"""
Memory-Optimized Inference Pipeline for Stable Diffusion WebUI
Dynamic CPU/GPU/VRAM offloading with priority-based model swapping
"""

import gc
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import psutil
import torch
import torch.nn as nn
from torch.cuda import nvtx

logger = logging.getLogger(__name__)

class ModelPriority(Enum):
    """Priority levels for model components"""
    CRITICAL = 0      # Always in VRAM (e.g., currently active UNet)
    HIGH = 1          # Frequently used (e.g., text encoder during prompt)
    MEDIUM = 2        # Used per-step (e.g., VAE during decode)
    LOW = 3           # Rarely used (e.g., secondary models)
    BACKGROUND = 4    # Can stay on CPU (e.g., LoRA weights)

class MemoryLocation(Enum):
    """Where model components can reside"""
    GPU_VRAM = "gpu_vram"
    GPU_SHARED = "gpu_shared"  # Unified memory
    CPU_PINNED = "cpu_pinned"  # Pinned memory for fast transfer
    CPU_REGULAR = "cpu_regular"
    DISK = "disk"  # For extremely large models

@dataclass
class ModelComponent:
    """Represents a model component with memory metadata"""
    name: str
    model: nn.Module
    size_mb: float
    priority: ModelPriority
    location: MemoryLocation
    last_used: float
    access_count: int
    cuda_stream: Optional[torch.cuda.Stream]
    pinned_memory: Optional[torch.Tensor]
    transfer_in_progress: bool = False
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class MemoryStats:
    """Tracks memory usage statistics"""
    def __init__(self):
        self.peak_vram = 0
        self.current_vram = 0
        self.total_vram = 0
        self.available_vram = 0
        self.cpu_memory_used = 0
        self.cpu_memory_total = 0
        self.swaps_total = 0
        self.swaps_last_minute = 0
        self.swap_history = deque(maxlen=60)  # Last 60 seconds
        self.hit_rate = 0.0
        self.miss_rate = 0.0
        
    def update(self):
        """Update memory statistics"""
        if torch.cuda.is_available():
            self.current_vram = torch.cuda.memory_allocated() / (1024 ** 2)
            self.peak_vram = max(self.peak_vram, self.current_vram)
            self.total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            self.available_vram = self.total_vram - self.current_vram
        
        process = psutil.Process()
        mem_info = process.memory_info()
        self.cpu_memory_used = mem_info.rss / (1024 ** 2)
        self.cpu_memory_total = psutil.virtual_memory().total / (1024 ** 2)
        
        # Calculate swap rate
        current_time = time.time()
        self.swap_history.append(current_time)
        self.swaps_last_minute = sum(1 for t in self.swap_history if current_time - t < 60)

class VRAMDefragmenter:
    """Handles VRAM defragmentation and compaction"""
    def __init__(self, threshold_mb: float = 100):
        self.threshold_mb = threshold_mb
        self.fragmentation_score = 0.0
        
    def should_defragment(self, memory_stats: MemoryStats) -> bool:
        """Determine if defragmentation is needed"""
        if memory_stats.available_vram < self.threshold_mb:
            return True
        return False
    
    def defragment(self, components: Dict[str, ModelComponent]):
        """Perform VRAM defragmentation"""
        logger.info("Starting VRAM defragmentation")
        
        # Group components by location
        gpu_components = [c for c in components.values() if c.location == MemoryLocation.GPU_VRAM]
        
        if not gpu_components:
            return
        
        # Sort by priority and size (move low priority, large components first)
        gpu_components.sort(key=lambda x: (x.priority.value, -x.size_mb))
        
        # Move components to CPU if needed
        for component in gpu_components[:len(gpu_components) // 2]:
            if component.priority.value >= ModelPriority.MEDIUM.value:
                self._move_to_cpu(component)
        
        # Clear cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("VRAM defragmentation completed")

class PrioritySwapper:
    """Handles priority-based model swapping"""
    def __init__(self, vram_limit_mb: float, cpu_pinned_mb: float = 2048):
        self.vram_limit_mb = vram_limit_mb
        self.cpu_pinned_limit_mb = cpu_pinned_mb
        self.components: Dict[str, ModelComponent] = OrderedDict()
        self.access_queue = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.transfer_threads = []
        self.running = True
        
        # Initialize CUDA streams for async transfers
        self.streams = {}
        if torch.cuda.is_available():
            for i in range(4):  # Multiple streams for parallel transfers
                self.streams[f"stream_{i}"] = torch.cuda.Stream()
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.worker_thread.start()
    
    def register_component(self, name: str, model: nn.Module, priority: ModelPriority,
                          dependencies: List[str] = None) -> ModelComponent:
        """Register a model component with the memory manager"""
        with self.lock:
            if name in self.components:
                logger.warning(f"Component {name} already registered, updating")
            
            # Calculate model size
            size_mb = self._calculate_model_size(model)
            
            # Determine initial location based on priority and size
            if priority == ModelPriority.CRITICAL:
                location = MemoryLocation.GPU_VRAM
            elif priority == ModelPriority.HIGH and size_mb < 500:
                location = MemoryLocation.GPU_VRAM
            else:
                location = MemoryLocation.CPU_PINNED
            
            # Allocate pinned memory for CPU components
            pinned_memory = None
            if location == MemoryLocation.CPU_PINNED and torch.cuda.is_available():
                pinned_memory = self._allocate_pinned_memory(model)
            
            # Create component
            component = ModelComponent(
                name=name,
                model=model,
                size_mb=size_mb,
                priority=priority,
                location=location,
                last_used=time.time(),
                access_count=0,
                cuda_stream=self.streams.get("stream_0"),
                pinned_memory=pinned_memory,
                dependencies=dependencies or []
            )
            
            self.components[name] = component
            
            # Move to initial location
            if location == MemoryLocation.GPU_VRAM:
                self._move_to_gpu(component)
            elif location == MemoryLocation.CPU_PINNED:
                self._move_to_cpu_pinned(component)
            
            logger.info(f"Registered component {name} ({size_mb:.1f}MB) at {location.value}")
            return component
    
    def get_component(self, name: str, blocking: bool = True) -> Optional[nn.Module]:
        """Get a component, moving it to appropriate location if needed"""
        with self.lock:
            if name not in self.components:
                logger.error(f"Component {name} not found")
                return None
            
            component = self.components[name]
            component.last_used = time.time()
            component.access_count += 1
            
            # Check if component needs to be moved
            if component.location != MemoryLocation.GPU_VRAM:
                if blocking:
                    self._ensure_on_gpu(component)
                else:
                    # Schedule async transfer
                    self._schedule_async_transfer(component)
            
            # Update access history
            self.access_queue.append((name, time.time()))
            
            return component.model
    
    def _ensure_on_gpu(self, component: ModelComponent):
        """Ensure component is in VRAM, moving other components if necessary"""
        if component.location == MemoryLocation.GPU_VRAM:
            return
        
        # Check available VRAM
        memory_stats = MemoryStats()
        memory_stats.update()
        
        if memory_stats.available_vram < component.size_mb:
            # Need to free up space
            self._free_vram(component.size_mb - memory_stats.available_vram)
        
        # Move component to GPU
        self._move_to_gpu(component)
    
    def _free_vram(self, required_mb: float):
        """Free up VRAM by offloading lower priority components"""
        logger.info(f"Attempting to free {required_mb:.1f}MB VRAM")
        
        # Sort components by priority (lowest first) and last used (oldest first)
        candidates = []
        for component in self.components.values():
            if component.location == MemoryLocation.GPU_VRAM:
                score = component.priority.value * 1000 + (time.time() - component.last_used)
                candidates.append((score, component))
        
        candidates.sort(reverse=True)  # Highest score = best candidate to move
        
        freed_mb = 0
        for _, component in candidates:
            if freed_mb >= required_mb:
                break
            
            if component.priority.value > ModelPriority.CRITICAL.value:
                self._move_to_cpu(component)
                freed_mb += component.size_mb
        
        if freed_mb < required_mb:
            logger.warning(f"Could only free {freed_mb:.1f}MB of {required_mb:.1f}MB required")
    
    def _move_to_gpu(self, component: ModelComponent):
        """Move component to GPU VRAM"""
        if component.transfer_in_progress:
            return
        
        component.transfer_in_progress = True
        nvtx.mark(f"Moving {component.name} to GPU")
        
        try:
            with torch.cuda.stream(component.cuda_stream):
                if component.location == MemoryLocation.CPU_PINNED and component.pinned_memory is not None:
                    # Fast transfer from pinned memory
                    component.model.to('cuda', non_blocking=True)
                else:
                    component.model.to('cuda')
            
            component.location = MemoryLocation.GPU_VRAM
            logger.debug(f"Moved {component.name} to GPU VRAM")
            
        except Exception as e:
            logger.error(f"Failed to move {component.name} to GPU: {e}")
            component.location = MemoryLocation.CPU_REGULAR
        finally:
            component.transfer_in_progress = False
    
    def _move_to_cpu(self, component: ModelComponent):
        """Move component to CPU regular memory"""
        if component.transfer_in_progress:
            return
        
        component.transfer_in_progress = True
        nvtx.mark(f"Moving {component.name} to CPU")
        
        try:
            with torch.cuda.stream(component.cuda_stream):
                component.model.to('cpu')
            
            component.location = MemoryLocation.CPU_REGULAR
            logger.debug(f"Moved {component.name} to CPU")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to move {component.name} to CPU: {e}")
        finally:
            component.transfer_in_progress = False
    
    def _move_to_cpu_pinned(self, component: ModelComponent):
        """Move component to CPU pinned memory for fast transfers"""
        if not torch.cuda.is_available():
            self._move_to_cpu(component)
            return
        
        if component.transfer_in_progress:
            return
        
        component.transfer_in_progress = True
        nvtx.mark(f"Moving {component.name} to CPU pinned")
        
        try:
            # Allocate pinned memory if not already done
            if component.pinned_memory is None:
                component.pinned_memory = self._allocate_pinned_memory(component.model)
            
            # Copy model to pinned memory (simplified - actual implementation would copy weights)
            component.model.to('cpu')
            component.location = MemoryLocation.CPU_PINNED
            logger.debug(f"Moved {component.name} to CPU pinned memory")
            
        except Exception as e:
            logger.error(f"Failed to move {component.name} to CPU pinned: {e}")
            component.location = MemoryLocation.CPU_REGULAR
        finally:
            component.transfer_in_progress = False
    
    def _schedule_async_transfer(self, component: ModelComponent):
        """Schedule an asynchronous transfer to GPU"""
        # Implementation would use CUDA streams for async transfers
        # For simplicity, we'll use a thread pool
        thread = threading.Thread(
            target=self._async_transfer_worker,
            args=(component,),
            daemon=True
        )
        thread.start()
        self.transfer_threads.append(thread)
    
    def _async_transfer_worker(self, component: ModelComponent):
        """Worker thread for async transfers"""
        try:
            self._move_to_gpu(component)
        except Exception as e:
            logger.error(f"Async transfer failed for {component.name}: {e}")
    
    def _background_worker(self):
        """Background worker for memory management tasks"""
        while self.running:
            time.sleep(5)  # Run every 5 seconds
            
            with self.lock:
                # Update component priorities based on access patterns
                self._update_priorities()
                
                # Check for components that should be moved
                self._check_and_move_components()
    
    def _update_priorities(self):
        """Update component priorities based on access patterns"""
        current_time = time.time()
        
        for component in self.components.values():
            # Components not used recently get lower priority
            time_since_use = current_time - component.last_used
            
            if time_since_use > 300:  # 5 minutes
                if component.priority.value < ModelPriority.BACKGROUND.value:
                    component.priority = ModelPriority.BACKGROUND
            elif time_since_use > 60:  # 1 minute
                if component.priority.value < ModelPriority.LOW.value:
                    component.priority = ModelPriority.LOW
    
    def _check_and_move_components(self):
        """Check and move components based on current memory state"""
        memory_stats = MemoryStats()
        memory_stats.update()
        
        # If VRAM is getting full, move low priority components to CPU
        if memory_stats.available_vram < 500:  # Less than 500MB available
            for component in self.components.values():
                if (component.location == MemoryLocation.GPU_VRAM and 
                    component.priority.value >= ModelPriority.MEDIUM.value):
                    self._move_to_cpu(component)
                    break
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        total_params = 0
        for param in model.parameters():
            total_params += param.numel()
        
        # Estimate size (assuming float32 = 4 bytes per parameter)
        size_mb = total_params * 4 / (1024 ** 2)
        return size_mb
    
    def _allocate_pinned_memory(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Allocate pinned memory for model weights"""
        if not torch.cuda.is_available():
            return None
        
        try:
            # Simplified: allocate a buffer of the same size as model
            total_params = sum(p.numel() for p in model.parameters())
            pinned_buffer = torch.empty(total_params, dtype=torch.float32, pin_memory=True)
            return pinned_buffer
        except Exception as e:
            logger.warning(f"Failed to allocate pinned memory: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the memory manager"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        for thread in self.transfer_threads:
            if thread.is_alive():
                thread.join(timeout=1)

class MemoryManager:
    """Main memory manager for Stable Diffusion WebUI"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.enabled = True
        self.vram_limit_mb = 0
        self.auto_optimize = True
        self.defragmenter = VRAMDefragmenter()
        self.memory_stats = MemoryStats()
        self.swapper = None
        self.component_registry = {}
        self.hooks_registered = False
        
        # Initialize based on available hardware
        self._detect_hardware()
        
        # Register hooks for automatic optimization
        if self.auto_optimize:
            self._register_hooks()
        
        logger.info(f"Memory Manager initialized (VRAM limit: {self.vram_limit_mb:.0f}MB)")
    
    def _detect_hardware(self):
        """Detect available hardware and set limits"""
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            total_vram = gpu_props.total_memory / (1024 ** 2)
            
            # Use 90% of available VRAM to leave headroom
            self.vram_limit_mb = total_vram * 0.9
            
            # Initialize swapper with detected limits
            self.swapper = PrioritySwapper(
                vram_limit_mb=self.vram_limit_mb,
                cpu_pinned_mb=min(4096, psutil.virtual_memory().total / (1024 ** 2) * 0.1)
            )
        else:
            logger.warning("CUDA not available, running in CPU-only mode")
            self.vram_limit_mb = 0
            self.enabled = False
    
    def _register_hooks(self):
        """Register hooks for automatic memory management"""
        if self.hooks_registered:
            return
        
        # Hook into model forward passes to track usage
        self._original_forwards = {}
        
        self.hooks_registered = True
        logger.debug("Registered memory management hooks")
    
    def register_model(self, name: str, model: nn.Module, 
                      priority: ModelPriority = ModelPriority.MEDIUM,
                      dependencies: List[str] = None) -> bool:
        """Register a model for memory management"""
        if not self.enabled or self.swapper is None:
            return False
        
        try:
            component = self.swapper.register_component(
                name=name,
                model=model,
                priority=priority,
                dependencies=dependencies
            )
            
            self.component_registry[name] = component
            
            # Register forward hook to track usage
            self._register_model_hook(name, model)
            
            return True
        except Exception as e:
            logger.error(f"Failed to register model {name}: {e}")
            return False
    
    def _register_model_hook(self, name: str, model: nn.Module):
        """Register forward hook for a model"""
        def hook_fn(module, input, output):
            # Update last used time
            if name in self.swapper.components:
                self.swapper.components[name].last_used = time.time()
        
        # Register hook on all submodules
        for submodule in model.modules():
            if len(list(submodule.children())) == 0:  # Leaf modules only
                submodule.register_forward_hook(hook_fn)
    
    def get_model(self, name: str, blocking: bool = True) -> Optional[nn.Module]:
        """Get a registered model, ensuring it's in the right location"""
        if not self.enabled or self.swapper is None:
            return None
        
        return self.swapper.get_component(name, blocking)
    
    def prefetch_model(self, name: str):
        """Prefetch a model to GPU for upcoming use"""
        if not self.enabled or self.swapper is None:
            return
        
        component = self.swapper.components.get(name)
        if component and component.location != MemoryLocation.GPU_VRAM:
            self.swapper._schedule_async_transfer(component)
    
    def optimize_for_inference(self):
        """Optimize memory layout for inference"""
        if not self.enabled:
            return
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Defragment if needed
        if self.defragmenter.should_defragment(self.memory_stats):
            self.defragmenter.defragment(self.swapper.components)
        
        # Update statistics
        self.memory_stats.update()
        
        logger.debug(f"Memory optimized: {self.memory_stats.current_vram:.0f}MB VRAM used")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        self.memory_stats.update()
        
        stats = {
            "vram_used_mb": self.memory_stats.current_vram,
            "vram_total_mb": self.memory_stats.total_vram,
            "vram_available_mb": self.memory_stats.available_vram,
            "vram_peak_mb": self.memory_stats.peak_vram,
            "cpu_memory_used_mb": self.memory_stats.cpu_memory_used,
            "cpu_memory_total_mb": self.memory_stats.cpu_memory_total,
            "swaps_last_minute": self.memory_stats.swaps_last_minute,
            "components": {}
        }
        
        if self.swapper:
            for name, component in self.swapper.components.items():
                stats["components"][name] = {
                    "size_mb": component.size_mb,
                    "location": component.location.value,
                    "priority": component.priority.name,
                    "last_used": component.last_used,
                    "access_count": component.access_count
                }
        
        return stats
    
    def set_vram_limit(self, limit_mb: float):
        """Set VRAM usage limit"""
        self.vram_limit_mb = limit_mb
        if self.swapper:
            self.swapper.vram_limit_mb = limit_mb
        logger.info(f"VRAM limit set to {limit_mb:.0f}MB")
    
    def enable(self):
        """Enable memory management"""
        self.enabled = True
        logger.info("Memory management enabled")
    
    def disable(self):
        """Disable memory management"""
        self.enabled = False
        logger.info("Memory management disabled")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.swapper:
            self.swapper.shutdown()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Memory manager cleaned up")

# Global instance
memory_manager = MemoryManager()

# Integration functions for existing codebase
def register_text_encoder(model: nn.Module):
    """Register text encoder model"""
    return memory_manager.register_model(
        name="text_encoder",
        model=model,
        priority=ModelPriority.HIGH
    )

def register_unet(model: nn.Module):
    """Register UNet model"""
    return memory_manager.register_model(
        name="unet",
        model=model,
        priority=ModelPriority.CRITICAL
    )

def register_vae(model: nn.Module):
    """Register VAE model"""
    return memory_manager.register_model(
        name="vae",
        model=model,
        priority=ModelPriority.MEDIUM
    )

def register_lora_model(name: str, model: nn.Module):
    """Register LoRA model"""
    return memory_manager.register_model(
        name=f"lora_{name}",
        model=model,
        priority=ModelPriority.LOW
    )

def get_text_encoder() -> Optional[nn.Module]:
    """Get text encoder, ensuring it's in VRAM"""
    return memory_manager.get_model("text_encoder")

def get_unet() -> Optional[nn.Module]:
    """Get UNet, ensuring it's in VRAM"""
    return memory_manager.get_model("unet")

def get_vae() -> Optional[nn.Module]:
    """Get VAE, ensuring it's in VRAM"""
    return memory_manager.get_model("vae")

def prefetch_for_step(step_type: str):
    """Prefetch models needed for the next step"""
    if step_type == "encode":
        memory_manager.prefetch_model("text_encoder")
    elif step_type == "diffusion":
        memory_manager.prefetch_model("unet")
    elif step_type == "decode":
        memory_manager.prefetch_model("vae")

def optimize_memory():
    """Optimize memory for inference"""
    memory_manager.optimize_for_inference()

def get_memory_report() -> Dict[str, Any]:
    """Get detailed memory report"""
    return memory_manager.get_memory_stats()

# Context manager for inference steps
class InferenceContext:
    """Context manager for memory-optimized inference"""
    
    def __init__(self, step_type: str):
        self.step_type = step_type
        self.start_stats = None
        
    def __enter__(self):
        self.start_stats = memory_manager.get_memory_stats()
        prefetch_for_step(self.step_type)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Optional: move models back to CPU after step
        # This can be configured based on usage patterns
        pass

# Decorator for memory-optimized functions
def memory_optimized(step_type: str = None):
    """Decorator to optimize memory for a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with InferenceContext(step_type or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Auto-initialization for existing modules
def initialize_for_existing_models():
    """Initialize memory manager for existing models in the codebase"""
    try:
        # Try to import and register models from existing modules
        from modules import sd_models
        
        # This would need to be adapted based on actual model loading
        logger.info("Memory manager ready for existing models")
        
    except ImportError:
        logger.debug("Could not import sd_models for auto-initialization")

# Initialize on import
initialize_for_existing_models()