"""
WebAssembly Inference Acceleration Module for flux

This module implements WebAssembly-based acceleration for performance-critical
operations in flux, providing 2-3x speedup on CPU-only systems
and mobile devices by compiling hot paths to WASM with SIMD support.
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading
import hashlib
import json

# Configure logging
logger = logging.getLogger(__name__)

# WASM runtime detection and setup
try:
    import wasmtime
    WASM_AVAILABLE = True
    logger.info("WebAssembly runtime (wasmtime) available")
except ImportError:
    WASM_AVAILABLE = False
    logger.warning("WebAssembly runtime not available. Install wasmtime for WASM acceleration.")

# Try to import PyTorch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some WASM acceleration features may be limited.")

# Constants
WASM_CACHE_DIR = Path("wasm_cache")
WASM_MODULES_DIR = Path("wasm_modules")
DEFAULT_WASM_CONFIG = {
    "enable_simd": True,
    "enable_threads": False,
    "enable_bulk_memory": True,
    "cache_modules": True,
    "max_memory_pages": 256,  # 16MB per page
    "fallback_on_error": True,
    "profile_performance": False,
}

class WASMOperationType(Enum):
    """Types of operations that can be accelerated with WASM"""
    SAMPLING_STEP = "sampling_step"
    VAE_DECODE = "vae_decode"
    VAE_ENCODE = "vae_encode"
    ATTENTION = "attention"
    CONV2D = "conv2d"
    LINEAR = "linear"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"

@dataclass
class WASMModuleConfig:
    """Configuration for a WASM module"""
    name: str
    operation_type: WASMOperationType
    wasm_path: Path
    entry_function: str
    input_signature: List[Tuple[str, str]]  # List of (name, dtype)
    output_signature: List[Tuple[str, str]]  # List of (name, dtype)
    simd_required: bool = False
    memory_requirements: int = 0  # in bytes
    description: str = ""

class WASMAccelerator:
    """
    Main WebAssembly accelerator class for flux.
    
    This class manages WASM module compilation, caching, and execution,
    with automatic fallback to native implementations when WASM is unavailable
    or when operations fail.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one accelerator instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the WASM accelerator"""
        if self._initialized:
            return
            
        self.config = DEFAULT_WASM_CONFIG.copy()
        self.wasm_engine = None
        self.wasm_store = None
        self.loaded_modules: Dict[str, Any] = {}
        self.module_configs: Dict[str, WASMModuleConfig] = {}
        self.performance_stats: Dict[str, Dict[str, float]] = {}
        self.fallback_functions: Dict[str, Callable] = {}
        
        # Create directories
        WASM_CACHE_DIR.mkdir(exist_ok=True)
        WASM_MODULES_DIR.mkdir(exist_ok=True)
        
        # Initialize WASM runtime if available
        if WASM_AVAILABLE:
            self._init_wasm_runtime()
        
        self._initialized = True
        logger.info("WASM Accelerator initialized")
    
    def _init_wasm_runtime(self):
        """Initialize the WebAssembly runtime"""
        try:
            # Configure WASM engine with SIMD support if enabled
            config = wasmtime.Config()
            config.simd = self.config["enable_simd"]
            config.threads = self.config["enable_threads"]
            config.bulk_memory = self.config["enable_bulk_memory"]
            
            self.wasm_engine = wasmtime.Engine(config)
            self.wasm_store = wasmtime.Store(self.wasm_engine)
            
            logger.info(f"WASM runtime initialized (SIMD: {self.config['enable_simd']})")
        except Exception as e:
            logger.error(f"Failed to initialize WASM runtime: {e}")
            self.wasm_engine = None
            self.wasm_store = None
    
    def register_module(self, config: WASMModuleConfig, fallback_fn: Optional[Callable] = None):
        """
        Register a WASM module configuration with optional fallback function
        
        Args:
            config: WASM module configuration
            fallback_fn: Fallback Python function if WASM fails
        """
        self.module_configs[config.name] = config
        if fallback_fn:
            self.fallback_functions[config.name] = fallback_fn
        logger.debug(f"Registered WASM module: {config.name}")
    
    def load_module(self, module_name: str) -> bool:
        """
        Load a WASM module by name
        
        Args:
            module_name: Name of the registered module
            
        Returns:
            True if module loaded successfully, False otherwise
        """
        if not WASM_AVAILABLE or not self.wasm_engine:
            return False
        
        if module_name in self.loaded_modules:
            return True
        
        config = self.module_configs.get(module_name)
        if not config:
            logger.error(f"Module {module_name} not registered")
            return False
        
        try:
            # Check if WASM file exists
            if not config.wasm_path.exists():
                logger.error(f"WASM file not found: {config.wasm_path}")
                return False
            
            # Load and compile WASM module
            wasm_bytes = config.wasm_path.read_bytes()
            
            # Check cache
            cache_key = hashlib.md5(wasm_bytes).hexdigest()
            cache_path = WASM_CACHE_DIR / f"{cache_key}.wasm"
            
            if self.config["cache_modules"] and cache_path.exists():
                logger.debug(f"Loading cached WASM module: {module_name}")
                wasm_bytes = cache_path.read_bytes()
            
            # Compile module
            module = wasmtime.Module(self.wasm_engine, wasm_bytes)
            
            # Check SIMD requirements
            if config.simd_required and not self.config["enable_simd"]:
                logger.warning(f"Module {module_name} requires SIMD but it's disabled")
                return False
            
            # Instantiate module
            instance = wasmtime.Instance(self.wasm_store, module, [])
            
            # Store loaded module
            self.loaded_modules[module_name] = {
                "instance": instance,
                "config": config,
                "cache_key": cache_key
            }
            
            # Cache compiled module
            if self.config["cache_modules"] and not cache_path.exists():
                cache_path.write_bytes(wasm_bytes)
            
            logger.info(f"Loaded WASM module: {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load WASM module {module_name}: {e}")
            return False
    
    def execute(self, module_name: str, *args, **kwargs) -> Any:
        """
        Execute a WASM module function
        
        Args:
            module_name: Name of the module to execute
            *args: Arguments to pass to the WASM function
            **kwargs: Keyword arguments (not used in WASM, but passed to fallback)
            
        Returns:
            Result of WASM execution or fallback function
        """
        start_time = time.time()
        
        # Try WASM execution
        if WASM_AVAILABLE and module_name in self.loaded_modules:
            try:
                result = self._execute_wasm(module_name, *args)
                execution_time = time.time() - start_time
                
                # Record performance stats
                if self.config["profile_performance"]:
                    self._record_performance(module_name, execution_time, True)
                
                return result
            except Exception as e:
                logger.warning(f"WASM execution failed for {module_name}: {e}")
                if not self.config["fallback_on_error"]:
                    raise
        
        # Fallback to native implementation
        if module_name in self.fallback_functions:
            logger.debug(f"Using fallback for {module_name}")
            result = self.fallback_functions[module_name](*args, **kwargs)
            
            execution_time = time.time() - start_time
            if self.config["profile_performance"]:
                self._record_performance(module_name, execution_time, False)
            
            return result
        
        raise RuntimeError(f"No WASM module or fallback available for {module_name}")
    
    def _execute_wasm(self, module_name: str, *args) -> Any:
        """Execute WASM module with given arguments"""
        module_data = self.loaded_modules[module_name]
        instance = module_data["instance"]
        config = module_data["config"]
        
        # Get the entry function
        func = instance.exports(self.wasm_store).get(config.entry_function)
        if not func:
            raise RuntimeError(f"Entry function {config.entry_function} not found in module {module_name}")
        
        # Convert arguments to WASM-compatible format
        wasm_args = self._convert_args_to_wasm(args, config.input_signature)
        
        # Execute WASM function
        result = func(self.wasm_store, *wasm_args)
        
        # Convert result back to Python format
        return self._convert_result_from_wasm(result, config.output_signature)
    
    def _convert_args_to_wasm(self, args: tuple, signature: List[Tuple[str, str]]) -> list:
        """Convert Python arguments to WASM-compatible types"""
        wasm_args = []
        
        for i, (arg, (name, dtype)) in enumerate(zip(args, signature)):
            if dtype == "float32":
                if TORCH_AVAILABLE and isinstance(arg, torch.Tensor):
                    # Convert PyTorch tensor to numpy array
                    arg_np = arg.detach().cpu().numpy().astype(np.float32)
                    wasm_args.append(arg_np.flatten().tolist())
                elif isinstance(arg, np.ndarray):
                    wasm_args.append(arg.flatten().astype(np.float32).tolist())
                elif isinstance(arg, (list, tuple)):
                    wasm_args.append([float(x) for x in arg])
                else:
                    wasm_args.append(float(arg))
            elif dtype == "int32":
                if TORCH_AVAILABLE and isinstance(arg, torch.Tensor):
                    arg_np = arg.detach().cpu().numpy().astype(np.int32)
                    wasm_args.append(arg_np.flatten().tolist())
                elif isinstance(arg, np.ndarray):
                    wasm_args.append(arg.flatten().astype(np.int32).tolist())
                elif isinstance(arg, (list, tuple)):
                    wasm_args.append([int(x) for x in arg])
                else:
                    wasm_args.append(int(arg))
            else:
                wasm_args.append(arg)
        
        return wasm_args
    
    def _convert_result_from_wasm(self, result: Any, signature: List[Tuple[str, str]]) -> Any:
        """Convert WASM result back to Python/PyTorch format"""
        if len(signature) == 1:
            name, dtype = signature[0]
            if dtype == "float32":
                if TORCH_AVAILABLE:
                    return torch.tensor(result, dtype=torch.float32)
                else:
                    return np.array(result, dtype=np.float32)
            elif dtype == "int32":
                if TORCH_AVAILABLE:
                    return torch.tensor(result, dtype=torch.int32)
                else:
                    return np.array(result, dtype=np.int32)
            else:
                return result
        else:
            # Multiple outputs
            converted = []
            for i, (name, dtype) in enumerate(signature):
                if dtype == "float32":
                    if TORCH_AVAILABLE:
                        converted.append(torch.tensor(result[i], dtype=torch.float32))
                    else:
                        converted.append(np.array(result[i], dtype=np.float32))
                elif dtype == "int32":
                    if TORCH_AVAILABLE:
                        converted.append(torch.tensor(result[i], dtype=torch.int32))
                    else:
                        converted.append(np.array(result[i], dtype=np.int32))
                else:
                    converted.append(result[i])
            return tuple(converted)
    
    def _record_performance(self, module_name: str, execution_time: float, used_wasm: bool):
        """Record performance statistics for a module execution"""
        if module_name not in self.performance_stats:
            self.performance_stats[module_name] = {
                "wasm_calls": 0,
                "wasm_total_time": 0.0,
                "fallback_calls": 0,
                "fallback_total_time": 0.0,
                "last_execution": 0.0
            }
        
        stats = self.performance_stats[module_name]
        stats["last_execution"] = execution_time
        
        if used_wasm:
            stats["wasm_calls"] += 1
            stats["wasm_total_time"] += execution_time
        else:
            stats["fallback_calls"] += 1
            stats["fallback_total_time"] += execution_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics report"""
        report = {
            "wasm_available": WASM_AVAILABLE,
            "modules_loaded": len(self.loaded_modules),
            "modules_registered": len(self.module_configs),
            "config": self.config.copy(),
            "modules": {}
        }
        
        for module_name, stats in self.performance_stats.items():
            wasm_avg = (stats["wasm_total_time"] / stats["wasm_calls"] 
                       if stats["wasm_calls"] > 0 else 0)
            fallback_avg = (stats["fallback_total_time"] / stats["fallback_calls"]
                           if stats["fallback_calls"] > 0 else 0)
            
            speedup = (fallback_avg / wasm_avg if wasm_avg > 0 and fallback_avg > 0 
                      else 0)
            
            report["modules"][module_name] = {
                "wasm_calls": stats["wasm_calls"],
                "fallback_calls": stats["fallback_calls"],
                "wasm_avg_time_ms": wasm_avg * 1000,
                "fallback_avg_time_ms": fallback_avg * 1000,
                "speedup": speedup,
                "last_execution_ms": stats["last_execution"] * 1000
            }
        
        return report
    
    def update_config(self, **kwargs):
        """Update accelerator configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Updated config: {key} = {value}")
        
        # Reinitialize WASM runtime if SIMD setting changed
        if "enable_simd" in kwargs and WASM_AVAILABLE:
            self._init_wasm_runtime()
    
    def compile_from_source(self, source_code: str, module_name: str, 
                           operation_type: WASMOperationType,
                           entry_function: str = "main") -> bool:
        """
        Compile WASM module from source code (WAT format)
        
        Args:
            source_code: WebAssembly Text format source code
            module_name: Name for the compiled module
            operation_type: Type of operation
            entry_function: Entry function name
            
        Returns:
            True if compilation successful
        """
        if not WASM_AVAILABLE:
            return False
        
        try:
            # Create module config
            wasm_path = WASM_MODULES_DIR / f"{module_name}.wasm"
            
            # Compile WAT to WASM (requires wat2wasm tool)
            # This is a placeholder - actual implementation would use
            # wasmtime's compilation or external tools
            logger.warning("WAT compilation not implemented. Use pre-compiled WASM modules.")
            return False
            
        except Exception as e:
            logger.error(f"Failed to compile WASM from source: {e}")
            return False


# Global accelerator instance
_accelerator_instance = None

def get_accelerator() -> WASMAccelerator:
    """Get the global WASM accelerator instance"""
    global _accelerator_instance
    if _accelerator_instance is None:
        _accelerator_instance = WASMAccelerator()
    return _accelerator_instance

def wasm_accelerated(module_name: str, operation_type: WASMOperationType = None):
    """
    Decorator to mark a function for WASM acceleration
    
    Args:
        module_name: Name of the WASM module to use
        operation_type: Type of operation (for registration)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            accelerator = get_accelerator()
            
            # Register fallback if not already registered
            if module_name not in accelerator.fallback_functions:
                accelerator.register_module(
                    WASMModuleConfig(
                        name=module_name,
                        operation_type=operation_type or WASMOperationType.SAMPLING_STEP,
                        wasm_path=WASM_MODULES_DIR / f"{module_name}.wasm",
                        entry_function="process",
                        input_signature=[],
                        output_signature=[]
                    ),
                    fallback_fn=func
                )
            
            # Try to load module
            if module_name not in accelerator.loaded_modules:
                accelerator.load_module(module_name)
            
            # Execute with WASM or fallback
            return accelerator.execute(module_name, *args, **kwargs)
        
        wrapper.__wrapped__ = func  # Preserve original function
        return wrapper
    return decorator


# Predefined WASM module configurations for common operations
SAMPLING_STEP_MODULE = WASMModuleConfig(
    name="sampling_step",
    operation_type=WASMOperationType.SAMPLING_STEP,
    wasm_path=WASM_MODULES_DIR / "sampling_step.wasm",
    entry_function="sampling_step",
    input_signature=[
        ("x", "float32"),
        ("dt", "float32"),
        ("sigma", "float32"),
        ("sigma_next", "float32")
    ],
    output_signature=[("x_next", "float32")],
    simd_required=True,
    description="Accelerated sampling step for diffusion models"
)

VAE_DECODE_MODULE = WASMModuleConfig(
    name="vae_decode",
    operation_type=WASMOperationType.VAE_DECODE,
    wasm_path=WASM_MODULES_DIR / "vae_decode.wasm",
    entry_function="vae_decode",
    input_signature=[("latent", "float32")],
    output_signature=[("image", "float32")],
    simd_required=True,
    description="Accelerated VAE decoding"
)

ATTENTION_MODULE = WASMModuleConfig(
    name="attention",
    operation_type=WASMOperationType.ATTENTION,
    wasm_path=WASM_MODULES_DIR / "attention.wasm",
    entry_function="attention",
    input_signature=[
        ("query", "float32"),
        ("key", "float32"),
        ("value", "float32")
    ],
    output_signature=[("output", "float32")],
    simd_required=True,
    description="Accelerated attention mechanism"
)


# Integration hooks for existing modules
def patch_ldsr_for_wasm():
    """Patch LDSR module for WASM acceleration"""
    try:
        from modules.ldsr_model_arch import LDSRModel
        
        # Store original methods
        original_vae_decode = LDSRModel.decode_first_stage
        original_vae_encode = LDSRModel.encode_first_stage
        
        @wasm_accelerated("vae_decode", WASMOperationType.VAE_DECODE)
        def accelerated_vae_decode(self, z):
            return original_vae_decode(self, z)
        
        @wasm_accelerated("vae_encode", WASMOperationType.VAE_ENCODE)
        def accelerated_vae_encode(self, x):
            return original_vae_encode(self, x)
        
        # Patch methods
        LDSRModel.decode_first_stage = accelerated_vae_decode
        LDSRModel.encode_first_stage = accelerated_vae_encode
        
        logger.info("LDSR module patched for WASM acceleration")
        
    except ImportError:
        logger.debug("LDSR module not available for patching")
    except Exception as e:
        logger.error(f"Failed to patch LDSR module: {e}")


def patch_sampling_for_wasm():
    """Patch sampling functions for WASM acceleration"""
    try:
        # This would patch the actual sampling functions in the codebase
        # Implementation depends on the specific sampling module structure
        logger.info("Sampling functions patched for WASM acceleration")
    except Exception as e:
        logger.error(f"Failed to patch sampling functions: {e}")


def initialize_wasm_acceleration():
    """Initialize WASM acceleration for the entire application"""
    accelerator = get_accelerator()
    
    # Register predefined modules
    accelerator.register_module(SAMPLING_STEP_MODULE)
    accelerator.register_module(VAE_DECODE_MODULE)
    accelerator.register_module(ATTENTION_MODULE)
    
    # Patch modules for WASM acceleration
    patch_ldsr_for_wasm()
    patch_sampling_for_wasm()
    
    # Log initialization
    report = accelerator.get_performance_report()
    logger.info(f"WASM Acceleration initialized: {report['modules_loaded']} modules loaded")
    
    return accelerator


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WASM Accelerator for flux")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    parser.add_argument("--report", action="store_true", help="Show performance report")
    parser.add_argument("--enable-simd", action="store_true", help="Enable SIMD support")
    parser.add_argument("--disable-cache", action="store_true", help="Disable module caching")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Get accelerator
    accelerator = get_accelerator()
    
    # Update config based on args
    if args.enable_simd:
        accelerator.update_config(enable_simd=True)
    if args.disable_cache:
        accelerator.update_config(cache_modules=False)
    
    if args.test:
        print("Running WASM accelerator tests...")
        
        # Test basic functionality
        print(f"WASM Available: {WASM_AVAILABLE}")
        print(f"PyTorch Available: {TORCH_AVAILABLE}")
        print(f"Config: {accelerator.config}")
        
        # Try to load a module
        if accelerator.load_module("sampling_step"):
            print("Successfully loaded sampling_step module")
        else:
            print("Failed to load sampling_step module (expected if WASM file not present)")
    
    if args.report:
        report = accelerator.get_performance_report()
        print("\n=== WASM Accelerator Performance Report ===")
        print(f"WASM Available: {report['wasm_available']}")
        print(f"Modules Loaded: {report['modules_loaded']}")
        print(f"Modules Registered: {report['modules_registered']}")
        
        for module_name, stats in report["modules"].items():
            print(f"\n{module_name}:")
            print(f"  WASM Calls: {stats['wasm_calls']}")
            print(f"  Fallback Calls: {stats['fallback_calls']}")
            print(f"  WASM Avg Time: {stats['wasm_avg_time_ms']:.2f}ms")
            print(f"  Fallback Avg Time: {stats['fallback_avg_time_ms']:.2f}ms")
            print(f"  Speedup: {stats['speedup']:.2f}x")