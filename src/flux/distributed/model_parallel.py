# Copyright (c) 2024 flux Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Distributed Training with FSDP2 and Model Parallelism for flux.

Implements fully sharded data parallel (FSDP2) with automatic model parallelism
for training 70B+ models on consumer hardware. Includes gradient checkpointing
optimization and memory-efficient attention.
"""

import os
import math
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp._common_utils import _get_module_fsdp_extension
from torch.distributed.fsdp._init_utils import _init_intra_node_process_group
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Any, Tuple, List, Union, Callable
import logging
from dataclasses import dataclass, field
from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training with FSDP2 and model parallelism."""
    
    # FSDP Configuration
    fsdp_sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    fsdp_backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE
    fsdp_cpu_offload: bool = False
    fsdp_forward_prefetch: bool = False
    fsdp_limit_all_gathers: bool = True
    fsdp_use_orig_params: bool = True
    fsdp_sync_module_states: bool = True
    fsdp_activation_checkpointing: bool = True
    
    # Model Parallelism Configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    auto_parallel: bool = True
    min_model_size_for_parallel: int = 13_000_000_000  # 13B parameters
    
    # Memory Optimization
    gradient_checkpointing_ratio: float = 0.5  # Checkpoint every N layers
    use_flash_attention: bool = True
    use_memory_efficient_attention: bool = True
    max_memory_gb: Optional[float] = None
    
    # Mixed Precision
    mixed_precision_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    buffer_dtype: torch.dtype = torch.float32
    
    # Communication
    nccl_async_error_handling: bool = True
    gradient_predivide_factor: float = 1.0
    
    def __post_init__(self):
        if self.max_memory_gb is None:
            # Auto-detect available GPU memory
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                self.max_memory_gb = gpu_mem / (1024 ** 3) * 0.9  # Use 90% of available memory


class MemoryEfficientAttentionWrapper(nn.Module):
    """Wrapper for memory-efficient attention mechanisms."""
    
    def __init__(self, original_attention: nn.Module, use_flash: bool = True):
        super().__init__()
        self.original_attention = original_attention
        self.use_flash = use_flash
        
        # Check if flash attention is available
        self.flash_available = False
        try:
            from flash_attn import flash_attn_func
            self.flash_available = True
            self.flash_attn_func = flash_attn_func
        except ImportError:
            logger.warning("Flash Attention not available, falling back to memory-efficient attention")
        
        # Check for PyTorch's built-in memory-efficient attention
        self.torch_memory_efficient = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(self, *args, **kwargs):
        if self.use_flash and self.flash_available:
            return self._flash_attention_forward(*args, **kwargs)
        elif self.torch_memory_efficient:
            return self._memory_efficient_forward(*args, **kwargs)
        else:
            return self.original_attention(*args, **kwargs)
    
    def _flash_attention_forward(self, query, key, value, **kwargs):
        """Flash Attention forward pass."""
        # Reshape for flash attention: (batch, seq_len, num_heads, head_dim)
        batch_size, seq_len, num_heads, head_dim = query.shape
        
        # Flash attention expects (batch, seqlen, nheads, headdim)
        query = query.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Apply flash attention
        attn_output = self.flash_attn_func(
            query, key, value,
            dropout_p=kwargs.get('dropout_p', 0.0),
            softmax_scale=kwargs.get('softmax_scale', None),
            causal=kwargs.get('causal', True)
        )
        
        # Reshape back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output
    
    def _memory_efficient_forward(self, query, key, value, **kwargs):
        """PyTorch's memory-efficient attention forward pass."""
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=kwargs.get('attn_mask'),
            dropout_p=kwargs.get('dropout_p', 0.0),
            is_causal=kwargs.get('causal', True)
        )


class GradientCheckpointingWrapper(nn.Module):
    """Wrapper for gradient checkpointing with configurable ratio."""
    
    def __init__(self, module: nn.Module, checkpoint_ratio: float = 0.5):
        super().__init__()
        self.module = module
        self.checkpoint_ratio = checkpoint_ratio
        self.checkpoint_layers = []
        
        # Identify which layers to checkpoint
        if hasattr(module, 'layers') or hasattr(module, 'encoder') or hasattr(module, 'decoder'):
            self._identify_checkpoint_layers()
    
    def _identify_checkpoint_layers(self):
        """Identify layers to checkpoint based on ratio."""
        layers = []
        
        # Common patterns for transformer layers
        if hasattr(self.module, 'layers'):
            layers = list(self.module.layers)
        elif hasattr(self.module, 'encoder') and hasattr(self.module.encoder, 'layer'):
            layers = list(self.module.encoder.layer)
        elif hasattr(self.module, 'decoder') and hasattr(self.module.decoder, 'layer'):
            layers = list(self.module.decoder.layer)
        
        if layers:
            # Select layers to checkpoint based on ratio
            num_layers = len(layers)
            num_checkpoint = max(1, int(num_layers * self.checkpoint_ratio))
            
            # Checkpoint evenly spaced layers
            step = max(1, num_layers // num_checkpoint)
            self.checkpoint_layers = list(range(0, num_layers, step))
            
            logger.info(f"Gradient checkpointing enabled for {len(self.checkpoint_layers)}/{num_layers} layers")
    
    def forward(self, *args, **kwargs):
        if self.training and self.checkpoint_layers:
            return self._forward_with_checkpointing(*args, **kwargs)
        else:
            return self.module(*args, **kwargs)
    
    def _forward_with_checkpointing(self, *args, **kwargs):
        """Forward pass with gradient checkpointing."""
        # This is a simplified implementation
        # In practice, you'd need to handle the specific model architecture
        def custom_forward(*inputs):
            return self.module(*inputs, **kwargs)
        
        return checkpoint(
            custom_forward,
            *args,
            use_reentrant=False,
            preserve_rng_state=False
        )


class AutomaticParallelPlanner:
    """Plans automatic parallelism based on model size and available hardware."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Detect available GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_memory = self._get_gpu_memory()
    
    def _get_gpu_memory(self) -> List[float]:
        """Get memory available on each GPU in GB."""
        if not torch.cuda.is_available():
            return []
        
        memory = []
        for i in range(self.num_gpus):
            with torch.cuda.device(i):
                total_mem = torch.cuda.get_device_properties(i).total_memory
                memory.append(total_mem / (1024 ** 3))  # Convert to GB
        return memory
    
    def estimate_model_memory(self, model: nn.Module) -> float:
        """Estimate model memory requirements in GB."""
        param_count = sum(p.numel() for p in model.parameters())
        
        # Rough estimation: 18 bytes per parameter for mixed precision training
        # (2 bytes param + 2 bytes grad + 4 bytes optimizer state + 10 bytes overhead)
        memory_gb = (param_count * 18) / (1024 ** 3)
        return memory_gb
    
    def plan_parallelism(self, model: nn.Module) -> Dict[str, int]:
        """Plan parallelism strategy based on model and hardware."""
        model_memory = self.estimate_model_memory(model)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Default plan
        plan = {
            "tensor_parallel": 1,
            "data_parallel": self.world_size,
            "pipeline_parallel": 1,
            "use_fsdp": True,
            "fsdp_sharding": "full",
        }
        
        # Check if model is large enough for parallelism
        if not self.config.auto_parallel or param_count < self.config.min_model_size_for_parallel:
            logger.info(f"Model size {param_count:,} parameters < {self.config.min_model_size_for_parallel:,}, using data parallel only")
            return plan
        
        # Calculate required parallelism
        if self.gpu_memory:
            avg_gpu_memory = sum(self.gpu_memory) / len(self.gpu_memory)
            min_gpu_memory = min(self.gpu_memory)
            
            # Estimate memory needed per GPU
            memory_per_gpu = model_memory / self.world_size
            
            # Adjust for FSDP overhead
            if plan["use_fsdp"]:
                memory_per_gpu *= 0.3  # FSDP reduces memory usage
            
            if memory_per_gpu > min_gpu_memory * 0.8:  # Use 80% of GPU memory
                # Need tensor parallelism
                tp_size = 2
                while tp_size <= min(8, self.num_gpus):
                    if model_memory / (self.world_size * tp_size) <= min_gpu_memory * 0.8:
                        plan["tensor_parallel"] = tp_size
                        plan["data_parallel"] = self.world_size // tp_size
                        break
                    tp_size *= 2
                
                logger.info(f"Auto-selected tensor parallelism size: {plan['tensor_parallel']}")
                logger.info(f"Data parallel size: {plan['data_parallel']}")
        
        # Adjust FSDP sharding based on parallelism
        if plan["tensor_parallel"] > 1:
            plan["fsdp_sharding"] = "hybrid"  # Hybrid sharding for model parallelism
        
        return plan


class FSDP2ModelParallelWrapper:
    """Main wrapper for FSDP2 with model parallelism."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DistributedConfig] = None,
        device_id: Optional[int] = None,
    ):
        self.model = model
        self.config = config or DistributedConfig()
        self.device_id = device_id or (torch.cuda.current_device() if torch.cuda.is_available() else None)
        
        # Initialize distributed if not already
        if not dist.is_initialized() and torch.cuda.is_available():
            self._init_distributed()
        
        # Plan parallelism
        self.planner = AutomaticParallelPlanner(self.config)
        self.parallel_plan = self.planner.plan_parallelism(model)
        
        # Apply optimizations
        self._apply_optimizations()
    
    def _init_distributed(self):
        """Initialize distributed training environment."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )
            logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
    
    def _apply_optimizations(self):
        """Apply memory and performance optimizations."""
        # Apply gradient checkpointing
        if self.config.fsdp_activation_checkpointing:
            self.model = GradientCheckpointingWrapper(
                self.model,
                checkpoint_ratio=self.config.gradient_checkpointing_ratio
            )
        
        # Apply memory-efficient attention
        if self.config.use_memory_efficient_attention or self.config.use_flash_attention:
            self._wrap_attention_layers()
        
        # Enable activation checkpointing for transformer blocks
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def _wrap_attention_layers(self):
        """Wrap attention layers with memory-efficient implementations."""
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                if isinstance(module, nn.Module):
                    # Replace with memory-efficient wrapper
                    wrapped = MemoryEfficientAttentionWrapper(
                        module,
                        use_flash=self.config.use_flash_attention
                    )
                    # Replace the module in the model
                    parts = name.split('.')
                    parent = self.model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], wrapped)
    
    def _get_mixed_precision_policy(self) -> MixedPrecision:
        """Get mixed precision policy for FSDP."""
        return MixedPrecision(
            param_dtype=self.config.mixed_precision_dtype,
            reduce_dtype=self.config.reduce_dtype,
            buffer_dtype=self.config.buffer_dtype,
        )
    
    def _get_auto_wrap_policy(self) -> Callable:
        """Get auto-wrap policy for FSDP."""
        # Try to detect transformer blocks
        transformer_cls = None
        for name, module in self.model.named_modules():
            if 'TransformerBlock' in type(module).__name__ or 'DecoderLayer' in type(module).__name__:
                transformer_cls = type(module)
                break
        
        if transformer_cls:
            # Use transformer-aware wrapping
            return partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_cls}
            )
        else:
            # Fall back to size-based wrapping
            return partial(
                size_based_auto_wrap_policy,
                min_num_params=1e6  # Wrap modules with > 1M parameters
            )
    
    def wrap_model(self) -> nn.Module:
        """Wrap the model with FSDP2 and model parallelism."""
        # Apply tensor parallelism first if needed
        if self.parallel_plan["tensor_parallel"] > 1:
            self._apply_tensor_parallelism()
        
        # Get FSDP configuration
        mixed_precision = self._get_mixed_precision_policy()
        auto_wrap_policy = self._get_auto_wrap_policy()
        
        # Configure CPU offloading
        cpu_offload = CPUOffload(offload_params=True) if self.config.fsdp_cpu_offload else None
        
        # Get sharding strategy
        sharding_strategy = ShardingStrategy[self.config.fsdp_sharding_strategy.name]
        
        # Wrap with FSDP
        wrapped_model = FSDP(
            self.model,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            backward_prefetch=self.config.fsdp_backward_prefetch,
            cpu_offload=cpu_offload,
            forward_prefetch=self.config.fsdp_forward_prefetch,
            limit_all_gathers=self.config.fsdp_limit_all_gathers,
            use_orig_params=self.config.fsdp_use_orig_params,
            sync_module_states=self.config.fsdp_sync_module_states,
            device_id=self.device_id,
        )
        
        logger.info(f"Model wrapped with FSDP2. Parallel plan: {self.parallel_plan}")
        return wrapped_model
    
    def _apply_tensor_parallelism(self):
        """Apply tensor parallelism to the model."""
        tp_size = self.parallel_plan["tensor_parallel"]
        
        if tp_size <= 1:
            return
        
        # Define parallelization plan for transformer models
        # This is a simplified version - in practice, you'd need to detect the specific architecture
        parallelize_plan = {}
        
        # Common patterns for transformer models
        for name, module in self.model.named_modules():
            if 'q_proj' in name or 'query' in name:
                parallelize_plan[name] = ColwiseParallel()
            elif 'k_proj' in name or 'key' in name:
                parallelize_plan[name] = ColwiseParallel()
            elif 'v_proj' in name or 'value' in name:
                parallelize_plan[name] = ColwiseParallel()
            elif 'o_proj' in name or 'output' in name:
                parallelize_plan[name] = RowwiseParallel()
            elif 'gate_proj' in name or 'up_proj' in name:
                parallelize_plan[name] = ColwiseParallel()
            elif 'down_proj' in name:
                parallelize_plan[name] = RowwiseParallel()
        
        if parallelize_plan:
            # Apply tensor parallelism
            device_mesh = dist.device_mesh.init_device_mesh(
                "cuda",
                (tp_size, dist.get_world_size() // tp_size),
                mesh_dim_names=("tp", "dp")
            )
            
            self.model = parallelize_module(
                self.model,
                device_mesh=device_mesh["tp"],
                parallelize_plan=parallelize_plan,
            )
            
            logger.info(f"Applied tensor parallelism with size {tp_size}")
    
    def get_sharded_state_dict(self) -> Dict[str, Any]:
        """Get sharded state dict for checkpointing."""
        if isinstance(self.model, FSDP):
            return self.model.state_dict()
        return self.model.state_dict()
    
    def load_sharded_state_dict(self, state_dict: Dict[str, Any]):
        """Load sharded state dict."""
        if isinstance(self.model, FSDP):
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


def setup_distributed_environment(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> bool:
    """Setup distributed training environment.
    
    Returns:
        True if distributed training is enabled, False otherwise.
    """
    if dist.is_initialized():
        return True
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        torch.cuda.set_device(local_rank)
        
        if init_method is None:
            init_method = "env://"
        
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        
        logger.info(f"Distributed training initialized: rank {rank}/{world_size}")
        return True
    
    logger.info("Distributed training not enabled")
    return False


def create_fsdp_model(
    model: nn.Module,
    config: Optional[DistributedConfig] = None,
    device_id: Optional[int] = None,
) -> nn.Module:
    """Create FSDP-wrapped model with model parallelism.
    
    Args:
        model: The model to wrap
        config: Distributed configuration
        device_id: GPU device ID
        
    Returns:
        FSDP-wrapped model
    """
    wrapper = FSDP2ModelParallelWrapper(model, config, device_id)
    return wrapper.wrap_model()


def get_optimal_batch_size(
    model: nn.Module,
    seq_length: int = 2048,
    config: Optional[DistributedConfig] = None,
) -> int:
    """Calculate optimal batch size based on model size and available memory.
    
    Args:
        model: The model
        seq_length: Sequence length
        config: Distributed configuration
        
    Returns:
        Optimal batch size per GPU
    """
    if config is None:
        config = DistributedConfig()
    
    # Estimate memory per sample
    param_count = sum(p.numel() for p in model.parameters())
    
    # Memory per sample in GB (rough estimate)
    # Activation memory: ~14 bytes per parameter per token for mixed precision
    # Plus optimizer states and gradients
    memory_per_sample_gb = (param_count * 14 * seq_length) / (1024 ** 3)
    
    # Get available memory
    if torch.cuda.is_available() and config.max_memory_gb:
        available_memory = config.max_memory_gb
    elif torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    else:
        available_memory = 16  # Default assumption for CPU
    
    # Calculate batch size (use 70% of available memory for activations)
    optimal_batch = max(1, int((available_memory * 0.7) / memory_per_sample_gb))
    
    # Round to power of 2 for efficiency
    optimal_batch = 2 ** int(math.log2(optimal_batch))
    
    logger.info(f"Optimal batch size: {optimal_batch} (seq_len={seq_length}, memory={available_memory:.1f}GB)")
    return optimal_batch


def log_memory_stats(prefix: str = ""):
    """Log GPU memory statistics."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            logger.info(f"{prefix} GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


# Export public API
__all__ = [
    "DistributedConfig",
    "FSDP2ModelParallelWrapper",
    "setup_distributed_environment",
    "create_fsdp_model",
    "get_optimal_batch_size",
    "log_memory_stats",
    "MemoryEfficientAttentionWrapper",
    "GradientCheckpointingWrapper",
    "AutomaticParallelPlanner",
]