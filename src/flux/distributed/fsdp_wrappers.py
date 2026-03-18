"""
flux Distributed Training with FSDP2 and Model Parallelism
"""
import os
import math
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
)
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp._init_utils import ProcessGroupType

try:
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        PrepareModuleInput,
    )
    HAS_TP = True
except ImportError:
    HAS_TP = False

from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

logger = logging.getLogger(__name__)


@dataclass
class FSDPConfig:
    """Configuration for FSDP2 distributed training."""
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    mixed_precision: Optional[MixedPrecision] = None
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE
    cpu_offload: Optional[CPUOffload] = None
    activation_checkpointing: bool = True
    auto_wrap_policy: Optional[str] = "size_based"  # "size_based", "transformer", "custom"
    min_num_params: int = 1e6  # For size-based auto wrap
    transformer_layer_cls: Optional[List[type]] = None
    sync_module_states: bool = True
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = True
    param_init_fn: Optional[Any] = None
    device_id: Optional[int] = None
    
    # Tensor Parallelism settings
    enable_tensor_parallel: bool = False
    tp_size: int = 1
    tp_parallel_plan: Optional[Dict[str, Any]] = None
    
    # Memory optimization
    gradient_checkpointing_ratio: float = 1.0  # Fraction of layers to checkpoint
    use_flash_attention: bool = True
    memory_efficient_attention: bool = True
    
    # Model size thresholds for auto-parallelism
    auto_parallel_model_size_gb: float = 14.0  # Threshold for auto TP
    min_gpu_memory_gb: float = 24.0  # Minimum GPU memory to consider
    
    def __post_init__(self):
        if self.transformer_layer_cls is None:
            self.transformer_layer_cls = [
                LlamaDecoderLayer,
                MistralDecoderLayer,
                Qwen2DecoderLayer,
                Qwen3DecoderLayer,
                GemmaDecoderLayer,
            ]
        
        if self.mixed_precision is None:
            self.mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )


class MemoryEfficientAttentionWrapper(nn.Module):
    """Wrapper for memory-efficient attention mechanisms."""
    
    def __init__(self, module: nn.Module, config: FSDPConfig):
        super().__init__()
        self.module = module
        self.config = config
        
        # Try to enable flash attention if available
        if config.use_flash_attention:
            self._enable_flash_attention()
    
    def _enable_flash_attention(self):
        """Enable flash attention if available."""
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
        except ImportError:
            logger.warning("Flash attention not available, falling back to standard attention")
            self.flash_attn_func = None
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class GradientCheckpointingWrapper(nn.Module):
    """Wrapper for gradient checkpointing with selective layer checkpointing."""
    
    def __init__(self, module: nn.Module, config: FSDPConfig):
        super().__init__()
        self.module = module
        self.config = config
        self._setup_gradient_checkpointing()
    
    def _setup_gradient_checkpointing(self):
        """Apply gradient checkpointing to transformer layers."""
        if not self.config.activation_checkpointing:
            return
        
        # Find transformer layers and wrap them
        for name, child in self.module.named_children():
            if any(isinstance(child, cls) for cls in self.config.transformer_layer_cls):
                # Apply checkpointing to this layer
                wrapped = checkpoint_wrapper(
                    child,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    preserve_rng_state=False,
                )
                setattr(self.module, name, wrapped)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FSDPModelWrapper:
    """Main wrapper for FSDP2 distributed training with model parallelism."""
    
    def __init__(self, model: PreTrainedModel, config: Optional[FSDPConfig] = None):
        self.model = model
        self.config = config or FSDPConfig()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Auto-detect tensor parallelism needs
        if self.config.enable_tensor_parallel and HAS_TP:
            self._auto_detect_tensor_parallel()
    
    def _auto_detect_tensor_parallel(self):
        """Automatically determine if tensor parallelism is needed based on model size."""
        model_size_gb = self._estimate_model_size_gb()
        gpu_memory_gb = self._get_gpu_memory_gb()
        
        logger.info(f"Model size: {model_size_gb:.2f} GB, GPU memory: {gpu_memory_gb:.2f} GB")
        
        # Determine if tensor parallelism is needed
        if model_size_gb > self.config.auto_parallel_model_size_gb:
            # Calculate optimal TP size
            optimal_tp = min(
                self.world_size,
                max(1, int(model_size_gb / (gpu_memory_gb * 0.7)))  # Use 70% of GPU memory
            )
            
            if optimal_tp > 1 and self.world_size >= optimal_tp:
                self.config.tp_size = optimal_tp
                self.config.enable_tensor_parallel = True
                logger.info(f"Auto-enabling tensor parallelism with TP size: {optimal_tp}")
            else:
                self.config.enable_tensor_parallel = False
                logger.info("Tensor parallelism not needed or not enough GPUs")
        else:
            self.config.enable_tensor_parallel = False
    
    def _estimate_model_size_gb(self) -> float:
        """Estimate model size in GB."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        # Account for optimizer states (Adam: 2 states per param)
        optimizer_size = param_size * 2
        
        # Account for gradients
        gradient_size = param_size
        
        total_size = param_size + optimizer_size + gradient_size
        return total_size / (1024 ** 3)  # Convert to GB
    
    def _get_gpu_memory_gb(self) -> float:
        """Get available GPU memory in GB."""
        if torch.cuda.is_available():
            gpu_id = self.local_rank % torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            return gpu_memory / (1024 ** 3)
        return self.config.min_gpu_memory_gb
    
    def _get_auto_wrap_policy(self):
        """Get the appropriate auto wrap policy."""
        if self.config.auto_wrap_policy == "size_based":
            return size_based_auto_wrap_policy(
                min_num_params=self.config.min_num_params
            )
        elif self.config.auto_wrap_policy == "transformer":
            return transformer_auto_wrap_policy(
                transformer_layer_cls=self.config.transformer_layer_cls
            )
        elif self.config.auto_wrap_policy == "custom":
            # Custom lambda policy for LLMs
            def lambda_policy(module):
                # Wrap attention and MLP blocks
                if hasattr(module, 'self_attn') or hasattr(module, 'mlp'):
                    return True
                # Wrap large linear layers
                if isinstance(module, nn.Linear) and module.weight.numel() > 1e6:
                    return True
                return False
            return lambda_auto_wrap_policy(lambda_policy)
        else:
            return None
    
    def _apply_tensor_parallelism(self):
        """Apply tensor parallelism to the model."""
        if not self.config.enable_tensor_parallel or not HAS_TP:
            return self.model
        
        # Create tensor parallel plan
        if self.config.tp_parallel_plan is None:
            # Default parallel plan for transformer models
            self.config.tp_parallel_plan = self._create_default_tp_plan()
        
        # Apply tensor parallelism
        device_mesh = dist.device_mesh.init_device_mesh(
            "cuda", (self.config.tp_size,)
        )
        
        parallelized_model = parallelize_module(
            self.model,
            device_mesh=device_mesh,
            parallelize_plan=self.config.tp_parallel_plan,
        )
        
        logger.info(f"Applied tensor parallelism with TP size: {self.config.tp_size}")
        return parallelized_model
    
    def _create_default_tp_plan(self) -> Dict[str, Any]:
        """Create default tensor parallel plan for transformer models."""
        plan = {}
        
        # Parallelize attention layers
        for name, module in self.model.named_modules():
            if "self_attn" in name or "attention" in name:
                if hasattr(module, 'q_proj'):
                    plan[f"{name}.q_proj"] = ColwiseParallel()
                if hasattr(module, 'k_proj'):
                    plan[f"{name}.k_proj"] = ColwiseParallel()
                if hasattr(module, 'v_proj'):
                    plan[f"{name}.v_proj"] = ColwiseParallel()
                if hasattr(module, 'o_proj'):
                    plan[f"{name}.o_proj"] = RowwiseParallel()
            
            # Parallelize MLP layers
            elif "mlp" in name or "feed_forward" in name:
                if hasattr(module, 'gate_proj'):
                    plan[f"{name}.gate_proj"] = ColwiseParallel()
                if hasattr(module, 'up_proj'):
                    plan[f"{name}.up_proj"] = ColwiseParallel()
                if hasattr(module, 'down_proj'):
                    plan[f"{name}.down_proj"] = RowwiseParallel()
        
        return plan
    
    def _apply_memory_optimizations(self):
        """Apply memory optimization techniques."""
        # Apply gradient checkpointing
        if self.config.activation_checkpointing:
            self.model = GradientCheckpointingWrapper(self.model, self.config)
        
        # Apply memory-efficient attention
        if self.config.memory_efficient_attention:
            self.model = MemoryEfficientAttentionWrapper(self.model, self.config)
        
        # Enable gradient checkpointing in model config if available
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False  # Disable KV cache for training
    
    def wrap(self) -> FSDP:
        """Wrap the model with FSDP2 and apply all optimizations."""
        # Step 1: Apply tensor parallelism if enabled
        if self.config.enable_tensor_parallel:
            self.model = self._apply_tensor_parallelism()
        
        # Step 2: Apply memory optimizations
        self._apply_memory_optimizations()
        
        # Step 3: Get auto wrap policy
        auto_wrap_policy = self._get_auto_wrap_policy()
        
        # Step 4: Determine device ID
        device_id = self.config.device_id
        if device_id is None and torch.cuda.is_available():
            device_id = self.local_rank % torch.cuda.device_count()
        
        # Step 5: Create FSDP model
        fsdp_model = FSDP(
            self.model,
            sharding_strategy=self.config.sharding_strategy,
            mixed_precision=self.config.mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=self.config.backward_prefetch,
            cpu_offload=self.config.cpu_offload,
            sync_module_states=self.config.sync_module_states,
            forward_prefetch=self.config.forward_prefetch,
            limit_all_gathers=self.config.limit_all_gathers,
            use_orig_params=self.config.use_orig_params,
            param_init_fn=self.config.param_init_fn,
            device_id=device_id,
        )
        
        logger.info("Model wrapped with FSDP2 successfully")
        return fsdp_model
    
    def get_fsdp_state(self, module: nn.Module) -> Optional[Any]:
        """Get FSDP state for a module."""
        return _get_module_fsdp_state(module)
    
    def get_sharded_parameters(self, module: nn.Module) -> List[nn.Parameter]:
        """Get sharded parameters from FSDP module."""
        sharded_params = []
        fsdp_state = self.get_fsdp_state(module)
        
        if fsdp_state is not None:
            for param in module.parameters():
                if hasattr(param, '_is_sharded'):
                    sharded_params.append(param)
        
        return sharded_params


def create_fsdp_model(
    model: PreTrainedModel,
    config: Optional[FSDPConfig] = None,
    **kwargs
) -> FSDP:
    """
    Convenience function to create FSDP-wrapped model.
    
    Args:
        model: The model to wrap
        config: FSDP configuration
        **kwargs: Additional arguments to override config
        
    Returns:
        FSDP-wrapped model
    """
    if config is None:
        config = FSDPConfig()
    
    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    wrapper = FSDPModelWrapper(model, config)
    return wrapper.wrap()


def setup_distributed_environment(
    backend: str = "nccl",
    init_method: str = "env://",
    timeout_seconds: int = 1800,
) -> None:
    """
    Setup distributed training environment.
    
    Args:
        backend: Distributed backend (nccl, gloo, etc.)
        init_method: Initialization method
        timeout_seconds: Timeout for initialization
    """
    if not dist.is_initialized():
        # Set environment variables if not set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timeout_seconds,
        )
        
        logger.info(f"Distributed environment initialized: "
                   f"rank={dist.get_rank()}, world_size={dist.get_world_size()}")


def cleanup_distributed_environment() -> None:
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device_mesh(
    mesh_shape: Tuple[int, ...],
    mesh_dim_names: Optional[List[str]] = None,
) -> Any:
    """
    Create device mesh for hybrid parallelism.
    
    Args:
        mesh_shape: Shape of the device mesh
        mesh_dim_names: Names of mesh dimensions
        
    Returns:
        Device mesh
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized")
    
    if mesh_dim_names is None:
        mesh_dim_names = ["data", "tensor"] if len(mesh_shape) == 2 else ["data"]
    
    return dist.device_mesh.init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=mesh_dim_names
    )


class FSDPCheckpointManager:
    """Manager for FSDP checkpointing."""
    
    @staticmethod
    def save_checkpoint(
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        loss: float,
        save_path: str,
        **kwargs
    ) -> None:
        """Save FSDP checkpoint."""
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        
        # Configure state dict type
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        
        # Create checkpoint
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            **kwargs
        }
        
        # Save only on rank 0
        if dist.get_rank() == 0:
            torch.save(checkpoint, save_path)
            logger.info(f"Checkpoint saved to {save_path}")
    
    @staticmethod
    def load_checkpoint(
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        load_path: str,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load FSDP checkpoint."""
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        
        # Load checkpoint on all ranks
        checkpoint = torch.load(load_path, map_location="cpu")
        
        # Configure state dict type
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {load_path}")
        return checkpoint


# Utility functions for memory optimization
def optimize_memory_usage() -> None:
    """Optimize PyTorch memory usage."""
    # Enable memory efficient attention
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    
    # Enable gradient checkpointing
    torch.utils.checkpoint.checkpoint_sequential = True


def get_model_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """Get detailed memory footprint of model."""
    param_mem = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_mem = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    return {
        "parameters_gb": param_mem / (1024 ** 3),
        "buffers_gb": buffer_mem / (1024 ** 3),
        "total_gb": (param_mem + buffer_mem) / (1024 ** 3),
    }


# Export main classes and functions
__all__ = [
    "FSDPConfig",
    "FSDPModelWrapper",
    "create_fsdp_model",
    "setup_distributed_environment",
    "cleanup_distributed_environment",
    "get_device_mesh",
    "FSDPCheckpointManager",
    "optimize_memory_usage",
    "get_model_memory_footprint",
]