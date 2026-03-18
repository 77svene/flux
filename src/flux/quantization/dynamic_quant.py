import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Params4bit
from bitsandbytes.functional import dequantize_4bit, quantize_4bit
from bitsandbytes.optim import PagedAdamW8bit, PagedAdamW32bit
import math
from typing import Optional, Tuple, Dict, Any, Union
import warnings

# Custom CUDA kernel compilation
try:
    dynamic_quant_cuda = load(
        name="dynamic_quant_cuda",
        sources=[
            "src/flux/quantization/cuda/dynamic_quant_kernel.cu",
            "src/flux/quantization/cuda/dynamic_quant.cpp"
        ],
        verbose=False
    )
except:
    warnings.warn("Custom CUDA kernels not compiled. Using fallback implementations.")
    dynamic_quant_cuda = None

class DynamicQuantConfig:
    """Configuration for dynamic quantization during training."""
    
    def __init__(
        self,
        quant_method: str = "nf4",
        double_quant: bool = True,
        quant_type: str = "fp4",
        compress_statistics: bool = True,
        quant_storage: torch.dtype = torch.uint8,
        dynamic_quant_bits: int = 8,
        dynamic_quant_group_size: int = 128,
        paged_optimizer: bool = True,
        gradient_checkpointing: bool = False,
        mixed_precision: str = "bf16",
        gradient_scaling_factor: float = 1.0,
        min_scaling_factor: float = 1e-7,
        use_custom_kernels: bool = True
    ):
        self.quant_method = quant_method
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.compress_statistics = compress_statistics
        self.quant_storage = quant_storage
        self.dynamic_quant_bits = dynamic_quant_bits
        self.dynamic_quant_group_size = dynamic_quant_group_size
        self.paged_optimizer = paged_optimizer
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.gradient_scaling_factor = gradient_scaling_factor
        self.min_scaling_factor = min_scaling_factor
        self.use_custom_kernels = use_custom_kernels


class DynamicQuantLinear(nn.Module):
    """Dynamic quantization linear layer with QLoRA support.
    
    Supports 4-bit NormalFloat with double quantization and dynamic
    activation quantization during training.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[DynamicQuantConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or DynamicQuantConfig()
        
        # Initialize 4-bit quantized weights
        self.weight = Params4bit(
            torch.empty(out_features, in_features, dtype=dtype or torch.float16, device=device),
            requires_grad=True,
            compress_statistics=self.config.compress_statistics,
            quant_type=self.config.quant_type,
            quant_storage=self.config.quant_storage,
            blocksize=64 if self.config.double_quant else 128
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype or torch.float16, device=device)
            )
        else:
            self.register_parameter('bias', None)
        
        # Dynamic quantization scales
        self.register_buffer('input_scale', torch.ones(1, dtype=torch.float32))
        self.register_buffer('output_scale', torch.ones(1, dtype=torch.float32))
        self.register_buffer('grad_scale', torch.ones(1, dtype=torch.float32))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with proper scaling for quantized training."""
        nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.data)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic quantization."""
        if self.config.gradient_checkpointing and self.training:
            return self._forward_checkpoint(x)
        return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Main forward implementation."""
        original_dtype = x.dtype
        
        # Dynamic input quantization
        if self.training and self.config.dynamic_quant_bits < 16:
            x_quant, input_scale = self._dynamic_quantize_input(x)
            self.input_scale.copy_(input_scale.mean())
        else:
            x_quant = x
        
        # Dequantize weights for computation
        weight_dequant = dequantize_4bit(
            self.weight.data,
            self.weight.quant_state,
            quant_type=self.config.quant_type
        ).to(x.dtype)
        
        # Compute output with custom kernel if available
        if dynamic_quant_cuda is not None and self.config.use_custom_kernels:
            output = dynamic_quant_cuda.dynamic_linear_forward(
                x_quant, weight_dequant, self.bias, 
                self.config.dynamic_quant_bits, self.config.dynamic_quant_group_size
            )
        else:
            output = F.linear(x_quant, weight_dequant, self.bias)
        
        # Dynamic output quantization
        if self.training and self.config.dynamic_quant_bits < 16:
            output_quant, output_scale = self._dynamic_quantize_output(output)
            self.output_scale.copy_(output_scale.mean())
            return output_quant.to(original_dtype)
        
        return output.to(original_dtype)
    
    def _forward_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with gradient checkpointing for memory efficiency."""
        def custom_forward(*inputs):
            return self._forward_impl(inputs[0])
        
        return torch.utils.checkpoint.checkpoint(
            custom_forward, x, use_reentrant=False
        )
    
    def _dynamic_quantize_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamically quantize input activations."""
        if dynamic_quant_cuda is not None and self.config.use_custom_kernels:
            return dynamic_quant_cuda.dynamic_quantize(
                x, self.config.dynamic_quant_bits, 
                self.config.dynamic_quant_group_size, True
            )
        else:
            # Fallback implementation
            return self._fallback_dynamic_quantize(x)
    
    def _dynamic_quantize_output(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamically quantize output activations."""
        if dynamic_quant_cuda is not None and self.config.use_custom_kernels:
            return dynamic_quant_cuda.dynamic_quantize(
                x, self.config.dynamic_quant_bits, 
                self.config.dynamic_quant_group_size, False
            )
        else:
            return self._fallback_dynamic_quantize(x)
    
    def _fallback_dynamic_quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback dynamic quantization when CUDA kernels are not available."""
        # Per-token quantization
        if x.dim() == 3:  # [batch, seq_len, hidden]
            batch, seq_len, hidden = x.shape
            x_flat = x.reshape(-1, hidden)
        else:
            x_flat = x
        
        # Compute scale per token
        abs_max = torch.abs(x_flat).max(dim=-1, keepdim=True)[0]
        scale = abs_max / (2 ** (self.config.dynamic_quant_bits - 1) - 1)
        scale = torch.clamp(scale, min=self.config.min_scaling_factor)
        
        # Quantize
        x_quant = torch.round(x_flat / scale).to(torch.int8)
        
        if x.dim() == 3:
            x_quant = x_quant.reshape(batch, seq_len, hidden)
        
        return x_quant, scale
    
    def backward_hook(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Custom backward hook for gradient scaling."""
        if self.config.gradient_scaling_factor != 1.0:
            grad_output = grad_output * self.config.gradient_scaling_factor
        
        # Gradient clipping for stability
        grad_norm = torch.norm(grad_output)
        if grad_norm > 1.0:
            grad_output = grad_output / grad_norm
        
        self.grad_scale.copy_(grad_norm.mean())
        return grad_output


class DynamicQuantQLoRA:
    """QLoRA with dynamic quantization support.
    
    Implements QLoRA with 4-bit NormalFloat, double quantization,
    paged optimizers, and dynamic activation quantization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DynamicQuantConfig] = None,
        target_modules: Optional[list] = None
    ):
        self.model = model
        self.config = config or DynamicQuantConfig()
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Convert target modules to dynamic quantized layers
        self._convert_to_dynamic_quant()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
    
    def _convert_to_dynamic_quant(self):
        """Convert target modules to dynamic quantized layers."""
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with dynamic quantized linear
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.model.get_submodule(parent_name)
                    
                    new_module = DynamicQuantLinear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        config=self.config,
                        device=module.weight.device,
                        dtype=module.weight.dtype
                    )
                    
                    # Copy weights
                    with torch.no_grad():
                        new_module.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_module.bias.data.copy_(module.bias.data)
                    
                    setattr(parent, child_name, new_module)
    
    def _create_optimizer(self):
        """Create paged optimizer for memory efficiency."""
        # Separate parameters for different learning rates
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        if self.config.paged_optimizer:
            # Use paged optimizer for memory efficiency
            if self.config.mixed_precision == "bf16":
                return PagedAdamW8bit(
                    param_groups,
                    lr=2e-5,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            else:
                return PagedAdamW32bit(
                    param_groups,
                    lr=2e-5,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
        else:
            return torch.optim.AdamW(
                param_groups,
                lr=2e-5,
                betas=(0.9, 0.999),
                eps=1e-8
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step with dynamic quantization."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Backward pass with gradient scaling
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'grad_scale': self._get_average_grad_scale(),
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'memory_reserved': torch.cuda.memory_reserved() / 1024**3
        }
        
        return metrics
    
    def _get_average_grad_scale(self) -> float:
        """Get average gradient scale from dynamic quant layers."""
        scales = []
        for module in self.model.modules():
            if isinstance(module, DynamicQuantLinear):
                scales.append(module.grad_scale.item())
        
        return sum(scales) / len(scales) if scales else 1.0
    
    def save_pretrained(self, path: str):
        """Save model with quantization state."""
        # Dequantize weights for saving
        state_dict = {}
        for name, param in self.model.named_parameters():
            if isinstance(param, Params4bit):
                # Save dequantized weights
                dequant = dequantize_4bit(
                    param.data,
                    param.quant_state,
                    quant_type=self.config.quant_type
                )
                state_dict[name] = dequant.cpu()
            else:
                state_dict[name].cpu()
        
        torch.save({
            'model_state_dict': state_dict,
            'config': self.config.__dict__,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    @classmethod
    def from_pretrained(cls, path: str, model: nn.Module) -> 'DynamicQuantQLoRA':
        """Load model with quantization state."""
        checkpoint = torch.load(path, map_location='cpu')
        config = DynamicQuantConfig(**checkpoint['config'])
        
        instance = cls(model, config)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return instance


class GradientScaler:
    """Gradient scaler for low-precision training stability."""
    
    def __init__(self, init_scale: float = 2**16, growth_factor: float = 2.0, 
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.iter = 0
        self.found_inf = False
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation."""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients after backward pass."""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Perform optimizer step with gradient scaling."""
        self.iter += 1
        
        # Check for NaN/Inf gradients
        self.found_inf = False
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self.found_inf = True
                        break
            if self.found_inf:
                break
        
        if self.found_inf:
            # Skip step and reduce scale
            self.scale *= self.backoff_factor
            optimizer.zero_grad()
        else:
            # Unscale gradients and step
            self.unscale_gradients(optimizer)
            optimizer.step()
            optimizer.zero_grad()
            
            # Increase scale periodically
            if self.iter % self.growth_interval == 0:
                self.scale *= self.growth_factor
    
    def get_scale(self) -> float:
        """Get current scaling factor."""
        return self.scale


def create_dynamic_quant_model(
    model: nn.Module,
    quant_config: Optional[DynamicQuantConfig] = None,
    target_modules: Optional[list] = None
) -> DynamicQuantQLoRA:
    """Factory function to create dynamic quantization model."""
    config = quant_config or DynamicQuantConfig()
    return DynamicQuantQLoRA(model, config, target_modules)


def get_paged_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw_8bit",
    lr: float = 2e-5,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """Get paged optimizer for memory-efficient training."""
    # Separate parameters
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if optimizer_type == "adamw_8bit":
        return PagedAdamW8bit(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type == "adamw_32bit":
        return PagedAdamW32bit(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


# Integration with existing flux modules
def patch_flux_quantization():
    """Patch flux's quantization module with dynamic quantization."""
    try:
        from flux.model import quantization
        
        # Add dynamic quantization to available methods
        if not hasattr(quantization.QuantizationMethod, 'DYNAMIC_NF4'):
            quantization.QuantizationMethod.DYNAMIC_NF4 = "dynamic_nf4"
        
        # Extend get_quantization_config
        original_get_config = quantization.get_quantization_config
        
        def extended_get_config(model_args, finetuning_args):
            config = original_get_config(model_args, finetuning_args)
            
            if getattr(model_args, 'quantization_method', None) == 'dynamic_nf4':
                config.update({
                    'quant_method': 'dynamic_nf4',
                    'dynamic_quant_bits': getattr(model_args, 'dynamic_quant_bits', 8),
                    'dynamic_quant_group_size': getattr(model_args, 'dynamic_quant_group_size', 128),
                    'paged_optimizer': getattr(model_args, 'paged_optimizer', True),
                    'gradient_scaling_factor': getattr(model_args, 'gradient_scaling_factor', 1.0)
                })
            
            return config
        
        quantization.get_quantization_config = extended_get_config
        
    except ImportError:
        warnings.warn("flux not found. Skipping integration patch.")


# Apply patch on import
patch_flux_quantization()