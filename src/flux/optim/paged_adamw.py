import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Optional, Tuple, List, Dict, Any, Union
import bitsandbytes as bnb
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
import warnings
from dataclasses import dataclass
from enum import Enum

class QuantizationType(Enum):
    """Supported quantization types for optimizer states."""
    FP4 = "fp4"
    NF4 = "nf4"
    INT8 = "int8"
    NONE = "none"

@dataclass
class PagedAdamWConfig:
    """Configuration for PagedAdamW optimizer."""
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False
    maximize: bool = False
    foreach: Optional[bool] = None
    capturable: bool = False
    differentiable: bool = False
    fused: Optional[bool] = None
    
    # Quantization settings
    quant_type: QuantizationType = QuantizationType.NF4
    quantize_stats: bool = True
    double_quant: bool = True
    compress_statistics: bool = True
    
    # Memory optimization settings
    paged_optimizer: bool = True
    page_size: int = 4096
    max_memory: Optional[int] = None
    
    # Mixed precision settings
    use_gradient_scaling: bool = True
    dynamic_scaling: bool = True
    initial_scale: float = 2**16
    scale_window: int = 1000
    min_scale: float = 1.0
    max_scale: float = 2**24
    scale_growth_factor: float = 2.0
    scale_backoff_factor: float = 0.5
    
    # Performance settings
    use_cuda_kernels: bool = True
    optimize_memory: bool = True

class PagedAdamW(Optimizer):
    """
    Implements AdamW algorithm with paged memory and quantization support for QLoRA training.
    
    This optimizer extends standard AdamW with:
    1. 4-bit NormalFloat quantization with double quantization for optimizer states
    2. Paged memory management for CPU offloading
    3. Gradient scaling for mixed precision training
    4. Dynamic quantization during training
    5. Integration with bitsandbytes for efficient 4-bit operations
    
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): Coefficients for computing running averages (default: (0.9, 0.999))
        eps (float, optional): Term added to denominator for numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay coefficient (default: 0.01)
        amsgrad (bool, optional): Whether to use the AMSGrad variant (default: False)
        maximize (bool, optional): Maximize the objective with respect to the params (default: False)
        foreach (bool, optional): Whether foreach implementation of optimizer is used (default: None)
        capturable (bool, optional): Whether this instance is safe to capture in a CUDA graph (default: False)
        differentiable (bool, optional): Whether autograd should occur through the optimizer step (default: False)
        fused (bool, optional): Whether the fused implementation is used (default: None)
        
        # Quantization parameters
        quant_type (QuantizationType, optional): Type of quantization to use (default: NF4)
        quantize_stats (bool, optional): Whether to quantize optimizer statistics (default: True)
        double_quant (bool, optional): Whether to use double quantization (default: True)
        compress_statistics (bool, optional): Whether to compress statistics (default: True)
        
        # Memory optimization parameters
        paged_optimizer (bool, optional): Whether to use paged memory (default: True)
        page_size (int, optional): Size of memory pages in elements (default: 4096)
        max_memory (int, optional): Maximum memory usage in bytes (default: None)
        
        # Mixed precision parameters
        use_gradient_scaling (bool, optional): Whether to use gradient scaling (default: True)
        dynamic_scaling (bool, optional): Whether to dynamically adjust scale (default: True)
        initial_scale (float, optional): Initial gradient scale (default: 2**16)
        scale_window (int, optional): Window for scaling adjustments (default: 1000)
        min_scale (float, optional): Minimum gradient scale (default: 1.0)
        max_scale (float, optional): Maximum gradient scale (default: 2**24)
        scale_growth_factor (float, optional): Factor to increase scale (default: 2.0)
        scale_backoff_factor (float, optional): Factor to decrease scale (default: 0.5)
        
        # Performance parameters
        use_cuda_kernels (bool, optional): Whether to use custom CUDA kernels (default: True)
        optimize_memory (bool, optional): Whether to optimize memory usage (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        
        # Quantization parameters
        quant_type: Union[str, QuantizationType] = QuantizationType.NF4,
        quantize_stats: bool = True,
        double_quant: bool = True,
        compress_statistics: bool = True,
        
        # Memory optimization parameters
        paged_optimizer: bool = True,
        page_size: int = 4096,
        max_memory: Optional[int] = None,
        
        # Mixed precision parameters
        use_gradient_scaling: bool = True,
        dynamic_scaling: bool = True,
        initial_scale: float = 2**16,
        scale_window: int = 1000,
        min_scale: float = 1.0,
        max_scale: float = 2**24,
        scale_growth_factor: float = 2.0,
        scale_backoff_factor: float = 0.5,
        
        # Performance parameters
        use_cuda_kernels: bool = True,
        optimize_memory: bool = True,
    ):
        if isinstance(quant_type, str):
            quant_type = QuantizationType(quant_type)
        
        if quant_type not in [QuantizationType.NF4, QuantizationType.FP4, QuantizationType.INT8, QuantizationType.NONE]:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            quant_type=quant_type,
            quantize_stats=quantize_stats,
            double_quant=double_quant,
            compress_statistics=compress_statistics,
            paged_optimizer=paged_optimizer,
            page_size=page_size,
            max_memory=max_memory,
            use_gradient_scaling=use_gradient_scaling,
            dynamic_scaling=dynamic_scaling,
            initial_scale=initial_scale,
            scale_window=scale_window,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_growth_factor=scale_growth_factor,
            scale_backoff_factor=scale_backoff_factor,
            use_cuda_kernels=use_cuda_kernels,
            optimize_memory=optimize_memory,
        )
        
        super().__init__(params, defaults)
        
        # Initialize gradient scaling
        if use_gradient_scaling:
            self._init_gradient_scaling()
        
        # Initialize memory management
        self._init_memory_management()
        
        # Register with bitsandbytes manager if using quantization
        if quant_type != QuantizationType.NONE:
            self._register_with_bitsandbytes()
        
        # Initialize CUDA kernels if available
        if use_cuda_kernels and torch.cuda.is_available():
            self._init_cuda_kernels()
    
    def _init_gradient_scaling(self):
        """Initialize gradient scaling for mixed precision training."""
        self.scale = self.defaults['initial_scale']
        self.scale_growth_tracker = 0
        self.scale_backoff_tracker = 0
        self._growth_interval = self.defaults['scale_window']
        self._backoff_interval = 1  # Immediate backoff on overflow
        
        # Track if we've had a recent overflow
        self._recent_overflow = False
        
        # Store the device for scaling operations
        self._scale_device = None
    
    def _init_memory_management(self):
        """Initialize paged memory management."""
        self.page_size = self.defaults['page_size']
        self.max_memory = self.defaults['max_memory']
        self.paged_optimizer = self.defaults['paged_optimizer']
        
        # Track memory usage
        self._current_memory_usage = 0
        self._page_table = {}  # Maps parameter ID to page locations
        self._free_pages = []  # List of free page indices
        
        # Initialize memory pools
        if self.paged_optimizer:
            self._init_paged_memory()
    
    def _init_paged_memory(self):
        """Initialize paged memory system."""
        # Pre-allocate memory pages if max_memory is specified
        if self.max_memory is not None:
            num_pages = self.max_memory // (self.page_size * 4)  # 4 bytes per float32
            self._page_pool = torch.zeros(num_pages * self.page_size, dtype=torch.float32, device='cpu')
            self._free_pages = list(range(num_pages))
        else:
            self._page_pool = None
            self._free_pages = []
    
    def _register_with_bitsandbytes(self):
        """Register optimizer with bitsandbytes for 4-bit optimization."""
        try:
            self.manager = GlobalOptimManager.get_instance()
            self.manager.register_optimizer(self)
        except Exception as e:
            warnings.warn(f"Failed to register with bitsandbytes manager: {e}")
    
    def _init_cuda_kernels(self):
        """Initialize custom CUDA kernels for quantized operations."""
        try:
            # Import custom CUDA kernels if available
            from . import cuda_kernels
            self._cuda_kernels = cuda_kernels
            self._use_custom_kernels = True
        except ImportError:
            self._cuda_kernels = None
            self._use_custom_kernels = False
            warnings.warn("Custom CUDA kernels not available, falling back to PyTorch operations")
    
    def _allocate_page(self, param_id: int, size: int) -> torch.Tensor:
        """Allocate a memory page for a parameter."""
        if not self.paged_optimizer:
            # Allocate directly on device
            return torch.zeros(size, dtype=torch.float32, device=self._get_param_device(param_id))
        
        # Check if we have a free page
        if self._free_pages:
            page_idx = self._free_pages.pop()
            start_idx = page_idx * self.page_size
            end_idx = start_idx + size
            
            # Ensure we don't exceed page bounds
            if end_idx > start_idx + self.page_size:
                # Need multiple pages
                num_pages_needed = (size + self.page_size - 1) // self.page_size
                page_indices = [self._free_pages.pop() for _ in range(min(num_pages_needed, len(self._free_pages)))]
                if len(page_indices) < num_pages_needed:
                    # Not enough free pages, allocate new memory
                    return self._allocate_new_memory(size, param_id)
                
                # Use multiple pages
                self._page_table[param_id] = page_indices
                return self._get_paged_tensor(page_indices, size)
            else:
                # Single page is enough
                self._page_table[param_id] = [page_idx]
                return self._page_pool[start_idx:end_idx].view(size)
        else:
            # No free pages, allocate new memory
            return self._allocate_new_memory(size, param_id)
    
    def _allocate_new_memory(self, size: int, param_id: int) -> torch.Tensor:
        """Allocate new memory for optimizer states."""
        if self.max_memory is not None:
            # Check if we exceed memory limit
            required_memory = size * 4  # 4 bytes per float32
            if self._current_memory_usage + required_memory > self.max_memory:
                # Try to free some memory
                self._free_unused_memory()
                
                if self._current_memory_usage + required_memory > self.max_memory:
                    raise RuntimeError(f"Memory limit exceeded: {self._current_memory_usage + required_memory} > {self.max_memory}")
        
        # Allocate new memory
        tensor = torch.zeros(size, dtype=torch.float32, device='cpu')
        self._current_memory_usage += size * 4
        
        if self.paged_optimizer:
            # Track as non-paged memory
            self._page_table[param_id] = [-1]  # -1 indicates non-paged memory
        
        return tensor
    
    def _get_paged_tensor(self, page_indices: List[int], size: int) -> torch.Tensor:
        """Get a tensor from paged memory."""
        if len(page_indices) == 1:
            start_idx = page_indices[0] * self.page_size
            return self._page_pool[start_idx:start_idx + size].view(size)
        else:
            # Concatenate multiple pages
            tensors = []
            remaining = size
            for page_idx in page_indices:
                start_idx = page_idx * self.page_size
                chunk_size = min(remaining, self.page_size)
                tensors.append(self._page_pool[start_idx:start_idx + chunk_size])
                remaining -= chunk_size
            return torch.cat(tensors).view(size)
    
    def _free_unused_memory(self):
        """Free memory from unused optimizer states."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated memory management
        pass
    
    def _get_param_device(self, param_id: int) -> torch.device:
        """Get the device for a parameter."""
        for group in self.param_groups:
            for p in group['params']:
                if id(p) == param_id:
                    return p.device
        return torch.device('cpu')
    
    def _quantize_state(self, state: torch.Tensor, quant_type: QuantizationType) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize optimizer state tensor."""
        if quant_type == QuantizationType.NONE:
            return state, {}
        
        if not state.is_cuda and self.defaults['use_cuda_kernels']:
            # Move to GPU for quantization if using CUDA kernels
            state = state.cuda()
        
        if quant_type == QuantizationType.NF4:
            # Use bitsandbytes NF4 quantization
            if hasattr(bnb, 'functional') and hasattr(bnb.functional, 'quantize_4bit'):
                quantized, quant_state = bnb.functional.quantize_4bit(
                    state,
                    quant_type='nf4',
                    compress_statistics=self.defaults['compress_statistics']
                )
                return quantized, quant_state
            else:
                # Fallback to blockwise quantization
                quantized, absmax, shape, code, blocksize = quantize_blockwise(
                    state, blocksize=64, nested=self.defaults['double_quant']
                )
                return quantized, {
                    'absmax': absmax,
                    'shape': shape,
                    'code': code,
                    'blocksize': blocksize,
                    'quant_type': 'nf4'
                }
        
        elif quant_type == QuantizationType.FP4:
            # Use bitsandbytes FP4 quantization
            if hasattr(bnb, 'functional') and hasattr(bnb.functional, 'quantize_4bit'):
                quantized, quant_state = bnb.functional.quantize_4bit(
                    state,
                    quant_type='fp4',
                    compress_statistics=self.defaults['compress_statistics']
                )
                return quantized, quant_state
            else:
                # Fallback to blockwise quantization
                quantized, absmax, shape, code, blocksize = quantize_blockwise(
                    state, blocksize=64, nested=self.defaults['double_quant']
                )
                return quantized, {
                    'absmax': absmax,
                    'shape': shape,
                    'code': code,
                    'blocksize': blocksize,
                    'quant_type': 'fp4'
                }
        
        elif quant_type == QuantizationType.INT8:
            # Simple int8 quantization
            min_val, max_val = state.min(), state.max()
            scale = (max_val - min_val) / 255
            zero_point = -min_val / scale
            quantized = torch.clamp(torch.round(state / scale + zero_point), 0, 255).to(torch.uint8)
            return quantized, {
                'scale': scale,
                'zero_point': zero_point,
                'min_val': min_val,
                'max_val': max_val,
                'quant_type': 'int8'
            }
        
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
    
    def _dequantize_state(self, quantized: torch.Tensor, quant_state: Dict[str, Any]) -> torch.Tensor:
        """Dequantize optimizer state tensor."""
        if not quant_state:
            return quantized
        
        quant_type = quant_state.get('quant_type', 'none')
        
        if quant_type == 'nf4' or quant_type == 'fp4':
            if 'absmax' in quant_state:
                # Using blockwise quantization
                return dequantize_blockwise(
                    quantized,
                    absmax=quant_state['absmax'],
                    code=quant_state['code'],
                    blocksize=quant_state['blocksize']
                )
            else:
                # Using bitsandbytes 4-bit quantization
                return bnb.functional.dequantize_4bit(quantized, quant_state)
        
        elif quant_type == 'int8':
            scale = quant_state['scale']
            zero_point = quant_state['zero_point']
            return (quantized.float() - zero_point) * scale
        
        else:
            return quantized
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Check for gradient overflow if using gradient scaling
        if self.defaults['use_gradient_scaling']:
            has_overflow = self._check_gradient_overflow()
            if has_overflow:
                self._handle_gradient_overflow()
                return loss
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if p.grad.is_sparse:
                    raise RuntimeError('PagedAdamW does not support sparse gradients')
                
                params_with_grad.append(p)
                
                # Apply gradient scaling if enabled
                grad = p.grad
                if self.defaults['use_gradient_scaling']:
                    grad = grad / self.scale
                grads.append(grad)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    # Initialize optimizer states
                    param_id = id(p)
                    size = p.numel()
                    
                    if self.defaults['quantize_stats']:
                        # Allocate quantized states
                        exp_avg = self._allocate_page(param_id, size)
                        exp_avg_sq = self._allocate_page(param_id, size)
                        
                        # Store quantized states
                        state['exp_avg'] = exp_avg
                        state['exp_avg_sq'] = exp_avg_sq
                        state['exp_avg_quantized'] = False
                        state['exp_avg_sq_quantized'] = False
                        state['exp_avg_quant_state'] = {}
                        state['exp_avg_sq_quant_state'] = {}
                    else:
                        # Allocate regular states
                        exp_avg = self._allocate_page(param_id, size)
                        exp_avg_sq = self._allocate_page(param_id, size)
                        state['exp_avg'] = exp_avg
                        state['exp_avg_sq'] = exp_avg_sq
                    
                    if group['amsgrad']:
                        max_exp_avg_sq = self._allocate_page(param_id, size)
                        state['max_exp_avg_sq'] = max_exp_avg_sq
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                
                state['step'] += 1
                state_steps.append(state['step'])
            
            # Perform optimization step
            self._adamw_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group
            )
        
        # Update gradient scaling
        if self.defaults['use_gradient_scaling']:
            self._update_gradient_scale()
        
        return loss
    
    def _check_gradient_overflow(self) -> bool:
        """Check if any gradient has overflowed."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if torch.any(torch.isinf(p.grad)) or torch.any(torch.isnan(p.grad)):
                        return True
        return False
    
    def _handle_gradient_overflow(self):
        """Handle gradient overflow by reducing scale."""
        self.scale *= self.defaults['scale_backoff_factor']
        self.scale = max(self.scale, self.defaults['min_scale'])
        self._recent_overflow = True
        self.scale_backoff_tracker += 1
        
        # Skip the optimization step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
    
    def _update_gradient_scale(self):
        """Update gradient scale based on recent history."""
        if self._recent_overflow:
            self._recent_overflow = False
            return
        
        self.scale_growth_tracker += 1
        
        if self.scale_growth_tracker >= self._growth_interval:
            # Increase scale
            self.scale *= self.defaults['scale_growth_factor']
            self.scale = min(self.scale, self.defaults['max_scale'])
            self.scale_growth_tracker = 0
    
    def _adamw_step(
        self,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        state_steps: List[int],
        group: Dict[str, Any]
    ):
        """Perform AdamW optimization step with quantization support."""
        beta1, beta2 = group['betas']
        
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            # Dequantize states if they are quantized
            state = self.state[param]
            if state.get('exp_avg_quantized', False):
                exp_avg = self._dequantize_state(exp_avg, state['exp_avg_quant_state'])
                state['exp_avg_quantized'] = False
            
            if state.get('exp_avg_sq_quantized', False):
                exp_avg_sq = self._dequantize_state(exp_avg_sq, state['exp_avg_sq_quant_state'])
                state['exp_avg_sq_quantized'] = False
            
            # Perform AdamW update
            if self.defaults['use_cuda_kernels'] and self._cuda_kernels is not None:
                # Use custom CUDA kernels for efficiency
                self._cuda_kernels.adamw_update(
                    param, grad, exp_avg, exp_avg_sq,
                    beta1, beta2, group['lr'], group['weight_decay'],
                    group['eps'], step
                )
            else:
                # Fallback to PyTorch implementation
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute step size
                step_size = group['lr'] / bias_correction1
                
                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    param.add_(param, alpha=-group['weight_decay'] * group['lr'])
                
                # Update parameters
                param.addcdiv_(exp_avg, denom, value=-step_size)
            
            # Re-quantize states if quantization is enabled
            if self.defaults['quantize_stats']:
                quant_type = group['quant_type']
                if quant_type != QuantizationType.NONE:
                    # Quantize exp_avg
                    quantized_exp_avg, quant_state_exp_avg = self._quantize_state(exp_avg, quant_type)
                    state['exp_avg'] = quantized_exp_avg
                    state['exp_avg_quant_state'] = quant_state_exp_avg
                    state['exp_avg_quantized'] = True
                    
                    # Quantize exp_avg_sq
                    quantized_exp_avg_sq, quant_state_exp_avg_sq = self._quantize_state(exp_avg_sq, quant_type)
                    state['exp_avg_sq'] = quantized_exp_avg_sq
                    state['exp_avg_sq_quant_state'] = quant_state_exp_avg_sq
                    state['exp_avg_sq_quantized'] = True
    
    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients of all optimized parameters."""
        super().zero_grad(set_to_none=set_to_none)
        
        # Reset gradient scaling trackers if needed
        if self.defaults['use_gradient_scaling'] and self._recent_overflow:
            self._recent_overflow = False
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict."""
        state_dict = super().state_dict()
        
        # Add optimizer-specific state
        state_dict['paged_adamw_state'] = {
            'scale': self.scale,
            'scale_growth_tracker': self.scale_growth_tracker,
            'scale_backoff_tracker': self.scale_backoff_tracker,
            '_recent_overflow': self._recent_overflow,
            '_current_memory_usage': self._current_memory_usage,
            '_page_table': self._page_table,
            '_free_pages': self._free_pages,
        }
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from a state dict."""
        # Load optimizer-specific state
        if 'paged_adamw_state' in state_dict:
            paged_state = state_dict['paged_adamw_state']
            self.scale = paged_state['scale']
            self.scale_growth_tracker = paged_state['scale_growth_tracker']
            self.scale_backoff_tracker = paged_state['scale_backoff_tracker']
            self._recent_overflow = paged_state['_recent_overflow']
            self._current_memory_usage = paged_state['_current_memory_usage']
            self._page_table = paged_state['_page_table']
            self._free_pages = paged_state['_free_pages']
        
        # Load parent state
        super().load_state_dict(state_dict)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        return {
            'current_memory_bytes': self._current_memory_usage,
            'max_memory_bytes': self.max_memory,
            'page_size': self.page_size,
            'num_free_pages': len(self._free_pages),
            'num_paged_params': len(self._page_table),
        }
    
    def get_scaling_info(self) -> Dict[str, float]:
        """Get gradient scaling information."""
        return {
            'current_scale': self.scale,
            'min_scale': self.defaults['min_scale'],
            'max_scale': self.defaults['max_scale'],
            'growth_factor': self.defaults['scale_growth_factor'],
            'backoff_factor': self.defaults['scale_backoff_factor'],
            'recent_overflow': self._recent_overflow,
        }
    
    def __repr__(self) -> str:
        return (
            f"PagedAdamW("
            f"lr={self.defaults['lr']}, "
            f"betas={self.defaults['betas']}, "
            f"eps={self.defaults['eps']}, "
            f"weight_decay={self.defaults['weight_decay']}, "
            f"quant_type={self.defaults['quant_type']}, "
            f"paged={self.defaults['paged_optimizer']}, "
            f"gradient_scaling={self.defaults['use_gradient_scaling']}"
            f")"
        )


# Convenience function for creating optimizer with default QLoRA settings
def create_qlora_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    quant_type: Union[str, QuantizationType] = QuantizationType.NF4,
    **kwargs
) -> PagedAdamW:
    """
    Create a PagedAdamW optimizer configured for QLoRA training.
    
    This is a convenience function that sets up the optimizer with
    recommended settings for QLoRA fine-tuning.
    
    Args:
        model: The model to optimize
        lr: Learning rate (default: 2e-4)
        weight_decay: Weight decay coefficient (default: 0.01)
        quant_type: Quantization type (default: NF4)
        **kwargs: Additional arguments to pass to PagedAdamW
    
    Returns:
        Configured PagedAdamW optimizer
    """
    # Get all parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Set up optimizer with QLoRA defaults
    optimizer = PagedAdamW(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        quant_type=quant_type,
        quantize_stats=True,
        double_quant=True,
        compress_statistics=True,
        paged_optimizer=True,
        page_size=4096,
        use_gradient_scaling=True,
        dynamic_scaling=True,
        initial_scale=2**16,
        scale_window=1000,
        use_cuda_kernels=True,
        optimize_memory=True,
        **kwargs
    )
    
    return optimizer


# Example usage in training loop
if __name__ == "__main__":
    # Example model
    model = nn.Linear(100, 100)
    
    # Create optimizer
    optimizer = create_qlora_optimizer(
        model,
        lr=2e-4,
        quant_type="nf4"
    )
    
    # Example training step
    x = torch.randn(32, 100)
    y = torch.randn(32, 100)
    
    for epoch in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        print(f"Memory usage: {optimizer.get_memory_usage()}")
        print(f"Scaling info: {optimizer.get_scaling_info()}")
        print("---")