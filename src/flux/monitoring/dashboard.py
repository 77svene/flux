"""
Real-time Training Dashboard with Weights & Biases Integration for flux.

This module provides comprehensive training visualization with live loss curves,
gradient norms, memory usage, and hyperparameter tracking. Includes custom W&B
callbacks for LLM-specific metrics with decorator-based monitoring system.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from enum import Enum
import psutil
import GPUtil

# Third-party imports with fallback handling
try:
    import wandb
    from wandb.sdk.data_types.base_types import WBValue
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked by the dashboard."""
    LOSS = "loss"
    GRADIENT = "gradient"
    MEMORY = "memory"
    LEARNING_RATE = "learning_rate"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


@dataclass
class MonitoringConfig:
    """Configuration for the training dashboard."""
    project_name: str = "flux-training"
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["llm", "training"])
    log_interval: int = 10  # Steps between logging
    save_interval: int = 100  # Steps between model checkpoints
    gradient_logging: bool = True
    memory_logging: bool = True
    system_metrics: bool = True
    model_architecture: bool = True
    hyperparameters: bool = True
    custom_metrics: List[str] = field(default_factory=list)
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", "disabled"
    resume_from: Optional[str] = None
    group_name: Optional[str] = None
    job_type: Optional[str] = "training"
    notes: Optional[str] = None
    config_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for W&B."""
        config_dict = asdict(self)
        # Remove non-serializable fields if any
        return config_dict


@dataclass
class TrainingMetrics:
    """Container for training metrics at a given step."""
    step: int
    epoch: Optional[float] = None
    global_step: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    gradient_norms_by_layer: Optional[Dict[str, float]] = None
    memory_allocated: Optional[float] = None
    memory_reserved: Optional[float] = None
    memory_used_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_wandb_dict(self) -> Dict[str, Any]:
        """Convert metrics to W&B loggable dictionary."""
        metrics_dict = {}
        
        # Add standard metrics
        if self.loss is not None:
            metrics_dict["train/loss"] = self.loss
        if self.learning_rate is not None:
            metrics_dict["train/learning_rate"] = self.learning_rate
        if self.gradient_norm is not None:
            metrics_dict["train/gradient_norm"] = self.gradient_norm
        if self.epoch is not None:
            metrics_dict["train/epoch"] = self.epoch
        if self.global_step is not None:
            metrics_dict["train/global_step"] = self.global_step
            
        # Add memory metrics
        if self.memory_allocated is not None:
            metrics_dict["system/memory_allocated_gb"] = self.memory_allocated
        if self.memory_reserved is not None:
            metrics_dict["system/memory_reserved_gb"] = self.memory_reserved
        if self.memory_used_percent is not None:
            metrics_dict["system/memory_used_percent"] = self.memory_used_percent
        if self.gpu_utilization is not None:
            metrics_dict["system/gpu_utilization"] = self.gpu_utilization
            
        # Add throughput metrics
        if self.throughput_samples_per_sec is not None:
            metrics_dict["performance/throughput_samples_per_sec"] = self.throughput_samples_per_sec
        if self.throughput_tokens_per_sec is not None:
            metrics_dict["performance/throughput_tokens_per_sec"] = self.throughput_tokens_per_sec
            
        # Add gradient norms by layer
        if self.gradient_norms_by_layer:
            for layer_name, norm in self.gradient_norms_by_layer.items():
                metrics_dict[f"gradients/{layer_name}"] = norm
                
        # Add custom metrics
        for key, value in self.custom_metrics.items():
            metrics_dict[f"custom/{key}"] = value
            
        return metrics_dict


class GradientMonitor:
    """Monitor and compute gradient statistics."""
    
    def __init__(self, model: Optional[Any] = None):
        self.model = model
        self.gradient_norms_history = []
        
    def compute_gradient_norm(self, model: Any = None) -> Tuple[float, Dict[str, float]]:
        """Compute total gradient norm and per-layer norms."""
        if model is None:
            model = self.model
            
        if model is None or not TORCH_AVAILABLE:
            return 0.0, {}
            
        total_norm = 0.0
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                layer_norms[name] = param_norm.item()
                
        total_norm = total_norm ** 0.5
        return total_norm, layer_norms
    
    def log_gradient_histogram(self, step: int, wandb_run: Any = None):
        """Log gradient histograms to W&B."""
        if not TORCH_AVAILABLE or self.model is None or wandb_run is None:
            return
            
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.data.cpu().numpy()
                
        if gradients and WANDB_AVAILABLE:
            wandb_run.log({f"gradients/{name}": wandb.Histogram(grad) 
                          for name, grad in gradients.items()}, step=step)


class MemoryMonitor:
    """Monitor system and GPU memory usage."""
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        # System memory
        memory = psutil.virtual_memory()
        stats["system_memory_used_percent"] = memory.percent
        stats["system_memory_available_gb"] = memory.available / (1024 ** 3)
        
        # GPU memory if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming single GPU
                stats["gpu_memory_used_gb"] = gpu.memoryUsed / 1024
                stats["gpu_memory_total_gb"] = gpu.memoryTotal / 1024
                stats["gpu_memory_used_percent"] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                stats["gpu_utilization"] = gpu.load * 100
        except:
            pass
            
        # PyTorch CUDA memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            stats["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
            stats["cuda_max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
        return stats


class TrainingDashboard:
    """Main dashboard class for real-time training monitoring with W&B integration."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.run = None
        self.gradient_monitor = GradientMonitor()
        self.memory_monitor = MemoryMonitor()
        self.metrics_history = []
        self.current_step = 0
        self.start_time = None
        self._lock = threading.Lock()
        self._system_metrics_thread = None
        self._stop_system_metrics = threading.Event()
        
        if not WANDB_AVAILABLE and config.wandb_mode != "disabled":
            logger.warning("Weights & Biases not installed. Dashboard will run in limited mode.")
            
    def initialize(self, model: Optional[Any] = None, 
                  hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the dashboard and W&B run."""
        if self.config.wandb_mode == "disabled":
            logger.info("Dashboard disabled by configuration")
            return
            
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, running in local-only mode")
            return
            
        # Prepare config for W&B
        wandb_config = self.config.to_dict()
        if hyperparameters:
            wandb_config.update(hyperparameters)
            
        # Generate experiment name if not provided
        experiment_name = self.config.experiment_name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"flux_{timestamp}"
            
        # Initialize W&B run
        self.run = wandb.init(
            project=self.config.project_name,
            name=experiment_name,
            entity=self.config.wandb_entity,
            config=wandb_config,
            tags=self.config.tags,
            group=self.config.group_name,
            job_type=self.config.job_type,
            notes=self.config.notes,
            resume=self.config.resume_from,
            mode=self.config.wandb_mode,
            save_code=True
        )
        
        # Watch model for gradient/parameter logging
        if model is not None and self.config.gradient_logging:
            self.gradient_monitor.model = model
            if hasattr(wandb, 'watch'):
                wandb.watch(model, log="all", log_freq=self.config.log_interval)
                
        # Log model architecture
        if model is not None and self.config.model_architecture:
            self._log_model_architecture(model)
            
        # Start system metrics logging thread
        if self.config.system_metrics:
            self._start_system_metrics_logging()
            
        self.start_time = time.time()
        logger.info(f"Dashboard initialized for experiment: {experiment_name}")
        
    def _log_model_architecture(self, model: Any) -> None:
        """Log model architecture summary to W&B."""
        if not WANDB_AVAILABLE or self.run is None:
            return
            
        try:
            if TORCH_AVAILABLE and hasattr(model, 'named_parameters'):
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                architecture_summary = {
                    "model_type": model.__class__.__name__,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "non_trainable_parameters": total_params - trainable_params,
                }
                
                # Add layer information
                layer_info = []
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Leaf modules
                        layer_info.append({
                            "name": name,
                            "type": module.__class__.__name__,
                            "parameters": sum(p.numel() for p in module.parameters())
                        })
                        
                architecture_summary["layers"] = layer_info[:50]  # Limit to first 50 layers
                self.run.summary["model_architecture"] = architecture_summary
                
        except Exception as e:
            logger.warning(f"Failed to log model architecture: {e}")
            
    def _start_system_metrics_logging(self) -> None:
        """Start background thread for system metrics logging."""
        def log_system_metrics():
            while not self._stop_system_metrics.is_set():
                try:
                    if self.run and WANDB_AVAILABLE:
                        memory_stats = self.memory_monitor.get_memory_stats()
                        system_metrics = {f"system/{k}": v for k, v in memory_stats.items()}
                        system_metrics["system/timestamp"] = time.time()
                        wandb.log(system_metrics)
                except Exception as e:
                    logger.debug(f"Error logging system metrics: {e}")
                    
                time.sleep(5)  # Log system metrics every 5 seconds
                
        self._system_metrics_thread = threading.Thread(
            target=log_system_metrics, 
            daemon=True,
            name="system-metrics-logger"
        )
        self._system_metrics_thread.start()
        
    def log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics to dashboard and W&B."""
        with self._lock:
            self.current_step = metrics.step
            self.metrics_history.append(metrics)
            
            # Check if we should log based on interval
            if metrics.step % self.config.log_interval != 0:
                return
                
            if self.run and WANDB_AVAILABLE:
                # Prepare metrics for W&B
                wandb_metrics = metrics.to_wandb_dict()
                
                # Add timing information
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    wandb_metrics["timing/elapsed_seconds"] = elapsed
                    if metrics.throughput_samples_per_sec:
                        wandb_metrics["timing/estimated_remaining_seconds"] = (
                            (1000000 - metrics.step) / metrics.throughput_samples_per_sec
                            if metrics.throughput_samples_per_sec > 0 else 0
                        )
                
                # Log to W&B
                self.run.log(wandb_metrics, step=metrics.step)
                
            # Log to local logger
            log_msg = f"Step {metrics.step}"
            if metrics.loss is not None:
                log_msg += f" | Loss: {metrics.loss:.4f}"
            if metrics.learning_rate is not None:
                log_msg += f" | LR: {metrics.learning_rate:.2e}"
            if metrics.gradient_norm is not None:
                log_msg += f" | Grad Norm: {metrics.gradient_norm:.4f}"
                
            logger.info(log_msg)
            
    def log_custom_metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Log a custom metric to the dashboard."""
        if step is None:
            step = self.current_step
            
        if self.run and WANDB_AVAILABLE:
            self.run.log({f"custom/{name}": value}, step=step)
            
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to W&B config."""
        if self.run and WANDB_AVAILABLE:
            self.run.config.update(hparams, allow_val_change=True)
            
    def save_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint and log to W&B."""
        if self.run and WANDB_AVAILABLE:
            try:
                artifact = wandb.Artifact(
                    name=f"model_checkpoint_step_{self.current_step}",
                    type="model",
                    metadata=metadata or {}
                )
                artifact.add_file(checkpoint_path)
                self.run.log_artifact(artifact)
                logger.info(f"Checkpoint saved to W&B: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint to W&B: {e}")
                
    def create_summary_table(self, metrics_list: List[TrainingMetrics]) -> Any:
        """Create a W&B table summarizing training metrics."""
        if not WANDB_AVAILABLE or not metrics_list:
            return None
            
        columns = ["step", "loss", "learning_rate", "gradient_norm", "memory_used"]
        data = []
        
        for metrics in metrics_list:
            if metrics.step % (self.config.log_interval * 10) == 0:  # Sample every 10 log intervals
                data.append([
                    metrics.step,
                    metrics.loss,
                    metrics.learning_rate,
                    metrics.gradient_norm,
                    metrics.memory_allocated
                ])
                
        return wandb.Table(columns=columns, data=data)
    
    def finish(self, exit_code: int = 0) -> None:
        """Finish the dashboard and W&B run."""
        self._stop_system_metrics.set()
        
        if self._system_metrics_thread:
            self._system_metrics_thread.join(timeout=5)
            
        if self.run and WANDB_AVAILABLE:
            # Log final summary
            if self.metrics_history:
                summary_table = self.create_summary_table(self.metrics_history)
                if summary_table:
                    self.run.log({"training_summary": summary_table})
                    
                # Log final metrics
                final_metrics = self.metrics_history[-1]
                self.run.summary.update({
                    "final_loss": final_metrics.loss,
                    "total_steps": final_metrics.step,
                    "total_training_time": time.time() - self.start_time if self.start_time else 0
                })
                
            self.run.finish(exit_code=exit_code)
            logger.info("Dashboard finished")
            
        self.run = None


def monitor_training(
    project_name: str = "flux-training",
    experiment_name: Optional[str] = None,
    log_interval: int = 10,
    tags: Optional[List[str]] = None,
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
    **kwargs
) -> Callable:
    """
    Decorator for monitoring training functions with W&B integration.
    
    Args:
        project_name: W&B project name
        experiment_name: Name for this experiment run
        log_interval: Steps between metric logging
        tags: Tags for the W&B run
        wandb_entity: W&B entity (username or team)
        wandb_mode: W&B mode ("online", "offline", "disabled")
        **kwargs: Additional configuration parameters
        
    Returns:
        Decorated function with monitoring capabilities
    """
    def decorator(train_func: Callable) -> Callable:
        @wraps(train_func)
        def wrapper(*args, **kwargs_inner):
            # Extract model and hyperparameters from function arguments if available
            model = None
            hyperparameters = {}
            
            # Try to find model in arguments
            for arg in args:
                if TORCH_AVAILABLE and hasattr(arg, 'parameters') and hasattr(arg, 'named_parameters'):
                    model = arg
                    break
                    
            # Try to find hyperparameters in kwargs
            for key in ['config', 'training_args', 'hyperparameters', 'args']:
                if key in kwargs_inner:
                    hyperparameters.update(kwargs_inner[key] if isinstance(kwargs_inner[key], dict) else {})
                    
            # Create monitoring configuration
            config = MonitoringConfig(
                project_name=project_name,
                experiment_name=experiment_name,
                tags=tags or ["llm", "training"],
                log_interval=log_interval,
                wandb_entity=wandb_entity,
                wandb_mode=wandb_mode,
                **kwargs
            )
            
            # Initialize dashboard
            dashboard = TrainingDashboard(config)
            dashboard.initialize(model=model, hyperparameters=hyperparameters)
            
            # Add dashboard to function kwargs
            kwargs_inner['dashboard'] = dashboard
            kwargs_inner['log_metrics'] = dashboard.log_metrics
            
            try:
                # Execute training function
                result = train_func(*args, **kwargs_inner)
                
                # Log final results if available
                if isinstance(result, dict) and 'metrics' in result:
                    dashboard.log_custom_metric("final_results", result['metrics'])
                    
                return result
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                if dashboard.run:
                    dashboard.run.alert(title="Training Failed", text=str(e))
                raise
                
            finally:
                dashboard.finish()
                
        return wrapper
    return decorator


def create_dashboard_from_config(config_path: str) -> TrainingDashboard:
    """Create a dashboard instance from a configuration file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    config = MonitoringConfig(**config_dict)
    return TrainingDashboard(config)


# Example usage and integration helpers
class DashboardCallback:
    """Callback class for integration with training loops."""
    
    def __init__(self, dashboard: TrainingDashboard):
        self.dashboard = dashboard
        self.last_log_time = time.time()
        self.samples_since_last_log = 0
        
    def on_step_end(self, step: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each training step."""
        # Convert metrics dict to TrainingMetrics
        training_metrics = TrainingMetrics(
            step=step,
            loss=metrics.get('loss'),
            learning_rate=metrics.get('learning_rate'),
            gradient_norm=metrics.get('gradient_norm'),
            custom_metrics={k: v for k, v in metrics.items() 
                          if k not in ['loss', 'learning_rate', 'gradient_norm']}
        )
        
        self.dashboard.log_metrics(training_metrics)
        self.samples_since_last_log += metrics.get('batch_size', 1)
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        self.dashboard.log_custom_metric(f"epoch_{epoch}_metrics", metrics)
        
    def on_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called at the end of training."""
        self.dashboard.log_custom_metric("final_training_metrics", final_metrics)


# Utility functions for common monitoring patterns
def log_model_summary(model: Any, dashboard: TrainingDashboard) -> None:
    """Log detailed model summary to dashboard."""
    if not TORCH_AVAILABLE or not hasattr(model, 'named_parameters'):
        return
        
    summary = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layer_count": len(list(model.named_modules())),
    }
    
    # Add layer-wise parameter counts
    layer_params = {}
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0] if '.' in name else name
        layer_params[layer_name] = layer_params.get(layer_name, 0) + param.numel()
        
    summary["parameters_by_layer"] = layer_params
    dashboard.log_custom_metric("model_summary", summary)


def create_comparison_dashboard(experiment_names: List[str], 
                               project_name: str = "flux-comparison") -> Optional[Any]:
    """Create a dashboard comparing multiple experiments."""
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available for comparison dashboard")
        return None
        
    api = wandb.Api()
    runs = []
    
    for exp_name in experiment_names:
        try:
            run = api.run(f"{project_name}/{exp_name}")
            runs.append(run)
        except Exception as e:
            logger.warning(f"Could not fetch run {exp_name}: {e}")
            
    if not runs:
        return None
        
    # Create comparison plots would be done in W&B UI
    # This function returns the runs for further processing
    return runs