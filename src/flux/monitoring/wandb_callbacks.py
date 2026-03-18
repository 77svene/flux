"""
src/flux/monitoring/wandb_callbacks.py

Real-time Training Dashboard with Weights & Biases Integration

This module provides comprehensive training visualization with live loss curves,
gradient norms, memory usage, and hyperparameter tracking. Includes custom
W&B callbacks for LLM-specific metrics with decorator-based monitoring system
that captures training metrics at configurable intervals, with automatic
experiment grouping and model versioning.
"""

import os
import time
import functools
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np

try:
    import wandb
    from wandb.sdk.data_types import Histogram
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    # Create mock classes for type hints when wandb is not available
    class Histogram:
        pass

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_callback import TrainerCallback


class MonitoringLevel(Enum):
    """Monitoring intensity levels for different use cases."""
    MINIMAL = "minimal"      # Only essential metrics (loss, learning rate)
    STANDARD = "standard"    # Standard metrics (loss, gradients, memory)
    DETAILED = "detailed"    # Detailed metrics (all above + activations, weights)
    DEBUG = "debug"          # Debug level with maximum verbosity


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases monitoring."""
    project: str = "flux"
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    save_code: bool = True
    log_model: bool = True
    log_gradients: bool = True
    log_parameters: bool = True
    log_memory: bool = True
    log_learning_rate: bool = True
    log_gpu_utilization: bool = True
    log_system_metrics: bool = True
    log_frequency: int = 10  # Log every N steps
    log_epoch_frequency: int = 1  # Log every N epochs
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
    watch_model: bool = True
    watch_frequency: int = 1000  # Log model parameters every N steps
    reinit: bool = True
    resume: Optional[str] = None  # "allow", "must", "never", or None
    anonymous: Optional[str] = None  # "allow", "must", "never"
    mode: Optional[str] = None  # "online", "offline", "disabled"


class WandbMetricsLogger:
    """Core metrics logging functionality for W&B integration."""
    
    def __init__(self, config: WandbConfig, model: Optional[torch.nn.Module] = None):
        self.config = config
        self.model = model
        self.initialized = False
        self.run = None
        self.step_counter = 0
        self.epoch_counter = 0
        self._lock = threading.Lock()
        
        if not WANDB_AVAILABLE:
            raise ImportError(
                "Weights & Biases (wandb) is required for monitoring. "
                "Install with: pip install wandb"
            )
    
    def init_run(self, training_args: Optional[Dict[str, Any]] = None):
        """Initialize W&B run with configuration."""
        if self.initialized:
            return
        
        with self._lock:
            if self.initialized:
                return
                
            # Prepare config for W&B
            wandb_config = {
                "monitoring_level": self.config.monitoring_level.value,
                "log_frequency": self.config.log_frequency,
                "log_gradients": self.config.log_gradients,
                "log_parameters": self.config.log_parameters,
            }
            
            if training_args:
                wandb_config.update(training_args)
            
            # Initialize W&B run
            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                group=self.config.group,
                job_type=self.config.job_type,
                tags=self.config.tags,
                notes=self.config.notes,
                config=wandb_config,
                save_code=self.config.save_code,
                reinit=self.config.reinit,
                resume=self.config.resume,
                anonymous=self.config.anonymous,
                mode=self.config.mode,
            )
            
            # Watch model if enabled
            if self.config.watch_model and self.model is not None:
                wandb.watch(
                    self.model,
                    log="all" if self.config.monitoring_level == MonitoringLevel.DEBUG else "gradients",
                    log_freq=self.config.watch_frequency,
                    log_graph=(self.config.monitoring_level in [MonitoringLevel.DETAILED, MonitoringLevel.DEBUG])
                )
            
            self.initialized = True
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True
    ):
        """Log metrics to W&B."""
        if not self.initialized or self.config.mode == "disabled":
            return
        
        with self._lock:
            if step is None:
                step = self.step_counter
                self.step_counter += 1
            
            wandb.log(metrics, step=step, commit=commit)
    
    def log_training_step(
        self,
        loss: torch.Tensor,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
        batch_size: Optional[int] = None,
        step: Optional[int] = None,
        epoch: Optional[float] = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log training step metrics."""
        if not self.initialized or self.config.mode == "disabled":
            return
        
        # Check if we should log at this step
        if step is not None and step % self.config.log_frequency != 0:
            return
        
        metrics = {
            "train/loss": loss.item(),
            "train/learning_rate": learning_rate,
            "train/step": step if step is not None else self.step_counter,
        }
        
        if epoch is not None:
            metrics["train/epoch"] = epoch
        
        if gradient_norm is not None and self.config.log_gradients:
            metrics["train/gradient_norm"] = gradient_norm
        
        if batch_size is not None:
            metrics["train/batch_size"] = batch_size
        
        # Add custom metrics
        if custom_metrics:
            for key, value in custom_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[f"custom/{key}"] = value.item()
                else:
                    metrics[f"custom/{key}"] = value
        
        # Log memory usage if enabled
        if self.config.log_memory:
            memory_metrics = self._get_memory_metrics()
            metrics.update(memory_metrics)
        
        # Log GPU utilization if enabled
        if self.config.log_gpu_utilization and torch.cuda.is_available():
            gpu_metrics = self._get_gpu_metrics()
            metrics.update(gpu_metrics)
        
        self.log_metrics(metrics, step=step)
    
    def log_validation_step(
        self,
        eval_loss: float,
        metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None,
        epoch: Optional[float] = None
    ):
        """Log validation metrics."""
        if not self.initialized or self.config.mode == "disabled":
            return
        
        log_metrics = {
            "eval/loss": eval_loss,
        }
        
        if epoch is not None:
            log_metrics["eval/epoch"] = epoch
        
        if metrics:
            for key, value in metrics.items():
                log_metrics[f"eval/{key}"] = value
        
        self.log_metrics(log_metrics, step=step)
    
    def log_model_checkpoint(
        self,
        checkpoint_path: str,
        metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None
    ):
        """Log model checkpoint to W&B."""
        if not self.initialized or not self.config.log_model:
            return
        
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            metadata=metrics or {}
        )
        artifact.add_dir(checkpoint_path)
        wandb.log_artifact(artifact)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters to W&B."""
        if not self.initialized:
            return
        
        wandb.config.update(hparams, allow_val_change=True)
    
    def log_system_metrics(self):
        """Log system-level metrics."""
        if not self.initialized or not self.config.log_system_metrics:
            return
        
        system_metrics = {}
        
        # CPU metrics
        try:
            import psutil
            system_metrics["system/cpu_percent"] = psutil.cpu_percent()
            system_metrics["system/memory_percent"] = psutil.virtual_memory().percent
            system_metrics["system/disk_usage"] = psutil.disk_usage('/').percent
        except ImportError:
            pass
        
        # GPU metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem = torch.cuda.get_device_properties(i).total_memory
                gpu_mem_allocated = torch.cuda.memory_allocated(i)
                gpu_mem_cached = torch.cuda.memory_reserved(i)
                
                system_metrics[f"system/gpu_{i}_memory_total"] = gpu_mem
                system_metrics[f"system/gpu_{i}_memory_allocated"] = gpu_mem_allocated
                system_metrics[f"system/gpu_{i}_memory_cached"] = gpu_mem_cached
                system_metrics[f"system/gpu_{i}_memory_utilization"] = (
                    gpu_mem_allocated / gpu_mem * 100 if gpu_mem > 0 else 0
                )
        
        self.log_metrics(system_metrics, commit=False)
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage metrics."""
        metrics = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics[f"memory/gpu_{i}_allocated"] = torch.cuda.memory_allocated(i) / 1024**2  # MB
                metrics[f"memory/gpu_{i}_cached"] = torch.cuda.memory_reserved(i) / 1024**2  # MB
                metrics[f"memory/gpu_{i}_max_allocated"] = torch.cuda.max_memory_allocated(i) / 1024**2  # MB
        
        # CPU memory
        try:
            import psutil
            process = psutil.Process()
            metrics["memory/cpu_rss"] = process.memory_info().rss / 1024**2  # MB
            metrics["memory/cpu_percent"] = process.memory_percent()
        except ImportError:
            pass
        
        return metrics
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization metrics."""
        metrics = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # This requires pynvml
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f"gpu_{i}_utilization"] = util.gpu
                    metrics[f"gpu_{i}_memory_utilization"] = util.memory
                    pynvml.nvmlShutdown()
                except (ImportError, Exception):
                    # Fallback to basic metrics
                    metrics[f"gpu_{i}_memory_allocated"] = torch.cuda.memory_allocated(i) / 1024**2
        
        return metrics
    
    def finish(self):
        """Finish the W&B run."""
        if self.initialized and self.run is not None:
            wandb.finish()
            self.initialized = False


class WandbTrainerCallback(TrainerCallback):
    """Hugging Face Trainer callback for W&B integration."""
    
    def __init__(
        self,
        config: WandbConfig,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = WandbMetricsLogger(config, model)
        self.training_start_time = None
        self.epoch_start_time = None
        self.step_start_time = None
        
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of initialization."""
        # Prepare training args for W&B config
        training_config = {
            "model_name": getattr(args, "model_name_or_path", "unknown"),
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "optim": args.optim,
            "lr_scheduler_type": str(args.lr_scheduler_type),
        }
        
        # Add model-specific config if available
        if self.model is not None:
            training_config["model_parameters"] = sum(p.numel() for p in self.model.parameters())
            training_config["trainable_parameters"] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        
        self.logger.init_run(training_config)
        self.logger.log_hyperparameters(training_config)
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        self.training_start_time = time.time()
        
        # Log initial system metrics
        if self.config.log_system_metrics:
            self.logger.log_system_metrics()
        
        # Log model architecture if in debug mode
        if self.config.monitoring_level == MonitoringLevel.DEBUG and self.model is not None:
            model_summary = self._get_model_summary()
            wandb.run.summary["model_architecture"] = model_summary
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of an epoch."""
        self.epoch_start_time = time.time()
        self.logger.epoch_counter = state.epoch
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of a training step."""
        self.step_start_time = time.time()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of a training step."""
        # Log training metrics at configured frequency
        if state.global_step % self.config.log_frequency == 0:
            # Calculate step time
            step_time = time.time() - self.step_start_time if self.step_start_time else 0
            
            # Get current loss from state
            loss = state.loss if hasattr(state, 'loss') else None
            
            # Calculate gradient norm if enabled
            gradient_norm = None
            if self.config.log_gradients and self.model is not None:
                gradient_norm = self._calculate_gradient_norm()
            
            # Log metrics
            self.logger.log_training_step(
                loss=torch.tensor(loss) if loss is not None else torch.tensor(0.0),
                learning_rate=self._get_learning_rate(args, state),
                gradient_norm=gradient_norm,
                step=state.global_step,
                epoch=state.epoch,
                custom_metrics={
                    "step_time": step_time,
                    "samples_per_second": args.per_device_train_batch_size / step_time if step_time > 0 else 0,
                }
            )
            
            # Log system metrics periodically
            if state.global_step % (self.config.log_frequency * 10) == 0 and self.config.log_system_metrics:
                self.logger.log_system_metrics()
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        """Called after evaluation."""
        # Log evaluation metrics
        eval_loss = metrics.get("eval_loss", 0.0)
        
        # Extract all eval metrics
        eval_metrics = {k: v for k, v in metrics.items() if k.startswith("eval_")}
        
        self.logger.log_validation_step(
            eval_loss=eval_loss,
            metrics=eval_metrics,
            step=state.global_step,
            epoch=state.epoch
        )
        
        # Calculate and log perplexity if we have loss
        if "eval_loss" in metrics:
            perplexity = np.exp(metrics["eval_loss"])
            wandb.log({"eval/perplexity": perplexity}, step=state.global_step)
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when saving a checkpoint."""
        if self.config.log_model:
            checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            if os.path.exists(checkpoint_dir):
                self.logger.log_model_checkpoint(
                    checkpoint_path=checkpoint_dir,
                    metrics={"step": state.global_step, "epoch": state.epoch},
                    step=state.global_step
                )
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        # Calculate total training time
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            wandb.run.summary["total_training_time"] = total_time
            wandb.run.summary["total_steps"] = state.global_step
            wandb.run.summary["total_epochs"] = state.epoch
        
        # Log final metrics
        if hasattr(state, 'log_history') and state.log_history:
            final_metrics = state.log_history[-1]
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    wandb.run.summary[f"final_{key}"] = value
        
        # Finish W&B run
        self.logger.finish()
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Called when logging."""
        # This is called by the trainer's logging mechanism
        # We can use this to capture additional metrics
        pass
    
    def _calculate_gradient_norm(self) -> float:
        """Calculate the gradient norm of model parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _get_learning_rate(self, args: TrainingArguments, state: TrainerState) -> float:
        """Get current learning rate."""
        # This is a simplified version; in practice, you might need to access the optimizer
        if hasattr(state, 'learning_rate'):
            return state.learning_rate
        return args.learning_rate
    
    def _get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        if self.model is None:
            return "No model available"
        
        summary = []
        summary.append(f"Model: {self.model.__class__.__name__}")
        summary.append(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        summary.append(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Add layer information for debug mode
        if self.config.monitoring_level == MonitoringLevel.DEBUG:
            summary.append("\nLayer details:")
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    params = sum(p.numel() for p in module.parameters())
                    if params > 0:
                        summary.append(f"  {name}: {params:,} parameters")
        
        return "\n".join(summary)


def monitor_with_wandb(
    config: Optional[WandbConfig] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    log_frequency: int = 10,
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
):
    """
    Decorator for monitoring training functions with W&B.
    
    Args:
        config: WandbConfig object. If provided, other parameters are ignored.
        project: W&B project name
        entity: W&B entity (team) name
        tags: List of tags for the run
        log_frequency: Log every N steps
        monitoring_level: Level of monitoring detail
        
    Returns:
        Decorated function with W&B monitoring
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not WANDB_AVAILABLE:
                # If wandb is not available, just run the function
                return func(*args, **kwargs)
            
            # Create config if not provided
            nonlocal config
            if config is None:
                config = WandbConfig(
                    project=project or "flux",
                    entity=entity,
                    tags=tags or [],
                    log_frequency=log_frequency,
                    monitoring_level=monitoring_level
                )
            
            # Initialize W&B
            logger = WandbMetricsLogger(config)
            
            # Try to extract model from function arguments
            model = None
            for arg in args:
                if isinstance(arg, torch.nn.Module):
                    model = arg
                    break
            if model is None and 'model' in kwargs:
                model = kwargs['model']
            
            if model is not None:
                logger.model = model
            
            # Initialize run
            logger.init_run()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # If the function returns a dict, log it as final metrics
                if isinstance(result, dict):
                    logger.log_metrics(result, commit=True)
                
                return result
            finally:
                # Always finish the run
                logger.finish()
        
        return wrapper
    return decorator


class WandbModelVersioning:
    """Model versioning and experiment tracking with W&B."""
    
    def __init__(self, config: WandbConfig):
        self.config = config
        self.artifacts = {}
        
    def log_model_version(
        self,
        model: torch.nn.Module,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None
    ):
        """Log a model version to W&B."""
        if not WANDB_AVAILABLE or self.config.mode == "disabled":
            return
        
        # Create artifact
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=metadata or {},
            description=f"Model version {version}"
        )
        
        # Save model state dict
        model_path = f"/tmp/{model_name}_{version}.pt"
        torch.save(model.model.state_dict(), model_path)
        artifact.add_file(model_path)
        
        # Log artifact
        aliases = aliases or ["latest", version]
        wandb.log_artifact(artifact, aliases=aliases)
        
        # Clean up temporary file
        os.remove(model_path)
        
        # Store artifact reference
        self.artifacts[f"{model_name}_{version}"] = artifact
    
    def load_model_version(
        self,
        model: torch.nn.Module,
        model_name: str,
        version: str = "latest"
    ) -> torch.nn.Module:
        """Load a model version from W&B."""
        if not WANDB_AVAILABLE:
            return model
        
        # Get artifact
        artifact = wandb.use_artifact(f"{model_name}:{version}")
        artifact_dir = artifact.download()
        
        # Load model weights
        model_path = os.path.join(artifact_dir, f"{model_name}_{version}.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        
        return model


def create_wandb_callback(
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
    project: str = "flux",
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    log_frequency: int = 10,
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
    **kwargs
) -> WandbTrainerCallback:
    """
    Factory function to create a W&B callback with sensible defaults.
    
    Args:
        model: The model being trained
        tokenizer: The tokenizer (optional)
        project: W&B project name
        entity: W&B entity name
        tags: List of tags for the run
        log_frequency: Log every N steps
        monitoring_level: Level of monitoring detail
        **kwargs: Additional configuration options
        
    Returns:
        Configured WandbTrainerCallback instance
    """
    config = WandbConfig(
        project=project,
        entity=entity,
        tags=tags or [],
        log_frequency=log_frequency,
        monitoring_level=monitoring_level,
        **kwargs
    )
    
    return WandbTrainerCallback(
        config=config,
        model=model,
        tokenizer=tokenizer
    )


# Utility functions for common monitoring patterns
def log_llm_specific_metrics(
    logger: WandbMetricsLogger,
    logits: torch.Tensor,
    labels: torch.Tensor,
    step: Optional[int] = None
):
    """Log LLM-specific metrics like token accuracy and perplexity."""
    if not logger.initialized or logger.config.mode == "disabled":
        return
    
    with torch.no_grad():
        # Calculate token accuracy
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100  # Ignore padding tokens
        correct = (predictions[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate perplexity
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss).item()
        
        # Log metrics
        metrics = {
            "llm/token_accuracy": accuracy,
            "llm/perplexity": perplexity,
            "llm/num_tokens": total,
        }
        
        logger.log_metrics(metrics, step=step, commit=False)


def log_attention_metrics(
    logger: WandbMetricsLogger,
    attention_weights: torch.Tensor,
    step: Optional[int] = None
):
    """Log attention weight statistics."""
    if not logger.initialized or logger.config.mode == "disabled":
        return
    
    with torch.no_grad():
        # Calculate attention statistics
        attention_mean = attention_weights.mean().item()
        attention_std = attention_weights.std().item()
        attention_max = attention_weights.max().item()
        attention_min = attention_weights.min().item()
        
        # Calculate entropy (measure of attention spread)
        attention_entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-10),
            dim=-1
        ).mean().item()
        
        metrics = {
            "attention/mean": attention_mean,
            "attention/std": attention_std,
            "attention/max": attention_max,
            "attention/min": attention_min,
            "attention/entropy": attention_entropy,
        }
        
        logger.log_metrics(metrics, step=step, commit=False)


# Example usage in training script:
"""
from flux.monitoring.wandb_callbacks import (
    WandbTrainerCallback, WandbConfig, MonitoringLevel, create_wandb_callback
)

# Method 1: Using the factory function (recommended)
wandb_callback = create_wandb_callback(
    model=model,
    project="my-llm-project",
    tags=["llama2", "finetuning"],
    log_frequency=50,
    monitoring_level=MonitoringLevel.STANDARD
)

# Method 2: Using the decorator for custom training loops
@monitor_with_wandb(project="my-project", tags=["experiment1"])
def train_model(model, train_dataloader, optimizer, num_epochs):
    # Training loop here
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # Training step
            loss = model(batch)
            loss.backward()
            optimizer.step()
            
            # W&B logging happens automatically via decorator
    
    return {"final_loss": loss.item()}

# Method 3: Manual logging
config = WandbConfig(project="my-project", log_frequency=10)
logger = WandbMetricsLogger(config, model)
logger.init_run()

for step in range(num_steps):
    # Training step
    loss = train_step()
    
    # Manual logging
    logger.log_training_step(
        loss=loss,
        learning_rate=current_lr,
        step=step
    )
"""