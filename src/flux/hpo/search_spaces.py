"""
Hyperparameter Optimization Module for flux
Integrates Bayesian optimization with Optuna for automated hyperparameter selection.
"""

import logging
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for hyperparameter search."""
    ACCURACY = "accuracy"
    LOSS = "loss"
    SPEED = "speed"
    MEMORY = "memory"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class SearchSpaceConfig:
    """Configuration for hyperparameter search spaces."""
    
    # Learning rate parameters
    lr_min: float = 1e-6
    lr_max: float = 1e-3
    lr_log: bool = True
    
    # Batch size parameters
    batch_size_min: int = 1
    batch_size_max: int = 128
    batch_size_step: int = 2
    
    # LoRA parameters
    lora_rank_min: int = 4
    lora_rank_max: int = 64
    lora_rank_step: int = 4
    lora_alpha_min: float = 8.0
    lora_alpha_max: float = 64.0
    lora_dropout_min: float = 0.0
    lora_dropout_max: float = 0.5
    
    # Optimizer parameters
    weight_decay_min: float = 0.0
    weight_decay_max: float = 0.1
    adam_beta1_min: float = 0.8
    adam_beta1_max: float = 0.99
    adam_beta2_min: float = 0.9
    adam_beta2_max: float = 0.999
    
    # Training parameters
    warmup_ratio_min: float = 0.0
    warmup_ratio_max: float = 0.2
    warmup_steps_min: int = 0
    warmup_steps_max: int = 1000
    max_grad_norm_min: float = 0.1
    max_grad_norm_max: float = 10.0
    
    # Scheduler parameters
    scheduler_type: List[str] = field(default_factory=lambda: ["cosine", "linear", "constant"])
    num_cycles_min: int = 1
    num_cycles_max: int = 5
    
    # Quantization parameters (if applicable)
    quantization_bits: List[int] = field(default_factory=lambda: [4, 8, 16])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "lr_range": [self.lr_min, self.lr_max],
            "batch_size_range": [self.batch_size_min, self.batch_size_max],
            "lora_rank_range": [self.lora_rank_min, self.lora_rank_max],
            "lora_alpha_range": [self.lora_alpha_min, self.lora_alpha_max],
            "lora_dropout_range": [self.lora_dropout_min, self.lora_dropout_max],
            "weight_decay_range": [self.weight_decay_min, self.weight_decay_max],
            "warmup_ratio_range": [self.warmup_ratio_min, self.warmup_ratio_max],
            "scheduler_types": self.scheduler_type,
        }


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    
    n_trials: int = 50
    timeout: Optional[int] = 3600  # 1 hour
    n_jobs: int = 1
    study_name: str = "flux_hpo"
    storage: Optional[str] = None  # e.g., "sqlite:///hpo.db"
    direction: Union[str, List[str]] = "minimize"  # or "maximize" for accuracy
    pruner_type: str = "median"  # "median", "halving", "percentile"
    sampler_type: str = "tpe"  # "tpe", "cmaes", "random"
    seed: int = 42
    
    # Multi-objective weights (if using multi-objective)
    accuracy_weight: float = 1.0
    speed_weight: float = 0.0
    memory_weight: float = 0.0
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Warmup-based scheduling
    enable_warmup_scheduling: bool = True
    warmup_trials: int = 10
    
    # Search space configuration
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.direction, str):
            self.direction = [self.direction]
        
        if self.pruner_type == "median":
            self.pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            )
        elif self.pruner_type == "halving":
            self.pruner = SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=3,
                min_early_stopping_rate=0,
            )
        else:
            self.pruner = optuna.pruners.PercentilePruner(
                percentile=25.0,
                n_startup_trials=5,
                n_warmup_steps=10,
            )
        
        if self.sampler_type == "tpe":
            self.sampler = TPESampler(seed=self.seed, multivariate=True)
        elif self.sampler_type == "cmaes":
            self.sampler = CmaEsSampler(seed=self.seed)
        else:
            self.sampler = optuna.samplers.RandomSampler(seed=self.seed)


class LLMOptunaPruningCallback:
    """
    Custom pruning callback for LLM training that monitors training dynamics.
    """
    
    def __init__(
        self,
        trial: optuna.trial.Trial,
        monitor: str = "eval_loss",
        patience: int = 3,
        min_delta: float = 0.001,
    ):
        self.trial = trial
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.patience_counter = 0
        self.step = 0
    
    def __call__(self, eval_metrics: Dict[str, float]) -> bool:
        """
        Check if trial should be pruned based on evaluation metrics.
        
        Args:
            eval_metrics: Dictionary containing evaluation metrics
            
        Returns:
            True if trial should be pruned, False otherwise
        """
        self.step += 1
        current_value = eval_metrics.get(self.monitor)
        
        if current_value is None:
            return False
        
        # Report intermediate value to Optuna
        self.trial.report(current_value, self.step)
        
        # Check if trial should be pruned by Optuna
        if self.trial.should_prune():
            logger.info(f"Trial {self.trial.number} pruned by Optuna at step {self.step}")
            return True
        
        # Custom early stopping logic
        if self.best_value is None:
            self.best_value = current_value
        else:
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Trial {self.trial.number} pruned by custom early stopping "
                        f"at step {self.step} (patience={self.patience})"
                    )
                    return True
        
        return False


class HyperparameterSearchSpace:
    """
    Defines hyperparameter search spaces for LLM fine-tuning.
    """
    
    def __init__(self, config: SearchSpaceConfig):
        self.config = config
    
    def sample_hyperparameters(
        self, trial: optuna.trial.Trial
    ) -> Dict[str, Any]:
        """
        Sample hyperparameters from the search space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {}
        
        # Learning rate
        if self.config.lr_log:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate",
                self.config.lr_min,
                self.config.lr_max,
                log=True,
            )
        else:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate",
                self.config.lr_min,
                self.config.lr_max,
            )
        
        # Batch size (power of 2)
        batch_size_exp = trial.suggest_int(
            "batch_size_exp",
            int(np.log2(self.config.batch_size_min)),
            int(np.log2(self.config.batch_size_max)),
        )
        params["batch_size"] = 2 ** batch_size_exp
        
        # Gradient accumulation steps (to maintain effective batch size)
        params["gradient_accumulation_steps"] = trial.suggest_int(
            "gradient_accumulation_steps", 1, 16
        )
        
        # LoRA parameters
        params["lora_rank"] = trial.suggest_int(
            "lora_rank",
            self.config.lora_rank_min,
            self.config.lora_rank_max,
            step=self.config.lora_rank_step,
        )
        
        params["lora_alpha"] = trial.suggest_float(
            "lora_alpha",
            self.config.lora_alpha_min,
            self.config.lora_alpha_max,
        )
        
        params["lora_dropout"] = trial.suggest_float(
            "lora_dropout",
            self.config.lora_dropout_min,
            self.config.lora_dropout_max,
        )
        
        # Optimizer parameters
        params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            self.config.weight_decay_min,
            self.config.weight_decay_max,
        )
        
        params["adam_beta1"] = trial.suggest_float(
            "adam_beta1",
            self.config.adam_beta1_min,
            self.config.adam_beta1_max,
        )
        
        params["adam_beta2"] = trial.suggest_float(
            "adam_beta2",
            self.config.adam_beta2_min,
            self.config.adam_beta2_max,
        )
        
        # Training parameters
        params["max_grad_norm"] = trial.suggest_float(
            "max_grad_norm",
            self.config.max_grad_norm_min,
            self.config.max_grad_norm_max,
        )
        
        # Scheduler parameters
        params["scheduler_type"] = trial.suggest_categorical(
            "scheduler_type", self.config.scheduler_type
        )
        
        if params["scheduler_type"] in ["cosine_with_restarts", "polynomial"]:
            params["num_cycles"] = trial.suggest_int(
                "num_cycles",
                self.config.num_cycles_min,
                self.config.num_cycles_max,
            )
        
        # Warmup parameters
        use_warmup_steps = trial.suggest_categorical("use_warmup_steps", [True, False])
        if use_warmup_steps:
            params["warmup_steps"] = trial.suggest_int(
                "warmup_steps",
                self.config.warmup_steps_min,
                self.config.warmup_steps_max,
            )
            params["warmup_ratio"] = None
        else:
            params["warmup_ratio"] = trial.suggest_float(
                "warmup_ratio",
                self.config.warmup_ratio_min,
                self.config.warmup_ratio_max,
            )
            params["warmup_steps"] = None
        
        # Quantization (if applicable)
        if trial.suggest_categorical("use_quantization", [True, False]):
            params["quantization_bits"] = trial.suggest_categorical(
                "quantization_bits", self.config.quantization_bits
            )
        else:
            params["quantization_bits"] = None
        
        return params
    
    def get_warmup_scheduled_params(
        self,
        trial_number: int,
        total_trials: int,
        base_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply warmup-based scheduling to hyperparameters.
        Gradually increases exploration as trials progress.
        
        Args:
            trial_number: Current trial number
            total_trials: Total number of trials
            base_params: Base hyperparameters
            
        Returns:
            Scheduled hyperparameters
        """
        if not self.config.enable_warmup_scheduling:
            return base_params
        
        warmup_ratio = min(1.0, trial_number / self.config.warmup_trials)
        
        scheduled_params = base_params.copy()
        
        # Gradually increase learning rate range exploration
        if "learning_rate" in scheduled_params:
            # Start with conservative learning rates
            lr_scale = 0.1 + 0.9 * warmup_ratio
            scheduled_params["learning_rate"] *= lr_scale
        
        # Gradually increase batch size exploration
        if "batch_size" in scheduled_params:
            # Start with smaller batch sizes
            batch_scale = 0.5 + 0.5 * warmup_ratio
            scheduled_params["batch_size"] = max(
                1, int(scheduled_params["batch_size"] * batch_scale)
            )
        
        # Gradually increase LoRA rank exploration
        if "lora_rank" in scheduled_params:
            # Start with lower ranks
            rank_scale = 0.3 + 0.7 * warmup_ratio
            scheduled_params["lora_rank"] = max(
                self.config.lora_rank_min,
                int(scheduled_params["lora_rank"] * rank_scale),
            )
        
        logger.info(
            f"Applied warmup scheduling (ratio={warmup_ratio:.2f}) "
            f"to trial {trial_number}"
        )
        
        return scheduled_params


class MultiObjectiveOptimizer:
    """
    Handles multi-objective optimization for speed vs. accuracy tradeoffs.
    """
    
    def __init__(
        self,
        objectives: List[OptimizationObjective],
        weights: Optional[List[float]] = None,
    ):
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        
        if len(self.weights) != len(self.objectives):
            raise ValueError("Number of weights must match number of objectives")
    
    def compute_combined_metric(
        self, metrics: Dict[str, float]
    ) -> Union[float, List[float]]:
        """
        Compute combined metric for optimization.
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Returns:
            Combined metric value(s)
        """
        if len(self.objectives) == 1:
            # Single objective
            obj = self.objectives[0]
            if obj == OptimizationObjective.ACCURACY:
                return -metrics.get("accuracy", 0.0)  # Negative for minimization
            elif obj == OptimizationObjective.LOSS:
                return metrics.get("eval_loss", float("inf"))
            elif obj == OptimizationObjective.SPEED:
                return metrics.get("training_time_per_epoch", float("inf"))
            elif obj == OptimizationObjective.MEMORY:
                return metrics.get("peak_memory_mb", float("inf"))
        
        # Multi-objective
        combined = []
        for obj, weight in zip(self.objectives, self.weights):
            if obj == OptimizationObjective.ACCURACY:
                value = -metrics.get("accuracy", 0.0) * weight
            elif obj == OptimizationObjective.LOSS:
                value = metrics.get("eval_loss", float("inf")) * weight
            elif obj == OptimizationObjective.SPEED:
                value = metrics.get("training_time_per_epoch", float("inf")) * weight
            elif obj == OptimizationObjective.MEMORY:
                value = metrics.get("peak_memory_mb", float("inf")) * weight
            else:
                value = float("inf")
            combined.append(value)
        
        return combined


class HyperparameterOptimizer:
    """
    Main class for hyperparameter optimization using Optuna.
    """
    
    def __init__(
        self,
        config: HPOConfig,
        train_fn: Callable[[Dict[str, Any], optuna.trial.Trial], Dict[str, float]],
        model_name: str = "flux",
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            config: HPO configuration
            train_fn: Training function that takes hyperparams and trial,
                     returns evaluation metrics
            model_name: Name of the model being optimized
        """
        self.config = config
        self.train_fn = train_fn
        self.model_name = model_name
        self.search_space = HyperparameterSearchSpace(config.search_space)
        
        # Setup study
        self.study = self._create_study()
        
        # Results storage
        self.best_params = None
        self.best_value = None
        self.trial_history = []
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with specified configuration."""
        if len(self.config.direction) == 1:
            # Single objective study
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=self.config.sampler,
                pruner=self.config.pruner,
                direction=self.config.direction[0],
                load_if_exists=True,
            )
        else:
            # Multi-objective study
            study = optuna.multi_objective.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=self.config.sampler,
                pruner=self.config.pruner,
                directions=self.config.direction,
                load_if_exists=True,
            )
        
        return study
    
    def objective(self, trial: optuna.trial.Trial) -> Union[float, List[float]]:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric value(s) to optimize
        """
        # Sample hyperparameters
        params = self.search_space.sample_hyperparameters(trial)
        
        # Apply warmup scheduling
        params = self.search_space.get_warmup_scheduled_params(
            trial.number, self.config.n_trials, params
        )
        
        logger.info(f"Trial {trial.number} - Hyperparameters: {params}")
        
        try:
            # Run training with sampled hyperparameters
            metrics = self.train_fn(params, trial)
            
            # Store trial information
            trial_info = {
                "trial_number": trial.number,
                "params": params,
                "metrics": metrics,
                "state": trial.state.name,
            }
            self.trial_history.append(trial_info)
            
            # Compute objective value(s)
            if hasattr(self.study, "directions"):  # Multi-objective
                optimizer = MultiObjectiveOptimizer(
                    objectives=[OptimizationObjective(d) for d in self.config.direction],
                    weights=[
                        self.config.accuracy_weight,
                        self.config.speed_weight,
                        self.config.memory_weight,
                    ],
                )
                return optimizer.compute_combined_metric(metrics)
            else:  # Single objective
                optimizer = MultiObjectiveOptimizer(
                    objectives=[OptimizationObjective(self.config.direction[0])]
                )
                return optimizer.compute_combined_metric(metrics)
        
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            trial.set_user_attr("error", str(e))
            raise optuna.exceptions.TrialPruned()
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {self.config.n_trials} trials")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )
        
        # Get best trial(s)
        if hasattr(self.study, "best_trials"):  # Multi-objective
            best_trials = self.study.best_trials
            self.best_params = [trial.params for trial in best_trials]
            self.best_value = [trial.values for trial in best_trials]
        else:  # Single objective
            best_trial = self.study.best_trial
            self.best_params = best_trial.params
            self.best_value = best_trial.value
        
        # Generate results
        results = {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "study_name": self.config.study_name,
        }
        
        logger.info(f"Optimization completed. Best value: {self.best_value}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return results
    
    def get_pareto_front(self) -> List[optuna.trial.FrozenTrial]:
        """
        Get Pareto front for multi-objective optimization.
        
        Returns:
            List of trials on the Pareto front
        """
        if not hasattr(self.study, "best_trials"):
            raise ValueError("Pareto front only available for multi-objective studies")
        
        return self.study.best_trials
    
    def visualize_results(self, output_dir: str = "./hpo_results"):
        """
        Generate visualization plots for optimization results.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Optimization history
            fig1 = plot_optimization_history(self.study)
            fig1.write_html(os.path.join(output_dir, "optimization_history.html"))
            
            # Parameter importances
            fig2 = plot_param_importances(self.study)
            fig2.write_html(os.path.join(output_dir, "param_importances.html"))
            
            # Save trial history
            with open(os.path.join(output_dir, "trial_history.json"), "w") as f:
                json.dump(self.trial_history, f, indent=2, default=str)
            
            # Save best parameters
            with open(os.path.join(output_dir, "best_params.json"), "w") as f:
                json.dump(self.best_params, f, indent=2, default=str)
            
            logger.info(f"Visualizations saved to {output_dir}")
        
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {str(e)}")
    
    def get_suggested_config(self) -> Dict[str, Any]:
        """
        Get suggested configuration based on optimization results.
        
        Returns:
            Dictionary with suggested hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # For multi-objective, return the first Pareto-optimal solution
        if isinstance(self.best_params, list):
            suggested = self.best_params[0]
        else:
            suggested = self.best_params
        
        # Add metadata
        suggested_config = {
            "hyperparameters": suggested,
            "optimization_info": {
                "model_name": self.model_name,
                "study_name": self.config.study_name,
                "n_trials": self.config.n_trials,
                "best_value": self.best_value,
            },
        }
        
        return suggested_config


# Integration with existing flux training
def create_training_wrapper(
    base_train_fn: Callable,
    eval_dataset: Any = None,
    callbacks: Optional[List] = None,
) -> Callable:
    """
    Create a training wrapper for Optuna integration.
    
    Args:
        base_train_fn: Original training function
        eval_dataset: Evaluation dataset
        callbacks: Additional callbacks
        
    Returns:
        Wrapped training function compatible with Optuna
    """
    def train_fn_with_optuna(
        hyperparams: Dict[str, Any],
        trial: optuna.trial.Trial,
    ) -> Dict[str, float]:
        """
        Training function with Optuna integration.
        
        Args:
            hyperparams: Hyperparameters for this trial
            trial: Optuna trial object
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Add pruning callback
        pruning_callback = LLMOptunaPruningCallback(
            trial=trial,
            monitor="eval_loss",
            patience=3,
        )
        
        # Combine with existing callbacks
        all_callbacks = callbacks or []
        all_callbacks.append(pruning_callback)
        
        # Modify training arguments with sampled hyperparams
        # This would integrate with flux's training arguments
        training_args = {
            "learning_rate": hyperparams.get("learning_rate", 5e-5),
            "per_device_train_batch_size": hyperparams.get("batch_size", 4),
            "gradient_accumulation_steps": hyperparams.get("gradient_accumulation_steps", 1),
            "weight_decay": hyperparams.get("weight_decay", 0.01),
            "adam_beta1": hyperparams.get("adam_beta1", 0.9),
            "adam_beta2": hyperparams.get("adam_beta2", 0.999),
            "max_grad_norm": hyperparams.get("max_grad_norm", 1.0),
            "warmup_ratio": hyperparams.get("warmup_ratio", 0.1),
            "warmup_steps": hyperparams.get("warmup_steps", 0),
            "lr_scheduler_type": hyperparams.get("scheduler_type", "cosine"),
            "num_train_epochs": 3,  # Fixed for HPO
            "evaluation_strategy": "steps",
            "eval_steps": 50,
            "save_strategy": "steps",
            "save_steps": 50,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
        
        # Add LoRA configuration if using LoRA
        lora_config = {
            "r": hyperparams.get("lora_rank", 8),
            "lora_alpha": hyperparams.get("lora_alpha", 16),
            "lora_dropout": hyperparams.get("lora_dropout", 0.05),
            "target_modules": ["q_proj", "v_proj"],  # Default for LLaMA
        }
        
        # Call base training function
        # Note: This would need to be adapted to flux's actual API
        try:
            results = base_train_fn(
                training_args=training_args,
                lora_config=lora_config,
                eval_dataset=eval_dataset,
                callbacks=all_callbacks,
            )
            
            # Extract metrics
            metrics = {
                "eval_loss": results.get("eval_loss", float("inf")),
                "accuracy": results.get("eval_accuracy", 0.0),
                "training_time_per_epoch": results.get("training_time", 0.0),
                "peak_memory_mb": results.get("memory_usage", 0.0),
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    return train_fn_with_optuna


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = HPOConfig(
        n_trials=100,
        study_name="llama2_7b_lora_hpo",
        direction=["minimize", "minimize"],  # Multi-objective: loss and time
        accuracy_weight=1.0,
        speed_weight=0.3,
        sampler_type="tpe",
        pruner_type="median",
    )
    
    # Example training function (placeholder)
    def example_train_fn(hyperparams: Dict[str, Any], trial: optuna.trial.Trial) -> Dict[str, float]:
        """Example training function for demonstration."""
        # Simulate training
        import time
        time.sleep(1)
        
        # Simulate metrics
        metrics = {
            "eval_loss": 0.5 + np.random.random() * 0.5,
            "accuracy": 0.7 + np.random.random() * 0.2,
            "training_time_per_epoch": 100 + np.random.random() * 50,
            "peak_memory_mb": 8000 + np.random.random() * 2000,
        }
        
        return metrics
    
    # Run optimization
    optimizer = HyperparameterOptimizer(
        config=config,
        train_fn=example_train_fn,
        model_name="llama2-7b-lora",
    )
    
    results = optimizer.optimize()
    optimizer.visualize_results("./hpo_results")
    
    # Get suggested configuration
    suggested_config = optimizer.get_suggested_config()
    print("Suggested configuration:", json.dumps(suggested_config, indent=2))