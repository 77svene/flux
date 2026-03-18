"""
src/flux/hpo/optimizer.py
Automated Hyperparameter Optimization with Optuna for flux
"""

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import optuna
from optuna.pruners import BasePruner, MedianPruner, PatientPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.storages import BaseStorage
from optuna.trial import FrozenTrial, Trial, TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_pareto_front

from ..data import get_dataset
from ..model import load_model
from ..train import train
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    
    # Optimization objectives
    objectives: List[str] = field(default_factory=lambda: ["eval_loss", "train_time"])
    directions: List[str] = field(default_factory=lambda: ["minimize", "minimize"])
    
    # Search space
    lr_range: Tuple[float, float] = (1e-6, 1e-3)
    batch_size_range: Tuple[int, int] = (2, 64)
    lora_rank_range: Tuple[int, int] = (4, 64)
    lora_alpha_range: Tuple[float, float] = (8.0, 64.0)
    warmup_ratio_range: Tuple[float, float] = (0.0, 0.2)
    
    # Training constraints
    max_epochs_per_trial: int = 5
    max_steps_per_trial: int = 1000
    early_stopping_patience: int = 3
    
    # Optimization settings
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1
    seed: int = 42
    
    # Pruning settings
    pruning_patience: int = 2
    pruning_warmup_steps: int = 100
    min_trials_for_pruning: int = 5
    
    # Storage and study
    study_name: str = "flux_hpo"
    storage: Optional[Union[str, BaseStorage]] = None
    load_if_exists: bool = True
    
    # Visualization
    output_dir: str = "./hpo_results"
    save_visualizations: bool = True


class LLMPruner(BasePruner):
    """Custom pruner for LLM training with warmup and stability considerations."""
    
    def __init__(
        self,
        warmup_steps: int = 100,
        patience: int = 2,
        min_trials: int = 5,
        percentile: float = 25.0,
        n_startup_trials: int = 5,
        interval_steps: int = 1,
    ):
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.min_trials = min_trials
        self.percentile = percentile
        self.n_startup_trials = n_startup_trials
        self.interval_steps = interval_steps
        self._median_pruner = MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=warmup_steps,
            interval_steps=interval_steps,
        )
        self._patient_pruner = PatientPruner(
            wrapped_pruner=self._median_pruner,
            patience=patience,
        )
    
    def prune(self, study: optuna.Study, trial: FrozenTrial) -> bool:
        """Determine if trial should be pruned."""
        
        # Don't prune during warmup
        step = trial.last_step
        if step is None or step < self.warmup_steps:
            return False
        
        # Not enough completed trials for reliable pruning
        completed_trials = len(
            [t for t in study.trials if t.state == TrialState.COMPLETE]
        )
        if completed_trials < self.min_trials:
            return False
        
        # Use patient pruner for stability
        return self._patient_pruner.prune(study, trial)


class WarmupScheduler:
    """Warmup-based hyperparameter scheduling during trials."""
    
    def __init__(self, warmup_ratio: float = 0.1, schedule_type: str = "linear"):
        self.warmup_ratio = warmup_ratio
        self.schedule_type = schedule_type
        self._step = 0
        self._total_steps = 0
    
    def set_total_steps(self, total_steps: int):
        """Set total training steps for warmup calculation."""
        self._total_steps = total_steps
        self._step = 0
    
    def step(self) -> float:
        """Get warmup factor for current step."""
        self._step += 1
        warmup_steps = int(self._total_steps * self.warmup_ratio)
        
        if self._step >= warmup_steps:
            return 1.0
        
        if self.schedule_type == "linear":
            return self._step / warmup_steps
        elif self.schedule_type == "cosine":
            import math
            return 0.5 * (1.0 + math.cos(math.pi * (1.0 - self._step / warmup_steps)))
        else:
            return min(1.0, self._step / warmup_steps)


class fluxHPO:
    """Main class for hyperparameter optimization in flux."""
    
    def __init__(
        self,
        config: HPOConfig,
        base_model_args: Dict[str, Any],
        data_args: Dict[str, Any],
        training_args: Dict[str, Any],
        callbacks: Optional[List[Callable]] = None,
    ):
        self.config = config
        self.base_model_args = base_model_args
        self.data_args = data_args
        self.training_args = training_args
        self.callbacks = callbacks or []
        
        # Initialize study
        self.study = self._create_study()
        self.best_trials = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _create_study(self) -> optuna.Study:
        """Create or load Optuna study."""
        
        # Create sampler
        sampler = TPESampler(seed=self.config.seed)
        
        # Create pruner
        pruner = LLMPruner(
            warmup_steps=self.config.pruning_warmup_steps,
            patience=self.config.pruning_patience,
            min_trials=self.config.min_trials_for_pruning,
        )
        
        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            directions=self.config.directions,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=self.config.load_if_exists,
        )
        
        return study
    
    def _setup_logging(self):
        """Setup logging for HPO."""
        log_file = Path(self.config.output_dir) / "hpo.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        
        hyperparams = {}
        
        # Learning rate (log scale)
        hyperparams["learning_rate"] = trial.suggest_float(
            "learning_rate",
            self.config.lr_range[0],
            self.config.lr_range[1],
            log=True,
        )
        
        # Batch size (integer)
        # Power of 2 for better GPU utilization
        batch_size_exp = trial.suggest_int(
            "batch_size_exp",
            int(self.config.batch_size_range[0]).bit_length() - 1,
            int(self.config.batch_size_range[1]).bit_length(),
        )
        hyperparams["batch_size"] = 2 ** batch_size_exp
        
        # LoRA parameters
        hyperparams["lora_rank"] = trial.suggest_int(
            "lora_rank",
            self.config.lora_rank_range[0],
            self.config.lora_rank_range[1],
            step=4,  # Common LoRA ranks are multiples of 4
        )
        
        hyperparams["lora_alpha"] = trial.suggest_float(
            "lora_alpha",
            self.config.lora_alpha_range[0],
            self.config.lora_alpha_range[1],
        )
        
        # LoRA scaling factor
        hyperparams["lora_scaling"] = hyperparams["lora_alpha"] / hyperparams["lora_rank"]
        
        # Warmup ratio
        hyperparams["warmup_ratio"] = trial.suggest_float(
            "warmup_ratio",
            self.config.warmup_ratio_range[0],
            self.config.warmup_ratio_range[1],
        )
        
        # Optimizer choice
        hyperparams["optimizer"] = trial.suggest_categorical(
            "optimizer", ["adamw_torch", "adamw_hf", "adafactor"]
        )
        
        # Weight decay
        hyperparams["weight_decay"] = trial.suggest_float(
            "weight_decay", 0.0, 0.1, step=0.01
        )
        
        # Gradient accumulation steps (for effective batch size)
        hyperparams["gradient_accumulation_steps"] = trial.suggest_int(
            "gradient_accumulation_steps", 1, 16
        )
        
        # Learning rate scheduler
        hyperparams["lr_scheduler_type"] = trial.suggest_categorical(
            "lr_scheduler_type",
            ["linear", "cosine", "cosine_with_restarts", "polynomial"],
        )
        
        # Gradient clipping
        hyperparams["max_grad_norm"] = trial.suggest_float(
            "max_grad_norm", 0.5, 2.0, step=0.5
        )
        
        return hyperparams
    
    def _create_training_args(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Create training arguments from hyperparameters."""
        
        training_args = self.training_args.copy()
        
        # Update with hyperparameters
        training_args.update({
            "learning_rate": hyperparams["learning_rate"],
            "per_device_train_batch_size": hyperparams["batch_size"],
            "per_device_eval_batch_size": hyperparams["batch_size"] * 2,
            "gradient_accumulation_steps": hyperparams["gradient_accumulation_steps"],
            "warmup_ratio": hyperparams["warmup_ratio"],
            "weight_decay": hyperparams["weight_decay"],
            "lr_scheduler_type": hyperparams["lr_scheduler_type"],
            "max_grad_norm": hyperparams["max_grad_norm"],
            "optim": hyperparams["optimizer"],
            "max_steps": self.config.max_steps_per_trial,
            "num_train_epochs": self.config.max_epochs_per_trial,
            "evaluation_strategy": "steps",
            "eval_steps": 50,
            "save_strategy": "steps",
            "save_steps": 100,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "none",  # Disable external reporting during HPO
        })
        
        # Add LoRA config
        if "lora_config" not in training_args:
            training_args["lora_config"] = {}
        
        training_args["lora_config"].update({
            "r": hyperparams["lora_rank"],
            "lora_alpha": hyperparams["lora_alpha"],
            "lora_scaling": hyperparams["lora_scaling"],
        })
        
        return training_args
    
    def _train_and_evaluate(
        self,
        trial: Trial,
        hyperparams: Dict[str, Any],
    ) -> Dict[str, float]:
        """Train model with given hyperparameters and return metrics."""
        
        # Create temporary directory for this trial
        trial_dir = Path(self.config.output_dir) / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)
        
        # Save hyperparameters
        with open(trial_dir / "hyperparams.json", "w") as f:
            json.dump(hyperparams, f, indent=2)
        
        # Create training arguments
        training_args = self._create_training_args(hyperparams)
        training_args["output_dir"] = str(trial_dir / "checkpoints")
        
        # Setup warmup scheduler
        warmup_scheduler = WarmupScheduler(
            warmup_ratio=hyperparams["warmup_ratio"],
            schedule_type="linear",
        )
        
        # Callback for Optuna integration
        def optuna_callback(metrics: Dict[str, float], step: int):
            # Report intermediate metrics to Optuna
            for objective in self.config.objectives:
                if objective in metrics:
                    trial.report(metrics[objective], step)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Update warmup scheduler
            warmup_scheduler.step()
        
        # Add to callbacks
        trial_callbacks = self.callbacks + [optuna_callback]
        
        try:
            # Load dataset
            dataset = get_dataset(self.data_args)
            
            # Load model with LoRA config
            model = load_model(
                self.base_model_args,
                training_args,
            )
            
            # Train model
            start_time = time.time()
            train_result = train(
                model=model,
                dataset=dataset,
                training_args=training_args,
                callbacks=trial_callbacks,
            )
            train_time = time.time() - start_time
            
            # Extract metrics
            metrics = {
                "eval_loss": train_result.get("eval_loss", float("inf")),
                "train_loss": train_result.get("train_loss", float("inf")),
                "train_time": train_time,
                "throughput": train_result.get("train_samples_per_second", 0),
                "memory_usage": train_result.get("memory_usage", 0),
            }
            
            # Calculate composite score if multi-objective
            if len(self.config.objectives) > 1:
                # Normalize and combine objectives
                normalized_loss = min(1.0, metrics["eval_loss"] / 10.0)
                normalized_time = min(1.0, metrics["train_time"] / 3600.0)  # 1 hour max
                metrics["composite_score"] = (
                    0.7 * normalized_loss + 0.3 * normalized_time
                )
            
            # Save metrics
            with open(trial_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Return worst possible metrics
            return {obj: float("inf") for obj in self.config.objectives}
    
    def objective(self, trial: Trial) -> Union[float, List[float]]:
        """Objective function for Optuna optimization."""
        
        logger.info(f"Starting trial {trial.number}")
        
        # Suggest hyperparameters
        hyperparams = self._suggest_hyperparameters(trial)
        
        logger.info(f"Trial {trial.number} hyperparameters: {hyperparams}")
        
        # Train and evaluate
        metrics = self._train_and_evaluate(trial, hyperparams)
        
        # Return objectives
        if len(self.config.objectives) == 1:
            return metrics[self.config.objectives[0]]
        else:
            return [metrics[obj] for obj in self.config.objectives]
    
    def optimize(self) -> optuna.Study:
        """Run hyperparameter optimization."""
        
        logger.info(f"Starting HPO with {self.config.n_trials} trials")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
            callbacks=[self._log_callback],
        )
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        if self.config.save_visualizations:
            self._generate_visualizations()
        
        logger.info("HPO completed")
        
        return self.study
    
    def _log_callback(self, study: optuna.Study, trial: FrozenTrial):
        """Callback for logging trial progress."""
        
        logger.info(
            f"Trial {trial.number} finished with values: {trial.values} "
            f"and params: {trial.params}"
        )
        
        if trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"Trial {trial.number} was pruned")
        
        # Update best trials
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.best_trials.append(trial)
            self.best_trials.sort(key=lambda t: t.values[0] if t.values else float("inf"))
            self.best_trials = self.best_trials[:10]  # Keep top 10
    
    def _save_results(self):
        """Save optimization results."""
        
        results_dir = Path(self.config.output_dir)
        
        # Save best trial
        best_trial = self.study.best_trial
        with open(results_dir / "best_trial.json", "w") as f:
            json.dump({
                "number": best_trial.number,
                "values": best_trial.values,
                "params": best_trial.params,
                "datetime_start": str(best_trial.datetime_start),
                "datetime_complete": str(best_trial.datetime_complete),
                "duration": str(best_trial.duration),
            }, f, indent=2)
        
        # Save all trials
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                "number": trial.number,
                "values": trial.values,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": str(trial.datetime_start),
                "datetime_complete": str(trial.datetime_complete),
                "duration": str(trial.duration) if trial.duration else None,
            })
        
        with open(results_dir / "all_trials.json", "w") as f:
            json.dump(trials_data, f, indent=2)
        
        # Save study statistics
        stats = {
            "n_trials": len(self.study.trials),
            "n_complete": len([t for t in self.study.trials if t.state == TrialState.COMPLETE]),
            "n_pruned": len([t for t in self.study.trials if t.state == TrialState.PRUNED]),
            "n_failed": len([t for t in self.study.trials if t.state == TrialState.FAIL]),
            "best_values": self.study.best_values if hasattr(self.study, "best_values") else None,
        }
        
        with open(results_dir / "study_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    
    def _generate_visualizations(self):
        """Generate optimization visualizations."""
        
        viz_dir = Path(self.config.output_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_image(str(viz_dir / "optimization_history.png"))
            fig.write_html(str(viz_dir / "optimization_history.html"))
            
            # Parameter importances
            if len(self.study.trials) > 10:
                fig = plot_param_importances(self.study)
                fig.write_image(str(viz_dir / "param_importances.png"))
                fig.write_html(str(viz_dir / "param_importances.html"))
            
            # Pareto front for multi-objective
            if len(self.config.objectives) > 1:
                fig = plot_pareto_front(self.study)
                fig.write_image(str(viz_dir / "pareto_front.png"))
                fig.write_html(str(viz_dir / "pareto_front.html"))
            
            logger.info(f"Visualizations saved to {viz_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {str(e)}")
    
    def get_best_config(self, objective_index: int = 0) -> Dict[str, Any]:
        """Get best configuration for specified objective."""
        
        if len(self.config.objectives) == 1:
            best_trial = self.study.best_trial
        else:
            # For multi-objective, get best for specified objective
            best_trial = min(
                [t for t in self.study.trials if t.state == TrialState.COMPLETE],
                key=lambda t: t.values[objective_index] if t.values else float("inf"),
            )
        
        return {
            "hyperparameters": best_trial.params,
            "values": best_trial.values,
            "trial_number": best_trial.number,
        }
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get Pareto-optimal configurations for multi-objective optimization."""
        
        if len(self.config.objectives) == 1:
            return [self.get_best_config()]
        
        pareto_trials = []
        complete_trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        
        for trial in complete_trials:
            is_dominated = False
            for other_trial in complete_trials:
                if trial.number == other_trial.number:
                    continue
                
                # Check if other_trial dominates trial
                dominates = all(
                    other_val <= trial_val
                    for other_val, trial_val in zip(other_trial.values, trial.values)
                ) and any(
                    other_val < trial_val
                    for other_val, trial_val in zip(other_trial.values, trial.values)
                )
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_trials.append({
                    "hyperparameters": trial.params,
                    "values": trial.values,
                    "trial_number": trial.number,
                })
        
        return pareto_trials


def run_hpo(
    config: HPOConfig,
    base_model_args: Dict[str, Any],
    data_args: Dict[str, Any],
    training_args: Dict[str, Any],
    callbacks: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """Convenience function to run HPO."""
    
    hpo = fluxHPO(
        config=config,
        base_model_args=base_model_args,
        data_args=data_args,
        training_args=training_args,
        callbacks=callbacks,
    )
    
    study = hpo.optimize()
    
    # Get results
    results = {
        "best_config": hpo.get_best_config(),
        "study_stats": {
            "n_trials": len(study.trials),
            "best_value": study.best_value if hasattr(study, "best_value") else None,
            "best_params": study.best_params if hasattr(study, "best_params") else None,
        },
    }
    
    if len(config.objectives) > 1:
        results["pareto_front"] = hpo.get_pareto_front()
    
    # Save final results
    results_path = Path(config.output_dir) / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = HPOConfig(
        objectives=["eval_loss", "train_time"],
        directions=["minimize", "minimize"],
        n_trials=50,
        output_dir="./hpo_experiment",
    )
    
    # Example arguments (would come from actual flux config)
    base_model_args = {
        "model_name_or_path": "meta-llama/Llama-2-7b-hf",
        "quantization": "bitsandbytes",
    }
    
    data_args = {
        "dataset": "alpaca",
        "max_samples": 1000,
    }
    
    training_args = {
        "output_dir": "./output",
        "fp16": True,
    }
    
    # Run HPO
    results = run_hpo(
        config=config,
        base_model_args=base_model_args,
        data_args=data_args,
        training_args=training_args,
    )
    
    print("Best configuration:", results["best_config"])