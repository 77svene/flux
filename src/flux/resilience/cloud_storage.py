"""src/flux/resilience/cloud_storage.py

Fault-tolerant training with cloud-backed checkpoint resilience.
Implements async checkpointing, heartbeat monitoring, and automatic recovery.
"""

import asyncio
import json
import os
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    S3 = "s3"
    GCS = "gcs"
    HDFS = "hdfs"
    LOCAL = "local"


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint resilience."""
    storage_backend: StorageBackend = StorageBackend.S3
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000  # steps
    save_total_limit: int = 5
    async_save: bool = True
    heartbeat_interval: int = 30  # seconds
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    compression: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_gradient_accumulation: bool = True
    save_rng_state: bool = True
    cloud_credentials: Optional[Dict[str, str]] = None
    endpoint_url: Optional[str] = None


class StorageInterface(ABC):
    """Abstract interface for cloud storage backends."""
    
    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> bool:
        """Upload file to storage."""
        pass
    
    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> bool:
        """Download file from storage."""
        pass
    
    @abstractmethod
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix."""
        pass
    
    @abstractmethod
    def delete(self, remote_path: str) -> bool:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if file exists."""
        pass


class S3Storage(StorageInterface):
    """AWS S3 storage backend."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=config.cloud_credentials.get('aws_access_key_id'),
                aws_secret_access_key=config.cloud_credentials.get('aws_secret_access_key'),
                endpoint_url=config.endpoint_url
            )
            self.bucket = config.cloud_credentials.get('bucket_name')
            self.ClientError = ClientError
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            self.s3.upload_file(local_path, self.bucket, remote_path)
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            self.s3.download_file(self.bucket, remote_path, local_path)
            return True
        except self.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception:
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception:
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except self.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise


class GCSStorage(StorageInterface):
    """Google Cloud Storage backend."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        try:
            from google.cloud import storage
            self.client = storage.Client.from_service_account_json(
                config.cloud_credentials.get('credentials_path')
            )
            self.bucket = self.client.bucket(config.cloud_credentials.get('bucket_name'))
        except ImportError:
            raise ImportError("google-cloud-storage is required for GCS. Install with: pip install google-cloud-storage")
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            return True
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            blob = self.bucket.blob(remote_path)
            if not blob.exists():
                return False
            blob.download_to_filename(local_path)
            return True
        except Exception as e:
            logger.error(f"GCS download failed: {e}")
            return False
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception:
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            blob = self.bucket.blob(remote_path)
            blob.delete()
            return True
        except Exception:
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            blob = self.bucket.blob(remote_path)
            return blob.exists()
        except Exception:
            return False


class HDFSStorage(StorageInterface):
    """Hadoop HDFS storage backend."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        try:
            from hdfs import InsecureClient
            self.client = InsecureClient(
                config.endpoint_url,
                user=config.cloud_credentials.get('user', 'hadoop')
            )
        except ImportError:
            raise ImportError("hdfs is required for HDFS storage. Install with: pip install hdfs")
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            self.client.upload(remote_path, local_path, overwrite=True)
            return True
        except Exception as e:
            logger.error(f"HDFS upload failed: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            if not self.client.status(remote_path, strict=False):
                return False
            self.client.download(remote_path, local_path, overwrite=True)
            return True
        except Exception as e:
            logger.error(f"HDFS download failed: {e}")
            return False
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            files = []
            for path, dirs, file_list in self.client.walk(prefix):
                files.extend([f"{path}/{f}" for f in file_list])
            return files
        except Exception:
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            self.client.delete(remote_path)
            return True
        except Exception:
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            return self.client.status(remote_path, strict=False) is not None
        except Exception:
            return False


class LocalStorage(StorageInterface):
    """Local filesystem storage (for testing/fallback)."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.base_path = Path(config.checkpoint_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            import shutil
            dest = self.base_path / remote_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            return True
        except Exception as e:
            logger.error(f"Local upload failed: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            src = self.base_path / remote_path
            if not src.exists():
                return False
            import shutil
            shutil.copy2(src, local_path)
            return True
        except Exception as e:
            logger.error(f"Local download failed: {e}")
            return False
    
    def list_files(self, prefix: str) -> List[str]:
        try:
            prefix_path = self.base_path / prefix
            if not prefix_path.exists():
                return []
            return [str(p.relative_to(self.base_path)) for p in prefix_path.rglob('*') if p.is_file()]
        except Exception:
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            path = self.base_path / remote_path
            if path.exists():
                path.unlink()
            return True
        except Exception:
            return False
    
    def exists(self, remote_path: str) -> bool:
        return (self.base_path / remote_path).exists()


class CheckpointMetadata:
    """Metadata for checkpoint management."""
    
    def __init__(self):
        self.checkpoints: List[Dict[str, Any]] = []
        self.latest_step: int = 0
        self.latest_checkpoint: Optional[str] = None
        self.heartbeat: Optional[float] = None
    
    def add_checkpoint(self, step: int, path: str, timestamp: float):
        self.checkpoints.append({
            'step': step,
            'path': path,
            'timestamp': timestamp
        })
        self.checkpoints.sort(key=lambda x: x['step'])
        self.latest_step = step
        self.latest_checkpoint = path
    
    def prune_checkpoints(self, keep_last_n: int):
        if len(self.checkpoints) > keep_last_n:
            to_remove = self.checkpoints[:-keep_last_n]
            self.checkpoints = self.checkpoints[-keep_last_n:]
            return [c['path'] for c in to_remove]
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'checkpoints': self.checkpoints,
            'latest_step': self.latest_step,
            'latest_checkpoint': self.latest_checkpoint,
            'heartbeat': self.heartbeat
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        metadata = cls()
        metadata.checkpoints = data.get('checkpoints', [])
        metadata.latest_step = data.get('latest_step', 0)
        metadata.latest_checkpoint = data.get('latest_checkpoint')
        metadata.heartbeat = data.get('heartbeat')
        return metadata


class ResilientCheckpointManager:
    """Manages fault-tolerant checkpointing with cloud storage."""
    
    def __init__(self, config: CheckpointConfig, local_rank: int = 0):
        self.config = config
        self.local_rank = local_rank
        self.is_main_process = local_rank == 0
        
        # Initialize storage backend
        self.storage = self._create_storage_backend()
        
        # State
        self.metadata = CheckpointMetadata()
        self.current_step = 0
        self.save_executor = ThreadPoolExecutor(max_workers=2) if config.async_save else None
        self.heartbeat_thread = None
        self.stop_heartbeat = threading.Event()
        self.gradient_accumulation_step = 0
        
        # Load existing metadata
        self._load_metadata()
        
        # Start heartbeat if enabled
        if config.heartbeat_interval > 0 and self.is_main_process:
            self._start_heartbeat()
    
    def _create_storage_backend(self) -> StorageInterface:
        """Create storage backend based on configuration."""
        if self.config.storage_backend == StorageBackend.S3:
            return S3Storage(self.config)
        elif self.config.storage_backend == StorageBackend.GCS:
            return GCSStorage(self.config)
        elif self.config.storage_backend == StorageBackend.HDFS:
            return HDFSStorage(self.config)
        else:
            return LocalStorage(self.config)
    
    def _load_metadata(self):
        """Load checkpoint metadata from storage."""
        metadata_path = f"{self.config.checkpoint_dir}/metadata.json"
        local_path = "/tmp/checkpoint_metadata.json"
        
        if self.storage.download(metadata_path, local_path):
            try:
                with open(local_path, 'r') as f:
                    data = json.load(f)
                self.metadata = CheckpointMetadata.from_dict(data)
                logger.info(f"Loaded metadata: {len(self.metadata.checkpoints)} checkpoints")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save checkpoint metadata to storage."""
        if not self.is_main_process:
            return
        
        metadata_path = f"{self.config.checkpoint_dir}/metadata.json"
        local_path = "/tmp/checkpoint_metadata.json"
        
        try:
            with open(local_path, 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
            
            self.storage.upload(local_path, metadata_path)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _start_heartbeat(self):
        """Start heartbeat monitoring thread."""
        def heartbeat_loop():
            while not self.stop_heartbeat.is_set():
                try:
                    self.metadata.heartbeat = time.time()
                    self._save_metadata()
                except Exception as e:
                    logger.error(f"Heartbeat failed: {e}")
                
                # Sleep with interruption check
                for _ in range(self.config.heartbeat_interval):
                    if self.stop_heartbeat.is_set():
                        break
                    time.sleep(1)
        
        self.heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            daemon=True,
            name="checkpoint-heartbeat"
        )
        self.heartbeat_thread.start()
        logger.info("Started heartbeat monitoring")
    
    def _stop_heartbeat(self):
        """Stop heartbeat monitoring thread."""
        if self.heartbeat_thread:
            self.stop_heartbeat.set()
            self.heartbeat_thread.join(timeout=5)
            self.heartbeat_thread = None
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        gradient_accumulation_step: int = 0,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save checkpoint asynchronously to cloud storage."""
        if not self.is_main_process:
            return ""
        
        checkpoint_id = f"checkpoint-{step}-{uuid.uuid4().hex[:8]}"
        checkpoint_dir = f"{self.config.checkpoint_dir}/{checkpoint_id}"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'model_state_dict': self._get_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'gradient_accumulation_step': gradient_accumulation_step,
            'timestamp': time.time()
        }
        
        # Add optional states
        if self.config.save_rng_state:
            checkpoint_data['rng_state'] = self._get_rng_state()
        
        if extra_state:
            checkpoint_data['extra_state'] = extra_state
        
        # Save locally first
        local_checkpoint_path = f"/tmp/{checkpoint_id}.pt"
        torch.save(checkpoint_data, local_checkpoint_path)
        
        # Upload to cloud storage
        remote_path = f"{checkpoint_dir}/checkpoint.pt"
        
        if self.config.async_save and self.save_executor:
            future = self.save_executor.submit(
                self._upload_with_retry,
                local_checkpoint_path,
                remote_path,
                checkpoint_id,
                step
            )
            future.add_done_callback(lambda f: self._cleanup_local(local_checkpoint_path))
        else:
            success = self._upload_with_retry(
                local_checkpoint_path,
                remote_path,
                checkpoint_id,
                step
            )
            self._cleanup_local(local_checkpoint_path)
            if not success:
                raise RuntimeError(f"Failed to save checkpoint at step {step}")
        
        return checkpoint_id
    
    def _get_model_state_dict(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get model state dict, handling DDP wrapper."""
        if isinstance(model, DDP):
            return model.module.state_dict()
        return model.state_dict()
    
    def _get_rng_state(self) -> Dict[str, Any]:
        """Get RNG state for reproducibility."""
        return {
            'cpu_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng_state': None,  # Would need numpy import
            'python_rng_state': None  # Would need random module
        }
    
    def _upload_with_retry(
        self,
        local_path: str,
        remote_path: str,
        checkpoint_id: str,
        step: int
    ) -> bool:
        """Upload checkpoint with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                success = self.storage.upload(local_path, remote_path)
                if success:
                    # Update metadata
                    self.metadata.add_checkpoint(step, checkpoint_id, time.time())
                    
                    # Prune old checkpoints
                    to_remove = self.metadata.prune_checkpoints(self.config.save_total_limit)
                    for old_path in to_remove:
                        self._delete_checkpoint(old_path)
                    
                    self._save_metadata()
                    logger.info(f"Saved checkpoint at step {step} (attempt {attempt + 1})")
                    return True
            except Exception as e:
                logger.error(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        return False
    
    def _cleanup_local(self, local_path: str):
        """Clean up local temporary file."""
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass
    
    def _delete_checkpoint(self, checkpoint_id: str):
        """Delete checkpoint from storage."""
        checkpoint_dir = f"{self.config.checkpoint_dir}/{checkpoint_id}"
        remote_path = f"{checkpoint_dir}/checkpoint.pt"
        self.storage.delete(remote_path)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_id: Optional[str] = None
    ) -> Tuple[int, int, Dict[str, Any]]:
        """Load checkpoint from cloud storage.
        
        Returns:
            Tuple of (step, gradient_accumulation_step, extra_state)
        """
        # Find checkpoint to load
        if checkpoint_id is None:
            # Check for recovery from preemption
            if self._is_preempted():
                logger.warning("Detected preemption, recovering from latest checkpoint")
            
            # Load latest checkpoint
            if not self.metadata.checkpoints:
                logger.info("No checkpoints found, starting from scratch")
                return 0, 0, {}
            
            checkpoint_id = self.metadata.latest_checkpoint
        
        # Download checkpoint
        checkpoint_dir = f"{self.config.checkpoint_dir}/{checkpoint_id}"
        remote_path = f"{checkpoint_dir}/checkpoint.pt"
        local_path = f"/tmp/load_{checkpoint_id}.pt"
        
        success = False
        for attempt in range(self.config.max_retries):
            try:
                success = self.storage.download(remote_path, local_path)
                if success:
                    break
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        if not success:
            raise RuntimeError(f"Failed to download checkpoint {checkpoint_id}")
        
        # Load checkpoint data
        try:
            checkpoint_data = torch.load(local_path, map_location='cpu')
            
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load optimizer state
            if self.config.save_optimizer and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Load scheduler state
            if (self.config.save_scheduler and 
                scheduler is not None and 
                'scheduler_state_dict' in checkpoint_data):
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Load RNG state
            if self.config.save_rng_state and 'rng_state' in checkpoint_data:
                self._restore_rng_state(checkpoint_data['rng_state'])
            
            step = checkpoint_data['step']
            grad_accum_step = checkpoint_data.get('gradient_accumulation_step', 0)
            extra_state = checkpoint_data.get('extra_state', {})
            
            logger.info(f"Loaded checkpoint from step {step}")
            
            # Cleanup
            self._cleanup_local(local_path)
            
            return step, grad_accum_step, extra_state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _restore_rng_state(self, rng_state: Dict[str, Any]):
        """Restore RNG state from checkpoint."""
        if 'cpu_rng_state' in rng_state:
            torch.set_rng_state(rng_state['cpu_rng_state'])
        
        if (torch.cuda.is_available() and 
            'cuda_rng_state' in rng_state and 
            rng_state['cuda_rng_state'] is not None):
            torch.cuda.set_rng_state_all(rng_state['cuda_rng_state'])
    
    def _is_preempted(self) -> bool:
        """Check if training was preempted based on heartbeat."""
        if not self.metadata.heartbeat:
            return False
        
        time_since_heartbeat = time.time() - self.metadata.heartbeat
        # Consider preempted if no heartbeat for 3x the interval
        return time_since_heartbeat > (self.config.heartbeat_interval * 3)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the ID of the latest checkpoint."""
        return self.metadata.latest_checkpoint
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.metadata.checkpoints.copy()
    
    def cleanup(self):
        """Cleanup resources."""
        self._stop_heartbeat()
        if self.save_executor:
            self.save_executor.shutdown(wait=True)


class GradientAccumulationState:
    """Preserves gradient accumulation state across interruptions."""
    
    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_gradients = {}
    
    def should_sync_gradients(self) -> bool:
        """Check if gradients should be synchronized."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def step(self):
        """Increment accumulation step."""
        self.current_step += 1
    
    def reset(self):
        """Reset accumulation state."""
        self.current_step = 0
        self.accumulated_gradients.clear()
    
    def save_state(self) -> Dict[str, Any]:
        """Save accumulation state for checkpointing."""
        return {
            'current_step': self.current_step,
            'accumulation_steps': self.accumulation_steps
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load accumulation state from checkpoint."""
        self.current_step = state.get('current_step', 0)
        self.accumulation_steps = state.get('accumulation_steps', self.accumulation_steps)


class ResilientTrainer:
    """Wrapper for training with checkpoint resilience."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: CheckpointConfig,
        local_rank: int = 0,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.local_rank = local_rank
        
        # Initialize checkpoint manager
        self.checkpoint_manager = ResilientCheckpointManager(config, local_rank)
        
        # Gradient accumulation state
        self.gradient_accumulation = GradientAccumulationState(gradient_accumulation_steps)
        
        # Training state
        self.global_step = 0
        self.start_step = 0
        
        # Try to resume from checkpoint
        self._resume_from_checkpoint()
    
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint if available."""
        try:
            step, grad_accum_step, extra_state = self.checkpoint_manager.load_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler
            )
            
            self.global_step = step
            self.start_step = step
            self.gradient_accumulation.current_step = grad_accum_step
            
            # Restore any extra state
            if 'epoch' in extra_state:
                self.current_epoch = extra_state['epoch']
            
            logger.info(f"Resumed training from step {step}")
            
        except Exception as e:
            logger.warning(f"Could not resume from checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    def training_step(self, batch: Any, step: int) -> torch.Tensor:
        """Execute a single training step with gradient accumulation."""
        # Forward pass
        outputs = self.model(batch)
        loss = outputs.loss / self.gradient_accumulation.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Check if we should sync gradients
        if self.gradient_accumulation.should_sync_gradients():
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Update global step
            self.global_step += 1
            
            # Check if we should save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint()
        
        # Update accumulation step
        self.gradient_accumulation.step()
        
        return loss * self.gradient_accumulation.accumulation_steps
    
    def _save_checkpoint(self):
        """Save checkpoint with current state."""
        extra_state = {
            'epoch': getattr(self, 'current_epoch', 0),
            'global_step': self.global_step
        }
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            gradient_accumulation_step=self.gradient_accumulation.current_step,
            extra_state=extra_state
        )
    
    def save_final_checkpoint(self):
        """Save final checkpoint at end of training."""
        if self.local_rank == 0:
            self._save_checkpoint()
    
    def cleanup(self):
        """Cleanup resources."""
        self.checkpoint_manager.cleanup()


# Utility functions for integration with existing codebase
def create_resilient_checkpoint_manager(
    checkpoint_dir: str,
    storage_backend: str = "s3",
    save_interval: int = 1000,
    **kwargs
) -> ResilientCheckpointManager:
    """Factory function to create checkpoint manager."""
    backend = StorageBackend(storage_backend)
    config = CheckpointConfig(
        storage_backend=backend,
        checkpoint_dir=checkpoint_dir,
        save_interval=save_interval,
        **kwargs
    )
    return ResilientCheckpointManager(config)


def wrap_model_with_resilience(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_dir: str,
    storage_backend: str = "s3",
    gradient_accumulation_steps: int = 1,
    **kwargs
) -> ResilientTrainer:
    """Wrap model and optimizer with resilience features."""
    backend = StorageBackend(storage_backend)
    config = CheckpointConfig(
        storage_backend=backend,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )
    
    return ResilientTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        gradient_accumulation_steps=gradient_accumulation_steps
    )


# Example usage in training loop
def example_training_loop():
    """Example of how to use the resilience features."""
    # Initialize model, optimizer, scheduler
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    # Create resilient trainer
    trainer = wrap_model_with_resilience(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir="s3://my-bucket/checkpoints",
        storage_backend="s3",
        save_interval=100,
        gradient_accumulation_steps=4,
        cloud_credentials={
            'aws_access_key_id': 'YOUR_KEY',
            'aws_secret_access_key': 'YOUR_SECRET',
            'bucket_name': 'my-bucket'
        }
    )
    
    # Training loop
    for epoch in range(10):
        trainer.current_epoch = epoch
        for step, batch in enumerate(get_training_batches()):
            loss = trainer.training_step(batch, step)
            
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
    
    # Save final checkpoint
    trainer.save_final_checkpoint()
    trainer.cleanup()


def get_training_batches():
    """Dummy function for example."""
    for i in range(1000):
        yield torch.randn(32, 10)