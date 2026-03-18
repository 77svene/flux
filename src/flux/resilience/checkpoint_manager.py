"""
Fault-Tolerant Training Checkpoint Manager for flux
"""

import os
import time
import json
import logging
import threading
import tempfile
import shutil
import hashlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
import atexit

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_METADATA_FILE = "checkpoint_metadata.json"
TRAINING_STATE_FILE = "training_state.pt"
MODEL_WEIGHTS_FILE = "model.safetensors"  # Using safetensors for safety
OPTIMIZER_STATE_FILE = "optimizer.pt"
SCHEDULER_STATE_FILE = "scheduler.pt"
GRADIENT_ACCUMULATION_FILE = "gradient_accumulation.pt"
RNG_STATE_FILE = "rng_state.pt"
CHECKPOINT_INDEX_FILE = "latest_checkpoint.txt"
HEARTBEAT_FILE = ".heartbeat"

@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    global_step: int
    epoch: int
    timestamp: str
    model_hash: str
    optimizer_hash: str
    training_args: Dict[str, Any]
    metrics: Dict[str, float]
    gradient_accumulation_step: int
    is_complete: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(**data)

class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def upload(self, local_path: Union[str, Path], remote_path: str) -> bool:
        """Upload a file to storage"""
        pass
    
    @abstractmethod
    def download(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        """Download a file from storage"""
        pass
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with given prefix"""
        pass
    
    @abstractmethod
    def delete(self, remote_path: str) -> bool:
        """Delete a file from storage"""
        pass
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if a file exists"""
        pass

class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload(self, local_path: Union[str, Path], remote_path: str) -> bool:
        try:
            dest = self.base_path / remote_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {remote_path}: {e}")
            return False
    
    def download(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        try:
            src = self.base_path / remote_path
            if not src.exists():
                return False
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            return True
        except Exception as e:
            logger.error(f"Failed to download {remote_path} to {local_path}: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        files = []
        prefix_path = self.base_path / prefix if prefix else self.base_path
        for file_path in prefix_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.base_path)
                files.append(str(rel_path))
        return files
    
    def delete(self, remote_path: str) -> bool:
        try:
            file_path = self.base_path / remote_path
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        return (self.base_path / remote_path).exists()

class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend"""
    
    def __init__(self, bucket_name: str, prefix: str = "", endpoint_url: Optional[str] = None):
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")
        self.endpoint_url = endpoint_url
        self._s3_client = None
    
    @property
    def s3_client(self):
        if self._s3_client is None:
            try:
                import boto3
                self._s3_client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url
                )
            except ImportError:
                raise ImportError("boto3 is required for S3 storage backend. Install with: pip install boto3")
        return self._s3_client
    
    def _get_full_path(self, path: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path
    
    def upload(self, local_path: Union[str, Path], remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            self.s3_client.upload_file(str(local_path), self.bucket_name, full_path)
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
    
    def download(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, full_path, str(dest))
            return True
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        try:
            full_prefix = self._get_full_path(prefix)
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix)
            
            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if self.prefix:
                            # Remove the prefix from the key
                            if key.startswith(self.prefix + "/"):
                                key = key[len(self.prefix) + 1:]
                        files.append(key)
            return files
        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=full_path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=full_path)
            return True
        except:
            return False

class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend"""
    
    def __init__(self, bucket_name: str, prefix: str = ""):
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")
        self._client = None
        self._bucket = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client()
                self._bucket = self._client.bucket(self.bucket_name)
            except ImportError:
                raise ImportError("google-cloud-storage is required for GCS backend. Install with: pip install google-cloud-storage")
        return self._client
    
    @property
    def bucket(self):
        if self._bucket is None:
            _ = self.client  # Initialize client and bucket
        return self._bucket
    
    def _get_full_path(self, path: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path
    
    def upload(self, local_path: Union[str, Path], remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            blob = self.bucket.blob(full_path)
            blob.upload_from_filename(str(local_path))
            return True
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            return False
    
    def download(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            blob = self.bucket.blob(full_path)
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
            return True
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        try:
            full_prefix = self._get_full_path(prefix)
            blobs = self.bucket.list_blobs(prefix=full_prefix)
            
            files = []
            for blob in blobs:
                name = blob.name
                if self.prefix:
                    # Remove the prefix from the name
                    if name.startswith(self.prefix + "/"):
                        name = name[len(self.prefix) + 1:]
                files.append(name)
            return files
        except Exception as e:
            logger.error(f"Failed to list GCS files: {e}")
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            blob = self.bucket.blob(full_path)
            blob.delete()
            return True
        except Exception as e:
            logger.error(f"Failed to delete from GCS: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            blob = self.bucket.blob(full_path)
            return blob.exists()
        except:
            return False

class HDFSStorageBackend(StorageBackend):
    """HDFS storage backend"""
    
    def __init__(self, base_path: str, user: Optional[str] = None):
        self.base_path = base_path.rstrip("/")
        self.user = user
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import hdfs
                self._client = hdfs.InsecureClient(self.base_path, user=self.user)
            except ImportError:
                raise ImportError("hdfs is required for HDFS backend. Install with: pip install hdfs")
        return self._client
    
    def _get_full_path(self, path: str) -> str:
        return f"{self.base_path}/{path}"
    
    def upload(self, local_path: Union[str, Path], remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            self.client.upload(full_path, str(local_path), overwrite=True)
            return True
        except Exception as e:
            logger.error(f"Failed to upload to HDFS: {e}")
            return False
    
    def download(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.client.download(full_path, str(dest), overwrite=True)
            return True
        except Exception as e:
            logger.error(f"Failed to download from HDFS: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        try:
            full_prefix = self._get_full_path(prefix)
            files = []
            for path, dirs, file_names in self.client.walk(full_prefix):
                for file_name in file_names:
                    full_file_path = f"{path}/{file_name}"
                    # Make path relative to base_path
                    if full_file_path.startswith(self.base_path + "/"):
                        rel_path = full_file_path[len(self.base_path) + 1:]
                        files.append(rel_path)
            return files
        except Exception as e:
            logger.error(f"Failed to list HDFS files: {e}")
            return []
    
    def delete(self, remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            self.client.delete(full_path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete from HDFS: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        try:
            full_path = self._get_full_path(remote_path)
            status = self.client.status(full_path, strict=False)
            return status is not None
        except:
            return False

class HeartbeatMonitor:
    """Monitors training heartbeat and triggers recovery on timeout"""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        timeout_seconds: int = 300,
        check_interval: int = 30
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self.heartbeat_file = self.checkpoint_dir / HEARTBEAT_FILE
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._on_timeout_callback = None
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self, on_timeout_callback: Optional[Callable] = None):
        """Start heartbeat monitoring"""
        self._on_timeout_callback = on_timeout_callback
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="heartbeat-monitor"
        )
        self._monitor_thread.start()
        logger.info(f"Heartbeat monitor started (timeout: {self.timeout_seconds}s)")
    
    def stop(self):
        """Stop heartbeat monitoring"""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("Heartbeat monitor stopped")
    
    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        try:
            self.heartbeat_file.write_text(str(time.time()))
        except Exception as e:
            logger.warning(f"Failed to update heartbeat: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                if self.heartbeat_file.exists():
                    last_heartbeat = float(self.heartbeat_file.read_text())
                    elapsed = time.time() - last_heartbeat
                    
                    if elapsed > self.timeout_seconds:
                        logger.error(f"Heartbeat timeout detected (elapsed: {elapsed:.1f}s)")
                        if self._on_timeout_callback:
                            self._on_timeout_callback()
                        break
                else:
                    # No heartbeat file yet, create one
                    self.update_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
            
            # Wait for next check
            self._stop_event.wait(self.check_interval)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class CheckpointManager:
    """
    Manages fault-tolerant checkpointing for distributed training.
    
    Features:
    - Async checkpointing with configurable storage backends
    - Training resumption from any point
    - Gradient accumulation state preservation
    - Heartbeat monitoring for preemption detection
    - Automatic cleanup of old checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        storage_backend: Optional[StorageBackend] = None,
        checkpoint_interval: int = 1000,
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_rng_state: bool = True,
        save_gradient_accumulation: bool = True,
        async_upload: bool = True,
        heartbeat_timeout: int = 300,
        use_safetensors: bool = True,
        model_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Local directory for temporary checkpoint storage
            storage_backend: Backend for persistent storage (defaults to local)
            checkpoint_interval: Steps between checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            save_rng_state: Whether to save RNG states
            save_gradient_accumulation: Whether to save gradient accumulation state
            async_upload: Whether to upload checkpoints asynchronously
            heartbeat_timeout: Seconds before heartbeat timeout triggers recovery
            use_safetensors: Use safetensors for model weights (safer)
            model_config: Model configuration for checkpoint metadata
            training_args: Training arguments for checkpoint metadata
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.storage_backend = storage_backend or LocalStorageBackend(self.checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_rng_state = save_rng_state
        self.save_gradient_accumulation = save_gradient_accumulation
        self.async_upload = async_upload
        self.use_safetensors = use_safetensors
        self.model_config = model_config or {}
        self.training_args = training_args or {}
        
        # State tracking
        self._global_step = 0
        self._epoch = 0
        self._gradient_accumulation_step = 0
        self._last_checkpoint_step = 0
        self._checkpoints: List[str] = []
        self._upload_futures: List[Future] = []
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="checkpoint-upload")
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize heartbeat monitor
        self.heartbeat_monitor = HeartbeatMonitor(
            checkpoint_dir=self.checkpoint_dir,
            timeout_seconds=heartbeat_timeout
        )
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        # Load existing checkpoint metadata
        self._load_checkpoint_index()
        
        logger.info(f"CheckpointManager initialized (dir: {checkpoint_dir}, interval: {checkpoint_interval})")
    
    def _load_checkpoint_index(self):
        """Load checkpoint index from storage"""
        index_file = self.checkpoint_dir / CHECKPOINT_INDEX_FILE
        
        # Try to download from storage if not exists locally
        if not index_file.exists():
            self.storage_backend.download(CHECKPOINT_INDEX_FILE, index_file)
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self._checkpoints = content.split('\n')
                        logger.info(f"Loaded {len(self._checkpoints)} existing checkpoints")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint index: {e}")
    
    def _save_checkpoint_index(self):
        """Save checkpoint index to storage"""
        index_file = self.checkpoint_dir / CHECKPOINT_INDEX_FILE
        try:
            with open(index_file, 'w') as f:
                f.write('\n'.join(self._checkpoints))
            
            # Upload to storage
            if self.async_upload:
                future = self._executor.submit(
                    self.storage_backend.upload,
                    index_file,
                    CHECKPOINT_INDEX_FILE
                )
                self._upload_futures.append(future)
            else:
                self.storage_backend.upload(index_file, CHECKPOINT_INDEX_FILE)
        except Exception as e:
            logger.error(f"Failed to save checkpoint index: {e}")
    
    def _generate_checkpoint_id(self, step: int) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint-{step}-{timestamp}"
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for integrity checking"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy().tobytes()
        elif hasattr(data, 'state_dict'):
            data = pickle.dumps(data.state_dict())
        else:
            data = pickle.dumps(data)
        
        return hashlib.sha256(data).hexdigest()[:16]
    
    def _save_model_weights(self, model: torch.nn.Module, save_path: Path) -> str:
        """Save model weights and return hash"""
        model_to_save = model.module if isinstance(model, DDP) else model
        
        if self.use_safetensors:
            try:
                from safetensors.torch import save_file
                state_dict = model_to_save.state_dict()
                # Convert to CPU and ensure contiguous
                state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
                save_file(state_dict, save_path / MODEL_WEIGHTS_FILE)
            except ImportError:
                logger.warning("safetensors not installed, falling back to torch.save")
                torch.save(model_to_save.state_dict(), save_path / MODEL_WEIGHTS_FILE)
        else:
            torch.save(model_to_save.state_dict(), save_path / MODEL_WEIGHTS_FILE)
        
        return self._compute_hash(model_to_save.state_dict())
    
    def _save_optimizer_state(self, optimizer: torch.optim.Optimizer, save_path: Path) -> str:
        """Save optimizer state and return hash"""
        torch.save(optimizer.state_dict(), save_path / OPTIMIZER_STATE_FILE)
        return self._compute_hash(optimizer.state_dict())
    
    def _save_scheduler_state(self, scheduler: Any, save_path: Path) -> bool:
        """Save scheduler state"""
        if scheduler is None:
            return False
        
        try:
            torch.save(scheduler.state_dict(), save_path / SCHEDULER_STATE_FILE)
            return True
        except Exception as e:
            logger.warning(f"Failed to save scheduler state: {e}")
            return False
    
    def _save_rng_state(self, save_path: Path) -> bool:
        """Save RNG states for reproducibility"""
        try:
            rng_states = {
                'torch': torch.get_rng_state(),
                'numpy': None,
                'cuda': None,
                'python': None
            }
            
            # Try to save numpy RNG state
            try:
                import numpy as np
                rng_states['numpy'] = np.random.get_state()
            except ImportError:
                pass
            
            # Try to save CUDA RNG states
            if torch.cuda.is_available():
                rng_states['cuda'] = torch.cuda.get_rng_state_all()
            
            # Try to save Python RNG state
            try:
                import random
                rng_states['python'] = random.getstate()
            except:
                pass
            
            torch.save(rng_states, save_path / RNG_STATE_FILE)
            return True
        except Exception as e:
            logger.warning(f"Failed to save RNG state: {e}")
            return False
    
    def _save_gradient_accumulation_state(self, save_path: Path) -> bool:
        """Save gradient accumulation state"""
        if not self.save_gradient_accumulation:
            return False
        
        try:
            state = {
                'global_step': self._global_step,
                'gradient_accumulation_step': self._gradient_accumulation_step,
                'epoch': self._epoch
            }
            torch.save(state, save_path / GRADIENT_ACCUMULATION_FILE)
            return True
        except Exception as e:
            logger.warning(f"Failed to save gradient accumulation state: {e}")
            return False
    
    def _save_training_state(self, save_path: Path, metrics: Dict[str, float]) -> bool:
        """Save training state and metadata"""
        try:
            metadata = CheckpointMetadata(
                checkpoint_id=save_path.name,
                global_step=self._global_step,
                epoch=self._epoch,
                timestamp=datetime.now().isoformat(),
                model_hash="",  # Will be filled later
                optimizer_hash="",  # Will be filled later
                training_args=self.training_args,
                metrics=metrics,
                gradient_accumulation_step=self._gradient_accumulation_step,
                is_complete=True
            )
            
            with open(save_path / CHECKPOINT_METADATA_FILE, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")
            return False
    
    def _upload_checkpoint(self, checkpoint_id: str, local_path: Path):
        """Upload checkpoint to storage backend"""
        try:
            # Upload all files in checkpoint directory
            for file_path in local_path.iterdir():
                if file_path.is_file():
                    remote_path = f"{checkpoint_id}/{file_path.name}"
                    success = self.storage_backend.upload(file_path, remote_path)
                    if not success:
                        logger.error(f"Failed to upload {file_path.name}")
                        return False
            
            logger.info(f"Checkpoint {checkpoint_id} uploaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to upload checkpoint {checkpoint_id}: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints"""
        if len(self._checkpoints) <= self.max_checkpoints:
            return
        
        # Sort checkpoints by step number
        sorted_checkpoints = sorted(
            self._checkpoints,
            key=lambda x: int(x.split('-')[1]) if x.startswith('checkpoint-') else 0
        )
        
        # Remove oldest checkpoints
        to_remove = sorted_checkpoints[:len(sorted_checkpoints) - self.max_checkpoints]
        
        for checkpoint_id in to_remove:
            try:
                # Remove from storage
                files = self.storage_backend.list_files(checkpoint_id)
                for file in files:
                    self.storage_backend.delete(f"{checkpoint_id}/{file}")
                
                # Remove from local
                local_dir = self.checkpoint_dir / checkpoint_id
                if local_dir.exists():
                    shutil.rmtree(local_dir)
                
                # Remove from index
                if checkpoint_id in self._checkpoints:
                    self._checkpoints.remove(checkpoint_id)
                
                logger.info(f"Removed old checkpoint: {checkpoint_id}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {checkpoint_id}: {e}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        force: bool = False
    ) -> Optional[str]:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Optional scheduler to save
            metrics: Training metrics to save
            force: Force save regardless of interval
        
        Returns:
            Checkpoint ID if saved, None otherwise
        """
        metrics = metrics or {}
        
        # Check if we should save
        steps_since_last = self._global_step - self._last_checkpoint_step
        if not force and steps_since_last < self.checkpoint_interval:
            return None
        
        # Update heartbeat
        self.heartbeat_monitor.update_heartbeat()
        
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(self._global_step)
        local_path = self.checkpoint_dir / checkpoint_id
        local_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint {checkpoint_id} at step {self._global_step}")
        
        try:
            # Save model weights
            model_hash = self._save_model_weights(model, local_path)
            
            # Save optimizer state
            optimizer_hash = ""
            if self.save_optimizer:
                optimizer_hash = self._save_optimizer_state(optimizer, local_path)
            
            # Save scheduler state
            if self.save_scheduler and scheduler is not None:
                self._save_scheduler_state(scheduler, local_path)
            
            # Save RNG state
            if self.save_rng_state:
                self._save_rng_state(local_path)
            
            # Save gradient accumulation state
            if self.save_gradient_accumulation:
                self._save_gradient_accumulation_state(local_path)
            
            # Save training state and metadata
            self._save_training_state(local_path, metrics)
            
            # Update metadata with hashes
            metadata_path = local_path / CHECKPOINT_METADATA_FILE
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['model_hash'] = model_hash
                metadata['optimizer_hash'] = optimizer_hash
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Upload checkpoint
            if self.async_upload:
                future = self._executor.submit(
                    self._upload_checkpoint,
                    checkpoint_id,
                    local_path
                )
                self._upload_futures.append(future)
            else:
                self._upload_checkpoint(checkpoint_id, local_path)
            
            # Update checkpoint index
            self._checkpoints.append(checkpoint_id)
            self._save_checkpoint_index()
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Update state
            self._last_checkpoint_step = self._global_step
            
            logger.info(f"Checkpoint {checkpoint_id} saved successfully")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up partial checkpoint
            if local_path.exists():
                shutil.rmtree(local_path)
            return None
    
    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint to load (loads latest if None)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load tensors to
        
        Returns:
            Dictionary with loaded state information
        """
        device = device or torch.device('cpu')
        
        # Find checkpoint to load
        if checkpoint_id is None:
            checkpoint_id = self._get_latest_checkpoint_id()
        
        if checkpoint_id is None:
            logger.info("No checkpoint found to load")
            return {}
        
        logger.info(f"Loading checkpoint {checkpoint_id}")
        
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            local_path = self.checkpoint_dir / checkpoint_id
            
            # Download checkpoint if not exists locally
            if not local_path.exists():
                local_path.mkdir(parents=True, exist_ok=True)
                
                # Download all files
                files = self.storage_backend.list_files(checkpoint_id)
                for file in files:
                    remote_path = f"{checkpoint_id}/{file}"
                    local_file = local_path / file
                    success = self.storage_backend.download(remote_path, local_file)
                    if not success:
                        logger.error(f"Failed to download {file}")
                        raise RuntimeError(f"Failed to download checkpoint file: {file}")
            
            # Load metadata
            metadata_path = local_path / CHECKPOINT_METADATA_FILE
            if not metadata_path.exists():
                raise RuntimeError(f"Checkpoint metadata not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata = CheckpointMetadata.from_dict(json.load(f))
            
            # Load model weights
            if model is not None:
                model_path = local_path / MODEL_WEIGHTS_FILE
                if model_path.exists():
                    if self.use_safetensors and model_path.suffix == '.safetensors':
                        try:
                            from safetensors.torch import load_file
                            state_dict = load_file(model_path, device=str(device))
                        except ImportError:
                            logger.warning("safetensors not installed, falling back to torch.load")
                            state_dict = torch.load(model_path, map_location=device)
                    else:
                        state_dict = torch.load(model_path, map_location=device)
                    
                    model_to_load = model.module if isinstance(model, DDP) else model
                    model_to_load.load_state_dict(state_dict)
                    logger.info("Model weights loaded")
            
            # Load optimizer state
            if optimizer is not None and self.save_optimizer:
                optimizer_path = local_path / OPTIMIZER_STATE_FILE
                if optimizer_path.exists():
                    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                    logger.info("Optimizer state loaded")
            
            # Load scheduler state
            if scheduler is not None and self.save_scheduler:
                scheduler_path = local_path / SCHEDULER_STATE_FILE
                if scheduler_path.exists():
                    scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
                    logger.info("Scheduler state loaded")
            
            # Load RNG state
            if self.save_rng_state:
                rng_path = local_path / RNG_STATE_FILE
                if rng_path.exists():
                    rng_states = torch.load(rng_path, map_location=device)
                    
                    # Restore torch RNG state
                    if 'torch' in rng_states:
                        torch.set_rng_state(rng_states['torch'])
                    
                    # Restore numpy RNG state
                    if 'numpy' in rng_states and rng_states['numpy'] is not None:
                        try:
                            import numpy as np
                            np.random.set_state(rng_states['numpy'])
                        except ImportError:
                            pass
                    
                    # Restore CUDA RNG states
                    if 'cuda' in rng_states and rng_states['cuda'] is not None:
                        if torch.cuda.is_available():
                            torch.cuda.set_rng_state_all(rng_states['cuda'])
                    
                    # Restore Python RNG state
                    if 'python' in rng_states and rng_states['python'] is not None:
                        try:
                            import random
                            random.setstate(rng_states['python'])
                        except:
                            pass
                    
                    logger.info("RNG states restored")
            
            # Load gradient accumulation state
            gradient_state = {}
            if self.save_gradient_accumulation:
                grad_path = local_path / GRADIENT_ACCUMULATION_FILE
                if grad_path.exists():
                    gradient_state = torch.load(grad_path, map_location=device)
                    self._global_step = gradient_state.get('global_step', 0)
                    self._gradient_accumulation_step = gradient_state.get('gradient_accumulation_step', 0)
                    self._epoch = gradient_state.get('epoch', 0)
                    logger.info(f"Gradient accumulation state restored (step: {self._gradient_accumulation_step})")
            
            # Update internal state
            self._last_checkpoint_step = metadata.global_step
            
            logger.info(f"Checkpoint {checkpoint_id} loaded successfully (step: {metadata.global_step})")
            
            return {
                'metadata': metadata,
                'gradient_state': gradient_state,
                'global_step': metadata.global_step,
                'epoch': metadata.epoch,
                'metrics': metadata.metrics
            }
    
    def _get_latest_checkpoint_id(self) -> Optional[str]:
        """Get the latest checkpoint ID"""
        if not self._checkpoints:
            return None
        
        # Sort by step number
        sorted_checkpoints = sorted(
            self._checkpoints,
            key=lambda x: int(x.split('-')[1]) if x.startswith('checkpoint-') else 0,
            reverse=True
        )
        
        return sorted_checkpoints[0] if sorted_checkpoints else None
    
    def update_step(self, step: int, gradient_accumulation_step: int = 0, epoch: int = 0):
        """Update current training step"""
        self._global_step = step
        self._gradient_accumulation_step = gradient_accumulation_step
        self._epoch = epoch
        
        # Update heartbeat
        self.heartbeat_monitor.update_heartbeat()
    
    def should_save(self) -> bool:
        """Check if checkpoint should be saved at current step"""
        steps_since_last = self._global_step - self._last_checkpoint_step
        return steps_since_last >= self.checkpoint_interval
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about existing checkpoints"""
        return {
            'total_checkpoints': len(self._checkpoints),
            'checkpoints': self._checkpoints.copy(),
            'latest_checkpoint': self._get_latest_checkpoint_id(),
            'global_step': self._global_step,
            'epoch': self._epoch
        }
    
    def wait_for_uploads(self, timeout: Optional[float] = None):
        """Wait for all async uploads to complete"""
        if not self._upload_futures:
            return
        
        logger.info(f"Waiting for {len(self._upload_futures)} uploads to complete...")
        
        for future in self._upload_futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Upload failed: {e}")
        
        self._upload_futures.clear()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up CheckpointManager...")
        
        # Stop heartbeat monitor
        self.heartbeat_monitor.stop()
        
        # Wait for uploads
        self.wait_for_uploads(timeout=30)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("CheckpointManager cleanup complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

def create_storage_backend(
    backend_type: str,
    **kwargs
) -> StorageBackend:
    """
    Factory function to create storage backend.
    
    Args:
        backend_type: Type of backend ('local', 's3', 'gcs', 'hdfs')
        **kwargs: Backend-specific arguments
    
    Returns:
        StorageBackend instance
    """
    backend_type = backend_type.lower()
    
    if backend_type == 'local':
        return LocalStorageBackend(**kwargs)
    elif backend_type == 's3':
        return S3StorageBackend(**kwargs)
    elif backend_type == 'gcs':
        return GCSStorageBackend(**kwargs)
    elif backend_type == 'hdfs':
        return HDFSStorageBackend(**kwargs)
    else:
        raise ValueError(f"Unknown storage backend type: {backend_type}")

def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    storage_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> CheckpointManager:
    """
    Factory function to create checkpoint manager with storage backend.
    
    Args:
        checkpoint_dir: Checkpoint directory
        storage_config: Storage backend configuration
        **kwargs: Additional CheckpointManager arguments
    
    Returns:
        CheckpointManager instance
    """
    storage_backend = None
    
    if storage_config:
        backend_type = storage_config.pop('type', 'local')
        storage_backend = create_storage_backend(backend_type, **storage_config)
    
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        storage_backend=storage_backend,
        **kwargs
    )

# Integration with existing flux training
class TrainingResumer:
    """Helper class to integrate checkpoint manager with training loops"""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Recovery state
        self.recovered_step = 0
        self.recovered_epoch = 0
        self.recovered_metrics = {}
    
    def attempt_recovery(self) -> bool:
        """Attempt to recover from latest checkpoint"""
        try:
            load_result = self.checkpoint_manager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device
            )
            
            if load_result:
                self.recovered_step = load_result.get('global_step', 0)
                self.recovered_epoch = load_result.get('epoch', 0)
                self.recovered_metrics = load_result.get('metrics', {})
                logger.info(f"Recovered from step {self.recovered_step}, epoch {self.recovered_epoch}")
                return True
        except Exception as e:
            logger.warning(f"Recovery failed: {e}")
        
        return False
    
    def on_training_step(
        self,
        step: int,
        gradient_accumulation_step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        force_save: bool = False
    ):
        """Called after each training step"""
        self.checkpoint_manager.update_step(step, gradient_accumulation_step, epoch)
        
        if force_save or self.checkpoint_manager.should_save():
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=metrics,
                force=force_save
            )
    
    def on_training_end(self, final_metrics: Optional[Dict[str, float]] = None):
        """Called at end of training"""
        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics=final_metrics,
            force=True
        )
        
        # Wait for uploads
        self.checkpoint_manager.wait_for_uploads()

# Example usage and integration
if __name__ == "__main__":
    # Example configuration
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint Manager Example")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--storage_type", type=str, default="local", 
                       choices=["local", "s3", "gcs", "hdfs"])
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--max_checkpoints", type=int, default=5)
    parser.add_argument("--heartbeat_timeout", type=int, default=300)
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create storage backend config
    storage_config = {"type": args.storage_type}
    
    if args.storage_type == "s3":
        storage_config.update({
            "bucket_name": os.getenv("S3_BUCKET", "my-checkpoints"),
            "prefix": "flux/checkpoints"
        })
    elif args.storage_type == "gcs":
        storage_config.update({
            "bucket_name": os.getenv("GCS_BUCKET", "my-checkpoints"),
            "prefix": "flux/checkpoints"
        })
    
    # Create checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=args.checkpoint_dir,
        storage_config=storage_config,
        checkpoint_interval=args.checkpoint_interval,
        max_checkpoints=args.max_checkpoints,
        heartbeat_timeout=args.heartbeat_timeout,
        async_upload=True
    )
    
    # Example model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create training resumer
    resumer = TrainingResumer(
        checkpoint_manager=checkpoint_manager,
        model=model,
        optimizer=optimizer
    )
    
    # Attempt recovery
    if resumer.attempt_recovery():
        start_step = resumer.recovered_step
        start_epoch = resumer.recovered_epoch
    else:
        start_step = 0
        start_epoch = 0
    
    # Simulate training loop
    try:
        for epoch in range(start_epoch, 10):
            for step in range(start_step, 1000):
                # Simulate training step
                loss = 1.0 / (step + 1)
                
                # Update checkpoint manager
                resumer.on_training_step(
                    step=step,
                    gradient_accumulation_step=step % 4,
                    epoch=epoch,
                    metrics={"loss": loss, "lr": 0.001}
                )
                
                if step % 100 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            
            start_step = 0  # Reset for next epoch
        
        # End training
        resumer.on_training_end(final_metrics={"final_loss": 0.01})
        
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        checkpoint_manager.cleanup()