import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import queue
import time
import logging
import pickle
import socket
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
import gc
import psutil
import GPUtil
from collections import defaultdict
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Import from existing modules
from modules import devices, shared, sd_models, sd_samplers
from modules.shared import opts, cmd_opts
from modules.processing import StableDiffusionProcessing, Processed
from modules.sd_hijack import model_hijack
from modules.sd_models import load_model, unload_model_weights

logger = logging.getLogger(__name__)

class ClusterState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SCALING = "scaling"

@dataclass
class ModelShard:
    """Represents a shard of the model assigned to a worker"""
    shard_id: str
    layer_indices: List[int]
    device: torch.device
    parameters: Dict[str, torch.Tensor]
    memory_required: float  # in MB
    computation_weight: float  # relative computation cost

@dataclass
class InferenceTask:
    """Task for distributed inference"""
    task_id: str
    prompt: str
    negative_prompt: str
    seed: int
    sampler_name: str
    steps: int
    cfg_scale: float
    width: int
    height: int
    batch_size: int
    priority: int = 0
    created_at: float = None
    assigned_worker: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class WorkerNode:
    """Represents a worker node in the cluster"""
    def __init__(self, worker_id: str, host: str, port: int, 
                 device: torch.device, capabilities: Dict):
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.device = device
        self.capabilities = capabilities
        self.state = ClusterState.INITIALIZING
        self.assigned_shards: List[ModelShard] = []
        self.current_tasks: Dict[str, InferenceTask] = {}
        self.last_heartbeat = time.time()
        self.load_average = 0.0
        self.memory_used = 0.0
        self.available_memory = 0.0
        self._lock = threading.Lock()
        
    def update_heartbeat(self):
        self.last_heartbeat = time.time()
        
    def is_alive(self, timeout: float = 30.0) -> bool:
        return time.time() - self.last_heartbeat < timeout
    
    def assign_shard(self, shard: ModelShard):
        with self._lock:
            self.assigned_shards.append(shard)
            self.memory_used += shard.memory_required
            
    def remove_shard(self, shard_id: str):
        with self._lock:
            self.assigned_shards = [s for s in self.assigned_shards 
                                   if s.shard_id != shard_id]
            self.recalculate_memory()
            
    def recalculate_memory(self):
        self.memory_used = sum(s.memory_required for s in self.assigned_shards)
        
    def to_dict(self) -> Dict:
        return {
            'worker_id': self.worker_id,
            'host': self.host,
            'port': self.port,
            'device': str(self.device),
            'state': self.state.value,
            'assigned_shards': len(self.assigned_shards),
            'current_tasks': len(self.current_tasks),
            'memory_used': self.memory_used,
            'load_average': self.load_average
        }

class ParameterServer:
    """Manages model parameters and sharding across workers"""
    def __init__(self, model: nn.Module):
        self.model = model
        self.shards: Dict[str, ModelShard] = {}
        self.parameter_locations: Dict[str, str] = {}  # param_name -> worker_id
        self.gradient_checkpointing = True
        self._lock = threading.Lock()
        
    def analyze_model(self) -> Dict:
        """Analyze model structure for optimal sharding"""
        analysis = {
            'total_parameters': 0,
            'layers': [],
            'memory_per_layer': [],
            'computation_per_layer': []
        }
        
        for name, param in self.model.named_parameters():
            analysis['total_parameters'] += param.numel()
            analysis['layers'].append(name)
            
            # Estimate memory (4 bytes per float32 parameter)
            memory_mb = (param.numel() * 4) / (1024 * 1024)
            analysis['memory_per_layer'].append(memory_mb)
            
            # Estimate computation (simplified)
            if 'weight' in name:
                analysis['computation_per_layer'].append(memory_mb * 2)
            else:
                analysis['computation_per_layer'].append(memory_mb)
                
        return analysis
    
    def create_shards(self, num_shards: int, 
                     worker_capabilities: List[Dict]) -> List[ModelShard]:
        """Create optimal shards based on worker capabilities"""
        analysis = self.analyze_model()
        total_memory = sum(analysis['memory_per_layer'])
        
        # Sort workers by available memory
        sorted_workers = sorted(worker_capabilities, 
                               key=lambda x: x.get('available_memory', 0),
                               reverse=True)
        
        shards = []
        layers_per_shard = len(analysis['layers']) // num_shards
        
        for i in range(num_shards):
            start_idx = i * layers_per_shard
            end_idx = start_idx + layers_per_shard if i < num_shards - 1 else len(analysis['layers'])
            
            shard_layers = analysis['layers'][start_idx:end_idx]
            shard_memory = sum(analysis['memory_per_layer'][start_idx:end_idx])
            shard_computation = sum(analysis['computation_per_layer'][start_idx:end_idx])
            
            shard = ModelShard(
                shard_id=f"shard_{i}",
                layer_indices=list(range(start_idx, end_idx)),
                device=torch.device('cpu'),  # Will be assigned later
                parameters={},
                memory_required=shard_memory,
                computation_weight=shard_computation
            )
            
            # Extract parameters for this shard
            for idx in range(start_idx, end_idx):
                param_name = analysis['layers'][idx]
                param = dict(self.model.named_parameters())[param_name]
                shard.parameters[param_name] = param.data.clone()
                
            shards.append(shard)
            
        return shards
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        self.gradient_checkpointing = True
        
    def get_parameter(self, param_name: str) -> Optional[torch.Tensor]:
        """Get parameter from wherever it's stored"""
        with self._lock:
            if param_name in self.parameter_locations:
                worker_id = self.parameter_locations[param_name]
                # In real implementation, would fetch from worker
                return None
            return None

class TaskScheduler:
    """Distributes and manages inference tasks across workers"""
    def __init__(self, max_queue_size: int = 1000):
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}
        self.worker_queues: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._task_counter = 0
        
    def add_task(self, task: InferenceTask) -> str:
        """Add a task to the queue with priority"""
        with self._lock:
            self._task_counter += 1
            task.task_id = f"task_{self._task_counter}_{int(time.time())}"
            
            # Negative priority for max-heap behavior
            self.task_queue.put((-task.priority, task.created_at, task))
            
            logger.info(f"Task {task.task_id} added to queue")
            return task.task_id
    
    def get_next_task(self, worker_id: str) -> Optional[InferenceTask]:
        """Get next task for a specific worker"""
        try:
            # Non-blocking get with timeout
            _, _, task = self.task_queue.get(timeout=0.1)
            task.assigned_worker = worker_id
            return task
        except queue.Empty:
            return None
    
    def complete_task(self, task_id: str, result: Any):
        """Mark task as completed"""
        with self._lock:
            self.completed_tasks[task_id] = {
                'result': result,
                'completed_at': time.time()
            }
            logger.info(f"Task {task_id} completed")
    
    def fail_task(self, task_id: str, error: Exception):
        """Mark task as failed"""
        with self._lock:
            self.failed_tasks[task_id] = {
                'error': str(error),
                'failed_at': time.time()
            }
            logger.error(f"Task {task_id} failed: {error}")
    
    def retry_failed_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        with self._lock:
            if task_id in self.failed_tasks:
                # Would need to recreate task from stored info
                del self.failed_tasks[task_id]
                return True
            return False
    
    def get_queue_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            'queue_size': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'throughput': self._calculate_throughput()
        }
    
    def _calculate_throughput(self) -> float:
        """Calculate tasks per minute"""
        if not self.completed_tasks:
            return 0.0
            
        recent_tasks = [
            t for t in self.completed_tasks.values()
            if time.time() - t['completed_at'] < 60
        ]
        return len(recent_tasks)

class DistributedInferenceCluster:
    """Main cluster manager for distributed inference"""
    def __init__(self, 
                 master_host: str = "localhost",
                 master_port: int = 29500,
                 discovery_port: int = 29501):
        self.master_host = master_host
        self.master_port = master_port
        self.discovery_port = discovery_port
        
        self.state = ClusterState.INITIALIZING
        self.workers: Dict[str, WorkerNode] = {}
        self.parameter_server: Optional[ParameterServer] = None
        self.task_scheduler = TaskScheduler()
        
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._discovery_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self.cluster_id = self._generate_cluster_id()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Performance metrics
        self.metrics = {
            'total_tasks_processed': 0,
            'total_processing_time': 0.0,
            'average_latency': 0.0,
            'gpu_utilization': defaultdict(list),
            'memory_usage': defaultdict(list)
        }
        
        logger.info(f"Initializing Distributed Inference Cluster {self.cluster_id}")
        
    def _generate_cluster_id(self) -> str:
        """Generate unique cluster ID"""
        hostname = socket.gethostname()
        timestamp = str(time.time())
        return hashlib.md5(f"{hostname}_{timestamp}".encode()).hexdigest()[:8]
    
    def initialize_cluster(self, model: nn.Module = None):
        """Initialize the cluster with optional model"""
        try:
            # Initialize parameter server if model provided
            if model is not None:
                self.parameter_server = ParameterServer(model)
                self.parameter_server.enable_gradient_checkpointing()
            
            # Start background threads
            self._start_background_threads()
            
            # Discover available GPUs
            self._discover_local_gpus()
            
            self.state = ClusterState.READY
            logger.info("Cluster initialized successfully")
            
        except Exception as e:
            self.state = ClusterState.ERROR
            logger.error(f"Failed to initialize cluster: {e}")
            raise
    
    def _start_background_threads(self):
        """Start all background threads"""
        self._stop_event.clear()
        
        # Heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="HeartbeatThread"
        )
        self._heartbeat_thread.start()
        
        # Task scheduler thread
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="SchedulerThread"
        )
        self._scheduler_thread.start()
        
        # Discovery thread
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop,
            daemon=True,
            name="DiscoveryThread"
        )
        self._discovery_thread.start()
    
    def _heartbeat_loop(self):
        """Monitor worker health"""
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                dead_workers = []
                
                for worker_id, worker in self.workers.items():
                    if not worker.is_alive():
                        dead_workers.append(worker_id)
                        logger.warning(f"Worker {worker_id} is dead")
                
                # Handle dead workers
                for worker_id in dead_workers:
                    self._handle_worker_failure(worker_id)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    def _scheduler_loop(self):
        """Distribute tasks to workers"""
        while not self._stop_event.is_set():
            try:
                # Find available workers
                available_workers = [
                    w for w in self.workers.values() 
                    if w.state == ClusterState.READY and len(w.current_tasks) < 2
                ]
                
                if available_workers and not self.task_scheduler.task_queue.empty():
                    # Simple round-robin assignment
                    for worker in available_workers:
                        task = self.task_scheduler.get_next_task(worker.worker_id)
                        if task:
                            self._assign_task_to_worker(worker, task)
                
                time.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    def _discovery_loop(self):
        """Discover new workers on the network"""
        while not self._stop_event.is_set():
            try:
                # In production, would listen for worker announcements
                # For now, just monitor local GPUs
                self._discover_local_gpus()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
    
    def _discover_local_gpus(self):
        """Discover and register local GPUs"""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                worker_id = f"local_gpu_{gpu.id}"
                
                if worker_id not in self.workers:
                    capabilities = {
                        'gpu_memory': gpu.memoryTotal,
                        'gpu_name': gpu.name,
                        'compute_capability': f"{gpu.major}.{gpu.minor}"
                    }
                    
                    worker = WorkerNode(
                        worker_id=worker_id,
                        host=self.master_host,
                        port=self.master_port + gpu.id,
                        device=torch.device(f'cuda:{gpu.id}'),
                        capabilities=capabilities
                    )
                    
                    self.workers[worker_id] = worker
                    logger.info(f"Discovered local GPU: {gpu.name} ({gpu.memoryTotal}MB)")
                    
        except Exception as e:
            logger.warning(f"Could not discover GPUs: {e}")
    
    def _handle_worker_failure(self, worker_id: str):
        """Handle worker failure and redistribute work"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            
            # Reassign tasks from failed worker
            for task_id, task in worker.current_tasks.items():
                task.assigned_worker = None
                self.task_scheduler.add_task(task)
                logger.info(f"Reassigned task {task_id} from failed worker")
            
            # Remove shards from failed worker
            for shard in worker.assigned_shards:
                if self.parameter_server:
                    # Would need to reassign shard to another worker
                    pass
            
            # Remove worker
            del self.workers[worker_id]
            logger.warning(f"Removed failed worker {worker_id}")
    
    def _assign_task_to_worker(self, worker: WorkerNode, task: InferenceTask):
        """Assign a task to a specific worker"""
        worker.current_tasks[task.task_id] = task
        worker.state = ClusterState.PROCESSING
        
        # Execute task asynchronously
        self.executor.submit(self._execute_task, worker, task)
    
    def _execute_task(self, worker: WorkerNode, task: InferenceTask):
        """Execute a task on a worker"""
        start_time = time.time()
        
        try:
            logger.info(f"Executing task {task.task_id} on worker {worker.worker_id}")
            
            # In production, would send task to worker via RPC
            # For now, simulate processing
            result = self._simulate_inference(task, worker.device)
            
            # Mark task as completed
            self.task_scheduler.complete_task(task.task_id, result)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['total_tasks_processed'] += 1
            self.metrics['total_processing_time'] += processing_time
            self.metrics['average_latency'] = (
                self.metrics['total_processing_time'] / 
                self.metrics['total_tasks_processed']
            )
            
            # Update worker state
            del worker.current_tasks[task.task_id]
            if not worker.current_tasks:
                worker.state = ClusterState.READY
                
            logger.info(f"Task {task.task_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            self.task_scheduler.fail_task(task.task_id, e)
            
            # Update worker state
            if task.task_id in worker.current_tasks:
                del worker.current_tasks[task.task_id]
            if not worker.current_tasks:
                worker.state = ClusterState.READY
    
    def _simulate_inference(self, task: InferenceTask, device: torch.device) -> Any:
        """Simulate inference (replace with actual inference in production)"""
        # This would be replaced with actual model inference
        time.sleep(0.5)  # Simulate computation
        
        # Return simulated result
        return {
            'task_id': task.task_id,
            'images': [],  # Would contain generated images
            'parameters': {
                'prompt': task.prompt,
                'seed': task.seed,
                'steps': task.steps
            },
            'device': str(device)
        }
    
    def submit_inference_task(self, 
                            prompt: str,
                            negative_prompt: str = "",
                            seed: int = -1,
                            sampler_name: str = "Euler a",
                            steps: int = 20,
                            cfg_scale: float = 7.0,
                            width: int = 512,
                            height: int = 512,
                            batch_size: int = 1,
                            priority: int = 0) -> str:
        """Submit an inference task to the cluster"""
        if self.state != ClusterState.READY:
            raise RuntimeError(f"Cluster not ready. Current state: {self.state}")
        
        task = InferenceTask(
            task_id="",  # Will be assigned by scheduler
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed if seed != -1 else int(time.time()),
            sampler_name=sampler_name,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            batch_size=batch_size,
            priority=priority
        )
        
        task_id = self.task_scheduler.add_task(task)
        logger.info(f"Submitted inference task: {task_id}")
        
        return task_id
    
    def get_task_result(self, task_id: str, timeout: float = 60.0) -> Optional[Any]:
        """Get result of a completed task"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.task_scheduler.completed_tasks:
                return self.task_scheduler.completed_tasks[task_id]['result']
            elif task_id in self.task_scheduler.failed_tasks:
                raise RuntimeError(
                    f"Task failed: {self.task_scheduler.failed_tasks[task_id]['error']}"
                )
            time.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} timed out")
    
    def scale_cluster(self, target_workers: int):
        """Scale cluster to target number of workers"""
        self.state = ClusterState.SCALING
        current_workers = len(self.workers)
        
        logger.info(f"Scaling cluster from {current_workers} to {target_workers} workers")
        
        if target_workers > current_workers:
            # Add workers (in production, would spawn new processes/machines)
            for i in range(target_workers - current_workers):
                self._add_worker()
        elif target_workers < current_workers:
            # Remove workers
            workers_to_remove = list(self.workers.keys())[:current_workers - target_workers]
            for worker_id in workers_to_remove:
                self._remove_worker(worker_id)
        
        self.state = ClusterState.READY
    
    def _add_worker(self):
        """Add a new worker to the cluster"""
        # In production, would spawn a new process or connect to remote machine
        worker_id = f"worker_{len(self.workers)}_{int(time.time())}"
        
        # For simulation, create a CPU worker
        worker = WorkerNode(
            worker_id=worker_id,
            host=self.master_host,
            port=self.master_port + len(self.workers),
            device=torch.device('cpu'),
            capabilities={'type': 'simulated'}
        )
        
        self.workers[worker_id] = worker
        logger.info(f"Added worker {worker_id}")
    
    def _remove_worker(self, worker_id: str):
        """Remove a worker from the cluster"""
        if worker_id in self.workers:
            self._handle_worker_failure(worker_id)
            logger.info(f"Removed worker {worker_id}")
    
    def get_cluster_status(self) -> Dict:
        """Get comprehensive cluster status"""
        return {
            'cluster_id': self.cluster_id,
            'state': self.state.value,
            'workers': {wid: w.to_dict() for wid, w in self.workers.items()},
            'queue_stats': self.task_scheduler.get_queue_stats(),
            'metrics': self.metrics,
            'parameter_server': {
                'shards': len(self.parameter_server.shards) if self.parameter_server else 0,
                'gradient_checkpointing': self.parameter_server.gradient_checkpointing if self.parameter_server else False
            }
        }
    
    def optimize_for_model(self, model: nn.Module, 
                          optimization_strategy: str = "balanced"):
        """Optimize cluster configuration for a specific model"""
        if not self.parameter_server:
            self.parameter_server = ParameterServer(model)
        
        analysis = self.parameter_server.analyze_model()
        total_memory_mb = sum(analysis['memory_per_layer'])
        
        logger.info(f"Model analysis: {analysis['total_parameters']} parameters, "
                   f"{total_memory_mb:.2f} MB memory required")
        
        # Calculate optimal number of shards
        available_gpus = len([w for w in self.workers.values() 
                            if 'cuda' in str(w.device)])
        
        if available_gpus > 0:
            # Create shards based on available GPUs
            worker_capabilities = [w.capabilities for w in self.workers.values()]
            shards = self.parameter_server.create_shards(
                min(available_gpus, 4),  # Max 4 shards for now
                worker_capabilities
            )
            
            # Assign shards to workers
            gpu_workers = [w for w in self.workers.values() 
                          if 'cuda' in str(w.device)]
            
            for i, shard in enumerate(shards):
                if i < len(gpu_workers):
                    worker = gpu_workers[i]
                    shard.device = worker.device
                    worker.assign_shard(shard)
                    self.parameter_server.shards[shard.shard_id] = shard
                    
                    logger.info(f"Assigned shard {shard.shard_id} to {worker.worker_id}")
        
        # Enable optimizations based on strategy
        if optimization_strategy == "memory":
            self.parameter_server.enable_gradient_checkpointing()
        elif optimization_strategy == "speed":
            # Would enable other optimizations
            pass
    
    def shutdown(self):
        """Shutdown the cluster gracefully"""
        logger.info("Shutting down cluster...")
        
        self._stop_event.set()
        
        # Wait for threads to finish
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        if self._discovery_thread:
            self._discovery_thread.join(timeout=5)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.state = ClusterState.INITIALIZING
        logger.info("Cluster shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

class DistributedStableDiffusionPipeline:
    """Wrapper for flux that uses distributed inference"""
    def __init__(self, cluster: DistributedInferenceCluster):
        self.cluster = cluster
        self.original_model = None
        
    def hijack_pipeline(self):
        """Hijack the existing pipeline for distributed inference"""
        from modules import processing
        
        # Store original processing function
        self.original_process_images = processing.process_images
        
        # Replace with distributed version
        processing.process_images = self.distributed_process_images
        
        logger.info("Pipeline hijacked for distributed inference")
    
    def restore_pipeline(self):
        """Restore original pipeline"""
        if self.original_model:
            from modules import processing
            processing.process_images = self.original_process_images
            logger.info("Original pipeline restored")
    
    def distributed_process_images(self, p: StableDiffusionProcessing) -> Processed:
        """Distributed version of process_images"""
        logger.info(f"Processing distributed request: {p.prompt}")
        
        # Submit task to cluster
        task_id = self.cluster.submit_inference_task(
            prompt=p.prompt,
            negative_prompt=p.negative_prompt,
            seed=p.seed,
            sampler_name=p.sampler_name,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
            batch_size=p.batch_size
        )
        
        # Wait for result
        try:
            result = self.cluster.get_task_result(task_id, timeout=300)
            
            # Convert to Processed object
            processed = Processed(
                p=p,
                images_list=result.get('images', []),
                seed=result.get('parameters', {}).get('seed', p.seed),
                info=json.dumps(result.get('parameters', {}))
            )
            
            return processed
            
        except Exception as e:
            logger.error(f"Distributed processing failed: {e}")
            # Fall back to local processing
            return self.original_process_images(p)
    
    def generate_8k_image(self, prompt: str, **kwargs) -> List[Any]:
        """Generate ultra-high resolution images using tiled processing"""
        # Calculate tiles for 8K (7680x4320)
        tile_size = 1024
        tiles_x = 8  # 8 * 1024 = 8192 (slightly larger than 8K)
        tiles_y = 5  # 5 * 1024 = 5120
        
        logger.info(f"Generating 8K image with {tiles_x}x{tiles_y} tiles")
        
        # Submit tile generation tasks
        task_ids = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                tile_prompt = f"{prompt}, tile {x}_{y}"
                task_id = self.cluster.submit_inference_task(
                    prompt=tile_prompt,
                    width=tile_size,
                    height=tile_size,
                    priority=10,  # High priority for 8K generation
                    **kwargs
                )
                task_ids.append((x, y, task_id))
        
        # Collect results
        tiles = {}
        for x, y, task_id in task_ids:
            try:
                result = self.cluster.get_task_result(task_id, timeout=600)
                if result and 'images' in result and result['images']:
                    tiles[(x, y)] = result['images'][0]
            except Exception as e:
                logger.error(f"Tile {x},{y} failed: {e}")
        
        # Stitch tiles together
        if len(tiles) == tiles_x * tiles_y:
            return self._stitch_tiles(tiles, tiles_x, tiles_y, tile_size)
        
        return []
    
    def _stitch_tiles(self, tiles: Dict, tiles_x: int, tiles_y: int, 
                     tile_size: int) -> List[Any]:
        """Stitch tiles into final image"""
        from PIL import Image
        
        # Create blank canvas
        final_width = tiles_x * tile_size
        final_height = tiles_y * tile_size
        final_image = Image.new('RGB', (final_width, final_height))
        
        # Paste tiles
        for (x, y), tile in tiles.items():
            if isinstance(tile, Image.Image):
                final_image.paste(tile, (x * tile_size, y * tile_size))
        
        return [final_image]

# Global cluster instance
_cluster_instance: Optional[DistributedInferenceCluster] = None

def get_cluster() -> DistributedInferenceCluster:
    """Get or create global cluster instance"""
    global _cluster_instance
    if _cluster_instance is None:
        _cluster_instance = DistributedInferenceCluster()
    return _cluster_instance

def initialize_distributed_inference(model: nn.Module = None):
    """Initialize distributed inference system"""
    cluster = get_cluster()
    cluster.initialize_cluster(model)
    return cluster

def enable_distributed_for_webui():
    """Enable distributed inference for the webui"""
    cluster = get_cluster()
    pipeline = DistributedStableDiffusionPipeline(cluster)
    pipeline.hijack_pipeline()
    return pipeline

# Integration with existing modules
def patch_sd_hijack():
    """Patch SD hijack for distributed inference"""
    from modules import sd_hijack
    
    original_apply_optimizations = sd_hijack.apply_optimizations
    
    def distributed_apply_optimizations():
        original_apply_optimizations()
        
        # Add distributed optimizations
        if hasattr(shared.opts, 'enable_distributed_inference') and shared.opts.enable_distributed_inference:
            cluster = get_cluster()
            if hasattr(sd_hijack, 'model'):
                cluster.optimize_for_model(sd_hijack.model)
    
    sd_hijack.apply_optimizations = distributed_apply_optimizations

# Add settings
def on_ui_settings():
    """Add distributed inference settings to UI"""
    from modules import shared
    
    section = ('distributed_inference', "Distributed Inference")
    
    shared.opts.add_option("enable_distributed_inference", shared.OptionInfo(
        False, "Enable distributed inference across multiple GPUs",
        section=section
    ))
    
    shared.opts.add_option("distributed_cluster_size", shared.OptionInfo(
        1, "Number of workers in cluster (0 for auto)",
        section=section
    ))
    
    shared.opts.add_option("distributed_optimization_strategy", shared.OptionInfo(
        "balanced", "Optimization strategy",
        gr.Radio, {"choices": ["balanced", "memory", "speed"]},
        section=section
    ))
    
    shared.opts.add_option("distributed_enable_8k", shared.OptionInfo(
        False, "Enable 8K+ image generation",
        section=section
    ))

# Register callbacks
def on_app_started(demo, app):
    """Initialize when app starts"""
    if hasattr(shared.opts, 'enable_distributed_inference') and shared.opts.enable_distributed_inference:
        try:
            initialize_distributed_inference()
            enable_distributed_for_webui()
            logger.info("Distributed inference initialized on app start")
        except Exception as e:
            logger.error(f"Failed to initialize distributed inference: {e}")

# Export main classes and functions
__all__ = [
    'DistributedInferenceCluster',
    'DistributedStableDiffusionPipeline',
    'ParameterServer',
    'TaskScheduler',
    'WorkerNode',
    'get_cluster',
    'initialize_distributed_inference',
    'enable_distributed_for_webui',
    'on_ui_settings',
    'on_app_started'
]