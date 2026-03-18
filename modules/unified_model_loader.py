"""
Unified Model Format Support
Single loader supporting all model formats (ckpt, safetensors, diffusers, ONNX)
with automatic conversion and caching, eliminating format-specific code duplication.
"""

import os
import gc
import sys
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import numpy as np
from safetensors.torch import load_file, save_file
import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Import existing modules
from modules import shared, devices, modelloader, errors
from modules.paths import models_path, data_path
from modules.sd_models import CheckpointInfo, list_models, checkpoints_list, checkpoint_tiles
from modules.sd_vae import vae_dict, vae_list
from modules.shared import opts, cmd_opts

# Try to import optional dependencies
try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoencoderKL
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global cache for converted models
_converted_models_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()

@dataclass
class ModelMetadata:
    """Metadata about a model file"""
    path: str
    format: str
    architecture: Optional[str] = None
    is_inpainting: bool = False
    is_sd2: bool = False
    is_sd2_depth: bool = False
    is_sd21: bool = False
    is_sdxl: bool = False
    is_sdxl_refiner: bool = False
    vae_path: Optional[str] = None
    config_path: Optional[str] = None
    hash: Optional[str] = None
    file_size: int = 0

class ModelBackend(ABC):
    """Abstract base class for model format backends"""
    
    @abstractmethod
    def can_load(self, path: str) -> bool:
        """Check if this backend can load the given model file"""
        pass
    
    @abstractmethod
    def load_model(self, path: str, metadata: ModelMetadata, device: torch.device) -> Any:
        """Load model from path"""
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """Get format name"""
        pass
    
    @abstractmethod
    def supports_conversion_to(self, target_format: str) -> bool:
        """Check if conversion to target format is supported"""
        pass
    
    @abstractmethod
    def convert_model(self, model: Any, target_format: str, save_path: str) -> bool:
        """Convert model to target format"""
        pass

class CheckpointBackend(ModelBackend):
    """Backend for loading .ckpt/.pt checkpoint files"""
    
    def can_load(self, path: str) -> bool:
        return path.endswith(('.ckpt', '.pt', '.pth', '.bin'))
    
    def get_format_name(self) -> str:
        return "checkpoint"
    
    def supports_conversion_to(self, target_format: str) -> bool:
        return target_format in ["safetensors", "diffusers"]
    
    def load_model(self, path: str, metadata: ModelMetadata, device: torch.device) -> Dict[str, torch.Tensor]:
        """Load checkpoint file"""
        logger.info(f"Loading checkpoint from {path}")
        
        # Check for cached converted version
        cache_key = f"ckpt_{metadata.hash}"
        with _cache_lock:
            if cache_key in _converted_models_cache:
                logger.info(f"Using cached checkpoint for {path}")
                return _converted_models_cache[cache_key]
        
        # Load checkpoint
        pl_sd = torch.load(path, map_location="cpu")
        
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
        
        # Cache the loaded state dict
        with _cache_lock:
            _converted_models_cache[cache_key] = sd
        
        return sd
    
    def convert_model(self, model: Dict[str, torch.Tensor], target_format: str, save_path: str) -> bool:
        """Convert checkpoint to another format"""
        if target_format == "safetensors":
            try:
                save_file(model, save_path)
                return True
            except Exception as e:
                logger.error(f"Failed to convert checkpoint to safetensors: {e}")
                return False
        elif target_format == "diffusers":
            if not DIFFUSERS_AVAILABLE:
                logger.error("Diffusers library not available for conversion")
                return False
            try:
                # Convert to diffusers format
                pipeline = load_pipeline_from_original_stable_diffusion_ckpt(
                    checkpoint_path_or_dict=model,
                    original_config_file=None,
                    image_size=512,
                    prediction_type="epsilon",
                    model_type=None,
                    extract_ema=False,
                    scheduler_type="pndm",
                    num_in_channels=None,
                    upcast_attention=None,
                    load_safety_checker=True
                )
                pipeline.save_pretrained(save_path)
                return True
            except Exception as e:
                logger.error(f"Failed to convert checkpoint to diffusers: {e}")
                return False
        return False

class SafetensorsBackend(ModelBackend):
    """Backend for loading .safetensors files"""
    
    def can_load(self, path: str) -> bool:
        return path.endswith('.safetensors')
    
    def get_format_name(self) -> str:
        return "safetensors"
    
    def supports_conversion_to(self, target_format: str) -> bool:
        return target_format in ["checkpoint", "diffusers"]
    
    def load_model(self, path: str, metadata: ModelMetadata, device: torch.device) -> Dict[str, torch.Tensor]:
        """Load safetensors file"""
        logger.info(f"Loading safetensors from {path}")
        
        # Check for cached converted version
        cache_key = f"st_{metadata.hash}"
        with _cache_lock:
            if cache_key in _converted_models_cache:
                logger.info(f"Using cached safetensors for {path}")
                return _converted_models_cache[cache_key]
        
        # Load safetensors
        sd = load_file(path, device="cpu")
        
        # Cache the loaded state dict
        with _cache_lock:
            _converted_models_cache[cache_key] = sd
        
        return sd
    
    def convert_model(self, model: Dict[str, torch.Tensor], target_format: str, save_path: str) -> bool:
        """Convert safetensors to another format"""
        if target_format == "checkpoint":
            try:
                torch.save(model, save_path)
                return True
            except Exception as e:
                logger.error(f"Failed to convert safetensors to checkpoint: {e}")
                return False
        elif target_format == "diffusers":
            if not DIFFUSERS_AVAILABLE:
                logger.error("Diffusers library not available for conversion")
                return False
            try:
                # Convert to diffusers format
                pipeline = load_pipeline_from_original_stable_diffusion_ckpt(
                    checkpoint_path_or_dict=model,
                    original_config_file=None,
                    image_size=512,
                    prediction_type="epsilon",
                    model_type=None,
                    extract_ema=False,
                    scheduler_type="pndm",
                    num_in_channels=None,
                    upcast_attention=None,
                    load_safety_checker=True
                )
                pipeline.save_pretrained(save_path)
                return True
            except Exception as e:
                logger.error(f"Failed to convert safetensors to diffusers: {e}")
                return False
        return False

class DiffusersBackend(ModelBackend):
    """Backend for loading diffusers models"""
    
    def can_load(self, path: str) -> bool:
        if not DIFFUSERS_AVAILABLE:
            return False
        
        # Check if it's a diffusers directory
        path_obj = Path(path)
        if path_obj.is_dir():
            # Check for model_index.json which indicates a diffusers model
            model_index = path_obj / "model_index.json"
            if model_index.exists():
                return True
            
            # Check for common diffusers subdirectories
            diffusers_dirs = {"unet", "vae", "text_encoder", "tokenizer", "scheduler"}
            existing_dirs = {d.name for d in path_obj.iterdir() if d.is_dir()}
            if diffusers_dirs.intersection(existing_dirs):
                return True
        
        return False
    
    def get_format_name(self) -> str:
        return "diffusers"
    
    def supports_conversion_to(self, target_format: str) -> bool:
        return target_format in ["checkpoint", "safetensors"]
    
    def load_model(self, path: str, metadata: ModelMetadata, device: torch.device) -> Any:
        """Load diffusers model"""
        logger.info(f"Loading diffusers model from {path}")
        
        # Check for cached version
        cache_key = f"diffusers_{metadata.hash}"
        with _cache_lock:
            if cache_key in _converted_models_cache:
                logger.info(f"Using cached diffusers model for {path}")
                return _converted_models_cache[cache_key]
        
        try:
            # Load diffusers pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                path,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Cache the pipeline
            with _cache_lock:
                _converted_models_cache[cache_key] = pipeline
            
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load diffusers model: {e}")
            raise
    
    def convert_model(self, model: Any, target_format: str, save_path: str) -> bool:
        """Convert diffusers model to another format"""
        if not DIFFUSERS_AVAILABLE:
            return False
        
        try:
            if target_format in ["checkpoint", "safetensors"]:
                # Extract state dict from diffusers pipeline
                if hasattr(model, 'unet'):
                    # For StableDiffusionPipeline
                    state_dict = {}
                    
                    # UNet
                    unet_state = model.unet.state_dict()
                    for k, v in unet_state.items():
                        state_dict[f"model.diffusion_model.{k}"] = v
                    
                    # Text encoder
                    text_encoder_state = model.text_encoder.state_dict()
                    for k, v in text_encoder_state.items():
                        state_dict[f"cond_stage_model.transformer.{k}"] = v
                    
                    # VAE
                    vae_state = model.vae.state_dict()
                    for k, v in vae_state.items():
                        state_dict[f"first_stage_model.{k}"] = v
                    
                    if target_format == "checkpoint":
                        torch.save({"state_dict": state_dict}, save_path)
                    else:  # safetensors
                        save_file(state_dict, save_path)
                    
                    return True
        except Exception as e:
            logger.error(f"Failed to convert diffusers model: {e}")
        
        return False

class ONNXBackend(ModelBackend):
    """Backend for loading ONNX models"""
    
    def can_load(self, path: str) -> bool:
        if not ONNX_AVAILABLE:
            return False
        return path.endswith('.onnx')
    
    def get_format_name(self) -> str:
        return "onnx"
    
    def supports_conversion_to(self, target_format: str) -> bool:
        # ONNX models are typically used for inference only
        return False
    
    def load_model(self, path: str, metadata: ModelMetadata, device: torch.device) -> Any:
        """Load ONNX model"""
        logger.info(f"Loading ONNX model from {path}")
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX runtime not available")
        
        # Check for cached session
        cache_key = f"onnx_{metadata.hash}"
        with _cache_lock:
            if cache_key in _converted_models_cache:
                logger.info(f"Using cached ONNX session for {path}")
                return _converted_models_cache[cache_key]
        
        try:
            # Create ONNX runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
            session = ort.InferenceSession(path, providers=providers)
            
            # Cache the session
            with _cache_lock:
                _converted_models_cache[cache_key] = session
            
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def convert_model(self, model: Any, target_format: str, save_path: str) -> bool:
        # ONNX conversion not typically supported
        return False

class UnifiedModelLoader:
    """
    Unified model loader that supports multiple formats with automatic detection,
    conversion, and caching.
    """
    
    def __init__(self):
        self.backends: List[ModelBackend] = []
        self._register_default_backends()
        self._model_cache_dir = Path(data_path) / "model_cache"
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Register with existing model list
        self._patch_existing_model_loading()
    
    def _register_default_backends(self):
        """Register default format backends"""
        self.register_backend(CheckpointBackend())
        self.register_backend(SafetensorsBackend())
        self.register_backend(DiffusersBackend())
        self.register_backend(ONNXBackend())
    
    def register_backend(self, backend: ModelBackend):
        """Register a new model format backend"""
        self.backends.append(backend)
        logger.info(f"Registered model backend: {backend.get_format_name()}")
    
    def detect_format(self, path: str) -> Optional[str]:
        """Detect model format from file path"""
        path_obj = Path(path)
        
        # Check each backend
        for backend in self.backends:
            if backend.can_load(path):
                return backend.get_format_name()
        
        # Fallback to extension-based detection
        suffix = path_obj.suffix.lower()
        format_map = {
            '.ckpt': 'checkpoint',
            '.pt': 'checkpoint',
            '.pth': 'checkpoint',
            '.bin': 'checkpoint',
            '.safetensors': 'safetensors',
            '.onnx': 'onnx',
        }
        
        if suffix in format_map:
            return format_map[suffix]
        
        # Check if directory (could be diffusers)
        if path_obj.is_dir():
            return 'diffusers'
        
        return None
    
    def get_backend_for_format(self, format_name: str) -> Optional[ModelBackend]:
        """Get backend for specific format"""
        for backend in self.backends:
            if backend.get_format_name() == format_name:
                return backend
        return None
    
    def calculate_model_hash(self, path: str) -> str:
        """Calculate hash for model file or directory"""
        if os.path.isfile(path):
            # For files, hash the first 1MB and file size
            hasher = hashlib.sha256()
            with open(path, 'rb') as f:
                chunk = f.read(1024 * 1024)  # Read first 1MB
                hasher.update(chunk)
            hasher.update(str(os.path.getsize(path)).encode())
            return hasher.hexdigest()[:16]
        else:
            # For directories, hash directory structure
            hasher = hashlib.sha256()
            path_obj = Path(path)
            for file_path in sorted(path_obj.rglob("*")):
                if file_path.is_file():
                    hasher.update(str(file_path.relative_to(path_obj)).encode())
                    hasher.update(str(file_path.stat().st_mtime).encode())
            return hasher.hexdigest()[:16]
    
    def get_model_metadata(self, path: str) -> ModelMetadata:
        """Get metadata about a model file"""
        path_obj = Path(path)
        format_name = self.detect_format(path)
        model_hash = self.calculate_model_hash(path)
        
        # Try to detect architecture from filename
        filename = path_obj.name.lower()
        is_inpainting = "inpainting" in filename
        is_sd2 = "sd2" in filename or "v2" in filename
        is_sd2_depth = "depth" in filename
        is_sd21 = "sd2.1" in filename or "v2.1" in filename
        is_sdxl = "sdxl" in filename or "xl" in filename
        is_sdxl_refiner = "refiner" in filename
        
        # Get file size
        if path_obj.is_file():
            file_size = path_obj.stat().st_size
        else:
            file_size = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
        
        # Look for config file
        config_path = None
        possible_configs = [
            path_obj.with_suffix('.yaml'),
            path_obj.with_suffix('.yml'),
            path_obj.parent / f"{path_obj.stem}.yaml",
        ]
        for config in possible_configs:
            if config.exists():
                config_path = str(config)
                break
        
        return ModelMetadata(
            path=path,
            format=format_name or "unknown",
            is_inpainting=is_inpainting,
            is_sd2=is_sd2,
            is_sd2_depth=is_sd2_depth,
            is_sd21=is_sd21,
            is_sdxl=is_sdxl,
            is_sdxl_refiner=is_sdxl_refiner,
            config_path=config_path,
            hash=model_hash,
            file_size=file_size
        )
    
    def load_model(
        self,
        path: str,
        device: Optional[torch.device] = None,
        target_format: Optional[str] = None,
        convert_if_needed: bool = True
    ) -> Tuple[Any, ModelMetadata]:
        """
        Load model from path with automatic format detection and conversion.
        
        Args:
            path: Path to model file or directory
            device: Target device for model
            target_format: If specified, convert model to this format
            convert_if_needed: Whether to convert if format doesn't match target
            
        Returns:
            Tuple of (model, metadata)
        """
        if device is None:
            device = devices.device
        
        # Get metadata
        metadata = self.get_model_metadata(path)
        logger.info(f"Loading model: {path} (format: {metadata.format})")
        
        # Get appropriate backend
        backend = self.get_backend_for_format(metadata.format)
        if backend is None:
            raise ValueError(f"No backend available for format: {metadata.format}")
        
        # Load model using backend
        try:
            model = backend.load_model(path, metadata, device)
        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}")
            raise
        
        # Convert if needed
        if target_format and target_format != metadata.format:
            if not convert_if_needed:
                raise ValueError(
                    f"Model format is {metadata.format}, but {target_format} was requested. "
                    "Set convert_if_needed=True to enable conversion."
                )
            
            if not backend.supports_conversion_to(target_format):
                raise ValueError(
                    f"Cannot convert from {metadata.format} to {target_format}"
                )
            
            # Generate cache path for converted model
            cache_filename = f"{metadata.hash}_{target_format}"
            if target_format == "diffusers":
                cache_path = self._model_cache_dir / cache_filename
                cache_path.mkdir(parents=True, exist_ok=True)
            else:
                cache_path = self._model_cache_dir / f"{cache_filename}.{target_format}"
            
            # Convert if not already cached
            if not cache_path.exists():
                logger.info(f"Converting model from {metadata.format} to {target_format}")
                success = backend.convert_model(model, target_format, str(cache_path))
                if not success:
                    raise RuntimeError(f"Failed to convert model to {target_format}")
            
            # Load the converted model
            target_backend = self.get_backend_for_format(target_format)
            if target_backend:
                model = target_backend.load_model(str(cache_path), metadata, device)
        
        # Move model to device if needed
        if hasattr(model, 'to') and device is not None:
            model = model.to(device)
        
        return model, metadata
    
    def load_model_as_state_dict(
        self,
        path: str,
        convert_to_sd: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Load model and return as state dict (for compatibility with existing code).
        This is the main integration point with existing flux code.
        """
        metadata = self.get_model_metadata(path)
        
        # If already a checkpoint/safetensors, load directly
        if metadata.format in ["checkpoint", "safetensors"]:
            backend = self.get_backend_for_format(metadata.format)
            if backend:
                return backend.load_model(path, metadata, devices.device)
        
        # For diffusers, convert to state dict
        if metadata.format == "diffusers" and convert_to_sd:
            model, _ = self.load_model(path, target_format="safetensors")
            if isinstance(model, dict):
                return model
        
        # For ONNX or other formats, try to extract state dict
        # This is a simplified implementation
        model, _ = self.load_model(path)
        if hasattr(model, 'state_dict'):
            return model.state_dict()
        elif isinstance(model, dict):
            return model
        
        raise ValueError(f"Cannot convert model at {path} to state dict")
    
    def preload_model(self, path: str, callback=None):
        """Preload model in background thread"""
        def _preload():
            try:
                self.load_model(path)
                if callback:
                    callback(True, None)
            except Exception as e:
                logger.error(f"Preload failed for {path}: {e}")
                if callback:
                    callback(False, e)
        
        self._executor.submit(_preload)
    
    def clear_cache(self):
        """Clear the model cache"""
        with _cache_lock:
            _converted_models_cache.clear()
        
        # Clear disk cache
        if self._model_cache_dir.exists():
            shutil.rmtree(self._model_cache_dir)
            self._model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        with _cache_lock:
            memory_cache_count = len(_converted_models_cache)
        
        disk_cache_count = 0
        disk_cache_size = 0
        if self._model_cache_dir.exists():
            for item in self._model_cache_dir.rglob("*"):
                if item.is_file():
                    disk_cache_count += 1
                    disk_cache_size += item.stat().st_size
        
        return {
            "memory_cache_count": memory_cache_count,
            "disk_cache_count": disk_cache_count,
            "disk_cache_size_mb": disk_cache_size / (1024 * 1024),
            "cache_dir": str(self._model_cache_dir)
        }
    
    def _patch_existing_model_loading(self):
        """Patch existing model loading functions to use unified loader"""
        # Store original functions
        self._original_load_model_weights = None
        self._original_load_checkpoint = None
        
        # Try to patch sd_models module
        try:
            import modules.sd_models as sd_models
            
            # Store originals
            self._original_load_model_weights = getattr(sd_models, 'load_model_weights', None)
            self._original_load_checkpoint = getattr(sd_models, 'load_checkpoint', None)
            
            # Create wrapper for load_model_weights
            def unified_load_model_weights(model, checkpoint_info: CheckpointInfo, *args, **kwargs):
                if checkpoint_info and checkpoint_info.filename:
                    try:
                        # Use unified loader
                        state_dict = self.load_model_as_state_dict(checkpoint_info.filename)
                        
                        # Load state dict into model
                        model.load_state_dict(state_dict, strict=False)
                        
                        # Set model metadata
                        if hasattr(model, 'sd_model_hash'):
                            model.sd_model_hash = checkpoint_info.hash
                        if hasattr(model, 'sd_model_checkpoint'):
                            model.sd_model_checkpoint = checkpoint_info.filename
                        
                        return
                    except Exception as e:
                        logger.warning(f"Unified loader failed, falling back to original: {e}")
                
                # Fallback to original function
                if self._original_load_model_weights:
                    return self._original_load_model_weights(model, checkpoint_info, *args, **kwargs)
                else:
                    raise RuntimeError("Original load_model_weights not available")
            
            # Create wrapper for load_checkpoint
            def unified_load_checkpoint(model, path, *args, **kwargs):
                if path:
                    try:
                        # Use unified loader
                        state_dict = self.load_model_as_state_dict(path)
                        
                        # Load state dict into model
                        model.load_state_dict(state_dict, strict=False)
                        return
                    except Exception as e:
                        logger.warning(f"Unified loader failed, falling back to original: {e}")
                
                # Fallback to original function
                if self._original_load_checkpoint:
                    return self._original_load_checkpoint(model, path, *args, **kwargs)
                else:
                    raise RuntimeError("Original load_checkpoint not available")
            
            # Apply patches
            sd_models.load_model_weights = unified_load_model_weights
            sd_models.load_checkpoint = unified_load_checkpoint
            
            logger.info("Patched sd_models module with unified loader")
            
        except ImportError:
            logger.warning("Could not patch sd_models module")
        
        # Also patch modelloader module
        try:
            import modules.modelloader as modelloader
            
            # Store original
            self._original_load_models = getattr(modelloader, 'load_models', None)
            
            def unified_load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None) -> str:
                # Use unified loader for actual loading
                if model_path and os.path.exists(model_path):
                    try:
                        # Just validate we can load it
                        self.detect_format(model_path)
                    except:
                        pass
                
                # Call original function
                if self._original_load_models:
                    return self._original_load_models(
                        model_path, model_url, command_path, ext_filter, download_name, ext_blacklist
                    )
                return model_path
            
            # Apply patch
            modelloader.load_models = unified_load_models
            
            logger.info("Patched modelloader module with unified loader")
            
        except ImportError:
            logger.warning("Could not patch modelloader module")

# Global instance
unified_loader = UnifiedModelLoader()

# Convenience functions for external use
def load_model(path: str, **kwargs) -> Tuple[Any, ModelMetadata]:
    """Load model using unified loader"""
    return unified_loader.load_model(path, **kwargs)

def load_model_as_state_dict(path: str, **kwargs) -> Dict[str, torch.Tensor]:
    """Load model as state dict using unified loader"""
    return unified_loader.load_model_as_state_dict(path, **kwargs)

def detect_model_format(path: str) -> Optional[str]:
    """Detect model format"""
    return unified_loader.detect_format(path)

def get_model_metadata(path: str) -> ModelMetadata:
    """Get model metadata"""
    return unified_loader.get_model_metadata(path)

def clear_model_cache():
    """Clear model cache"""
    unified_loader.clear_cache()

def get_cache_info() -> Dict[str, Any]:
    """Get cache information"""
    return unified_loader.get_cache_info()

# Integration with existing model listing
def list_models_with_unified_loader():
    """List models using unified loader for format detection"""
    from modules.sd_models import model_path as sd_model_path
    
    model_list = []
    model_path = Path(sd_model_path)
    
    if not model_path.exists():
        return model_list
    
    # Scan for model files
    for ext in ['.ckpt', '.safetensors', '.pt', '.pth', '.bin', '.onnx']:
        for file_path in model_path.rglob(f"*{ext}"):
            try:
                metadata = get_model_metadata(str(file_path))
                model_list.append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'format': metadata.format,
                    'size_mb': metadata.file_size / (1024 * 1024),
                    'hash': metadata.hash,
                    'metadata': metadata
                })
            except Exception as e:
                logger.warning(f"Failed to get metadata for {file_path}: {e}")
    
    # Scan for diffusers directories
    for dir_path in model_path.iterdir():
        if dir_path.is_dir():
            try:
                if detect_model_format(str(dir_path)) == 'diffusers':
                    metadata = get_model_metadata(str(dir_path))
                    model_list.append({
                        'path': str(dir_path),
                        'name': dir_path.name,
                        'format': 'diffusers',
                        'size_mb': metadata.file_size / (1024 * 1024),
                        'hash': metadata.hash,
                        'metadata': metadata
                    })
            except Exception as e:
                logger.warning(f"Failed to get metadata for {dir_path}: {e}")
    
    return model_list

# Auto-initialization when module is imported
def initialize():
    """Initialize unified loader and integrate with webui"""
    logger.info("Initializing Unified Model Loader")
    
    # Add cache info to shared state for UI display
    if hasattr(shared, 'state'):
        shared.state.cache_info = get_cache_info
    
    # Register model scanning function
    try:
        from modules import sd_models
        sd_models.list_models = list_models_with_unified_loader
    except:
        pass

# Initialize on import
initialize()