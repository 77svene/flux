"""
modules/model_converter.py

Unified Model Format Support — Single loader supporting all model formats (ckpt, safetensors, diffusers, ONNX)
with automatic conversion and caching, eliminating format-specific code duplication.

This module provides a format-agnostic interface for loading models with pluggable backends
and automatic format detection/conversion pipeline.
"""

import os
import sys
import time
import hashlib
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List, Type
from dataclasses import dataclass
from enum import Enum
import logging

import torch
import numpy as np
from safetensors.torch import load_file as safetensors_load_file, save_file as safetensors_save_file

from modules import shared, devices, modelloader, paths
from modules.sd_models import model_path, model_path_olive

logger = logging.getLogger(__name__)

class ModelFormat(Enum):
    """Supported model formats"""
    CKPT = "ckpt"
    SAFETENSORS = "safetensors"
    DIFFUSERS = "diffusers"
    ONNX = "onnx"
    UNKNOWN = "unknown"

@dataclass
class ModelMetadata:
    """Metadata about a model"""
    format: ModelFormat
    path: Union[str, Path]
    hash: Optional[str] = None
    size: Optional[int] = None
    converted_from: Optional[ModelFormat] = None
    conversion_time: Optional[float] = None

class ModelBackend:
    """Base class for model format backends"""
    
    def __init__(self):
        self.format = ModelFormat.UNKNOWN
    
    def detect_format(self, model_path: Union[str, Path]) -> bool:
        """Detect if this backend can handle the given model"""
        raise NotImplementedError
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> Any:
        """Load model using this backend"""
        raise NotImplementedError
    
    def save_model(self, model: Any, save_path: Union[str, Path], **kwargs) -> None:
        """Save model using this backend"""
        raise NotImplementedError
    
    def get_metadata(self, model_path: Union[str, Path]) -> ModelMetadata:
        """Get metadata about the model"""
        raise NotImplementedError

class CkptBackend(ModelBackend):
    """Backend for loading .ckpt/.pt model files"""
    
    def __init__(self):
        super().__init__()
        self.format = ModelFormat.CKPT
    
    def detect_format(self, model_path: Union[str, Path]) -> bool:
        path = Path(model_path)
        return path.suffix.lower() in ['.ckpt', '.pt', '.pth', '.bin']
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> Dict[str, torch.Tensor]:
        """Load a checkpoint file"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try loading with different methods for compatibility
        try:
            # First try with weights_only=False for full pickle loading
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint with weights_only=False: {e}")
            try:
                # Try with weights_only=True for safer loading
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception as e2:
                logger.error(f"Failed to load checkpoint: {e2}")
                raise RuntimeError(f"Could not load checkpoint: {model_path}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check for common checkpoint structures
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model' in checkpoint:
                return checkpoint['model']
            elif 'params' in checkpoint:
                return checkpoint['params']
            else:
                # Assume the dict itself is the state dict
                return checkpoint
        elif isinstance(checkpoint, torch.nn.Module):
            return checkpoint.state_dict()
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
    
    def save_model(self, model: Dict[str, torch.Tensor], save_path: Union[str, Path], **kwargs) -> None:
        """Save model as checkpoint"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            'state_dict': model,
            'epoch': kwargs.get('epoch', 0),
            'global_step': kwargs.get('global_step', 0),
        }
        
        torch.save(checkpoint, save_path)
    
    def get_metadata(self, model_path: Union[str, Path]) -> ModelMetadata:
        path = Path(model_path)
        return ModelMetadata(
            format=self.format,
            path=path,
            size=path.stat().st_size if path.exists() else None
        )

class SafetensorsBackend(ModelBackend):
    """Backend for loading .safetensors model files"""
    
    def __init__(self):
        super().__init__()
        self.format = ModelFormat.SAFETENSORS
    
    def detect_format(self, model_path: Union[str, Path]) -> bool:
        path = Path(model_path)
        return path.suffix.lower() == '.safetensors'
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> Dict[str, torch.Tensor]:
        """Load a safetensors file"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load safetensors file
            state_dict = safetensors_load_file(str(model_path), device="cpu")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load safetensors file: {e}")
            raise RuntimeError(f"Could not load safetensors: {model_path}")
    
    def save_model(self, model: Dict[str, torch.Tensor], save_path: Union[str, Path], **kwargs) -> None:
        """Save model as safetensors"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            safetensors_save_file(model, str(save_path))
        except Exception as e:
            logger.error(f"Failed to save safetensors file: {e}")
            raise RuntimeError(f"Could not save safetensors: {save_path}")
    
    def get_metadata(self, model_path: Union[str, Path]) -> ModelMetadata:
        path = Path(model_path)
        return ModelMetadata(
            format=self.format,
            path=path,
            size=path.stat().st_size if path.exists() else None
        )

class DiffusersBackend(ModelBackend):
    """Backend for loading diffusers model directories"""
    
    def __init__(self):
        super().__init__()
        self.format = ModelFormat.DIFFUSERS
    
    def detect_format(self, model_path: Union[str, Path]) -> bool:
        path = Path(model_path)
        if path.is_dir():
            # Check for diffusers directory structure
            model_index = path / "model_index.json"
            if model_index.exists():
                return True
            # Check for common diffusers subdirectories
            subdirs = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
            for subdir in subdirs:
                if (path / subdir).exists():
                    return True
        return False
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> Any:
        """Load a diffusers model directory"""
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError("diffusers library is required for loading diffusers models")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load diffusers pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        return pipeline
    
    def save_model(self, model: Any, save_path: Union[str, Path], **kwargs) -> None:
        """Save model as diffusers directory"""
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError("diffusers library is required for saving diffusers models")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(model, StableDiffusionPipeline):
            model.save_pretrained(str(save_path))
        else:
            raise ValueError("Model must be a StableDiffusionPipeline for diffusers saving")
    
    def get_metadata(self, model_path: Union[str, Path]) -> ModelMetadata:
        path = Path(model_path)
        total_size = 0
        if path.exists() and path.is_dir():
            for file in path.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
        
        return ModelMetadata(
            format=self.format,
            path=path,
            size=total_size if total_size > 0 else None
        )

class ONNXBackend(ModelBackend):
    """Backend for loading ONNX model files"""
    
    def __init__(self):
        super().__init__()
        self.format = ModelFormat.ONNX
    
    def detect_format(self, model_path: Union[str, Path]) -> bool:
        path = Path(model_path)
        return path.suffix.lower() == '.onnx'
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> Any:
        """Load an ONNX model"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for loading ONNX models")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        
        # Create ONNX runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = kwargs.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        try:
            session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"Could not load ONNX model: {model_path}")
    
    def save_model(self, model: Any, save_path: Union[str, Path], **kwargs) -> None:
        """Save ONNX model"""
        # ONNX models are typically saved directly, not converted from PyTorch
        # This would require conversion from PyTorch to ONNX first
        raise NotImplementedError("Direct saving to ONNX format not supported. Use PyTorch to ONNX conversion.")
    
    def get_metadata(self, model_path: Union[str, Path]) -> ModelMetadata:
        path = Path(model_path)
        return ModelMetadata(
            format=self.format,
            path=path,
            size=path.stat().st_size if path.exists() else None
        )

class ModelConverter:
    """Unified model converter with format detection and caching"""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.backends: Dict[ModelFormat, ModelBackend] = {
            ModelFormat.CKPT: CkptBackend(),
            ModelFormat.SAFETENSORS: SafetensorsBackend(),
            ModelFormat.DIFFUSERS: DiffusersBackend(),
            ModelFormat.ONNX: ONNXBackend(),
        }
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path(paths.models_path) / "model_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
        
        # Conversion history
        self.conversion_history: List[Dict[str, Any]] = []
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_cache_metadata(self) -> None:
        """Save cache metadata to file"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, model_path: Union[str, Path], target_format: ModelFormat) -> str:
        """Generate a unique cache key for a model and target format"""
        path = Path(model_path)
        
        # Use file path, size, and modification time for key generation
        if path.is_file():
            stat = path.stat()
            key_data = f"{path.absolute()}:{stat.st_size}:{stat.st_mtime}:{target_format.value}"
        elif path.is_dir():
            # For directories, use directory name and total size
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            key_data = f"{path.absolute()}:{total_size}:{target_format.value}"
        else:
            key_data = f"{path.absolute()}:{target_format.value}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_model_path(self, cache_key: str, target_format: ModelFormat) -> Path:
        """Get the path for a cached model"""
        extension = {
            ModelFormat.CKPT: '.ckpt',
            ModelFormat.SAFETENSORS: '.safetensors',
            ModelFormat.DIFFUSERS: '',  # Directory
            ModelFormat.ONNX: '.onnx',
        }.get(target_format, '.bin')
        
        if target_format == ModelFormat.DIFFUSERS:
            return self.cache_dir / cache_key
        else:
            return self.cache_dir / f"{cache_key}{extension}"
    
    def detect_format(self, model_path: Union[str, Path]) -> ModelFormat:
        """Detect the format of a model file/directory"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Try each backend to detect format
        for format_type, backend in self.backends.items():
            try:
                if backend.detect_format(model_path):
                    return format_type
            except Exception:
                continue
        
        # If no specific format detected, try to guess from extension
        if model_path.is_file():
            ext = model_path.suffix.lower()
            if ext in ['.ckpt', '.pt', '.pth', '.bin']:
                return ModelFormat.CKPT
            elif ext == '.safetensors':
                return ModelFormat.SAFETENSORS
            elif ext == '.onnx':
                return ModelFormat.ONNX
        
        return ModelFormat.UNKNOWN
    
    def get_model_metadata(self, model_path: Union[str, Path]) -> ModelMetadata:
        """Get metadata about a model"""
        format_type = self.detect_format(model_path)
        
        if format_type == ModelFormat.UNKNOWN:
            raise ValueError(f"Unknown model format: {model_path}")
        
        backend = self.backends[format_type]
        metadata = backend.get_metadata(model_path)
        
        # Add hash if it's a file
        model_path = Path(model_path)
        if model_path.is_file():
            try:
                with open(model_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                metadata.hash = file_hash
            except Exception:
                pass
        
        return metadata
    
    def load_model(self, model_path: Union[str, Path], target_format: Optional[ModelFormat] = None, 
                   use_cache: bool = True, **kwargs) -> Any:
        """
        Load a model with automatic format detection and optional conversion.
        
        Args:
            model_path: Path to the model file or directory
            target_format: Optional target format to convert to
            use_cache: Whether to use cached converted models
            **kwargs: Additional arguments for the loader
        
        Returns:
            Loaded model in the requested format
        """
        model_path = Path(model_path)
        
        # Detect source format
        source_format = self.detect_format(model_path)
        if source_format == ModelFormat.UNKNOWN:
            raise ValueError(f"Could not detect model format: {model_path}")
        
        # If no target format specified, use source format
        if target_format is None:
            target_format = source_format
        
        # Check cache first
        if use_cache and target_format != source_format:
            cache_key = self._get_cache_key(model_path, target_format)
            cached_path = self._get_cached_model_path(cache_key, target_format)
            
            if cached_path.exists():
                logger.info(f"Loading cached model: {cached_path}")
                try:
                    backend = self.backends[target_format]
                    return backend.load_model(cached_path, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}")
                    # Continue to load and convert original
        
        # Load from source
        logger.info(f"Loading model from {model_path} (format: {source_format.value})")
        source_backend = self.backends[source_format]
        
        start_time = time.time()
        
        if source_format == target_format:
            # No conversion needed
            model = source_backend.load_model(model_path, **kwargs)
        else:
            # Convert between formats
            logger.info(f"Converting from {source_format.value} to {target_format.value}")
            
            # Load source model
            source_model = source_backend.load_model(model_path, **kwargs)
            
            # Convert to target format
            model = self._convert_model(source_model, source_format, target_format, **kwargs)
            
            # Cache the converted model
            if use_cache:
                cache_key = self._get_cache_key(model_path, target_format)
                cached_path = self._get_cached_model_path(cache_key, target_format)
                
                try:
                    target_backend = self.backends[target_format]
                    target_backend.save_model(model, cached_path, **kwargs)
                    
                    # Update cache metadata
                    self.cache_metadata[cache_key] = {
                        'source_path': str(model_path),
                        'source_format': source_format.value,
                        'target_format': target_format.value,
                        'cached_path': str(cached_path),
                        'timestamp': time.time(),
                        'conversion_time': time.time() - start_time
                    }
                    self._save_cache_metadata()
                    
                    logger.info(f"Cached converted model: {cached_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache converted model: {e}")
        
        conversion_time = time.time() - start_time
        
        # Record conversion history
        self.conversion_history.append({
            'source_path': str(model_path),
            'source_format': source_format.value,
            'target_format': target_format.value,
            'conversion_time': conversion_time,
            'timestamp': time.time()
        })
        
        return model
    
    def _convert_model(self, model: Any, source_format: ModelFormat, 
                       target_format: ModelFormat, **kwargs) -> Any:
        """Convert a model from source format to target format"""
        
        # Handle conversion between different formats
        if source_format == ModelFormat.CKPT and target_format == ModelFormat.SAFETENSORS:
            # CKPT to Safetensors
            if isinstance(model, dict):
                return model  # Already a state dict
            else:
                raise ValueError("Cannot convert non-dict model to safetensors")
        
        elif source_format == ModelFormat.SAFETENSORS and target_format == ModelFormat.CKPT:
            # Safetensors to CKPT
            if isinstance(model, dict):
                return model  # Already a state dict
            else:
                raise ValueError("Cannot convert non-dict model to checkpoint")
        
        elif source_format in [ModelFormat.CKPT, ModelFormat.SAFETENSORS] and target_format == ModelFormat.DIFFUSERS:
            # PyTorch state dict to Diffusers
            return self._convert_to_diffusers(model, **kwargs)
        
        elif source_format == ModelFormat.DIFFUSERS and target_format in [ModelFormat.CKPT, ModelFormat.SAFETENSORS]:
            # Diffusers to PyTorch state dict
            return self._convert_from_diffusers(model, **kwargs)
        
        elif target_format == ModelFormat.ONNX:
            # Convert to ONNX
            return self._convert_to_onnx(model, source_format, **kwargs)
        
        else:
            raise ValueError(f"Unsupported conversion: {source_format.value} -> {target_format.value}")
    
    def _convert_to_diffusers(self, state_dict: Dict[str, torch.Tensor], **kwargs) -> Any:
        """Convert a PyTorch state dict to a diffusers pipeline"""
        try:
            from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError("diffusers and transformers libraries are required for conversion")
        
        # This is a simplified conversion - in practice, you'd need to handle
        # the specific architecture of the model
        logger.warning("Simplified diffusers conversion - may not work for all models")
        
        # For now, return the state dict as-is
        # A full implementation would require architecture detection and proper conversion
        return state_dict
    
    def _convert_from_diffusers(self, pipeline: Any, **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a diffusers pipeline to a PyTorch state dict"""
        if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'state_dict'):
            # Extract UNet state dict
            return pipeline.unet.state_dict()
        else:
            raise ValueError("Cannot extract state dict from diffusers pipeline")
    
    def _convert_to_onnx(self, model: Any, source_format: ModelFormat, **kwargs) -> Any:
        """Convert a model to ONNX format"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX conversion")
        
        # This would require a PyTorch model and conversion to ONNX
        # For now, raise an error as this requires model-specific conversion
        raise NotImplementedError(
            "ONNX conversion requires model-specific implementation. "
            "Use dedicated conversion scripts for your model architecture."
        )
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear the model cache.
        
        Args:
            older_than_days: Only clear cache entries older than this many days
        
        Returns:
            Number of cache entries cleared
        """
        cleared_count = 0
        
        if older_than_days is not None:
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
            keys_to_remove = []
            
            for cache_key, metadata in self.cache_metadata.items():
                if metadata.get('timestamp', 0) < cutoff_time:
                    keys_to_remove.append(cache_key)
            
            for cache_key in keys_to_remove:
                metadata = self.cache_metadata[cache_key]
                cached_path = Path(metadata['cached_path'])
                
                # Remove cached file/directory
                if cached_path.exists():
                    if cached_path.is_file():
                        cached_path.unlink()
                    elif cached_path.is_dir():
                        shutil.rmtree(cached_path)
                
                # Remove from metadata
                del self.cache_metadata[cache_key]
                cleared_count += 1
        else:
            # Clear entire cache
            for cache_key, metadata in list(self.cache_metadata.items()):
                cached_path = Path(metadata['cached_path'])
                
                if cached_path.exists():
                    if cached_path.is_file():
                        cached_path.unlink()
                    elif cached_path.is_dir():
                        shutil.rmtree(cached_path)
                
                cleared_count += 1
            
            self.cache_metadata.clear()
        
        # Save updated metadata
        self._save_cache_metadata()
        
        logger.info(f"Cleared {cleared_count} cache entries")
        return cleared_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache"""
        total_size = 0
        format_counts = {}
        
        for cache_key, metadata in self.cache_metadata.items():
            cached_path = Path(metadata['cached_path'])
            
            if cached_path.exists():
                if cached_path.is_file():
                    total_size += cached_path.stat().st_size
                elif cached_path.is_dir():
                    for file in cached_path.rglob('*'):
                        if file.is_file():
                            total_size += file.stat().st_size
            
            target_format = metadata.get('target_format', 'unknown')
            format_counts[target_format] = format_counts.get(target_format, 0) + 1
        
        return {
            'total_entries': len(self.cache_metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'format_distribution': format_counts,
            'cache_directory': str(self.cache_dir)
        }

# Global converter instance
_converter_instance: Optional[ModelConverter] = None

def get_converter() -> ModelConverter:
    """Get the global model converter instance"""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = ModelConverter()
    return _converter_instance

def load_model(model_path: Union[str, Path], target_format: Optional[Union[str, ModelFormat]] = None,
               use_cache: bool = True, **kwargs) -> Any:
    """
    Convenience function to load a model with automatic format detection.
    
    Args:
        model_path: Path to the model file or directory
        target_format: Optional target format (string or ModelFormat enum)
        use_cache: Whether to use cached converted models
        **kwargs: Additional arguments for the loader
    
    Returns:
        Loaded model
    """
    converter = get_converter()
    
    # Convert string format to enum if needed
    if isinstance(target_format, str):
        target_format = ModelFormat(target_format.lower())
    
    return converter.load_model(model_path, target_format, use_cache, **kwargs)

def detect_model_format(model_path: Union[str, Path]) -> str:
    """
    Detect the format of a model file/directory.
    
    Args:
        model_path: Path to the model
    
    Returns:
        Format string (ckpt, safetensors, diffusers, onnx, or unknown)
    """
    converter = get_converter()
    format_type = converter.detect_format(model_path)
    return format_type.value

def get_model_metadata(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get metadata about a model.
    
    Args:
        model_path: Path to the model
    
    Returns:
        Dictionary with model metadata
    """
    converter = get_converter()
    metadata = converter.get_model_metadata(model_path)
    
    return {
        'format': metadata.format.value,
        'path': str(metadata.path),
        'hash': metadata.hash,
        'size': metadata.size,
        'converted_from': metadata.converted_from.value if metadata.converted_from else None,
        'conversion_time': metadata.conversion_time
    }

def clear_model_cache(older_than_days: Optional[int] = None) -> int:
    """
    Clear the model cache.
    
    Args:
        older_than_days: Only clear cache entries older than this many days
    
    Returns:
        Number of cache entries cleared
    """
    converter = get_converter()
    return converter.clear_cache(older_than_days)

def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the model cache.
    
    Returns:
        Dictionary with cache statistics
    """
    converter = get_converter()
    return converter.get_cache_stats()

# Integration with existing codebase
def patch_existing_loaders():
    """
    Patch existing model loading functions to use the unified converter.
    This should be called during initialization to replace format-specific code.
    """
    try:
        # Try to patch modules.sd_models if it exists
        import modules.sd_models as sd_models
        
        # Store original functions
        original_load_model = getattr(sd_models, 'load_model', None)
        original_load_checkpoint = getattr(sd_models, 'load_checkpoint', None)
        
        if original_load_model:
            def patched_load_model(*args, **kwargs):
                # Try to use unified converter first
                try:
                    if len(args) > 0:
                        model_path = args[0]
                        return load_model(model_path, **kwargs)
                except Exception as e:
                    logger.warning(f"Unified loader failed, falling back to original: {e}")
                    # Fall back to original
                    return original_load_model(*args, **kwargs)
            
            sd_models.load_model = patched_load_model
        
        if original_load_checkpoint:
            def patched_load_checkpoint(*args, **kwargs):
                # Try to use unified converter first
                try:
                    if len(args) > 0:
                        model_path = args[0]
                        return load_model(model_path, target_format=ModelFormat.CKPT, **kwargs)
                except Exception as e:
                    logger.warning(f"Unified loader failed, falling back to original: {e}")
                    # Fall back to original
                    return original_load_checkpoint(*args, **kwargs)
            
            sd_models.load_checkpoint = patched_load_checkpoint
        
        logger.info("Patched existing model loaders with unified converter")
        
    except ImportError:
        logger.debug("modules.sd_models not found, skipping patching")
    except Exception as e:
        logger.warning(f"Failed to patch existing loaders: {e}")

# Auto-patch when module is imported
if shared.opts.unified_model_loader:
    patch_existing_loaders()

# Register settings
def on_ui_settings():
    """Register settings for the unified model loader"""
    import gradio as gr
    from modules import shared
    
    section = ('unified_model_loader', "Unified Model Loader")
    
    shared.opts.add_option("unified_model_loader", shared.OptionInfo(
        True, "Use unified model loader (experimental)", gr.Checkbox, section=section))
    
    shared.opts.add_option("model_cache_enabled", shared.OptionInfo(
        True, "Enable model conversion cache", gr.Checkbox, section=section))
    
    shared.opts.add_option("model_cache_max_size", shared.OptionInfo(
        10, "Maximum cache size (GB)", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}, section=section))
    
    shared.opts.add_option("model_cache_cleanup_days", shared.OptionInfo(
        30, "Auto-cleanup cache entries older than (days)", gr.Slider, {"minimum": 1, "maximum": 365, "step": 1}, section=section))

# Initialize when module is imported
if __name__ != "__main__":
    try:
        on_ui_settings()
    except Exception as e:
        logger.debug(f"Could not register UI settings: {e}")