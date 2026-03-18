import os
import logging
from typing import Optional, Dict, List, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedModel, AutoProcessor, AutoConfig
from accelerate import Accelerator
import torch.nn.init as init
from einops import rearrange

logger = logging.getLogger(__name__)

@dataclass
class VisionConfig:
    """Configuration for vision encoder processing."""
    image_size: int = 336
    patch_size: int = 14
    num_crops: int = 1
    crop_size: Tuple[int, int] = field(default_factory=lambda: (336, 336))
    aspect_ratio: Optional[float] = None
    min_resolution: int = 256
    max_resolution: int = 1024
    processor_name: str = "default"

@dataclass
class AlignmentConfig:
    """Configuration for vision-language alignment."""
    projection_dim: int = 4096
    hidden_dim: int = 1024
    loss_type: str = "cross_entropy"
    temperature: float = 0.07
    dropout: float = 0.1
    freeze_vision_encoder: bool = True

class AbstractVisionEncoder(ABC):
    """Abstract base class for vision encoder integration."""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self._model = None
    
    @abstractmethod
    def load_model(self, path: str, device: str = "cpu") -> None:
        """Load the specific vision encoder model."""
        pass
    
    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features from input images."""
        pass
    
    @abstractmethod
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get image features with optional pooling."""
        pass
    
    def set_model(self, model: PreTrainedModel) -> None:
        """Set the underlying model instance."""
        self._model = model

class QwenVisionEncoder(AbstractVisionEncoder):
    """Wrapper for Qwen-VL style vision encoders."""
    
    def __init__(self, config: VisionConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
    
    def load_model(self, path: str, device: str = "cpu") -> None:
        """Load Qwen-VL vision encoder."""
        self._model = AutoProcessor.from_pretrained(path, trust_remote_code=True).vision_processor
        self._model = self._model.vision_tower
        self._model = self._model.to(device)
        self._model.eval()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Process images through Qwen vision tower."""
        if images.dim() == 4:  # B, C, H, W
            images = rearrange(images, "b c h w -> b h w c")
        with torch.no_grad():
            features = self._model(images)
        return features
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features and apply pooling."""
        features = self.forward(images)
        # Qwen often returns list of tensors or single tensor
        if isinstance(features, list):
            features = torch.cat(features, dim=-1)
        return features

class LlavaVisionEncoder(AbstractVisionEncoder):
    """Wrapper for LLaVA style vision encoders (CLIP/ViT)."""
    
    def __init__(self, config: VisionConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
    
    def load_model(self, path: str, device: str = "cpu") -> None:
        """Load CLIP or ViT vision encoder."""
        from transformers import CLIPVisionModel, ViTModel
        try:
            self._model = CLIPVisionModel.from_pretrained(path)
        except Exception:
            self._model = ViTModel.from_pretrained(path)
        self._model = self._model.to(device)
        self._model.eval()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Process images through CLIP/ViT."""
        with torch.no_grad():
            features = self._model(images)
        return features.last_hidden_state
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get pooled image features."""
        features = self.forward(images)
        # Average pooling over sequence dimension
        features = features[:, 0, :] if features.shape[1] > 1 else features[:, 0, :]
        return features

class DynamicImagePatcher:
    """Handles dynamic image patching and resizing."""
    
    def __init__(self, config: VisionConfig):
        self.config = config
    
    def process_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Process a batch of images to a uniform size with dynamic patching.
        Supports aspect ratio preservation and padding.
        """
        if not images:
            return torch.empty((0, 3, self.config.image_size, self.config.image_size))
        
        # Determine max resolution from images
        max_h = max(img.height for img in images)
        max_w = max(img.width for img in images)
        
        # Resize images to max resolution maintaining aspect ratio
        resized_images