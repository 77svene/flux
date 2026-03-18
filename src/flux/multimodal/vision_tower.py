"""
Vision Tower Module for Multi-Modal Training Pipeline
Supports LLaVA, Qwen-VL, and other Vision-Language Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import math
import warnings
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    SiglipVisionModel,
    SiglipImageProcessor,
    AutoModel,
    AutoImageProcessor,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import BaseModelOutput


class VisionEncoderType(Enum):
    """Supported vision encoder architectures."""
    CLIP = "clip"
    SIGLIP = "siglip"
    OPENCLIP = "openclip"
    CUSTOM = "custom"


@dataclass
class VisionTowerConfig:
    """Configuration for vision tower."""
    vision_encoder_type: VisionEncoderType = VisionEncoderType.CLIP
    vision_encoder_name_or_path: str = "openai/clip-vit-large-patch14-336"
    image_size: int = 336
    patch_size: int = 14
    num_image_tokens: int = 576  # (image_size // patch_size) ** 2
    hidden_size: int = 1024
    projection_dim: int = 4096
    use_dynamic_padding: bool = True
    max_image_size: int = 1024
    min_image_size: int = 224
    do_normalize: bool = True
    do_resize: bool = True
    do_center_crop: bool = False
    do_rescale: bool = True
    image_mean: Optional[List[float]] = None
    image_std: Optional[List[float]] = None
    freeze_vision_encoder: bool = True
    gradient_checkpointing: bool = False
    output_hidden_states: bool = False
    output_attentions: bool = False


class VisionProjection(nn.Module):
    """Projects vision encoder outputs to language model dimension."""
    
    def __init__(self, config: VisionTowerConfig):
        super().__init__()
        self.config = config
        
        # Linear projection layer
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.linear_2 = nn.Linear(config.hidden_size * 4, config.projection_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Activation function
        self.act = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for projection.
        
        Args:
            x: Vision encoder outputs [batch_size, num_patches, hidden_size]
            
        Returns:
            Projected features [batch_size, num_patches, projection_dim]
        """
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class DynamicImagePatcher:
    """Handles dynamic image patching for variable resolution inputs."""
    
    def __init__(self, config: VisionTowerConfig):
        self.config = config
        self.patch_size = config.patch_size
        
    def compute_optimal_grid(
        self, 
        image_height: int, 
        image_width: int
    ) -> Tuple[int, int]:
        """
        Compute optimal grid size for dynamic patching.
        
        Args:
            image_height: Height of input image
            image_width: Width of input image
            
        Returns:
            Tuple of (grid_height, grid_width)
        """
        # Ensure image dimensions are divisible by patch size
        target_height = max(
            self.config.min_image_size,
            min(
                self.config.max_image_size,
                (image_height // self.patch_size) * self.patch_size
            )
        )
        target_width = max(
            self.config.min_image_size,
            min(
                self.config.max_image_size,
                (image_width // self.patch_size) * self.patch_size
            )
        )
        
        grid_height = target_height // self.patch_size
        grid_width = target_width // self.patch_size
        
        return grid_height, grid_width
    
    def pad_image(
        self, 
        image: torch.Tensor, 
        grid_height: int, 
        grid_width: int
    ) -> torch.Tensor:
        """
        Pad image to match grid dimensions.
        
        Args:
            image: Input image tensor [C, H, W]
            grid_height: Target grid height
            grid_width: Target grid width
            
        Returns:
            Padded image tensor
        """
        target_height = grid_height * self.patch_size
        target_width = grid_width * self.patch_size
        
        current_height, current_width = image.shape[-2:]
        
        if current_height == target_height and current_width == target_width:
            return image
        
        # Pad with zeros (or could use reflection padding)
        pad_height = target_height - current_height
        pad_width = target_width - current_width
        
        padding = [
            pad_width // 2, 
            pad_width - pad_width // 2,
            pad_height // 2, 
            pad_height - pad_height // 2
        ]
        
        return F.pad(image, padding, mode='constant', value=0)
    
    def extract_patches(
        self, 
        image: torch.Tensor, 
        grid_height: int, 
        grid_width: int
    ) -> torch.Tensor:
        """
        Extract patches from image.
        
        Args:
            image: Input image tensor [C, H, W]
            grid_height: Grid height
            grid_width: Grid width
            
        Returns:
            Patches tensor [num_patches, patch_size * patch_size * C]
        """
        # Unfold to extract patches
        patches = image.unfold(1, self.patch_size, self.patch_size)
        patches = patches.unfold(2, self.patch_size, self.patch_size)
        
        # Reshape to [num_patches, patch_dim]
        patches = patches.contiguous().view(
            image.shape[0], 
            -1, 
            self.patch_size * self.patch_size
        )
        patches = patches.permute(1, 0, 2).contiguous().view(
            -1, 
            image.shape[0] * self.patch_size * self.patch_size
        )
        
        return patches


class VisionTower(nn.Module):
    """
    Unified vision tower for multi-modal training.
    Supports multiple vision encoder architectures with dynamic patching.
    """
    
    def __init__(
        self, 
        config: Optional[VisionTowerConfig] = None,
        vision_encoder: Optional[PreTrainedModel] = None,
        image_processor: Optional[Any] = None,
    ):
        super().__init__()
        
        if config is None:
            config = VisionTowerConfig()
        
        self.config = config
        self.vision_encoder_type = config.vision_encoder_type
        
        # Initialize vision encoder
        if vision_encoder is not None:
            self.vision_encoder = vision_encoder
        else:
            self.vision_encoder = self._create_vision_encoder()
        
        # Initialize image processor
        if image_processor is not None:
            self.image_processor = image_processor
        else:
            self.image_processor = self._create_image_processor()
        
        # Initialize projection layer
        self.projection = VisionProjection(config)
        
        # Initialize dynamic patcher
        self.patcher = DynamicImagePatcher(config)
        
        # Freeze vision encoder if specified
        if config.freeze_vision_encoder:
            self._freeze_vision_encoder()
        
        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing:
            self.vision_encoder.gradient_checkpointing_enable()
    
    def _create_vision_encoder(self) -> PreTrainedModel:
        """Create vision encoder based on configuration."""
        if self.vision_encoder_type == VisionEncoderType.CLIP:
            return CLIPVisionModel.from_pretrained(
                self.config.vision_encoder_name_or_path
            )
        elif self.vision_encoder_type == VisionEncoderType.SIGLIP:
            return SiglipVisionModel.from_pretrained(
                self.config.vision_encoder_name_or_path
            )
        elif self.vision_encoder_type == VisionEncoderType.OPENCLIP:
            # OpenCLIP models can be loaded via AutoModel
            return AutoModel.from_pretrained(
                self.config.vision_encoder_name_or_path
            )
        else:
            raise ValueError(
                f"Unsupported vision encoder type: {self.vision_encoder_type}"
            )
    
    def _create_image_processor(self) -> Any:
        """Create image processor based on configuration."""
        if self.vision_encoder_type == VisionEncoderType.CLIP:
            return CLIPImageProcessor.from_pretrained(
                self.config.vision_encoder_name_or_path,
                do_resize=self.config.do_resize,
                do_normalize=self.config.do_normalize,
                do_center_crop=self.config.do_center_crop,
                do_rescale=self.config.do_rescale,
                size=self.config.image_size,
                image_mean=self.config.image_mean,
                image_std=self.config.image_std,
            )
        elif self.vision_encoder_type == VisionEncoderType.SIGLIP:
            return SiglipImageProcessor.from_pretrained(
                self.config.vision_encoder_name_or_path,
                do_resize=self.config.do_resize,
                do_normalize=self.config.do_normalize,
                size=self.config.image_size,
            )
        else:
            return AutoImageProcessor.from_pretrained(
                self.config.vision_encoder_name_or_path
            )
    
    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def _unfreeze_vision_encoder(self):
        """Unfreeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
    
    def preprocess_images(
        self, 
        images: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess images using the image processor.
        
        Args:
            images: List of image tensors
            
        Returns:
            Dictionary with preprocessed images
        """
        # Convert tensors to PIL images if needed
        processed = self.image_processor(
            images=images,
            return_tensors="pt",
        )
        return processed
    
    def encode_images(
        self, 
        pixel_values: torch.Tensor,
        **kwargs
    ) -> BaseModelOutput:
        """
        Encode images using vision encoder.
        
        Args:
            pixel_values: Preprocessed image tensors
            **kwargs: Additional arguments for vision encoder
            
        Returns:
            Vision encoder outputs
        """
        # Forward pass through vision encoder
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=self.config.output_hidden_states,
            output_attentions=self.config.output_attentions,
            **kwargs,
        )
        
        return outputs
    
    def extract_image_features(
        self, 
        vision_outputs: BaseModelOutput,
        use_cls_token: bool = True,
    ) -> torch.Tensor:
        """
        Extract image features from vision encoder outputs.
        
        Args:
            vision_outputs: Outputs from vision encoder
            use_cls_token: Whether to use CLS token or average pooling
            
        Returns:
            Image features tensor
        """
        if hasattr(vision_outputs, "pooler_output") and use_cls_token:
            # Use pooled CLS token
            image_features = vision_outputs.pooler_output
        else:
            # Use last hidden state
            last_hidden_state = vision_outputs.last_hidden_state
            
            if use_cls_token:
                # Extract CLS token (first token)
                image_features = last_hidden_state[:, 0, :]
            else:
                # Average pooling over spatial dimensions
                image_features = last_hidden_state.mean(dim=1)
        
        return image_features
    
    def dynamic_encode(
        self, 
        images: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Encode images with dynamic patching for variable resolutions.
        
        Args:
            images: List of image tensors with variable sizes
            **kwargs: Additional arguments
            
        Returns:
            Projected image features
        """
        batch_features = []
        
        for image in images:
            # Get image dimensions
            if image.dim() == 3:
                c, h, w = image.shape
            else:
                # Assume batch of 1
                c, h, w = image.shape[1:]
                image = image[0]
            
            # Compute optimal grid
            grid_h, grid_w = self.patcher.compute_optimal_grid(h, w)
            
            # Pad image if needed
            padded_image = self.patcher.pad_image(image, grid_h, grid_w)
            
            # Preprocess single image
            processed = self.image_processor(
                images=[padded_image],
                return_tensors="pt",
            )
            
            # Encode image
            with torch.no_grad() if self.config.freeze_vision_encoder else torch.enable_grad():
                vision_outputs = self.encode_images(
                    processed["pixel_values"].to(image.device),
                    **kwargs
                )
            
            # Extract features
            features = self.extract_image_features(vision_outputs)
            batch_features.append(features)
        
        # Stack batch features
        batch_features = torch.stack(batch_features, dim=0)
        
        return batch_features
    
    def forward(
        self,
        images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for vision tower.
        
        Args:
            images: Raw image tensors (optional)
            pixel_values: Preprocessed pixel values (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with image features and projections
        """
        if pixel_values is None and images is not None:
            # Preprocess images if raw images provided
            if isinstance(images, list):
                # Handle dynamic batching with variable resolutions
                image_features = self.dynamic_encode(images, **kwargs)
            else:
                # Standard batch processing
                processed = self.preprocess_images(images)
                pixel_values = processed["pixel_values"]
        
        if pixel_values is not None:
            # Encode images
            vision_outputs = self.encode_images(pixel_values, **kwargs)
            
            # Extract features
            image_features = self.extract_image_features(vision_outputs)
        else:
            raise ValueError("Either images or pixel_values must be provided")
        
        # Project features to language model dimension
        projected_features = self.projection(image_features)
        
        return {
            "image_features": image_features,
            "projected_features": projected_features,
            "vision_outputs": vision_outputs if 'vision_outputs' in locals() else None,
        }
    
    def get_image_embeddings(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Get normalized image embeddings for contrastive learning.
        
        Args:
            images: Input images
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Image embeddings tensor
        """
        outputs = self.forward(images=images)
        embeddings = outputs["image_features"]
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class MultiModalLoss(nn.Module):
    """
    Multi-modal loss functions for vision-language alignment.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.1,
        contrastive_weight: float = 1.0,
        captioning_weight: float = 1.0,
        alignment_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight
        self.captioning_weight = captioning_weight
        self.alignment_weight = alignment_weight
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))
    
    def contrastive_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings [batch_size, embed_dim]
            text_embeddings: Text embeddings [batch_size, embed_dim]
            labels: Optional labels for supervised contrastive loss
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embeddings @ text_embeddings.t()
        
        if labels is None:
            # Symmetric contrastive loss (CLIP-style)
            batch_size = image_embeddings.shape[0]
            labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Cross entropy loss
        loss_i2t = F.cross_entropy(
            logits, 
            labels, 
            label_smoothing=self.label_smoothing
        )
        loss_t2i = F.cross_entropy(
            logits.t(), 
            labels, 
            label_smoothing=self.label_smoothing
        )
        
        return (loss_i2t + loss_t2i) / 2
    
    def captioning_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Compute captioning/language modeling loss.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            ignore_index: Index to ignore in loss calculation
            
        Returns:
            Captioning loss value
        """
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def alignment_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_to_text_matrix: torch.Tensor,
        text_to_image_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute vision-language alignment loss.
        
        Args:
            image_features: Image features [batch_size, num_patches, dim]
            text_features: Text features [batch_size, seq_len, dim]
            image_to_text_matrix: Attention matrix from image to text
            text_to_image_matrix: Attention matrix from text to image
            
        Returns:
            Alignment loss value
        """
        # Compute KL divergence between attention distributions
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        # Create uniform target distributions
        batch_size, num_patches, _ = image_features.shape
        _, seq_len, _ = text_features.shape
        
        uniform_image = torch.ones(batch_size, num_patches, device=image_features.device) / num_patches
        uniform_text = torch.ones(batch_size, seq_len, device=text_features.device) / seq_len
        
        # Compute alignment losses
        loss_i2t = kl_loss(
            F.log_softmax(image_to_text_matrix, dim=-1),
            uniform_text.unsqueeze(1).expand_as(image_to_text_matrix)
        )
        loss_t2i = kl_loss(
            F.log_softmax(text_to_image_matrix, dim=-1),
            uniform_image.unsqueeze(1).expand_as(text_to_image_matrix)
        )
        
        return (loss_i2t + loss_t2i) / 2
    
    def forward(
        self,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        image_to_text_matrix: Optional[torch.Tensor] = None,
        text_to_image_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined multi-modal loss.
        
        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}
        total_loss = 0.0
        
        # Contrastive loss
        if image_embeddings is not None and text_embeddings is not None:
            contrastive_loss = self.contrastive_loss(
                image_embeddings, 
                text_embeddings
            )
            losses["contrastive_loss"] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss
        
        # Captioning loss
        if logits is not None and labels is not None:
            captioning_loss = self.captioning_loss(logits, labels)
            losses["captioning_loss"] = captioning_loss
            total_loss += self.captioning_weight * captioning_loss
        
        # Alignment loss
        if (image_features is not None and text_features is not None and
            image_to_text_matrix is not None and text_to_image_matrix is not None):
            alignment_loss = self.alignment_loss(
                image_features,
                text_features,
                image_to_text_matrix,
                text_to_image_matrix,
            )
            losses["alignment_loss"] = alignment_loss
            total_loss += self.alignment_weight * alignment_loss
        
        losses["total_loss"] = total_loss
        
        return losses


class MultiModalDataLoader:
    """
    Data loader for multi-modal training with image-text pairs.
    """
    
    def __init__(
        self,
        vision_tower: VisionTower,
        tokenizer: Any,
        image_processor: Optional[Any] = None,
        max_length: int = 512,
        image_token: str = "<image>",
        ignore_index: int = -100,
    ):
        self.vision_tower = vision_tower
        self.tokenizer = tokenizer
        self.image_processor = image_processor or vision_tower.image_processor
        self.max_length = max_length
        self.image_token = image_token
        self.ignore_index = ignore_index
        
        # Add special tokens if not present
        if image_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    
    def preprocess_text(
        self, 
        text: str, 
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text with tokenization.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tokenized text dictionary
        """
        if max_length is None:
            max_length = self.max_length
        
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return tokenized
    
    def create_image_text_sample(
        self,
        image: torch.Tensor,
        text: str,
        instruction: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Create a single image-text training sample.
        
        Args:
            image: Image tensor
            text: Target text/caption
            instruction: Optional instruction for visual question answering
            
        Returns:
            Dictionary with processed sample
        """
        # Preprocess image
        processed_image = self.image_processor(
            images=[image],
            return_tensors="pt",
        )
        
        # Create input text with image token
        if instruction:
            input_text = f"{instruction}\n{self.image_token}"
        else:
            input_text = f"Describe the image: {self.image_token}"
        
        # Tokenize input and output
        input_tokens = self.preprocess_text(input_text)
        output_tokens = self.preprocess_text(text)
        
        # Create labels (ignore input tokens and padding)
        labels = output_tokens["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        return {
            "pixel_values": processed_image["pixel_values"].squeeze(0),
            "input_ids": input_tokens["input_ids"].squeeze(0),
            "attention_mask": input_tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
    
    def create_batch(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Create a batch from multiple samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Batched dictionary
        """
        batch = {}
        
        for key in samples[0].keys():
            if key == "pixel_values":
                # Stack images
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                # Stack token tensors
                batch[key] = torch.stack([s[key] for s in samples])
        
        return batch
    
    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for multi-modal data.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        # Extract images and texts
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        instructions = [item.get("instruction") for item in batch]
        
        # Process each sample
        processed_samples = []
        for image, text, instruction in zip(images, texts, instructions):
            sample = self.create_image_text_sample(image, text, instruction)
            processed_samples.append(sample)
        
        # Create batch
        return self.create_batch(processed_samples)


def build_vision_tower(
    config: Optional[VisionTowerConfig] = None,
    **kwargs,
) -> VisionTower:
    """
    Factory function to build vision tower.
    
    Args:
        config: Vision tower configuration
        **kwargs: Additional arguments
        
    Returns:
        VisionTower instance
    """
    if config is None:
        config = VisionTowerConfig()
    
    # Merge kwargs into config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return VisionTower(config=config)


def create_vision_tower_from_pretrained(
    model_name_or_path: str,
    vision_encoder_type: Optional[VisionEncoderType] = None,
    **kwargs,
) -> VisionTower:
    """
    Create vision tower from pretrained model.
    
    Args:
        model_name_or_path: Path to pretrained model
        vision_encoder_type: Type of vision encoder
        **kwargs: Additional arguments
        
    Returns:
        VisionTower instance
    """
    # Determine vision encoder type from model name if not specified
    if vision_encoder_type is None:
        if "clip" in model_name_or_path.lower():
            vision_encoder_type = VisionEncoderType.CLIP
        elif "siglip" in model_name_or_path.lower():
            vision_encoder_type = VisionEncoderType.SIGLIP
        else:
            vision_encoder_type = VisionEncoderType.CLIP  # Default
    
    config = VisionTowerConfig(
        vision_encoder_type=vision_encoder_type,
        vision_encoder_name_or_path=model_name_or_path,
        **kwargs,
    )
    
    return VisionTower(config=config)


# Example usage and integration points
if __name__ == "__main__":
    # Example: Create a vision tower for LLaVA
    print("Building Vision Tower for Multi-Modal Training...")
    
    # Create vision tower
    vision_tower = build_vision_tower(
        config=VisionTowerConfig(
            vision_encoder_type=VisionEncoderType.CLIP,
            vision_encoder_name_or_path="openai/clip-vit-large-patch14-336",
            image_size=336,
            patch_size=14,
            projection_dim=4096,
            freeze_vision_encoder=True,
        )
    )
    
    # Create multi-modal loss
    loss_fn = MultiModalLoss(
        temperature=0.07,
        contrastive_weight=1.0,
        captioning_weight=1.0,
        alignment_weight=0.5,
    )
    
    print(f"Vision Tower created with {sum(p.numel() for p in vision_tower.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in vision_tower.parameters() if p.requires_grad)}")