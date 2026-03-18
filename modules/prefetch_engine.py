import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, OrderedDict
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# Constants
MAX_SEQUENCE_LENGTH = 64
PREDICTION_BATCH_SIZE = 32
MODEL_CACHE_SIZE = 3  # Number of models to keep preloaded
TRAINING_INTERVAL = 300  # Train every 5 minutes
PREFETCH_IDLE_THRESHOLD = 2.0  # Seconds of idle time before prefetching
SESSION_FILE = "prefetch_sessions.json"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LightweightTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class ModelRegistry:
    """Registry for tracking available models and LoRAs with their identifiers"""
    def __init__(self):
        self.model_to_id = {}
        self.id_to_model = {}
        self.next_id = 0
        self.lock = threading.Lock()
        
    def get_id(self, model_name: str) -> int:
        with self.lock:
            if model_name not in self.model_to_id:
                self.model_to_id[model_name] = self.next_id
                self.id_to_model[self.next_id] = model_name
                self.next_id += 1
            return self.model_to_id[model_name]
    
    def get_model(self, model_id: int) -> Optional[str]:
        with self.lock:
            return self.id_to_model.get(model_id)
    
    @property
    def vocab_size(self) -> int:
        with self.lock:
            return max(self.next_id, 1)  # Ensure at least 1 for empty vocab

class SessionTracker:
    """Tracks user sessions and model usage patterns"""
    def __init__(self, session_file: str = SESSION_FILE):
        self.session_file = Path(session_file)
        self.current_session = []
        self.sessions = []
        self.lock = threading.Lock()
        self.load_sessions()
    
    def load_sessions(self):
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', [])[-1000:]  # Keep last 1000 sessions
                    logger.info(f"Loaded {len(self.sessions)} previous sessions")
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
            self.sessions = []
    
    def save_sessions(self):
        try:
            with self.session_file, 'w') as f:
                json.dump({
                    'sessions': self.sessions[-1000:],  # Keep last 1000
                    'last_updated': time.time()
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save sessions: {e}")
    
    def add_transition(self, from_model: int, to_model: int):
        with self.lock:
            self.current_session.append({
                'from': from_model,
                'to': to_model,
                'timestamp': time.time()
            })
    
    def end_session(self):
        with self.lock:
            if self.current_session:
                self.sessions.append(self.current_session.copy())
                self.current_session = []
                # Save periodically
                if len(self.sessions) % 10 == 0:
                    self.save_sessions()
    
    def get_training_data(self, sequence_length: int = MAX_SEQUENCE_LENGTH) -> List[List[int]]:
        """Get sequences of model transitions for training"""
        sequences = []
        with self.lock:
            # Use current session
            if len(self.current_session) >= 2:
                seq = [self.current_session[0]['from']]
                for transition in self.current_session:
                    seq.append(transition['to'])
                if len(seq) >= 2:
                    sequences.append(seq[-sequence_length:])
            
            # Use historical sessions
            for session in self.sessions[-100:]:  # Last 100 sessions
                if len(session) >= 2:
                    seq = [session[0]['from']]
                    for transition in session:
                        seq.append(transition['to'])
                    sequences.append(seq[-sequence_length:])
        
        return sequences

class LRUCache:
    """LRU Cache for preloaded models"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def contains(self, key: str) -> bool:
        with self.lock:
            return key in self.cache
    
    def keys(self) -> List[str]:
        with self.lock:
            return list(self.cache.keys())
    
    def remove(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]

class PrefetchEngine:
    """Main prefetch engine that coordinates prediction and preloading"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.registry = ModelRegistry()
        self.session_tracker = SessionTracker()
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.model_cache = LRUCache(MODEL_CACHE_SIZE)
        self.is_training = False
        self.is_prefetching = False
        self.last_activity_time = time.time()
        self.last_training_time = 0
        self.enabled = True
        self.training_thread = None
        self.prefetch_thread = None
        self.lock = threading.Lock()
        
        # Callbacks for integration
        self.load_model_callback = None
        self.load_lora_callback = None
        self.unload_model_callback = None
        
        # Initialize model
        self._initialize_model()
        
        # Start background threads
        self._start_background_threads()
    
    def _initialize_model(self):
        """Initialize or load the transformer model"""
        try:
            vocab_size = max(self.registry.vocab_size, 100)  # Start with reasonable size
            self.model = LightweightTransformer(
                vocab_size=vocab_size,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Try to load pre-trained weights
            model_path = Path("models/prefetch_model.pt")
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded pre-trained prefetch model")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to initialize prefetch model: {e}")
            self.model = None
    
    def _start_background_threads(self):
        """Start background threads for training and prefetching"""
        # Training thread
        self.training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="prefetch-training"
        )
        self.training_thread.start()
        
        # Prefetch thread
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name="prefetch-worker"
        )
        self.prefetch_thread.start()
    
    def _training_loop(self):
        """Background thread for training the model"""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                if not self.enabled or self.is_training:
                    continue
                
                current_time = time.time()
                if current_time - self.last_training_time < TRAINING_INTERVAL:
                    continue
                
                # Get training data
                sequences = self.session_tracker.get_training_data()
                if len(sequences) < 5:  # Need minimum data
                    continue
                
                self._train_on_sequences(sequences)
                self.last_training_time = current_time
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                time.sleep(30)
    
    def _train_on_sequences(self, sequences: List[List[int]]):
        """Train model on sequences of model transitions"""
        try:
            self.is_training = True
            self.model.train()
            
            # Prepare training data
            inputs = []
            targets = []
            
            for seq in sequences:
                if len(seq) < 2:
                    continue
                
                for i in range(1, len(seq)):
                    input_seq = seq[:i][-MAX_SEQUENCE_LENGTH:]
                    target = seq[i]
                    
                    # Pad sequence
                    padded = [0] * (MAX_SEQUENCE_LENGTH - len(input_seq)) + input_seq
                    inputs.append(padded)
                    targets.append(target)
            
            if not inputs:
                return
            
            # Convert to tensors
            inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=self.device)
            targets_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
            
            # Training loop
            dataset_size = inputs_tensor.size(0)
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, PREDICTION_BATCH_SIZE):
                end = min(start + PREDICTION_BATCH_SIZE, dataset_size)
                batch_indices = indices[start:end]
                
                batch_inputs = inputs_tensor[batch_indices]
                batch_targets = targets_tensor[batch_indices]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                
                # Get predictions for last token in sequence
                last_outputs = outputs[:, -1, :]
                loss = self.criterion(last_outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Save model periodically
            self._save_model()
            
            logger.debug(f"Trained prefetch model on {len(inputs)} samples")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
        finally:
            self.model.eval()
            self.is_training = False
    
    def _save_model(self):
        """Save model weights to disk"""
        try:
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'registry': {
                    'model_to_id': self.registry.model_to_id,
                    'id_to_model': self.registry.id_to_model,
                    'next_id': self.registry.next_id
                }
            }, model_dir / "prefetch_model.pt")
            
        except Exception as e:
            logger.warning(f"Failed to save prefetch model: {e}")
    
    def _prefetch_loop(self):
        """Background thread for prefetching models during idle time"""
        while True:
            try:
                time.sleep(0.5)
                
                if not self.enabled or self.is_prefetching:
                    continue
                
                current_time = time.time()
                idle_time = current_time - self.last_activity_time
                
                # Only prefetch during sufficient idle time
                if idle_time < PREFETCH_IDLE_THRESHOLD:
                    continue
                
                # Get predictions
                predictions = self._get_predictions()
                if not predictions:
                    continue
                
                # Prefetch top predictions
                self._prefetch_models(predictions[:2])  # Top 2 predictions
                
            except Exception as e:
                logger.error(f"Prefetch loop error: {e}")
                time.sleep(5)
    
    def _get_predictions(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get model predictions for next likely models"""
        if not self.model or not self.session_tracker.current_session:
            return []
        
        try:
            self.model.eval()
            
            # Get current sequence
            current_seq = []
            for transition in self.session_tracker.current_session[-MAX_SEQUENCE_LENGTH:]:
                current_seq.append(transition['to'])
            
            if not current_seq:
                return []
            
            # Prepare input
            padded = [0] * (MAX_SEQUENCE_LENGTH - len(current_seq)) + current_seq
            input_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_tensor)
                last_output = outputs[0, -1, :]
                probabilities = F.softmax(last_output, dim=0)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                model_name = self.registry.get_model(idx.item())
                if model_name:
                    predictions.append((model_name, prob.item()))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
    
    def _prefetch_models(self, predictions: List[Tuple[str, float]]):
        """Prefetch predicted models"""
        try:
            self.is_prefetching = True
            
            for model_name, confidence in predictions:
                if confidence < 0.1:  # Skip low confidence predictions
                    continue
                
                # Check if already cached
                if self.model_cache.contains(model_name):
                    continue
                
                # Check if model exists
                model_path = self._find_model_path(model_name)
                if not model_path:
                    continue
                
                # Load model in background
                logger.info(f"Prefetching {model_name} (confidence: {confidence:.2f})")
                
                if model_name.endswith('.safetensors') or model_name.endswith('.ckpt'):
                    # It's a main model
                    if self.load_model_callback:
                        self.load_model_callback(model_path, preload=True)
                elif model_name.endswith('.safetensors') and 'lora' in model_name.lower():
                    # It's a LoRA
                    if self.load_lora_callback:
                        self.load_lora_callback(model_path, preload=True)
                
                # Add to cache
                self.model_cache.put(model_name, {
                    'path': model_path,
                    'timestamp': time.time(),
                    'confidence': confidence
                })
                
                # Respect memory limits
                if len(self.model_cache.keys()) >= MODEL_CACHE_SIZE:
                    oldest_key = self.model_cache.keys()[0]
                    self.model_cache.remove(oldest_key)
                    
                    # Unload if callback exists
                    if self.unload_model_callback:
                        self.unload_model_callback(oldest_key)
        
        except Exception as e:
            logger.error(f"Prefetching failed: {e}")
        finally:
            self.is_prefetching = False
    
    def _find_model_path(self, model_name: str) -> Optional[str]:
        """Find the actual path for a model name"""
        # Search in common directories
        search_dirs = [
            "models/Stable-diffusion",
            "models/Lora",
            "models/LyCORIS",
            "embeddings"
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file == model_name or model_name in file:
                        return os.path.join(root, file)
        
        return None
    
    def record_transition(self, from_model: str, to_model: str):
        """Record a model transition for learning"""
        if not self.enabled:
            return
        
        try:
            from_id = self.registry.get_id(from_model)
            to_id = self.registry.get_id(to_model)
            
            self.session_tracker.add_transition(from_id, to_id)
            self.last_activity_time = time.time()
            
            # Update vocab size if needed
            if self.registry.vocab_size > self.model.vocab_size:
                self._update_model_vocab()
                
        except Exception as e:
            logger.error(f"Failed to record transition: {e}")
    
    def _update_model_vocab(self):
        """Update model vocabulary when new models are added"""
        try:
            new_vocab_size = self.registry.vocab_size
            if new_vocab_size <= self.model.vocab_size:
                return
            
            # Create new model with expanded vocabulary
            old_model = self.model
            self.model = LightweightTransformer(
                vocab_size=new_vocab_size,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1
            ).to(self.device)
            
            # Copy weights for existing vocabulary
            with torch.no_grad():
                self.model.embedding.weight[:old_model.vocab_size] = old_model.embedding.weight
                self.model.decoder.weight[:old_model.vocab_size] = old_model.decoder.weight
                self.model.decoder.bias[:old_model.vocab_size] = old_model.decoder.bias
            
            # Update optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            logger.info(f"Updated model vocabulary from {old_model.vocab_size} to {new_vocab_size}")
            
        except Exception as e:
            logger.error(f"Failed to update model vocab: {e}")
    
    def set_callbacks(self, load_model=None, load_lora=None, unload_model=None):
        """Set callbacks for model loading/unloading"""
        self.load_model_callback = load_model
        self.load_lora_callback = load_lora
        self.unload_model_callback = unload_model
    
    def enable(self):
        """Enable prefetching"""
        self.enabled = True
        logger.info("Prefetch engine enabled")
    
    def disable(self):
        """Disable prefetching"""
        self.enabled = False
        logger.info("Prefetch engine disabled")
    
    def clear_cache(self):
        """Clear the prefetch cache"""
        self.model_cache = LRUCache(MODEL_CACHE_SIZE)
        logger.info("Prefetch cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the prefetch engine"""
        return {
            'enabled': self.enabled,
            'vocab_size': self.registry.vocab_size,
            'current_session_length': len(self.session_tracker.current_session),
            'total_sessions': len(self.session_tracker.sessions),
            'cached_models': self.model_cache.keys(),
            'last_training': self.last_training_time,
            'is_training': self.is_training,
            'is_prefetching': self.is_prefetching,
            'device': str(self.device)
        }
    
    def end_session(self):
        """End current session and save data"""
        self.session_tracker.end_session()
        self._save_model()

# Global instance
_prefetch_engine = None

def get_prefetch_engine() -> PrefetchEngine:
    """Get or create the global prefetch engine instance"""
    global _prefetch_engine
    if _prefetch_engine is None:
        _prefetch_engine = PrefetchEngine()
    return _prefetch_engine

def initialize_prefetch_engine():
    """Initialize the prefetch engine (call during startup)"""
    engine = get_prefetch_engine()
    
    # Integration with existing modules
    try:
        # Import here to avoid circular imports
        from modules import sd_models, extra_networks
        
        # Set up callbacks
        def load_model_callback(model_path, preload=False):
            if preload:
                # Preload in background
                threading.Thread(
                    target=lambda: sd_models.load_model(model_path),
                    daemon=True
                ).start()
            else:
                sd_models.load_model(model_path)
        
        def load_lora_callback(lora_path, preload=False):
            if preload:
                threading.Thread(
                    target=lambda: extra_networks.activate(['lora', lora_path, '']),
                    daemon=True
                ).start()
            else:
                extra_networks.activate(['lora', lora_path, ''])
        
        def unload_model_callback(model_name):
            # This would need to be implemented based on the actual unloading mechanism
            pass
        
        engine.set_callbacks(
            load_model=load_model_callback,
            load_lora=load_lora_callback,
            unload_model=unload_model_callback
        )
        
        logger.info("Prefetch engine initialized with callbacks")
        
    except ImportError as e:
        logger.warning(f"Could not integrate with existing modules: {e}")
    
    return engine

# Hook functions for integration with existing code
def on_model_loaded(model_name: str):
    """Call when a model is loaded"""
    engine = get_prefetch_engine()
    
    # Get previous model from session
    if engine.session_tracker.current_session:
        last_transition = engine.session_tracker.current_session[-1]
        prev_model_id = last_transition['to']
        prev_model = engine.registry.get_model(prev_model_id)
    else:
        prev_model = "unknown"
    
    engine.record_transition(prev_model, model_name)

def on_lora_loaded(lora_name: str):
    """Call when a LoRA is loaded"""
    engine = get_prefetch_engine()
    
    # Treat LoRA as a model for prediction purposes
    lora_id = f"lora:{lora_name}"
    
    if engine.session_tracker.current_session:
        last_transition = engine.session_tracker.current_session[-1]
        prev_model_id = last_transition['to']
        prev_model = engine.registry.get_model(prev_model_id)
    else:
        prev_model = "unknown"
    
    engine.record_transition(prev_model, lora_id)

def on_generation_start():
    """Call when generation starts"""
    engine = get_prefetch_engine()
    engine.last_activity_time = time.time()

def on_generation_end():
    """Call when generation ends"""
    pass  # Activity time is already updated

# Cleanup on exit
import atexit

@atexit.register
def cleanup_prefetch_engine():
    """Cleanup when the application exits"""
    global _prefetch_engine
    if _prefetch_engine:
        _prefetch_engine.end_session()
        logger.info("Prefetch engine cleaned up")