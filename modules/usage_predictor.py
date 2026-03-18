"""
Usage Predictor Module for Stable Diffusion WebUI
ML-powered prediction of next likely model/LoRA based on usage patterns
"""

import os
import json
import time
import threading
import queue
import hashlib
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import gc

import torch
import numpy as np
from modules import sd_models, shared, extra_networks, paths
from modules.sd_models import model_data
from modules.script_callbacks import on_model_loaded, on_script_unloaded
from modules.timer import Timer

# Constants
CACHE_DIR = Path(paths.script_path) / "models" / "usage_predictor_cache"
MODEL_SAVE_PATH = CACHE_DIR / "predictor_model.pkl"
SESSION_LOG_PATH = CACHE_DIR / "session_logs"
MAX_SESSION_HISTORY = 1000
PREDICTION_WINDOW = 5  # Predict next N models
PRELOAD_QUEUE_SIZE = 3
LRU_CACHE_SIZE = 5
TRAINING_INTERVAL = 3600  # Retrain every hour
IDLE_THRESHOLD = 2.0  # Seconds of idle time before preloading


class ModelType(Enum):
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    LYCORIS = "lycoris"
    HYPERNETWORK = "hypernetwork"
    EMBEDDING = "embedding"


@dataclass
class ModelUsageEvent:
    timestamp: float
    model_name: str
    model_type: ModelType
    model_hash: Optional[str] = None
    session_id: Optional[str] = None
    prompt_hash: Optional[str] = None


@dataclass
class PredictionResult:
    model_name: str
    model_type: ModelType
    confidence: float
    predicted_at: float


class LRUCache:
    """Thread-safe LRU cache for preloaded models"""
    
    def __init__(self, max_size: int = LRU_CACHE_SIZE):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.order: deque = deque()
        self.lock = threading.RLock()
        self.access_count: Dict[str, int] = defaultdict(int)
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
                self.order.append(key)
                self.access_count[key] += 1
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> Optional[str]:
        """Returns evicted key if any"""
        with self.lock:
            evicted = None
            if key in self.cache:
                self.order.remove(key)
            elif len(self.cache) >= self.max_size:
                evicted = self.order.popleft()
                del self.cache[evicted]
                if evicted in self.access_count:
                    del self.access_count[evicted]
            
            self.cache[key] = value
            self.order.append(key)
            self.access_count[key] = 1
            return evicted
    
    def remove(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
                return True
            return False
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.order.clear()
            self.access_count.clear()
    
    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.cache
    
    def __len__(self) -> int:
        with self.lock:
            return len(self.cache)


class SimpleTransformerPredictor:
    """Lightweight transformer for model transition prediction"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, max_seq_len: int = 50):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Initialize simple attention-based model
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, embed_dim)
        
        # Simple transformer layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = torch.nn.Linear(embed_dim, vocab_size)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def to(self, device):
        self.embeddings = self.embeddings.to(device)
        self.positional_encoding = self.positional_encoding.to(device)
        self.transformer = self.transformer.to(device)
        self.output_layer = self.output_layer.to(device)
        return self
    
    def parameters(self):
        return list(self.embeddings.parameters()) + \
               list(self.transformer.parameters()) + \
               list(self.output_layer.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        
        # Embed and add positional encoding
        x = self.embeddings(x)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Get predictions for next token
        x = self.output_layer(x[:, -1, :])  # Only last position
        return x
    
    def train_step(self, sequences: torch.Tensor, targets: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        outputs = self.forward(sequences)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict(self, sequence: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sequence.unsqueeze(0))
            probs = torch.softmax(outputs, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            return list(zip(top_indices.cpu().numpy()[0], top_probs.cpu().numpy()[0]))


class UsagePredictor:
    """Main class for predicting model usage and preloading"""
    
    def __init__(self):
        self.enabled = shared.opts.data.get("usage_predictor_enabled", True)
        self.session_id = self._generate_session_id()
        self.usage_history: List[ModelUsageEvent] = []
        self.transition_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.model_vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.predictor: Optional[SimpleTransformerPredictor] = None
        self.preload_queue = queue.Queue(maxsize=PRELOAD_QUEUE_SIZE)
        self.preload_cache = LRUCache()
        self.is_preloading = False
        self.last_activity_time = time.time()
        self.training_thread: Optional[threading.Thread] = None
        self.preload_thread: Optional[threading.Thread] = None
        self.model_lock = threading.RLock()
        self.data_lock = threading.RLock()
        
        # Initialize
        self._setup_directories()
        self._load_session_data()
        self._start_background_threads()
        
        # Register callbacks
        on_model_loaded(self._on_model_loaded)
        on_script_unloaded(self._on_unload)
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        return hashlib.md5(f"{time.time()}_{os.getpid()}".encode()).hexdigest()[:8]
    
    def _setup_directories(self):
        """Create necessary directories"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_LOG_PATH.mkdir(parents=True, exist_ok=True)
    
    def _load_session_data(self):
        """Load previous session data and model"""
        try:
            if MODEL_SAVE_PATH.exists():
                with open(MODEL_SAVE_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.transition_matrix = data.get('transition_matrix', {})
                    self.model_vocab = data.get('model_vocab', {})
                    self.reverse_vocab = {v: k for k, v in self.model_vocab.items()}
                    print(f"[UsagePredictor] Loaded model with {len(self.model_vocab)} models in vocabulary")
            
            # Load recent session logs
            self._load_recent_sessions()
            
            # Initialize predictor if we have enough data
            if len(self.model_vocab) > 10:
                self._initialize_predictor()
        
        except Exception as e:
            print(f"[UsagePredictor] Error loading data: {e}")
    
    def _load_recent_sessions(self, max_sessions: int = 10):
        """Load recent session logs"""
        try:
            session_files = sorted(SESSION_LOG_PATH.glob("*.json"), 
                                 key=lambda x: x.stat().st_mtime, reverse=True)[:max_sessions]
            
            for session_file in session_files:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    for event in session_data.get('events', []):
                        self.usage_history.append(ModelUsageEvent(**event))
            
            # Trim history if too long
            if len(self.usage_history) > MAX_SESSION_HISTORY:
                self.usage_history = self.usage_history[-MAX_SESSION_HISTORY:]
        
        except Exception as e:
            print(f"[UsagePredictor] Error loading sessions: {e}")
    
    def _save_session_data(self):
        """Save current session data"""
        try:
            with self.data_lock:
                # Save model
                model_data = {
                    'transition_matrix': dict(self.transition_matrix),
                    'model_vocab': self.model_vocab,
                    'last_updated': time.time()
                }
                with open(MODEL_SAVE_PATH, 'wb') as f:
                    pickle.dump(model_data, f)
                
                # Save current session
                session_file = SESSION_LOG_PATH / f"session_{self.session_id}_{int(time.time())}.json"
                session_data = {
                    'session_id': self.session_id,
                    'timestamp': time.time(),
                    'events': [asdict(e) for e in self.usage_history[-100:]]  # Last 100 events
                }
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
        
        except Exception as e:
            print(f"[UsagePredictor] Error saving data: {e}")
    
    def _initialize_predictor(self):
        """Initialize the transformer predictor"""
        try:
            vocab_size = len(self.model_vocab) + 1  # +1 for unknown
            self.predictor = SimpleTransformerPredictor(vocab_size=vocab_size)
            print(f"[UsagePredictor] Initialized predictor with vocab size {vocab_size}")
        except Exception as e:
            print(f"[UsagePredictor] Error initializing predictor: {e}")
            self.predictor = None
    
    def _start_background_threads(self):
        """Start background processing threads"""
        if not self.enabled:
            return
        
        # Training thread
        self.training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="UsagePredictor-Training"
        )
        self.training_thread.start()
        
        # Preloading thread
        self.preload_thread = threading.Thread(
            target=self._preload_loop,
            daemon=True,
            name="UsagePredictor-Preload"
        )
        self.preload_thread.start()
    
    def _training_loop(self):
        """Background thread for model training"""
        while self.enabled:
            try:
                time.sleep(TRAINING_INTERVAL)
                if len(self.usage_history) > 100:  # Minimum data for training
                    self._train_model()
            except Exception as e:
                print(f"[UsagePredictor] Training error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _preload_loop(self):
        """Background thread for model preloading"""
        while self.enabled:
            try:
                # Check if we should preload
                idle_time = time.time() - self.last_activity_time
                if idle_time > IDLE_THRESHOLD and not self.preload_queue.empty():
                    self._process_preload_queue()
                time.sleep(0.5)  # Check every 500ms
            except Exception as e:
                print(f"[UsagePredictor] Preload error: {e}")
                time.sleep(5)
    
    def _on_model_loaded(self, model, **kwargs):
        """Callback when a model is loaded"""
        try:
            model_name = getattr(model, 'name', None) or getattr(model, 'model_name', None)
            if not model_name:
                return
            
            # Determine model type
            model_type = ModelType.CHECKPOINT
            if hasattr(model, 'lora_name'):
                model_type = ModelType.LORA
            elif hasattr(model, 'lyco_name'):
                model_type = ModelType.LYCORIS
            
            # Record usage
            self.record_model_usage(model_name, model_type)
            
            # Update last activity time
            self.last_activity_time = time.time()
        
        except Exception as e:
            print(f"[UsagePredictor] Error in model loaded callback: {e}")
    
    def _on_unload(self, **kwargs):
        """Cleanup on unload"""
        self.enabled = False
        self._save_session_data()
        self.preload_cache.clear()
        
        # Clear GPU memory if predictor exists
        if self.predictor:
            del self.predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def record_model_usage(self, model_name: str, model_type: ModelType, 
                          prompt_hash: Optional[str] = None):
        """Record a model usage event"""
        if not self.enabled:
            return
        
        try:
            # Create event
            event = ModelUsageEvent(
                timestamp=time.time(),
                model_name=model_name,
                model_type=model_type,
                session_id=self.session_id,
                prompt_hash=prompt_hash
            )
            
            with self.data_lock:
                # Add to history
                self.usage_history.append(event)
                
                # Trim history if needed
                if len(self.usage_history) > MAX_SESSION_HISTORY:
                    self.usage_history = self.usage_history[-MAX_SESSION_HISTORY:]
                
                # Update vocabulary
                if model_name not in self.model_vocab:
                    vocab_id = len(self.model_vocab) + 1  # 0 reserved for unknown
                    self.model_vocab[model_name] = vocab_id
                    self.reverse_vocab[vocab_id] = model_name
                
                # Update transition matrix
                if len(self.usage_history) >= 2:
                    prev_event = self.usage_history[-2]
                    self.transition_matrix[prev_event.model_name][model_name] += 1
            
            # Trigger prediction
            self._queue_predictions()
        
        except Exception as e:
            print(f"[UsagePredictor] Error recording usage: {e}")
    
    def _queue_predictions(self):
        """Generate and queue predictions based on recent usage"""
        try:
            predictions = self.predict_next_models(PREDICTION_WINDOW)
            
            for pred in predictions:
                if pred.model_name not in self.preload_cache:
                    try:
                        self.preload_queue.put_nowait(pred)
                    except queue.Full:
                        pass  # Queue full, skip
        
        except Exception as e:
            print(f"[UsagePredictor] Error queuing predictions: {e}")
    
    def predict_next_models(self, n: int = 3) -> List[PredictionResult]:
        """Predict next n likely models"""
        predictions = []
        
        try:
            with self.data_lock:
                if len(self.usage_history) < 2:
                    return predictions
                
                # Get recent model sequence
                recent_models = [e.model_name for e in self.usage_history[-10:]]
                
                # Use transformer if available
                if self.predictor and len(self.model_vocab) > 10:
                    predictions = self._transformer_predict(recent_models, n)
                
                # Fallback to Markov chain
                if not predictions:
                    predictions = self._markov_predict(recent_models, n)
        
        except Exception as e:
            print(f"[UsagePredictor] Prediction error: {e}")
        
        return predictions
    
    def _transformer_predict(self, recent_models: List[str], n: int) -> List[PredictionResult]:
        """Make predictions using transformer model"""
        predictions = []
        
        try:
            # Convert to sequence
            sequence = [self.model_vocab.get(m, 0) for m in recent_models]
            sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.predictor.device)
            
            # Get predictions
            top_predictions = self.predictor.predict(sequence_tensor, top_k=n)
            
            current_time = time.time()
            for model_id, confidence in top_predictions:
                if model_id in self.reverse_vocab:
                    model_name = self.reverse_vocab[model_id]
                    predictions.append(PredictionResult(
                        model_name=model_name,
                        model_type=self._get_model_type(model_name),
                        confidence=float(confidence),
                        predicted_at=current_time
                    ))
        
        except Exception as e:
            print(f"[UsagePredictor] Transformer prediction error: {e}")
        
        return predictions
    
    def _markov_predict(self, recent_models: List[str], n: int) -> List[PredictionResult]:
        """Fallback Markov chain prediction"""
        predictions = []
        
        try:
            if not recent_models:
                return predictions
            
            last_model = recent_models[-1]
            transitions = self.transition_matrix.get(last_model, {})
            
            if not transitions:
                return predictions
            
            # Sort by frequency
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            total = sum(transitions.values())
            
            current_time = time.time()
            for model_name, count in sorted_transitions[:n]:
                confidence = count / total if total > 0 else 0
                predictions.append(PredictionResult(
                    model_name=model_name,
                    model_type=self._get_model_type(model_name),
                    confidence=confidence,
                    predicted_at=current_time
                ))
        
        except Exception as e:
            print(f"[UsagePredictor] Markov prediction error: {e}")
        
        return predictions
    
    def _get_model_type(self, model_name: str) -> ModelType:
        """Determine model type from name or cache"""
        # Check recent usage for this model
        for event in reversed(self.usage_history):
            if event.model_name == model_name:
                return event.model_type
        
        # Default to checkpoint
        return ModelType.CHECKPOINT
    
    def _process_preload_queue(self):
        """Process models in preload queue"""
        if self.is_preloading:
            return
        
        self.is_preloading = True
        
        try:
            while not self.preload_queue.empty():
                prediction = self.preload_queue.get_nowait()
                
                # Check if already cached
                if prediction.model_name in self.preload_cache:
                    continue
                
                # Preload the model
                self._preload_model(prediction)
                
                # Small delay between preloads
                time.sleep(0.1)
        
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[UsagePredictor] Preload processing error: {e}")
        finally:
            self.is_preloading = False
    
    def _preload_model(self, prediction: PredictionResult):
        """Preload a single model"""
        try:
            model_name = prediction.model_name
            model_type = prediction.model_type
            
            print(f"[UsagePredictor] Preloading {model_type.value}: {model_name} "
                  f"(confidence: {prediction.confidence:.2f})")
            
            # Load based on type
            if model_type == ModelType.CHECKPOINT:
                self._preload_checkpoint(model_name)
            elif model_type == ModelType.LORA:
                self._preload_lora(model_name)
            elif model_type == ModelType.LYCORIS:
                self._preload_lycoris(model_name)
            
            # Add to cache
            self.preload_cache.put(model_name, {
                'type': model_type,
                'loaded_at': time.time(),
                'prediction': prediction
            })
        
        except Exception as e:
            print(f"[UsagePredictor] Error preloading {prediction.model_name}: {e}")
    
    def _preload_checkpoint(self, model_name: str):
        """Preload a checkpoint model"""
        try:
            # Get model info
            model_info = sd_models.get_closet_checkpoint_match(model_name)
            if not model_info:
                return
            
            # Load to CPU first
            sd_models.load_model(model_info, already_loaded_state_dict={})
            
            # Move to GPU if available
            if hasattr(shared, 'sd_model') and shared.sd_model:
                if torch.cuda.is_available():
                    shared.sd_model = shared.sd_model.to('cuda')
        
        except Exception as e:
            print(f"[UsagePredictor] Error preloading checkpoint: {e}")
    
    def _preload_lora(self, lora_name: str):
        """Preload a LoRA model"""
        try:
            from modules import extra_networks
            extra_networks.activate('lora', [lora_name])
            # Immediately deactivate to keep in memory but not active
            extra_networks.deactivate('lora', [lora_name])
        except Exception as e:
            print(f"[UsagePredictor] Error preloading LoRA: {e}")
    
    def _preload_lycoris(self, lycoris_name: str):
        """Preload a LyCORIS model"""
        try:
            from modules import extra_networks
            extra_networks.activate('lyco', [lycoris_name])
            extra_networks.deactivate('lyco', [lycoris_name])
        except Exception as e:
            print(f"[UsagePredictor] Error preloading LyCORIS: {e}")
    
    def _train_model(self):
        """Train the transformer model on usage data"""
        if not self.predictor or len(self.usage_history) < 100:
            return
        
        print("[UsagePredictor] Training model...")
        timer = Timer()
        
        try:
            # Prepare training data
            sequences = []
            targets = []
            
            with self.data_lock:
                events = list(self.usage_history)
            
            # Create sequences
            for i in range(len(events) - 10):
                seq = [self.model_vocab.get(e.model_name, 0) for e in events[i:i+10]]
                target = self.model_vocab.get(events[i+10].model_name, 0)
                sequences.append(seq)
                targets.append(target)
            
            if not sequences:
                return
            
            # Convert to tensors
            seq_tensor = torch.tensor(sequences, dtype=torch.long).to(self.predictor.device)
            target_tensor = torch.tensor(targets, dtype=torch.long).to(self.predictor.device)
            
            # Train for a few epochs
            batch_size = 32
            num_batches = len(sequences) // batch_size
            
            for epoch in range(3):  # 3 epochs
                total_loss = 0
                indices = torch.randperm(len(sequences))
                
                for batch_idx in range(num_batches):
                    batch_indices = indices[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    batch_seq = seq_tensor[batch_indices]
                    batch_target = target_tensor[batch_indices]
                    
                    loss = self.predictor.train_step(batch_seq, batch_target)
                    total_loss += loss
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"[UsagePredictor] Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
            print(f"[UsagePredictor] Training completed in {timer.time():.2f}s")
        
        except Exception as e:
            print(f"[UsagePredictor] Training error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        with self.data_lock:
            stats = {
                'enabled': self.enabled,
                'session_id': self.session_id,
                'total_events': len(self.usage_history),
                'unique_models': len(self.model_vocab),
                'preload_cache_size': len(self.preload_cache),
                'queue_size': self.preload_queue.qsize(),
                'last_activity': self.last_activity_time,
                'idle_time': time.time() - self.last_activity_time
            }
            
            # Add top transitions
            if self.transition_matrix:
                top_transitions = []
                for from_model, to_models in list(self.transition_matrix.items())[:5]:
                    top_to = sorted(to_models.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_transitions.append({
                        'from': from_model,
                        'to': [{'model': m, 'count': c} for m, c in top_to]
                    })
                stats['top_transitions'] = top_transitions
            
            return stats
    
    def clear_cache(self):
        """Clear preload cache"""
        self.preload_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset(self):
        """Reset predictor state"""
        with self.data_lock:
            self.usage_history.clear()
            self.transition_matrix.clear()
            self.model_vocab.clear()
            self.reverse_vocab.clear()
            self.preload_cache.clear()
        
        if self.predictor:
            del self.predictor
            self.predictor = None
        
        self._initialize_predictor()
        print("[UsagePredictor] Reset complete")


# Global instance
usage_predictor = UsagePredictor()


# Public API functions
def record_model_usage(model_name: str, model_type: str = "checkpoint", **kwargs):
    """Record model usage (public API)"""
    try:
        model_type_enum = ModelType(model_type)
    except ValueError:
        model_type_enum = ModelType.CHECKPOINT
    
    usage_predictor.record_model_usage(model_name, model_type_enum, **kwargs)


def predict_next_models(n: int = 3) -> List[Dict[str, Any]]:
    """Get predictions (public API)"""
    predictions = usage_predictor.predict_next_models(n)
    return [
        {
            'model_name': p.model_name,
            'model_type': p.model_type.value,
            'confidence': p.confidence,
            'predicted_at': p.predicted_at
        }
        for p in predictions
    ]


def get_usage_statistics() -> Dict[str, Any]:
    """Get predictor statistics (public API)"""
    return usage_predictor.get_statistics()


def clear_preload_cache():
    """Clear preload cache (public API)"""
    usage_predictor.clear_cache()


def reset_predictor():
    """Reset predictor (public API)"""
    usage_predictor.reset()


# Settings integration
def on_ui_settings():
    """Add settings to UI"""
    from modules import shared
    
    section = ('usage_predictor', "Usage Predictor")
    shared.opts.add_option("usage_predictor_enabled", shared.OptionInfo(
        True, "Enable usage prediction and preloading", section=section))
    shared.opts.add_option("usage_predictor_cache_size", shared.OptionInfo(
        5, "Maximum models to keep in preload cache", section=section))
    shared.opts.add_option("usage_predictor_idle_threshold", shared.OptionInfo(
        2.0, "Idle time (seconds) before preloading", section=section))


# Hook into settings
from modules import script_callbacks
script_callbacks.on_ui_settings(on_ui_settings)


# Monkey-patch model loading to record usage
_original_load_model = sd_models.load_model


def _patched_load_model(*args, **kwargs):
    result = _original_load_model(*args, **kwargs)
    
    # Extract model name from args/kwargs
    model_name = None
    if args and hasattr(args[0], 'name'):
        model_name = args[0].name
    elif 'sd_model' in kwargs:
        model_name = kwargs['sd_model']
    
    if model_name:
        record_model_usage(model_name, "checkpoint")
    
    return result


# Apply monkey patch
sd_models.load_model = _patched_load_model


# Cleanup on exit
import atexit
atexit.register(usage_predictor._save_session_data)


print("[UsagePredictor] Module loaded successfully")