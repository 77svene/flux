"""
Real-Time Collaboration System for Stable Diffusion WebUI
Multi-user shared workspace with live parameter synchronization, collaborative prompting, and shared model pools
"""

import asyncio
import json
import threading
import time
import uuid
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets library not installed. Collaboration features disabled. Install with: pip install websockets")

from modules import shared, sd_models, sd_vae, sd_samplers, extra_networks
from modules.ui import create_refresh_button
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("collaboration")


class MessageType(Enum):
    """WebSocket message types"""
    JOIN = "join"
    LEAVE = "leave"
    PARAMETER_UPDATE = "parameter_update"
    PROMPT_UPDATE = "prompt_update"
    MODEL_UPDATE = "model_update"
    LORA_UPDATE = "lora_update"
    SAMPLER_UPDATE = "sampler_update"
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    CURSOR_POSITION = "cursor_position"
    SELECTION_UPDATE = "selection_update"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class CollaborationUser:
    """Represents a connected collaboration user"""
    user_id: str
    username: str
    color: str
    cursor_position: int = 0
    selection_start: int = 0
    selection_end: int = 0
    last_active: float = field(default_factory=time.time)
    is_host: bool = False
    permissions: Set[str] = field(default_factory=lambda: {"edit", "generate"})


@dataclass
class CollaborationSession:
    """Represents a collaboration session"""
    session_id: str
    host_user_id: str
    created_at: float
    users: Dict[str, CollaborationUser] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    positive_prompt: str = ""
    negative_prompt: str = ""
    prompt_history: List[Dict] = field(default_factory=list)
    max_users: int = 8
    is_locked: bool = False
    password: Optional[str] = None


class OperationalTransform:
    """Operational Transformation for concurrent text editing"""
    
    @staticmethod
    def transform_operations(op1: Dict, op2: Dict) -> tuple:
        """
        Transform two concurrent operations
        Returns transformed (op1', op2') that can be applied in any order
        """
        if op1["type"] == "insert" and op2["type"] == "insert":
            if op1["position"] < op2["position"]:
                return op1, {"type": "insert", "position": op2["position"] + len(op1["text"]), "text": op2["text"]}
            elif op1["position"] > op2["position"]:
                return {"type": "insert", "position": op1["position"] + len(op2["text"]), "text": op1["text"]}, op2
            else:
                # Same position - use user_id for tie-breaking
                if op1.get("user_id", "") < op2.get("user_id", ""):
                    return op1, {"type": "insert", "position": op2["position"] + len(op1["text"]), "text": op2["text"]}
                else:
                    return {"type": "insert", "position": op1["position"] + len(op2["text"]), "text": op1["text"]}, op2
        
        elif op1["type"] == "insert" and op2["type"] == "delete":
            if op1["position"] <= op2["position"]:
                return op1, {"type": "delete", "position": op2["position"] + len(op1["text"]), "length": op2["length"]}
            elif op1["position"] >= op2["position"] + op2["length"]:
                return {"type": "insert", "position": op1["position"] - op2["length"], "text": op1["text"]}, op2
            else:
                # Insert within delete range
                return {"type": "insert", "position": op2["position"], "text": op1["text"]}, op2
        
        elif op1["type"] == "delete" and op2["type"] == "insert":
            transformed_op2, transformed_op1 = OperationalTransform.transform_operations(op2, op1)
            return transformed_op1, transformed_op2
        
        elif op1["type"] == "delete" and op2["type"] == "delete":
            if op1["position"] >= op2["position"] + op2["length"]:
                return {"type": "delete", "position": op1["position"] - op2["length"], "length": op1["length"]}, op2
            elif op2["position"] >= op1["position"] + op1["length"]:
                return op1, {"type": "delete", "position": op2["position"] - op1["length"], "length": op2["length"]}
            else:
                # Overlapping deletes
                start1, end1 = op1["position"], op1["position"] + op1["length"]
                start2, end2 = op2["position"], op2["position"] + op2["length"]
                
                new_start1 = min(start1, start2)
                new_end1 = max(start1, start2)
                new_length1 = new_end1 - new_start1
                
                new_start2 = min(end1, end2)
                new_end2 = max(end1, end2)
                new_length2 = new_end2 - new_start2
                
                return (
                    {"type": "delete", "position": new_start1, "length": new_length1},
                    {"type": "delete", "position": new_start2 - new_length1, "length": new_length2}
                )
        
        return op1, op2


class CollaborationServer:
    """WebSocket server for real-time collaboration"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 7861):
        self.host = host
        self.port = port
        self.sessions: Dict[str, CollaborationSession] = {}
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.running = False
        self.server = None
        self.loop = None
        self.thread = None
        
        # Callbacks for UI integration
        self.on_parameter_update: Optional[Callable] = None
        self.on_prompt_update: Optional[Callable] = None
        self.on_user_joined: Optional[Callable] = None
        self.on_user_left: Optional[Callable] = None
        
        # User colors for visualization
        self.user_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
        ]
    
    async def start(self):
        """Start the WebSocket server"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start collaboration server: websockets library not available")
            return
        
        if self.running:
            return
        
        self.running = True
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            logger.info(f"Collaboration server started on ws://{self.host}:{self.port}")
            
            # Start heartbeat task
            asyncio.create_task(self.heartbeat_task())
            
        except Exception as e:
            logger.error(f"Failed to start collaboration server: {e}")
            self.running = False
    
    def start_in_thread(self):
        """Start server in a separate thread"""
        if self.thread and self.thread.is_alive():
            return
        
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.start())
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
    
    async def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        if self.loop:
            self.loop.stop()
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        user_id = str(uuid.uuid4())
        self.connections[user_id] = websocket
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(user_id, data)
                except json.JSONDecodeError:
                    await self.send_error(user_id, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.send_error(user_id, str(e))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for user {user_id}")
        finally:
            await self.handle_disconnect(user_id)
    
    async def process_message(self, user_id: str, data: Dict):
        """Process incoming WebSocket message"""
        message_type = data.get("type")
        session_id = self.user_sessions.get(user_id)
        
        if message_type == MessageType.JOIN.value:
            await self.handle_join(user_id, data)
        
        elif message_type == MessageType.LEAVE.value:
            await self.handle_leave(user_id)
        
        elif message_type == MessageType.SYNC_REQUEST.value:
            await self.handle_sync_request(user_id, session_id)
        
        elif message_type == MessageType.PARAMETER_UPDATE.value:
            await self.handle_parameter_update(user_id, session_id, data)
        
        elif message_type == MessageType.PROMPT_UPDATE.value:
            await self.handle_prompt_update(user_id, session_id, data)
        
        elif message_type == MessageType.MODEL_UPDATE.value:
            await self.handle_model_update(user_id, session_id, data)
        
        elif message_type == MessageType.LORA_UPDATE.value:
            await self.handle_lora_update(user_id, session_id, data)
        
        elif message_type == MessageType.SAMPLER_UPDATE.value:
            await self.handle_sampler_update(user_id, session_id, data)
        
        elif message_type == MessageType.GENERATION_START.value:
            await self.handle_generation_start(user_id, session_id, data)
        
        elif message_type == MessageType.GENERATION_COMPLETE.value:
            await self.handle_generation_complete(user_id, session_id, data)
        
        elif message_type == MessageType.CURSOR_POSITION.value:
            await self.handle_cursor_update(user_id, session_id, data)
        
        elif message_type == MessageType.SELECTION_UPDATE.value:
            await self.handle_selection_update(user_id, session_id, data)
        
        elif message_type == MessageType.HEARTBEAT.value:
            await self.handle_heartbeat(user_id)
    
    async def handle_join(self, user_id: str, data: Dict):
        """Handle user joining a session"""
        session_id = data.get("session_id")
        username = data.get("username", f"User_{user_id[:4]}")
        password = data.get("password")
        
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = CollaborationSession(
                session_id=session_id,
                host_user_id=user_id,
                created_at=time.time()
            )
            logger.info(f"Created new session: {session_id}")
        
        session = self.sessions[session_id]
        
        # Check password if set
        if session.password and session.password != password:
            await self.send_error(user_id, "Invalid session password")
            return
        
        # Check max users
        if len(session.users) >= session.max_users:
            await self.send_error(user_id, "Session is full")
            return
        
        # Assign color to user
        color_index = len(session.users) % len(self.user_colors)
        color = self.user_colors[color_index]
        
        # Add user to session
        user = CollaborationUser(
            user_id=user_id,
            username=username,
            color=color,
            is_host=(user_id == session.host_user_id)
        )
        session.users[user_id] = user
        self.user_sessions[user_id] = session_id
        
        # Send sync response to new user
        await self.send_sync_response(user_id, session)
        
        # Notify other users
        await self.broadcast_to_session(session_id, {
            "type": MessageType.JOIN.value,
            "user": asdict(user),
            "users": [asdict(u) for u in session.users.values()]
        }, exclude_user=user_id)
        
        # Trigger callback
        if self.on_user_joined:
            self.on_user_joined(session_id, user)
        
        logger.info(f"User {username} joined session {session_id}")
    
    async def handle_leave(self, user_id: str):
        """Handle user leaving a session"""
        session_id = self.user_sessions.get(user_id)
        if not session_id:
            return
        
        await self.remove_user_from_session(user_id, session_id)
    
    async def handle_disconnect(self, user_id: str):
        """Handle user disconnection"""
        session_id = self.user_sessions.get(user_id)
        if session_id:
            await self.remove_user_from_session(user_id, session_id)
        
        if user_id in self.connections:
            del self.connections[user_id]
    
    async def remove_user_from_session(self, user_id: str, session_id: str):
        """Remove user from session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        user = session.users.get(user_id)
        
        if user:
            # Notify other users
            await self.broadcast_to_session(session_id, {
                "type": MessageType.LEAVE.value,
                "user_id": user_id,
                "username": user.username,
                "users": [asdict(u) for u in session.users.values() if u.user_id != user_id]
            })
            
            # Remove user
            del session.users[user_id]
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            
            # Trigger callback
            if self.on_user_left:
                self.on_user_left(session_id, user)
            
            logger.info(f"User {user.username} left session {session_id}")
            
            # Delete session if empty
            if not session.users:
                del self.sessions[session_id]
                logger.info(f"Deleted empty session: {session_id}")
    
    async def handle_sync_request(self, user_id: str, session_id: str):
        """Handle sync request from client"""
        if not session_id or session_id not in self.sessions:
            await self.send_error(user_id, "Session not found")
            return
        
        await self.send_sync_response(user_id, self.sessions[session_id])
    
    async def send_sync_response(self, user_id: str, session: CollaborationSession):
        """Send full session state to user"""
        await self.send_to_user(user_id, {
            "type": MessageType.SYNC_RESPONSE.value,
            "session_id": session.session_id,
            "users": [asdict(u) for u in session.users.values()],
            "parameters": session.parameters,
            "positive_prompt": session.positive_prompt,
            "negative_prompt": session.negative_prompt,
            "prompt_history": session.prompt_history[-10:],  # Last 10 changes
            "is_locked": session.is_locked
        })
    
    async def handle_parameter_update(self, user_id: str, session_id: str, data: Dict):
        """Handle parameter update"""
        if not self.validate_session(user_id, session_id):
            return
        
        session = self.sessions[session_id]
        user = session.users.get(user_id)
        
        if not user or "edit" not in user.permissions:
            await self.send_error(user_id, "No edit permission")
            return
        
        parameter = data.get("parameter")
        value = data.get("value")
        
        if parameter and value is not None:
            session.parameters[parameter] = value
            
            # Broadcast to other users
            await self.broadcast_to_session(session_id, {
                "type": MessageType.PARAMETER_UPDATE.value,
                "parameter": parameter,
                "value": value,
                "user_id": user_id,
                "username": user.username,
                "timestamp": time.time()
            }, exclude_user=user_id)
            
            # Trigger callback
            if self.on_parameter_update:
                self.on_parameter_update(session_id, parameter, value, user)
    
    async def handle_prompt_update(self, user_id: str, session_id: str, data: Dict):
        """Handle prompt update with operational transformation"""
        if not self.validate_session(user_id, session_id):
            return
        
        session = self.sessions[session_id]
        user = session.users.get(user_id)
        
        if not user or "edit" not in user.permissions:
            await self.send_error(user_id, "No edit permission")
            return
        
        prompt_type = data.get("prompt_type", "positive")
        operation = data.get("operation")
        
        if not operation:
            # Simple full update
            new_text = data.get("text", "")
            if prompt_type == "positive":
                session.positive_prompt = new_text
            else:
                session.negative_prompt = new_text
        else:
            # Operational transformation
            if prompt_type == "positive":
                current_text = session.positive_prompt
            else:
                current_text = session.negative_prompt
            
            # Apply operation
            if operation["type"] == "insert":
                pos = min(operation["position"], len(current_text))
                new_text = current_text[:pos] + operation["text"] + current_text[pos:]
            elif operation["type"] == "delete":
                start = operation["position"]
                end = min(start + operation["length"], len(current_text))
                new_text = current_text[:start] + current_text[end:]
            else:
                new_text = current_text
            
            if prompt_type == "positive":
                session.positive_prompt = new_text
            else:
                session.negative_prompt = new_text
            
            # Store in history
            session.prompt_history.append({
                "timestamp": time.time(),
                "user_id": user_id,
                "username": user.username,
                "prompt_type": prompt_type,
                "operation": operation,
                "result_length": len(new_text)
            })
        
        # Broadcast to other users
        await self.broadcast_to_session(session_id, {
            "type": MessageType.PROMPT_UPDATE.value,
            "prompt_type": prompt_type,
            "operation": operation,
            "text": new_text if not operation else None,
            "user_id": user_id,
            "username": user.username,
            "timestamp": time.time()
        }, exclude_user=user_id)
        
        # Trigger callback
        if self.on_prompt_update:
            self.on_prompt_update(session_id, prompt_type, new_text, user)
    
    async def handle_model_update(self, user_id: str, session_id: str, data: Dict):
        """Handle model selection update"""
        if not self.validate_session(user_id, session_id):
            return
        
        session = self.sessions[session_id]
        model_name = data.get("model_name")
        
        if model_name:
            session.parameters["sd_model_checkpoint"] = model_name
            
            await self.broadcast_to_session(session_id, {
                "type": MessageType.MODEL_UPDATE.value,
                "model_name": model_name,
                "user_id": user_id,
                "timestamp": time.time()
            }, exclude_user=user_id)
    
    async def handle_lora_update(self, user_id: str, session_id: str, data: Dict):
        """Handle LoRA update"""
        if not self.validate_session(user_id, session_id):
            return
        
        session = self.sessions[session_id]
        lora_name = data.get("lora_name")
        lora_weight = data.get("lora_weight", 1.0)
        action = data.get("action", "add")  # add, remove, update
        
        if lora_name:
            loras = session.parameters.get("loras", [])
            
            if action == "add":
                # Check if already exists
                existing = next((l for l in loras if l["name"] == lora_name), None)
                if existing:
                    existing["weight"] = lora_weight
                else:
                    loras.append({"name": lora_name, "weight": lora_weight})
            
            elif action == "remove":
                loras = [l for l in loras if l["name"] != lora_name]
            
            elif action == "update":
                for lora in loras:
                    if lora["name"] == lora_name:
                        lora["weight"] = lora_weight
                        break
            
            session.parameters["loras"] = loras
            
            await self.broadcast_to_session(session_id, {
                "type": MessageType.LORA_UPDATE.value,
                "loras": loras,
                "user_id": user_id,
                "timestamp": time.time()
            }, exclude_user=user_id)
    
    async def handle_sampler_update(self, user_id: str, session_id: str, data: Dict):
        """Handle sampler update"""
        if not self.validate_session(user_id, session_id):
            return
        
        session = self.sessions[session_id]
        sampler_name = data.get("sampler_name")
        steps = data.get("steps")
        cfg_scale = data.get("cfg_scale")
        
        if sampler_name:
            session.parameters["sampler_name"] = sampler_name
        if steps:
            session.parameters["steps"] = steps
        if cfg_scale:
            session.parameters["cfg_scale"] = cfg_scale
        
        await self.broadcast_to_session(session_id, {
            "type": MessageType.SAMPLER_UPDATE.value,
            "sampler_name": sampler_name,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "user_id": user_id,
            "timestamp": time.time()
        }, exclude_user=user_id)
    
    async def handle_generation_start(self, user_id: str, session_id: str, data: Dict):
        """Handle generation start notification"""
        if not self.validate_session(user_id, session_id):
            return
        
        session = self.sessions[session_id]
        user = session.users.get(user_id)
        
        if not user or "generate" not in user.permissions:
            await self.send_error(user_id, "No generate permission")
            return
        
        await self.broadcast_to_session(session_id, {
            "type": MessageType.GENERATION_START.value,
            "user_id": user_id,
            "username": user.username,
            "timestamp": time.time()
        })
    
    async def handle_generation_complete(self, user_id: str, session_id: str, data: Dict):
        """Handle generation complete notification"""
        if not self.validate_session(user_id, session_id):
            return
        
        await self.broadcast_to_session(session_id, {
            "type": MessageType.GENERATION_COMPLETE.value,
            "user_id": user_id,
            "image_info": data.get("image_info"),
            "timestamp": time.time()
        })
    
    async def handle_cursor_update(self, user_id: str, session_id: str, data: Dict):
        """Handle cursor position update"""
        if not self.validate_session(user_id, session_id):
            return
        
        session = self.sessions[session_id]
        user = session.users.get(user_id)
        
        if user:
            cursor_position = data.get("position", 0)
            user.cursor_position = cursor_position
            user.last_active = time.time()
            
            await self.broadcast_to_session(session_id, {
                "type": MessageType.CURSOR_POSITION.value,
                "user_id": user_id,
                "position": cursor_position,
                "color": user.color
            }, exclude_user=user_id)
    
    async def handle_selection_update(self, user_id: str, session_id: str, data: Dict):
        """Handle text selection update"""
        if not self.validate_session(user_id, session_id):
            return
        
        session = self.sessions[session_id]
        user = session.users.get(user_id)
        
        if user:
            selection_start = data.get("start", 0)
            selection_end = data.get("end", 0)
            user.selection_start = selection_start
            user.selection_end = selection_end
            user.last_active = time.time()
            
            await self.broadcast_to_session(session_id, {
                "type": MessageType.SELECTION_UPDATE.value,
                "user_id": user_id,
                "start": selection_start,
                "end": selection_end,
                "color": user.color
            }, exclude_user=user_id)
    
    async def handle_heartbeat(self, user_id: str):
        """Handle heartbeat from client"""
        session_id = self.user_sessions.get(user_id)
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            user = session.users.get(user_id)
            if user:
                user.last_active = time.time()
        
        await self.send_to_user(user_id, {
            "type": MessageType.HEARTBEAT.value,
            "timestamp": time.time()
        })
    
    async def heartbeat_task(self):
        """Send periodic heartbeats and clean up inactive users"""
        while self.running:
            await asyncio.sleep(60)  # Check every minute
            
            current_time = time.time()
            inactive_threshold = 300  # 5 minutes
            
            for session_id in list(self.sessions.keys()):
                session = self.sessions[session_id]
                inactive_users = []
                
                for user_id, user in list(session.users.items()):
                    if current_time - user.last_active > inactive_threshold:
                        inactive_users.append(user_id)
                
                for user_id in inactive_users:
                    logger.info(f"Removing inactive user {user_id} from session {session_id}")
                    await self.remove_user_from_session(user_id, session_id)
    
    def validate_session(self, user_id: str, session_id: str) -> bool:
        """Validate user belongs to session"""
        if not session_id or session_id not in self.sessions:
            return False
        
        if user_id not in self.user_sessions:
            return False
        
        return self.user_sessions[user_id] == session_id
    
    async def send_to_user(self, user_id: str, data: Dict):
        """Send message to specific user"""
        if user_id in self.connections:
            try:
                await self.connections[user_id].send(json.dumps(data))
            except websockets.exceptions.ConnectionClosed:
                await self.handle_disconnect(user_id)
    
    async def send_error(self, user_id: str, message: str):
        """Send error message to user"""
        await self.send_to_user(user_id, {
            "type": MessageType.ERROR.value,
            "message": message
        })
    
    async def broadcast_to_session(self, session_id: str, data: Dict, exclude_user: str = None):
        """Broadcast message to all users in session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        message = json.dumps(data)
        
        for user_id in list(session.users.keys()):
            if user_id != exclude_user and user_id in self.connections:
                try:
                    await self.connections[user_id].send(message)
                except websockets.exceptions.ConnectionClosed:
                    await self.handle_disconnect(user_id)
    
    def create_session(self, password: str = None) -> str:
        """Create a new collaboration session"""
        session_id = str(uuid.uuid4())[:8]
        return session_id
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "user_count": len(session.users),
            "max_users": session.max_users,
            "created_at": session.created_at,
            "has_password": bool(session.password)
        }


class CollaborationClient:
    """Client for connecting to collaboration server"""
    
    def __init__(self, server_url: str = "ws://localhost:7861"):
        self.server_url = server_url
        self.websocket = None
        self.user_id = None
        self.session_id = None
        self.connected = False
        self.callbacks: Dict[str, List[Callable]] = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Start connection in background
        self.connection_thread = None
        self.loop = None
    
    def connect(self, session_id: str, username: str, password: str = None):
        """Connect to collaboration session"""
        if self.connected:
            return
        
        self.session_id = session_id
        
        def run_connection():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._connect(session_id, username, password))
        
        self.connection_thread = threading.Thread(target=run_connection, daemon=True)
        self.connection_thread.start()
    
    async def _connect(self, session_id: str, username: str, password: str = None):
        """Async connection method"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.websocket = websocket
                self.connected = True
                self.reconnect_attempts = 0
                
                # Send join message
                await self.send({
                    "type": MessageType.JOIN.value,
                    "session_id": session_id,
                    "username": username,
                    "password": password
                })
                
                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.handle_message(data)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON received")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            self.connected = False
            self.websocket = None
            
            # Attempt reconnect
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                logger.info(f"Attempting reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
                await asyncio.sleep(2)
                await self._connect(session_id, username, password)
    
    async def handle_message(self, data: Dict):
        """Handle incoming message"""
        message_type = data.get("type")
        
        # Trigger callbacks
        if message_type in self.callbacks:
            for callback in self.callbacks[message_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    async def send(self, data: Dict):
        """Send message to server"""
        if self.websocket and self.connected:
            try:
                await self.websocket.send(json.dumps(data))
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
    
    def send_sync(self, data: Dict):
        """Synchronous send wrapper"""
        if self.loop and self.connected:
            asyncio.run_coroutine_threadsafe(self.send(data), self.loop)
    
    def on(self, message_type: str, callback: Callable):
        """Register callback for message type"""
        if message_type not in self.callbacks:
            self.callbacks[message_type] = []
        self.callbacks[message_type].append(callback)
    
    def update_parameter(self, parameter: str, value: Any):
        """Update a parameter"""
        self.send_sync({
            "type": MessageType.PARAMETER_UPDATE.value,
            "parameter": parameter,
            "value": value
        })
    
    def update_prompt(self, prompt_type: str, operation: Dict = None, text: str = None):
        """Update prompt with operational transformation"""
        message = {
            "type": MessageType.PROMPT_UPDATE.value,
            "prompt_type": prompt_type
        }
        
        if operation:
            message["operation"] = operation
        elif text is not None:
            message["text"] = text
        
        self.send_sync(message)
    
    def update_model(self, model_name: str):
        """Update model selection"""
        self.send_sync({
            "type": MessageType.MODEL_UPDATE.value,
            "model_name": model_name
        })
    
    def update_lora(self, lora_name: str, weight: float = 1.0, action: str = "add"):
        """Update LoRA"""
        self.send_sync({
            "type": MessageType.LORA_UPDATE.value,
            "lora_name": lora_name,
            "lora_weight": weight,
            "action": action
        })
    
    def update_sampler(self, sampler_name: str = None, steps: int = None, cfg_scale: float = None):
        """Update sampler settings"""
        message = {"type": MessageType.SAMPLER_UPDATE.value}
        if sampler_name:
            message["sampler_name"] = sampler_name
        if steps:
            message["steps"] = steps
        if cfg_scale:
            message["cfg_scale"] = cfg_scale
        
        self.send_sync(message)
    
    def update_cursor(self, position: int):
        """Update cursor position"""
        self.send_sync({
            "type": MessageType.CURSOR_POSITION.value,
            "position": position
        })
    
    def update_selection(self, start: int, end: int):
        """Update text selection"""
        self.send_sync({
            "type": MessageType.SELECTION_UPDATE.value,
            "start": start,
            "end": end
        })
    
    def notify_generation_start(self):
        """Notify that generation has started"""
        self.send_sync({
            "type": MessageType.GENERATION_START.value
        })
    
    def notify_generation_complete(self, image_info: Dict = None):
        """Notify that generation has completed"""
        self.send_sync({
            "type": MessageType.GENERATION_COMPLETE.value,
            "image_info": image_info
        })
    
    def request_sync(self):
        """Request full state synchronization"""
        self.send_sync({
            "type": MessageType.SYNC_REQUEST.value
        })
    
    def disconnect(self):
        """Disconnect from session"""
        if self.loop and self.connected:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.connected = False


# Global instances
collaboration_server = CollaborationServer()
collaboration_client = CollaborationClient()


def setup_collaboration_ui():
    """Setup collaboration UI components in Gradio"""
    with gr.Accordion("🌐 Real-Time Collaboration", open=False):
        with gr.Row():
            session_id_input = gr.Textbox(
                label="Session ID",
                placeholder="Enter session ID or create new",
                scale=3
            )
            create_session_btn = gr.Button("Create New", scale=1)
            join_session_btn = gr.Button("Join", scale=1)
        
        with gr.Row():
            username_input = gr.Textbox(
                label="Username",
                placeholder="Your display name",
                value=f"User_{str(uuid.uuid4())[:4]}"
            )
            password_input = gr.Textbox(
                label="Session Password (optional)",
                type="password"
            )
        
        with gr.Row():
            connection_status = gr.HTML(
                value="<div style='color: #666;'>Not connected</div>",
                label="Status"
            )
            user_count = gr.Number(
                label="Users in Session",
                value=0,
                interactive=False
            )
        
        with gr.Row():
            share_link = gr.Textbox(
                label="Share Link",
                interactive=False,
                visible=False
            )
            copy_link_btn = gr.Button("Copy Link", visible=False)
        
        # Collaboration settings
        with gr.Row():
            auto_sync = gr.Checkbox(
                label="Auto-sync parameters",
                value=True
            )
            show_cursors = gr.Checkbox(
                label="Show other users' cursors",
                value=True
            )
        
        # User list
        user_list = gr.Dataframe(
            headers=["Username", "Status", "Color"],
            datatype=["str", "str", "str"],
            label="Connected Users",
            interactive=False
        )
    
    return {
        "session_id_input": session_id_input,
        "create_session_btn": create_session_btn,
        "join_session_btn": join_session_btn,
        "username_input": username_input,
        "password_input": password_input,
        "connection_status": connection_status,
        "user_count": user_count,
        "share_link": share_link,
        "copy_link_btn": copy_link_btn,
        "auto_sync": auto_sync,
        "show_cursors": show_cursors,
        "user_list": user_list
    }


def integrate_with_existing_ui():
    """Integrate collaboration features with existing UI elements"""
    # This function would be called during UI setup
    # It would add collaboration controls to existing parameter groups
    
    # Example: Add collaboration indicator to prompt fields
    def add_collaboration_indicator(component, param_name):
        # Add visual indicator for collaborative editing
        original_change = component.change
        
        def new_change(fn, inputs, outputs):
            def wrapped_fn(*args):
                # Send update to collaboration server
                if hasattr(collaboration_client, 'connected') and collaboration_client.connected:
                    collaboration_client.update_parameter(param_name, args[0] if args else None)
                return fn(*args)
            
            return original_change(wrapped_fn, inputs, outputs)
        
        component.change = new_change
    
    return add_collaboration_indicator


# Utility functions for external use
def start_collaboration_server(host: str = "0.0.0.0", port: int = 7861):
    """Start the collaboration server"""
    collaboration_server.host = host
    collaboration_server.port = port
    collaboration_server.start_in_thread()


def get_collaboration_server():
    """Get the collaboration server instance"""
    return collaboration_server


def get_collaboration_client():
    """Get the collaboration client instance"""
    return collaboration_client


def create_collaboration_session(password: str = None) -> str:
    """Create a new collaboration session"""
    return collaboration_server.create_session(password)


# Export main classes and functions
__all__ = [
    'CollaborationServer',
    'CollaborationClient',
    'CollaborationSession',
    'CollaborationUser',
    'MessageType',
    'OperationalTransform',
    'setup_collaboration_ui',
    'integrate_with_existing_ui',
    'start_collaboration_server',
    'get_collaboration_server',
    'get_collaboration_client',
    'create_collaboration_session',
    'collaboration_server',
    'collaboration_client'
]