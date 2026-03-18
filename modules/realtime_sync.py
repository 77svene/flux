"""
Real-Time Collaboration System for Stable Diffusion WebUI
Multi-user shared workspace with live parameter synchronization, collaborative prompting, and shared model pools.
WebSocket-based synchronization with operational transformation for concurrent editing.
"""

import asyncio
import json
import time
import uuid
import threading
import logging
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import websockets
from websockets.server import WebSocketServerProtocol

# Import from existing modules
from modules import shared, sd_models, sd_samplers, scripts, ui_common
from modules.processing import StableDiffusionProcessing, Processed
from modules.shared import opts, cmd_opts

logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Types of collaborative operations"""
    PARAM_UPDATE = "param_update"
    PROMPT_UPDATE = "prompt_update"
    NEGATIVE_PROMPT_UPDATE = "negative_prompt_update"
    SEED_UPDATE = "seed_update"
    MODEL_CHANGE = "model_change"
    SAMPLER_CHANGE = "sampler_change"
    STEPS_UPDATE = "steps_update"
    CFG_UPDATE = "cfg_update"
    SIZE_UPDATE = "size_update"
    BATCH_UPDATE = "batch_update"
    LORA_UPDATE = "lora_update"
    EMBEDDING_UPDATE = "embedding_update"
    GENERATION_START = "generation_start"
    GENERATION_PROGRESS = "generation_progress"
    GENERATION_COMPLETE = "generation_complete"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    CURSOR_POSITION = "cursor_position"
    SELECTION_RANGE = "selection_range"
    CHAT_MESSAGE = "chat_message"
    WORKSPACE_SYNC = "workspace_sync"
    CONFLICT_RESOLUTION = "conflict_resolution"

@dataclass
class Operation:
    """Represents a single collaborative operation"""
    id: str
    type: OperationType
    user_id: str
    timestamp: float
    target: str  # Parameter name or target component
    value: Any
    version: int = 0
    parent_version: int = 0
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert operation to dictionary for serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Operation':
        """Create operation from dictionary"""
        data['type'] = OperationType(data['type'])
        return cls(**data)

class OperationalTransform:
    """Operational Transformation engine for conflict resolution"""
    
    @staticmethod
    def transform(op1: Operation, op2: Operation) -> tuple:
        """
        Transform two concurrent operations against each other.
        Returns transformed (op1', op2') where op1' can be applied after op2.
        """
        # Same target parameter - need to resolve conflict
        if op1.target == op2.target:
            # For numeric parameters, use latest timestamp wins
            if op1.type in [OperationType.PARAM_UPDATE, OperationType.CFG_UPDATE, 
                           OperationType.STEPS_UPDATE, OperationType.SIZE_UPDATE]:
                if op1.timestamp > op2.timestamp:
                    return op1, None
                else:
                    return None, op2
            
            # For text parameters (prompts), use operational transformation
            elif op1.type in [OperationType.PROMPT_UPDATE, OperationType.NEGATIVE_PROMPT_UPDATE]:
                return OperationalTransform.transform_text(op1, op2)
        
        # Different targets - no conflict
        return op1, op2
    
    @staticmethod
    def transform_text(op1: Operation, op2: Operation) -> tuple:
        """Transform text operations using character-level OT"""
        if not isinstance(op1.value, dict) or not isinstance(op2.value, dict):
            return op1, op2
        
        # Simple character-level transformation
        # In production, you'd want a more sophisticated algorithm
        pos1 = op1.value.get('position', 0)
        pos2 = op2.value.get('position', 0)
        
        # If operations are at different positions, no conflict
        if abs(pos1 - pos2) > 10:  # Threshold for "same area"
            return op1, op2
        
        # Adjust positions based on insertions/deletions
        if op1.value.get('type') == 'insert' and op2.value.get('type') == 'insert':
            if pos2 <= pos1:
                op1.value['position'] = pos1 + len(op2.value.get('text', ''))
            return op1, op2
        
        # For simplicity, last-write-wins for complex transformations
        if op1.timestamp > op2.timestamp:
            return op1, None
        else:
            return None, op2

class UserSession:
    """Represents a connected user session"""
    
    def __init__(self, user_id: str, websocket: WebSocketServerProtocol, username: str = None):
        self.user_id = user_id
        self.websocket = websocket
        self.username = username or f"User_{user_id[:8]}"
        self.connected_at = time.time()
        self.last_active = time.time()
        self.color = self._generate_color()
        self.cursor_position = 0
        self.selection_start = 0
        self.selection_end = 0
        self.current_tab = "txt2img"
        
    def _generate_color(self) -> str:
        """Generate a unique color for the user"""
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
                 "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]
        return colors[hash(self.user_id) % len(colors)]
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'color': self.color,
            'connected_at': self.connected_at,
            'current_tab': self.current_tab,
            'cursor_position': self.cursor_position,
            'selection': [self.selection_start, self.selection_end]
        }

class SharedWorkspace:
    """Manages the shared collaborative workspace state"""
    
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.created_at = time.time()
        self.version = 0
        self.users: Dict[str, UserSession] = {}
        self.operation_history: List[Operation] = []
        self.state = {
            'prompt': '',
            'negative_prompt': '',
            'seed': -1,
            'sampler_name': 'Euler a',
            'steps': 20,
            'cfg_scale': 7.0,
            'width': 512,
            'height': 512,
            'batch_size': 1,
            'model': None,
            'vae': None,
            'lora': [],
            'embedding': [],
            'controlnet': [],
            'generation_active': False,
            'progress': 0,
            'current_image': None
        }
        self.lock = threading.RLock()
        
    def apply_operation(self, operation: Operation) -> bool:
        """Apply an operation to the workspace state"""
        with self.lock:
            # Check for conflicts with recent operations
            recent_ops = [op for op in self.operation_history[-50:] 
                         if op.target == operation.target and op.user_id != operation.user_id]
            
            for recent_op in recent_ops:
                transformed_op, transformed_recent = OperationalTransform.transform(operation, recent_op)
                if transformed_op is None:
                    return False  # Operation was superseded
                operation = transformed_op
            
            # Apply the operation
            if operation.type == OperationType.PROMPT_UPDATE:
                if isinstance(operation.value, dict):
                    # Handle incremental update
                    pos = operation.value.get('position', 0)
                    text = operation.value.get('text', '')
                    action = operation.value.get('action', 'insert')
                    
                    current = self.state['prompt']
                    if action == 'insert':
                        self.state['prompt'] = current[:pos] + text + current[pos:]
                    elif action == 'delete':
                        length = operation.value.get('length', 1)
                        self.state['prompt'] = current[:pos] + current[pos+length:]
                else:
                    # Full replacement
                    self.state['prompt'] = str(operation.value)
                    
            elif operation.type == OperationType.NEGATIVE_PROMPT_UPDATE:
                self.state['negative_prompt'] = str(operation.value)
                
            elif operation.type == OperationType.SEED_UPDATE:
                self.state['seed'] = int(operation.value)
                
            elif operation.type == OperationType.MODEL_CHANGE:
                self.state['model'] = operation.value
                # Trigger model load if different
                if sd_models.model_data.sd_model and sd_models.model_data.sd_model.sd_model_checkpoint != operation.value:
                    sd_models.load_model(operation.value)
                    
            elif operation.type == OperationType.SAMPLER_CHANGE:
                self.state['sampler_name'] = operation.value
                
            elif operation.type == OperationType.STEPS_UPDATE:
                self.state['steps'] = int(operation.value)
                
            elif operation.type == OperationType.CFG_UPDATE:
                self.state['cfg_scale'] = float(operation.value)
                
            elif operation.type == OperationType.SIZE_UPDATE:
                if isinstance(operation.value, dict):
                    self.state['width'] = int(operation.value.get('width', 512))
                    self.state['height'] = int(operation.value.get('height', 512))
                    
            elif operation.type == OperationType.BATCH_UPDATE:
                self.state['batch_size'] = int(operation.value)
                
            elif operation.type == OperationType.LORA_UPDATE:
                if operation.value.get('action') == 'add':
                    if operation.value['name'] not in self.state['lora']:
                        self.state['lora'].append(operation.value['name'])
                elif operation.value.get('action') == 'remove':
                    if operation.value['name'] in self.state['lora']:
                        self.state['lora'].remove(operation.value['name'])
                        
            elif operation.type == OperationType.GENERATION_START:
                self.state['generation_active'] = True
                self.state['progress'] = 0
                
            elif operation.type == OperationType.GENERATION_PROGRESS:
                self.state['progress'] = float(operation.value)
                
            elif operation.type == OperationType.GENERATION_COMPLETE:
                self.state['generation_active'] = False
                self.state['progress'] = 100
                self.state['current_image'] = operation.value
            
            # Update version and history
            self.version += 1
            operation.version = self.version
            self.operation_history.append(operation)
            
            # Trim history to prevent memory issues
            if len(self.operation_history) > 1000:
                self.operation_history = self.operation_history[-500:]
            
            return True
    
    def get_full_state(self) -> Dict:
        """Get complete workspace state for synchronization"""
        with self.lock:
            return {
                'workspace_id': self.workspace_id,
                'version': self.version,
                'state': self.state.copy(),
                'users': [user.to_dict() for user in self.users.values()],
                'timestamp': time.time()
            }
    
    def add_user(self, user: UserSession):
        """Add a user to the workspace"""
        with self.lock:
            self.users[user.user_id] = user
            
    def remove_user(self, user_id: str):
        """Remove a user from the workspace"""
        with self.lock:
            if user_id in self.users:
                del self.users[user_id]

class RealtimeSyncServer:
    """WebSocket server for real-time collaboration"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 7861):
        self.host = host
        self.port = port
        self.workspaces: Dict[str, SharedWorkspace] = {}
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.user_sessions: Dict[str, UserSession] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        self.server = None
        
        # Register default handlers
        self._register_handlers()
        
    def _register_handlers(self):
        """Register message handlers"""
        self.message_handlers = {
            'join_workspace': self._handle_join_workspace,
            'leave_workspace': self._handle_leave_workspace,
            'operation': self._handle_operation,
            'sync_request': self._handle_sync_request,
            'cursor_update': self._handle_cursor_update,
            'chat_message': self._handle_chat_message,
            'generate': self._handle_generate,
            'interrupt': self._handle_interrupt,
            'progress_sync': self._handle_progress_sync
        }
    
    async def start(self):
        """Start the WebSocket server"""
        self.running = True
        logger.info(f"Starting RealtimeSync server on {self.host}:{self.port}")
        
        try:
            self.server = await websockets.serve(
                self._connection_handler,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            logger.info(f"RealtimeSync server started on ws://{self.host}:{self.port}")
            
            # Keep server running
            await self.server.wait_closed()
        except Exception as e:
            logger.error(f"Failed to start RealtimeSync server: {e}")
            self.running = False
    
    async def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("RealtimeSync server stopped")
    
    async def _connection_handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connections"""
        user_id = str(uuid.uuid4())
        self.connections[user_id] = websocket
        
        logger.info(f"New connection: {user_id}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(user_id, data, websocket)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self._send_error(websocket, str(e))
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {user_id}")
        finally:
            # Clean up on disconnect
            await self._cleanup_user(user_id)
    
    async def _process_message(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Process incoming messages"""
        message_type = data.get('type')
        
        if message_type in self.message_handlers:
            await self.message_handlers[message_type](user_id, data, websocket)
        else:
            await self._send_error(websocket, f"Unknown message type: {message_type}")
    
    async def _handle_join_workspace(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle user joining a workspace"""
        workspace_id = data.get('workspace_id', 'default')
        username = data.get('username', f'User_{user_id[:8]}')
        
        # Create workspace if it doesn't exist
        if workspace_id not in self.workspaces:
            self.workspaces[workspace_id] = SharedWorkspace(workspace_id)
        
        workspace = self.workspaces[workspace_id]
        
        # Create user session
        user_session = UserSession(user_id, websocket, username)
        self.user_sessions[user_id] = user_session
        workspace.add_user(user_session)
        
        # Send current state to new user
        state = workspace.get_full_state()
        await websocket.send(json.dumps({
            'type': 'workspace_joined',
            'workspace_id': workspace_id,
            'user_id': user_id,
            'state': state
        }))
        
        # Notify other users
        join_operation = Operation(
            id=str(uuid.uuid4()),
            type=OperationType.USER_JOIN,
            user_id=user_id,
            timestamp=time.time(),
            target='users',
            value=user_session.to_dict()
        )
        
        await self._broadcast_to_workspace(workspace_id, join_operation.to_dict(), exclude_user=user_id)
        
        logger.info(f"User {username} ({user_id}) joined workspace {workspace_id}")
    
    async def _handle_leave_workspace(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle user leaving a workspace"""
        workspace_id = data.get('workspace_id')
        
        if workspace_id in self.workspaces:
            workspace = self.workspaces[workspace_id]
            workspace.remove_user(user_id)
            
            # Notify other users
            leave_operation = Operation(
                id=str(uuid.uuid4()),
                type=OperationType.USER_LEAVE,
                user_id=user_id,
                timestamp=time.time(),
                target='users',
                value={'user_id': user_id}
            )
            
            await self._broadcast_to_workspace(workspace_id, leave_operation.to_dict())
            
            # Clean up empty workspaces
            if not workspace.users:
                del self.workspaces[workspace_id]
        
        logger.info(f"User {user_id} left workspace {workspace_id}")
    
    async def _handle_operation(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle collaborative operations"""
        workspace_id = data.get('workspace_id')
        
        if workspace_id not in self.workspaces:
            await self._send_error(websocket, "Workspace not found")
            return
        
        workspace = self.workspaces[workspace_id]
        
        try:
            operation = Operation.from_dict(data['operation'])
            operation.user_id = user_id
            
            # Apply operation to workspace
            success = workspace.apply_operation(operation)
            
            if success:
                # Broadcast operation to all users in workspace
                await self._broadcast_to_workspace(workspace_id, {
                    'type': 'operation_applied',
                    'operation': operation.to_dict(),
                    'workspace_version': workspace.version
                })
                
                # Send confirmation to sender
                await websocket.send(json.dumps({
                    'type': 'operation_confirmed',
                    'operation_id': operation.id,
                    'version': operation.version
                }))
            else:
                # Operation was rejected (conflict)
                await websocket.send(json.dumps({
                    'type': 'operation_rejected',
                    'operation_id': operation.id,
                    'reason': 'Conflict detected'
                }))
                
        except Exception as e:
            logger.error(f"Error applying operation: {e}")
            await self._send_error(websocket, f"Failed to apply operation: {e}")
    
    async def _handle_sync_request(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle synchronization requests"""
        workspace_id = data.get('workspace_id')
        
        if workspace_id in self.workspaces:
            workspace = self.workspaces[workspace_id]
            state = workspace.get_full_state()
            
            await websocket.send(json.dumps({
                'type': 'full_sync',
                'state': state
            }))
    
    async def _handle_cursor_update(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle cursor position updates"""
        workspace_id = data.get('workspace_id')
        
        if workspace_id in self.workspaces and user_id in self.user_sessions:
            workspace = self.workspaces[workspace_id]
            user_session = self.user_sessions[user_id]
            
            # Update cursor position
            user_session.cursor_position = data.get('position', 0)
            user_session.selection_start = data.get('selection_start', 0)
            user_session.selection_end = data.get('selection_end', 0)
            user_session.last_active = time.time()
            
            # Broadcast cursor update
            cursor_operation = Operation(
                id=str(uuid.uuid4()),
                type=OperationType.CURSOR_POSITION,
                user_id=user_id,
                timestamp=time.time(),
                target='cursor',
                value={
                    'position': user_session.cursor_position,
                    'selection': [user_session.selection_start, user_session.selection_end]
                }
            )
            
            await self._broadcast_to_workspace(workspace_id, cursor_operation.to_dict(), exclude_user=user_id)
    
    async def _handle_chat_message(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle chat messages"""
        workspace_id = data.get('workspace_id')
        message = data.get('message', '')
        
        if workspace_id in self.workspaces and user_id in self.user_sessions:
            user_session = self.user_sessions[user_id]
            
            chat_operation = Operation(
                id=str(uuid.uuid4()),
                type=OperationType.CHAT_MESSAGE,
                user_id=user_id,
                timestamp=time.time(),
                target='chat',
                value={
                    'message': message,
                    'username': user_session.username,
                    'color': user_session.color
                }
            )
            
            await self._broadcast_to_workspace(workspace_id, chat_operation.to_dict())
    
    async def _handle_generate(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle generation requests"""
        workspace_id = data.get('workspace_id')
        
        if workspace_id not in self.workspaces:
            await self._send_error(websocket, "Workspace not found")
            return
        
        workspace = self.workspaces[workspace_id]
        
        # Notify generation start
        start_operation = Operation(
            id=str(uuid.uuid4()),
            type=OperationType.GENERATION_START,
            user_id=user_id,
            timestamp=time.time(),
            target='generation',
            value={'user_id': user_id}
        )
        
        await self._broadcast_to_workspace(workspace_id, start_operation.to_dict())
        
        # Start generation in background thread
        threading.Thread(
            target=self._run_generation,
            args=(workspace_id, workspace.state.copy(), user_id),
            daemon=True
        ).start()
    
    async def _handle_interrupt(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle generation interrupt requests"""
        workspace_id = data.get('workspace_id')
        
        if workspace_id in self.workspaces:
            # Set interrupt flag
            shared.state.interrupt()
            
            # Notify workspace
            interrupt_operation = Operation(
                id=str(uuid.uuid4()),
                type=OperationType.GENERATION_COMPLETE,
                user_id=user_id,
                timestamp=time.time(),
                target='generation',
                value={'status': 'interrupted'}
            )
            
            await self._broadcast_to_workspace(workspace_id, interrupt_operation.to_dict())
    
    async def _handle_progress_sync(self, user_id: str, data: Dict, websocket: WebSocketServerProtocol):
        """Handle progress synchronization"""
        workspace_id = data.get('workspace_id')
        progress = data.get('progress', 0)
        
        if workspace_id in self.workspaces:
            workspace = self.workspaces[workspace_id]
            workspace.state['progress'] = progress
            
            progress_operation = Operation(
                id=str(uuid.uuid4()),
                type=OperationType.GENERATION_PROGRESS,
                user_id=user_id,
                timestamp=time.time(),
                target='progress',
                value=progress
            )
            
            await self._broadcast_to_workspace(workspace_id, progress_operation.to_dict(), exclude_user=user_id)
    
    def _run_generation(self, workspace_id: str, params: Dict, user_id: str):
        """Run generation in background thread"""
        try:
            # Create processing object
            p = StableDiffusionProcessing(
                sd_model=sd_models.get_closet_checkpoint_match(params.get('model')),
                outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
                outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
                prompt=params.get('prompt', ''),
                negative_prompt=params.get('negative_prompt', ''),
                seed=params.get('seed', -1),
                sampler_name=params.get('sampler_name', 'Euler a'),
                steps=params.get('steps', 20),
                cfg_scale=params.get('cfg_scale', 7.0),
                width=params.get('width', 512),
                height=params.get('height', 512),
                batch_size=params.get('batch_size', 1),
            )
            
            # Add LoRA if specified
            lora_list = params.get('lora', [])
            if lora_list:
                # This would need integration with LoRA extension
                pass
            
            # Run generation
            processed = process_images(p)
            
            # Send result back to workspace
            if processed.images:
                # Save first image
                image = processed.images[0]
                import base64
                from io import BytesIO
                
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Create completion operation
                asyncio.run_coroutine_threadsafe(
                    self._send_generation_complete(workspace_id, user_id, img_str, processed),
                    asyncio.get_event_loop()
                )
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            asyncio.run_coroutine_threadsafe(
                self._send_generation_error(workspace_id, user_id, str(e)),
                asyncio.get_event_loop()
            )
    
    async def _send_generation_complete(self, workspace_id: str, user_id: str, image_data: str, processed: Processed):
        """Send generation completion notification"""
        if workspace_id in self.workspaces:
            complete_operation = Operation(
                id=str(uuid.uuid4()),
                type=OperationType.GENERATION_COMPLETE,
                user_id=user_id,
                timestamp=time.time(),
                target='generation',
                value={
                    'image': image_data,
                    'seed': processed.seed,
                    'info': processed.info,
                    'parameters': processed.parameters
                }
            )
            
            await self._broadcast_to_workspace(workspace_id, complete_operation.to_dict())
    
    async def _send_generation_error(self, workspace_id: str, user_id: str, error: str):
        """Send generation error notification"""
        if workspace_id in self.workspaces:
            error_operation = Operation(
                id=str(uuid.uuid4()),
                type=OperationType.GENERATION_COMPLETE,
                user_id=user_id,
                timestamp=time.time(),
                target='generation',
                value={'status': 'error', 'error': error}
            )
            
            await self._broadcast_to_workspace(workspace_id, error_operation.to_dict())
    
    async def _broadcast_to_workspace(self, workspace_id: str, message: Dict, exclude_user: str = None):
        """Broadcast message to all users in a workspace"""
        if workspace_id not in self.workspaces:
            return
        
        workspace = self.workspaces[workspace_id]
        message_json = json.dumps(message)
        
        tasks = []
        for user_id, user_session in workspace.users.items():
            if user_id != exclude_user and user_id in self.connections:
                try:
                    tasks.append(self.connections[user_id].send(message_json))
                except:
                    pass
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_error(self, websocket: WebSocketServerProtocol, message: str):
        """Send error message to client"""
        try:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': message,
                'timestamp': time.time()
            }))
        except:
            pass
    
    async def _cleanup_user(self, user_id: str):
        """Clean up user data on disconnect"""
        # Remove from all workspaces
        for workspace_id, workspace in list(self.workspaces.items()):
            if user_id in workspace.users:
                workspace.remove_user(user_id)
                
                # Notify other users
                leave_operation = Operation(
                    id=str(uuid.uuid4()),
                    type=OperationType.USER_LEAVE,
                    user_id=user_id,
                    timestamp=time.time(),
                    target='users',
                    value={'user_id': user_id}
                )
                
                await self._broadcast_to_workspace(workspace_id, leave_operation.to_dict())
        
        # Clean up user data
        if user_id in self.connections:
            del self.connections[user_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        
        logger.info(f"Cleaned up user: {user_id}")

class RealtimeSyncClient:
    """Client for connecting to RealtimeSync server"""
    
    def __init__(self, server_url: str = None):
        self.server_url = server_url or f"ws://{cmd_opts.listen or 'localhost'}:{cmd_opts.port + 1}"
        self.websocket = None
        self.user_id = None
        self.workspace_id = None
        self.connected = False
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    async def connect(self, workspace_id: str = "default", username: str = None):
        """Connect to the collaboration server"""
        self.workspace_id = workspace_id
        
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.connected = True
            self.reconnect_attempts = 0
            
            # Join workspace
            await self.send({
                'type': 'join_workspace',
                'workspace_id': workspace_id,
                'username': username or f"User_{uuid.uuid4().hex[:8]}"
            })
            
            # Start message listener
            asyncio.create_task(self._listen_for_messages())
            
            logger.info(f"Connected to collaboration server at {self.server_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to collaboration server: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the collaboration server"""
        if self.websocket:
            await self.send({
                'type': 'leave_workspace',
                'workspace_id': self.workspace_id
            })
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from collaboration server")
    
    async def send(self, data: Dict):
        """Send data to the server"""
        if self.websocket and self.connected:
            try:
                await self.websocket.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                await self._handle_disconnect()
    
    async def send_operation(self, operation_type: OperationType, target: str, value: Any):
        """Send a collaborative operation"""
        operation = Operation(
            id=str(uuid.uuid4()),
            type=operation_type,
            user_id=self.user_id,
            timestamp=time.time(),
            target=target,
            value=value
        )
        
        await self.send({
            'type': 'operation',
            'workspace_id': self.workspace_id,
            'operation': operation.to_dict()
        })
    
    async def update_prompt(self, prompt: str, position: int = None, action: str = "replace"):
        """Update the prompt collaboratively"""
        if position is not None:
            value = {
                'action': action,
                'position': position,
                'text': prompt if action == 'insert' else '',
                'length': len(prompt) if action == 'delete' else 0
            }
        else:
            value = prompt
        
        await self.send_operation(OperationType.PROMPT_UPDATE, 'prompt', value)
    
    async def update_negative_prompt(self, prompt: str):
        """Update the negative prompt"""
        await self.send_operation(OperationType.NEGATIVE_PROMPT_UPDATE, 'negative_prompt', prompt)
    
    async def update_seed(self, seed: int):
        """Update the seed"""
        await self.send_operation(OperationType.SEED_UPDATE, 'seed', seed)
    
    async def update_model(self, model_name: str):
        """Update the model"""
        await self.send_operation(OperationType.MODEL_CHANGE, 'model', model_name)
    
    async def update_sampler(self, sampler_name: str):
        """Update the sampler"""
        await self.send_operation(OperationType.SAMPLER_CHANGE, 'sampler_name', sampler_name)
    
    async def update_steps(self, steps: int):
        """Update the steps"""
        await self.send_operation(OperationType.STEPS_UPDATE, 'steps', steps)
    
    async def update_cfg_scale(self, cfg_scale: float):
        """Update CFG scale"""
        await self.send_operation(OperationType.CFG_UPDATE, 'cfg_scale', cfg_scale)
    
    async def update_size(self, width: int, height: int):
        """Update image size"""
        await self.send_operation(OperationType.SIZE_UPDATE, 'size', {'width': width, 'height': height})
    
    async def update_batch_size(self, batch_size: int):
        """Update batch size"""
        await self.send_operation(OperationType.BATCH_UPDATE, 'batch_size', batch_size)
    
    async def add_lora(self, lora_name: str):
        """Add LoRA to generation"""
        await self.send_operation(OperationType.LORA_UPDATE, 'lora', {
            'action': 'add',
            'name': lora_name
        })
    
    async def remove_lora(self, lora_name: str):
        """Remove LoRA from generation"""
        await self.send_operation(OperationType.LORA_UPDATE, 'lora', {
            'action': 'remove',
            'name': lora_name
        })
    
    async def start_generation(self):
        """Start generation"""
        await self.send({
            'type': 'generate',
            'workspace_id': self.workspace_id
        })
    
    async def interrupt_generation(self):
        """Interrupt current generation"""
        await self.send({
            'type': 'interrupt',
            'workspace_id': self.workspace_id
        })
    
    async def send_chat_message(self, message: str):
        """Send a chat message"""
        await self.send({
            'type': 'chat_message',
            'workspace_id': self.workspace_id,
            'message': message
        })
    
    async def update_cursor(self, position: int, selection_start: int = None, selection_end: int = None):
        """Update cursor position"""
        await self.send({
            'type': 'cursor_update',
            'workspace_id': self.workspace_id,
            'position': position,
            'selection_start': selection_start or position,
            'selection_end': selection_end or position
        })
    
    def on(self, event: str, callback: Callable):
        """Register event callback"""
        self.callbacks[event].append(callback)
    
    async def _listen_for_messages(self):
        """Listen for incoming messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            await self._handle_disconnect()
    
    async def _handle_message(self, data: Dict):
        """Handle incoming messages"""
        message_type = data.get('type')
        
        # Call registered callbacks
        for callback in self.callbacks.get(message_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {message_type}: {e}")
        
        # Handle specific message types
        if message_type == 'workspace_joined':
            self.user_id = data.get('user_id')
            logger.info(f"Joined workspace as {self.user_id}")
            
        elif message_type == 'operation_applied':
            operation = data.get('operation')
            logger.debug(f"Operation applied: {operation['type']} on {operation['target']}")
            
        elif message_type == 'error':
            logger.error(f"Server error: {data.get('message')}")
    
    async def _handle_disconnect(self):
        """Handle disconnection and attempt reconnection"""
        self.connected = False
        
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting reconnection ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
            
            await asyncio.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
            await self.connect(self.workspace_id)
        else:
            logger.error("Max reconnection attempts reached")

# Integration with existing WebUI
class RealtimeSyncExtension:
    """Extension for integrating real-time sync with WebUI"""
    
    def __init__(self):
        self.server = None
        self.client = None
        self.enabled = cmd_opts.enable_collaboration if hasattr(cmd_opts, 'enable_collaboration') else False
        
    def initialize(self):
        """Initialize the collaboration system"""
        if not self.enabled:
            return
        
        # Start server if enabled
        if cmd_opts.collaboration_server if hasattr(cmd_opts, 'collaboration_server') else False:
            self.server = RealtimeSyncServer(
                host=cmd_opts.listen or "0.0.0.0",
                port=cmd_opts.port + 1
            )
            
            # Start server in background thread
            threading.Thread(
                target=self._run_server,
                daemon=True
            ).start()
        
        # Initialize client
        self.client = RealtimeSyncClient()
        
        logger.info("Real-time collaboration system initialized")
    
    def _run_server(self):
        """Run the WebSocket server in background"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.server.start())
        except Exception as e:
            logger.error(f"Failed to start collaboration server: {e}")
        finally:
            loop.close()
    
    def setup_ui(self):
        """Add collaboration UI elements to the WebUI"""
        if not self.enabled:
            return
        
        # This would be called during UI setup to add collaboration controls
        pass
    
    async def connect_to_workspace(self, workspace_id: str, username: str = None):
        """Connect to a collaboration workspace"""
        if self.client:
            return await self.client.connect(workspace_id, username)
        return False
    
    async def disconnect_from_workspace(self):
        """Disconnect from current workspace"""
        if self.client:
            await self.client.disconnect()

# Global instance
realtime_sync = RealtimeSyncExtension()

# Command line arguments
def add_commandline_arguments(parser):
    """Add command line arguments for collaboration"""
    parser.add_argument(
        "--enable-collaboration",
        action="store_true",
        help="Enable real-time collaboration features",
        default=False
    )
    parser.add_argument(
        "--collaboration-server",
        action="store_true",
        help="Start collaboration WebSocket server",
        default=False
    )

# Hook into existing modules
def on_app_started(demo, app):
    """Called when the app starts"""
    realtime_sync.initialize()
    realtime_sync.setup_ui()

# Export for use in other modules
__all__ = [
    'RealtimeSyncServer',
    'RealtimeSyncClient', 
    'RealtimeSyncExtension',
    'Operation',
    'OperationType',
    'SharedWorkspace',
    'UserSession',
    'realtime_sync'
]