"""
Extension Hot-Reload System for flux
Enables live extension reloading without restarting the entire UI
with dependency isolation and version conflict resolution.
"""

import importlib
import importlib.util
import sys
import os
import time
import threading
import traceback
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pkg_resources
from packaging import version, requirements

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ExtensionState:
    """Represents the state of a loaded extension"""
    name: str
    path: Path
    modules: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    version: str = "0.0.0"
    last_modified: float = 0.0
    checksum: str = ""
    is_loaded: bool = False
    load_time: float = 0.0
    error: Optional[str] = None

class DependencyGraph:
    """Manages extension dependency relationships and conflict resolution"""
    
    def __init__(self):
        self.graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.extension_versions = {}
        self.dependency_cache = {}
        
    def add_extension(self, ext_name: str, dependencies: List[str], version_str: str):
        """Add an extension to the dependency graph"""
        self.graph[ext_name] = set(dependencies)
        self.extension_versions[ext_name] = version_str
        
        for dep in dependencies:
            self.reverse_graph[dep].add(ext_name)
    
    def get_load_order(self, changed_extension: str) -> List[str]:
        """Get topologically sorted load order for reloading an extension and its dependents"""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected: {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            
            # Visit dependencies first
            for dep in self.graph.get(node, []):
                if dep in self.graph:  # Only if dependency is an extension
                    visit(dep)
            
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        
        # Start with changed extension and all its dependents
        to_visit = {changed_extension}
        to_visit.update(self.reverse_graph.get(changed_extension, set()))
        
        for ext in to_visit:
            if ext not in visited:
                visit(ext)
        
        return order
    
    def check_conflicts(self, ext_name: str) -> List[str]:
        """Check for version conflicts with existing extensions"""
        conflicts = []
        ext_deps = self.graph.get(ext_name, [])
        
        for dep in ext_deps:
            if dep in self.extension_versions:
                # Parse requirement
                try:
                    req = requirements.Requirement(dep)
                    installed_version = self.extension_versions.get(req.name, "0.0.0")
                    
                    if not version.parse(installed_version) in req.specifier:
                        conflicts.append(
                            f"Extension {ext_name} requires {dep}, "
                            f"but {req.name} {installed_version} is installed"
                        )
                except Exception as e:
                    logger.warning(f"Failed to parse requirement {dep}: {e}")
        
        return conflicts

class NamespaceIsolator:
    """Provides namespace isolation for extension modules"""
    
    def __init__(self):
        self.namespace_cache = {}
        self.original_modules = {}
        
    def create_isolated_namespace(self, ext_name: str) -> Dict[str, Any]:
        """Create an isolated namespace for an extension"""
        namespace = {
            '__name__': f'extensions.{ext_name}',
            '__package__': f'extensions.{ext_name}',
            '__path__': [],
            '__file__': '',
            '__loader__': None,
            '__cached__': None,
        }
        self.namespace_cache[ext_name] = namespace
        return namespace
    
    def backup_modules(self, module_names: List[str]):
        """Backup original modules before isolation"""
        for name in module_names:
            if name in sys.modules:
                self.original_modules[name] = sys.modules[name]
    
    def restore_modules(self):
        """Restore original modules after extension unload"""
        for name, module in self.original_modules.items():
            sys.modules[name] = module
        self.original_modules.clear()

class ExtensionHotReloader:
    """
    Main hot-reload system for extensions
    Monitors extension directories and reloads changed extensions
    with dependency isolation and conflict resolution
    """
    
    def __init__(self, 
                 extensions_dirs: List[str] = None,
                 check_interval: float = 1.0,
                 enable_hot_reload: bool = True):
        
        self.extensions_dirs = extensions_dirs or [
            "extensions",
            "extensions-builtin"
        ]
        
        self.check_interval = check_interval
        self.enable_hot_reload = enable_hot_reload
        
        self.extensions: Dict[str, ExtensionState] = {}
        self.dependency_graph = DependencyGraph()
        self.namespace_isolator = NamespaceIsolator()
        
        self.file_hashes: Dict[str, str] = {}
        self.reload_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.watcher_thread = None
        
        self.reload_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Cache for module specs
        self.spec_cache = {}
        
        # Initialize
        self._discover_extensions()
        
        logger.info(f"ExtensionHotReloader initialized with {len(self.extensions)} extensions")
    
    def _discover_extensions(self):
        """Discover all extensions in the configured directories"""
        for ext_dir in self.extensions_dirs:
            ext_path = Path(ext_dir)
            if not ext_path.exists():
                continue
                
            for item in ext_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    self._register_extension(item)
    
    def _register_extension(self, ext_path: Path):
        """Register an extension for hot-reloading"""
        ext_name = ext_path.name
        
        # Calculate initial checksum
        checksum = self._calculate_directory_hash(ext_path)
        
        # Read extension metadata if available
        metadata = self._read_extension_metadata(ext_path)
        
        # Create extension state
        ext_state = ExtensionState(
            name=ext_name,
            path=ext_path,
            dependencies=metadata.get("dependencies", []),
            version=metadata.get("version", "0.0.0"),
            last_modified=time.time(),
            checksum=checksum
        )
        
        self.extensions[ext_name] = ext_state
        
        # Add to dependency graph
        self.dependency_graph.add_extension(
            ext_name,
            ext_state.dependencies,
            ext_state.version
        )
        
        # Initialize file hashes for monitoring
        self._update_file_hashes(ext_path)
        
        logger.debug(f"Registered extension: {ext_name}")
    
    def _read_extension_metadata(self, ext_path: Path) -> Dict[str, Any]:
        """Read extension metadata from various possible locations"""
        metadata = {
            "dependencies": [],
            "version": "0.0.0"
        }
        
        # Check for requirements.txt
        req_file = ext_path / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    metadata["dependencies"] = deps
            except Exception as e:
                logger.warning(f"Failed to read requirements.txt for {ext_path.name}: {e}")
        
        # Check for setup.py or pyproject.toml for version
        setup_file = ext_path / "setup.py"
        if setup_file.exists():
            try:
                # Simple extraction of version from setup.py
                with open(setup_file, 'r') as f:
                    content = f.read()
                    if 'version=' in content:
                        # Basic extraction - could be improved
                        for line in content.split('\n'):
                            if 'version=' in line:
                                version_str = line.split('version=')[1].split(',')[0].strip().strip('"\'')
                                metadata["version"] = version_str
                                break
            except Exception as e:
                logger.warning(f"Failed to read setup.py for {ext_path.name}: {e}")
        
        # Check for extension.json (custom metadata)
        metadata_file = ext_path / "extension.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    custom_metadata = json.load(f)
                    metadata.update(custom_metadata)
            except Exception as e:
                logger.warning(f"Failed to read extension.json for {ext_path.name}: {e}")
        
        return metadata
    
    def _calculate_directory_hash(self, directory: Path) -> str:
        """Calculate hash of directory contents for change detection"""
        hasher = hashlib.md5()
        
        for root, dirs, files in os.walk(directory):
            for file in sorted(files):
                if file.endswith(('.py', '.json', '.txt', '.yaml', '.yml')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'rb') as f:
                            hasher.update(f.read())
                    except Exception:
                        pass
        
        return hasher.hexdigest()
    
    def _update_file_hashes(self, directory: Path):
        """Update file hashes for a directory"""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.py', '.json', '.txt', '.yaml', '.yml')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            self.file_hashes[str(file_path)] = file_hash
                    except Exception:
                        pass
    
    def _has_files_changed(self, directory: Path) -> bool:
        """Check if any files in directory have changed"""
        current_hashes = {}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.py', '.json', '.txt', '.yaml', '.yml')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            current_hashes[str(file_path)] = file_hash
                    except Exception:
                        pass
        
        # Compare with stored hashes
        for file_path, file_hash in current_hashes.items():
            if file_path not in self.file_hashes or self.file_hashes[file_path] != file_hash:
                # Update hash
                self.file_hashes[file_path] = file_hash
                return True
        
        return False
    
    def start_watching(self):
        """Start the file watcher thread"""
        if not self.enable_hot_reload:
            logger.info("Hot reload is disabled")
            return
        
        if self.watcher_thread and self.watcher_thread.is_alive():
            logger.warning("Watcher thread already running")
            return
        
        self.stop_event.clear()
        self.watcher_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="ExtensionHotReloadWatcher"
        )
        self.watcher_thread.start()
        logger.info("Started extension hot-reload watcher")
    
    def stop_watching(self):
        """Stop the file watcher thread"""
        self.stop_event.set()
        if self.watcher_thread:
            self.watcher_thread.join(timeout=5.0)
            logger.info("Stopped extension hot-reload watcher")
    
    def _watch_loop(self):
        """Main loop for watching extension changes"""
        while not self.stop_event.is_set():
            try:
                self._check_for_changes()
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
                logger.debug(traceback.format_exc())
            
            # Wait for next check
            self.stop_event.wait(self.check_interval)
    
    def _check_for_changes(self):
        """Check all extensions for changes"""
        with self.reload_lock:
            for ext_name, ext_state in self.extensions.items():
                try:
                    # Check if directory has changed
                    if self._has_files_changed(ext_state.path):
                        current_hash = self._calculate_directory_hash(ext_state.path)
                        
                        if current_hash != ext_state.checksum:
                            logger.info(f"Detected changes in extension: {ext_name}")
                            self.reload_extension(ext_name)
                            ext_state.checksum = current_hash
                except Exception as e:
                    logger.error(f"Error checking extension {ext_name}: {e}")
                    logger.debug(traceback.format_exc())
    
    def reload_extension(self, ext_name: str, force: bool = False) -> bool:
        """
        Reload a specific extension
        
        Args:
            ext_name: Name of extension to reload
            force: Force reload even if no changes detected
        
        Returns:
            bool: True if reload successful
        """
        with self.reload_lock:
            if ext_name not in self.extensions:
                logger.error(f"Extension not found: {ext_name}")
                return False
            
            ext_state = self.extensions[ext_name]
            
            # Check for dependency conflicts
            conflicts = self.dependency_graph.check_conflicts(ext_name)
            if conflicts:
                error_msg = f"Dependency conflicts for {ext_name}: {conflicts}"
                logger.error(error_msg)
                ext_state.error = error_msg
                self._notify_error(ext_name, error_msg)
                return False
            
            # Get reload order (extension and its dependents)
            try:
                reload_order = self.dependency_graph.get_load_order(ext_name)
            except ValueError as e:
                logger.error(f"Failed to get reload order: {e}")
                return False
            
            logger.info(f"Reloading extensions in order: {reload_order}")
            
            # Unload extensions in reverse order
            for ext in reversed(reload_order):
                if ext in self.extensions:
                    self._unload_extension(ext)
            
            # Reload extensions in order
            success = True
            for ext in reload_order:
                if ext in self.extensions:
                    if not self._load_extension(ext):
                        success = False
                        logger.error(f"Failed to reload extension: {ext}")
                        break
            
            if success:
                logger.info(f"Successfully reloaded {len(reload_order)} extensions")
                self._notify_reload(ext_name, reload_order)
            else:
                logger.error(f"Failed to reload extension chain starting with {ext_name}")
            
            return success
    
    def _unload_extension(self, ext_name: str):
        """Unload an extension and clean up its modules"""
        if ext_name not in self.extensions:
            return
        
        ext_state = self.extensions[ext_name]
        
        # Remove modules from sys.modules
        modules_to_remove = []
        for module_name in list(sys.modules.keys()):
            if module_name.startswith(f"extensions.{ext_name}.") or module_name == f"extensions.{ext_name}":
                modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            try:
                # Call module's cleanup function if available
                module = sys.modules.get(module_name)
                if module and hasattr(module, 'on_unload'):
                    module.on_unload()
                
                del sys.modules[module_name]
            except Exception as e:
                logger.warning(f"Error removing module {module_name}: {e}")
        
        # Clear extension state
        ext_state.modules.clear()
        ext_state.is_loaded = False
        ext_state.error = None
        
        # Restore original modules
        self.namespace_isolator.restore_modules()
        
        logger.debug(f"Unloaded extension: {ext_name}")
    
    def _load_extension(self, ext_name: str) -> bool:
        """Load an extension with namespace isolation"""
        if ext_name not in self.extensions:
            return False
        
        ext_state = self.extensions[ext_name]
        start_time = time.time()
        
        try:
            # Create isolated namespace
            namespace = self.namespace_isolator.create_isolated_namespace(ext_name)
            
            # Backup modules that might be modified
            modules_to_backup = self._get_modules_to_backup(ext_state.path)
            self.namespace_isolator.backup_modules(modules_to_backup)
            
            # Load extension modules
            loaded_modules = self._load_extension_modules(ext_state.path, ext_name, namespace)
            
            # Update extension state
            ext_state.modules = loaded_modules
            ext_state.is_loaded = True
            ext_state.load_time = time.time() - start_time
            ext_state.error = None
            
            logger.info(f"Loaded extension {ext_name} in {ext_state.load_time:.2f}s with {len(loaded_modules)} modules")
            
            # Call extension's init function if available
            self._call_extension_init(ext_name, loaded_modules)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load extension {ext_name}: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            ext_state.error = error_msg
            ext_state.is_loaded = False
            
            self._notify_error(ext_name, error_msg)
            return False
    
    def _get_modules_to_backup(self, ext_path: Path) -> List[str]:
        """Get list of modules that might be modified by extension"""
        modules = []
        
        # Common modules that extensions might modify
        common_modules = [
            'modules.sd_hijack',
            'modules.sd_hijack_optimizations',
            'modules.sd_models',
            'modules.sd_vae',
            'modules.ui',
            'modules.shared',
            'modules.processing',
            'modules.devices',
        ]
        
        # Check if extension has a preload script
        preload_file = ext_path / "preload.py"
        if preload_file.exists():
            modules.extend(common_modules)
        
        return modules
    
    def _load_extension_modules(self, ext_path: Path, ext_name: str, namespace: Dict) -> Dict[str, Any]:
        """Load all Python modules from an extension directory"""
        loaded_modules = {}
        
        # Find all Python files
        py_files = list(ext_path.glob("**/*.py"))
        
        for py_file in py_files:
            try:
                # Create module spec
                module_name = f"extensions.{ext_name}.{py_file.stem}"
                
                # Use importlib to load module
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    py_file,
                    submodule_search_locations=[]
                )
                
                if spec is None:
                    logger.warning(f"Could not create spec for {py_file}")
                    continue
                
                # Create module from spec
                module = importlib.util.module_from_spec(spec)
                
                # Add to sys.modules
                sys.modules[module_name] = module
                
                # Execute module
                spec.loader.exec_module(module)
                
                # Store in loaded modules
                loaded_modules[module_name] = module
                
                logger.debug(f"Loaded module: {module_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {py_file}: {e}")
                logger.debug(traceback.format_exc())
                # Continue loading other modules
        
        return loaded_modules
    
    def _call_extension_init(self, ext_name: str, modules: Dict[str, Any]):
        """Call extension initialization functions"""
        for module_name, module in modules.items():
            # Look for common init functions
            init_functions = ['init', 'setup', 'initialize', 'on_load']
            
            for func_name in init_functions:
                if hasattr(module, func_name):
                    try:
                        func = getattr(module, func_name)
                        if callable(func):
                            logger.debug(f"Calling {module_name}.{func_name}")
                            func()
                    except Exception as e:
                        logger.error(f"Error calling {module_name}.{func_name}: {e}")
    
    def reload_all_extensions(self) -> bool:
        """Reload all extensions"""
        with self.reload_lock:
            logger.info("Reloading all extensions")
            
            # Unload all extensions
            for ext_name in list(self.extensions.keys()):
                self._unload_extension(ext_name)
            
            # Reload all extensions
            success_count = 0
            for ext_name in self.extensions.keys():
                if self._load_extension(ext_name):
                    success_count += 1
            
            logger.info(f"Reloaded {success_count}/{len(self.extensions)} extensions")
            return success_count == len(self.extensions)
    
    def get_extension_info(self, ext_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an extension"""
        if ext_name not in self.extensions:
            return None
        
        ext_state = self.extensions[ext_name]
        
        return {
            'name': ext_state.name,
            'path': str(ext_state.path),
            'version': ext_state.version,
            'dependencies': ext_state.dependencies,
            'is_loaded': ext_state.is_loaded,
            'load_time': ext_state.load_time,
            'error': ext_state.error,
            'modules': list(ext_state.modules.keys()),
            'last_modified': ext_state.last_modified,
        }
    
    def get_all_extensions_info(self) -> List[Dict[str, Any]]:
        """Get information about all extensions"""
        return [self.get_extension_info(ext_name) for ext_name in self.extensions.keys()]
    
    def register_reload_callback(self, callback: Callable):
        """Register a callback to be called after successful reload"""
        self.reload_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable):
        """Register a callback to be called on reload error"""
        self.error_callbacks.append(callback)
    
    def _notify_reload(self, changed_ext: str, reloaded_exts: List[str]):
        """Notify all registered callbacks of successful reload"""
        for callback in self.reload_callbacks:
            try:
                callback(changed_ext, reloaded_exts)
            except Exception as e:
                logger.error(f"Error in reload callback: {e}")
    
    def _notify_error(self, ext_name: str, error: str):
        """Notify all registered callbacks of error"""
        for callback in self.error_callbacks:
            try:
                callback(ext_name, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def force_reload_extension(self, ext_name: str) -> bool:
        """Force reload an extension regardless of changes"""
        return self.reload_extension(ext_name, force=True)
    
    def get_changed_extensions(self) -> List[str]:
        """Get list of extensions that have changed since last load"""
        changed = []
        
        for ext_name, ext_state in self.extensions.items():
            try:
                current_hash = self._calculate_directory_hash(ext_state.path)
                if current_hash != ext_state.checksum:
                    changed.append(ext_name)
            except Exception as e:
                logger.error(f"Error checking {ext_name}: {e}")
        
        return changed
    
    def enable(self):
        """Enable hot reloading"""
        self.enable_hot_reload = True
        if not self.watcher_thread or not self.watcher_thread.is_alive():
            self.start_watching()
        logger.info("Hot reload enabled")
    
    def disable(self):
        """Disable hot reloading"""
        self.enable_hot_reload = False
        self.stop_watching()
        logger.info("Hot reload disabled")
    
    def is_watching(self) -> bool:
        """Check if watcher is running"""
        return (self.watcher_thread is not None and 
                self.watcher_thread.is_alive() and 
                self.enable_hot_reload)

# Global instance for easy access
_hot_reloader = None

def get_hot_reloader() -> ExtensionHotReloader:
    """Get or create the global hot reloader instance"""
    global _hot_reloader
    if _hot_reloader is None:
        _hot_reloader = ExtensionHotReloader()
    return _hot_reloader

def initialize_hot_reload(extensions_dirs: List[str] = None, 
                         check_interval: float = 1.0,
                         enable: bool = True):
    """Initialize the hot reload system"""
    global _hot_reloader
    
    if _hot_reloader is not None:
        _hot_reloader.stop_watching()
    
    _hot_reloader = ExtensionHotReloader(
        extensions_dirs=extensions_dirs,
        check_interval=check_interval,
        enable_hot_reload=enable
    )
    
    if enable:
        _hot_reloader.start_watching()
    
    return _hot_reloader

def reload_extension(ext_name: str) -> bool:
    """Reload a specific extension"""
    reloader = get_hot_reloader()
    return reloader.reload_extension(ext_name)

def reload_all_extensions() -> bool:
    """Reload all extensions"""
    reloader = get_hot_reloader()
    return reloader.reload_all_extensions()

def get_extension_status(ext_name: str) -> Optional[Dict[str, Any]]:
    """Get status of an extension"""
    reloader = get_hot_reloader()
    return reloader.get_extension_info(ext_name)

def get_all_extension_status() -> List[Dict[str, Any]]:
    """Get status of all extensions"""
    reloader = get_hot_reloader()
    return reloader.get_all_extensions_info()

# Integration with existing flux
def patch_extension_loader():
    """
    Patch the existing extension loader to use hot-reload
    This function should be called during UI initialization
    """
    try:
        # Import existing extension system
        import modules.extensions as extensions
        
        # Store original functions
        original_load_extensions = extensions.load_extensions
        original_reload_extensions = getattr(extensions, 'reload_extensions', None)
        
        def patched_load_extensions():
            """Patched version that uses hot-reload"""
            # Call original first
            result = original_load_extensions()
            
            # Initialize hot reload
            reloader = initialize_hot_reload(enable=True)
            
            # Load all extensions through hot-reload system
            for ext_name in list(reloader.extensions.keys()):
                reloader._load_extension(ext_name)
            
            return result
        
        def patched_reload_extensions():
            """Patched version for reloading"""
            reloader = get_hot_reloader()
            return reloader.reload_all_extensions()
        
        # Apply patches
        extensions.load_extensions = patched_load_extensions
        if original_reload_extensions:
            extensions.reload_extensions = patched_reload_extensions
        
        logger.info("Successfully patched extension loader for hot-reload")
        
    except ImportError as e:
        logger.warning(f"Could not patch extension loader: {e}")
    except Exception as e:
        logger.error(f"Error patching extension loader: {e}")
        logger.debug(traceback.format_exc())

# Auto-initialization when module is imported
if __name__ != "__main__":
    try:
        # Try to initialize hot reload
        initialize_hot_reload(enable=True)
    except Exception as e:
        logger.warning(f"Failed to auto-initialize hot reload: {e}")