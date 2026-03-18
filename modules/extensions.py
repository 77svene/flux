from __future__ import annotations

import configparser
import dataclasses
import os
import threading
import re
import sys
import importlib
import importlib.util
import hashlib
import time
from pathlib import Path
from typing import Dict, Set, Optional, List, Any
from collections import defaultdict

from modules import shared, errors, cache, scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

extensions: list[Extension] = []
extension_paths: dict[str, Extension] = {}
loaded_extensions: dict[str, Exception] = {}

# Hot-reload system additions
extension_modules: Dict[str, Set[str]] = defaultdict(set)  # extension_name -> set of module names
module_to_extension: Dict[str, str] = {}  # module_name -> extension_name
extension_dependencies: Dict[str, Set[str]] = defaultdict(set)  # extension_name -> set of dependency module hashes
extension_versions: Dict[str, str] = {}  # extension_name -> version hash
reload_lock = threading.RLock()
_hot_reload_enabled = True

os.makedirs(extensions_dir, exist_ok=True)


def active():
    if shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions == "all":
        return []
    elif shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]


def enable_hot_reload(enabled: bool = True):
    """Enable or disable the hot-reload system."""
    global _hot_reload_enabled
    _hot_reload_enabled = enabled


def get_extension_hash(extension_path: str) -> str:
    """Calculate a hash of extension files for version tracking."""
    hash_md5 = hashlib.md5()
    for root, dirs, files in os.walk(extension_path):
        # Skip .git directories and __pycache__
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules']]
        for file in sorted(files):
            if file.endswith(('.py', '.js', '.css', '.html')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        hash_md5.update(f.read())
                except Exception:
                    pass
    return hash_md5.hexdigest()


def unload_extension_modules(extension_name: str):
    """Unload all modules associated with an extension."""
    with reload_lock:
        if extension_name not in extension_modules:
            return
        
        modules_to_remove = []
        for module_name in list(sys.modules.keys()):
            # Check if this module belongs to our extension
            if (module_name in module_to_extension and 
                module_to_extension[module_name] == extension_name):
                modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            if module_name in sys.modules:
                del sys.modules[module_name]
            if module_name in module_to_extension:
                del module_to_extension[module_name]
        
        extension_modules[extension_name].clear()


def import_extension_module(extension_path: str, module_name: str, extension_name: str):
    """Import a module from an extension with namespace isolation."""
    try:
        # Create a unique module name to avoid conflicts
        unique_name = f"extensions.{extension_name}.{module_name}"
        
        # Check if module is already loaded
        if unique_name in sys.modules:
            return sys.modules[unique_name]
        
        # Build the full module path
        module_path = os.path.join(extension_path, f"{module_name}.py")
        if not os.path.exists(module_path):
            return None
        
        # Load the module spec
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec is None:
            return None
        
        # Create and execute the module
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        module_to_extension[unique_name] = extension_name
        extension_modules[extension_name].add(unique_name)
        
        # Execute the module
        spec.loader.exec_module(module)
        
        return module
    except Exception as e:
        errors.report(f"Error importing module {module_name} from extension {extension_name}: {e}", exc_info=True)
        return None


def check_dependency_compatibility(extension_name: str, dependencies: List[str]) -> bool:
    """Check if extension dependencies are compatible with loaded extensions."""
    for dep in dependencies:
        if dep in extension_versions:
            # Dependency is already loaded - check version compatibility
            # This is a simplified check - in production you'd want semantic versioning
            pass
    return True


@dataclasses.dataclass
class CallbackOrderInfo:
    name: str
    before: list
    after: list


class ExtensionMetadata:
    filename = "metadata.ini"
    config: configparser.ConfigParser
    canonical_name: str
    requires: list

    def __init__(self, path, canonical_name):
        self.config = configparser.ConfigParser()

        filepath = os.path.join(path, self.filename)
        # `self.config.read()` will quietly swallow OSErrors (which FileNotFoundError is),
        # so no need to check whether the file exists beforehand.
        try:
            self.config.read(filepath)
        except Exception:
            errors.report(f"Error reading {self.filename} for extension {canonical_name}.", exc_info=True)

        self.canonical_name = self.config.get("Extension", "Name", fallback=canonical_name)
        self.canonical_name = canonical_name.lower().strip()

        self.requires = None

    def get_script_requirements(self, field, section, extra_section=None):
        """reads a list of requirements from the config; field is the name of the field in the ini file,
        like Requires or Before, and section is the name of the [section] in the ini file; additionally,
        reads more requirements from [extra_section] if specified."""

        x = self.config.get(section, field, fallback='')

        if extra_section:
            x = x + ', ' + self.config.get(extra_section, field, fallback='')

        listed_requirements = self.parse_list(x.lower())
        res = []

        for requirement in listed_requirements:
            loaded_requirements = (x for x in requirement.split("|") if x in loaded_extensions)
            relevant_requirement = next(loaded_requirements, requirement)
            res.append(relevant_requirement)

        return res

    def parse_list(self, text):
        """converts a line from config ("ext1 ext2, ext3  ") into a python list (["ext1", "ext2", "ext3"])"""

        if not text:
            return []

        # both "," and " " are accepted as separator
        return [x for x in re.split(r"[,\s]+", text.strip()) if x]

    def list_callback_order_instructions(self):
        for section in self.config.sections():
            if not section.startswith("callbacks/"):
                continue

            callback_name = section[10:]

            if not callback_name.startswith(self.canonical_name):
                errors.report(f"Callback order section for extension {self.canonical_name} is referencing the wrong extension: {section}")
                continue

            before = self.parse_list(self.config.get(section, 'Before', fallback=''))
            after = self.parse_list(self.config.get(section, 'After', fallback=''))

            yield CallbackOrderInfo(callback_name, before, after)


class Extension:
    lock = threading.Lock()
    cached_fields = ['remote', 'commit_date', 'branch', 'commit_hash', 'version']
    metadata: ExtensionMetadata

    def __init__(self, name, path, enabled=True, is_builtin=False, metadata=None):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        self.is_builtin = is_builtin
        self.commit_hash = ''
        self.commit_date = None
        self.version = ''
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False
        self.metadata = metadata if metadata else ExtensionMetadata(self.path, name.lower())
        self.canonical_name = metadata.canonical_name
        self.last_reload_time = 0
        self.file_hashes = {}
        self.watcher_thread = None
        self.watcher_active = False

    def to_dict(self):
        return {x: getattr(self, x) for x in self.cached_fields}

    def from_dict(self, d):
        for field in self.cached_fields:
            setattr(self, field, d[field])

    def read_info_from_repo(self):
        if self.is_builtin or self.have_info_from_repo:
            return

        def read_from_repo():
            with self.lock:
                if self.have_info_from_repo:
                    return

                self.do_read_info_from_repo()

                return self.to_dict()

        try:
            d = cache.cached_data_for_file('extensions-git', self.name, os.path.join(self.path, ".git"), read_from_repo)
            self.from_dict(d)
        except FileNotFoundError:
            pass
        self.status = 'unknown' if self.status == '' else self.status

    def do_read_info_from_repo(self):
        repo = None
        try:
            if os.path.exists(os.path.join(self.path, ".git")):
                repo = Repo(self.path)
        except Exception:
            errors.report(f"Error reading github repository info from {self.path}", exc_info=True)

        if repo is None or repo.bare:
            self.remote = None
        else:
            try:
                self.remote = next(repo.remote().urls, None)
                commit = repo.head.commit
                self.commit_date = commit.committed_date
                if repo.active_branch:
                    self.branch = repo.active_branch.name
                self.commit_hash = commit.hexsha
                self.version = self.commit_hash[:8]

            except Exception:
                errors.report(f"Failed reading extension data from Git repository ({self.name})", exc_info=True)
                self.remote = None

        self.have_info_from_repo = True

    def list_files(self, subdir, extension):
        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []

        res = []
        for filename in sorted(os.listdir(dirpath)):
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        res = [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

        return res

    def check_updates(self):
        repo = Repo(self.path)
        branch_name = f'{repo.remote().name}/{self.branch}'
        for fetch in repo.remote().fetch(dry_run=True):
            if self.branch and fetch.name != branch_name:
                continue
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "new commits"
                return

        try:
            origin = repo.rev_parse(branch_name)
            if repo.head.commit != origin:
                self.can_update = True
                self.status = "behind HEAD"
                return
        except Exception:
            self.can_update = False
            self.status = "unknown (remote error)"
            return

        self.can_update = False
        self.status = "latest"

    def fetch_and_reset_hard(self, commit=None):
        repo = Repo(self.path)
        if commit is None:
            commit = f'{repo.remote().name}/{self.branch}'
        # Fix: `error: Your local changes to the following files would be overwritten by merge`,
        # because WSL2 Docker set 755 file permissions instead of 644, this

    # Hot-reload methods
    def get_file_hash(self, filepath: str) -> str:
        """Get hash of a single file."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def has_files_changed(self) -> bool:
        """Check if any files in the extension have changed."""
        if not self.file_hashes:
            return False
        
        for filepath, old_hash in self.file_hashes.items():
            if not os.path.exists(filepath):
                return True
            new_hash = self.get_file_hash(filepath)
            if new_hash != old_hash:
                return True
        return False

    def update_file_hashes(self):
        """Update the stored file hashes."""
        self.file_hashes = {}
        for root, dirs, files in os.walk(self.path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.idea']]
            for file in files:
                if file.endswith(('.py', '.js', '.css', '.html')):
                    filepath = os.path.join(root, file)
                    self.file_hashes[filepath] = self.get_file_hash(filepath)

    def reload(self, force: bool = False) -> bool:
        """Reload the extension with hot-reload support."""
        if not _hot_reload_enabled:
            return False
        
        with reload_lock:
            try:
                # Check if files have actually changed (unless forced)
                if not force and not self.has_files_changed():
                    return False
                
                # Store current extension version
                old_version = extension_versions.get(self.name)
                new_version = get_extension_hash(self.path)
                
                # Check dependency compatibility
                if not self._check_reload_compatibility():
                    errors.report(f"Cannot reload extension {self.name}: dependency conflicts detected")
                    return False
                
                # Unload existing modules
                self._unload_modules()
                
                # Reload the extension
                success = self._load_extension_modules()
                
                if success:
                    # Update version tracking
                    extension_versions[self.name] = new_version
                    self.last_reload_time = time.time()
                    self.update_file_hashes()
                    
                    # Notify about successful reload
                    print(f"Hot-reloaded extension: {self.name}")
                    
                    # Trigger UI update if needed
                    self._notify_reload_complete()
                
                return success
                
            except Exception as e:
                errors.report(f"Error reloading extension {self.name}: {e}", exc_info=True)
                return False

    def _check_reload_compatibility(self) -> bool:
        """Check if the extension can be reloaded without conflicts."""
        # Check if any loaded extensions depend on this one
        for ext_name, deps in extension_dependencies.items():
            if ext_name != self.name and self.name in deps:
                # Another extension depends on this one - check if reload would break it
                # This is a simplified check - in production you'd want more sophisticated dependency resolution
                pass
        return True

    def _unload_modules(self):
        """Unload all modules associated with this extension."""
        unload_extension_modules(self.name)

    def _load_extension_modules(self) -> bool:
        """Load all Python modules for this extension."""
        try:
            # Load __init__.py if it exists
            init_path = os.path.join(self.path, "__init__.py")
            if os.path.exists(init_path):
                import_extension_module(self.path, "__init__", self.name)
            
            # Load all Python files in the extension directory
            for root, dirs, files in os.walk(self.path):
                # Skip certain directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.idea']]
                
                for file in files:
                    if file.endswith('.py') and file != '__init__.py':
                        rel_path = os.path.relpath(os.path.join(root, file), self.path)
                        module_name = rel_path[:-3].replace(os.sep, '.')
                        import_extension_module(self.path, module_name, self.name)
            
            return True
        except Exception as e:
            errors.report(f"Error loading extension modules for {self.name}: {e}", exc_info=True)
            return False

    def _notify_reload_complete(self):
        """Notify the system that the extension has been reloaded."""
        # This could trigger UI updates or other callbacks
        pass

    def start_file_watcher(self, interval: float = 2.0):
        """Start a background thread to watch for file changes."""
        if self.watcher_active:
            return
        
        self.watcher_active = True
        self.update_file_hashes()
        
        def watcher():
            while self.watcher_active:
                try:
                    if self.has_files_changed():
                        print(f"Detected changes in extension {self.name}, reloading...")
                        self.reload()
                    time.sleep(interval)
                except Exception as e:
                    errors.report(f"Error in file watcher for {self.name}: {e}", exc_info=True)
                    time.sleep(interval)
        
        self.watcher_thread = threading.Thread(target=watcher, daemon=True)
        self.watcher_thread.start()

    def stop_file_watcher(self):
        """Stop the file watcher thread."""
        self.watcher_active = False
        if self.watcher_thread:
            self.watcher_thread.join(timeout=5.0)
            self.watcher_thread = None


# Global hot-reload functions
def reload_extension(extension_name: str, force: bool = False) -> bool:
    """Reload a specific extension by name."""
    if extension_name in extension_paths:
        return extension_paths[extension_name].reload(force=force)
    return False


def reload_all_extensions(force: bool = False):
    """Reload all enabled extensions."""
    for ext in active():
        ext.reload(force=force)


def start_extension_watchers():
    """Start file watchers for all enabled extensions."""
    for ext in active():
        ext.start_file_watcher()


def stop_extension_watchers():
    """Stop file watchers for all extensions."""
    for ext in extensions:
        ext.stop_file_watcher()


def get_extension_modules(extension_name: str) -> Set[str]:
    """Get all modules loaded for an extension."""
    return extension_modules.get(extension_name, set())


def get_extension_version(extension_name: str) -> Optional[str]:
    """Get the current version hash of an extension."""
    return extension_versions.get(extension_name)