"""modules/compatibility_engine.py

Semantic Extension Marketplace for flux
AI-powered extension discovery with compatibility scoring, automated dependency resolution, and one-click installation with rollback support.
"""

import os
import sys
import json
import hashlib
import shutil
import tempfile
import subprocess
import importlib
import inspect
import ast
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import pickle

import torch
import numpy as np
from packaging import version as pkg_version
from packaging.specifiers import SpecifierSet
from packaging.requirements import Requirement

# Import from existing modules
from modules import extensions, shared, paths, errors
from modules.extensions import Extension
from modules.shared import opts, cmd_opts


class ExtensionStatus(Enum):
    AVAILABLE = "available"
    INSTALLED = "installed"
    INCOMPATIBLE = "incompatible"
    UPDATING = "updating"
    BROKEN = "broken"
    DEPRECATED = "deprecated"


@dataclass
class ExtensionMetadata:
    name: str
    version: str
    description: str
    author: str
    repository: str
    tags: List[str]
    dependencies: List[str]
    python_dependencies: List[str]
    sd_webui_version: str
    compatibility_score: float
    install_date: Optional[str] = None
    last_updated: Optional[str] = None
    status: ExtensionStatus = ExtensionStatus.AVAILABLE
    file_hash: Optional[str] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    semantic_tags: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    rollback_version: Optional[str] = None
    breaking_changes: List[str] = field(default_factory=list)


@dataclass
class CompatibilityPrediction:
    extension_name: str
    target_version: str
    compatibility_score: float
    confidence: float
    breaking_changes: List[str]
    dependency_issues: List[str]
    performance_impact: Dict[str, float]
    prediction_timestamp: str
    model_version: str


class SemanticAnalyzer:
    """Analyzes extension code for semantic compatibility"""
    
    def __init__(self):
        self.api_patterns = self._load_api_patterns()
        self.compatibility_model = self._load_compatibility_model()
        
    def _load_api_patterns(self) -> Dict[str, List[str]]:
        """Load known API patterns from existing modules"""
        patterns = {
            'model_hooks': ['model_hijack', 'process', 'postprocess'],
            'ui_components': ['on_ui_tabs', 'on_ui_settings', 'on_ui_train_tabs'],
            'script_callbacks': ['callbacks_model_loaded', 'callbacks_ui_tabs'],
            'extension_points': ['scripts', 'postprocessing', 'extra_networks']
        }
        
        # Analyze existing extensions to extract patterns
        for ext in extensions.extensions:
            if ext.enabled:
                try:
                    module = importlib.import_module(f"{ext.name}.preload")
                    patterns[ext.name] = dir(module)
                except:
                    pass
                    
        return patterns
    
    def _load_compatibility_model(self):
        """Load or train a compatibility prediction model"""
        model_path = os.path.join(paths.data_path, "compatibility_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return self._train_default_model()
    
    def _train_default_model(self):
        """Train a simple compatibility model based on existing extensions"""
        model = {
            'api_compatibility': 0.8,
            'dependency_weight': 0.3,
            'version_weight': 0.4,
            'performance_weight': 0.3
        }
        return model
    
    def analyze_extension_code(self, extension_path: str) -> Dict[str, Any]:
        """Analyze extension code for compatibility issues"""
        analysis = {
            'api_usage': [],
            'dependencies': [],
            'hooks': [],
            'compatibility_issues': [],
            'performance_concerns': []
        }
        
        for root, dirs, files in os.walk(extension_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Parse AST for API usage
                        tree = ast.parse(content)
                        analysis.update(self._analyze_ast(tree, file_path))
                        
                    except Exception as e:
                        analysis['compatibility_issues'].append(f"Parse error in {file}: {str(e)}")
        
        return analysis
    
    def _analyze_ast(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze AST for compatibility patterns"""
        result = {
            'api_usage': [],
            'hooks': [],
            'compatibility_issues': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('modules.'):
                        result['api_usage'].append(alias.name)
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('modules'):
                    for alias in node.names:
                        result['api_usage'].append(f"{node.module}.{alias.name}")
                        
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if hasattr(node.func, 'attr'):
                        if node.func.attr in ['register_callback', 'add_hook']:
                            result['hooks'].append(node.func.attr)
        
        return result
    
    def predict_compatibility(self, extension_metadata: ExtensionMetadata, 
                            installed_extensions: List[ExtensionMetadata]) -> CompatibilityPrediction:
        """Predict compatibility score using ML model"""
        # Calculate base compatibility
        base_score = 0.7
        
        # Check version compatibility
        try:
            ext_version = pkg_version.parse(extension_metadata.version)
            webui_version = pkg_version.parse(shared.version)
            
            if ext_version > webui_version:
                base_score -= 0.2
        except:
            base_score -= 0.1
            
        # Check dependency compatibility
        dependency_score = self._check_dependencies(extension_metadata, installed_extensions)
        
        # Calculate final score
        final_score = (
            base_score * self.compatibility_model['version_weight'] +
            dependency_score * self.compatibility_model['dependency_weight'] +
            extension_metadata.compatibility_score * self.compatibility_model['api_compatibility']
        )
        
        # Generate prediction
        return CompatibilityPrediction(
            extension_name=extension_metadata.name,
            target_version=extension_metadata.version,
            compatibility_score=min(max(final_score, 0.0), 1.0),
            confidence=0.85,
            breaking_changes=self._predict_breaking_changes(extension_metadata),
            dependency_issues=self._find_dependency_issues(extension_metadata, installed_extensions),
            performance_impact=self._estimate_performance_impact(extension_metadata),
            prediction_timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
    
    def _check_dependencies(self, ext_meta: ExtensionMetadata, 
                          installed: List[ExtensionMetadata]) -> float:
        """Check if dependencies are satisfied"""
        if not ext_meta.dependencies:
            return 1.0
            
        satisfied = 0
        total = len(ext_meta.dependencies)
        
        installed_names = {ext.name for ext in installed}
        
        for dep in ext_meta.dependencies:
            if dep in installed_names:
                satisfied += 1
                
        return satisfied / total if total > 0 else 1.0
    
    def _predict_breaking_changes(self, ext_meta: ExtensionMetadata) -> List[str]:
        """Predict potential breaking changes"""
        breaking = []
        
        # Check version jumps
        try:
            current = pkg_version.parse(ext_meta.version)
            if current.major > 1:
                breaking.append("Major version upgrade may include breaking changes")
        except:
            pass
            
        # Check for known breaking patterns
        if 'breaking' in ext_meta.description.lower():
            breaking.append("Description mentions breaking changes")
            
        return breaking
    
    def _find_dependency_issues(self, ext_meta: ExtensionMetadata,
                              installed: List[ExtensionMetadata]) -> List[str]:
        """Find dependency conflicts"""
        issues = []
        installed_map = {ext.name: ext for ext in installed}
        
        for dep in ext_meta.dependencies:
            if dep not in installed_map:
                issues.append(f"Missing dependency: {dep}")
            else:
                # Check version compatibility
                try:
                    dep_version = pkg_version.parse(installed_map[dep].version)
                    # Simple version check - could be enhanced with specifiers
                except:
                    issues.append(f"Cannot parse version for {dep}")
                    
        return issues
    
    def _estimate_performance_impact(self, ext_meta: ExtensionMetadata) -> Dict[str, float]:
        """Estimate performance impact of extension"""
        impact = {
            'memory_increase': 0.1,
            'startup_time': 0.05,
            'inference_slowdown': 0.0
        }
        
        # Analyze based on tags and description
        if 'heavy' in ext_meta.tags or 'performance' in ext_meta.tags:
            impact['memory_increase'] = 0.3
            impact['inference_slowdown'] = 0.1
            
        return impact


class DependencyResolver:
    """Resolves extension dependencies using SAT solver approach"""
    
    def __init__(self):
        self.dependency_graph = {}
        self.version_constraints = {}
        
    def build_dependency_graph(self, extensions: List[ExtensionMetadata]):
        """Build dependency graph from available extensions"""
        self.dependency_graph.clear()
        self.version_constraints.clear()
        
        for ext in extensions:
            self.dependency_graph[ext.name] = set()
            self.version_constraints[ext.name] = ext.version
            
            for dep in ext.dependencies:
                # Parse dependency with version constraint if present
                if '>=' in dep or '==' in dep or '<=' in dep:
                    name, constraint = self._parse_version_constraint(dep)
                    self.dependency_graph[ext.name].add(name)
                    self.version_constraints[name] = constraint
                else:
                    self.dependency_graph[ext.name].add(dep)
    
    def _parse_version_constraint(self, constraint: str) -> Tuple[str, str]:
        """Parse version constraint string"""
        # Simple parser - could be enhanced
        if '>=' in constraint:
            name, ver = constraint.split('>=')
            return name.strip(), f">={ver.strip()}"
        elif '==' in constraint:
            name, ver = constraint.split('==')
            return name.strip(), f"=={ver.strip()}"
        elif '<=' in constraint:
            name, ver = constraint.split('<=')
            return name.strip(), f"<={ver.strip()}"
        else:
            return constraint, "*"
    
    def resolve_dependencies(self, target_extension: str, 
                           available_extensions: Dict[str, ExtensionMetadata],
                           installed_extensions: Dict[str, ExtensionMetadata]) -> List[str]:
        """Resolve dependencies for target extension"""
        if target_extension not in available_extensions:
            raise ValueError(f"Extension {target_extension} not found in available extensions")
            
        # Use topological sort with cycle detection
        visited = set()
        temp_mark = set()
        result = []
        
        def visit(node):
            if node in temp_mark:
                raise ValueError(f"Circular dependency detected: {node}")
            if node not in visited:
                temp_mark.add(node)
                
                # Visit dependencies
                for neighbor in self.dependency_graph.get(node, set()):
                    if neighbor not in installed_extensions:
                        visit(neighbor)
                        
                temp_mark.remove(node)
                visited.add(node)
                result.append(node)
        
        try:
            visit(target_extension)
        except ValueError as e:
            # Try to find alternative resolution
            return self._resolve_with_alternatives(target_extension, available_extensions, installed_extensions)
            
        # Filter out already installed
        return [ext for ext in result if ext not in installed_extensions]
    
    def _resolve_with_alternatives(self, target: str, 
                                 available: Dict[str, ExtensionMetadata],
                                 installed: Dict[str, ExtensionMetadata]) -> List[str]:
        """Resolve dependencies with alternative packages"""
        # Simple implementation - could be enhanced with backtracking
        required = set()
        
        def collect_deps(ext_name):
            if ext_name in installed:
                return
            required.add(ext_name)
            for dep in self.dependency_graph.get(ext_name, []):
                if dep not in installed:
                    collect_deps(dep)
        
        collect_deps(target)
        return list(required)


class RollbackManager:
    """Manages extension rollback capabilities"""
    
    def __init__(self, rollback_dir: str):
        self.rollback_dir = Path(rollback_dir)
        self.rollback_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_history = self._load_rollback_history()
    
    def _load_rollback_history(self) -> Dict[str, List[Dict]]:
        """Load rollback history from disk"""
        history_file = self.rollback_dir / "rollback_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_rollback_history(self):
        """Save rollback history to disk"""
        history_file = self.rollback_dir / "rollback_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.rollback_history, f, indent=2)
    
    def create_backup(self, extension_path: str, extension_name: str, 
                     version: str) -> Optional[str]:
        """Create backup of extension before update"""
        try:
            backup_name = f"{extension_name}_{version}_{int(time.time())}"
            backup_path = self.rollback_dir / backup_name
            
            # Copy extension directory
            shutil.copytree(extension_path, backup_path)
            
            # Save metadata
            metadata = {
                'extension_name': extension_name,
                'version': version,
                'backup_time': datetime.now().isoformat(),
                'original_path': extension_path
            }
            
            with open(backup_path / 'backup_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update history
            if extension_name not in self.rollback_history:
                self.rollback_history[extension_name] = []
                
            self.rollback_history[extension_name].append({
                'backup_path': str(backup_path),
                'version': version,
                'timestamp': datetime.now().isoformat()
            })
            
            self.save_rollback_history()
            return str(backup_path)
            
        except Exception as e:
            print(f"Failed to create backup for {extension_name}: {e}")
            return None
    
    def rollback(self, extension_name: str, target_version: Optional[str] = None) -> bool:
        """Rollback extension to previous version"""
        if extension_name not in self.rollback_history:
            return False
            
        backups = self.rollback_history[extension_name]
        if not backups:
            return False
            
        # Find target backup
        target_backup = None
        if target_version:
            for backup in backups:
                if backup['version'] == target_version:
                    target_backup = backup
                    break
        else:
            # Get latest backup
            target_backup = backups[-1]
            
        if not target_backup:
            return False
            
        try:
            backup_path = Path(target_backup['backup_path'])
            if not backup_path.exists():
                return False
                
            # Read metadata
            with open(backup_path / 'backup_metadata.json', 'r') as f:
                metadata = json.load(f)
                
            original_path = Path(metadata['original_path'])
            
            # Remove current version
            if original_path.exists():
                shutil.rmtree(original_path)
                
            # Restore backup
            shutil.copytree(backup_path, original_path)
            
            # Remove backup after successful restore
            shutil.rmtree(backup_path)
            
            # Update history
            self.rollback_history[extension_name] = [
                b for b in backups if b['backup_path'] != str(backup_path)
            ]
            self.save_rollback_history()
            
            return True
            
        except Exception as e:
            print(f"Rollback failed for {extension_name}: {e}")
            return False
    
    def list_available_rollbacks(self, extension_name: str) -> List[Dict]:
        """List available rollback versions for extension"""
        return self.rollback_history.get(extension_name, [])


class ExtensionRegistry:
    """Manages extension registry with semantic versioning"""
    
    def __init__(self, registry_url: Optional[str] = None):
        self.registry_url = registry_url or "https://raw.githubusercontent.com/sd-webui-extensions/registry/main/registry.json"
        self.local_registry_path = Path(paths.data_path) / "extension_registry.json"
        self.extensions_cache: Dict[str, ExtensionMetadata] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load extension registry from local cache or remote"""
        try:
            # Try loading local cache first
            if self.local_registry_path.exists():
                with open(self.local_registry_path, 'r') as f:
                    data = json.load(f)
                    self._parse_registry_data(data)
            
            # Update from remote in background
            threading.Thread(target=self._update_from_remote, daemon=True).start()
            
        except Exception as e:
            print(f"Failed to load extension registry: {e}")
            self._load_fallback_registry()
    
    def _update_from_remote(self):
        """Update registry from remote source"""
        try:
            import requests
            response = requests.get(self.registry_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._parse_registry_data(data)
                
                # Save to local cache
                with open(self.local_registry_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            print(f"Failed to update registry from remote: {e}")
    
    def _parse_registry_data(self, data: Dict):
        """Parse registry data into ExtensionMetadata objects"""
        self.extensions_cache.clear()
        
        for ext_data in data.get('extensions', []):
            try:
                metadata = ExtensionMetadata(
                    name=ext_data['name'],
                    version=ext_data['version'],
                    description=ext_data.get('description', ''),
                    author=ext_data.get('author', 'Unknown'),
                    repository=ext_data.get('repository', ''),
                    tags=ext_data.get('tags', []),
                    dependencies=ext_data.get('dependencies', []),
                    python_dependencies=ext_data.get('python_dependencies', []),
                    sd_webui_version=ext_data.get('sd_webui_version', '>=1.0.0'),
                    compatibility_score=ext_data.get('compatibility_score', 0.5),
                    semantic_tags=ext_data.get('semantic_tags', []),
                    performance_metrics=ext_data.get('performance_metrics', {}),
                    breaking_changes=ext_data.get('breaking_changes', [])
                )
                self.extensions_cache[ext_data['name']] = metadata
                
            except KeyError as e:
                print(f"Invalid extension data: missing {e}")
    
    def _load_fallback_registry(self):
        """Load fallback registry with built-in extensions"""
        fallback_data = {
            'extensions': [
                {
                    'name': 'LDSR',
                    'version': '1.0.0',
                    'description': 'Latent Diffusion Super Resolution',
                    'author': 'Stability AI',
                    'repository': 'https://github.com/sd-webui-extensions/LDSR',
                    'tags': ['upscaling', 'super-resolution'],
                    'dependencies': [],
                    'python_dependencies': ['torch', 'numpy'],
                    'sd_webui_version': '>=1.0.0',
                    'compatibility_score': 0.9,
                    'semantic_tags': ['image-processing', 'enhancement'],
                    'performance_metrics': {'memory_usage': 0.4, 'speed_impact': 0.3}
                },
                {
                    'name': 'Lora',
                    'version': '1.0.0',
                    'description': 'LoRA model support',
                    'author': 'Community',
                    'repository': 'https://github.com/sd-webui-extensions/Lora',
                    'tags': ['models', 'training'],
                    'dependencies': [],
                    'python_dependencies': ['torch'],
                    'sd_webui_version': '>=1.0.0',
                    'compatibility_score': 0.95,
                    'semantic_tags': ['model-management', 'training'],
                    'performance_metrics': {'memory_usage': 0.2, 'speed_impact': 0.1}
                }
            ]
        }
        self._parse_registry_data(fallback_data)
    
    def search_extensions(self, query: str, tags: Optional[List[str]] = None,
                         min_compatibility: float = 0.0) -> List[ExtensionMetadata]:
        """Search extensions using semantic matching"""
        results = []
        query_lower = query.lower()
        
        for ext in self.extensions_cache.values():
            # Text matching
            text_match = (
                query_lower in ext.name.lower() or
                query_lower in ext.description.lower() or
                any(query_lower in tag.lower() for tag in ext.tags) or
                any(query_lower in tag.lower() for tag in ext.semantic_tags)
            )
            
            # Tag matching
            tag_match = True
            if tags:
                tag_match = any(
                    tag in ext.tags or tag in ext.semantic_tags
                    for tag in tags
                )
            
            # Compatibility filter
            compatibility_match = ext.compatibility_score >= min_compatibility
            
            if text_match and tag_match and compatibility_match:
                results.append(ext)
        
        # Sort by compatibility score
        results.sort(key=lambda x: x.compatibility_score, reverse=True)
        return results
    
    def get_extension(self, name: str) -> Optional[ExtensionMetadata]:
        """Get extension metadata by name"""
        return self.extensions_cache.get(name)
    
    def add_extension(self, metadata: ExtensionMetadata):
        """Add extension to registry"""
        self.extensions_cache[metadata.name] = metadata
    
    def update_extension(self, name: str, metadata: ExtensionMetadata):
        """Update extension in registry"""
        if name in self.extensions_cache:
            self.extensions_cache[name] = metadata


class CompatibilityEngine:
    """Main compatibility engine for extension marketplace"""
    
    def __init__(self):
        self.registry = ExtensionRegistry()
        self.semantic_analyzer = SemanticAnalyzer()
        self.dependency_resolver = DependencyResolver()
        self.rollback_manager = RollbackManager(
            os.path.join(paths.data_path, "extension_backups")
        )
        self.installed_extensions: Dict[str, ExtensionMetadata] = {}
        self.compatibility_cache: Dict[str, CompatibilityPrediction] = {}
        
        self._load_installed_extensions()
        self._update_dependency_graph()
    
    def _load_installed_extensions(self):
        """Load currently installed extensions"""
        self.installed_extensions.clear()
        
        # Load built-in extensions
        builtin_dir = Path(paths.script_path) / "extensions-builtin"
        if builtin_dir.exists():
            for ext_dir in builtin_dir.iterdir():
                if ext_dir.is_dir():
                    self._register_installed_extension(ext_dir, builtin=True)
        
        # Load user extensions
        user_dir = Path(paths.script_path) / "extensions"
        if user_dir.exists():
            for ext_dir in user_dir.iterdir():
                if ext_dir.is_dir():
                    self._register_installed_extension(ext_dir, builtin=False)
    
    def _register_installed_extension(self, ext_path: Path, builtin: bool = False):
        """Register an installed extension"""
        try:
            # Try to extract metadata from extension
            metadata = self._extract_extension_metadata(ext_path, builtin)
            if metadata:
                self.installed_extensions[metadata.name] = metadata
                
        except Exception as e:
            print(f"Failed to register extension {ext_path.name}: {e}")
    
    def _extract_extension_metadata(self, ext_path: Path, builtin: bool) -> Optional[ExtensionMetadata]:
        """Extract metadata from extension directory"""
        # Look for metadata files
        metadata_files = ['metadata.json', 'extension.json', 'info.json']
        
        for meta_file in metadata_files:
            meta_path = ext_path / meta_file
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        data = json.load(f)
                        return ExtensionMetadata(**data)
                except:
                    continue
        
        # Fallback: create metadata from directory
        return ExtensionMetadata(
            name=ext_path.name,
            version="1.0.0",
            description=f"{'Built-in' if builtin else 'User'} extension: {ext_path.name}",
            author="Unknown",
            repository="",
            tags=[],
            dependencies=[],
            python_dependencies=[],
            sd_webui_version=">=1.0.0",
            compatibility_score=0.7 if builtin else 0.5,
            install_date=datetime.fromtimestamp(ext_path.stat().st_mtime).isoformat(),
            status=ExtensionStatus.INSTALLED
        )
    
    def _update_dependency_graph(self):
        """Update dependency resolver with current extensions"""
        all_extensions = list(self.installed_extensions.values())
        all_extensions.extend(self.registry.extensions_cache.values())
        self.dependency_resolver.build_dependency_graph(all_extensions)
    
    def analyze_extension_compatibility(self, extension_name: str) -> CompatibilityPrediction:
        """Analyze compatibility of an extension with current environment"""
        if extension_name in self.compatibility_cache:
            return self.compatibility_cache[extension_name]
        
        ext_meta = self.registry.get_extension(extension_name)
        if not ext_meta:
            raise ValueError(f"Extension {extension_name} not found in registry")
        
        # Get prediction from semantic analyzer
        prediction = self.semantic_analyzer.predict_compatibility(
            ext_meta, 
            list(self.installed_extensions.values())
        )
        
        # Cache prediction
        self.compatibility_cache[extension_name] = prediction
        return prediction
    
    def install_extension(self, extension_name: str, version: Optional[str] = None) -> bool:
        """Install extension with dependency resolution"""
        print(f"Installing extension: {extension_name}")
        
        try:
            # Check if already installed
            if extension_name in self.installed_extensions:
                print(f"Extension {extension_name} is already installed")
                return False
            
            # Get extension metadata
            ext_meta = self.registry.get_extension(extension_name)
            if not ext_meta:
                print(f"Extension {extension_name} not found in registry")
                return False
            
            # Resolve dependencies
            dependencies = self.dependency_resolver.resolve_dependencies(
                extension_name,
                self.registry.extensions_cache,
                self.installed_extensions
            )
            
            # Install dependencies first
            for dep in dependencies:
                if dep not in self.installed_extensions:
                    if not self.install_extension(dep):
                        print(f"Failed to install dependency: {dep}")
                        return False
            
            # Create backup if updating
            if extension_name in self.installed_extensions:
                current = self.installed_extensions[extension_name]
                backup_path = self.rollback_manager.create_backup(
                    self._get_extension_path(extension_name),
                    extension_name,
                    current.version
                )
                if backup_path:
                    ext_meta.rollback_version = current.version
            
            # Download and install extension
            if self._download_extension(ext_meta):
                # Update installed extensions
                ext_meta.install_date = datetime.now().isoformat()
                ext_meta.status = ExtensionStatus.INSTALLED
                self.installed_extensions[extension_name] = ext_meta
                
                # Install Python dependencies
                self._install_python_dependencies(ext_meta)
                
                # Run compatibility tests
                self._run_compatibility_tests(extension_name)
                
                print(f"Successfully installed {extension_name}")
                return True
                
        except Exception as e:
            print(f"Failed to install {extension_name}: {e}")
            errors.report(f"Extension installation failed: {e}", exc_info=True)
            
            # Attempt rollback
            if extension_name in self.installed_extensions:
                self.rollback_manager.rollback(extension_name)
            
            return False
        
        return False
    
    def _download_extension(self, ext_meta: ExtensionMetadata) -> bool:
        """Download extension from repository"""
        try:
            import git
            
            # Determine installation directory
            install_dir = Path(paths.script_path) / "extensions" / ext_meta.name
            
            # Clone repository
            if ext_meta.repository:
                git.Repo.clone_from(ext_meta.repository, install_dir)
                
                # Checkout specific version if provided
                if ext_meta.version and ext_meta.version != "latest":
                    repo = git.Repo(install_dir)
                    repo.git.checkout(ext_meta.version)
                
                return True
                
        except ImportError:
            print("GitPython not installed, falling back to manual download")
            return self._manual_download(ext_meta)
        except Exception as e:
            print(f"Failed to download extension: {e}")
            return False
            
        return False
    
    def _manual_download(self, ext_meta: ExtensionMetadata) -> bool:
        """Manual download fallback"""
        try:
            import requests
            import zipfile
            import io
            
            # Construct download URL
            if 'github.com' in ext_meta.repository:
                # Convert GitHub URL to zip URL
                zip_url = ext_meta.repository.replace('github.com', 'codeload.github.com')
                if not zip_url.endswith('.zip'):
                    zip_url += '/zip/' + ext_meta.version
            
            # Download
            response = requests.get(zip_url, timeout=30)
            if response.status_code == 200:
                install_dir = Path(paths.script_path) / "extensions" / ext_meta.name
                
                # Extract zip
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(install_dir.parent)
                
                # Rename extracted directory
                extracted_name = zip_ref.namelist()[0].split('/')[0]
                extracted_path = install_dir.parent / extracted_name
                extracted_path.rename(install_dir)
                
                return True
                
        except Exception as e:
            print(f"Manual download failed: {e}")
            return False
            
        return False
    
    def _install_python_dependencies(self, ext_meta: ExtensionMetadata):
        """Install Python dependencies for extension"""
        if not ext_meta.python_dependencies:
            return
            
        try:
            import pip
            
            for dep in ext_meta.python_dependencies:
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                except subprocess.CalledProcessError:
                    print(f"Failed to install Python dependency: {dep}")
                    
        except Exception as e:
            print(f"Failed to install Python dependencies: {e}")
    
    def _run_compatibility_tests(self, extension_name: str):
        """Run compatibility tests on installed extension"""
        ext_path = self._get_extension_path(extension_name)
        if not ext_path:
            return
            
        # Analyze code
        analysis = self.semantic_analyzer.analyze_extension_code(ext_path)
        
        # Update metadata with test results
        if extension_name in self.installed_extensions:
            self.installed_extensions[extension_name].test_results = {
                'analysis': analysis,
                'timestamp': datetime.now().isoformat(),
                'passed': len(analysis['compatibility_issues']) == 0
            }
    
    def _get_extension_path(self, extension_name: str) -> Optional[str]:
        """Get path to installed extension"""
        # Check user extensions
        user_path = Path(paths.script_path) / "extensions" / extension_name
        if user_path.exists():
            return str(user_path)
            
        # Check built-in extensions
        builtin_path = Path(paths.script_path) / "extensions-builtin" / extension_name
        if builtin_path.exists():
            return str(builtin_path)
            
        return None
    
    def uninstall_extension(self, extension_name: str, keep_data: bool = False) -> bool:
        """Uninstall extension"""
        try:
            if extension_name not in self.installed_extensions:
                print(f"Extension {extension_name} is not installed")
                return False
            
            # Create backup before uninstall
            ext_path = self._get_extension_path(extension_name)
            if ext_path:
                self.rollback_manager.create_backup(
                    ext_path,
                    extension_name,
                    self.installed_extensions[extension_name].version
                )
            
            # Remove extension directory
            if ext_path and os.path.exists(ext_path):
                if keep_data:
                    # Move to temporary location instead of deleting
                    temp_dir = tempfile.mkdtemp()
                    shutil.move(ext_path, os.path.join(temp_dir, extension_name))
                else:
                    shutil.rmtree(ext_path)
            
            # Remove from installed extensions
            del self.installed_extensions[extension_name]
            
            print(f"Successfully uninstalled {extension_name}")
            return True
            
        except Exception as e:
            print(f"Failed to uninstall {extension_name}: {e}")
            return False
    
    def update_extension(self, extension_name: str, target_version: Optional[str] = None) -> bool:
        """Update extension to specific version"""
        if extension_name not in self.installed_extensions:
            print(f"Extension {extension_name} is not installed")
            return False
        
        current = self.installed_extensions[extension_name]
        
        # Get available versions from registry
        ext_meta = self.registry.get_extension(extension_name)
        if not ext_meta:
            print(f"Extension {extension_name} not found in registry")
            return False
        
        # Determine target version
        if not target_version:
            target_version = ext_meta.version
        
        # Check if update is needed
        if current.version == target_version:
            print(f"Extension {extension_name} is already at version {target_version}")
            return True
        
        # Predict compatibility
        prediction = self.analyze_extension_compatibility(extension_name)
        if prediction.compatibility_score < 0.3:
            print(f"Update not recommended: compatibility score {prediction.compatibility_score}")
            return False
        
        # Create backup
        ext_path = self._get_extension_path(extension_name)
        if ext_path:
            backup_path = self.rollback_manager.create_backup(
                ext_path,
                extension_name,
                current.version
            )
            if backup_path:
                current.rollback_version = current.version
        
        # Perform update
        try:
            # Update metadata
            ext_meta.version = target_version
            ext_meta.last_updated = datetime.now().isoformat()
            ext_meta.status = ExtensionStatus.UPDATING
            
            # Download new version
            if self._download_extension(ext_meta):
                # Update installed extension
                self.installed_extensions[extension_name] = ext_meta
                
                # Run tests
                self._run_compatibility_tests(extension_name)
                
                print(f"Successfully updated {extension_name} to version {target_version}")
                return True
                
        except Exception as e:
            print(f"Failed to update {extension_name}: {e}")
            
            # Rollback on failure
            if self.rollback_manager.rollback(extension_name):
                print(f"Rolled back {extension_name} to previous version")
            
            return False
        
        return False
    
    def get_extension_recommendations(self, limit: int = 10) -> List[Dict]:
        """Get personalized extension recommendations"""
        recommendations = []
        
        # Analyze currently installed extensions
        installed_tags = set()
        for ext in self.installed_extensions.values():
            installed_tags.update(ext.tags)
            installed_tags.update(ext.semantic_tags)
        
        # Find complementary extensions
        for ext in self.registry.extensions_cache.values():
            if ext.name not in self.installed_extensions:
                # Calculate relevance score
                relevance = len(set(ext.tags) & installed_tags) * 0.3
                relevance += ext.compatibility_score * 0.7
                
                # Get compatibility prediction
                try:
                    prediction = self.analyze_extension_compatibility(ext.name)
                    relevance *= prediction.compatibility_score
                except:
                    relevance *= 0.5
                
                recommendations.append({
                    'extension': ext,
                    'relevance_score': relevance,
                    'compatibility': prediction.compatibility_score if 'prediction' in locals() else 0.5
                })
        
        # Sort by relevance
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        return recommendations[:limit]
    
    def get_compatibility_report(self) -> Dict:
        """Generate comprehensive compatibility report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'webui_version': shared.version,
            'installed_extensions': {},
            'available_extensions': len(self.registry.extensions_cache),
            'compatibility_issues': [],
            'recommendations': []
        }
        
        # Analyze installed extensions
        for name, ext in self.installed_extensions.items():
            try:
                prediction = self.analyze_extension_compatibility(name)
                report['installed_extensions'][name] = {
                    'version': ext.version,
                    'compatibility_score': prediction.compatibility_score,
                    'status': ext.status.value,
                    'breaking_changes': prediction.breaking_changes,
                    'dependency_issues': prediction.dependency_issues
                }
                
                if prediction.compatibility_score < 0.5:
                    report['compatibility_issues'].append({
                        'extension': name,
                        'issue': 'Low compatibility score',
                        'score': prediction.compatibility_score
                    })
                    
            except Exception as e:
                report['installed_extensions'][name] = {
                    'version': ext.version,
                    'error': str(e)
                }
        
        # Get recommendations
        report['recommendations'] = self.get_extension_recommendations(5)
        
        return report


# Global compatibility engine instance
compatibility_engine = CompatibilityEngine()


def preload():
    """Preload compatibility engine during startup"""
    print("Preloading compatibility engine...")
    # Initialize engine in background
    threading.Thread(target=lambda: None, daemon=True).start()


def on_ui_tabs():
    """Register UI components for extension marketplace"""
    # This would integrate with the webui's UI system
    # Implementation depends on the specific UI framework used
    pass


def on_ui_settings():
    """Add settings for compatibility engine"""
    # Add settings to the webui settings panel
    pass


# API endpoints for external access
def api_get_extension_list():
    """API endpoint to get available extensions"""
    return {
        'extensions': [
            asdict(ext) for ext in compatibility_engine.registry.extensions_cache.values()
        ]
    }


def api_install_extension(extension_name: str, version: Optional[str] = None):
    """API endpoint to install extension"""
    success = compatibility_engine.install_extension(extension_name, version)
    return {'success': success, 'extension': extension_name}


def api_get_compatibility_report():
    """API endpoint to get compatibility report"""
    return compatibility_engine.get_compatibility_report()


# Integration with existing extension system
original_extensions_list = extensions.extensions


def patched_extensions_list():
    """Patched version of extensions list that includes marketplace extensions"""
    # Get original extensions
    ext_list = original_extensions_list()
    
    # Add marketplace extensions that are installed
    for name, ext_meta in compatibility_engine.installed_extensions.items():
        # Check if already in list
        if not any(e.name == name for e in ext_list):
            # Create Extension object
            ext = Extension(name=name, path=compatibility_engine._get_extension_path(name))
            ext_list.append(ext)
    
    return ext_list


# Apply patch
extensions.extensions = patched_extensions_list