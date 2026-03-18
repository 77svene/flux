"""modules/extension_marketplace.py"""

import os
import json
import shutil
import hashlib
import tempfile
import subprocess
import importlib
import sys
import traceback
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import packaging.version
import packaging.specifiers
import packaging.requirements
import requests
import git
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from modules import extensions, shared, paths, errors
from modules.extensions import extensions as installed_extensions


class ExtensionStatus(Enum):
    AVAILABLE = "available"
    INSTALLED = "installed"
    UPDATE_AVAILABLE = "update_available"
    INCOMPATIBLE = "incompatible"
    DEPRECATED = "deprecated"
    TESTING = "testing"


class CompatibilityLevel(Enum):
    EXCELLENT = 0.9
    GOOD = 0.7
    FAIR = 0.5
    POOR = 0.3
    INCOMPATIBLE = 0.1


@dataclass
class ExtensionMetadata:
    name: str
    version: str
    description: str
    author: str
    repository_url: str
    branch: str = "main"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    python_dependencies: List[str] = field(default_factory=list)
    min_webui_version: Optional[str] = None
    max_webui_version: Optional[str] = None
    platforms: List[str] = field(default_factory=lambda: ["any"])
    gpu_requirements: Dict[str, Any] = field(default_factory=dict)
    performance_impact: str = "low"
    last_updated: Optional[str] = None
    download_count: int = 0
    rating: float = 0.0
    compatibility_score: float = 0.0
    test_results: Dict[str, Any] = field(default_factory=dict)
    semantic_version: packaging.version.Version = field(init=False)
    
    def __post_init__(self):
        self.semantic_version = packaging.version.parse(self.version)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['semantic_version'] = str(self.semantic_version)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtensionMetadata':
        if 'semantic_version' in data:
            del data['semantic_version']
        return cls(**data)


@dataclass
class ExtensionDependency:
    name: str
    version_constraint: str
    optional: bool = False
    
    def is_satisfied_by(self, version: str) -> bool:
        try:
            spec = packaging.specifiers.SpecifierSet(self.version_constraint)
            return packaging.version.parse(version) in spec
        except:
            return False


@dataclass
class InstallationRecord:
    extension_name: str
    version: str
    installation_path: str
    installed_at: str
    dependencies_installed: List[str]
    backup_path: Optional[str] = None
    rollback_available: bool = True


class CompatibilityPredictor:
    """ML-based compatibility prediction system"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(paths.models_path, "extension_compatibility_model.pkl")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = []
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize with default training data"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                self.feature_names = model_data.get('feature_names', [])
                self.is_trained = True
            else:
                self._initialize_default_model()
        except Exception as e:
            print(f"Error loading compatibility model: {e}")
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize with synthetic training data based on common patterns"""
        # Generate synthetic training data for initial model
        X_text = []
        y = []
        
        # Common compatible patterns
        compatible_patterns = [
            "simple extension no dependencies",
            "basic ui extension",
            "lightweight utility",
            "extension with standard dependencies",
            "well-maintained popular extension"
        ]
        
        # Common incompatible patterns
        incompatible_patterns = [
            "extension with deprecated api calls",
            "extension requiring specific torch version",
            "extension with complex system dependencies",
            "experimental extension with breaking changes",
            "extension with outdated requirements"
        ]
        
        for pattern in compatible_patterns:
            X_text.append(pattern)
            y.append(1)  # Compatible
        
        for pattern in incompatible_patterns:
            X_text.append(pattern)
            y.append(0)  # Incompatible
        
        # Train initial model
        if X_text:
            X = self.vectorizer.fit_transform(X_text)
            self.classifier.fit(X, y)
            self.is_trained = True
    
    def extract_features(self, metadata: ExtensionMetadata, 
                        system_info: Dict[str, Any]) -> np.ndarray:
        """Extract features from extension metadata for compatibility prediction"""
        features = []
        
        # Version compatibility features
        try:
            if metadata.min_webui_version:
                min_ver = packaging.version.parse(metadata.min_webui_version)
                current_ver = packaging.version.parse(
                    getattr(shared, 'version', '1.0.0') or '1.0.0'
                )
                features.append(1.0 if current_ver >= min_ver else 0.0)
            else:
                features.append(1.0)  # No minimum specified
                
            if metadata.max_webui_version:
                max_ver = packaging.version.parse(metadata.max_webui_version)
                current_ver = packaging.version.parse(
                    getattr(shared, 'version', '1.0.0') or '1.0.0'
                )
                features.append(1.0 if current_ver <= max_ver else 0.0)
            else:
                features.append(1.0)  # No maximum specified
        except:
            features.extend([0.5, 0.5])  # Default for unparseable versions
        
        # Dependency complexity
        features.append(len(metadata.dependencies) / 10.0)  # Normalized
        features.append(len(metadata.python_dependencies) / 10.0)
        
        # Platform compatibility
        current_platform = sys.platform
        platform_match = 1.0 if ("any" in metadata.platforms or 
                                current_platform in metadata.platforms) else 0.0
        features.append(platform_match)
        
        # GPU requirements check
        gpu_compatible = 1.0
        if metadata.gpu_requirements:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                required_mem = metadata.gpu_requirements.get('min_memory_gb', 0) * 1e9
                if gpu_mem < required_mem:
                    gpu_compatible = 0.5
            else:
                gpu_compatible = 0.0 if metadata.gpu_requirements.get('requires_gpu', False) else 1.0
        features.append(gpu_compatible)
        
        # Text features from description and tags
        text_features = f"{metadata.description} {' '.join(metadata.tags)}"
        
        if self.is_trained:
            text_vector = self.vectorizer.transform([text_features]).toarray()[0]
            features.extend(text_vector)
        else:
            # Fallback to simple text length feature
            features.append(len(text_features) / 1000.0)
        
        return np.array(features).reshape(1, -1)
    
    def predict_compatibility(self, metadata: ExtensionMetadata,
                            system_info: Optional[Dict[str, Any]] = None) -> float:
        """Predict compatibility score for an extension"""
        if system_info is None:
            system_info = self._get_system_info()
        
        try:
            features = self.extract_features(metadata, system_info)
            
            if self.is_trained:
                proba = self.classifier.predict_proba(features)[0]
                # Return probability of being compatible (class 1)
                return proba[1] if len(proba) > 1 else 0.5
            else:
                # Fallback heuristic scoring
                return self._heuristic_compatibility_score(metadata, system_info)
        except Exception as e:
            print(f"Error predicting compatibility: {e}")
            return 0.5  # Default moderate compatibility
    
    def _heuristic_compatibility_score(self, metadata: ExtensionMetadata,
                                     system_info: Dict[str, Any]) -> float:
        """Fallback heuristic compatibility scoring"""
        score = 0.7  # Start with moderate score
        
        # Version checks
        try:
            current_ver = packaging.version.parse(
                getattr(shared, 'version', '1.0.0') or '1.0.0'
            )
            
            if metadata.min_webui_version:
                min_ver = packaging.version.parse(metadata.min_webui_version)
                if current_ver < min_ver:
                    score *= 0.3
            
            if metadata.max_webui_version:
                max_ver = packaging.version.parse(metadata.max_webui_version)
                if current_ver > max_ver:
                    score *= 0.5
        except:
            score *= 0.8  # Penalize unparseable versions
        
        # Platform check
        current_platform = sys.platform
        if ("any" not in metadata.platforms and 
            current_platform not in metadata.platforms):
            score *= 0.1
        
        # GPU requirements
        if metadata.gpu_requirements.get('requires_gpu', False):
            if not torch.cuda.is_available():
                score *= 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for compatibility prediction"""
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        
        try:
            info['webui_version'] = getattr(shared, 'version', 'unknown')
        except:
            info['webui_version'] = 'unknown'
        
        return info
    
    def update_model(self, training_data: List[Tuple[ExtensionMetadata, bool]]):
        """Update the compatibility model with new training data"""
        if not training_data:
            return
        
        X_text = []
        y = []
        
        for metadata, is_compatible in training_data:
            text = f"{metadata.description} {' '.join(metadata.tags)}"
            X_text.append(text)
            y.append(1 if is_compatible else 0)
        
        try:
            X = self.vectorizer.fit_transform(X_text)
            self.classifier.fit(X, y)
            self.is_trained = True
            
            # Save updated model
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'feature_names': self.feature_names
            }
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(model_data, self.model_path)
        except Exception as e:
            print(f"Error updating compatibility model: {e}")


class DependencyResolver:
    """Automated dependency resolution with conflict detection"""
    
    def __init__(self, registry: Dict[str, ExtensionMetadata]):
        self.registry = registry
        self.installed_versions: Dict[str, str] = {}
        self._load_installed_versions()
    
    def _load_installed_versions(self):
        """Load currently installed extension versions"""
        for ext in installed_extensions:
            if hasattr(ext, 'version'):
                self.installed_versions[ext.name] = ext.version
    
    def resolve_dependencies(self, extension_name: str, 
                           version: Optional[str] = None) -> List[Tuple[str, str]]:
        """Resolve all dependencies for an extension"""
        if extension_name not in self.registry:
            raise ValueError(f"Extension {extension_name} not found in registry")
        
        metadata = self.registry[extension_name]
        if version:
            # Find specific version metadata
            metadata = self._find_version_metadata(extension_name, version)
            if not metadata:
                raise ValueError(f"Version {version} not found for {extension_name}")
        
        resolved = []
        to_process = [(extension_name, metadata.version)]
        processed = set()
        
        while to_process:
            current_name, current_version = to_process.pop(0)
            
            if current_name in processed:
                continue
            
            processed.add(current_name)
            resolved.append((current_name, current_version))
            
            # Get dependencies for current extension
            current_metadata = self._find_version_metadata(current_name, current_version)
            if not current_metadata:
                continue
            
            for dep in current_metadata.dependencies:
                dep_name, dep_constraint = self._parse_dependency(dep)
                
                if dep_name in self.registry:
                    # Find compatible version
                    compatible_version = self._find_compatible_version(
                        dep_name, dep_constraint
                    )
                    if compatible_version:
                        to_process.append((dep_name, compatible_version))
                    else:
                        raise DependencyConflictError(
                            f"No compatible version found for {dep_name} {dep_constraint}"
                        )
        
        # Check for conflicts
        self._check_conflicts(resolved)
        
        # Return in installation order (dependencies first)
        return self._topological_sort(resolved)
    
    def _parse_dependency(self, dep_string: str) -> Tuple[str, str]:
        """Parse dependency string like 'extension_name>=1.0.0'"""
        try:
            req = packaging.requirements.Requirement(dep_string)
            return req.name, str(req.specifier) if req.specifier else "*"
        except:
            # Fallback for simple format
            if '>=' in dep_string:
                name, spec = dep_string.split('>=', 1)
                return name.strip(), f">={spec.strip()}"
            elif '==' in dep_string:
                name, spec = dep_string.split('==', 1)
                return name.strip(), f"=={spec.strip()}"
            else:
                return dep_string.strip(), "*"
    
    def _find_compatible_version(self, extension_name: str, 
                               constraint: str) -> Optional[str]:
        """Find a compatible version of an extension"""
        if extension_name not in self.registry:
            return None
        
        metadata = self.registry[extension_name]
        
        # Check if already installed version satisfies constraint
        if extension_name in self.installed_versions:
            installed_ver = self.installed_versions[extension_name]
            if self._version_satisfies(installed_ver, constraint):
                return installed_ver
        
        # Check if latest version satisfies constraint
        if self._version_satisfies(metadata.version, constraint):
            return metadata.version
        
        # Would need to query registry for all versions - simplified here
        return None
    
    def _version_satisfies(self, version: str, constraint: str) -> bool:
        """Check if version satisfies constraint"""
        try:
            spec = packaging.specifiers.SpecifierSet(constraint)
            return packaging.version.parse(version) in spec
        except:
            return constraint == "*"
    
    def _find_version_metadata(self, name: str, version: str) -> Optional[ExtensionMetadata]:
        """Find metadata for specific version (simplified)"""
        if name in self.registry and self.registry[name].version == version:
            return self.registry[name]
        return None
    
    def _check_conflicts(self, resolved: List[Tuple[str, str]]):
        """Check for dependency conflicts"""
        version_map: Dict[str, str] = {}
        
        for name, version in resolved:
            if name in version_map:
                if version_map[name] != version:
                    raise DependencyConflictError(
                        f"Version conflict for {name}: {version_map[name]} vs {version}"
                    )
            else:
                version_map[name] = version
    
    def _topological_sort(self, dependencies: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Sort dependencies in installation order (dependencies first)"""
        # Build dependency graph
        graph: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}
        version_map: Dict[str, str] = {}
        
        for name, version in dependencies:
            version_map[name] = version
            if name not in graph:
                graph[name] = []
                in_degree[name] = 0
        
        # Build edges
        for name, version in dependencies:
            metadata = self._find_version_metadata(name, version)
            if metadata:
                for dep in metadata.dependencies:
                    dep_name, _ = self._parse_dependency(dep)
                    if dep_name in graph:
                        graph[dep_name].append(name)
                        in_degree[name] += 1
        
        # Kahn's algorithm
        queue = [name for name in graph if in_degree[name] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append((node, version_map[node]))
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(dependencies):
            raise DependencyConflictError("Circular dependency detected")
        
        return result


class DependencyConflictError(Exception):
    """Exception raised when dependency conflicts are detected"""
    pass


class ExtensionMarketplace:
    """Main marketplace manager with AI-powered discovery and installation"""
    
    def __init__(self, registry_url: str = None):
        self.registry_url = registry_url or "https://raw.githubusercontent.com/sd-webui-extensions/registry/main/registry.json"
        self.registry: Dict[str, ExtensionMetadata] = {}
        self.compatibility_predictor = CompatibilityPredictor()
        self.dependency_resolver = DependencyResolver(self.registry)
        self.installation_history: List[InstallationRecord] = []
        self.marketplace_dir = os.path.join(paths.data_path, "extension_marketplace")
        self.cache_dir = os.path.join(self.marketplace_dir, "cache")
        self.backup_dir = os.path.join(self.marketplace_dir, "backups")
        self.history_file = os.path.join(self.marketplace_dir, "installation_history.json")
        
        # Create directories
        os.makedirs(self.marketplace_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Load data
        self._load_registry()
        self._load_installation_history()
        
        # Initialize with installed extensions
        self._sync_installed_extensions()
    
    def _load_registry(self):
        """Load extension registry from cache or remote"""
        cache_file = os.path.join(self.cache_dir, "registry.json")
        
        try:
            # Try to load from cache first
            if os.path.exists(cache_file):
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < 3600:  # 1 hour cache
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        registry_data = json.load(f)
                    self._parse_registry_data(registry_data)
                    return
            
            # Fetch from remote
            response = requests.get(self.registry_url, timeout=10)
            response.raise_for_status()
            registry_data = response.json()
            
            # Save to cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2)
            
            self._parse_registry_data(registry_data)
            
        except Exception as e:
            print(f"Error loading registry: {e}")
            # Fallback to empty registry
            self.registry = {}
    
    def _parse_registry_data(self, data: Dict[str, Any]):
        """Parse registry data into ExtensionMetadata objects"""
        self.registry = {}
        
        for name, ext_data in data.get('extensions', {}).items():
            try:
                metadata = ExtensionMetadata.from_dict(ext_data)
                self.registry[name] = metadata
                
                # Update compatibility score
                metadata.compatibility_score = self.compatibility_predictor.predict_compatibility(
                    metadata
                )
            except Exception as e:
                print(f"Error parsing extension {name}: {e}")
    
    def _load_installation_history(self):
        """Load installation history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    self.installation_history = [
                        InstallationRecord(**record) for record in history_data
                    ]
        except Exception as e:
            print(f"Error loading installation history: {e}")
            self.installation_history = []
    
    def _save_installation_history(self):
        """Save installation history to file"""
        try:
            history_data = [asdict(record) for record in self.installation_history]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            print(f"Error saving installation history: {e}")
    
    def _sync_installed_extensions(self):
        """Sync installed extensions with registry"""
        for ext in installed_extensions:
            if ext.name in self.registry:
                # Update status
                metadata = self.registry[ext.name]
                installed_version = getattr(ext, 'version', '0.0.0')
                
                try:
                    if packaging.version.parse(installed_version) < metadata.semantic_version:
                        metadata.status = ExtensionStatus.UPDATE_AVAILABLE
                    else:
                        metadata.status = ExtensionStatus.INSTALLED
                except:
                    metadata.status = ExtensionStatus.INSTALLED
    
    def search_extensions(self, query: str = "", tags: List[str] = None,
                         min_compatibility: float = 0.0,
                         sort_by: str = "compatibility") -> List[ExtensionMetadata]:
        """Search extensions with AI-powered relevance ranking"""
        results = []
        
        for name, metadata in self.registry.items():
            # Filter by query
            if query:
                query_lower = query.lower()
                if (query_lower not in name.lower() and 
                    query_lower not in metadata.description.lower() and
                    not any(query_lower in tag.lower() for tag in metadata.tags)):
                    continue
            
            # Filter by tags
            if tags:
                if not any(tag in metadata.tags for tag in tags):
                    continue
            
            # Filter by compatibility
            if metadata.compatibility_score < min_compatibility:
                continue
            
            results.append(metadata)
        
        # Sort results
        if sort_by == "compatibility":
            results.sort(key=lambda x: x.compatibility_score, reverse=True)
        elif sort_by == "popularity":
            results.sort(key=lambda x: x.download_count, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda x: x.rating, reverse=True)
        elif sort_by == "recent":
            results.sort(key=lambda x: x.last_updated or "", reverse=True)
        
        return results
    
    def get_extension_details(self, extension_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an extension"""
        if extension_name not in self.registry:
            return None
        
        metadata = self.registry[extension_name]
        
        # Get installation status
        installed = False
        installed_version = None
        for ext in installed_extensions:
            if ext.name == extension_name:
                installed = True
                installed_version = getattr(ext, 'version', 'unknown')
                break
        
        # Get dependencies
        dependencies = []
        try:
            deps = self.dependency_resolver.resolve_dependencies(extension_name)
            for dep_name, dep_version in deps:
                if dep_name != extension_name:
                    dep_metadata = self.registry.get(dep_name)
                    dependencies.append({
                        'name': dep_name,
                        'version': dep_version,
                        'installed': dep_name in self.dependency_resolver.installed_versions,
                        'description': dep_metadata.description if dep_metadata else ""
                    })
        except DependencyConflictError as e:
            dependencies = [{'error': str(e)}]
        
        # Get installation history
        history = [
            record for record in self.installation_history
            if record.extension_name == extension_name
        ]
        
        return {
            'metadata': metadata.to_dict(),
            'installed': installed,
            'installed_version': installed_version,
            'dependencies': dependencies,
            'installation_history': [asdict(h) for h in history[-5:]],  # Last 5 entries
            'compatibility_details': {
                'score': metadata.compatibility_score,
                'level': self._get_compatibility_level(metadata.compatibility_score),
                'factors': self._get_compatibility_factors(metadata)
            }
        }
    
    def _get_compatibility_level(self, score: float) -> str:
        """Convert compatibility score to level"""
        if score >= CompatibilityLevel.EXCELLENT.value:
            return "excellent"
        elif score >= CompatibilityLevel.GOOD.value:
            return "good"
        elif score >= CompatibilityLevel.FAIR.value:
            return "fair"
        elif score >= CompatibilityLevel.POOR.value:
            return "poor"
        else:
            return "incompatible"
    
    def _get_compatibility_factors(self, metadata: ExtensionMetadata) -> List[str]:
        """Get factors affecting compatibility"""
        factors = []
        
        # Version compatibility
        try:
            current_ver = packaging.version.parse(
                getattr(shared, 'version', '1.0.0') or '1.0.0'
            )
            
            if metadata.min_webui_version:
                min_ver = packaging.version.parse(metadata.min_webui_version)
                if current_ver < min_ver:
                    factors.append(f"Requires webui version >= {metadata.min_webui_version}")
            
            if metadata.max_webui_version:
                max_ver = packaging.version.parse(metadata.max_webui_version)
                if current_ver > max_ver:
                    factors.append(f"Designed for webui version <= {metadata.max_webui_version}")
        except:
            factors.append("Version information unparseable")
        
        # Platform compatibility
        current_platform = sys.platform
        if ("any" not in metadata.platforms and 
            current_platform not in metadata.platforms):
            factors.append(f"Not compatible with {current_platform}")
        
        # GPU requirements
        if metadata.gpu_requirements.get('requires_gpu', False):
            if not torch.cuda.is_available():
                factors.append("Requires GPU but none available")
        
        # Dependencies
        if metadata.dependencies:
            factors.append(f"Has {len(metadata.dependencies)} dependencies")
        
        return factors
    
    def install_extension(self, extension_name: str, 
                         version: Optional[str] = None,
                         force: bool = False) -> Tuple[bool, str]:
        """Install an extension with dependency resolution and rollback support"""
        if extension_name not in self.registry:
            return False, f"Extension {extension_name} not found in registry"
        
        metadata = self.registry[extension_name]
        version = version or metadata.version
        
        # Check compatibility
        if not force and metadata.compatibility_score < CompatibilityLevel.POOR.value:
            return False, f"Extension has low compatibility score: {metadata.compatibility_score:.2f}"
        
        try:
            # Resolve dependencies
            dependencies = self.dependency_resolver.resolve_dependencies(
                extension_name, version
            )
            
            # Create backup of existing installations
            backup_paths = self._create_backups(dependencies)
            
            # Install dependencies first
            installed_deps = []
            for dep_name, dep_version in dependencies:
                if dep_name == extension_name:
                    continue  # Skip the main extension for now
                
                if dep_name not in self.dependency_resolver.installed_versions:
                    success, message = self._install_single_extension(dep_name, dep_version)
                    if not success:
                        # Rollback on failure
                        self._rollback_installation(backup_paths)
                        return False, f"Failed to install dependency {dep_name}: {message}"
                    installed_deps.append(dep_name)
            
            # Install main extension
            success, message = self._install_single_extension(extension_name, version)
            if not success:
                # Rollback on failure
                self._rollback_installation(backup_paths)
                return False, f"Failed to install {extension_name}: {message}"
            
            # Record installation
            record = InstallationRecord(
                extension_name=extension_name,
                version=version,
                installation_path=os.path.join(paths.extensions_dir, extension_name),
                installed_at=datetime.now().isoformat(),
                dependencies_installed=installed_deps,
                backup_path=backup_paths.get(extension_name)
            )
            self.installation_history.append(record)
            self._save_installation_history()
            
            # Update registry statistics
            metadata.download_count += 1
            metadata.last_updated = datetime.now().isoformat()
            
            return True, f"Successfully installed {extension_name} {version}"
            
        except DependencyConflictError as e:
            return False, f"Dependency conflict: {str(e)}"
        except Exception as e:
            return False, f"Installation failed: {str(e)}"
    
    def _install_single_extension(self, extension_name: str, 
                                version: str) -> Tuple[bool, str]:
        """Install a single extension"""
        metadata = self.registry.get(extension_name)
        if not metadata:
            return False, f"Extension {extension_name} not found"
        
        install_path = os.path.join(paths.extensions_dir, extension_name)
        
        try:
            # Remove existing installation if present
            if os.path.exists(install_path):
                shutil.rmtree(install_path)
            
            # Clone repository
            repo_url = metadata.repository_url
            branch = metadata.branch
            
            print(f"Cloning {repo_url} (branch: {branch})...")
            git.Repo.clone_from(
                repo_url,
                install_path,
                branch=branch,
                depth=1  # Shallow clone for efficiency
            )
            
            # Install Python dependencies if specified
            if metadata.python_dependencies:
                self._install_python_dependencies(metadata.python_dependencies)
            
            # Run post-install script if exists
            post_install_script = os.path.join(install_path, "post_install.py")
            if os.path.exists(post_install_script):
                self._run_post_install(post_install_script)
            
            # Update installed versions
            self.dependency_resolver.installed_versions[extension_name] = version
            
            return True, "Installation successful"
            
        except git.exc.GitCommandError as e:
            return False, f"Git error: {str(e)}"
        except Exception as e:
            return False, f"Installation error: {str(e)}"
    
    def _install_python_dependencies(self, dependencies: List[str]):
        """Install Python dependencies using pip"""
        if not dependencies:
            return
        
        try:
            # Use subprocess to install dependencies
            cmd = [sys.executable, "-m", "pip", "install"] + dependencies
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install some Python dependencies: {e}")
    
    def _run_post_install(self, script_path: str):
        """Run post-installation script"""
        try:
            spec = importlib.util.spec_from_file_location("post_install", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'post_install'):
                module.post_install()
        except Exception as e:
            print(f"Warning: Post-install script failed: {e}")
    
    def _create_backups(self, dependencies: List[Tuple[str, str]]) -> Dict[str, str]:
        """Create backups of existing extensions before installation"""
        backup_paths = {}
        
        for dep_name, _ in dependencies:
            install_path = os.path.join(paths.extensions_dir, dep_name)
            if os.path.exists(install_path):
                # Create backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{dep_name}_{timestamp}"
                backup_path = os.path.join(self.backup_dir, backup_name)
                
                try:
                    shutil.copytree(install_path, backup_path)
                    backup_paths[dep_name] = backup_path
                except Exception as e:
                    print(f"Warning: Failed to backup {dep_name}: {e}")
        
        return backup_paths
    
    def _rollback_installation(self, backup_paths: Dict[str, str]):
        """Rollback installation using backups"""
        for dep_name, backup_path in backup_paths.items():
            install_path = os.path.join(paths.extensions_dir, dep_name)
            
            try:
                # Remove failed installation
                if os.path.exists(install_path):
                    shutil.rmtree(install_path)
                
                # Restore from backup
                if os.path.exists(backup_path):
                    shutil.copytree(backup_path, install_path)
                    print(f"Restored {dep_name} from backup")
            except Exception as e:
                print(f"Failed to rollback {dep_name}: {e}")
    
    def uninstall_extension(self, extension_name: str, 
                          remove_data: bool = False) -> Tuple[bool, str]:
        """Uninstall an extension with optional data removal"""
        install_path = os.path.join(paths.extensions_dir, extension_name)
        
        if not os.path.exists(install_path):
            return False, f"Extension {extension_name} is not installed"
        
        try:
            # Check if other extensions depend on this
            dependent_extensions = self._find_dependent_extensions(extension_name)
            if dependent_extensions:
                return False, f"Cannot uninstall: {', '.join(dependent_extensions)} depend on this extension"
            
            # Create final backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{extension_name}_uninstall_{timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            shutil.copytree(install_path, backup_path)
            
            # Remove extension
            if remove_data:
                shutil.rmtree(install_path)
            else:
                # Move to temporary location for potential recovery
                temp_path = os.path.join(tempfile.gettempdir(), f"sd_webui_ext_{extension_name}")
                shutil.move(install_path, temp_path)
            
            # Update records
            if extension_name in self.dependency_resolver.installed_versions:
                del self.dependency_resolver.installed_versions[extension_name]
            
            # Update registry status
            if extension_name in self.registry:
                self.registry[extension_name].status = ExtensionStatus.AVAILABLE
            
            return True, f"Successfully uninstalled {extension_name}"
            
        except Exception as e:
            return False, f"Uninstallation failed: {str(e)}"
    
    def _find_dependent_extensions(self, extension_name: str) -> List[str]:
        """Find extensions that depend on the given extension"""
        dependents = []
        
        for name, metadata in self.registry.items():
            if name == extension_name:
                continue
            
            for dep in metadata.dependencies:
                dep_name, _ = self.dependency_resolver._parse_dependency(dep)
                if dep_name == extension_name:
                    # Check if this extension is installed
                    if name in self.dependency_resolver.installed_versions:
                        dependents.append(name)
                    break
        
        return dependents
    
    def update_extension(self, extension_name: str) -> Tuple[bool, str]:
        """Update an extension to the latest version"""
        if extension_name not in self.registry:
            return False, f"Extension {extension_name} not found in registry"
        
        # Check if installed
        installed_version = self.dependency_resolver.installed_versions.get(extension_name)
        if not installed_version:
            return False, f"Extension {extension_name} is not installed"
        
        metadata = self.registry[extension_name]
        
        # Check if update is available
        try:
            if packaging.version.parse(installed_version) >= metadata.semantic_version:
                return False, f"Already at latest version {installed_version}"
        except:
            pass
        
        # Install new version
        return self.install_extension(extension_name, metadata.version, force=True)
    
    def rollback_extension(self, extension_name: str, 
                          backup_timestamp: Optional[str] = None) -> Tuple[bool, str]:
        """Rollback an extension to a previous version"""
        # Find backup
        if backup_timestamp:
            backup_name = f"{extension_name}_{backup_timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_name)
        else:
            # Find latest backup
            backups = []
            for item in os.listdir(self.backup_dir):
                if item.startswith(f"{extension_name}_"):
                    backups.append(item)
            
            if not backups:
                return False, f"No backups found for {extension_name}"
            
            backups.sort(reverse=True)
            backup_path = os.path.join(self.backup_dir, backups[0])
        
        if not os.path.exists(backup_path):
            return False, f"Backup not found: {backup_path}"
        
        try:
            install_path = os.path.join(paths.extensions_dir, extension_name)
            
            # Remove current installation
            if os.path.exists(install_path):
                shutil.rmtree(install_path)
            
            # Restore from backup
            shutil.copytree(backup_path, install_path)
            
            return True, f"Successfully rolled back {extension_name}"
            
        except Exception as e:
            return False, f"Rollback failed: {str(e)}"
    
    def get_installed_extensions(self) -> List[Dict[str, Any]]:
        """Get list of installed extensions with status"""
        installed = []
        
        for ext in installed_extensions:
            ext_info = {
                'name': ext.name,
                'version': getattr(ext, 'version', 'unknown'),
                'enabled': getattr(ext, 'enabled', True),
                'path': ext.path,
                'status': 'installed'
            }
            
            # Check for updates
            if ext.name in self.registry:
                metadata = self.registry[ext.name]
                try:
                    installed_ver = packaging.version.parse(ext_info['version'])
                    if installed_ver < metadata.semantic_version:
                        ext_info['status'] = 'update_available'
                        ext_info['latest_version'] = metadata.version
                except:
                    pass
            
            installed.append(ext_info)
        
        return installed
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        total_extensions = len(self.registry)
        installed_count = len(self.dependency_resolver.installed_versions)
        
        # Calculate average compatibility
        if self.registry:
            avg_compatibility = sum(
                ext.compatibility_score for ext in self.registry.values()
            ) / total_extensions
        else:
            avg_compatibility = 0.0
        
        # Get popular tags
        tag_counts = {}
        for metadata in self.registry.values():
            for tag in metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_extensions': total_extensions,
            'installed_extensions': installed_count,
            'average_compatibility': avg_compatibility,
            'popular_tags': popular_tags,
            'cache_size': self._get_cache_size(),
            'backup_count': len(os.listdir(self.backup_dir)) if os.path.exists(self.backup_dir) else 0
        }
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
    
    def clear_cache(self):
        """Clear marketplace cache"""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def update_compatibility_model(self):
        """Update the compatibility prediction model with new data"""
        # Collect training data from installation history
        training_data = []
        
        for record in self.installation_history:
            if record.extension_name in self.registry:
                metadata = self.registry[record.extension_name]
                # Assume successful installation means compatible
                training_data.append((metadata, True))
        
        # Add negative examples (extensions with known issues)
        # This would come from user reports or automated testing
        
        if training_data:
            self.compatibility_predictor.update_model(training_data)


# Global marketplace instance
marketplace = None

def initialize_marketplace():
    """Initialize the global marketplace instance"""
    global marketplace
    if marketplace is None:
        marketplace = ExtensionMarketplace()
    return marketplace

def get_marketplace():
    """Get the global marketplace instance"""
    if marketplace is None:
        return initialize_marketplace()
    return marketplace

# Integration with existing extension system
original_extensions_load = extensions.load_extensions

def patched_load_extensions():
    """Patched version of load_extensions that integrates with marketplace"""
    # Call original function
    original_extensions_load()
    
    # Initialize marketplace if not already done
    try:
        get_marketplace()
    except Exception as e:
        print(f"Error initializing extension marketplace: {e}")

# Apply the patch
extensions.load_extensions = patched_load_extensions

# API endpoints for web UI
def api_search_extensions(query: str = "", tags: str = "", 
                         min_compatibility: float = 0.0,
                         sort_by: str = "compatibility") -> List[Dict[str, Any]]:
    """API endpoint for searching extensions"""
    mp = get_marketplace()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    results = mp.search_extensions(query, tag_list, min_compatibility, sort_by)
    return [ext.to_dict() for ext in results]

def api_get_extension_details(extension_name: str) -> Optional[Dict[str, Any]]:
    """API endpoint for getting extension details"""
    mp = get_marketplace()
    return mp.get_extension_details(extension_name)

def api_install_extension(extension_name: str, version: str = None,
                         force: bool = False) -> Dict[str, Any]:
    """API endpoint for installing an extension"""
    mp = get_marketplace()
    success, message = mp.install_extension(extension_name, version, force)
    return {'success': success, 'message': message}

def api_uninstall_extension(extension_name: str, 
                           remove_data: bool = False) -> Dict[str, Any]:
    """API endpoint for uninstalling an extension"""
    mp = get_marketplace()
    success, message = mp.uninstall_extension(extension_name, remove_data)
    return {'success': success, 'message': message}

def api_update_extension(extension_name: str) -> Dict[str, Any]:
    """API endpoint for updating an extension"""
    mp = get_marketplace()
    success, message = mp.update_extension(extension_name)
    return {'success': success, 'message': message}

def api_rollback_extension(extension_name: str, 
                          backup_timestamp: str = None) -> Dict[str, Any]:
    """API endpoint for rolling back an extension"""
    mp = get_marketplace()
    success, message = mp.rollback_extension(extension_name, backup_timestamp)
    return {'success': success, 'message': message}

def api_get_installed_extensions() -> List[Dict[str, Any]]:
    """API endpoint for getting installed extensions"""
    mp = get_marketplace()
    return mp.get_installed_extensions()

def api_get_marketplace_stats() -> Dict[str, Any]:
    """API endpoint for getting marketplace statistics"""
    mp = get_marketplace()
    return mp.get_marketplace_stats()

def api_clear_cache() -> Dict[str, Any]:
    """API endpoint for clearing marketplace cache"""
    mp = get_marketplace()
    mp.clear_cache()
    return {'success': True, 'message': 'Cache cleared'}

# Register API endpoints if webui is available
try:
    from modules import api
    
    # Add marketplace API endpoints
    if hasattr(api, 'add_api_route'):
        api.add_api_route("/sdapi/v1/marketplace/search", api_search_extensions, methods=["GET"])
        api.add_api_route("/sdapi/v1/marketplace/extension/{extension_name}", api_get_extension_details, methods=["GET"])
        api.add_api_route("/sdapi/v1/marketplace/install/{extension_name}", api_install_extension, methods=["POST"])
        api.add_api_route("/sdapi/v1/marketplace/uninstall/{extension_name}", api_uninstall_extension, methods=["POST"])
        api.add_api_route("/sdapi/v1/marketplace/update/{extension_name}", api_update_extension, methods=["POST"])
        api.add_api_route("/sdapi/v1/marketplace/rollback/{extension_name}", api_rollback_extension, methods=["POST"])
        api.add_api_route("/sdapi/v1/marketplace/installed", api_get_installed_extensions, methods=["GET"])
        api.add_api_route("/sdapi/v1/marketplace/stats", api_get_marketplace_stats, methods=["GET"])
        api.add_api_route("/sdapi/v1/marketplace/clear-cache", api_clear_cache, methods=["POST"])
except ImportError:
    pass  # API not available

# Command line interface
def marketplace_cli():
    """Command line interface for the marketplace"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extension Marketplace CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search extensions')
    search_parser.add_argument('query', nargs='?', default='', help='Search query')
    search_parser.add_argument('--tags', help='Comma-separated tags')
    search_parser.add_argument('--min-compatibility', type=float, default=0.0,
                              help='Minimum compatibility score')
    search_parser.add_argument('--sort', choices=['compatibility', 'popularity', 'rating', 'recent'],
                              default='compatibility', help='Sort order')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install extension')
    install_parser.add_argument('extension_name', help='Extension name')
    install_parser.add_argument('--version', help='Specific version to install')
    install_parser.add_argument('--force', action='store_true',
                               help='Force installation despite low compatibility')
    
    # Uninstall command
    uninstall_parser = subparsers.add_parser('uninstall', help='Uninstall extension')
    uninstall_parser.add_argument('extension_name', help='Extension name')
    uninstall_parser.add_argument('--remove-data', action='store_true',
                                 help='Remove extension data')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update extension')
    update_parser.add_argument('extension_name', help='Extension name')
    
    # List command
    subparsers.add_parser('list', help='List installed extensions')
    
    # Stats command
    subparsers.add_parser('stats', help='Show marketplace statistics')
    
    args = parser.parse_args()
    
    mp = get_marketplace()
    
    if args.command == 'search':
        tag_list = [t.strip() for t in args.tags.split(",")] if args.tags else None
        results = mp.search_extensions(args.query, tag_list, 
                                      args.min_compatibility, args.sort)
        
        print(f"Found {len(results)} extensions:")
        for ext in results:
            status = "✓" if ext.status == ExtensionStatus.INSTALLED else " "
            print(f"{status} {ext.name} v{ext.version} - {ext.description}")
            print(f"   Compatibility: {ext.compatibility_score:.2f} | "
                  f"Downloads: {ext.download_count} | Rating: {ext.rating:.1f}")
            print()
    
    elif args.command == 'install':
        success, message = mp.install_extension(args.extension_name, 
                                               args.version, args.force)
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.command == 'uninstall':
        success, message = mp.uninstall_extension(args.extension_name, 
                                                 args.remove_data)
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.command == 'update':
        success, message = mp.update_extension(args.extension_name)
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.command == 'list':
        installed = mp.get_installed_extensions()
        print(f"Installed extensions ({len(installed)}):")
        for ext in installed:
            status = "enabled" if ext['enabled'] else "disabled"
            update = " (update available)" if ext['status'] == 'update_available' else ""
            print(f"  {ext['name']} v{ext['version']} [{status}]{update}")
    
    elif args.command == 'stats':
        stats = mp.get_marketplace_stats()
        print("Marketplace Statistics:")
        print(f"  Total extensions: {stats['total_extensions']}")
        print(f"  Installed extensions: {stats['installed_extensions']}")
        print(f"  Average compatibility: {stats['average_compatibility']:.2f}")
        print(f"  Cache size: {stats['cache_size'] / 1024 / 1024:.1f} MB")
        print(f"  Backups available: {stats['backup_count']}")
        
        if stats['popular_tags']:
            print("\nPopular tags:")
            for tag, count in stats['popular_tags']:
                print(f"  {tag}: {count}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    marketplace_cli()