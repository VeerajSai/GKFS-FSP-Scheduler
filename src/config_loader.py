"""
Configuration Loader
====================
Load and validate configuration from YAML file.
Single source of truth for all project settings.

Maintainer: Project Team
Date: January 2026
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class Config:
    """Configuration manager for the GKFS Auto Switch project."""

    _instance = None
    _config = None

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._config is None:
            self.load(config_path)

    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try to find config file relative to this file or project root
            possible_paths = [
                # Repo root /configs/config.yaml (most common)
                Path(__file__).resolve().parent.parent / 'configs' / 'config.yaml',
                # Workspace root when script launched from repo root or nested folders
                Path.cwd() / 'configs' / 'config.yaml',
                Path.cwd().parent / 'configs' / 'config.yaml',
                # Fallback: one level above repo root (e.g., when src is symlinked)
                Path(__file__).resolve().parent.parent.parent / 'configs' / 'config.yaml',
            ]

            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                raise FileNotFoundError(
                    "config.yaml not found. Searched in: " +
                    ", ".join(str(p) for p in possible_paths)
                )

        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Store the config directory for relative path resolution
        self._config['_config_dir'] = str(config_path.parent.parent)

        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_path(self, key: str) -> Path:
        """Get a path configuration value, resolved relative to project root."""
        path_str = self.get(key)
        if path_str is None:
            raise KeyError(f"Path configuration '{key}' not found")

        path = Path(path_str)
        if not path.is_absolute():
            path = Path(self._config['_config_dir']) / path

        return path

    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self._config.get('data', {})

    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self._config.get('training', {})

    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration section."""
        return self._config.get('models', {})

    @property
    def outputs(self) -> Dict[str, Any]:
        """Get outputs configuration section."""
        return self._config.get('outputs', {})

    @property
    def mlflow(self) -> Dict[str, Any]:
        """Get MLflow configuration section."""
        return self._config.get('mlflow', {})

    @property
    def fsp_providers(self) -> list:
        """Get list of FSP providers."""
        return self._config.get('fsp_providers', [])

    @property
    def visualization(self) -> Dict[str, Any]:
        """Get visualization configuration section."""
        return self._config.get('visualization', {})

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking if key exists in config."""
        if not isinstance(key, str):
            return False
        return key in self._config


def load_config(config_path: Optional[str] = None) -> Config:
    """Load and return configuration singleton."""
    return Config(config_path)


# Convenience function for notebook usage
def get_config() -> Config:
    """Get the configuration singleton."""
    return Config()
