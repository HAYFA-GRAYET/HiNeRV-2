#!/usr/bin/env python3
"""
HiNeRV GUI - Configuration Manager
Handles loading, saving, and managing configuration settings for HiNeRV
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration settings for HiNeRV compression
    Handles model presets, training configurations, and user preferences
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_dir: Optional path to configuration directory
        """
        # Set default paths
        self.app_dir = self._get_app_directory()
        self.config_dir = config_dir or os.path.join(self.app_dir, "configs")
        self.models_dir = os.path.join(self.config_dir, "models")
        self.training_dir = os.path.join(self.config_dir, "training")
        self.output_dir = os.path.join(self.app_dir, "output")
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Cache for loaded configurations
        self.model_presets_cache = {}
        self.training_configs_cache = {}
        self.recent_configs = []
        
        # Load from HiNeRV original config files if available
        self._load_hinerv_configs()
        
        logger.debug(f"ConfigManager initialized with config_dir: {self.config_dir}")
    
    def _get_app_directory(self) -> str:
        """
        Get the application directory
        
        Returns:
            Path to application directory
        """
        # Try to detect HiNeRV installation directory
        possible_dirs = [
            os.path.join(os.path.expanduser("~"), "HiNeRV"),
            os.path.join(os.path.expanduser("~"), "Documents", "HiNeRV"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        ] 
        
        # Check if any directory contains HiNeRV files
        for directory in possible_dirs:
            if os.path.exists(os.path.join(directory, "cfgs")):
                return directory
        
        # Fall back to user's home directory
        return os.path.join(os.path.expanduser("~"), "HiNeRV_GUI")
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_hinerv_configs(self):
        """
        Load configuration files from HiNeRV if available
        """
        # Try to locate HiNeRV config files
        hinerv_cfgs_dir = os.path.join(self.app_dir, "cfgs")
        
        if os.path.exists(hinerv_cfgs_dir):
            # Copy model configs
            hinerv_models_dir = os.path.join(hinerv_cfgs_dir, "models")
            if os.path.exists(hinerv_models_dir):
                self._copy_configs(hinerv_models_dir, self.models_dir)
            
            # Copy training configs
            hinerv_train_dir = os.path.join(hinerv_cfgs_dir, "train")
            if os.path.exists(hinerv_train_dir):
                self._copy_configs(hinerv_train_dir, self.training_dir)
            
            logger.info(f"Loaded original HiNeRV configs from {hinerv_cfgs_dir}")
        else:
            # Create default configurations if HiNeRV configs not found
            self._create_default_configs()
            logger.info("Created default configurations")
    
    def _copy_configs(self, source_dir: str, target_dir: str):
        """
        Copy configuration files from source to target directory
        
        Args:
            source_dir: Source directory
            target_dir: Target directory
        """
        os.makedirs(target_dir, exist_ok=True)
        
        # Only copy if target is empty
        if not os.listdir(target_dir):
            import shutil
            for item in os.listdir(source_dir):
                s = os.path.join(source_dir, item)
                t = os.path.join(target_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, t)
    
    def _create_default_configs(self):
        """Create default configuration files"""
        # Create default model presets
        default_models = {
            "small.txt": {
                "description": "Small model for faster training (lower quality)",
                "hidden_features": 64,
                "hidden_layers": 3,
                "skip_conn": True
            },
            "medium.txt": {
                "description": "Balanced model for most videos",
                "hidden_features": 128,
                "hidden_layers": 5,
                "skip_conn": True
            },
            "large.txt": {
                "description": "Large model for best quality (slower training)",
                "hidden_features": 256,
                "hidden_layers": 8,
                "skip_conn": True
            }
        }
        
        for filename, config in default_models.items():
            path = os.path.join(self.models_dir, filename)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    f.write(f"# {config['description']}\n")
                    f.write(f"hidden_features={config['hidden_features']}\n")
                    f.write(f"hidden_layers={config['hidden_layers']}\n")
                    f.write(f"skip_conn={'true' if config['skip_conn'] else 'false'}\n")
        
        # Create default training configurations
        default_trainings = {
            "default.txt": {
                "description": "Default training configuration",
                "epochs": 30,
                "lr": 1e-3,
                "batch_size": 4,
                "l1_coef": 0.0,
                "ssim_coef": 0.2,
                "depth_regularization": 0.0
            },
            "quality.txt": {
                "description": "High quality training (slower)",
                "epochs": 50,
                "lr": 1e-3,
                "batch_size": 2,
                "l1_coef": 0.0,
                "ssim_coef": 0.5,
                "depth_regularization": 0.0
            },
            "speed.txt": {
                "description": "Fast training (lower quality)",
                "epochs": 15,
                "lr": 2e-3,
                "batch_size": 8,
                "l1_coef": 0.0,
                "ssim_coef": 0.1,
                "depth_regularization": 0.0
            }
        }
        
        for filename, config in default_trainings.items():
            path = os.path.join(self.training_dir, filename)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    f.write(f"# {config['description']}\n")
                    f.write(f"epochs={config['epochs']}\n")
                    f.write(f"lr={config['lr']}\n")
                    f.write(f"batch_size={config['batch_size']}\n")
                    f.write(f"l1_coef={config['l1_coef']}\n")
                    f.write(f"ssim_coef={config['ssim_coef']}\n")
                    f.write(f"depth_regularization={config['depth_regularization']}\n")
    
    def get_model_presets(self) -> List[Dict[str, Any]]:
        """
        Get all available model presets
        
        Returns:
            List of model preset configurations
        """
        presets = []
        
        # Load from models directory
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.txt'):
                preset = self.get_model_preset(filename)
                if preset:
                    presets.append(preset)
        
        return presets
    
    def get_model_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific model preset
        
        Args:
            preset_name: Name of the preset (filename)
        
        Returns:
            Model preset configuration or None if not found
        """
        if preset_name in self.model_presets_cache:
            return self.model_presets_cache[preset_name]
        
        path = os.path.join(self.models_dir, preset_name)
        if not os.path.exists(path):
            logger.warning(f"Model preset not found: {preset_name}")
            return None
        
        try:
            preset = self._parse_config_file(path)
            preset['name'] = os.path.basename(preset_name)
            
            # Extract description from comments if available
            description = ""
            with open(path, 'r') as f:
                for line in f:
                    if line.strip().startswith('#'):
                        description += line.strip()[1:].strip() + " "
                    else:
                        break
            
            preset['description'] = description.strip()
            
            # Cache the result
            self.model_presets_cache[preset_name] = preset
            return preset
        
        except Exception as e:
            logger.error(f"Error loading model preset {preset_name}: {e}")
            return None
    
    def get_training_configs(self) -> List[Dict[str, Any]]:
        """
        Get all available training configurations
        
        Returns:
            List of training configurations
        """
        configs = []
        
        # Load from training directory
        for filename in os.listdir(self.training_dir):
            if filename.endswith('.txt'):
                config = self.get_training_config(filename)
                if config:
                    configs.append(config)
        
        return configs
    
    def get_training_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific training configuration
        
        Args:
            config_name: Name of the configuration (filename)
        
        Returns:
            Training configuration or None if not found
        """
        if config_name in self.training_configs_cache:
            return self.training_configs_cache[config_name]
        
        path = os.path.join(self.training_dir, config_name)
        if not os.path.exists(path):
            logger.warning(f"Training config not found: {config_name}")
            return None
        
        try:
            config = self._parse_config_file(path)
            config['name'] = os.path.basename(config_name)
            
            # Extract description from comments if available
            description = ""
            with open(path, 'r') as f:
                for line in f:
                    if line.strip().startswith('#'):
                        description += line.strip()[1:].strip() + " "
                    else:
                        break
            
            config['description'] = description.strip()
            
            # Cache the result
            self.training_configs_cache[config_name] = config
            return config
        
        except Exception as e:
            logger.error(f"Error loading training config {config_name}: {e}")
            return None
    
    def _parse_config_file(self, path: str) -> Dict[str, Any]:
        """
        Parse a configuration file
        
        Args:
            path: Path to configuration file
        
        Returns:
            Parsed configuration
        """
        config = {}
        
        with open(path, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert values to appropriate types
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif self._is_float(value):
                        value = float(value)
                    
                    config[key] = value
        
        return config
    
    def _is_float(self, value: str) -> bool:
        """
        Check if a string can be converted to float
        
        Args:
            value: String to check
        
        Returns:
            True if the string can be converted to float
        """
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def save_config(self, config: Dict[str, Any], output_dir: str) -> str:
        """
        Save a complete configuration to an output directory
        
        Args:
            config: Configuration to save
            output_dir: Output directory
        
        Returns:
            Path to saved configuration file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as YAML
        config_path = os.path.join(output_dir, "args.yaml")
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {config_path}")
            return config_path
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return ""
    
    def load_saved_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a saved configuration
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Loaded configuration or None if failed
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    config = self._parse_config_file(config_path)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading configuration {config_path}: {e}")
            return None
    
    def get_recent_configs(self) -> List[Dict[str, Any]]:
        """
        Get recent compression configurations
        
        Returns:
            List of recent configurations
        """
        # Scan output directory for args.yaml files
        output_dirs = []
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                args_path = os.path.join(item_path, "args.yaml")
                if os.path.exists(args_path):
                    # Get directory creation time for sorting
                    output_dirs.append({
                        'path': item_path,
                        'args_path': args_path,
                        'time': os.path.getctime(item_path)
                    })
        
        # Sort by creation time (newest first)
        output_dirs.sort(key=lambda x: x['time'], reverse=True)
        
        # Load configurations
        recent_configs = []
        for dir_info in output_dirs[:10]:  # Limit to 10 recent configs
            config = self.load_saved_config(dir_info['args_path'])
            if config:
                config['output_dir'] = dir_info['path']
                recent_configs.append(config)
        
        return recent_configs
    
    def generate_hinerv_command(self, config: Dict[str, Any]) -> str:
        """
        Generate HiNeRV command line from configuration
        
        Args:
            config: Configuration dictionary
        
        Returns:
            HiNeRV command line string
        """
        cmd = ["accelerate", "launch", "--mixed_precision", "fp16", "--dynamo_backend", "inductor"]
        cmd.append("train_video.py")
        
        # Add model parameters
        if 'model_preset' in config:
            model = config['model_preset']
            if 'hidden_features' in model:
                cmd.append(f"--hidden_features={model['hidden_features']}")
            if 'hidden_layers' in model:
                cmd.append(f"--hidden_layers={model['hidden_layers']}")
            if 'skip_conn' in model:
                cmd.append(f"--skip_conn={str(model['skip_conn']).lower()}")
        
        # Add training options
        if 'training_options' in config:
            opts = config['training_options']
            if 'epochs' in opts:
                cmd.append(f"--epochs={opts['epochs']}")
            if 'lr' in opts:
                cmd.append(f"--lr={opts['lr']}")
            if 'batch_size' in opts:
                cmd.append(f"--batch_size={opts['batch_size']}")
            if 'l1_coef' in opts:
                cmd.append(f"--l1_coef={opts['l1_coef']}")
            if 'ssim_coef' in opts:
                cmd.append(f"--ssim_coef={opts['ssim_coef']}")
            if 'depth_regularization' in opts:
                cmd.append(f"--depth_regularization={opts['depth_regularization']}")
        
        # Add video path
        if 'video_path' in config:
            cmd.append(f"--video_path={config['video_path']}")
        
        # Add output directory
        if 'output_dir' in config:
            cmd.append(f"--output_dir={config['output_dir']}")
        
        # Additional parameters
        if 'max_frames' in config:
            cmd.append(f"--max_frames={config['max_frames']}")
        
        return " ".join(cmd)