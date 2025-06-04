#!/usr/bin/env python3
"""
ModelPresetsWidget - Widget for selecting model presets
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QTextEdit, QFormLayout, QSpinBox, QDoubleSpinBox,
    QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtCore import Signal, Qt


class ModelPresetsWidget(QWidget):
    """Widget for selecting and configuring model presets"""
    
    preset_changed = Signal(dict)
    
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.current_preset = None
        
        self.setup_ui()
        self.load_presets()
    
    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout(self)
        
        # Preset selection group
        preset_group = QGroupBox("Model Preset")
        preset_layout = QFormLayout(preset_group)
        
        # Preset dropdown
        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addRow("Preset:", self.preset_combo)
        
        # Preset description
        self.description_text = QTextEdit()
        self.description_text.setMaximumHeight(100)
        self.description_text.setReadOnly(True)
        self.description_text.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        preset_layout.addRow("Description:", self.description_text)
        
        layout.addWidget(preset_group)
        
        # Model parameters group
        params_group = QGroupBox("Model Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.params_layout = QFormLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        params_layout.addWidget(scroll_area)
        
        layout.addWidget(params_group)
        
        # Stretch to fill remaining space
        layout.addStretch()
    
    def load_presets(self):
        """Load available model presets"""
        try:
            # Find all preset files
            models_dir = Path("../cfgs/models")
            print(f"Looking for model presets in: {models_dir}")
            if not models_dir.exists():
                self.logger.warning(f"Models directory not found: {models_dir}")
                return
            
            preset_files = list(models_dir.glob("*.txt"))
            if not preset_files:
                self.logger.warning("No preset files found in models directory")
                return
            
            # Clear existing items
            self.preset_combo.clear()
            
            # Parse and add presets
            for file_path in sorted(preset_files):
                preset_name = self.get_preset_name(file_path)
                self.preset_combo.addItem(preset_name, str(file_path))
            
            # Set default preset
            default_preset = "uvg-hinerv-s_1920x1080.txt"
            default_index = -1
            
            for i in range(self.preset_combo.count()):
                file_path = self.preset_combo.itemData(i)
                if default_preset in file_path:
                    default_index = i
                    break
            
            if default_index >= 0:
                self.preset_combo.setCurrentIndex(default_index)
            else:
                # Fallback to first preset
                self.preset_combo.setCurrentIndex(0)
            
            self.logger.info(f"Loaded {len(preset_files)} model presets")
            
        except Exception as e:
            self.logger.error(f"Error loading presets: {e}")
    
    def get_preset_name(self, file_path: Path) -> str:
        """Extract a user-friendly name from preset file"""
        # Remove file extension and format name
        name = file_path.stem
        
        # Replace hyphens with spaces and title case
        name = name.replace('-', ' ').replace('_', ' ')
        return ' '.join(word.capitalize() for word in name.split())
    
    def on_preset_changed(self, preset_name: str):
        """Handle preset selection change"""
        if not preset_name:
            return
        
        # Get the file path for the selected preset
        current_index = self.preset_combo.currentIndex()
        if current_index < 0:
            return
        
        file_path = self.preset_combo.itemData(current_index)
        if not file_path:
            return
        
        try:
            # Parse the preset file
            preset_config = self.parse_preset_file(file_path)
            
            # Update UI with preset parameters
            self.update_parameters_ui(preset_config)
            
            # Store current preset
            self.current_preset = {
                'name': preset_name,
                'file_path': file_path,
                'config': preset_config
            }
            
            # Emit signal
            self.preset_changed.emit(self.current_preset)
            
        except Exception as e:
            self.logger.error(f"Error loading preset {preset_name}: {e}")
    
    def parse_preset_file(self, file_path: str) -> Dict:
        """Parse a preset configuration file"""
        config = {}
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse configuration lines
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Split the line into key-value pairs
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].lstrip('-')  # Remove leading dashes
                    value = ' '.join(parts[1:])
                    config[key] = value
            
            # Add description if available
            config['description'] = self.get_preset_description(file_path)
            
        except Exception as e:
            self.logger.error(f"Error parsing preset file {file_path}: {e}")
            raise
        
        return config
    
    def get_preset_description(self, file_path: str) -> str:
        """Get description for a preset based on its name"""
        preset_name = Path(file_path).stem.lower()
        
        # Built-in descriptions for common presets
        descriptions = {
            'bunny-hinerv-s': "Small model optimized for Big Buck Bunny dataset (1280x720)",
            'bunny-hinerv-xs': "Extra small model for quick testing on Big Buck Bunny",
            'bunny-hinerv-xxs': "Minimal model for very fast testing",
            'uvg-hinerv-s': "Small model optimized for UVG dataset (1920x1080)",
            'uvg-hinerv-m': "Medium model with balanced speed and quality for UVG",
            'uvg-hinerv-l': "Large model for high quality compression on UVG",
            'mcl-hinerv-s': "Small model optimized for MCL-JCV dataset"
        }
        
        # Check for exact match first
        for key, desc in descriptions.items():
            if key in preset_name:
                return desc
        
        # Default description
        return f"Model preset for {preset_name.replace('-', ' ').title()}"
    
    def update_parameters_ui(self, config: Dict):
        """Update the parameters UI with preset configuration"""
        # Clear existing parameters
        for i in reversed(range(self.params_layout.count())):
            item = self.params_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        # Update description
        self.description_text.setPlainText(config.get('description', 'No description available'))
        
        # Add key parameters as editable fields
        key_params = [
            ('channels', 'Channels', int),
            ('depths', 'Depths', str),
            ('stem-kernels', 'Stem Kernels', int),
            ('scales-hw', 'HW Scales', str),
            ('batch-size', 'Batch Size', int),
            ('lr', 'Learning Rate', float),
            ('epochs', 'Epochs', int)
        ]
        
        for param_key, param_label, param_type in key_params:
            if param_key in config:
                value = config[param_key]
                
                if param_type == int:
                    widget = QSpinBox()
                    widget.setMinimum(1)
                    widget.setMaximum(10000)
                    widget.setValue(int(value.split()[0]) if isinstance(value, str) else int(value))
                elif param_type == float:
                    widget = QDoubleSpinBox()
                    widget.setMinimum(0.0001)
                    widget.setMaximum(1.0)
                    widget.setDecimals(6)
                    widget.setValue(float(value))
                else:  # string
                    widget = QLabel(str(value))
                    widget.setWordWrap(True)
                    widget.setStyleSheet("color: #666; font-family: monospace;")
                
                self.params_layout.addRow(f"{param_label}:", widget)
    
    def get_selected_preset(self) -> Optional[Dict]:
        """Get the currently selected preset"""
        return self.current_preset