"""
Training Options Widget for HiNeRV GUI
Provides form controls for all training parameters with validation
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QCheckBox,
    QSlider, QLabel, QPushButton, QScrollArea, QFrame, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any
import os
import json
import logging


class TrainingOptionsWidget(QWidget):
    """Widget for configuring training parameters"""
    
    options_changed = Signal(dict)
    
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.setup_ui()
        self.load_defaults()
    
    def setup_ui(self):
        """Setup the training options UI"""
        layout = QVBoxLayout(self)
        
        # Create scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        main_layout = QVBoxLayout(scroll_widget)
        
        # Create tabs for different option categories
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Basic options tab
        basic_tab = self.create_basic_options()
        tabs.addTab(basic_tab, "Basic")
        
        # Advanced options tab
        advanced_tab = self.create_advanced_options()
        tabs.addTab(advanced_tab, "Advanced")
        
        # Model architecture tab
        model_tab = self.create_model_options()
        tabs.addTab(model_tab, "Model")
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        self.save_preset_btn = QPushButton("Save as Preset")
        self.save_preset_btn.clicked.connect(self.save_preset)
        preset_layout.addWidget(self.save_preset_btn)
        
        self.load_preset_btn = QPushButton("Load Preset")
        self.load_preset_btn.clicked.connect(self.load_preset)
        preset_layout.addWidget(self.load_preset_btn)
        
        preset_layout.addStretch()
        main_layout.addLayout(preset_layout)
    
    def create_basic_options(self) -> QWidget:
        """Create basic training options tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training parameters group
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout(training_group)
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(30)
        training_layout.addRow("Epochs:", self.epochs_spin)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(2)
        training_layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Evaluation batch size
        self.eval_batch_size_spin = QSpinBox()
        self.eval_batch_size_spin.setRange(1, 32)
        self.eval_batch_size_spin.setValue(1)
        training_layout.addRow("Eval Batch Size:", self.eval_batch_size_spin)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.002)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        training_layout.addRow("Learning Rate:", self.lr_spin)
        
        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "adamw", "sgd"])
        training_layout.addRow("Optimizer:", self.optimizer_combo)
        
        layout.addWidget(training_group)
        
        # Loss configuration group
        loss_group = QGroupBox("Loss Configuration")
        loss_layout = QFormLayout(loss_group)
        
        # Loss type
        self.loss_type_edit = QLineEdit("0.7 l1 0.3 ms-ssim_5x5")
        loss_layout.addRow("Loss Function:", self.loss_type_edit)
        
        # Training metric
        self.train_metric_combo = QComboBox()
        self.train_metric_combo.addItems(["psnr", "ssim", "ms-ssim"])
        loss_layout.addRow("Training Metric:", self.train_metric_combo)
        
        # Eval metric
        self.eval_metric_edit = QLineEdit("psnr ms-ssim")
        loss_layout.addRow("Eval Metrics:", self.eval_metric_edit)
        
        layout.addWidget(loss_group)
        
        # Video processing group
        video_group = QGroupBox("Video Processing")
        video_layout = QFormLayout(video_group)
        
        # Max frames
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, 100000)
        self.max_frames_spin.setValue(100)
        video_layout.addRow("Max Frames:", self.max_frames_spin)
        
        # Patch size
        self.patch_size_edit = QLineEdit("120 120")
        video_layout.addRow("Patch Size:", self.patch_size_edit)
        
        # Input size
        self.input_size_edit = QLineEdit("1080 1920")
        video_layout.addRow("Input Size:", self.input_size_edit)
        
        layout.addWidget(video_group)
        
        layout.addStretch()
        return widget
    
    def create_advanced_options(self) -> QWidget:
        """Create advanced training options tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Pruning group
        pruning_group = QGroupBox("Pruning Options")
        pruning_layout = QFormLayout(pruning_group)
        
        # Enable pruning
        self.enable_pruning_check = QCheckBox()
        self.enable_pruning_check.setChecked(True)
        pruning_layout.addRow("Enable Pruning:", self.enable_pruning_check)
        
        # Prune epochs
        self.prune_epochs_spin = QSpinBox()
        self.prune_epochs_spin.setRange(1, 2000)
        self.prune_epochs_spin.setValue(30)
        pruning_layout.addRow("Prune Epochs:", self.prune_epochs_spin)
        
        # Prune ratio
        self.prune_ratio_spin = QDoubleSpinBox()
        self.prune_ratio_spin.setRange(0.0, 0.5)
        self.prune_ratio_spin.setValue(0.15)
        self.prune_ratio_spin.setDecimals(3)
        pruning_layout.addRow("Prune Ratio:", self.prune_ratio_spin)
        
        layout.addWidget(pruning_group)
        
        # Quantization group
        quant_group = QGroupBox("Quantization Options")
        quant_layout = QFormLayout(quant_group)
        
        # Enable quantization
        self.enable_quant_check = QCheckBox()
        self.enable_quant_check.setChecked(True)
        quant_layout.addRow("Enable Quantization:", self.enable_quant_check)
        
        # Quant epochs
        self.quant_epochs_spin = QSpinBox()
        self.quant_epochs_spin.setRange(1, 2000)
        self.quant_epochs_spin.setValue(30)
        quant_layout.addRow("Quant Epochs:", self.quant_epochs_spin)
        
        # Quant levels
        self.quant_levels_edit = QLineEdit("8 7 6")
        quant_layout.addRow("Quant Levels:", self.quant_levels_edit)
        
        # Quant noise
        self.quant_noise_spin = QDoubleSpinBox()
        self.quant_noise_spin.setRange(0.0, 1.0)
        self.quant_noise_spin.setValue(0.9)
        self.quant_noise_spin.setDecimals(2)
        quant_layout.addRow("Quant Noise:", self.quant_noise_spin)
        
        layout.addWidget(quant_group)
        
        # Evaluation group
        eval_group = QGroupBox("Evaluation Options")
        eval_layout = QFormLayout(eval_group)
        
        # Eval epochs
        self.eval_epochs_spin = QSpinBox()
        self.eval_epochs_spin.setRange(1, 100)
        self.eval_epochs_spin.setValue(10)
        eval_layout.addRow("Eval Every N Epochs:", self.eval_epochs_spin)
        
        # Log eval
        self.log_eval_check = QCheckBox()
        self.log_eval_check.setChecked(True)
        eval_layout.addRow("Log Evaluations:", self.log_eval_check)
        
        layout.addWidget(eval_group)
        
        layout.addStretch()
        return widget
    
    def create_model_options(self) -> QWidget:
        """Create model architecture options tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model architecture group
        arch_group = QGroupBox("Model Architecture")
        arch_layout = QFormLayout(arch_group)
        
        # Block type
        self.block_type_combo = QComboBox()
        self.block_type_combo.addItems(["convnext", "resnet", "transformer"])
        arch_layout.addRow("Block Type:", self.block_type_combo)
        
        # Channels
        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(32, 1024)
        self.channels_spin.setValue(280)
        arch_layout.addRow("Channels:", self.channels_spin)
        
        # Depths
        self.depths_edit = QLineEdit("3 3 3 1")
        arch_layout.addRow("Layer Depths:", self.depths_edit)
        
        # Kernel sizes
        self.kernels_edit = QLineEdit("3 3 3 3")
        arch_layout.addRow("Kernel Sizes:", self.kernels_edit)
        
        layout.addWidget(arch_group)
        
        # Upsampling group
        upsample_group = QGroupBox("Upsampling Configuration")
        upsample_layout = QFormLayout(upsample_group)
        
        # Upsample type
        self.upsample_type_combo = QComboBox()
        self.upsample_type_combo.addItems(["trilinear", "nearest", "pixel_shuffle"])
        self.upsample_type_combo.setCurrentText("trilinear")
        upsample_layout.addRow("Upsample Type:", self.upsample_type_combo)
        
        # Scales
        self.scales_hw_edit = QLineEdit("5 3 2 2")
        upsample_layout.addRow("HW Scales:", self.scales_hw_edit)
        
        self.scales_t_edit = QLineEdit("1 1 1 1")
        upsample_layout.addRow("Time Scales:", self.scales_t_edit)
        
        layout.addWidget(upsample_group)
        
        layout.addStretch()
        return widget
    
    def load_defaults(self):
        """Load default training configuration"""
        default_file = "cfgs/train/hinerv_1920x1080.txt"
        if os.path.exists(default_file):
            self.load_from_file(default_file)
        
        # Connect all widgets to change signal
        self.connect_change_signals()
    
    def connect_change_signals(self):
        """Connect all widgets to emit options_changed signal"""
        # Spin boxes
        self.epochs_spin.valueChanged.connect(self.emit_options_changed)
        self.batch_size_spin.valueChanged.connect(self.emit_options_changed)
        self.eval_batch_size_spin.valueChanged.connect(self.emit_options_changed)
        self.lr_spin.valueChanged.connect(self.emit_options_changed)
        
        # Combo boxes
        self.optimizer_combo.currentTextChanged.connect(self.emit_options_changed)
        self.train_metric_combo.currentTextChanged.connect(self.emit_options_changed)
        self.block_type_combo.currentTextChanged.connect(self.emit_options_changed)
        self.upsample_type_combo.currentTextChanged.connect(self.emit_options_changed)
        
        # Line edits
        self.loss_type_edit.textChanged.connect(self.emit_options_changed)
        self.eval_metric_edit.textChanged.connect(self.emit_options_changed)
        self.patch_size_edit.textChanged.connect(self.emit_options_changed)
        self.input_size_edit.textChanged.connect(self.emit_options_changed)
        
        # Check boxes
        self.enable_pruning_check.toggled.connect(self.emit_options_changed)
        self.enable_quant_check.toggled.connect(self.emit_options_changed)
        self.log_eval_check.toggled.connect(self.emit_options_changed)
    
    def emit_options_changed(self):
        """Emit signal when options change"""
        self.options_changed.emit(self.get_options())
    
    def get_options(self) -> Dict[str, Any]:
        """Get current training options as dictionary"""
        options = {
            # Basic options
            'epochs': self.epochs_spin.value(),
            'batch-size': self.batch_size_spin.value(),
            'eval-batch-size': self.eval_batch_size_spin.value(),
            'lr': self.lr_spin.value(),
            'opt': self.optimizer_combo.currentText(),
            'loss': self.loss_type_edit.text(),
            'train-metric': self.train_metric_combo.currentText(),
            'eval-metric': self.eval_metric_edit.text(),
            'max-frames': self.max_frames_spin.value(),
            'patch-size': self.patch_size_edit.text(),
            'input-size': self.input_size_edit.text(),
            
            # Advanced options
            'enable-pruning': self.enable_pruning_check.isChecked(),
            'prune-epochs': self.prune_epochs_spin.value(),
            'prune-ratio': self.prune_ratio_spin.value(),
            'enable-quant': self.enable_quant_check.isChecked(),
            'quant-epochs': self.quant_epochs_spin.value(),
            'quant-level': self.quant_levels_edit.text(),
            'quant-noise': self.quant_noise_spin.value(),
            'eval-epochs': self.eval_epochs_spin.value(),
            'log-eval': 'true' if self.log_eval_check.isChecked() else 'false',
            
            # Model options
            'block-type': self.block_type_combo.currentText(),
            'channels': self.channels_spin.value(),
            'depths': self.depths_edit.text(),
            'kernels': self.kernels_edit.text(),
            'upsample-type': self.upsample_type_combo.currentText(),
            'scales-hw': self.scales_hw_edit.text(),
            'scales-t': self.scales_t_edit.text(),
        }
        
        return options
    
    def set_options(self, options: Dict[str, Any]):
        """Set training options from dictionary"""
        # Block signals temporarily
        self.blockSignals(True)
        
        try:
            # Basic options
            if 'epochs' in options:
                self.epochs_spin.setValue(options['epochs'])
            if 'batch-size' in options:
                self.batch_size_spin.setValue(options['batch-size'])
            if 'eval-batch-size' in options:
                self.eval_batch_size_spin.setValue(options['eval-batch-size'])
            if 'lr' in options:
                self.lr_spin.setValue(options['lr'])
            if 'opt' in options:
                self.optimizer_combo.setCurrentText(options['opt'])
            if 'loss' in options:
                self.loss_type_edit.setText(options['loss'])
            if 'train-metric' in options:
                self.train_metric_combo.setCurrentText(options['train-metric'])
            if 'eval-metric' in options:
                self.eval_metric_edit.setText(options['eval-metric'])
            if 'max-frames' in options:
                self.max_frames_spin.setValue(options['max-frames'])
            if 'patch-size' in options:
                self.patch_size_edit.setText(options['patch-size'])
            if 'input-size' in options:
                self.input_size_edit.setText(options['input-size'])
            
            # Advanced options
            if 'enable-pruning' in options:
                self.enable_pruning_check.setChecked(options['enable-pruning'])
            if 'prune-epochs' in options:
                self.prune_epochs_spin.setValue(options['prune-epochs'])
            if 'prune-ratio' in options:
                self.prune_ratio_spin.setValue(options['prune-ratio'])
            if 'enable-quant' in options:
                self.enable_quant_check.setChecked(options['enable-quant'])
            if 'quant-epochs' in options:
                self.quant_epochs_spin.setValue(options['quant-epochs'])
            if 'quant-level' in options:
                self.quant_levels_edit.setText(options['quant-level'])
            if 'quant-noise' in options:
                self.quant_noise_spin.setValue(options['quant-noise'])
            if 'eval-epochs' in options:
                self.eval_epochs_spin.setValue(options['eval-epochs'])
            if 'log-eval' in options:
                self.log_eval_check.setChecked(options['log-eval'] == 'true')
            
            # Model options
            if 'block-type' in options:
                self.block_type_combo.setCurrentText(options['block-type'])
            if 'channels' in options:
                self.channels_spin.setValue(options['channels'])
            if 'depths' in options:
                self.depths_edit.setText(options['depths'])
            if 'kernels' in options:
                self.kernels_edit.setText(options['kernels'])
            if 'upsample-type' in options:
                self.upsample_type_combo.setCurrentText(options['upsample-type'])
            if 'scales-hw' in options:
                self.scales_hw_edit.setText(options['scales-hw'])
            if 'scales-t' in options:
                self.scales_t_edit.setText(options['scales-t'])
        
        finally:
            self.blockSignals(False)
            self.emit_options_changed()
    
    def load_from_file(self, filepath: str):
        """Load options from a configuration file"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Parse configuration file
            options = {}
            for line in lines:
                line = line.strip()
                if line.startswith('--'):
                    parts = line.split()
                    key = parts[0][2:]  # Remove --
                    if len(parts) > 1:
                        value = ' '.join(parts[1:])
                        # Try to convert to appropriate type
                        if value.isdigit():
                            value = int(value)
                        elif value.replace('.', '').isdigit():
                            value = float(value)
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        options[key] = value
            
            self.set_options(options)
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    def save_preset(self):
        """Save current options as a preset"""
        # TODO: Implement preset saving dialog
        pass
    
    def load_preset(self, preset_info: Dict):
        """Load settings from a model preset"""
        try:
            # Get training options from preset
            options = preset_info.get('training_options', {})
            
            # Update widget values
            for key, value in options.items():
                if hasattr(self, f"{key}_spin"):
                    spinbox = getattr(self, f"{key}_spin")
                    spinbox.setValue(value)
                elif hasattr(self, f"{key}_check"):
                    checkbox = getattr(self, f"{key}_check")
                    checkbox.setChecked(value)
        except Exception as e:
            logging.error(f"Error loading preset: {e}")
    
    def update_for_video(self, video_info: Dict):
        """Update options based on video information"""
        # Set input size based on video resolution
        width = video_info.get('width', 1920)
        height = video_info.get('height', 1080)
        self.input_size_edit.setText(f"{height} {width}")
        

    
    def validate_options(self) -> tuple[bool, str]:
        """Validate current options and return (is_valid, error_message)"""
        # Check patch size
        try:
            patch_parts = self.patch_size_edit.text().split()
            if len(patch_parts) != 2 or not all(p.isdigit() for p in patch_parts):
                return False, "Patch size must be two integers (e.g., '120 120')"
        except:
            return False, "Invalid patch size format"
        
        # Check input size
        try:
            input_parts = self.input_size_edit.text().split()
            if len(input_parts) != 2 or not all(p.isdigit() for p in input_parts):
                return False, "Input size must be two integers (e.g., '1080 1920')"
        except:
            return False, "Invalid input size format"
        
        # Check quantization levels
        try:
            quant_parts = self.quant_levels_edit.text().split()
            if not all(p.isdigit() for p in quant_parts):
                return False, "Quantization levels must be integers"
        except:
            return False, "Invalid quantization levels format"
        
        # Check if batch size is reasonable
        if self.batch_size_spin.value() > 16:
            return False, "Batch size > 16 may cause GPU memory issues"
        
        return True, ""