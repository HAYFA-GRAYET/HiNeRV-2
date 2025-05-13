#!/usr/bin/env python3
"""
HiNeRV GUI - Modern Video Compression Interface
Main application entry point
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QGroupBox, QFormLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
    QCheckBox, QLineEdit, QTextEdit, QProgressBar, QListWidget,
    QScrollArea, QFrame, QGridLayout, QSizePolicy, QMessageBox,
    QSplashScreen, QSystemTrayIcon, QMenu
)
from PySide6.QtCore import (
    Qt, QThread, QTimer, Signal, QSettings, QStandardPaths,
    QSize, Slot, QEvent, QUrl
)
from PySide6.QtGui import (
    QFont, QPixmap, QIcon, QDesktopServices, QKeySequence,
    QShortcut, QAction, QDragEnterEvent, QDropEvent
)
from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
)

# Import our custom modules
from src.components import (
    VideoPreviewWidget, ModelPresetsWidget, TrainingOptionsWidget,
    ResourceGuardWidget, ProgressWidget, ResultsWidget, HistoryWidget,
    LogViewerWidget
)
from src.core import (
    HiNeRVProcessor, ConfigManager, VideoProcessor, SystemMonitor
)
from src.utils import (
    setup_logging, get_system_info, check_dependencies,
    format_duration, format_filesize, create_dark_theme
)

# Constants
APP_NAME = "HiNeRV Compressor"
APP_VERSION = "1.0.0"
CONFIG_FILE = "config.json"
LOG_FILE = "hinerv_gui.log"


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings()
        self.config_manager = ConfigManager()
        self.processor = None
        self.current_video = None
        self.training_thread = None
        
        # Set up logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI
        self.setup_ui()
        self.system_monitor = SystemMonitor()
        self.system_monitor.start()
        self.setup_connections()
        self.setup_shortcuts()
        self.load_settings()
        
        # Set up system monitoring
        self.system_monitor = SystemMonitor()
        self.system_monitor.start()
        
        self.logger.info(f"{APP_NAME} v{APP_VERSION} started")
    
    def setup_logging(self):
        """Set up application logging"""
        log_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        setup_logging(log_dir / LOG_FILE)
    
    def setup_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self.setStyleSheet(create_dark_theme())
        
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Video selection and settings
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Progress and results
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 600])
        
        # Create status bar
        self.create_status_bar()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def create_left_panel(self) -> QWidget:
        """Create the left panel with video selection and options"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Video preview and selection
        self.video_preview = VideoPreviewWidget()
        layout.addWidget(self.video_preview)
        
        # Tab widget for different options
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Model presets tab
        self.model_presets = ModelPresetsWidget(self.config_manager)
        tabs.addTab(self.model_presets, "Model Presets")
        
        # Training options tab
        self.training_options = TrainingOptionsWidget(self.config_manager)
        tabs.addTab(self.training_options, "Training Options")
        
        # Resource settings tab
        self.resource_guard = ResourceGuardWidget()
        tabs.addTab(self.resource_guard, "Resource Settings")
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.quick_test_btn = QPushButton("Quick Test (30s)")
        self.quick_test_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.quick_test_btn.clicked.connect(self.run_quick_test)
        controls_layout.addWidget(self.quick_test_btn)
        
        self.start_btn = QPushButton("Start Compression")
        self.start_btn.setIcon(QIcon.fromTheme("media-record"))
        self.start_btn.clicked.connect(self.start_compression)
        self.start_btn.setObjectName("primaryButton")
        controls_layout.addWidget(self.start_btn)
        
        layout.addLayout(controls_layout)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with progress and results"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for progress and results
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Progress tab
        self.progress_widget = ProgressWidget()
        tabs.addTab(self.progress_widget, "Progress")
        
        # Results tab
        self.results_widget = ResultsWidget()
        tabs.addTab(self.results_widget, "Results")
        
        # History tab
        self.history_widget = HistoryWidget()
        tabs.addTab(self.history_widget, "History")
        
        # Log viewer tab
        self.log_viewer = LogViewerWidget()
        tabs.addTab(self.log_viewer, "Logs")
        
        return panel
    
    def create_status_bar(self):
        """Create the status bar"""
        status_bar = self.statusBar()
        
        # System info label
        self.system_info_label = QLabel()
        self.update_system_info()
        status_bar.addPermanentWidget(self.system_info_label)
        
        # GPU info label
        self.gpu_info_label = QLabel()
        self.update_gpu_info()
        status_bar.addPermanentWidget(self.gpu_info_label)
        
        # Timer to update system info
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_system_info)
        self.status_timer.timeout.connect(self.update_gpu_info)
        self.status_timer.start(5000)  # Update every 5 seconds
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        open_action = QAction("&Open Video", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")
        
        preferences_action = QAction("&Preferences", self)
        preferences_action.triggered.connect(self.show_preferences)
        edit_menu.addAction(preferences_action)
        
        # Run menu
        run_menu = menu_bar.addMenu("&Run")
        
        start_action = QAction("&Start Compression", self)
        start_action.setShortcut(QKeySequence("Ctrl+R"))
        start_action.triggered.connect(self.start_compression)
        run_menu.addAction(start_action)
        
        pause_action = QAction("&Pause/Resume", self)
        pause_action.setShortcut(QKeySequence("Space"))
        pause_action.triggered.connect(self.pause_resume)
        run_menu.addAction(pause_action)
        
        stop_action = QAction("&Stop", self)
        stop_action.setShortcut(QKeySequence("Escape"))
        stop_action.triggered.connect(self.stop_compression)
        run_menu.addAction(stop_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = self.addToolBar("Main")
        
        # Open video action
        open_action = toolbar.addAction("Open Video")
        open_action.setIcon(QIcon.fromTheme("document-open"))
        open_action.triggered.connect(self.open_video)
        
        toolbar.addSeparator()
        
        # Start/Stop actions
        self.start_action = toolbar.addAction("Start")
        self.start_action.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_action.triggered.connect(self.start_compression)
        
        self.pause_action = toolbar.addAction("Pause")
        self.pause_action.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.pause_action.triggered.connect(self.pause_resume)
        self.pause_action.setEnabled(False)
        
        self.stop_action = toolbar.addAction("Stop")
        self.stop_action.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_action.triggered.connect(self.stop_compression)
        self.stop_action.setEnabled(False)
    
    def setup_connections(self):
        """Set up signal-slot connections"""
        # Video preview connections
        self.video_preview.video_loaded.connect(self.on_video_loaded)
        self.video_preview.video_error.connect(self.on_video_error)
        
        # Model presets connections
        self.model_presets.preset_changed.connect(self.on_preset_changed)
        
        # System monitor connections
        self.system_monitor.stats_updated.connect(self.on_system_stats_updated)
    
    def setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        # Open video shortcut
        QShortcut(QKeySequence.Open, self, self.open_video)
        
        # Run shortcut
        QShortcut(QKeySequence("Ctrl+R"), self, self.start_compression)
        
        # Pause/Resume shortcut
        QShortcut(QKeySequence("Space"), self, self.pause_resume)
        
        # Stop shortcut
        QShortcut(QKeySequence("Escape"), self, self.stop_compression)
    
    def load_settings(self):
        """Load application settings"""
        # Window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Window state
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
        
        # Last directory
        self.last_video_dir = self.settings.value("lastVideoDir", "")
        self.last_output_dir = self.settings.value("lastOutputDir", "")
    
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("lastVideoDir", self.last_video_dir)
        self.settings.setValue("lastOutputDir", self.last_output_dir)
    
    def open_video(self):
        """Open a video file"""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mkv *.mov *.webm)")
        file_dialog.setDirectory(self.last_video_dir or "")
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.last_video_dir = os.path.dirname(file_path)
            self.video_preview.load_video(file_path)
    
    def on_video_loaded(self, video_info: Dict):
        """Handle video loaded signal"""
        self.current_video = video_info
        self.logger.info(f"Loaded video: {video_info['path']}")
        
        # Update UI elements based on video info
        self.training_options.update_for_video(video_info)
        self.resource_guard.update_for_video(video_info)
        
        # Enable start button
        self.start_btn.setEnabled(True)
        self.start_action.setEnabled(True)
        self.quick_test_btn.setEnabled(True)
    
    def on_video_error(self, error_msg: str):
        """Handle video loading error"""
        self.logger.error(f"Video loading error: {error_msg}")
        QMessageBox.warning(self, "Video Error", f"Failed to load video:\n{error_msg}")
    
    def on_preset_changed(self, preset_info: Dict):
        """Handle model preset change"""
        self.logger.info(f"Model preset changed: {preset_info['name']}")
        # Update training options based on preset
        self.training_options.load_preset(preset_info)
    
    def on_system_stats_updated(self, stats: Dict):
        """Handle system stats update"""
        self.progress_widget.update_system_stats(stats)
    
    def update_system_info(self):
        """Update system information in status bar"""
        info = get_system_info()
        self.system_info_label.setText(
            f"CPU: {info['cpu_usage']}% | RAM: {info['ram_usage']}%"
        )
    
    def update_gpu_info(self):
        """Update GPU information in status bar"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get memory info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = mem_info.used / 1024 / 1024
            total_mb = mem_info.total / 1024 / 1024
            
            # Get GPU utilization
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            
            self.gpu_info_label.setText(
                f"GPU: {util.gpu}% | VRAM: {used_mb:.0f}/{total_mb:.0f} MB"
            )
        except:
            self.gpu_info_label.setText("GPU: N/A")
    
    def run_quick_test(self):
        """Run a quick test compression"""
        if not self.current_video:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        
        # Set up quick test configuration
        config = self.create_compression_config()
        config.update({
            'epochs': 1,
            'max_frames': 5,
            'batch_size': 2,
            'quick_test': True
        })
        
        self.start_compression_with_config(config)
    
    def start_compression(self):
        """Start the compression process"""
        if not self.current_video:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        
        # Create configuration from UI
        config = self.create_compression_config()
        self.start_compression_with_config(config)
    
    def create_compression_config(self) -> Dict:
        """Create compression configuration from UI settings"""
        config = {
            'video_path': self.current_video['path'],
            'output_dir': self.get_output_directory(),
            'model_preset': self.model_presets.get_selected_preset(),
            'training_options': self.training_options.get_options(),
            'resource_limits': self.resource_guard.get_limits(),
        }
        return config
    
    def get_output_directory(self) -> str:
        """Get the output directory for compression results"""
        if not self.last_output_dir:
            default_dir = os.path.join(os.path.expanduser("~"), "HiNeRV_Output")
        else:
            default_dir = self.last_output_dir
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(self.current_video['path']).stem
        output_dir = os.path.join(default_dir, f"{video_name}_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def start_compression_with_config(self, config: Dict):
        """Start compression with the given configuration"""
        # Update UI state
        self.start_btn.setEnabled(False)
        self.start_action.setEnabled(False)
        self.pause_action.setEnabled(True)
        self.stop_action.setEnabled(True)
        
        # Create and start processor thread
        self.processor = HiNeRVProcessor(config)
        self.processor.progress_updated.connect(self.progress_widget.update_progress)
        self.processor.status_updated.connect(self.progress_widget.update_status)
        self.processor.error_occurred.connect(self.on_compression_error)
        self.processor.finished.connect(self.on_compression_finished)
        
        # Switch to progress tab
        self.findChild(QTabWidget).setCurrentWidget(self.progress_widget)
        
        # Start the compression
        self.processor.start()
        
        self.logger.info("Started compression process")
    
    def pause_resume(self):
        """Pause or resume the compression process"""
        if self.processor and self.processor.isRunning():
            if self.processor.is_paused:
                self.processor.resume()
                self.pause_action.setText("Pause")
                self.pause_action.setIcon(QIcon.fromTheme("media-playback-pause"))
            else:
                self.processor.pause()
                self.pause_action.setText("Resume")
                self.pause_action.setIcon(QIcon.fromTheme("media-playback-start"))
    
    def stop_compression(self):
        """Stop the compression process"""
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self, "Stop Compression",
                "Are you sure you want to stop the compression process?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processor.stop()
    
    def on_compression_finished(self, results: Dict):
        """Handle compression completion"""
        self.logger.info("Compression finished successfully")
        
        # Update UI state
        self.start_btn.setEnabled(True)
        self.start_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        
        # Show results
        self.results_widget.show_results(results)
        self.findChild(QTabWidget).setCurrentWidget(self.results_widget)
        
        # Add to history
        self.history_widget.add_run(results)
        
        # Show completion notification
        self.show_completion_notification(results)
    
    def on_compression_error(self, error: str):
        """Handle compression error"""
        self.logger.error(f"Compression error: {error}")
        
        # Update UI state
        self.start_btn.setEnabled(True)
        self.start_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        
        # Show error dialog
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Compression Error")
        error_dialog.setText("An error occurred during compression:")
        error_dialog.setDetailedText(error)
        error_dialog.addButton("Copy to Clipboard", QMessageBox.ActionRole)
        error_dialog.addButton(QMessageBox.Ok)
        
        clicked = error_dialog.exec()
        if clicked == 0:  # Copy to clipboard button
            QApplication.clipboard().setText(error)
    
    def show_completion_notification(self, results: Dict):
        """Show compression completion notification"""
        if hasattr(self, 'tray_icon') and self.tray_icon.isVisible():
            self.tray_icon.showMessage(
                "Compression Complete",
                f"Video: {Path(results['video_path']).name}\n"
                f"Size: {format_filesize(results['output_size'])}",
                QSystemTrayIcon.Information,
                5000
            )
    
    def show_preferences(self):
        """Show preferences dialog"""
        # TODO: Implement preferences dialog
        QMessageBox.information(self, "Preferences", "Preferences dialog not implemented yet.")
    
    def show_about(self):
        """Show about dialog"""
        about_text = f"""
        <h2>{APP_NAME} v{APP_VERSION}</h2>
        <p>A modern GUI for HiNeRV video compression framework.</p>
        <p>Built with Python and PySide6.</p>
        <p><a href="https://github.com/megvii-research/HiNeRV">HiNeRV on GitHub</a></p>
        """
        QMessageBox.about(self, "About", about_text)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event for video files"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if any(file_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']):
                    event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event for video files"""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            self.video_preview.load_video(file_path)
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Save settings
        self.save_settings()
        
        # Stop any running processes
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self, "Quit",
                "Compression is still running. Do you want to stop it and quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processor.stop()
                self.processor.wait(5000)  # Wait up to 5 seconds
            else:
                event.ignore()
                return
        
        # Stop system monitor
        if hasattr(self, 'system_monitor'):
            self.system_monitor.stop()
        
        # Accept the close event
        event.accept()


def check_system_requirements():
    """Check system requirements and dependencies"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 11):
        issues.append("Python 3.11 or higher is required")
    
    # Check for required packages
    required_packages = [
        'torch', 'torchvision', 'accelerate', 'deepspeed',
        'pytorch_msssim', 'timm'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Required package '{package}' is not installed")
    
    # Check for CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA is not available. GPU acceleration is required.")
    except ImportError:
        pass
    
    return issues


def show_splash_screen():
    """Show splash screen during startup"""
    pixmap = QPixmap(400, 300)
    pixmap.fill(Qt.black)
    
    splash = QSplashScreen(pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.show()
    
    # Show loading messages
    messages = [
        "Loading HiNeRV GUI...",
        "Checking system requirements...",
        "Initializing components...",
        "Ready!"
    ]
    
    for i, message in enumerate(messages):
        splash.showMessage(message, Qt.AlignBottom | Qt.AlignCenter)
        QApplication.processEvents()
        QTimer.singleShot(500 * (i + 1), lambda m=message: splash.showMessage(m, Qt.AlignBottom | Qt.AlignCenter))
    
    return splash


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("HiNeRV")
    
    # Show splash screen
    splash = show_splash_screen()
    
    # Check system requirements
    issues = check_system_requirements()
    if issues:
        splash.close()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("System Requirements")
        msg.setText("Some system requirements are not met:")
        msg.setDetailedText("\n".join(issues))
        msg.exec()
        sys.exit(1)
    
    # Create and show main window
    window = MainWindow()
    splash.finish(window)
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()