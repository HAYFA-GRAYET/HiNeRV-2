# src/components/history_widget.py
import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QGroupBox, QFormLayout, QSplitter,
    QFileDialog, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QAction


class HistoryWidget(QWidget):
    """Widget for displaying compression run history"""
    
    run_selected = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_items = []
        self.setup_ui()
        self.load_history()
        
    def setup_ui(self):
        """Set up the history UI"""
        layout = QVBoxLayout(self)
        
        # Controls header
        header_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_history)
        header_layout.addWidget(self.refresh_btn)
        
        self.open_dir_btn = QPushButton("Set History Folder")
        self.open_dir_btn.clicked.connect(self.set_history_folder)
        header_layout.addWidget(self.open_dir_btn)
        
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Split view: list + details
        splitter = QSplitter(Qt.Horizontal)
        
        # History list
        self.history_list = QListWidget()
        self.history_list.setAlternatingRowColors(True)
        self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self.show_context_menu)
        self.history_list.currentItemChanged.connect(self.on_item_selected)
        splitter.addWidget(self.history_list)
        
        # Details panel
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        self.info_group = QGroupBox("Run Information")
        info_layout = QFormLayout(self.info_group)
        
        self.video_name_label = QLabel("-")
        info_layout.addRow("Video:", self.video_name_label)
        
        self.timestamp_label = QLabel("-")
        info_layout.addRow("Date/Time:", self.timestamp_label)
        
        self.model_preset_label = QLabel("-")
        info_layout.addRow("Model Preset:", self.model_preset_label)
        
        self.output_size_label = QLabel("-")
        info_layout.addRow("Output Size:", self.output_size_label)
        
        self.compression_ratio_label = QLabel("-")
        info_layout.addRow("Compression Ratio:", self.compression_ratio_label)
        
        self.psnr_label = QLabel("-")
        info_layout.addRow("PSNR:", self.psnr_label)
        
        self.ms_ssim_label = QLabel("-")
        info_layout.addRow("MS-SSIM:", self.ms_ssim_label)
        
        details_layout.addWidget(self.info_group)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        self.open_output_btn = QPushButton("Open Folder")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_output_btn.setEnabled(False)
        actions_layout.addWidget(self.open_output_btn)
        
        self.play_video_btn = QPushButton("Play Video")
        self.play_video_btn.clicked.connect(self.play_output_video)
        self.play_video_btn.setEnabled(False)
        actions_layout.addWidget(self.play_video_btn)
        
        self.resume_btn = QPushButton("Resume Training")
        self.resume_btn.clicked.connect(self.resume_training)
        self.resume_btn.setEnabled(False)
        actions_layout.addWidget(self.resume_btn)
        
        details_layout.addLayout(actions_layout)
        details_layout.addStretch()
        
        splitter.addWidget(details_widget)
        splitter.setSizes([200, 400])
        
        layout.addWidget(splitter)
        
    def load_history(self):
        """Load compression history from output folders"""
        self.history_list.clear()
        self.history_items = []
        
        # Get search paths
        search_paths = self.get_search_paths()
        
        # Search for output folders
        for base_path in search_paths:
            if not os.path.exists(base_path):
                continue
                
            for output_dir in Path(base_path).glob("*"):
                if not output_dir.is_dir():
                    continue
                    
                # Check for args.yaml
                args_file = output_dir / "args.yaml"
                if not args_file.exists():
                    continue
                    
                try:
                    # Load run data
                    with open(args_file, 'r') as f:
                        args = yaml.safe_load(f)
                    
                    # Load metrics if available
                    metrics_file = output_dir / "metrics.json"
                    metrics = {}
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                        except:
                            pass
                    
                    # Create run info
                    run_info = {
                        'output_dir': str(output_dir),
                        'video_path': args.get('video_path', ''),
                        'model_preset': args.get('model_preset', {}).get('name', 'Unknown'),
                        'timestamp': self.extract_timestamp(str(output_dir)),
                        'metrics': metrics,
                        'args': args
                    }
                    
                    # Add to list
                    self.add_history_item(run_info)
                    
                except Exception as e:
                    print(f"Error loading history from {output_dir}: {str(e)}")
        
        # Sort items by timestamp (newest first)
        self.history_items.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Rebuild list widget
        self.history_list.clear()
        for item in self.history_items:
            list_item = QListWidgetItem()
            video_name = os.path.basename(item['video_path'])
            timestamp = datetime.fromtimestamp(item['timestamp']).strftime("%Y-%m-%d %H:%M")
            list_item.setText(f"{video_name} - {timestamp}")
            list_item.setData(Qt.UserRole, item)
            self.history_list.addItem(list_item)
            
    def get_search_paths(self) -> List[str]:
        """Get paths to search for history items"""
        # Get from settings
        from PySide6.QtCore import QSettings
        settings = QSettings()
        custom_history_path = settings.value("historyPath", "")
        
        paths = []
        
        # Add custom path if set
        if custom_history_path and os.path.exists(custom_history_path):
            paths.append(custom_history_path)
        
        # Add default path
        default_path = os.path.join(os.path.expanduser("~"), "HiNeRV_Output")
        if os.path.exists(default_path):
            paths.append(default_path)
            
        return paths
    
    def extract_timestamp(self, folder_path: str) -> float:
        """Extract timestamp from folder name or use folder creation time"""
        try:
            # Try to extract timestamp from folder name (format: name_YYYYMMDD_HHMMSS)
            folder_name = os.path.basename(folder_path)
            parts = folder_name.split('_')
            if len(parts) >= 2:
                date_str = parts[-2]
                time_str = parts[-1]
                if len(date_str) == 8 and len(time_str) == 6:
                    dt_str = f"{date_str}_{time_str}"
                    dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
                    return dt.timestamp()
        except:
            pass
            
        # Fall back to folder creation time
        return os.path.getctime(folder_path)
    
    def add_history_item(self, run_info: Dict):
        """Add a history item to the list"""
        self.history_items.append(run_info)
    
    def on_item_selected(self, current, previous):
        """Handle history item selection"""
        if not current:
            self.clear_details()
            return
            
        run_info = current.data(Qt.UserRole)
        if not run_info:
            self.clear_details()
            return
            
        # Update details panel
        self.update_run_details(run_info)
        
        # Enable action buttons
        self.open_output_btn.setEnabled(True)
        self.play_video_btn.setEnabled(self.has_output_video(run_info))
        self.resume_btn.setEnabled(self.can_resume_training(run_info))
    
    def update_run_details(self, run_info: Dict):
        """Update run details display"""
        # Basic info
        self.video_name_label.setText(os.path.basename(run_info['video_path']))
        self.timestamp_label.setText(
            datetime.fromtimestamp(run_info['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        )
        self.model_preset_label.setText(run_info['model_preset'])
        
        # Metrics
        metrics = run_info.get('metrics', {})
        if metrics:
            self.output_size_label.setText(self.format_size(metrics.get('output_size', 0)))
            self.compression_ratio_label.setText(f"{metrics.get('compression_ratio', 0):.2f}x")
            self.psnr_label.setText(f"{metrics.get('psnr', 0):.2f} dB")
            self.ms_ssim_label.setText(f"{metrics.get('ms_ssim', 0):.4f}")
        else:
            self.output_size_label.setText("-")
            self.compression_ratio_label.setText("-")
            self.psnr_label.setText("-")
            self.ms_ssim_label.setText("-")
    
    def clear_details(self):
        """Clear run details display"""
        self.video_name_label.setText("-")
        self.timestamp_label.setText("-")
        self.model_preset_label.setText("-")
        self.output_size_label.setText("-")
        self.compression_ratio_label.setText("-")
        self.psnr_label.setText("-")
        self.ms_ssim_label.setText("-")
        
        self.open_output_btn.setEnabled(False)
        self.play_video_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable string"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.1f} MB"
        else:
            return f"{size_bytes / (1024**3):.2f} GB"
    
    def has_output_video(self, run_info: Dict) -> bool:
        """Check if run has output video file"""
        output_dir = run_info['output_dir']
        video_paths = [
            os.path.join(output_dir, "output.mp4"),
            os.path.join(output_dir, "output.webm"),
            os.path.join(output_dir, "reconstructed.mp4")
        ]
        return any(os.path.exists(path) for path in video_paths)
    
    def can_resume_training(self, run_info: Dict) -> bool:
        """Check if training can be resumed"""
        output_dir = run_info['output_dir']
        checkpoint_path = os.path.join(output_dir, "checkpoints")
        return os.path.exists(checkpoint_path) and any(Path(checkpoint_path).glob("*.pt"))
    
    def open_output_folder(self):
        """Open the selected run's output folder"""
        current_item = self.history_list.currentItem()
        if not current_item:
            return
            
        run_info = current_item.data(Qt.UserRole)
        if not run_info:
            return
            
        output_dir = run_info['output_dir']
        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "Folder Not Found", 
                               f"Output folder does not exist:\n{output_dir}")
            return
            
        # Open folder in file explorer
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
    
    def play_output_video(self):
        """Play the selected run's output video"""
        current_item = self.history_list.currentItem()
        if not current_item:
            return
            
        run_info = current_item.data(Qt.UserRole)
        if not run_info:
            return
            
        output_dir = run_info['output_dir']
        
        # Find video file
        video_paths = [
            os.path.join(output_dir, "output.mp4"),
            os.path.join(output_dir, "output.webm"),
            os.path.join(output_dir, "reconstructed.mp4")
        ]
        
        video_path = next((path for path in video_paths if os.path.exists(path)), None)
        
        if not video_path:
            QMessageBox.warning(self, "Video Not Found", 
                               "Output video file does not exist.")
            return
            
        # Open video with default player
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
    
    def resume_training(self):
        """Resume training from the selected run"""
        current_item = self.history_list.currentItem()
        if not current_item:
            return
            
        run_info = current_item.data(Qt.UserRole)
        if not run_info:
            return
            
        # Emit signal to resume training with this run's data
        self.run_selected.emit(run_info)
    
    def set_history_folder(self):
        """Set custom history folder"""
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        
        # Get current history path from settings
        from PySide6.QtCore import QSettings
        settings = QSettings()
        current_path = settings.value("historyPath", "")
        
        if current_path:
            dialog.setDirectory(current_path)
        
        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                new_path = selected_dirs[0]
                
                # Save to settings
                settings.setValue("historyPath", new_path)
                
                # Reload history
                self.load_history()
    
    def show_context_menu(self, position):
        """Show context menu for history items"""
        current_item = self.history_list.currentItem()
        if not current_item:
            return
            
        run_info = current_item.data(Qt.UserRole)
        if not run_info:
            return
            
        menu = QMenu(self)
        
        # Add actions
        open_action = QAction("Open Folder", self)
        open_action.triggered.connect(self.open_output_folder)
        menu.addAction(open_action)
        
        if self.has_output_video(run_info):
            play_action = QAction("Play Video", self)
            play_action.triggered.connect(self.play_output_video)
            menu.addAction(play_action)
        
        if self.can_resume_training(run_info):
            resume_action = QAction("Resume Training", self)
            resume_action.triggered.connect(self.resume_training)
            menu.addAction(resume_action)
            
        menu.addSeparator()
        
        delete_action = QAction("Delete Run", self)
        delete_action.triggered.connect(self.delete_current_run)
        menu.addAction(delete_action)
        
        # Show the menu
        menu.exec(self.history_list.mapToGlobal(position))
    
    def delete_current_run(self):
        """Delete the selected run"""
        current_item = self.history_list.currentItem()
        if not current_item:
            return
            
        run_info = current_item.data(Qt.UserRole)
        if not run_info:
            return
            
        output_dir = run_info['output_dir']
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Run",
            f"Are you sure you want to delete this run?\n"
            f"This will remove all files in:\n{output_dir}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # Delete directory
        try:
            import shutil
            shutil.rmtree(output_dir)
            
            # Remove from list
            row = self.history_list.row(current_item)
            self.history_list.takeItem(row)
            
            # Update details
            self.clear_details()
            
        except Exception as e:
            QMessageBox.critical(self, "Delete Failed", 
                              f"Failed to delete run:\n{str(e)}")
    
    def add_run(self, results: Dict):
        """Add a new run to the history"""
        # Add to history items
        run_info = {
            'output_dir': results.get('output_dir', ''),
            'video_path': results.get('video_path', ''),
            'model_preset': results.get('model_preset', {}).get('name', 'Unknown'),
            'timestamp': datetime.now().timestamp(),
            'metrics': results.get('metrics', {}),
            'args': results.get('args', {})
        }
        
        self.add_history_item(run_info)
        
        # Add to list widget
        list_item = QListWidgetItem()
        video_name = os.path.basename(run_info['video_path'])
        timestamp = datetime.fromtimestamp(run_info['timestamp']).strftime("%Y-%m-%d %H:%M")
        list_item.setText(f"{video_name} - {timestamp}")
        list_item.setData(Qt.UserRole, run_info)
        
        # Insert at the beginning (newest first)
        self.history_list.insertItem(0, list_item)
        self.history_list.setCurrentItem(list_item)