#!/usr/bin/env python3
"""
VideoPreviewWidget - Widget for video preview and selection
"""

import os
import cv2
import logging
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PySide6.QtGui import QPixmap, QMovie, QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget


class VideoAnalyzer(QThread):
    """Background thread for video analysis"""
    
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Analyze video file"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                self.error_occurred.emit(f"Could not open video: {self.video_path}")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get file size
            file_size = os.path.getsize(self.video_path)
            
            # Get codec info
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            
            # Extract thumbnail from middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            thumbnail_path = None
            if ret:
                # Save thumbnail
                thumbnail_dir = Path.home() / ".hinerv_gui" / "thumbnails"
                thumbnail_dir.mkdir(parents=True, exist_ok=True)
                
                video_name = Path(self.video_path).stem
                thumbnail_path = thumbnail_dir / f"{video_name}_thumb.jpg"
                cv2.imwrite(str(thumbnail_path), frame)
            
            cap.release()
            
            # Emit results
            info = {
                'path': self.video_path,
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'file_size': file_size,
                'codec': codec,
                'thumbnail': str(thumbnail_path) if thumbnail_path else None
            }
            
            self.analysis_complete.emit(info)
            
        except Exception as e:
            self.logger.error(f"Video analysis error: {e}")
            self.error_occurred.emit(str(e))


class VideoPreviewWidget(QWidget):
    """Widget for video preview and selection"""
    
    video_loaded = pyqtSignal(dict)
    video_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.video_info = None
        self.analyzer_thread = None
        self.logger = logging.getLogger(__name__)
        
        self.setup_ui()
        self.setAcceptDrops(True)
    
    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout(self)
        
        # Create video preview group
        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Thumbnail area
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setMinimumSize(320, 180)
        self.thumbnail_label.setMaximumSize(640, 360)
        self.thumbnail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px dashed #555;
                border-radius: 8px;
                color: #aaa;
                text-align: center;
                padding: 20px;
            }
        """)
        self.thumbnail_label.setText("No video loaded\nClick 'Open Video' or drag & drop a video file")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setScaledContents(True)
        preview_layout.addWidget(self.thumbnail_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.open_button = QPushButton("Open Video")
        self.open_button.clicked.connect(self.open_video)
        controls_layout.addWidget(self.open_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_video)
        self.clear_button.setEnabled(False)
        controls_layout.addWidget(self.clear_button)
        
        controls_layout.addStretch()
        preview_layout.addLayout(controls_layout)
        
        layout.addWidget(preview_group)
        
        # Video info group
        info_group = QGroupBox("Video Information")
        info_layout = QVBoxLayout(info_group)
        
        # Info labels
        self.info_labels = {
            'path': QLabel("Path: N/A"),
            'resolution': QLabel("Resolution: N/A"),
            'duration': QLabel("Duration: N/A"),
            'fps': QLabel("FPS: N/A"),
            'codec': QLabel("Codec: N/A"),
            'size': QLabel("Size: N/A")
        }
        
        for label in self.info_labels.values():
            label.setWordWrap(True)
            info_layout.addWidget(label)
        
        layout.addWidget(info_group)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def open_video(self):
        """Open video file dialog"""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mkv *.mov *.webm *.m4v *.flv *.3gp)")
        file_dialog.setWindowTitle("Select Video File")
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.load_video(file_path)
    
    def load_video(self, file_path: str):
        """Load a video file"""
        if not os.path.exists(file_path):
            self.video_error.emit(f"File not found: {file_path}")
            return
        
        # Show loading state
        self.thumbnail_label.setText("Loading video...")
        self.clear_button.setEnabled(False)
        
        # Start video analysis thread
        self.analyzer_thread = VideoAnalyzer(file_path)
        self.analyzer_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analyzer_thread.error_occurred.connect(self.on_analysis_error)
        self.analyzer_thread.start()
    
    def on_analysis_complete(self, info: Dict):
        """Handle video analysis completion"""
        self.video_info = info
        self.update_ui(info)
        self.clear_button.setEnabled(True)
        self.video_loaded.emit(info)
    
    def on_analysis_error(self, error: str):
        """Handle video analysis error"""
        self.thumbnail_label.setText(f"Error loading video:\n{error}")
        self.clear_button.setEnabled(False)
        self.video_error.emit(error)
    
    def update_ui(self, info: Dict):
        """Update UI with video information"""
        # Update thumbnail
        if info.get('thumbnail') and os.path.exists(info['thumbnail']):
            pixmap = QPixmap(info['thumbnail'])
            scaled_pixmap = pixmap.scaled(
                self.thumbnail_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled_pixmap)
        else:
            self.thumbnail_label.setText("Video loaded\n(No thumbnail available)")
        
        # Update info labels
        path = Path(info['path'])
        self.info_labels['path'].setText(f"Path: {path.name}")
        self.info_labels['resolution'].setText(f"Resolution: {info['width']}x{info['height']}")
        
        duration_str = self.format_duration(info['duration'])
        self.info_labels['duration'].setText(f"Duration: {duration_str}")
        self.info_labels['fps'].setText(f"FPS: {info['fps']:.2f}")
        self.info_labels['codec'].setText(f"Codec: {info['codec']}")
        
        size_str = self.format_file_size(info['file_size'])
        self.info_labels['size'].setText(f"Size: {size_str}")
    
    def clear_video(self):
        """Clear the loaded video"""
        self.video_info = None
        
        # Reset thumbnail
        self.thumbnail_label.clear()
        self.thumbnail_label.setText("No video loaded\nClick 'Open Video' or drag & drop a video file")
        
        # Reset info labels
        for key, label in self.info_labels.items():
            if key == 'path':
                label.setText("Path: N/A")
            elif key == 'resolution':
                label.setText("Resolution: N/A")
            elif key == 'duration':
                label.setText("Duration: N/A")
            elif key == 'fps':
                label.setText("FPS: N/A")
            elif key == 'codec':
                label.setText("Codec: N/A")
            elif key == 'size':
                label.setText("Size: N/A")
        
        self.clear_button.setEnabled(False)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if self.is_video_file(file_path):
                    event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                self.load_video(file_path)
    
    def is_video_file(self, file_path: str) -> bool:
        """Check if file is a video file"""
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.flv', '.3gp']
        return any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @staticmethod
    def format_file_size(size: int) -> str:
        """Format file size in bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"