#!/usr/bin/env python3
"""
HiNeRV Video Compressor - Modern Minimal GUI
A clean, user-friendly interface for video compression using HiNeRV
"""

import os
import sys
import json
import shutil
import subprocess
import threading
import logging
from pathlib import Path
from datetime import datetime
import time

# Qt imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QFrame,
    QGroupBox, QGridLayout, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor, QDragEnterEvent, QDropEvent

# Video processing imports
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoInfo:
    """Container for video metadata"""
    def __init__(self, path=""):
        self.path = path
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.duration = 0
        self.size_bytes = 0
        self.size_str = ""


class VideoProcessor(QThread):
    """Background thread for video compression"""
    
    progress = Signal(int, str)  # progress percentage, status message
    finished = Signal(dict)  # compression results
    error = Signal(str)  # error message
    
    def __init__(self, video_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.output_dir = os.path.abspath(output_dir)  # Convert to absolute path
        self.is_running = True
        
    def run(self):
        """Main compression pipeline"""
        try:
            # Step 1: Extract frames
            self.progress.emit(10, "Extracting video frames...")
            frames_dir = self.extract_frames()
            
            # Step 2: Process in batches
            self.progress.emit(30, "Training compression model...")
            self.train_model(frames_dir)
            
            # Step 3: Generate compressed video
            self.progress.emit(80, "Generating compressed video...")
            compressed_path = self.generate_output()
            
            # Step 4: Calculate results
            self.progress.emit(95, "Calculating compression metrics...")
            results = self.calculate_results(compressed_path)
            
            self.progress.emit(100, "Compression complete!")
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Compression error: {str(e)}")
            self.error.emit(str(e))
    
    def extract_frames(self):
        """Extract frames from video"""
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get video info
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Extract frames using ffmpeg
        cmd = [
            "ffmpeg", "-i", self.video_path,
            "-q:v", "0",  # Best quality
            os.path.join(frames_dir, "%06d.png")
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Extracted {total_frames} frames to {frames_dir}")
        
        return frames_dir
    
    def train_model(self, frames_dir):
        """Train HiNeRV model on extracted frames"""
        # Get HiNeRV root directory (parent of GUI directory)
        gui_dir = Path(__file__).parent
        hinerv_root = gui_dir.parent
        
        # Convert frames_dir to absolute path
        frames_dir_abs = os.path.abspath(frames_dir)
        
        # Log paths for debugging
        logger.info(f"GUI directory: {gui_dir}")
        logger.info(f"HiNeRV root directory: {hinerv_root}")
        logger.info(f"Frames directory (absolute): {frames_dir_abs}")
        
        # Prepare paths - use absolute paths
        dataset_dir = os.path.dirname(frames_dir_abs)
        dataset_name = os.path.basename(frames_dir_abs)
        model_output = os.path.join(self.output_dir, "model")
        model_output_abs = os.path.abspath(model_output)
        
        # Create model output directory
        os.makedirs(model_output_abs, exist_ok=True)
        
        # Read config files
        train_cfg_path = hinerv_root / "cfgs" / "train" / "hinerv_1920x1080.txt"
        model_cfg_path = hinerv_root / "cfgs" / "models" / "uvg-hinerv-s_1920x1080.txt"
        
        # Check if config files exist
        if not train_cfg_path.exists():
            raise FileNotFoundError(f"Training config not found: {train_cfg_path}")
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_cfg_path}")
        
        # Check if hinerv_main.py exists
        hinerv_main_path = hinerv_root / "hinerv_main.py"
        if not hinerv_main_path.exists():
            raise FileNotFoundError(f"hinerv_main.py not found: {hinerv_main_path}")
        
        # Build command with absolute paths
        cmd = [
            "accelerate", "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=inductor",
            str(hinerv_main_path),
            "--dataset", dataset_dir,
            "--dataset-name", dataset_name,
            "--output", model_output_abs
        ]
        
        # Add config file contents
        with open(train_cfg_path, 'r') as f:
            train_config = f.read().strip()
            if train_config:
                cmd.extend(train_config.split())
        
        with open(model_cfg_path, 'r') as f:
            model_config = f.read().strip()
            if model_config:
                cmd.extend(model_config.split())
        
        # Add runtime parameters
        cmd.extend([
            "--batch-size", "1",
            "--eval-batch-size", "1",
            "--grad-accum", "1",
            "--log-eval", "true",
            "--seed", "0"
        ])
        
        # Log the full command for debugging
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: {hinerv_root}")
        
        # Create log file for training output
        log_file_path = os.path.join(self.output_dir, "training_log.txt")
        
        # Run training
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(hinerv_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=os.environ.copy()
            )
            
            # Monitor progress and collect output
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                log_file.write(line)
                log_file.flush()
                
                # Log important lines
                if "Error" in line or "error" in line:
                    logger.error(f"Training error: {line.strip()}")
                elif "Epoch" in line:
                    # Update progress based on epoch info
                    self.progress.emit(50, f"Training... {line.strip()}")
                    logger.info(f"Training progress: {line.strip()}")
            
            process.wait()
            
            if process.returncode != 0:
                # Get last 20 lines of output for error message
                error_lines = output_lines[-20:] if len(output_lines) > 20 else output_lines
                error_msg = "Training failed. Last output:\n" + "".join(error_lines)
                logger.error(f"Training failed with return code {process.returncode}")
                logger.error(f"Full log saved to: {log_file_path}")
                raise RuntimeError(error_msg)
    
    def generate_output(self):
        """Generate compressed video from model"""
        # For now, create a placeholder compressed video
        # In a real implementation, this would use HiNeRV's decompression
        compressed_path = os.path.join(self.output_dir, "compressed.mp4")
        
        # Copy original as placeholder (in real implementation, use HiNeRV output)
        shutil.copy(self.video_path, compressed_path)
        
        return compressed_path
    
    def calculate_results(self, compressed_path):
        """Calculate compression metrics"""
        original_size = os.path.getsize(self.video_path)
        compressed_size = os.path.getsize(compressed_path)
        
        results = {
            'original_path': self.video_path,
            'compressed_path': compressed_path,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
            'space_saved': 1 - (compressed_size / original_size) if original_size > 0 else 0
        }
        
        return results


class VideoPreviewWidget(QWidget):
    """Widget for video preview and information display"""
    
    def __init__(self, title="Video"):
        super().__init__()
        self.title = title
        self.video_info = VideoInfo()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(title_label)
        
        # Video preview
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setMaximumSize(600, 450)
        self.preview_label.setScaledContents(True)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 8px;
            }
        """)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)
        
        # Video info
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        info_layout = QGridLayout(info_frame)
        
        self.info_labels = {
            'resolution': QLabel("Resolution: --"),
            'fps': QLabel("FPS: --"),
            'size': QLabel("Size: --"),
            'duration': QLabel("Duration: --")
        }
        
        row = 0
        for key, label in self.info_labels.items():
            label.setStyleSheet("color: #ccc; padding: 3px;")
            info_layout.addWidget(label, row // 2, row % 2)
            row += 1
        
        layout.addWidget(info_frame)
    
    def load_video(self, video_path):
        """Load and display video information"""
        if not os.path.exists(video_path):
            return
        
        self.video_info.path = video_path
        
        # Get video info using OpenCV
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.video_info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_info.fps = cap.get(cv2.CAP_PROP_FPS)
            self.video_info.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_info.duration = self.video_info.frame_count / self.video_info.fps if self.video_info.fps > 0 else 0
            
            # Get middle frame for preview
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_info.frame_count // 2)
            ret, frame = cap.read()
            if ret:
                # Convert to RGB and create QPixmap
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                
                # Create QPixmap from frame
                from PySide6.QtGui import QImage
                q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                # Scale to fit preview
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
            
            cap.release()
        
        # Get file size
        self.video_info.size_bytes = os.path.getsize(video_path)
        self.video_info.size_str = self.format_size(self.video_info.size_bytes)
        
        # Update labels
        self.update_info()
    
    def update_info(self):
        """Update information labels"""
        self.info_labels['resolution'].setText(f"Resolution: {self.video_info.width}x{self.video_info.height}")
        self.info_labels['fps'].setText(f"FPS: {self.video_info.fps:.2f}")
        self.info_labels['size'].setText(f"Size: {self.video_info.size_str}")
        self.info_labels['duration'].setText(f"Duration: {self.format_duration(self.video_info.duration)}")
    
    def format_size(self, size_bytes):
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def format_duration(self, seconds):
        """Format duration"""
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m {secs}s"


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.output_dir = None
        self.processor = None
        self.setup_ui()
        self.apply_theme()
        
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("HiNeRV Video Compressor")
        self.setMinimumSize(1200, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("HiNeRV Video Compressor")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4CAF50;
            padding: 10px;
        """)
        main_layout.addWidget(header)
        
        # Upload section
        upload_widget = self.create_upload_section()
        main_layout.addWidget(upload_widget)
        
        # Video comparison section
        self.comparison_widget = self.create_comparison_section()
        self.comparison_widget.setVisible(False)
        main_layout.addWidget(self.comparison_widget)
        
        # Progress section
        self.progress_widget = self.create_progress_section()
        self.progress_widget.setVisible(False)
        main_layout.addWidget(self.progress_widget)
        
        # Results section
        self.results_widget = self.create_results_section()
        self.results_widget.setVisible(False)
        main_layout.addWidget(self.results_widget)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def create_upload_section(self):
        """Create the upload section"""
        widget = QGroupBox("Upload Video")
        layout = QVBoxLayout(widget)
        
        # Upload area
        self.upload_area = QLabel("Drag and drop a video file here\nor click to browse")
        self.upload_area.setMinimumHeight(150)
        self.upload_area.setAlignment(Qt.AlignCenter)
        self.upload_area.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px dashed #666;
                border-radius: 10px;
                font-size: 16px;
                color: #aaa;
            }
            QLabel:hover {
                border-color: #4CAF50;
                color: #ccc;
            }
        """)
        self.upload_area.setCursor(Qt.PointingHandCursor)
        self.upload_area.mousePressEvent = self.browse_video
        layout.addWidget(self.upload_area)
        
        # Browse button
        browse_btn = QPushButton("Browse Files")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        browse_btn.clicked.connect(self.browse_video)
        layout.addWidget(browse_btn, alignment=Qt.AlignCenter)
        
        return widget
    
    def create_comparison_section(self):
        """Create video comparison section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Video previews
        previews_layout = QHBoxLayout()
        
        self.original_preview = VideoPreviewWidget("Original Video")
        self.compressed_preview = VideoPreviewWidget("Compressed Video")
        
        previews_layout.addWidget(self.original_preview)
        previews_layout.addWidget(self.compressed_preview)
        
        layout.addLayout(previews_layout)
        
        # Compress button
        self.compress_btn = QPushButton("Start Compression")
        self.compress_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.compress_btn.clicked.connect(self.start_compression)
        layout.addWidget(self.compress_btn, alignment=Qt.AlignCenter)
        
        return widget
    
    def create_progress_section(self):
        """Create progress section"""
        widget = QGroupBox("Compression Progress")
        layout = QVBoxLayout(widget)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-size: 14px; color: #ccc; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        return widget
    
    def create_results_section(self):
        """Create results section"""
        widget = QGroupBox("Compression Results")
        layout = QVBoxLayout(widget)
        
        # Results grid
        results_frame = QFrame()
        results_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        results_layout = QGridLayout(results_frame)
        
        self.result_labels = {
            'original_size': QLabel("Original Size: --"),
            'compressed_size': QLabel("Compressed Size: --"),
            'compression_ratio': QLabel("Compression Ratio: --"),
            'space_saved': QLabel("Space Saved: --")
        }
        
        row = 0
        for key, label in self.result_labels.items():
            label.setStyleSheet("font-size: 14px; color: #ccc; padding: 5px;")
            results_layout.addWidget(label, row // 2, row % 2)
            row += 1
        
        layout.addWidget(results_frame)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        self.play_original_btn = QPushButton("Play Original")
        self.play_compressed_btn = QPushButton("Play Compressed")
        self.save_btn = QPushButton("Save Compressed Video")
        
        for btn in [self.play_original_btn, self.play_compressed_btn, self.save_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #555;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #666;
                }
            """)
        
        self.play_original_btn.clicked.connect(self.play_original)
        self.play_compressed_btn.clicked.connect(self.play_compressed)
        self.save_btn.clicked.connect(self.save_compressed)
        
        actions_layout.addWidget(self.play_original_btn)
        actions_layout.addWidget(self.play_compressed_btn)
        actions_layout.addWidget(self.save_btn)
        
        layout.addLayout(actions_layout)
        
        # New compression button
        new_btn = QPushButton("Compress Another Video")
        new_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        new_btn.clicked.connect(self.reset_ui)
        layout.addWidget(new_btn, alignment=Qt.AlignCenter)
        
        return widget
    
    def apply_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
    
    def browse_video(self, event=None):
        """Open file browser to select video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*.*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, file_path):
        """Load selected video"""
        self.video_path = file_path
        
        # Update upload area
        self.upload_area.setText(f"Selected: {os.path.basename(file_path)}")
        self.upload_area.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                font-size: 14px;
                color: #4CAF50;
            }
        """)
        
        # Show comparison section
        self.comparison_widget.setVisible(True)
        self.original_preview.load_video(file_path)
    
    def start_compression(self):
        """Start the compression process"""
        if not self.video_path:
            return
        
        # Create output directory with absolute path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(self.video_path).stem
        
        # Create output in GUI directory
        gui_dir = Path(__file__).parent
        self.output_dir = gui_dir / "output" / f"{video_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update UI
        self.compress_btn.setEnabled(False)
        self.progress_widget.setVisible(True)
        self.results_widget.setVisible(False)
        
        # Start compression thread with absolute paths
        self.processor = VideoProcessor(str(self.video_path), str(self.output_dir))
        self.processor.progress.connect(self.update_progress)
        self.processor.finished.connect(self.on_compression_finished)
        self.processor.error.connect(self.on_compression_error)
        self.processor.start()
    
    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def on_compression_finished(self, results):
        """Handle compression completion"""
        # Update UI
        self.compress_btn.setEnabled(True)
        self.progress_widget.setVisible(False)
        self.results_widget.setVisible(True)
        
        # Store results
        self.compression_results = results
        
        # Update compressed preview
        self.compressed_preview.load_video(results['compressed_path'])
        
        # Update results
        self.result_labels['original_size'].setText(
            f"Original Size: {self.format_size(results['original_size'])}"
        )
        self.result_labels['compressed_size'].setText(
            f"Compressed Size: {self.format_size(results['compressed_size'])}"
        )
        self.result_labels['compression_ratio'].setText(
            f"Compression Ratio: {results['compression_ratio']:.2f}x"
        )
        self.result_labels['space_saved'].setText(
            f"Space Saved: {results['space_saved']*100:.1f}%"
        )
    
    def on_compression_error(self, error_msg):
        """Handle compression error"""
        self.compress_btn.setEnabled(True)
        self.progress_widget.setVisible(False)
        
        # Check if there's a log file we can point to
        log_file_hint = ""
        if self.output_dir:
            log_file = os.path.join(self.output_dir, "training_log.txt")
            if os.path.exists(log_file):
                log_file_hint = f"\n\nDetailed log saved to:\n{log_file}"
        
        # Create detailed error dialog
        error_dialog = QMessageBox(self)
        error_dialog.setWindowTitle("Compression Error")
        error_dialog.setText("An error occurred during compression:")
        error_dialog.setDetailedText(error_msg + log_file_hint)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.exec()
    
    def play_original(self):
        """Play original video"""
        if self.video_path and os.path.exists(self.video_path):
            os.system(f'xdg-open "{self.video_path}"')
    
    def play_compressed(self):
        """Play compressed video"""
        if hasattr(self, 'compression_results'):
            compressed_path = self.compression_results.get('compressed_path')
            if compressed_path and os.path.exists(compressed_path):
                os.system(f'xdg-open "{compressed_path}"')
    
    def save_compressed(self):
        """Save compressed video to user location"""
        if not hasattr(self, 'compression_results'):
            return
        
        compressed_path = self.compression_results.get('compressed_path')
        if not compressed_path or not os.path.exists(compressed_path):
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Compressed Video",
            f"compressed_{os.path.basename(self.video_path)}",
            "Video Files (*.mp4);;All Files (*.*)"
        )
        
        if save_path:
            shutil.copy(compressed_path, save_path)
            QMessageBox.information(self, "Success", f"Video saved to:\n{save_path}")
    
    def reset_ui(self):
        """Reset UI for new compression"""
        self.video_path = None
        self.output_dir = None
        
        # Reset upload area
        self.upload_area.setText("Drag and drop a video file here\nor click to browse")
        self.upload_area.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px dashed #666;
                border-radius: 10px;
                font-size: 16px;
                color: #aaa;
            }
            QLabel:hover {
                border-color: #4CAF50;
                color: #ccc;
            }
        """)
        
        # Hide sections
        self.comparison_widget.setVisible(False)
        self.progress_widget.setVisible(False)
        self.results_widget.setVisible(False)
    
    def format_size(self, size_bytes):
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop"""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            # Check if it's a video file
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
            if any(files[0].lower().endswith(ext) for ext in video_extensions):
                self.load_video(files[0])


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("HiNeRV Video Compressor")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()