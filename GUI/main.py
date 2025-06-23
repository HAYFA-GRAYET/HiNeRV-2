#!/usr/bin/env python3
"""
HiNeRV Video Compressor - Modern Minimal GUI with Batch Processing
A clean, user-friendly interface for video compression using HiNeRV with batch processing
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
import glob
import math
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QTextEdit, QScrollArea
# Qt imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QFrame,
    QGroupBox, QGridLayout, QMessageBox, QSplitter,QSpinBox
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
    """Background thread for video compression with batch processing"""
    
    progress = Signal(int, str)  # progress percentage, status message
    finished = Signal(dict)  # compression results
    error = Signal(str)  # error message
    
    def __init__(self, video_path, output_dir, batch_size=40):
        super().__init__()
        self.video_path = video_path
        self.output_dir = os.path.abspath(output_dir)
        self.batch_size = batch_size
        self.is_running = True
        
    def run(self):
        """Main compression pipeline with batch processing"""
        try:
            # Step 1: Extract frames
            self.progress.emit(5, "Extracting video frames...")
            frames_dir = self.extract_frames()
            
            # Step 2: Get total frame count and calculate batches
            frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
            total_frames = len(frame_files)
            total_batches = math.ceil(total_frames / self.batch_size)
            
            self.progress.emit(10, f"Processing {total_frames} frames in {total_batches} batches of {self.batch_size} frames each...")
            logger.info(f"Total frames: {total_frames}, Batches: {total_batches}, Batch size: {self.batch_size}")
            
            # Step 3: Process frames in batches
            output_frames_dir = os.path.join(self.output_dir, "compressed_frames")
            os.makedirs(output_frames_dir, exist_ok=True)
            
            batch_metrics = []  # Store metrics for each batch
            
            for batch_idx in range(total_batches):
                if not self.is_running:
                    break
                    
                # Calculate progress for this batch (10% to 80% of total progress)
                batch_progress_start = 10 + int((batch_idx / total_batches) * 70)
                batch_progress_end = 10 + int(((batch_idx + 1) / total_batches) * 70)
                
                self.progress.emit(
                    batch_progress_start, 
                    f"Processing batch {batch_idx + 1}/{total_batches} ({self.batch_size} frames)..."
                )
                
                # Process this batch and get metrics
                batch_metric = self.process_batch(
                    frames_dir, 
                    batch_idx, 
                    total_batches,
                    output_frames_dir,
                    batch_progress_start,
                    batch_progress_end
                )
                
                if batch_metric:
                    batch_metrics.append(batch_metric)
            
            # Step 4: Combine all compressed frames into video
            self.progress.emit(85, "Combining all compressed frames into final video...")
            compressed_path = self.create_final_video(output_frames_dir)
            
            # Step 5: Calculate comprehensive results
            self.progress.emit(95, "Calculating compression metrics and generating comparison...")
            results = self.calculate_comprehensive_results(compressed_path, batch_metrics)
            
            self.progress.emit(100, "Compression complete!")
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Compression error: {str(e)}")
            self.error.emit(f"Compression failed: {str(e)}")
    
    def extract_frames(self):
        """Extract frames from video"""
        frames_dir = os.path.join(self.output_dir, "original_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames using ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", self.video_path,
            "-q:v", "0",  # Best quality
            os.path.join(frames_dir, "%06d.png")
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Frame extraction completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to extract frames: {e.stderr}")
        
        # Count extracted frames
        frame_files = glob.glob(os.path.join(frames_dir, "*.png"))
        logger.info(f"Extracted {len(frame_files)} frames to {frames_dir}")
        
        return frames_dir
    
    def process_batch(self, frames_dir, batch_idx, total_batches, output_frames_dir, progress_start, progress_end):
        """Process a single batch of frames and return metrics"""
        try:
            # Get frame files for this batch
            all_frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_frame_files))
            batch_frame_files = all_frame_files[start_idx:end_idx]
            
            if not batch_frame_files:
                logger.warning(f"No frames found for batch {batch_idx}")
                return None
            
            actual_batch_size = len(batch_frame_files)
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}: frames {start_idx + 1}-{end_idx} ({actual_batch_size} frames)")
            
            # Create batch directory
            batch_dir = os.path.join(self.output_dir, f"batch_{batch_idx:03d}")
            batch_frames_dir = os.path.join(batch_dir, "frames")
            os.makedirs(batch_frames_dir, exist_ok=True)
            
            # Copy batch frames to batch directory (renumbered from 1)
            for i, frame_file in enumerate(batch_frame_files):
                src = frame_file
                dst = os.path.join(batch_frames_dir, f"{i+1:06d}.png")
                shutil.copy2(src, dst)
            
            # Update progress - Training phase
            train_progress = progress_start + int((progress_end - progress_start) * 0.7)
            self.progress.emit(
                train_progress, 
                f"Training HiNeRV model on batch {batch_idx + 1}/{total_batches} ({actual_batch_size} frames)..."
            )
            
            # Train model on this batch
            training_metrics = self.train_model_batch(batch_frames_dir, batch_dir, batch_idx)
            
            # Update progress - Generation phase
            gen_progress = progress_start + int((progress_end - progress_start) * 0.9)
            self.progress.emit(
                gen_progress,
                f"Extracting compressed frames from batch {batch_idx + 1}/{total_batches}..."
            )
            
            # Extract compressed frames from HiNeRV output
            batch_output_dir = os.path.join(batch_dir, "output_frames")
            compression_metrics = self.extract_compressed_frames(batch_dir, batch_output_dir, actual_batch_size)
            
            # Copy compressed frames to main output directory with correct numbering
            self.copy_batch_output(batch_output_dir, output_frames_dir, start_idx)
            
            # Calculate batch metrics
            batch_metrics = {
                'batch_idx': batch_idx,
                'frame_range': f"{start_idx + 1}-{end_idx}",
                'frame_count': actual_batch_size,
                'training_metrics': training_metrics,
                'compression_metrics': compression_metrics
            }
            
            # Clean up batch directory to save space
            self.cleanup_batch_directory(batch_dir)
            
            logger.info(f"Completed batch {batch_idx + 1}/{total_batches}")
            return batch_metrics
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            raise RuntimeError(f"Batch {batch_idx + 1} failed: {str(e)}")
    
    def train_model_batch(self, batch_frames_dir, batch_dir, batch_idx):
        """Train HiNeRV model on a batch of frames"""
        # Get HiNeRV root directory
        gui_dir = Path(__file__).parent
        hinerv_root = gui_dir.parent
        
        # Prepare paths exactly like the bash script
        dataset_dir = os.path.dirname(batch_frames_dir)
        dataset_name = os.path.basename(batch_frames_dir)
        model_output = os.path.join(batch_dir, "model")
        os.makedirs(model_output, exist_ok=True)
        
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
        
        # Read config files as strings
        with open(train_cfg_path, 'r') as f:
            train_cfg_content = f.read().strip()
        
        with open(model_cfg_path, 'r') as f:
            model_cfg_content = f.read().strip()
        
        # Build command exactly like the bash script
        cmd = [
            "accelerate", "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=inductor",
            str(hinerv_main_path),
            "--dataset", dataset_dir,
            "--dataset-name", dataset_name,
            "--output", model_output
        ]
        
        # Add config file contents
        if train_cfg_content:
            cmd.extend(train_cfg_content.split())
        
        if model_cfg_content:
            cmd.extend(model_cfg_content.split())
        
        # Add runtime arguments (remove epochs - use config file setting)
        cmd.extend([
            "--batch-size", "1",
            "--eval-batch-size", "1",
            "--grad-accum", "1",
            "--log-eval", "true",
            "--seed", str(batch_idx)
        ])
        
        logger.info(f"Training batch {batch_idx} with config file epochs setting")
        
        # Create log file
        log_file_path = os.path.join(batch_dir, f"training_log_batch_{batch_idx}.txt")
        
        # Run training
        training_metrics = {'psnr': [], 'loss': [], 'final_psnr': 0}
        
        try:
            with open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(hinerv_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=os.environ.copy()
                )
                
                # Monitor training progress
                for line in process.stdout:
                    line = line.strip()
                    log_file.write(line + '\n')
                    log_file.flush()
                    
                    # Extract metrics from training output
                    if 'psnr' in line.lower() and 'epoch' in line.lower():
                        try:
                            # Try to extract PSNR value
                            import re
                            psnr_match = re.search(r'psnr[:\s]+([0-9.]+)', line.lower())
                            if psnr_match:
                                psnr_val = float(psnr_match.group(1))
                                training_metrics['psnr'].append(psnr_val)
                                training_metrics['final_psnr'] = psnr_val
                        except:
                            pass
                    
                    # Log important lines
                    if any(keyword in line.lower() for keyword in ['epoch', 'psnr', 'loss']):
                        logger.info(f"Batch {batch_idx}: {line}")
                
                process.wait()
                
                if process.returncode != 0:
                    raise RuntimeError(f"Training failed with return code {process.returncode}")
                
                logger.info(f"Batch {batch_idx} training completed successfully")
                return training_metrics
                
        except Exception as e:
            logger.error(f"Training batch {batch_idx} failed: {str(e)}")
            raise RuntimeError(f"Training batch {batch_idx} failed: {str(e)}")
    
    def extract_compressed_frames(self, batch_dir, output_dir, expected_frames):
        """Extract compressed frames from HiNeRV's eval_output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Find HiNeRV output directory
        model_dir = os.path.join(batch_dir, "model")
        hinerv_output_dirs = []
        
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path) and ("HiNeRV" in item or "hinerv" in item.lower()):
                    hinerv_output_dirs.append(item_path)
        
        if not hinerv_output_dirs:
            logger.warning(f"No HiNeRV output directory found in {model_dir}")
            return self.fallback_extract_frames(batch_dir, output_dir, expected_frames)
        
        # Use the most recent HiNeRV output directory
        hinerv_output_dir = max(hinerv_output_dirs, key=os.path.getctime)
        logger.info(f"Using HiNeRV output: {os.path.basename(hinerv_output_dir)}")
        
        # Look for eval_output directory
        eval_output_dir = os.path.join(hinerv_output_dir, "eval_output")
        
        if not os.path.exists(eval_output_dir):
            logger.warning(f"No eval_output directory found")
            return self.fallback_extract_frames(batch_dir, output_dir, expected_frames)
        
        # Find the best compressed frames
        compressed_frames = self.find_best_compressed_frames(eval_output_dir)
        
        if not compressed_frames:
            logger.warning("No compressed frames found in eval_output")
            return self.fallback_extract_frames(batch_dir, output_dir, expected_frames)
        
        # Copy compressed frames to output directory
        copied_count = self.copy_compressed_frames(compressed_frames, output_dir)
        
        # Calculate compression metrics
        compression_metrics = self.calculate_batch_compression_metrics(
            os.path.join(batch_dir, "frames"), 
            output_dir
        )
        
        logger.info(f"Successfully extracted {copied_count} compressed frames")
        return compression_metrics
    
    def find_best_compressed_frames(self, eval_output_dir):
        """Find the best quality compressed frames from HiNeRV output"""
        potential_dirs = []
        
        # List all directories in eval_output
        for item in os.listdir(eval_output_dir):
            item_path = os.path.join(eval_output_dir, item)
            if os.path.isdir(item_path):
                potential_dirs.append((item, item_path))
        
        if not potential_dirs:
            return []
        
        # Sort by preference: highest epoch number, then highest quality
        def sort_key(dir_info):
            dir_name, _ = dir_info
            
            # Extract epoch number
            epoch_num = 0
            parts = dir_name.split('_')
            try:
                if len(parts) > 1:
                    epoch_num = int(parts[0])
                else:
                    epoch_num = int(dir_name)
            except ValueError:
                epoch_num = 0
            
            # Extract quality (q8 > q7 > q6, no q = medium priority)
            quality_score = 1.5  # Default
            if 'q8' in dir_name:
                quality_score = 3
            elif 'q7' in dir_name:
                quality_score = 2
            elif 'q6' in dir_name:
                quality_score = 1
            
            return (epoch_num, quality_score)
        
        # Sort by epoch and quality (highest first)
        potential_dirs.sort(key=sort_key, reverse=True)
        
        # Try each directory until we find one with frames
        for dir_name, dir_path in potential_dirs:
            frame_files = glob.glob(os.path.join(dir_path, "*.png"))
            if frame_files:
                logger.info(f"Using compressed frames from: {dir_name} ({len(frame_files)} frames)")
                return sorted(frame_files)
        
        return []
    
    def copy_compressed_frames(self, source_frames, output_dir):
        """Copy compressed frames to output directory with sequential numbering"""
        copied_count = 0
        
        for i, source_frame in enumerate(source_frames):
            output_filename = f"{i+1:06d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            shutil.copy2(source_frame, output_path)
            copied_count += 1
        
        return copied_count
    
    def fallback_extract_frames(self, batch_dir, output_dir, expected_frames):
        """Fallback: copy original frames when HiNeRV output not found"""
        logger.warning("Using fallback: copying original frames")
        
        batch_frames_dir = os.path.join(batch_dir, "frames")
        frame_files = sorted(glob.glob(os.path.join(batch_frames_dir, "*.png")))
        
        copied_count = 0
        for i, frame_file in enumerate(frame_files):
            if i >= expected_frames:
                break
            output_file = os.path.join(output_dir, f"{i+1:06d}.png")
            shutil.copy2(frame_file, output_file)
            copied_count += 1
        
        # Return basic metrics
        return {
            'compression_ratio': 1.0,  # No compression
            'space_saved': 0.0,
            'avg_frame_size_original': 0,
            'avg_frame_size_compressed': 0,
            'fallback_used': True
        }
    
    def calculate_batch_compression_metrics(self, original_frames_dir, compressed_frames_dir):
        """Calculate compression metrics for a batch"""
        try:
            original_frames = glob.glob(os.path.join(original_frames_dir, "*.png"))
            compressed_frames = glob.glob(os.path.join(compressed_frames_dir, "*.png"))
            
            if not original_frames or not compressed_frames:
                return {'compression_ratio': 1.0, 'space_saved': 0.0, 'error': 'No frames found'}
            
            # Calculate total sizes
            original_size = sum(os.path.getsize(f) for f in original_frames)
            compressed_size = sum(os.path.getsize(f) for f in compressed_frames)
            
            # Calculate metrics
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            space_saved = 1 - (compressed_size / original_size) if original_size > 0 else 0.0
            
            return {
                'compression_ratio': compression_ratio,
                'space_saved': space_saved,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'avg_frame_size_original': original_size / len(original_frames),
                'avg_frame_size_compressed': compressed_size / len(compressed_frames),
                'fallback_used': False
            }
            
        except Exception as e:
            logger.error(f"Error calculating batch metrics: {e}")
            return {'compression_ratio': 1.0, 'space_saved': 0.0, 'error': str(e)}
    
    def copy_batch_output(self, batch_output_dir, main_output_dir, start_frame_idx):
        """Copy batch output frames to main output directory with correct numbering"""
        batch_files = sorted(glob.glob(os.path.join(batch_output_dir, "*.png")))
        
        for i, batch_file in enumerate(batch_files):
            final_frame_num = start_frame_idx + i + 1
            final_output_file = os.path.join(main_output_dir, f"{final_frame_num:06d}.png")
            shutil.copy2(batch_file, final_output_file)
        
        logger.info(f"Copied {len(batch_files)} frames to main output starting from frame {start_frame_idx + 1}")
    
    def cleanup_batch_directory(self, batch_dir):
        """Clean up batch directory to save space"""
        try:
            # Remove frames directories to save space
            frames_dir = os.path.join(batch_dir, "frames")
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            
            output_frames_dir = os.path.join(batch_dir, "output_frames")
            if os.path.exists(output_frames_dir):
                shutil.rmtree(output_frames_dir)
            
            logger.debug(f"Cleaned up batch directory: {batch_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup batch directory {batch_dir}: {e}")
    
    def create_final_video(self, compressed_frames_dir):
        """Create final compressed video from all compressed frames"""
        compressed_path = os.path.join(self.output_dir, "compressed.mp4")
        
        # Get original video properties
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Check frame count
        frame_files = glob.glob(os.path.join(compressed_frames_dir, "*.png"))
        logger.info(f"Creating final video from {len(frame_files)} compressed frames")
        
        if len(frame_files) == 0:
            raise RuntimeError("No compressed frames found to create video")
        
        # Create video with lossless encoding (frames already compressed by HiNeRV)
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", os.path.join(compressed_frames_dir, "%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "0",  # Lossless since frames are already compressed
            "-preset", "fast",
            compressed_path
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Created final compressed video: {compressed_path}")
            
            final_size = os.path.getsize(compressed_path)
            logger.info(f"Final video size: {final_size / (1024*1024):.2f} MB")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error creating final video: {e.stderr}")
            raise RuntimeError(f"Failed to create final video: {e.stderr}")
        
        return compressed_path
    
    def calculate_comprehensive_results(self, compressed_path, batch_metrics):
        """Calculate comprehensive compression results with detailed metrics"""
        try:
            original_size = os.path.getsize(self.video_path)
            compressed_size = os.path.getsize(compressed_path)
            
            # Aggregate batch metrics
            total_batches = len(batch_metrics)
            avg_compression_ratio = 0
            avg_space_saved = 0
            total_frames_processed = 0
            avg_psnr = 0
            psnr_values = []
            
            fallback_count = 0
            
            for batch in batch_metrics:
                if batch and 'compression_metrics' in batch:
                    metrics = batch['compression_metrics']
                    if 'compression_ratio' in metrics:
                        avg_compression_ratio += metrics['compression_ratio']
                    if 'space_saved' in metrics:
                        avg_space_saved += metrics['space_saved']
                    if 'fallback_used' in metrics and metrics['fallback_used']:
                        fallback_count += 1
                
                total_frames_processed += batch.get('frame_count', 0)
                
                if batch and 'training_metrics' in batch:
                    training = batch['training_metrics']
                    if 'final_psnr' in training and training['final_psnr'] > 0:
                        psnr_values.append(training['final_psnr'])
            
            # Calculate averages
            if total_batches > 0:
                avg_compression_ratio /= total_batches
                avg_space_saved /= total_batches
            
            if psnr_values:
                avg_psnr = sum(psnr_values) / len(psnr_values)
            
            # Comprehensive results
            results = {
                'original_path': self.video_path,
                'compressed_path': compressed_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'video_compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
                'video_space_saved': 1 - (compressed_size / original_size) if original_size > 0 else 0,
                
                # Batch processing metrics
                'total_batches': total_batches,
                'batch_size': self.batch_size,
                'total_frames_processed': total_frames_processed,
                'fallback_batches': fallback_count,
                
                # Neural compression metrics
                'avg_frame_compression_ratio': avg_compression_ratio,
                'avg_frame_space_saved': avg_space_saved,
                'avg_psnr': avg_psnr,
                'psnr_values': psnr_values,
                
                # Detailed batch metrics
                'batch_metrics': batch_metrics,
                
                # Performance info
                'compression_method': 'HiNeRV Neural Compression',
                'processing_successful': fallback_count < total_batches
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive results: {e}")
            # Return basic results as fallback
            return {
                'original_path': self.video_path,
                'compressed_path': compressed_path,
                'original_size': os.path.getsize(self.video_path),
                'compressed_size': os.path.getsize(compressed_path),
                'video_compression_ratio': 1.0,
                'video_space_saved': 0.0,
                'error': str(e)
            }
    
    def stop(self):
        """Stop the processing"""
        self.is_running = False
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
        
        # Video preview - larger when upload section is hidden
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 280)  # Increased from 300x200
        self.preview_label.setMaximumSize(600, 400)  # Increased from 450x300
        self.preview_label.setScaledContents(True)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 6px;
            }
        """)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)
        
        # Video info - readable size
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        info_layout = QGridLayout(info_frame)
        info_layout.setSpacing(3)
        
        self.info_labels = {
            'resolution': QLabel("Resolution: --"),
            'fps': QLabel("FPS: --"),
            'size': QLabel("Size: --"),
            'duration': QLabel("Duration: --")
        }
        
        row = 0
        for key, label in self.info_labels.items():
            label.setStyleSheet("color: #ccc; padding: 2px; font-size: 12px;")
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
        self.precompressed_mode = False  
        self.upload_widget = None  
        self.setup_ui()
        self.apply_theme()
        self.setup_shortcuts() 
        
    def setup_ui(self):
        """Set up the user interface optimized for laptop screens"""
        self.setWindowTitle("HiNeRV Video Compressor - Batch Processing")
        self.setMinimumSize(1000, 600)  # Reduced from 1200x700
        self.setMaximumSize(1400, 800)  # Add maximum size for laptop compatibility
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)  # Reduced spacing
        main_layout.setContentsMargins(15, 15, 15, 15)  # Reduced margins
        
        # Header - more compact
        header = QLabel("HiNeRV Video Compressor")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            padding: 8px;
        """)
        main_layout.addWidget(header)
        
        # Batch info - more compact
        batch_info = QLabel("Processing in batches of 40 frames • Ctrl+Shift+L for pre-compressed videos")
        batch_info.setAlignment(Qt.AlignCenter)
        batch_info.setStyleSheet("""
            font-size: 11px;
            color: #888;
            padding: 3px;
        """)
        main_layout.addWidget(batch_info)
        
        # Upload section - store reference for hiding
        self.upload_widget = self.create_upload_section()
        main_layout.addWidget(self.upload_widget)
        
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
    def setup_shortcuts(self):
        """Set up keyboard shortcuts for hidden features"""
        # Hidden shortcut: Ctrl+Shift+L to load pre-compressed video
        self.load_compressed_shortcut = QShortcut(QKeySequence("Ctrl+Shift+L"), self)
        self.load_compressed_shortcut.activated.connect(self.load_precompressed_video)
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
    def load_precompressed_video(self):
        """Hidden feature: Load pre-compressed video with existing metrics"""
        if not self.video_path:
            QMessageBox.warning(self, "No Original Video", "Please load an original video first.")
            return
        
        if self.processor and self.processor.isRunning():
            QMessageBox.warning(self, "Processing Active", "Cannot load pre-compressed video while processing.")
            return
        
        # Ask for compressed video
        compressed_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pre-Compressed Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*.*)"
        )
        
        if not compressed_path:
            return
        
        # Ask for metrics JSON file
        metrics_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Metrics JSON File",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not metrics_path:
            return
        
        try:
            # Load metrics from JSON
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Update metrics with current paths
            metrics['original_path'] = self.video_path
            metrics['compressed_path'] = compressed_path
            
            # Recalculate file sizes
            metrics['original_size'] = os.path.getsize(self.video_path)
            metrics['compressed_size'] = os.path.getsize(compressed_path)
            metrics['video_compression_ratio'] = metrics['original_size'] / metrics['compressed_size']
            metrics['video_space_saved'] = 1 - (metrics['compressed_size'] / metrics['original_size'])
            
            # Set precompressed mode
            self.precompressed_mode = True
            
            # Show results as if compression just finished
            self.on_compression_finished(metrics)
            
            QMessageBox.information(
                self, 
                "Pre-compressed Video Loaded", 
                "Successfully loaded pre-compressed video with metrics!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Loading", 
                f"Failed to load pre-compressed video or metrics:\n{str(e)}"
            )
    def create_comparison_section(self):
        """Create video comparison section with enhanced layout"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Video previews with improved sizing for full screen usage
        previews_layout = QHBoxLayout()
        
        self.original_preview = VideoPreviewWidget("Original Video")
        self.compressed_preview = VideoPreviewWidget("Compressed Video")
        
        # Larger sizes when upload section is hidden
        for preview in [self.original_preview, self.compressed_preview]:
            preview.setMaximumHeight(450)  # Increased when upload is hidden
            preview.setMinimumHeight(350)  # Increased minimum
        
        previews_layout.addWidget(self.original_preview)
        previews_layout.addWidget(self.compressed_preview)
        
        layout.addLayout(previews_layout)
        
        # Training info - more compact
        info_widget = QGroupBox("Processing Info")
        info_layout = QVBoxLayout(info_widget)
        
        info_text = QLabel("• 40 frames/batch • Config file epochs • Independent batch training\n• Shortcut: Ctrl+Shift+L to load pre-compressed video")
        info_text.setStyleSheet("color: #888; font-size: 11px; padding: 5px;")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_widget)
        
        # Compress button
        self.compress_btn = QPushButton("Start HiNeRV Batch Compression")
        self.compress_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.compress_btn.clicked.connect(self.start_compression)
        layout.addWidget(self.compress_btn, alignment=Qt.AlignCenter)
        
        # Stop button
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_compression)
        self.stop_btn.setVisible(False)
        layout.addWidget(self.stop_btn, alignment=Qt.AlignCenter)
        
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
        """Create enhanced results section optimized for laptop screens"""
        widget = QGroupBox("HiNeRV Compression Results")
        layout = QVBoxLayout(widget)
        
        # Create scrollable area for better laptop compatibility
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(400)  # Limit height for laptop screens
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #444;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #666;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777;
            }
        """)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Main results in compact grid
        main_results_frame = QFrame()
        main_results_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        main_results_layout = QGridLayout(main_results_frame)
        main_results_layout.setSpacing(5)  # Reduced spacing
        
        # Video-level metrics with compact labels
        self.result_labels = {
            'original_size': QLabel("Original: --"),
            'compressed_size': QLabel("Compressed: --"),
            'video_compression_ratio': QLabel("Ratio: --"),
            'video_space_saved': QLabel("Saved: --"),
            'avg_psnr': QLabel("Avg PSNR: --"),
            'total_batches': QLabel("Batches: --")
        }
        
        row = 0
        for key, label in self.result_labels.items():
            label.setStyleSheet("font-size: 12px; color: #ccc; padding: 2px;")
            main_results_layout.addWidget(label, row // 2, row % 2)
            row += 1
        
        scroll_layout.addWidget(main_results_frame)
        
        # Performance indicators - horizontal layout for space efficiency
        perf_frame = QFrame()
        perf_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 6px;
                padding: 8px;
                margin-top: 5px;
            }
        """)
        perf_layout = QHBoxLayout(perf_frame)
        perf_layout.setSpacing(10)
        
        self.performance_labels = {
            'neural_success': QLabel("Neural: --"),
            'fallback_count': QLabel("Fallback: --"),
            'avg_frame_compression': QLabel("Frame Ratio: --")
        }
        
        for label in self.performance_labels.values():
            label.setStyleSheet("font-size: 11px; color: #ccc; padding: 2px;")
            perf_layout.addWidget(label)
        
        scroll_layout.addWidget(perf_frame)
        
        # Compact batch details
        detailed_frame = QFrame()
        detailed_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 6px;
                padding: 8px;
                margin-top: 5px;
            }
        """)
        detailed_layout = QVBoxLayout(detailed_frame)
        
        detailed_title = QLabel("Batch Analysis")
        detailed_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #4CAF50; padding: 2px;")
        detailed_layout.addWidget(detailed_title)
        
        self.batch_details_text = QTextEdit()
        self.batch_details_text.setMaximumHeight(100)  # More compact
        self.batch_details_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ccc;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        self.batch_details_text.setReadOnly(True)
        detailed_layout.addWidget(self.batch_details_text)
        
        scroll_layout.addWidget(detailed_frame)
        
        # Set scroll area content
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Action buttons - more compact layout
        actions_frame = QFrame()
        actions_layout = QGridLayout(actions_frame)
        actions_layout.setSpacing(5)
        
        self.play_original_btn = QPushButton("Play Original")
        self.play_compressed_btn = QPushButton("Play Compressed")
        self.save_btn = QPushButton("Save Video")
        self.export_metrics_btn = QPushButton("Export Metrics")
        
        buttons = [self.play_original_btn, self.play_compressed_btn, self.save_btn, self.export_metrics_btn]
        
        for i, btn in enumerate(buttons):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #555;
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #666;
                }
            """)
            actions_layout.addWidget(btn, i // 2, i % 2)
        
        self.play_original_btn.clicked.connect(self.play_original)
        self.play_compressed_btn.clicked.connect(self.play_compressed)
        self.save_btn.clicked.connect(self.save_compressed)
        self.export_metrics_btn.clicked.connect(self.export_metrics)
        
        layout.addWidget(actions_frame)
        
        # New compression button
        new_btn = QPushButton("Compress Another Video")
        new_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
                margin-top: 5px;
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
        """Start the compression process using config file epochs"""
        if not self.video_path:
            return
        
        # Check if we're in precompressed mode
        if self.precompressed_mode:
            QMessageBox.information(
                self, 
                "Pre-compressed Mode", 
                "This video was loaded with pre-existing compression results.\nUse 'Compress Another Video' to start fresh."
            )
            return
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(self.video_path).stem
        
        gui_dir = Path(__file__).parent
        self.output_dir = gui_dir / "output" / f"{video_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update UI
        self.compress_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self.progress_widget.setVisible(True)
        self.results_widget.setVisible(False)
        
        # Start compression thread
        self.processor = VideoProcessor(str(self.video_path), str(self.output_dir), batch_size=40)
        self.processor.progress.connect(self.update_progress)
        self.processor.finished.connect(self.on_compression_finished)
        self.processor.error.connect(self.on_compression_error)
        self.processor.start()


    def stop_compression(self):
        """Stop the compression process"""
        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait(5000)  # Wait up to 5 seconds
            
        # Reset UI
        self.compress_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.progress_widget.setVisible(False)
        
        QMessageBox.information(self, "Stopped", "Compression process has been stopped.")
    
    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def on_compression_finished(self, results):
        """Handle compression completion with comprehensive results"""
        # Update UI
        self.compress_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.progress_widget.setVisible(False)
        self.results_widget.setVisible(True)
        
        # Store results
        self.compression_results = results
        
        # Update compressed preview
        if 'compressed_path' in results:
            self.compressed_preview.load_video(results['compressed_path'])
        
        # Update main results
        self.result_labels['original_size'].setText(
            f"Original Size: {self.format_size(results.get('original_size', 0))}"
        )
        self.result_labels['compressed_size'].setText(
            f"Compressed Size: {self.format_size(results.get('compressed_size', 0))}"
        )
        self.result_labels['video_compression_ratio'].setText(
            f"Video Compression Ratio: {results.get('video_compression_ratio', 0):.2f}x"
        )
        self.result_labels['video_space_saved'].setText(
            f"Video Space Saved: {results.get('video_space_saved', 0)*100:.1f}%"
        )
        
        # PSNR
        avg_psnr = results.get('avg_psnr', 0)
        if avg_psnr > 0:
            self.result_labels['avg_psnr'].setText(f"Average PSNR: {avg_psnr:.2f} dB")
        else:
            self.result_labels['avg_psnr'].setText("Average PSNR: Not available")
        
        # Batch info
        total_batches = results.get('total_batches', 0)
        self.result_labels['total_batches'].setText(
            f"Batches Processed: {total_batches} (config file epochs)"
        )
        
        # Performance indicators
        fallback_count = results.get('fallback_batches', 0)
        successful_neural = total_batches - fallback_count
        
        if successful_neural > 0:
            self.performance_labels['neural_success'].setText(f"Neural Compression: ✓ {successful_neural}/{total_batches} batches")
            self.performance_labels['neural_success'].setStyleSheet("font-size: 12px; color: #4CAF50; padding: 3px;")
        else:
            self.performance_labels['neural_success'].setText(f"Neural Compression: ✗ Failed")
            self.performance_labels['neural_success'].setStyleSheet("font-size: 12px; color: #f44336; padding: 3px;")
        
        self.performance_labels['fallback_count'].setText(f"Fallback Batches: {fallback_count}")
        
        avg_frame_compression = results.get('avg_frame_compression_ratio', 1.0)
        self.performance_labels['avg_frame_compression'].setText(f"Avg Frame Compression: {avg_frame_compression:.2f}x")
        
        # Update detailed batch analysis
        self.update_batch_details(results)
        
        # Show completion message
        success_rate = (successful_neural / total_batches * 100) if total_batches > 0 else 0
        
        QMessageBox.information(
            self, 
            "HiNeRV Compression Complete", 
            f"Video compression completed!\n\n"
            f"• Processed {results.get('total_frames_processed', 0)} frames in {total_batches} batches\n"
            f"• Neural compression success: {success_rate:.1f}%\n"
            f"• Overall compression ratio: {results.get('video_compression_ratio', 1):.2f}x\n"
            f"• Space saved: {results.get('video_space_saved', 0)*100:.1f}%"
        )

    def on_compression_error(self, error_msg):
        """Handle compression error"""
        self.compress_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.progress_widget.setVisible(False)
        
        # Check if there's a log file we can point to
        log_file_hint = ""
        if self.output_dir:
            # Look for any batch log files
            log_files = glob.glob(os.path.join(self.output_dir, "**/training_log_batch_*.txt"), recursive=True)
            if log_files:
                log_file_hint = f"\n\nDetailed logs saved to:\n" + "\n".join(log_files[:3])
                if len(log_files) > 3:
                    log_file_hint += f"\n... and {len(log_files) - 3} more"
        
        # Create detailed error dialog
        error_dialog = QMessageBox(self)
        error_dialog.setWindowTitle("Compression Error")
        error_dialog.setText("An error occurred during batch compression:")
        error_dialog.setDetailedText(error_msg + log_file_hint)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.exec()
    
    def play_original(self):
        """Play original video"""
        if self.video_path and os.path.exists(self.video_path):
            if sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{self.video_path}"')
            elif sys.platform.startswith('win'):  # Windows
                os.system(f'start "" "{self.video_path}"')
            else:  # Linux
                os.system(f'xdg-open "{self.video_path}"')
    
    def play_compressed(self):
        """Play compressed video"""
        if hasattr(self, 'compression_results'):
            compressed_path = self.compression_results.get('compressed_path')
            if compressed_path and os.path.exists(compressed_path):
                if sys.platform.startswith('darwin'):  # macOS
                    os.system(f'open "{compressed_path}"')
                elif sys.platform.startswith('win'):  # Windows
                    os.system(f'start "" "{compressed_path}"')
                else:  # Linux
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
        self.precompressed_mode = False  # NEW: Reset precompressed mode
        
        # Stop any running processor
        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait(5000)
        
        # Show upload section again
        if self.upload_widget:
            self.upload_widget.setVisible(True)
        
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
        
        # Reset button states
        self.compress_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        
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
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self, 
                'Exit Application', 
                'Compression is in progress. Are you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processor.stop()
                self.processor.wait(5000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def update_batch_details(self, results):
        """Update the detailed batch analysis text"""
        details_text = "Batch-by-Batch Analysis:\n"
        details_text += "=" * 50 + "\n"
        
        batch_metrics = results.get('batch_metrics', [])
        
        for batch in batch_metrics:
            if not batch:
                continue
                
            batch_idx = batch.get('batch_idx', 0)
            frame_range = batch.get('frame_range', '--')
            frame_count = batch.get('frame_count', 0)
            
            details_text += f"Batch {batch_idx + 1}: Frames {frame_range} ({frame_count} frames)\n"
            
            # Training metrics
            training = batch.get('training_metrics', {})
            if 'final_psnr' in training and training['final_psnr'] > 0:
                details_text += f"  Training PSNR: {training['final_psnr']:.2f} dB\n"
            
            # Compression metrics
            compression = batch.get('compression_metrics', {})
            if 'compression_ratio' in compression:
                ratio = compression['compression_ratio']
                space_saved = compression.get('space_saved', 0) * 100
                fallback = compression.get('fallback_used', False)
                
                if fallback:
                    details_text += f"  Compression: FALLBACK (no neural compression)\n"
                else:
                    details_text += f"  Compression: {ratio:.2f}x ratio, {space_saved:.1f}% saved\n"
            
            details_text += "\n"
        
        # Summary
        total_batches = len(batch_metrics)
        fallback_count = results.get('fallback_batches', 0)
        
        details_text += f"Summary: {total_batches - fallback_count}/{total_batches} batches used neural compression\n"
        
        if results.get('avg_psnr', 0) > 0:
            details_text += f"Average PSNR across all batches: {results['avg_psnr']:.2f} dB\n"
        
        self.batch_details_text.setPlainText(details_text)

    def export_metrics(self):
        """Export detailed metrics to a file"""
        if not hasattr(self, 'compression_results'):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metrics",
            f"hinerv_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.compression_results, f, indent=2, default=str)
                else:
                    # Export as readable text
                    with open(file_path, 'w') as f:
                        f.write("HiNeRV Compression Results\n")
                        f.write("=" * 50 + "\n\n")
                        
                        results = self.compression_results
                        
                        f.write(f"Original Video: {results.get('original_path', 'N/A')}\n")
                        f.write(f"Compressed Video: {results.get('compressed_path', 'N/A')}\n\n")
                        
                        f.write("Video Metrics:\n")
                        f.write(f"  Original Size: {self.format_size(results.get('original_size', 0))}\n")
                        f.write(f"  Compressed Size: {self.format_size(results.get('compressed_size', 0))}\n")
                        f.write(f"  Compression Ratio: {results.get('video_compression_ratio', 0):.2f}x\n")
                        f.write(f"  Space Saved: {results.get('video_space_saved', 0)*100:.1f}%\n\n")
                        
                        f.write("Processing Details:\n")
                        f.write(f"  Total Batches: {results.get('total_batches', 0)}\n")
                        f.write(f"  Batch Size: {results.get('batch_size', 0)} frames\n")
                        f.write(f"  Epochs: From configuration files\n")
                        f.write(f"  Total Frames: {results.get('total_frames_processed', 0)}\n")
                        f.write(f"  Neural Compression Success: {results.get('total_batches', 0) - results.get('fallback_batches', 0)}/{results.get('total_batches', 0)} batches\n\n")
                        
                        if results.get('avg_psnr', 0) > 0:
                            f.write(f"Average PSNR: {results['avg_psnr']:.2f} dB\n\n")
                        
                        # Batch details
                        f.write("Batch Details:\n")
                        batch_metrics = results.get('batch_metrics', [])
                        for batch in batch_metrics:
                            if batch:
                                f.write(f"  Batch {batch.get('batch_idx', 0) + 1}: ")
                                f.write(f"Frames {batch.get('frame_range', '--')} ")
                                
                                compression = batch.get('compression_metrics', {})
                                if compression.get('fallback_used', False):
                                    f.write("(FALLBACK)\n")
                                else:
                                    ratio = compression.get('compression_ratio', 1.0)
                                    f.write(f"({ratio:.2f}x compression)\n")
                
                QMessageBox.information(self, "Export Complete", f"Metrics exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export metrics:\n{str(e)}")

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("HiNeRV Video Compressor - Batch Processing")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()