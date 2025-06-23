#!/usr/bin/env python3

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
import webbrowser
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QTextEdit, QScrollArea
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QFrame,
    QGroupBox, QGridLayout, QMessageBox, QSplitter, QSpinBox,
    QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor, QDragEnterEvent, QDropEvent

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoInfo:
    def __init__(self, path=""):
        self.path = path
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.duration = 0
        self.size_bytes = 0
        self.size_str = ""


class MetricsChart(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color: #1e1e1e;")
        
    def plot_compression_metrics(self, results):
        self.fig.clear()
        
        batch_metrics = results.get('batch_metrics', [])
        if not batch_metrics:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No batch metrics available', 
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)
            ax.set_facecolor('#2b2b2b')
            self.draw()
            return
        
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        batch_indices = []
        compression_ratios = []
        psnr_values = []
        space_saved_percentages = []
        
        for batch in batch_metrics:
            if batch and 'compression_metrics' in batch:
                batch_indices.append(batch.get('batch_idx', 0) + 1)
                
                compression = batch['compression_metrics']
                compression_ratios.append(compression.get('compression_ratio', 1.0))
                space_saved_percentages.append(compression.get('space_saved', 0) * 100)
                
                training = batch.get('training_metrics', {})
                psnr_values.append(training.get('final_psnr', 0))
        
        ax1 = self.fig.add_subplot(gs[0, 0])
        if compression_ratios:
            ax1.bar(batch_indices, compression_ratios, color='#4CAF50', alpha=0.8)
            ax1.set_title('Compression Ratio by Batch', color='white', fontsize=10)
            ax1.set_ylabel('Compression Ratio', color='white', fontsize=9)
            ax1.tick_params(colors='white', labelsize=8)
        ax1.set_facecolor('#2b2b2b')
        
        ax2 = self.fig.add_subplot(gs[0, 1])
        if psnr_values and any(p > 0 for p in psnr_values):
            valid_psnr = [p for p in psnr_values if p > 0]
            valid_indices = [batch_indices[i] for i, p in enumerate(psnr_values) if p > 0]
            ax2.plot(valid_indices, valid_psnr, 'o-', color='#2196F3', linewidth=2, markersize=4)
            ax2.set_title('PSNR by Batch', color='white', fontsize=10)
            ax2.set_ylabel('PSNR (dB)', color='white', fontsize=9)
            ax2.tick_params(colors='white', labelsize=8)
        ax2.set_facecolor('#2b2b2b')
        
        ax3 = self.fig.add_subplot(gs[1, 0])
        if space_saved_percentages:
            ax3.bar(batch_indices, space_saved_percentages, color='#FF9800', alpha=0.8)
            ax3.set_title('Space Saved by Batch', color='white', fontsize=10)
            ax3.set_ylabel('Space Saved (%)', color='white', fontsize=9)
            ax3.set_xlabel('Batch Number', color='white', fontsize=9)
            ax3.tick_params(colors='white', labelsize=8)
        ax3.set_facecolor('#2b2b2b')
        
        ax4 = self.fig.add_subplot(gs[1, 1])
        overall_ratio = results.get('video_compression_ratio', 1.0)
        overall_saved = results.get('video_space_saved', 0) * 100
        
        categories = ['Original Size', 'Compressed Size']
        sizes = [results.get('original_size', 0), results.get('compressed_size', 0)]
        colors = ['#f44336', '#4CAF50']
        
        if sum(sizes) > 0:
            ax4.pie([sizes[0] - sizes[1], sizes[1]], labels=['Saved', 'Compressed'], 
                   colors=['#4CAF50', '#2196F3'], autopct='%1.1f%%', startangle=90,
                   textprops={'color': 'white', 'fontsize': 8})
            ax4.set_title('Overall Size Reduction', color='white', fontsize=10)
        ax4.set_facecolor('#2b2b2b')
        
        self.draw()


class VideoProcessor(QThread):
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, video_path, output_dir, batch_size=40):
        super().__init__()
        self.video_path = video_path
        self.output_dir = os.path.abspath(output_dir)
        self.batch_size = batch_size
        self.is_running = True
        
    def run(self):
        try:
            self.progress.emit(5, "Extracting video frames...")
            frames_dir = self.extract_frames()
            
            frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
            total_frames = len(frame_files)
            total_batches = math.ceil(total_frames / self.batch_size)
            
            self.progress.emit(10, f"Processing {total_frames} frames in {total_batches} batches...")
            logger.info(f"Total frames: {total_frames}, Batches: {total_batches}")
            
            output_frames_dir = os.path.join(self.output_dir, "compressed_frames")
            os.makedirs(output_frames_dir, exist_ok=True)
            
            batch_metrics = []
            
            for batch_idx in range(total_batches):
                if not self.is_running:
                    break
                    
                batch_progress_start = 10 + int((batch_idx / total_batches) * 70)
                batch_progress_end = 10 + int(((batch_idx + 1) / total_batches) * 70)
                
                self.progress.emit(
                    batch_progress_start, 
                    f"Processing batch {batch_idx + 1}/{total_batches}..."
                )
                
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
            
            self.progress.emit(85, "Creating final compressed video...")
            compressed_path = self.create_final_video(output_frames_dir)
            
            self.progress.emit(95, "Calculating compression metrics...")
            results = self.calculate_comprehensive_results(compressed_path, batch_metrics)
            
            self.progress.emit(100, "Compression complete!")
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Compression error: {str(e)}")
            self.error.emit(f"Compression failed: {str(e)}")
    
    def extract_frames(self):
        frames_dir = os.path.join(self.output_dir, "original_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-i", self.video_path,
            "-q:v", "0",
            os.path.join(frames_dir, "%06d.png")
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Frame extraction completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to extract frames: {e.stderr}")
        
        frame_files = glob.glob(os.path.join(frames_dir, "*.png"))
        logger.info(f"Extracted {len(frame_files)} frames")
        
        return frames_dir
    
    def process_batch(self, frames_dir, batch_idx, total_batches, output_frames_dir, progress_start, progress_end):
        try:
            all_frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_frame_files))
            batch_frame_files = all_frame_files[start_idx:end_idx]
            
            if not batch_frame_files:
                logger.warning(f"No frames found for batch {batch_idx}")
                return None
            
            actual_batch_size = len(batch_frame_files)
            logger.info(f"Processing batch {batch_idx + 1}: frames {start_idx + 1}-{end_idx}")
            
            batch_dir = os.path.join(self.output_dir, f"batch_{batch_idx:03d}")
            batch_frames_dir = os.path.join(batch_dir, "frames")
            os.makedirs(batch_frames_dir, exist_ok=True)
            
            for i, frame_file in enumerate(batch_frame_files):
                src = frame_file
                dst = os.path.join(batch_frames_dir, f"{i+1:06d}.png")
                shutil.copy2(src, dst)
            
            train_progress = progress_start + int((progress_end - progress_start) * 0.7)
            self.progress.emit(
                train_progress, 
                f"Training HiNeRV model on batch {batch_idx + 1}..."
            )
            
            training_metrics = self.train_model_batch(batch_frames_dir, batch_dir, batch_idx)
            
            gen_progress = progress_start + int((progress_end - progress_start) * 0.9)
            self.progress.emit(
                gen_progress,
                f"Extracting compressed frames from batch {batch_idx + 1}..."
            )
            
            batch_output_dir = os.path.join(batch_dir, "output_frames")
            compression_metrics = self.extract_compressed_frames(batch_dir, batch_output_dir, actual_batch_size)
            
            self.copy_batch_output(batch_output_dir, output_frames_dir, start_idx)
            
            batch_metrics = {
                'batch_idx': batch_idx,
                'frame_range': f"{start_idx + 1}-{end_idx}",
                'frame_count': actual_batch_size,
                'training_metrics': training_metrics,
                'compression_metrics': compression_metrics
            }
            
            self.cleanup_batch_directory(batch_dir)
            
            logger.info(f"Completed batch {batch_idx + 1}")
            return batch_metrics
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            raise RuntimeError(f"Batch {batch_idx + 1} failed: {str(e)}")
    
    def train_model_batch(self, batch_frames_dir, batch_dir, batch_idx):
        gui_dir = Path(__file__).parent
        hinerv_root = gui_dir.parent
        
        dataset_dir = os.path.dirname(batch_frames_dir)
        dataset_name = os.path.basename(batch_frames_dir)
        model_output = os.path.join(batch_dir, "model")
        os.makedirs(model_output, exist_ok=True)
        
        train_cfg_path = hinerv_root / "cfgs" / "train" / "hinerv_1920x1080.txt"
        model_cfg_path = hinerv_root / "cfgs" / "models" / "uvg-hinerv-s_1920x1080.txt"
        
        if not train_cfg_path.exists():
            raise FileNotFoundError(f"Training config not found: {train_cfg_path}")
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_cfg_path}")
        
        hinerv_main_path = hinerv_root / "hinerv_main.py"
        if not hinerv_main_path.exists():
            raise FileNotFoundError(f"hinerv_main.py not found: {hinerv_main_path}")
        
        with open(train_cfg_path, 'r') as f:
            train_cfg_content = f.read().strip()
        
        with open(model_cfg_path, 'r') as f:
            model_cfg_content = f.read().strip()
        
        cmd = [
            "accelerate", "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=inductor",
            str(hinerv_main_path),
            "--dataset", dataset_dir,
            "--dataset-name", dataset_name,
            "--output", model_output
        ]
        
        if train_cfg_content:
            cmd.extend(train_cfg_content.split())
        
        if model_cfg_content:
            cmd.extend(model_cfg_content.split())
        
        cmd.extend([
            "--batch-size", "1",
            "--eval-batch-size", "1",
            "--grad-accum", "1",
            "--log-eval", "true",
            "--seed", str(batch_idx)
        ])
        
        logger.info(f"Training batch {batch_idx}")
        
        log_file_path = os.path.join(batch_dir, f"training_log_batch_{batch_idx}.txt")
        
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
                
                for line in process.stdout:
                    line = line.strip()
                    log_file.write(line + '\n')
                    log_file.flush()
                    
                    if 'psnr' in line.lower() and 'epoch' in line.lower():
                        try:
                            import re
                            psnr_match = re.search(r'psnr[:\s]+([0-9.]+)', line.lower())
                            if psnr_match:
                                psnr_val = float(psnr_match.group(1))
                                training_metrics['psnr'].append(psnr_val)
                                training_metrics['final_psnr'] = psnr_val
                        except:
                            pass
                    
                    if any(keyword in line.lower() for keyword in ['epoch', 'psnr', 'loss']):
                        logger.info(f"Batch {batch_idx}: {line}")
                
                process.wait()
                
                if process.returncode != 0:
                    raise RuntimeError(f"Training failed with return code {process.returncode}")
                
                logger.info(f"Batch {batch_idx} training completed")
                return training_metrics
                
        except Exception as e:
            logger.error(f"Training batch {batch_idx} failed: {str(e)}")
            raise RuntimeError(f"Training batch {batch_idx} failed: {str(e)}")
    
    def extract_compressed_frames(self, batch_dir, output_dir, expected_frames):
        os.makedirs(output_dir, exist_ok=True)
        
        model_dir = os.path.join(batch_dir, "model")
        hinerv_output_dirs = []
        
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path) and ("HiNeRV" in item or "hinerv" in item.lower()):
                    hinerv_output_dirs.append(item_path)
        
        if not hinerv_output_dirs:
            logger.warning(f"No HiNeRV output directory found")
            return self.fallback_extract_frames(batch_dir, output_dir, expected_frames)
        
        hinerv_output_dir = max(hinerv_output_dirs, key=os.path.getctime)
        logger.info(f"Using HiNeRV output: {os.path.basename(hinerv_output_dir)}")
        
        eval_output_dir = os.path.join(hinerv_output_dir, "eval_output")
        
        if not os.path.exists(eval_output_dir):
            logger.warning(f"No eval_output directory found")
            return self.fallback_extract_frames(batch_dir, output_dir, expected_frames)
        
        compressed_frames = self.find_best_compressed_frames(eval_output_dir)
        
        if not compressed_frames:
            logger.warning("No compressed frames found")
            return self.fallback_extract_frames(batch_dir, output_dir, expected_frames)
        
        copied_count = self.copy_compressed_frames(compressed_frames, output_dir)
        
        compression_metrics = self.calculate_batch_compression_metrics(
            os.path.join(batch_dir, "frames"), 
            output_dir
        )
        
        logger.info(f"Successfully extracted {copied_count} compressed frames")
        return compression_metrics
    
    def find_best_compressed_frames(self, eval_output_dir):
        potential_dirs = []
        
        for item in os.listdir(eval_output_dir):
            item_path = os.path.join(eval_output_dir, item)
            if os.path.isdir(item_path):
                potential_dirs.append((item, item_path))
        
        if not potential_dirs:
            return []
        
        def sort_key(dir_info):
            dir_name, _ = dir_info
            
            epoch_num = 0
            parts = dir_name.split('_')
            try:
                if len(parts) > 1:
                    epoch_num = int(parts[0])
                else:
                    epoch_num = int(dir_name)
            except ValueError:
                epoch_num = 0
            
            quality_score = 1.5
            if 'q8' in dir_name:
                quality_score = 3
            elif 'q7' in dir_name:
                quality_score = 2
            elif 'q6' in dir_name:
                quality_score = 1
            
            return (epoch_num, quality_score)
        
        potential_dirs.sort(key=sort_key, reverse=True)
        
        for dir_name, dir_path in potential_dirs:
            frame_files = glob.glob(os.path.join(dir_path, "*.png"))
            if frame_files:
                logger.info(f"Using compressed frames from: {dir_name}")
                return sorted(frame_files)
        
        return []
    
    def copy_compressed_frames(self, source_frames, output_dir):
        copied_count = 0
        
        for i, source_frame in enumerate(source_frames):
            output_filename = f"{i+1:06d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            shutil.copy2(source_frame, output_path)
            copied_count += 1
        
        return copied_count
    
    def fallback_extract_frames(self, batch_dir, output_dir, expected_frames):
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
        
        return {
            'compression_ratio': 1.0,
            'space_saved': 0.0,
            'avg_frame_size_original': 0,
            'avg_frame_size_compressed': 0,
            'fallback_used': True
        }
    
    def calculate_batch_compression_metrics(self, original_frames_dir, compressed_frames_dir):
        try:
            original_frames = glob.glob(os.path.join(original_frames_dir, "*.png"))
            compressed_frames = glob.glob(os.path.join(compressed_frames_dir, "*.png"))
            
            if not original_frames or not compressed_frames:
                return {'compression_ratio': 1.0, 'space_saved': 0.0, 'error': 'No frames found'}
            
            original_size = sum(os.path.getsize(f) for f in original_frames)
            compressed_size = sum(os.path.getsize(f) for f in compressed_frames)
            
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
        batch_files = sorted(glob.glob(os.path.join(batch_output_dir, "*.png")))
        
        for i, batch_file in enumerate(batch_files):
            final_frame_num = start_frame_idx + i + 1
            final_output_file = os.path.join(main_output_dir, f"{final_frame_num:06d}.png")
            shutil.copy2(batch_file, final_output_file)
        
        logger.info(f"Copied {len(batch_files)} frames starting from frame {start_frame_idx + 1}")
    
    def cleanup_batch_directory(self, batch_dir):
        try:
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
        compressed_path = os.path.join(self.output_dir, "compressed.mp4")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        frame_files = glob.glob(os.path.join(compressed_frames_dir, "*.png"))
        logger.info(f"Creating final video from {len(frame_files)} compressed frames")
        
        if len(frame_files) == 0:
            raise RuntimeError("No compressed frames found")
        
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", os.path.join(compressed_frames_dir, "%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "0",
            "-preset", "fast",
            compressed_path
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Created final compressed video: {compressed_path}")
            
            final_size = os.path.getsize(compressed_path)
            logger.info(f"Final video size: {final_size / (1024*1024):.2f} MB")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to create final video: {e.stderr}")
        
        return compressed_path
    
    def calculate_comprehensive_results(self, compressed_path, batch_metrics):
        try:
            original_size = os.path.getsize(self.video_path)
            compressed_size = os.path.getsize(compressed_path)
            
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
            
            if total_batches > 0:
                avg_compression_ratio /= total_batches
                avg_space_saved /= total_batches
            
            if psnr_values:
                avg_psnr = sum(psnr_values) / len(psnr_values)
            
            results = {
                'original_path': self.video_path,
                'compressed_path': compressed_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'video_compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
                'video_space_saved': 1 - (compressed_size / original_size) if original_size > 0 else 0,
                'total_batches': total_batches,
                'batch_size': self.batch_size,
                'total_frames_processed': total_frames_processed,
                'fallback_batches': fallback_count,
                'avg_frame_compression_ratio': avg_compression_ratio,
                'avg_frame_space_saved': avg_space_saved,
                'avg_psnr': avg_psnr,
                'psnr_values': psnr_values,
                'batch_metrics': batch_metrics,
                'compression_method': 'HiNeRV Neural Compression',
                'processing_successful': fallback_count < total_batches
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
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
        self.is_running = False


class VideoLoadWorker(QThread):
    """Background worker for loading video information without blocking UI"""
    video_loaded = Signal(dict)  # Signal to emit video info
    error_occurred = Signal(str)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        
    def run(self):
        try:
            video_info = {}
            video_info['path'] = self.video_path
            
            # Load video properties using OpenCV in background thread
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                video_info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                video_info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_info['duration'] = video_info['frame_count'] / video_info['fps'] if video_info['fps'] > 0 else 0
                
                # Get a preview frame from the middle of the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_info['frame_count'] // 2)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    
                    from PySide6.QtGui import QImage
                    q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    video_info['preview_image'] = q_image
                
                cap.release()
            else:
                video_info['width'] = 0
                video_info['height'] = 0
                video_info['fps'] = 0
                video_info['frame_count'] = 0
                video_info['duration'] = 0
                video_info['preview_image'] = None
            
            # Get file size
            video_info['size_bytes'] = os.path.getsize(self.video_path)
            
            self.video_loaded.emit(video_info)
            
        except Exception as e:
            self.error_occurred.emit(f"Error loading video: {str(e)}")


class VideoPreviewWidget(QWidget):
    def __init__(self, title="Video"):
        super().__init__()
        self.title = title
        self.video_info = VideoInfo()
        self.load_worker = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 12px; 
            font-weight: bold; 
            padding: 6px;
            background-color: #333;
            border-radius: 3px;
            margin-bottom: 4px;
        """)
        layout.addWidget(title_label)
        
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 280)
        self.preview_label.setMaximumSize(600, 420)
        self.preview_label.setScaledContents(True)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 6px;
            }
        """)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Loading...")  # Show loading text initially
        layout.addWidget(self.preview_label)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 4px;
                padding: 8px;
                margin-top: 4px;
            }
        """)
        info_layout = QGridLayout(info_frame)
        info_layout.setSpacing(4)
        
        self.info_labels = {
            'resolution': QLabel("Resolution: --"),
            'fps': QLabel("FPS: --"),
            'size': QLabel("Size: --"),
            'duration': QLabel("Duration: --")
        }
        
        row = 0
        for key, label in self.info_labels.items():
            label.setStyleSheet("""
                color: #ddd; 
                padding: 3px; 
                font-size: 10px;
                font-weight: 500;
            """)
            info_layout.addWidget(label, row // 2, row % 2)
            row += 1
        
        layout.addWidget(info_frame)
    
    def load_video(self, video_path):
        """Load video in background thread to prevent UI freezing"""
        if not os.path.exists(video_path):
            self.preview_label.setText("Video file not found")
            return
        
        # Stop any existing worker
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.quit()
            self.load_worker.wait()
        
        # Show loading state
        self.preview_label.setText("Loading video...")
        for label in self.info_labels.values():
            label.setText(label.text().split(':')[0] + ": Loading...")
        
        # Start background loading
        self.load_worker = VideoLoadWorker(video_path)
        self.load_worker.video_loaded.connect(self.on_video_loaded)
        self.load_worker.error_occurred.connect(self.on_video_error)
        self.load_worker.start()
    
    def on_video_loaded(self, video_info):
        """Handle video loaded signal from background thread"""
        # Update video info object
        self.video_info.path = video_info['path']
        self.video_info.width = video_info['width']
        self.video_info.height = video_info['height']
        self.video_info.fps = video_info['fps']
        self.video_info.frame_count = video_info['frame_count']
        self.video_info.duration = video_info['duration']
        self.video_info.size_bytes = video_info['size_bytes']
        self.video_info.size_str = self.format_size(self.video_info.size_bytes)
        
        # Update preview image
        if video_info.get('preview_image'):
            pixmap = QPixmap.fromImage(video_info['preview_image'])
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
        else:
            self.preview_label.setText("No preview available")
        
        # Update info labels
        self.update_info()
    
    def on_video_error(self, error_msg):
        """Handle video loading error"""
        self.preview_label.setText(f"Error: {error_msg}")
        logger.error(f"Video loading error: {error_msg}")
        
        # Reset info labels
        for key, label in self.info_labels.items():
            label.setText(f"{key.title()}: Error")
    
    def update_info(self):
        self.info_labels['resolution'].setText(f"Resolution: {self.video_info.width}x{self.video_info.height}")
        self.info_labels['fps'].setText(f"FPS: {self.video_info.fps:.2f}")
        self.info_labels['size'].setText(f"Size: {self.video_info.size_str}")
        self.info_labels['duration'].setText(f"Duration: {self.format_duration(self.video_info.duration)}")
    
    def format_size(self, size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def format_duration(self, seconds):
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
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.output_dir = None
        self.processor = None
        self.precompressed_mode = False  
        self.upload_widget = None
        self.compression_results = None
        self.setup_ui()
        self.apply_theme()
        self.setup_shortcuts() 
        
    def setup_ui(self):
        self.setWindowTitle("HiNeRV Video Compressor")
        self.setMinimumSize(1200, 700)
        self.setMaximumSize(1600, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        self.upload_widget = self.create_upload_section()
        main_layout.addWidget(self.upload_widget)
        
        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #333;
                color: #ddd;
                padding: 8px 16px;
                margin-right: 1px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #555;
            }
        """)
        self.main_tabs.setVisible(False)
        
        # Video comparison tab
        self.comparison_tab = self.create_comparison_tab()
        self.main_tabs.addTab(self.comparison_tab, "📹 Video Comparison")
        
        # Results and charts tab
        self.results_tab = self.create_results_tab()
        self.main_tabs.addTab(self.results_tab, "📊 Results & Analysis")
        
        main_layout.addWidget(self.main_tabs)
        
        self.progress_widget = self.create_progress_section()
        self.progress_widget.setVisible(False)
        main_layout.addWidget(self.progress_widget)
        
        self.setAcceptDrops(True)
    
    def setup_shortcuts(self):
        self.load_compressed_shortcut = QShortcut(QKeySequence("Ctrl+Shift+L"), self)
        self.load_compressed_shortcut.activated.connect(self.load_precompressed_video)
    
    def create_upload_section(self):
        widget = QGroupBox("Upload Video")
        widget.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                padding-top: 12px;
            }
        """)
        layout = QVBoxLayout(widget)
        
        self.upload_area = QLabel("Drag and drop a video file here or click to browse")
        self.upload_area.setMinimumHeight(120)
        self.upload_area.setAlignment(Qt.AlignCenter)
        self.upload_area.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px dashed #666;
                border-radius: 8px;
                font-size: 14px;
                color: #bbb;
                font-weight: 500;
            }
            QLabel:hover {
                border-color: #4CAF50;
                color: #ddd;
                background-color: #2f2f2f;
            }
        """)
        self.upload_area.setCursor(Qt.PointingHandCursor)
        self.upload_area.mousePressEvent = self.browse_video
        layout.addWidget(self.upload_area)
        
        browse_btn = QPushButton("Browse Files")
        browse_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #45a049, stop:1 #3d8b40);
            }
        """)
        browse_btn.clicked.connect(self.browse_video)
        layout.addWidget(browse_btn, alignment=Qt.AlignCenter)
        
        shortcut_hint = QLabel("Advanced: Ctrl+Shift+L to load pre-compressed video")
        shortcut_hint.setAlignment(Qt.AlignCenter)
        shortcut_hint.setStyleSheet("""
            font-size: 10px;
            color: #888;
            padding: 5px;
            font-style: italic;
        """)
        layout.addWidget(shortcut_hint)
        
        return widget
    
    def create_comparison_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        # Video previews
        previews_layout = QHBoxLayout()
        previews_layout.setSpacing(10)
        
        self.original_preview = VideoPreviewWidget("Original Video")
        self.compressed_preview = VideoPreviewWidget("Compressed Video")
        
        previews_layout.addWidget(self.original_preview)
        previews_layout.addWidget(self.compressed_preview)
        
        layout.addLayout(previews_layout)
        
        # Control buttons
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setSpacing(10)
        
        self.compress_btn = QPushButton("Start Compression")
        self.compress_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #1976D2, stop:1 #1565C0);
            }
        """)
        self.compress_btn.clicked.connect(self.start_compression)
        controls_layout.addWidget(self.compress_btn)
        
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #f44336, stop:1 #d32f2f);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #d32f2f, stop:1 #c62828);
            }
        """)
        self.stop_btn.clicked.connect(self.stop_compression)
        self.stop_btn.setVisible(False)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        
        # Video control buttons
        self.play_original_btn = QPushButton("▶ Play Original")
        self.play_compressed_btn = QPushButton("▶ Play Compressed")
        
        video_controls = [self.play_original_btn, self.play_compressed_btn]
        for btn in video_controls:
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #555, stop:1 #444);
                    color: white;
                    border: 1px solid #666;
                    padding: 8px 15px;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #666, stop:1 #555);
                }
            """)
            controls_layout.addWidget(btn)
        
        self.play_original_btn.clicked.connect(self.play_original)
        self.play_compressed_btn.clicked.connect(self.play_compressed)
        
        layout.addWidget(controls_frame)
        
        return widget
    
    def create_results_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(10)
        
        # Left side - Metrics and overview
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        
        # Key metrics
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 6px;
                padding: 12px;
            }
        """)
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setSpacing(8)
        
        self.result_labels = {
            'original_size': QLabel("Original: --"),
            'compressed_size': QLabel("Compressed: --"),
            'video_compression_ratio': QLabel("Ratio: --"),
            'video_space_saved': QLabel("Saved: --"),
            'avg_psnr': QLabel("PSNR: --"),
            'total_batches': QLabel("Batches: --")
        }
        
        row = 0
        for key, label in self.result_labels.items():
            label.setStyleSheet("""
                font-size: 11px; 
                color: #ddd; 
                padding: 6px;
                background-color: #2b2b2b;
                border-radius: 3px;
                border-left: 3px solid #4CAF50;
                font-weight: 500;
            """)
            metrics_layout.addWidget(label, row // 2, row % 2)
            row += 1
        
        left_layout.addWidget(metrics_frame)
        
        # Batch details
        details_frame = QFrame()
        details_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        details_layout = QVBoxLayout(details_frame)
        
        details_title = QLabel("Batch Analysis")
        details_title.setStyleSheet("""
            font-size: 12px; 
            font-weight: bold; 
            color: #4CAF50; 
            padding: 4px;
        """)
        details_layout.addWidget(details_title)
        
        self.batch_details_text = QTextEdit()
        self.batch_details_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ccc;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9px;
                padding: 6px;
            }
        """)
        self.batch_details_text.setReadOnly(True)
        details_layout.addWidget(self.batch_details_text)
        
        left_layout.addWidget(details_frame)
        
        # Action buttons
        actions_frame = QFrame()
        actions_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        actions_layout = QGridLayout(actions_frame)
        actions_layout.setSpacing(6)
        
        self.save_btn = QPushButton("Save Video")
        self.export_metrics_btn = QPushButton("Export Metrics")
        self.new_compression_btn = QPushButton("New Compression")
        
        action_buttons = [self.save_btn, self.export_metrics_btn, self.new_compression_btn]
        
        for i, btn in enumerate(action_buttons):
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #555, stop:1 #444);
                    color: white;
                    border: 1px solid #666;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #666, stop:1 #555);
                }
            """)
            actions_layout.addWidget(btn, i // 2, i % 2)
        
        if len(action_buttons) % 2:
            actions_layout.addWidget(QWidget(), (len(action_buttons)) // 2, 1)
        
        self.save_btn.clicked.connect(self.save_compressed)
        self.export_metrics_btn.clicked.connect(self.export_metrics)
        self.new_compression_btn.clicked.connect(self.reset_ui)
        
        left_layout.addWidget(actions_frame)
        
        # Right side - Charts
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        charts_title = QLabel("Performance Charts")
        charts_title.setAlignment(Qt.AlignCenter)
        charts_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #4CAF50;
            padding: 8px;
            background-color: #333;
            border-radius: 4px;
            margin-bottom: 5px;
        """)
        right_layout.addWidget(charts_title)
        
        self.metrics_chart = MetricsChart(right_panel, width=8, height=6)
        right_layout.addWidget(self.metrics_chart)
        
        # Set panel proportions
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)
        
        return widget
    
    def load_precompressed_video(self):
        if not self.video_path:
            QMessageBox.warning(self, "No Original Video", "Please load an original video first.")
            return
        
        if self.processor and self.processor.isRunning():
            QMessageBox.warning(self, "Processing Active", "Cannot load pre-compressed video while processing.")
            return
        
        compressed_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pre-Compressed Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*.*)"
        )
        
        if not compressed_path:
            return
        
        metrics_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Metrics JSON File",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not metrics_path:
            return
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            metrics['original_path'] = self.video_path
            metrics['compressed_path'] = compressed_path
            
            metrics['original_size'] = os.path.getsize(self.video_path)
            metrics['compressed_size'] = os.path.getsize(compressed_path)
            metrics['video_compression_ratio'] = metrics['original_size'] / metrics['compressed_size']
            metrics['video_space_saved'] = 1 - (metrics['compressed_size'] / metrics['original_size'])
            
            self.precompressed_mode = True
            
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
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title_label = QLabel("Video Comparison")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
            padding: 10px;
            background-color: #333;
            border-radius: 6px;
            margin-bottom: 10px;
        """)
        layout.addWidget(title_label)
        
        previews_layout = QHBoxLayout()
        previews_layout.setSpacing(15)
        
        self.original_preview = VideoPreviewWidget("Original Video")
        self.compressed_preview = VideoPreviewWidget("Compressed Video")
        
        previews_layout.addWidget(self.original_preview)
        previews_layout.addWidget(self.compressed_preview)
        
        layout.addLayout(previews_layout)
        
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
            }
        """)
        controls_layout = QHBoxLayout(controls_frame)
        
        self.compress_btn = QPushButton("🚀 Start HiNeRV Compression")
        self.compress_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #1976D2, stop:1 #1565C0);
            }
            QPushButton:pressed {
                background: #1565C0;
            }
        """)
        self.compress_btn.clicked.connect(self.start_compression)
        controls_layout.addWidget(self.compress_btn)
        
        self.stop_btn = QPushButton("⏹ Stop Processing")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #f44336, stop:1 #d32f2f);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #d32f2f, stop:1 #c62828);
            }
            QPushButton:pressed {
                background: #c62828;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_compression)
        self.stop_btn.setVisible(False)
        controls_layout.addWidget(self.stop_btn)
        
        layout.addWidget(controls_frame)
        
        return widget
    
    def create_progress_section(self):
        widget = QGroupBox("Processing Progress")
        widget.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                padding-top: 10px;
                background-color: #2b2b2b;
                border-radius: 6px;
            }
        """)
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            font-size: 12px; 
            color: #ddd; 
            padding: 6px;
            background-color: #333;
            border-radius: 3px;
            margin-bottom: 3px;
        """)
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 6px;
                text-align: center;
                height: 24px;
                font-size: 11px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                           stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        return widget
    
    def create_results_section(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title_label = QLabel("🎯 Compression Results & Analysis")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: #4CAF50;
            padding: 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #2b2b2b, stop:0.5 #333, stop:1 #2b2b2b);
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        layout.addWidget(title_label)
        
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #444;
                background-color: #2b2b2b;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #333;
                color: #ddd;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #555;
            }
        """)
        
        overview_tab = self.create_overview_tab()
        charts_tab = self.create_charts_tab()
        details_tab = self.create_details_tab()
        
        tabs.addTab(overview_tab, "📊 Overview")
        tabs.addTab(charts_tab, "📈 Charts")
        tabs.addTab(details_tab, "📋 Details")
        
        layout.addWidget(tabs)
        
        actions_frame = self.create_actions_frame()
        layout.addWidget(actions_frame)
        
        return widget
    
    def create_overview_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setSpacing(15)
        
        self.result_labels = {
            'original_size': QLabel("Original Size: --"),
            'compressed_size': QLabel("Compressed Size: --"),
            'video_compression_ratio': QLabel("Compression Ratio: --"),
            'video_space_saved': QLabel("Space Saved: --"),
            'avg_psnr': QLabel("Average PSNR: --"),
            'total_batches': QLabel("Total Batches: --"),
            'neural_success': QLabel("Neural Success: --"),
            'processing_time': QLabel("Processing: Complete")
        }
        
        row = 0
        for key, label in self.result_labels.items():
            label.setStyleSheet("""
                font-size: 14px; 
                color: #ddd; 
                padding: 8px;
                background-color: #2b2b2b;
                border-radius: 4px;
                border-left: 4px solid #4CAF50;
                font-weight: 500;
            """)
            metrics_layout.addWidget(label, row // 2, row % 2)
            row += 1
        
        layout.addWidget(metrics_frame)
        
        return widget
    
    def create_charts_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.metrics_chart = MetricsChart(widget, width=12, height=8)
        layout.addWidget(self.metrics_chart)
        
        return widget
    
    def create_details_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.batch_details_text = QTextEdit()
        self.batch_details_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ddd;
                border: 2px solid #444;
                border-radius: 6px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                padding: 10px;
            }
        """)
        self.batch_details_text.setReadOnly(True)
        layout.addWidget(self.batch_details_text)
        
        return widget
    
    def create_actions_frame(self):
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
            }
        """)
        layout = QGridLayout(frame)
        layout.setSpacing(10)
        
        self.play_original_btn = QPushButton("▶ Play Original")
        self.play_compressed_btn = QPushButton("▶ Play Compressed")
        self.save_btn = QPushButton("💾 Save Video")
        self.export_metrics_btn = QPushButton("📊 Export Metrics")
        
        buttons = [self.play_original_btn, self.play_compressed_btn, self.save_btn, self.export_metrics_btn]
        
        for i, btn in enumerate(buttons):
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #555, stop:1 #444);
                    color: white;
                    border: 2px solid #666;
                    padding: 12px 18px;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #666, stop:1 #555);
                    border-color: #777;
                }
                QPushButton:pressed {
                    background: #444;
                }
            """)
            layout.addWidget(btn, i // 2, i % 2)
        
        self.play_original_btn.clicked.connect(self.play_original)
        self.play_compressed_btn.clicked.connect(self.play_compressed)
        self.save_btn.clicked.connect(self.save_compressed)
        self.export_metrics_btn.clicked.connect(self.export_metrics)
        
        new_btn = QPushButton("🔄 Compress Another Video")
        new_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: none;
                padding: 15px 25px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #45a049, stop:1 #3d8b40);
            }
            QPushButton:pressed {
                background: #3d8b40;
            }
        """)
        new_btn.clicked.connect(self.reset_ui)
        layout.addWidget(new_btn, 2, 0, 1, 2)
        
        return frame
    
    def apply_theme(self):
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
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
            }
        """)
    
    def browse_video(self, event=None):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*.*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, file_path):
        self.video_path = file_path
        
        self.upload_area.setText(f"✅ Selected: {os.path.basename(file_path)}")
        self.upload_area.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                font-size: 12px;
                color: #4CAF50;
                font-weight: bold;
            }
        """)
        
        self.main_tabs.setVisible(True)
        self.original_preview.load_video(file_path)
    
    def start_compression(self):
        if not self.video_path:
            return
        
        if self.precompressed_mode:
            QMessageBox.information(
                self, 
                "Pre-compressed Mode", 
                "This video was loaded with pre-existing compression results.\nUse 'New Compression' to start fresh."
            )
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(self.video_path).stem
        
        gui_dir = Path(__file__).parent
        self.output_dir = gui_dir / "output" / f"{video_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.compress_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self.progress_widget.setVisible(True)
        
        self.processor = VideoProcessor(str(self.video_path), str(self.output_dir), batch_size=40)
        self.processor.progress.connect(self.update_progress)
        self.processor.finished.connect(self.on_compression_finished)
        self.processor.error.connect(self.on_compression_error)
        self.processor.start()
    
    def stop_compression(self):
        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait(5000)
            
        self.compress_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.progress_widget.setVisible(False)
        
        QMessageBox.information(self, "Stopped", "Compression process has been stopped.")
    
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def on_compression_finished(self, results):
        self.upload_widget.setVisible(False)
        
        self.compress_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.progress_widget.setVisible(False)
        
        # Switch to results tab
        self.main_tabs.setCurrentIndex(1)
        
        self.compression_results = results
        
        if 'compressed_path' in results:
            self.compressed_preview.load_video(results['compressed_path'])
        
        self.result_labels['original_size'].setText(
            f"Original: {self.format_size(results.get('original_size', 0))}"
        )
        self.result_labels['compressed_size'].setText(
            f"Compressed: {self.format_size(results.get('compressed_size', 0))}"
        )
        self.result_labels['video_compression_ratio'].setText(
            f"Ratio: {results.get('video_compression_ratio', 0):.2f}x"
        )
        self.result_labels['video_space_saved'].setText(
            f"Saved: {results.get('video_space_saved', 0)*100:.1f}%"
        )
        
        avg_psnr = results.get('avg_psnr', 0)
        if avg_psnr > 0:
            self.result_labels['avg_psnr'].setText(f"PSNR: {avg_psnr:.1f} dB")
        else:
            self.result_labels['avg_psnr'].setText("PSNR: N/A")
        
        total_batches = results.get('total_batches', 0)
        self.result_labels['total_batches'].setText(f"Batches: {total_batches}")
        
        self.update_batch_details(results)
        self.metrics_chart.plot_compression_metrics(results)
        
        fallback_count = results.get('fallback_batches', 0)
        successful_neural = total_batches - fallback_count
        success_rate = (successful_neural / total_batches * 100) if total_batches > 0 else 0
        
        QMessageBox.information(
            self, 
            "Compression Complete", 
            f"Video compression completed!\n\n"
            f"Processed {results.get('total_frames_processed', 0)} frames in {total_batches} batches\n"
            f"Neural compression success: {success_rate:.1f}%\n"
            f"Compression ratio: {results.get('video_compression_ratio', 1):.2f}x\n"
            f"Space saved: {results.get('video_space_saved', 0)*100:.1f}%"
        )

    def on_compression_error(self, error_msg):
        self.compress_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.progress_widget.setVisible(False)
        
        log_file_hint = ""
        if self.output_dir:
            log_files = glob.glob(os.path.join(self.output_dir, "**/training_log_batch_*.txt"), recursive=True)
            if log_files:
                log_file_hint = f"\n\nDetailed logs saved to:\n" + "\n".join(log_files[:3])
                if len(log_files) > 3:
                    log_file_hint += f"\n... and {len(log_files) - 3} more"
        
        error_dialog = QMessageBox(self)
        error_dialog.setWindowTitle("Compression Error")
        error_dialog.setText("An error occurred during batch compression:")
        error_dialog.setDetailedText(error_msg + log_file_hint)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.exec()
    
    def play_original(self):
        if self.video_path and os.path.exists(self.video_path):
            self.open_video_file(self.video_path)
    
    def play_compressed(self):
        if hasattr(self, 'compression_results'):
            compressed_path = self.compression_results.get('compressed_path')
            if compressed_path and os.path.exists(compressed_path):
                self.open_video_file(compressed_path)
    
    def open_video_file(self, file_path):
        success = False
        last_error = ""
        
        # Convert WSL path to Windows path if needed
        windows_path = self.convert_wsl_path_to_windows(file_path)
        
        # WSL-specific video playback methods
        if self.is_wsl_environment():
            success = self.try_wsl_video_playback(windows_path, file_path)
            if success:
                return
        
        # Try system default first
        try:
            if sys.platform.startswith('win'):
                os.startfile(file_path)
                return
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', file_path], check=True)
                return
            else:
                subprocess.run(['xdg-open', file_path], check=True, 
                             stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                return
        except Exception as e:
            last_error = str(e)
        
        # Try specific video players
        video_players = []
        if sys.platform.startswith('win'):
            video_players = [
                ['vlc.exe', file_path],
                ['wmplayer.exe', file_path],
                ['mpc-hc.exe', file_path],
                ['mpc-hc64.exe', file_path],
                ['potplayer.exe', file_path]
            ]
        elif sys.platform.startswith('darwin'):
            video_players = [
                ['open', '-a', 'VLC', file_path],
                ['open', '-a', 'QuickTime Player', file_path],
                ['open', '-a', 'IINA', file_path],
                ['open', '-a', 'Elmedia Player', file_path]
            ]
        else:
            # Linux/Unix systems
            video_players = [
                ['vlc', file_path],
                ['mpv', file_path],
                ['totem', file_path],
                ['mplayer', file_path],
                ['smplayer', file_path],
                ['kaffeine', file_path],
                ['dragon', file_path],
                ['celluloid', file_path],
                ['gnome-videos', file_path],
                ['parole', file_path]
            ]
        
        for player_cmd in video_players:
            try:
                # Check if player exists
                if sys.platform.startswith('win'):
                    result = subprocess.run(['where', player_cmd[0]], 
                                          capture_output=True, text=True)
                    if result.returncode != 0:
                        continue
                else:
                    result = subprocess.run(['which', player_cmd[0]], 
                                          capture_output=True, text=True)
                    if result.returncode != 0:
                        continue
                
                # Try to launch the player
                subprocess.Popen(player_cmd, 
                               stderr=subprocess.DEVNULL, 
                               stdout=subprocess.DEVNULL)
                success = True
                break
            except Exception as e:
                last_error = str(e)
                continue
        
        # Try flatpak apps on Linux
        if not success and not sys.platform.startswith('win') and not sys.platform.startswith('darwin'):
            flatpak_players = [
                ['flatpak', 'run', 'org.videolan.VLC', file_path],
                ['flatpak', 'run', 'io.github.celluloid_player.Celluloid', file_path],
                ['flatpak', 'run', 'org.gnome.Totem', file_path]
            ]
            
            for player_cmd in flatpak_players:
                try:
                    subprocess.run(['which', 'flatpak'], check=True, 
                                 capture_output=True)
                    subprocess.Popen(player_cmd,
                                   stderr=subprocess.DEVNULL, 
                                   stdout=subprocess.DEVNULL)
                    success = True
                    break
                except:
                    continue
        
        # Try snap apps on Linux
        if not success and not sys.platform.startswith('win') and not sys.platform.startswith('darwin'):
            snap_players = [
                ['snap', 'run', 'vlc', file_path],
                ['snap', 'run', 'mpv', file_path]
            ]
            
            for player_cmd in snap_players:
                try:
                    subprocess.run(['which', 'snap'], check=True, 
                                 capture_output=True)
                    subprocess.Popen(player_cmd,
                                   stderr=subprocess.DEVNULL, 
                                   stdout=subprocess.DEVNULL)
                    success = True
                    break
                except:
                    continue
        
        if not success:
            # Show helpful dialog for WSL users
            if self.is_wsl_environment():
                QMessageBox.information(
                    self, 
                    "WSL Video Playback", 
                    f"Cannot directly play video in WSL environment.\n\n"
                    f"Options:\n"
                    f"1. Copy this path to Windows: {windows_path}\n"
                    f"2. Install a video player: sudo apt install vlc\n"
                    f"3. Use Windows Subsystem for Linux GUI (WSLg)\n\n"
                    f"File location: {file_path}"
                )
            else:
                QMessageBox.warning(
                    self, 
                    "Cannot Play Video", 
                    f"Unable to find a video player to open the file.\n"
                    f"Please install VLC, MPV, or another video player.\n\n"
                    f"File location: {file_path}\n"
                    f"Last error: {last_error}"
                )
    
    def is_wsl_environment(self):
        """Check if running in WSL environment"""
        try:
            # Check for WSL-specific indicators
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                return 'microsoft' in version_info or 'wsl' in version_info
        except:
            # Alternative check
            try:
                result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
                return 'microsoft' in result.stdout.lower() or 'wsl' in result.stdout.lower()
            except:
                return False
    
    def convert_wsl_path_to_windows(self, wsl_path):
        """Convert WSL path to Windows path"""
        try:
            # Use wslpath if available
            result = subprocess.run(['wslpath', '-w', wsl_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Manual conversion as fallback
        if wsl_path.startswith('/mnt/'):
            # /mnt/c/path -> C:\path
            parts = wsl_path.split('/')
            if len(parts) >= 3:
                drive = parts[2].upper()
                path_parts = parts[3:]
                windows_path = f"{drive}:\\" + "\\".join(path_parts)
                return windows_path
        
        return wsl_path
    
    def try_wsl_video_playback(self, windows_path, linux_path):
        """Try WSL-specific video playback methods"""
        methods = [
            # Method 1: Use cmd.exe to launch Windows media player
            lambda: self.try_windows_cmd_playback(windows_path),
            
            # Method 2: Use powershell.exe to invoke Windows media
            lambda: self.try_powershell_playback(windows_path),
            
            # Method 3: Try WSLg if available
            lambda: self.try_wslg_playback(linux_path),
            
            # Method 4: Use explorer.exe to open file
            lambda: self.try_explorer_playback(windows_path)
        ]
        
        for method in methods:
            try:
                if method():
                    return True
            except Exception as e:
                continue
        
        return False
    
    def try_windows_cmd_playback(self, windows_path):
        """Try to play video using Windows cmd.exe"""
        try:
            # Use cmd.exe to start the file with default Windows association
            subprocess.Popen([
                'cmd.exe', '/c', 'start', '""', windows_path
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            return True
        except:
            return False
    
    def try_powershell_playback(self, windows_path):
        """Try to play video using PowerShell"""
        try:
            # Use PowerShell to invoke the file
            ps_command = f'Invoke-Item "{windows_path}"'
            subprocess.Popen([
                'powershell.exe', '-Command', ps_command
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            return True
        except:
            return False
    
    def try_wslg_playback(self, linux_path):
        """Try to play video using WSLg (Windows Subsystem for Linux GUI)"""
        try:
            # Check if DISPLAY is set (indicates WSLg or X11 forwarding)
            if 'DISPLAY' not in os.environ:
                return False
            
            # Try common Linux video players that work with WSLg
            players = ['vlc', 'mpv', 'totem', 'mplayer']
            
            for player in players:
                try:
                    # Check if player exists
                    result = subprocess.run(['which', player], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        # Launch the player
                        subprocess.Popen([player, linux_path],
                                       stderr=subprocess.DEVNULL, 
                                       stdout=subprocess.DEVNULL)
                        return True
                except:
                    continue
            
            return False
        except:
            return False
    
    def try_explorer_playback(self, windows_path):
        """Try to open file using Windows Explorer"""
        try:
            # Use explorer.exe to open the file
            subprocess.Popen([
                'explorer.exe', windows_path
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            return True
        except:
            return False
    
    def save_compressed(self):
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
        self.video_path = None
        self.output_dir = None
        self.precompressed_mode = False
        self.compression_results = None
        
        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait(5000)
        
        self.upload_widget.setVisible(True)
        self.main_tabs.setVisible(False)
        
        self.upload_area.setText("Drag and drop a video file here or click to browse")
        self.upload_area.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px dashed #666;
                border-radius: 8px;
                font-size: 14px;
                color: #bbb;
                font-weight: 500;
            }
            QLabel:hover {
                border-color: #4CAF50;
                color: #ddd;
                background-color: #2f2f2f;
            }
        """)
        
        self.compress_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.progress_widget.setVisible(False)
        
        # Reset to first tab
        self.main_tabs.setCurrentIndex(0)
    
    def format_size(self, size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
            if any(files[0].lower().endswith(ext) for ext in video_extensions):
                self.load_video(files[0])
    
    def closeEvent(self, event):
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
        details_text = "Batch Processing Analysis\n"
        details_text += "=" * 40 + "\n\n"
        
        batch_metrics = results.get('batch_metrics', [])
        
        for batch in batch_metrics:
            if not batch:
                continue
                
            batch_idx = batch.get('batch_idx', 0)
            frame_range = batch.get('frame_range', '--')
            frame_count = batch.get('frame_count', 0)
            
            details_text += f"Batch {batch_idx + 1:2d}: Frames {frame_range:>8} ({frame_count:2d})\n"
            
            training = batch.get('training_metrics', {})
            if 'final_psnr' in training and training['final_psnr'] > 0:
                details_text += f"   PSNR: {training['final_psnr']:5.2f} dB\n"
            
            compression = batch.get('compression_metrics', {})
            if 'compression_ratio' in compression:
                ratio = compression['compression_ratio']
                space_saved = compression.get('space_saved', 0) * 100
                fallback = compression.get('fallback_used', False)
                
                if fallback:
                    details_text += f"   Status: FALLBACK\n"
                else:
                    details_text += f"   Ratio: {ratio:4.2f}x, Saved: {space_saved:4.1f}%\n"
            
            details_text += "\n"
        
        total_batches = len(batch_metrics)
        fallback_count = results.get('fallback_batches', 0)
        
        details_text += f"Summary:\n"
        details_text += f"Neural: {total_batches - fallback_count}/{total_batches} batches\n"
        details_text += f"Fallback: {fallback_count} batches\n"
        
        if results.get('avg_psnr', 0) > 0:
            details_text += f"Avg PSNR: {results['avg_psnr']:.2f} dB\n"
        
        details_text += f"Overall: {results.get('video_compression_ratio', 1):.2f}x\n"
        details_text += f"Saved: {results.get('video_space_saved', 0)*100:.1f}%\n"
        
        self.batch_details_text.setPlainText(details_text)

    def export_metrics(self):
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
                    with open(file_path, 'w') as f:
                        f.write("HiNeRV Neural Video Compression Results\n")
                        f.write("=" * 50 + "\n\n")
                        
                        results = self.compression_results
                        
                        f.write(f"Original Video: {results.get('original_path', 'N/A')}\n")
                        f.write(f"Compressed Video: {results.get('compressed_path', 'N/A')}\n\n")
                        
                        f.write("Performance Metrics:\n")
                        f.write(f"  Original Size: {self.format_size(results.get('original_size', 0))}\n")
                        f.write(f"  Compressed Size: {self.format_size(results.get('compressed_size', 0))}\n")
                        f.write(f"  Compression Ratio: {results.get('video_compression_ratio', 0):.2f}x\n")
                        f.write(f"  Space Saved: {results.get('video_space_saved', 0)*100:.1f}%\n\n")
                        
                        f.write("Processing Details:\n")
                        f.write(f"  Total Batches: {results.get('total_batches', 0)}\n")
                        f.write(f"  Batch Size: {results.get('batch_size', 0)} frames\n")
                        f.write(f"  Total Frames: {results.get('total_frames_processed', 0)}\n")
                        f.write(f"  Neural Success: {results.get('total_batches', 0) - results.get('fallback_batches', 0)}/{results.get('total_batches', 0)} batches\n\n")
                        
                        if results.get('avg_psnr', 0) > 0:
                            f.write(f"Quality Metrics:\n")
                            f.write(f"  Average PSNR: {results['avg_psnr']:.2f} dB\n\n")
                        
                        f.write("Batch Analysis:\n")
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
    app = QApplication(sys.argv)
    app.setApplicationName("HiNeRV Video Compressor - Professional Edition")
    
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()