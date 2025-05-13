#!/usr/bin/env python3
"""
HiNeRV Processor - Core component that handles the compression process
"""

import os
import sys
import time
import json
import yaml
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from PySide6.QtCore import QThread, Signal, Slot

from .video_processor import VideoProcessor
from .system_monitor import SystemMonitor
from ..utils import format_filesize, format_duration


class HiNeRVProcessor(QThread):
    """
    Worker thread that handles the HiNeRV compression process.
    Manages video preprocessing, model training, and output generation.
    """
    
    # Signal definitions
    progress_updated = Signal(dict)  # Emit progress updates
    status_updated = Signal(str)     # Emit status messages
    error_occurred = Signal(str)     # Emit error messages
    finished = Signal(dict)          # Emit results when finished
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.is_paused = False
        self.pause_condition = threading.Condition()
        
        # Initialize video processor
        self.video_processor = VideoProcessor()
        
        # Initialize progress tracking
        self.start_time = None
        self.progress_data = {
            'epochs_completed': 0,
            'total_epochs': config['training_options'].get('epochs', 100),
            'frames_processed': 0,
            'total_frames': 0,
            'current_loss': 0.0,
            'loss_history': [],
            'psnr_history': [],
            'ms_ssim_history': [],
            'eta': None
        }
        
        # Save configuration
        self._save_config()
    
    def run(self):
        """Main execution method"""
        try:
            self.is_running = True
            self.start_time = time.time()
            
            # Emit initial status
            self.status_updated.emit("Initializing...")
            
            # Create output directory if it doesn't exist
            output_dir = self.config['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Preprocess video
            self._preprocess_video()
            if not self.is_running:
                return
                
            # Step 2: Train model
            self._train_model()
            if not self.is_running:
                return
                
            # Step 3: Generate output
            self._generate_output()
            if not self.is_running:
                return
                
            # Step 4: Evaluate results
            results = self._evaluate_results()
            if not self.is_running:
                return
                
            # Step 5: Clean up temporary files
            self._cleanup()
            
            # Emit final progress
            self.progress_data['epochs_completed'] = self.progress_data['total_epochs']
            self.progress_updated.emit(self.progress_data)
            
            # Emit finished signal with results
            self.finished.emit(results)
            
        except Exception as e:
            self.logger.exception("Exception in HiNeRV processor thread")
            self.error_occurred.emit(str(e))
        finally:
            self.is_running = False
    
    def pause(self):
        """Pause the compression process"""
        self.is_paused = True
        self.status_updated.emit("Paused")
        self.logger.info("Compression paused")
    
    def resume(self):
        """Resume the compression process"""
        with self.pause_condition:
            self.is_paused = False
            self.pause_condition.notify_all()
        self.status_updated.emit("Resumed")
        self.logger.info("Compression resumed")
    
    def stop(self):
        """Stop the compression process"""
        self.is_running = False
        self.status_updated.emit("Stopping...")
        self.logger.info("Compression stopping")
        
        # Wake up any paused thread
        with self.pause_condition:
            self.is_paused = False
            self.pause_condition.notify_all()
    
    def _check_pause(self):
        """Check if the process is paused and wait if needed"""
        if self.is_paused:
            with self.pause_condition:
                while self.is_paused and self.is_running:
                    self.pause_condition.wait()
    
    def _preprocess_video(self):
        """Preprocess the video: extract frames and metadata"""
        self.status_updated.emit("Preprocessing video...")
        
        video_path = self.config['video_path']
        output_dir = self.config['output_dir']
        frames_dir = os.path.join(output_dir, "frames")
        
        # Create frames directory
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract video metadata
        self.status_updated.emit("Extracting video metadata...")
        video_info = self.video_processor.get_video_info(video_path)
        
        # Determine frames to extract
        max_frames = self.config['training_options'].get('max_frames', 0)
        if max_frames <= 0:
            max_frames = video_info['frame_count']
        else:
            max_frames = min(max_frames, video_info['frame_count'])
        
        # Update total frames
        self.progress_data['total_frames'] = max_frames
        self.progress_updated.emit(self.progress_data)
        
        # Extract frames
        self.status_updated.emit(f"Extracting {max_frames} frames...")
        
        # Check if this is a quick test
        if self.config.get('quick_test', False):
            # For quick test, extract just a few frames
            max_frames = min(max_frames, 5)
            self.progress_data['total_frames'] = max_frames
        
        # Extract frames from video
        success = self.video_processor.extract_frames(
            video_path, frames_dir, max_frames=max_frames
        )
        
        if not success:
            raise RuntimeError("Failed to extract frames from video")
        
        # Update config with frame information
        self.config['frames_dir'] = frames_dir
        self.config['frame_count'] = max_frames
        self.config['video_info'] = video_info
        
        # Save updated config
        self._save_config()
        
        self.status_updated.emit("Preprocessing complete")
    
    def _train_model(self):
        """Train the HiNeRV model on extracted frames"""
        self.status_updated.emit("Preparing training...")
        
        # Get parameters from config
        output_dir = self.config['output_dir']
        frames_dir = self.config['frames_dir']
        training_options = self.config['training_options']
        model_preset = self.config['model_preset']
        resource_limits = self.config['resource_limits']
        
        # Create model directory
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create command line arguments for HiNeRV
        args = self._build_training_args(frames_dir, model_dir, training_options, model_preset, resource_limits)
        
        # Write args to file for debugging
        args_file = os.path.join(output_dir, "training_args.txt")
        with open(args_file, 'w') as f:
            f.write(" ".join(args))
        
        # Start training process
        self.status_updated.emit("Starting training...")
        self._run_training_process(args)
        
        # Update config
        self.config['model_dir'] = model_dir
        self._save_config()
        
        self.status_updated.emit("Training complete")
    
    def _build_training_args(self, frames_dir: str, model_dir: str, 
                           training_options: Dict, model_preset: Dict,
                           resource_limits: Dict) -> List[str]:
        """Build command line arguments for HiNeRV training"""
        # Base arguments
        args = [
            "accelerate", "launch",
            "--mixed_precision", "fp16",
            "--dynamo_backend", "inductor",
            "hinerv_train.py",
            "--frames_dir", frames_dir,
            "--output_dir", model_dir,
        ]
        
        # Add model preset arguments
        if 'config_file' in model_preset:
            args.extend(["--config", model_preset['config_file']])
        
        # Add training options
        for key, value in training_options.items():
            if value is None or (isinstance(value, bool) and not value):
                continue
                
            arg_name = f"--{key.replace('_', '-')}"
            
            # Handle boolean flags
            if isinstance(value, bool):
                args.append(arg_name)
            else:
                args.extend([arg_name, str(value)])
        
        # Add resource limits
        if resource_limits.get('max_memory_percentage', 100) < 100:
            memory_limit = resource_limits['max_memory_percentage']
            args.extend(["--max-memory-mb", str(int(memory_limit * 0.01 * self._get_total_gpu_memory()))])
        
        # Handle quick test
        if self.config.get('quick_test', False):
            # Override with minimal parameters for quick test
            for i, arg in enumerate(args):
                if arg == "--epochs":
                    args[i+1] = "1"
                elif arg == "--batch-size":
                    args[i+1] = "2"
        
        return args
    
    def _get_total_gpu_memory(self) -> int:
        """Get total GPU memory in MB"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            nvml.nvmlShutdown()
            return mem_info.total // (1024 * 1024)  # Convert to MB
        except:
            # Default to 8GB if we can't get the information
            return 8 * 1024
    
    def _run_training_process(self, args: List[str]):
        """Run the HiNeRV training process and monitor its progress"""
        log_file = os.path.join(self.config['output_dir'], "training_log.txt")
        rank0_log = os.path.join(self.config['model_dir'], "rank_0.txt")
        
        # Start process
        self.logger.info(f"Starting training process with args: {' '.join(args)}")
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Monitor process output
                for line in process.stdout:
                    # Write to log file
                    log.write(line)
                    log.flush()
                    
                    # Parse progress information
                    self._parse_progress_line(line)
                    
                    # Check if paused
                    self._check_pause()
                    
                    # Check if stopped
                    if not self.is_running:
                        process.terminate()
                        break
                
                # Wait for process to finish
                return_code = process.wait()
                
                if return_code != 0 and self.is_running:
                    raise RuntimeError(f"Training process failed with return code {return_code}")
                
        except Exception as e:
            self.logger.exception("Error running training process")
            raise RuntimeError(f"Error running training process: {str(e)}")
    
    def _parse_progress_line(self, line: str):
        """Parse a line of output from the training process"""
        try:
            # Check for epoch information
            if "Epoch:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Epoch:":
                        epoch = int(parts[i+1].strip(","))
                        self.progress_data['epochs_completed'] = epoch
                    elif part == "Loss:":
                        loss = float(parts[i+1])
                        self.progress_data['current_loss'] = loss
                        self.progress_data['loss_history'].append(loss)
                    elif part == "PSNR:":
                        psnr = float(parts[i+1])
                        self.progress_data['psnr_history'].append(psnr)
                    elif part == "MS-SSIM:":
                        ms_ssim = float(parts[i+1])
                        self.progress_data['ms_ssim_history'].append(ms_ssim)
                
                # Calculate ETA
                if self.start_time and self.progress_data['epochs_completed'] > 0:
                    elapsed = time.time() - self.start_time
                    total_epochs = self.progress_data['total_epochs']
                    completed_epochs = self.progress_data['epochs_completed']
                    
                    if completed_epochs > 0:
                        time_per_epoch = elapsed / completed_epochs
                        remaining_epochs = total_epochs - completed_epochs
                        eta_seconds = time_per_epoch * remaining_epochs
                        
                        eta = datetime.now() + timedelta(seconds=eta_seconds)
                        self.progress_data['eta'] = eta.strftime("%H:%M:%S")
                
                # Emit progress update
                self.progress_updated.emit(self.progress_data)
                
                # Update status
                self.status_updated.emit(
                    f"Training - Epoch {self.progress_data['epochs_completed']}/{self.progress_data['total_epochs']}, "
                    f"Loss: {self.progress_data['current_loss']:.4f}"
                )
        except Exception as e:
            # Just log errors in parsing, don't break the process
            self.logger.error(f"Error parsing progress line: {str(e)}")
    
    def _generate_output(self):
        """Generate the compressed output video"""
        self.status_updated.emit("Generating compressed output...")
        
        # Get parameters from config
        output_dir = self.config['output_dir']
        model_dir = self.config['model_dir']
        video_info = self.config['video_info']
        
        # Create output paths
        compressed_file = os.path.join(output_dir, "compressed.bin")
        output_video = os.path.join(output_dir, "output.mp4")
        
        # Build arguments for compression
        compress_args = [
            "python", "hinerv_compress.py",
            "--model_dir", model_dir,
            "--output", compressed_file
        ]
        
        # Run compression process
        self.status_updated.emit("Compressing model...")
        try:
            compress_log = os.path.join(output_dir, "compression_log.txt")
            with open(compress_log, 'w') as log:
                process = subprocess.run(
                    compress_args,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True
                )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Compression process failed: {str(e)}")
            raise RuntimeError(f"Failed to compress model: {str(e)}")
        
        # Generate output video from compressed bitstream
        self.status_updated.emit("Generating output video...")
        try:
            # Use the HiNeRV decompress script to convert bitstream to video
            decompress_args = [
                "python", "hinerv_decompress.py",
                "--input", compressed_file,
                "--output", output_video,
                "--fps", str(video_info['fps'])
            ]
            
            decompress_log = os.path.join(output_dir, "decompression_log.txt")
            with open(decompress_log, 'w') as log:
                process = subprocess.run(
                    decompress_args,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True
                )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Decompression process failed: {str(e)}")
            raise RuntimeError(f"Failed to generate output video: {str(e)}")
        
        # Update config
        self.config['compressed_file'] = compressed_file
        self.config['output_video'] = output_video
        self._save_config()
        
        self.status_updated.emit("Output generation complete")
    
    def _evaluate_results(self) -> Dict:
        """Evaluate the compression results"""
        self.status_updated.emit("Evaluating results...")
        
        # Get file sizes
        input_size = os.path.getsize(self.config['video_path'])
        output_size = os.path.getsize(self.config['compressed_file'])
        
        # Calculate compression ratio
        compression_ratio = input_size / output_size if output_size > 0 else 0
        
        # Calculate bitrate
        video_info = self.config['video_info']
        video_duration = video_info['duration']
        bitrate_kbps = (output_size * 8 / 1000) / video_duration if video_duration > 0 else 0
        
        # Get quality metrics from the last training iteration
        psnr = self.progress_data['psnr_history'][-1] if self.progress_data['psnr_history'] else 0
        ms_ssim = self.progress_data['ms_ssim_history'][-1] if self.progress_data['ms_ssim_history'] else 0
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Prepare results
        results = {
            'video_path': self.config['video_path'],
            'output_video': self.config['output_video'],
            'compressed_file': self.config['compressed_file'],
            'model_dir': self.config['model_dir'],
            'input_size': input_size,
            'output_size': output_size,
            'compression_ratio': compression_ratio,
            'bitrate_kbps': bitrate_kbps,
            'psnr': psnr,
            'ms_ssim': ms_ssim,
            'elapsed_time': elapsed,
            'elapsed_formatted': format_duration(elapsed),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'video_info': self.config['video_info']
        }
        
        # Save results to file
        results_file = os.path.join(self.config['output_dir'], "results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.status_updated.emit("Evaluation complete")
        return results
    
    def _cleanup(self):
        """Clean up temporary files"""
        self.status_updated.emit("Cleaning up...")
        
        # Only clean up if not in debug mode
        if not self.config.get('debug', False):
            # Remove frames directory
            frames_dir = self.config.get('frames_dir')
            if frames_dir and os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
        
        self.status_updated.emit("Cleanup complete")
    
    def _save_config(self):
        """Save the current configuration to a YAML file"""
        config_file = os.path.join(self.config['output_dir'], "args.yaml")
        
        # Make a copy of the config to avoid modifying the original
        config_copy = self.config.copy()
        
        # Remove large or unnecessary fields
        if 'video_info' in config_copy:
            video_info_copy = config_copy['video_info'].copy()
            # Remove any binary data or very large fields
            if 'thumbnail' in video_info_copy:
                del video_info_copy['thumbnail']
            config_copy['video_info'] = video_info_copy
        
        # Save to file
        with open(config_file, 'w') as f:
            yaml.dump(config_copy, f, default_flow_style=False)