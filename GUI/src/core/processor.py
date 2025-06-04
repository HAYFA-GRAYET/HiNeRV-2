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
            self._verify_paths()


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
        
        # Ensure frame_count exists in video_info
        if 'frame_count' not in video_info:
            # Calculate frame count from duration and fps
            fps = video_info.get('fps', 30)  # Default to 30 fps if not present
            duration = video_info.get('duration', 0)
            video_info['frame_count'] = int(duration * fps)
        
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
        success = self.video_processor._extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            frame_limit=max_frames  # Changed from max_frames to frame_limit
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
        
        # Store model_dir in config BEFORE using it
        self.config['model_dir'] = model_dir
        self._save_config()
        
        # Get total frames
        total_frames = self.config.get('frame_count', 0)
        max_frames_per_batch = 100  # Process 100 frames at a time
        
        if total_frames <= max_frames_per_batch:
            # Process all frames in one batch
            self._train_single_batch(frames_dir, model_dir, training_options, model_preset, resource_limits)
        else:
            # Process in batches
            self._train_multiple_batches(frames_dir, model_dir, training_options, model_preset, resource_limits, total_frames, max_frames_per_batch)
        
        self.status_updated.emit("Training complete")

    def _train_single_batch(self, frames_dir, model_dir, training_options, model_preset, resource_limits):
        """Train on a single batch of frames"""
        args = self._build_training_args(frames_dir, model_dir, training_options, model_preset, resource_limits)
        args_file = os.path.join(self.config['output_dir'], "training_args.txt")
        with open(args_file, 'w') as f:
            f.write(" ".join(args))
        
        self.status_updated.emit("Starting training...")
        self._run_training_process(args)

    def _train_multiple_batches(self, frames_dir, model_dir, training_options, model_preset, resource_limits, total_frames, max_frames_per_batch):
        """Train on multiple batches of frames"""
        num_batches = (total_frames + max_frames_per_batch - 1) // max_frames_per_batch
        
        self.status_updated.emit(f"Processing {total_frames} frames in {num_batches} batches...")
        
        # Create batch directories
        batch_models = []
        
        for batch_idx in range(num_batches):
            if not self.is_running:
                return
                
            # Calculate frame range for this batch
            start_frame = batch_idx * max_frames_per_batch
            end_frame = min((batch_idx + 1) * max_frames_per_batch, total_frames)
            batch_size = end_frame - start_frame
            
            self.status_updated.emit(f"Processing batch {batch_idx + 1}/{num_batches} (frames {start_frame}-{end_frame})...")
            
            # Create batch-specific directories
            batch_frames_dir = os.path.join(self.config['output_dir'], f"batch_{batch_idx}_frames")
            batch_model_dir = os.path.join(model_dir, f"batch_{batch_idx}")
            os.makedirs(batch_frames_dir, exist_ok=True)
            os.makedirs(batch_model_dir, exist_ok=True)
            
            # Copy frames for this batch
            import shutil
            for i in range(start_frame, end_frame):
                src_frame = os.path.join(frames_dir, f"frame_{i+1:06d}.png")
                dst_frame = os.path.join(batch_frames_dir, f"frame_{i-start_frame+1:06d}.png")
                if os.path.exists(src_frame):
                    shutil.copy2(src_frame, dst_frame)
            
            # Train on this batch
            args = self._build_training_args(batch_frames_dir, batch_model_dir, training_options, model_preset, resource_limits)
            
            # Update progress for overall process
            overall_progress = (batch_idx + 0.5) / num_batches
            self.progress_updated.emit({
                'progress': overall_progress,
                'status': f"Training batch {batch_idx + 1}/{num_batches}",
                'elapsed_time': time.time() - self.start_time
            })
            
            self._run_training_process(args)
            
            batch_models.append(batch_model_dir)
            
            # Clean up batch frames to save space
            shutil.rmtree(batch_frames_dir)
        
        # Store batch model directories
        self.config['batch_models'] = batch_models
        self._save_config()
    def _build_training_args(self, frames_dir: str, model_dir: str, 
                        training_options: Dict, model_preset: Dict,
                        resource_limits: Dict) -> List[str]:
        """Build command line arguments for HiNeRV training"""
        # Get the HiNeRV root directory (parent of GUI folder)
        gui_dir = Path(__file__).parent.parent.parent
        hinerv_root = gui_dir.parent
        
        # Change to HiNeRV root directory for execution
        os.chdir(hinerv_root)
        
        # For HiNeRV, we need to pass the parent directory as dataset
        # and the subdirectory name as dataset-name
        dataset_parent = os.path.dirname(frames_dir)
        dataset_name = os.path.basename(frames_dir)
        
        # Base arguments
        args = [
            "accelerate", "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=inductor",
            "hinerv_main.py",
            "--dataset", dataset_parent,  # Parent directory
            "--dataset-name", dataset_name,  # Subdirectory name
            "--output", model_dir,
        ]
        
        # Read config files and append their contents
        if 'config' in model_preset and 'file_path' in model_preset:
            config_path = model_preset['file_path']
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_content = f.read().strip()
                    # Split by whitespace and add each argument
                    config_args = config_content.split()
                    args.extend(config_args)
        
        # Read training config file
        training_config_path = os.path.join(hinerv_root, "cfgs/train/hinerv_1920x1080.txt")
        if os.path.exists(training_config_path):
            with open(training_config_path, 'r') as f:
                train_config_content = f.read().strip()
                train_args = train_config_content.split()
                args.extend(train_args)
        
        # Override with specific training options from GUI
        args.extend([
            "--batch-size", str(training_options.get('batch-size', 2)),
            "--eval-batch-size", str(training_options.get('eval-batch-size', 1)),
            "--grad-accum", "1",
            "--log-eval", "true",
            "--seed", "0"
        ])
        
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
        
        # Get the HiNeRV root directory
        gui_dir = Path(__file__).parent.parent.parent
        hinerv_root = gui_dir.parent
        
        # Set up environment to fix library issues
        env = os.environ.copy()
        # Force Python to use unbuffered output
        env['PYTHONUNBUFFERED'] = '1'
        
        # Add conda lib path to LD_LIBRARY_PATH if needed
        conda_env_path = os.environ.get('CONDA_PREFIX', '')
        if conda_env_path:
            lib_path = os.path.join(conda_env_path, 'lib')
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = f"{lib_path}:{env['LD_LIBRARY_PATH']}"
            else:
                env['LD_LIBRARY_PATH'] = lib_path
        
        # Start process
        self.logger.info(f"Starting training process with args: {' '.join(args)}")
        self.logger.info(f"Working directory: {hinerv_root}")
        
        try:
            with open(log_file, 'w') as log:
                # Use Popen with proper buffering settings
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,  # Line buffered
                    cwd=str(hinerv_root),
                    env=env
                )
                
                # Update status to show we're initializing
                self.status_updated.emit("Initializing training environment...")
                self.progress_updated.emit({
                    'status': 'Initializing training environment...',
                    'progress': 0.0,
                    'elapsed_time': time.time() - self.start_time
                })
                
                # Monitor process output
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                        
                    if line:
                        # Write to log file
                        log.write(line)
                        log.flush()
                        
                        # Log the line for debugging
                        self.logger.debug(f"Process output: {line.strip()}")
                        
                        # Parse progress information
                        self._parse_progress_line(line.strip())
                        
                    # Check if paused
                    self._check_pause()
                    
                    # Check if stopped
                    if not self.is_running:
                        process.terminate()
                        break
                
                # Wait for process to finish
                return_code = process.poll()
                
                if return_code != 0 and self.is_running:
                    # Read the log file to get error details
                    with open(log_file, 'r') as f:
                        error_details = f.read()
                    self.logger.error(f"Training failed. Log contents:\n{error_details}")
                    raise RuntimeError(f"Training process failed with return code {return_code}")
                
        except Exception as e:
            self.logger.exception("Error running training process")
            raise RuntimeError(f"Error running training process: {str(e)}")
    
    def _parse_progress_line(self, line: str):
        """Parse a line of output from the training process"""
        try:
            # Check if in dev mode
            try:
                from main import DEV_MODE_ENABLED
            except ImportError:
                DEV_MODE_ENABLED = False

            if not DEV_MODE_ENABLED:
                # Simplified progress for non-dev mode
                if "Start main training for" in line:
                    # Extract total epochs
                    import re
                    match = re.search(r'for (\d+) epochs', line)
                    if match:
                        total_epochs = int(match.group(1))
                        self.progress_data['total_epochs'] = total_epochs
                        self.progress_updated.emit({
                            'status': f'Starting training for {total_epochs} epochs...',
                            'progress': 0.0,
                            'elapsed_time': time.time() - self.start_time
                        })
                elif "Start training for" in line:
                    # Extract total epochs from overall training
                    import re
                    match = re.search(r'for (\d+) epochs', line)
                    if match:
                        total_epochs = int(match.group(1))
                        self.progress_updated.emit({
                            'status': f'Preparing to train for {total_epochs} total epochs...',
                            'elapsed_time': time.time() - self.start_time
                        })
                elif "Create training dataset" in line:
                    self.progress_updated.emit({
                        'status': "Creating training dataset...",
                        'elapsed_time': time.time() - self.start_time
                    })
                elif "Create model:" in line:
                    self.progress_updated.emit({
                        'status': "Initializing model architecture...",
                        'elapsed_time': time.time() - self.start_time
                    })
                elif "Number of parameters:" in line:
                    self.progress_updated.emit({
                        'status': "Model initialized, calculating parameters...",
                        'elapsed_time': time.time() - self.start_time
                    })
                elif "Flops profiler" in line:
                    self.progress_updated.emit({
                        'status': "Profiling model performance...",
                        'elapsed_time': time.time() - self.start_time
                    })
                elif "Epoch" in line and "/" in line:
                    # Parse epoch information
                    import re
                    # Match patterns like "Epoch: 1/30" or "Epoch 1/30"
                    epoch_match = re.search(r'Epoch[:\s]+(\d+)[/\s]+(\d+)', line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        total_epochs = int(epoch_match.group(2))
                        progress = current_epoch / total_epochs if total_epochs > 0 else 0
                        
                        self.progress_updated.emit({
                            'progress': progress,
                            'status': f"Training model... Epoch {current_epoch}/{total_epochs}",
                            'elapsed_time': time.time() - self.start_time
                        })
            else:
                # Original detailed parsing for dev mode
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
                    
                    # Calculate progress
                    if self.progress_data['total_epochs'] > 0:
                        progress = self.progress_data['epochs_completed'] / self.progress_data['total_epochs']
                        self.progress_data['progress'] = progress
                    
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
        
    def _verify_paths(self):
        """Verify all required paths exist"""
        gui_dir = Path(__file__).parent.parent.parent
        hinerv_root = gui_dir.parent
        
        paths_to_check = {
            "HiNeRV root": hinerv_root,
            "hinerv_main.py": hinerv_root / "hinerv_main.py",
            "cfgs/models": hinerv_root / "cfgs" / "models",
            "cfgs/train": hinerv_root / "cfgs" / "train",
            "Training config": hinerv_root / "cfgs" / "train" / "hinerv_1920x1080.txt",
            "Model config": hinerv_root / "cfgs" / "models" / "uvg-hinerv-s_1920x1080.txt"
        }
        
        for name, path in paths_to_check.items():
            if path.exists():
                self.logger.info(f"✓ {name}: {path}")
            else:
                self.logger.error(f"✗ {name}: {path} (NOT FOUND)")