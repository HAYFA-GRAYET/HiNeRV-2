"""
Video Processor Module
Handles preprocessing of input videos and postprocessing of compressed outputs
"""

import os
import subprocess
import logging
import re
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import concurrent.futures

from PySide6.QtCore import QObject, Signal, QThread, QProcess


class VideoProcessor(QObject):
    """Handles video preprocessing (splitting) and postprocessing (joining)"""
    
    # Signals for progress notifications
    progress_updated = Signal(int, str)  # percentage, status
    error_occurred = Signal(str)
    preprocessing_complete = Signal(dict)  # video info
    postprocessing_complete = Signal(dict)  # result info
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.process = None
        self.abort_requested = False
        
    def preprocess_video(self, video_path: str, output_dir: str, max_frames: Optional[int] = None) -> None:
        """
        Split a video into frames for processing
        
        Args:
            video_path: Path to the input video
            output_dir: Directory to save extracted frames
            max_frames: Maximum number of frames to extract (None for all)
        """
        try:
            # Create output directory for frames
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Get video information
            video_info = self.get_video_info(video_path)
            self.logger.info(f"Processing video: {video_info}")
            
            # Calculate frame limit
            total_frames = int(float(video_info.get('duration_secs', 0)) * 
                               float(video_info.get('fps', 0)))
            
            if max_frames is not None and max_frames > 0:
                frame_limit = min(total_frames, max_frames)
            else:
                frame_limit = total_frames
                
            # Update video_info with frame limit
            video_info['frame_limit'] = frame_limit
            video_info['original_total_frames'] = total_frames
            
            # Extract frames using ffmpeg
            self.extract_frames(video_path, frames_dir, frame_limit)
            
            # Save video info
            video_info['frames_dir'] = frames_dir
            video_info['extracted_frames'] = len([f for f in os.listdir(frames_dir) 
                                                if f.endswith('.png')])
            
            with open(os.path.join(output_dir, "video_info.json"), "w") as f:
                json.dump(video_info, f, indent=2)
            
            # Signal completion
            self.preprocessing_complete.emit(video_info)
            
        except Exception as e:
            error_msg = f"Video preprocessing failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
    
    def extract_frames(self, video_path: str, output_dir: str, frame_limit: Optional[int] = None) -> bool:
        """Extract frames from video"""
        try:
            # Ensure frame_limit is positive
            if frame_limit is not None and frame_limit <= 0:
                frame_limit = None
                
            # Build FFmpeg command
            frame_pattern = os.path.join(output_dir, "frame_%06d.png")
            if frame_limit:
                cmd = [
                    "ffmpeg", "-i", video_path,
                    "-vf", f"select=between(n\\,0\\,{frame_limit-1})",
                    "-vsync", "0", "-q:v", "0",
                    frame_pattern
                ]
            else:
                cmd = [
                    "ffmpeg", "-i", video_path,
                    "-vf", "select=between(n\\,0\\,-1)",
                    "-vsync", "0", "-q:v", "0",
                    frame_pattern
                ]
                
            # Run FFmpeg command
            logging.info(f"Running FFmpeg command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            current_frame = 0
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                    
                # Update progress
                if frame_limit:
                    progress = min(int(current_frame * 100 / frame_limit), 100)
                else:
                    progress = 0
                current_frame += 1
                
            return_code = process.wait()
            return return_code == 0
            
        except Exception as e:
            logging.error(f"Error extracting frames: {e}")
            return False
    
    def postprocess_video(self, 
                         compressed_dir: str, 
                         output_path: str,
                         original_video_path: str,
                         fps: float) -> None:
        """
        Join compressed frames back into a video
        
        Args:
            compressed_dir: Directory containing the compressed frames
            output_path: Path for the output video file
            original_video_path: Path to the original video (for audio)
            fps: Frames per second for the output video
        """
        try:
            # Check for compressed frames
            frames_pattern = os.path.join(compressed_dir, "*.png")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Step 1: Join frames into video (without audio)
            temp_video = os.path.join(os.path.dirname(output_path), "temp_output.mp4")
            
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-pattern_type", "glob",
                "-i", frames_pattern,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                temp_video
            ]
            
            self.logger.info(f"Running FFmpeg join command: {' '.join(ffmpeg_cmd)}")
            self.progress_updated.emit(0, "Joining frames into video...")
            
            # Run ffmpeg process
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for process to finish
            process.wait()
            
            if process.returncode != 0:
                error_output = process.stderr.read()
                raise Exception(f"FFmpeg frame joining failed: {error_output}")
                
            self.progress_updated.emit(50, "Adding audio from original...")
            
            # Step 2: Copy audio from original (if it exists)
            try:
                # Check if original has audio
                probe_cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=codec_type",
                    "-of", "json",
                    original_video_path
                ]
                
                probe_result = subprocess.run(
                    probe_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                has_audio = False
                if probe_result.returncode == 0:
                    probe_data = json.loads(probe_result.stdout)
                    has_audio = 'streams' in probe_data and len(probe_data['streams']) > 0
                
                if has_audio:
                    # Copy audio from original
                    final_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i", temp_video,
                        "-i", original_video_path,
                        "-c:v", "copy",
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-shortest",
                        output_path
                    ]
                    
                    subprocess.run(
                        final_cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                else:
                    # Just rename the temp video
                    shutil.move(temp_video, output_path)
            except Exception as e:
                self.logger.warning(f"Could not add audio: {str(e)}")
                # Fallback to temp video
                if os.path.exists(temp_video):
                    shutil.move(temp_video, output_path)
            
            # Clean up temp file if it still exists
            if os.path.exists(temp_video):
                os.remove(temp_video)
                
            # Calculate output video size and stats
            output_size = os.path.getsize(output_path)
            
            # Get video info for original and compressed
            original_info = self.get_video_info(original_video_path)
            compressed_info = self.get_video_info(output_path)
            
            # Calculate compression ratio and bitrate
            original_size = int(original_info.get('size_bytes', 0))
            compression_ratio = original_size / output_size if output_size > 0 else 0
            
            result_info = {
                'output_path': output_path,
                'output_size': output_size,
                'original_size': original_size,
                'compression_ratio': compression_ratio,
                'original_bitrate': original_info.get('bitrate', 0),
                'compressed_bitrate': compressed_info.get('bitrate', 0),
            }
            
            self.progress_updated.emit(100, "Video processing complete!")
            self.postprocessing_complete.emit(result_info)
            
        except Exception as e:
            error_msg = f"Video postprocessing failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata using ffprobe"""
        try:
            # Run ffprobe to get video info
            ffprobe_cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(
                ffprobe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True
            )
            
            probe_data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream is None:
                raise ValueError("No video stream found")
                
            # Extract relevant information
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            # Calculate frame rate
            frame_rate = video_stream.get('r_frame_rate', '0/0')
            if '/' in frame_rate:
                num, den = map(int, frame_rate.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(frame_rate)
            
            # Get duration and size
            format_info = probe_data.get('format', {})
            duration_secs = float(format_info.get('duration', 0))
            size_bytes = int(format_info.get('size', 0))
            bitrate = int(format_info.get('bit_rate', 0))
            
            # Return compiled info
            video_info = {
                'path': video_path,
                'width': width,
                'height': height,
                'fps': fps,
                'duration_secs': duration_secs,
                'duration_str': self.format_duration(duration_secs),
                'size_bytes': size_bytes,
                'size_str': self.format_size(size_bytes),
                'bitrate': bitrate,
                'bitrate_str': f"{bitrate/1000:.2f} Kbps"
            }
            
            return video_info
            
        except Exception as e:
            self.logger.error(f"Failed to get video info: {str(e)}", exc_info=True)
            return {
                'path': video_path,
                'error': str(e)
            }
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in bytes to human-readable string"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.2f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"
    
    def abort(self):
        """Abort current processing"""
        self.abort_requested = True
        
        if self.process is not None:
            try:
                self.process.terminate()
            except:
                pass


class VideoPreprocessThread(QThread):
    """Thread for video preprocessing"""
    
    def __init__(self, processor, video_path, output_dir, max_frames=None):
        super().__init__()
        self.processor = processor
        self.video_path = video_path
        self.output_dir = output_dir
        self.max_frames = max_frames
        
    def run(self):
        """Run the preprocessing in a separate thread"""
        self.processor.preprocess_video(self.video_path, self.output_dir, self.max_frames)


class VideoPostprocessThread(QThread):
    """Thread for video postprocessing"""
    
    def __init__(self, processor, compressed_dir, output_path, original_video_path, fps):
        super().__init__()
        self.processor = processor
        self.compressed_dir = compressed_dir
        self.output_path = output_path
        self.original_video_path = original_video_path
        self.fps = fps
        
    def run(self):
        """Run the postprocessing in a separate thread"""
        self.processor.postprocess_video(
            self.compressed_dir, 
            self.output_path,
            self.original_video_path,
            self.fps
        )