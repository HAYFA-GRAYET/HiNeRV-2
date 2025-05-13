"""
HiNeRV GUI - System Utilities
System information and dependency checking functions
"""

import os
import sys
import platform
import subprocess
import shutil
import logging
import psutil
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Union


# Set up module logger
logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, str]:
    """
    Get system information including OS, CPU, RAM, Python version.
    
    Returns:
        Dict with system information
    """
    system_info = {
        'os_name': platform.system(),
        'os_version': platform.version(),
        'os_release': platform.release(),
        'python_version': platform.python_version(),
        'cpu_name': get_cpu_name(),
        'cpu_count': str(psutil.cpu_count(logical=True)),
        'cpu_usage': str(int(psutil.cpu_percent())),
        'ram_total': format_filesize(psutil.virtual_memory().total),
        'ram_available': format_filesize(psutil.virtual_memory().available),
        'ram_usage': str(int(psutil.virtual_memory().percent)),
    }
    
    # Add GPU information if available
    gpu_info = get_gpu_info()
    if gpu_info:
        system_info.update(gpu_info)
    
    return system_info


def get_cpu_name() -> str:
    """
    Get CPU name/model.
    
    Returns:
        CPU name as string
    """
    if platform.system() == "Windows":
        return platform.processor()
    
    elif platform.system() == "Darwin":
        try:
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command.split()).strip().decode()
        except Exception:
            return platform.processor()
    
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                info = f.readlines()
            
            for line in info:
                if "model name" in line:
                    return line.split(":")[1].strip()
            return platform.processor()
        except Exception:
            return platform.processor()
    
    return platform.processor()


def get_gpu_info() -> Dict[str, str]:
    """
    Get NVIDIA GPU information using nvidia-smi.
    
    Returns:
        Dict with GPU information, or empty dict if not available
    """
    gpu_info = {}
    
    try:
        # Check if nvidia-smi is available
        if shutil.which("nvidia-smi") is None:
            return {}
        
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            # Parse the output
            output = result.stdout.strip().split(",")
            if len(output) >= 4:
                gpu_info["gpu_name"] = output[0].strip()
                gpu_info["gpu_memory_total"] = f"{float(output[1].strip())/1024:.2f} GB"
                gpu_info["gpu_memory_used"] = f"{float(output[2].strip())/1024:.2f} GB"
                gpu_info["gpu_utilization"] = f"{output[3].strip()}%"
        
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        logger.warning(f"Failed to get GPU info: {e}")
    
    return gpu_info


def get_gpu_memory_info() -> Optional[Dict[str, int]]:
    """
    Get available and total GPU memory in bytes.
    
    Returns:
        Dict with available and total GPU memory, or None if not available
    """
    try:
        # Check if nvidia-smi is available
        if shutil.which("nvidia-smi") is None:
            return None
        
        # Run nvidia-smi to get GPU memory info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            # Parse the output
            output = result.stdout.strip().split(",")
            if len(output) >= 2:
                free_mb = int(output[0].strip())
                total_mb = int(output[1].strip())
                
                return {
                    "free_bytes": free_mb * 1024 * 1024,
                    "total_bytes": total_mb * 1024 * 1024,
                    "free_mb": free_mb,
                    "total_mb": total_mb
                }
        
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
    
    return None


def check_dependencies() -> List[str]:
    """
    Check for required dependencies.
    
    Returns:
        List of missing dependencies or issues
    """
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 9):
        issues.append("Python 3.9 or higher is required.")
    
    # Check for required Python packages
    required_packages = [
        "torch", "torchvision", "numpy", "pillow", "ffmpeg-python",
        "accelerate", "timm", "pytorch-msssim"
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
            issues.append("CUDA is not available. GPU acceleration is required for HiNeRV.")
    except ImportError:
        pass  # Already reported above
    
    # Check for external programs
    external_dependencies = ["ffmpeg"]
    for program in external_dependencies:
        if shutil.which(program) is None:
            issues.append(f"External program '{program}' is not found in PATH")
    
    return issues


def check_cuda_capabilities() -> Dict[str, Union[bool, str]]:
    """
    Check CUDA capabilities and installation.
    
    Returns:
        Dict with CUDA info and status
    """
    cuda_info = {
        "available": False,
        "version": "N/A",
        "device_count": 0,
        "device_names": [],
        "error": None
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            cuda_info["available"] = True
            cuda_info["version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            cuda_info["error"] = "CUDA is not available in PyTorch"
            
    except ImportError:
        cuda_info["error"] = "PyTorch is not installed"
    except Exception as e:
        cuda_info["error"] = str(e)
    
    return cuda_info


def get_optimal_batch_size(video_height: int, video_width: int, max_vram_mb: int = None) -> int:
    """
    Calculate optimal batch size based on available GPU memory and video resolution.
    
    Args:
        video_height: Video height in pixels
        video_width: Video width in pixels
        max_vram_mb: Maximum VRAM to use in MB, or None to auto-detect
    
    Returns:
        Recommended batch size
    """
    # Default conservative batch size if we can't determine it
    default_batch_size = 1
    
    try:
        # Get available GPU memory
        if max_vram_mb is None:
            gpu_info = get_gpu_memory_info()
            if not gpu_info:
                return default_batch_size
            
            # Use 70% of available memory
            available_vram_mb = int(gpu_info["free_mb"] * 0.7)
        else:
            available_vram_mb = max_vram_mb
        
        # Estimate memory per frame (very rough approximation)
        # For HiNeRV with typical settings, each 1080p frame might need ~200-400MB
        # Adjust based on resolution compared to 1080p
        pixels_ratio = (video_height * video_width) / (1080 * 1920)
        mb_per_frame = 300 * pixels_ratio  # Base estimate for 1080p
        
        # Calculate batch size
        estimated_batch_size = int(available_vram_mb / mb_per_frame)
        
        # Ensure minimum of 1, maximum of 32
        batch_size = max(1, min(32, estimated_batch_size))
        
        logger.info(f"Estimated optimal batch size: {batch_size} " +
                   f"(available VRAM: {available_vram_mb}MB, " +
                   f"est. memory per frame: {mb_per_frame:.1f}MB)")
        
        return batch_size
        
    except Exception as e:
        logger.warning(f"Failed to determine optimal batch size: {e}")
        return default_batch_size


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0:
        return "N/A"
    
    td = timedelta(seconds=seconds)
    
    # Format differently based on duration length
    if td.total_seconds() < 60:
        return f"{td.seconds % 60}s"
    elif td.total_seconds() < 3600:
        return f"{td.seconds // 60}m {td.seconds % 60}s"
    elif td.days == 0:
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        return f"{td.days}d {hours}h {minutes}m"


def format_filesize(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes < 0:
        return "N/A"
    
    # Define size units
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    
    # Calculate appropriate unit
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1
    
    # Format with appropriate precision
    if i == 0:
        return f"{size_bytes:.0f} {units[i]}"
    else:
        return f"{size_bytes:.2f} {units[i]}"


def get_ffmpeg_version() -> Optional[str]:
    """
    Get FFmpeg version.
    
    Returns:
        FFmpeg version string or None if not found
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            # Extract version number from first line
            first_line = result.stdout.split('\n')[0]
            if 'ffmpeg version' in first_line:
                return first_line.split('ffmpeg version')[1].strip()
        return None
    except Exception:
        return None


def find_hinerv_root() -> Optional[Path]:
    """
    Find HiNeRV repository root path.
    Searches in current and parent directories for indicators of HiNeRV repo.
    
    Returns:
        Path to HiNeRV root, or None if not found
    """
    # Start with current directory
    current_dir = Path.cwd()
    
    # Files/dirs that would indicate HiNeRV root
    indicators = [
        "hinerv_compress.py",
        "hinerv_decompress.py",
        "cfgs/models",
        "cfgs/train"
    ]
    
    # Check if current dir is HiNeRV root
    if all(Path(current_dir / ind).exists() for ind in indicators):
        return current_dir
    
    # If not, check parent directories (up to 5 levels up)
    for _ in range(5):
        if current_dir.parent == current_dir:
            # Reached filesystem root
            break
        
        current_dir = current_dir.parent
        if all(Path(current_dir / ind).exists() for ind in indicators):
            return current_dir
    
    # If still not found, try to find in common locations
    common_locations = [
        Path.home() / "HiNeRV",
        Path.home() / "git" / "HiNeRV",
        Path.home() / "projects" / "HiNeRV",
        Path.home() / "Documents" / "HiNeRV",
    ]
    
    for location in common_locations:
        if location.exists() and all(Path(location / ind).exists() for ind in indicators):
            return location
    
    # Not found
    return None


def get_conda_envs() -> List[str]:
    """
    Get list of available conda environments.
    
    Returns:
        List of environment names
    """
    try:
        # Check for conda executable
        conda_executable = shutil.which("conda")
        if not conda_executable:
            return []
        
        # Run conda env list
        result = subprocess.run(
            [conda_executable, "env", "list"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            # Parse output to extract environment names
            envs = []
            for line in result.stdout.split('\n'):
                if line and not line.startswith('#'):
                    env_name = line.split()[0]
                    if env_name != "*":  # Skip the "*" marker for active env
                        envs.append(env_name)
            return envs
        
        return []
    
    except Exception as e:
        logger.warning(f"Failed to get conda environments: {e}")
        return []