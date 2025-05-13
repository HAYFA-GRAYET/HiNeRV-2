"""
System Monitor Module
Monitors system resources (CPU, RAM, GPU) during compression
"""

import os
import time
import logging
import threading
import psutil
from typing import Dict, List, Optional, Union

from PySide6.QtCore import QObject, Signal

try:
    import nvidia_ml_py3 as nvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    

class SystemMonitor(QObject):
    """Monitors system resources and emits updates periodically"""
    
    stats_updated = Signal(dict)  # Signal with current system stats
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize the system monitor
        
        Args:
            interval: Update interval in seconds
        """
        super().__init__()
        self.interval = interval
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.thread = None
        self.nvidia_initialized = False
        
        # Initialize NVIDIA monitoring if available
        if HAS_NVML:
            try:
                nvml.nvmlInit()
                self.nvidia_initialized = True
                self.logger.info("NVIDIA monitoring initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVIDIA monitoring: {e}")
    
    def start(self):
        """Start monitoring system resources"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info("System monitoring started")
    
    def stop(self):
        """Stop monitoring system resources"""
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        if self.nvidia_initialized:
            try:
                nvml.nvmlShutdown()
            except:
                pass
            
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system stats
                stats = self._collect_stats()
                
                # Emit signal with stats
                self.stats_updated.emit(stats)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}", exc_info=True)
                
            # Sleep for interval
            time.sleep(self.interval)
    
    def _collect_stats(self) -> Dict:
        """Collect current system statistics"""
        stats = {
            'timestamp': time.time(),
            'cpu': self._get_cpu_stats(),
            'memory': self._get_memory_stats(),
            'gpu': self._get_gpu_stats(),
        }
        return stats
    
    def _get_cpu_stats(self) -> Dict:
        """Get CPU usage statistics"""
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count(logical=True)
        
        # Get per-core usage
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        
        return {
            'usage': cpu_usage,
            'count': cpu_count,
            'per_cpu': per_cpu,
            'load_avg': self._get_load_average(),
        }
    
    def _get_load_average(self) -> List[float]:
        """Get system load average (1, 5, 15 min)"""
        try:
            return os.getloadavg()
        except:
            return [0.0, 0.0, 0.0]
    
    def _get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.total - memory.available,
            'percent': memory.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent,
        }
    
    def _get_gpu_stats(self) -> Union[Dict, None]:
        """Get GPU usage statistics (NVIDIA)"""
        if not self.nvidia_initialized:
            return None
            
        try:
            # Get device count
            device_count = nvml.nvmlDeviceGetCount()
            devices = []
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get device name
                name = nvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Get utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                # Get power usage
                try:
                    power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power_usage = 0
                    
                # Get clocks
                try:
                    sm_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                    mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                except:
                    sm_clock = 0
                    mem_clock = 0
                
                device_info = {
                    'index': i,
                    'name': name,
                    'utilization': {
                        'gpu': util.gpu,
                        'memory': util.memory,
                    },
                    'memory': {
                        'total': mem_info.total,
                        'used': mem_info.used,
                        'free': mem_info.free,
                        'percent': (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0,
                    },
                    'temperature': temp,
                    'power_usage': power_usage,
                    'clocks': {
                        'sm': sm_clock,
                        'memory': mem_clock,
                    }
                }
                
                devices.append(device_info)
            
            return {
                'count': device_count,
                'devices': devices,
            }
            
        except Exception as e:
            self.logger.warning(f"Error getting GPU stats: {e}")
            return None
    
    def get_free_gpu_memory(self, device_index: int = 0) -> int:
        """
        Get free GPU memory in bytes
        
        Args:
            device_index: GPU device index
            
        Returns:
            Free memory in bytes or 0 if error
        """
        if not self.nvidia_initialized:
            return 0
            
        try:
            # Try to get handle
            handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
            
            # Get memory info
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            return mem_info.free
            
        except Exception as e:
            self.logger.warning(f"Error getting GPU memory: {e}")
            return 0