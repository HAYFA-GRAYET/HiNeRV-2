#!/usr/bin/env python3
"""
Resource Guard Widget - Monitors and controls resource usage
"""

import os
import json
import logging
from typing import Dict, Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QComboBox,
    QPushButton, QProgressBar, QTextEdit, QTabWidget, QScrollArea,
    QGridLayout, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPixmap, QIcon
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
    nvml.nvmlInit()
except:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceGuardWidget(QWidget):
    """Widget for monitoring and controlling system resources"""
    
    limits_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_video_info = None
        self.monitoring_timer = QTimer()
        self.monitoring_timer.timeout.connect(self.update_monitoring)
        
        self.setup_ui()
        self.load_settings()
        
        # Start monitoring
        self.monitoring_timer.start(2000)  # Update every 2 seconds
    
    def setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Resource Limits Group
        limits_group = QGroupBox("Resource Limits")
        limits_layout = QFormLayout(limits_group)
        
        # Max VRAM Usage
        self.max_vram_slider = QSlider(Qt.Horizontal)
        self.max_vram_slider.setRange(10, 90)
        self.max_vram_slider.setValue(80)
        self.max_vram_slider.setTickPosition(QSlider.TicksBelow)
        self.max_vram_slider.setTickInterval(10)
        self.max_vram_slider.valueChanged.connect(self.on_limits_changed)
        
        vram_layout = QHBoxLayout()
        vram_layout.addWidget(self.max_vram_slider)
        self.max_vram_label = QLabel("80%")
        self.max_vram_label.setMinimumWidth(50)
        vram_layout.addWidget(self.max_vram_label)
        
        limits_layout.addRow("Max VRAM Usage:", vram_layout)
        
        # Frames per Batch
        self.frames_per_batch = QSpinBox()
        self.frames_per_batch.setRange(1, 1000)
        self.frames_per_batch.setValue(100)
        self.frames_per_batch.setSuffix(" frames")
        self.frames_per_batch.valueChanged.connect(self.on_limits_changed)
        limits_layout.addRow("Frames per Batch:", self.frames_per_batch)
        
        # Auto-scale Batch Size
        self.auto_scale_batch = QCheckBox("Auto-scale batch size based on VRAM")
        self.auto_scale_batch.setChecked(True)
        self.auto_scale_batch.stateChanged.connect(self.on_limits_changed)
        limits_layout.addRow(self.auto_scale_batch)
        
        # Max CPU Usage
        self.max_cpu_slider = QSlider(Qt.Horizontal)
        self.max_cpu_slider.setRange(10, 100)
        self.max_cpu_slider.setValue(80)
        self.max_cpu_slider.setTickPosition(QSlider.TicksBelow)
        self.max_cpu_slider.setTickInterval(10)
        self.max_cpu_slider.valueChanged.connect(self.on_limits_changed)
        
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(self.max_cpu_slider)
        self.max_cpu_label = QLabel("80%")
        self.max_cpu_label.setMinimumWidth(50)
        cpu_layout.addWidget(self.max_cpu_label)
        
        limits_layout.addRow("Max CPU Usage:", cpu_layout)
        
        # Max RAM Usage
        self.max_ram_slider = QSlider(Qt.Horizontal)
        self.max_ram_slider.setRange(10, 95)
        self.max_ram_slider.setValue(80)
        self.max_ram_slider.setTickPosition(QSlider.TicksBelow)
        self.max_ram_slider.setTickInterval(10)
        self.max_ram_slider.valueChanged.connect(self.on_limits_changed)
        
        ram_layout = QHBoxLayout()
        ram_layout.addWidget(self.max_ram_slider)
        self.max_ram_label = QLabel("80%")
        self.max_ram_label.setMinimumWidth(50)
        ram_layout.addWidget(self.max_ram_label)
        
        limits_layout.addRow("Max RAM Usage:", ram_layout)
        
        layout.addWidget(limits_group)
        
        # Current Usage Monitoring
        monitoring_group = QGroupBox("Current Usage")
        monitoring_layout = QVBoxLayout(monitoring_group)
        
        # Create monitoring tabs
        self.monitoring_tabs = QTabWidget()
        
        # Overview tab
        overview_tab = self.create_overview_tab()
        self.monitoring_tabs.addTab(overview_tab, "Overview")
        
        # Charts tab
        charts_tab = self.create_charts_tab()
        self.monitoring_tabs.addTab(charts_tab, "Charts")
        
        monitoring_layout.addWidget(self.monitoring_tabs)
        layout.addWidget(monitoring_group)
        
        # Frame Estimation Group
        estimation_group = QGroupBox("Video Analysis")
        estimation_layout = QFormLayout(estimation_group)
        
        self.total_frames_label = QLabel("Not calculated")
        estimation_layout.addRow("Total Frames:", self.total_frames_label)
        
        self.estimated_vram_label = QLabel("Not calculated")
        estimation_layout.addRow("Estimated VRAM:", self.estimated_vram_label)
        
        self.batch_size_rec_label = QLabel("Not calculated")
        estimation_layout.addRow("Recommended Batch Size:", self.batch_size_rec_label)
        
        layout.addWidget(estimation_group)
        
        # Connect slider events to update labels
        self.max_vram_slider.valueChanged.connect(
            lambda v: self.max_vram_label.setText(f"{v}%")
        )
        self.max_cpu_slider.valueChanged.connect(
            lambda v: self.max_cpu_label.setText(f"{v}%")
        )
        self.max_ram_slider.valueChanged.connect(
            lambda v: self.max_ram_label.setText(f"{v}%")
        )
    
    def create_overview_tab(self) -> QWidget:
        """Create the overview monitoring tab"""
        tab = QWidget()
        layout = QGridLayout(tab)
        
        # GPU Monitoring
        if NVML_AVAILABLE:
            gpu_frame = QFrame()
            gpu_frame.setFrameStyle(QFrame.StyledPanel)
            gpu_layout = QVBoxLayout(gpu_frame)
            
            gpu_title = QLabel("GPU Status")
            gpu_title.setFont(QFont("Arial", 12, QFont.Bold))
            gpu_layout.addWidget(gpu_title)
            
            self.gpu_name_label = QLabel("Detecting...")
            gpu_layout.addWidget(self.gpu_name_label)
            
            self.gpu_temp_label = QLabel("Temperature: --°C")
            gpu_layout.addWidget(self.gpu_temp_label)
            
            self.vram_progress = QProgressBar()
            self.vram_progress.setFormat("VRAM: %p%")
            gpu_layout.addWidget(self.vram_progress)
            
            self.gpu_util_progress = QProgressBar()
            self.gpu_util_progress.setFormat("GPU: %p%")
            gpu_layout.addWidget(self.gpu_util_progress)
            
            layout.addWidget(gpu_frame, 0, 0)
        
        # CPU Monitoring
        cpu_frame = QFrame()
        cpu_frame.setFrameStyle(QFrame.StyledPanel)
        cpu_layout = QVBoxLayout(cpu_frame)
        
        cpu_title = QLabel("CPU Status")
        cpu_title.setFont(QFont("Arial", 12, QFont.Bold))
        cpu_layout.addWidget(cpu_title)
        
        self.cpu_usage_label = QLabel("CPU Usage: --%")
        cpu_layout.addWidget(self.cpu_usage_label)
        
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setFormat("CPU: %p%")
        cpu_layout.addWidget(self.cpu_progress)
        
        layout.addWidget(cpu_frame, 0, 1)
        
        # RAM Monitoring
        ram_frame = QFrame()
        ram_frame.setFrameStyle(QFrame.StyledPanel)
        ram_layout = QVBoxLayout(ram_frame)
        
        ram_title = QLabel("RAM Status")
        ram_title.setFont(QFont("Arial", 12, QFont.Bold))
        ram_layout.addWidget(ram_title)
        
        self.ram_usage_label = QLabel("RAM Usage: --%")
        ram_layout.addWidget(self.ram_usage_label)
        
        self.ram_progress = QProgressBar()
        self.ram_progress.setFormat("RAM: %p%")
        ram_layout.addWidget(self.ram_progress)
        
        layout.addWidget(ram_frame, 1, 0)
        
        # Disk Monitoring
        disk_frame = QFrame()
        disk_frame.setFrameStyle(QFrame.StyledPanel)
        disk_layout = QVBoxLayout(disk_frame)
        
        disk_title = QLabel("Disk Status")
        disk_title.setFont(QFont("Arial", 12, QFont.Bold))
        disk_layout.addWidget(disk_title)
        
        self.disk_usage_label = QLabel("Disk Usage: --%")
        disk_layout.addWidget(self.disk_usage_label)
        
        self.disk_progress = QProgressBar()
        self.disk_progress.setFormat("Disk: %p%")
        disk_layout.addWidget(self.disk_progress)
        
        layout.addWidget(disk_frame, 1, 1)
        
        return tab
    
    def create_charts_tab(self) -> QWidget:
        """Create the charts monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create charts for different metrics
        self.usage_charts = {}
        
        # GPU Chart
        if NVML_AVAILABLE:
            gpu_chart = QChart()
            gpu_chart.setTitle("GPU Usage")
            self.gpu_series = QLineSeries()
            self.gpu_series.setName("GPU Utilization")
            gpu_chart.addSeries(self.gpu_series)
            
            gpu_chart_view = QChartView(gpu_chart)
            gpu_chart_view.setRenderHint(gpu_chart_view.Antialiasing)
            layout.addWidget(gpu_chart_view)
            
            self.usage_charts["gpu"] = {
                "chart": gpu_chart,
                "series": self.gpu_series,
                "data": []
            }
        
        # CPU Chart
        cpu_chart = QChart()
        cpu_chart.setTitle("CPU Usage")
        self.cpu_series = QLineSeries()
        self.cpu_series.setName("CPU Utilization")
        cpu_chart.addSeries(self.cpu_series)
        
        cpu_chart_view = QChartView(cpu_chart)
        cpu_chart_view.setRenderHint(cpu_chart_view.Antialiasing)
        layout.addWidget(cpu_chart_view)
        
        self.usage_charts["cpu"] = {
            "chart": cpu_chart,
            "series": self.cpu_series,
            "data": []
        }
        
        # RAM Chart
        ram_chart = QChart()
        ram_chart.setTitle("RAM Usage")
        self.ram_series = QLineSeries()
        self.ram_series.setName("RAM Utilization")
        ram_chart.addSeries(self.ram_series)
        
        ram_chart_view = QChartView(ram_chart)
        ram_chart_view.setRenderHint(ram_chart_view.Antialiasing)
        layout.addWidget(ram_chart_view)
        
        self.usage_charts["ram"] = {
            "chart": ram_chart,
            "series": self.ram_series,
            "data": []
        }
        
        return tab
    
    def update_monitoring(self):
        """Update resource monitoring information"""
        try:
            import psutil
            import time
            
            # Get timestamp
            timestamp = time.time()
            
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage_label.setText(f"CPU Usage: {cpu_percent:.1f}%")
            self.cpu_progress.setValue(int(cpu_percent))
            
            # Update CPU chart
            if "cpu" in self.usage_charts:
                self.update_chart_data("cpu", timestamp, cpu_percent)
            
            # RAM Usage
            ram_info = psutil.virtual_memory()
            ram_percent = ram_info.percent
            self.ram_usage_label.setText(f"RAM Usage: {ram_percent:.1f}%")
            self.ram_progress.setValue(int(ram_percent))
            
            # Update RAM chart
            if "ram" in self.usage_charts:
                self.update_chart_data("ram", timestamp, ram_percent)
            
            # Disk Usage
            disk_info = psutil.disk_usage('/')
            disk_percent = (disk_info.used / disk_info.total) * 100
            self.disk_usage_label.setText(f"Disk Usage: {disk_percent:.1f}%")
            self.disk_progress.setValue(int(disk_percent))
            
            # GPU Usage (if available)
            if NVML_AVAILABLE:
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # GPU Name
                    gpu_name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    self.gpu_name_label.setText(gpu_name)
                    
                    # GPU Temperature
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    self.gpu_temp_label.setText(f"Temperature: {temp}°C")
                    
                    # VRAM Usage
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_percent = (mem_info.used / mem_info.total) * 100
                    self.vram_progress.setValue(int(vram_percent))
                    
                    # GPU Utilization
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = util.gpu
                    self.gpu_util_progress.setValue(gpu_percent)
                    
                    # Update GPU chart
                    if "gpu" in self.usage_charts:
                        self.update_chart_data("gpu", timestamp, gpu_percent)
                        
                except nvml.NVMLError as e:
                    logger.warning(f"GPU monitoring error: {e}")
            
        except Exception as e:
            logger.error(f"Error updating monitoring: {e}")
    
    def update_chart_data(self, chart_type: str, timestamp: float, value: float):
        """Update chart data with new values"""
        if chart_type not in self.usage_charts:
            return
        
        chart_info = self.usage_charts[chart_type]
        data = chart_info["data"]
        series = chart_info["series"]
        
        # Add new data point
        data.append((timestamp, value))
        
        # Keep only last 60 seconds of data
        cutoff_time = timestamp - 60
        data = [(t, v) for t, v in data if t > cutoff_time]
        chart_info["data"] = data
        
        # Update series
        series.clear()
        for t, v in data:
            series.append(t - timestamp, v)  # Relative time
        
        # Update chart axes
        chart = chart_info["chart"]
        chart.createDefaultAxes()
        
        # Set y-axis range
        axes = chart.axes(Qt.Vertical)
        if axes:
            axes[0].setRange(0, 100)
        
        # Set x-axis range
        axes = chart.axes(Qt.Horizontal)
        if axes:
            axes[0].setRange(-60, 0)
            axes[0].setTitleText("Time (seconds ago)")
    
    def update_for_video(self, video_info: Dict):
        """Update estimates based on video information"""
        self.current_video_info = video_info
        
        # Calculate total frames
        fps = video_info.get('fps', 30)
        duration = video_info.get('duration', 0)
        total_frames = int(fps * duration)
        self.total_frames_label.setText(f"{total_frames:,}")
        
        # Estimate VRAM usage
        width = video_info.get('width', 1920)
        height = video_info.get('height', 1080)
        
        # Rough estimation: depends on model complexity
        # Base estimate: ~1GB per 100 frames at 1080p
        frame_size_mb = (width * height * 3) / (1024 * 1024)  # RGB values
        batch_size = self.frames_per_batch.value()
        estimated_vram_mb = frame_size_mb * batch_size * 2  # 2x for gradients
        
        self.estimated_vram_label.setText(f"{estimated_vram_mb:.0f} MB")
        
        # Recommend batch size based on VRAM limit
        if NVML_AVAILABLE:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                total_vram_gb = mem_info.total / (1024 ** 3)
                max_vram_pct = self.max_vram_slider.value()
                available_vram_gb = total_vram_gb * (max_vram_pct / 100)
                
                # Conservative estimate: 1GB per 50 frames at 1080p
                frames_per_gb = 50 * (1920 * 1080) / (width * height)
                recommended_batch = int(available_vram_gb * frames_per_gb)
                recommended_batch = max(1, min(recommended_batch, 200))  # Clamp to reasonable range
                
                self.batch_size_rec_label.setText(f"{recommended_batch}")
            except:
                self.batch_size_rec_label.setText("Unable to calculate")
    
    def get_limits(self) -> Dict:
        """Get current resource limits"""
        return {
            'max_vram_percent': self.max_vram_slider.value(),
            'frames_per_batch': self.frames_per_batch.value(),
            'auto_scale_batch': self.auto_scale_batch.isChecked(),
            'max_cpu_percent': self.max_cpu_slider.value(),
            'max_ram_percent': self.max_ram_slider.value()
        }
    
    def on_limits_changed(self):
        """Handle changes to resource limits"""
        self.limits_changed.emit(self.get_limits())
        self.save_settings()
    
    def save_settings(self):
        """Save current settings"""
        settings = {
            'max_vram_percent': self.max_vram_slider.value(),
            'frames_per_batch': self.frames_per_batch.value(),
            'auto_scale_batch': self.auto_scale_batch.isChecked(),
            'max_cpu_percent': self.max_cpu_slider.value(),
            'max_ram_percent': self.max_ram_slider.value()
        }
        
        settings_dir = Path.home() / '.hinerv_gui'
        settings_dir.mkdir(exist_ok=True)
        
        with open(settings_dir / 'resource_settings.json', 'w') as f:
            json.dump(settings, f, indent=2)
    
    def load_settings(self):
        """Load saved settings"""
        settings_file = Path.home() / '.hinerv_gui' / 'resource_settings.json'
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                
                self.max_vram_slider.setValue(settings.get('max_vram_percent', 80))
                self.frames_per_batch.setValue(settings.get('frames_per_batch', 100))
                self.auto_scale_batch.setChecked(settings.get('auto_scale_batch', True))
                self.max_cpu_slider.setValue(settings.get('max_cpu_percent', 80))
                self.max_ram_slider.setValue(settings.get('max_ram_percent', 80))
            except Exception as e:
                logger.warning(f"Failed to load resource settings: {e}")