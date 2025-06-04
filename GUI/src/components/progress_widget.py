"""
Progress monitoring widget for HiNeRV compression
Shows real-time training progress, system stats, and logs
"""

import os
import json
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit,
    QScrollArea, QFrame, QGroupBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem,QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QPixmap, QColor
from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
)

from ..utils import format_duration, format_filesize


class SystemStatsWidget(QWidget):
    """Widget for displaying real-time system statistics"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Data storage for charts
        self.max_points = 100
        self.cpu_data = deque(maxlen=self.max_points)
        self.ram_data = deque(maxlen=self.max_points)
        self.gpu_data = deque(maxlen=self.max_points)
        self.vram_data = deque(maxlen=self.max_points)
        self.timestamp_data = deque(maxlen=self.max_points)
    
    def setup_ui(self):
        """Initialize the system stats UI"""
        layout = QVBoxLayout(self)
        
        # Current stats labels
        stats_group = QGroupBox("System Status")
        stats_layout = QGridLayout(stats_group)
        
        # CPU
        stats_layout.addWidget(QLabel("CPU:"), 0, 0)
        self.cpu_label = QLabel("0%")
        self.cpu_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        stats_layout.addWidget(self.cpu_label, 0, 1)
        
        # RAM
        stats_layout.addWidget(QLabel("RAM:"), 0, 2)
        self.ram_label = QLabel("0%")
        self.ram_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        stats_layout.addWidget(self.ram_label, 0, 3)
        
        # GPU
        stats_layout.addWidget(QLabel("GPU:"), 1, 0)
        self.gpu_label = QLabel("N/A")
        self.gpu_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        stats_layout.addWidget(self.gpu_label, 1, 1)
        
        # VRAM
        stats_layout.addWidget(QLabel("VRAM:"), 1, 2)
        self.vram_label = QLabel("N/A")
        self.vram_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
        stats_layout.addWidget(self.vram_label, 1, 3)
        
        layout.addWidget(stats_group)
        
        # Charts
        charts_group = QGroupBox("Performance Charts")
        charts_layout = QHBoxLayout(charts_group)
        
        # CPU/RAM Chart
        self.cpu_ram_chart = QChart()
        self.cpu_ram_chart.setTitle("CPU & RAM Usage (%)")
        self.cpu_ram_chart.setAnimationOptions(QChart.NoAnimation)
        
        self.cpu_series = QLineSeries()
        self.cpu_series.setName("CPU")
        self.cpu_series.setColor(QColor("#4CAF50"))
        
        self.ram_series = QLineSeries()
        self.ram_series.setName("RAM")
        self.ram_series.setColor(QColor("#2196F3"))
        
        self.cpu_ram_chart.addSeries(self.cpu_series)
        self.cpu_ram_chart.addSeries(self.ram_series)
        
        # Set up axes
        self.cpu_ram_axis_x = QValueAxis()
        self.cpu_ram_axis_x.setRange(0, 60)  # 60 seconds
        self.cpu_ram_axis_x.setTitleText("Time (seconds)")
        
        self.cpu_ram_axis_y = QValueAxis()
        self.cpu_ram_axis_y.setRange(0, 100)
        self.cpu_ram_axis_y.setTitleText("Usage (%)")
        
        self.cpu_ram_chart.addAxis(self.cpu_ram_axis_x, Qt.AlignBottom)
        self.cpu_ram_chart.addAxis(self.cpu_ram_axis_y, Qt.AlignLeft)
        
        self.cpu_series.attachAxis(self.cpu_ram_axis_x)
        self.cpu_series.attachAxis(self.cpu_ram_axis_y)
        self.ram_series.attachAxis(self.cpu_ram_axis_x)
        self.ram_series.attachAxis(self.cpu_ram_axis_y)
        
        cpu_ram_view = QChartView(self.cpu_ram_chart)
        charts_layout.addWidget(cpu_ram_view)
        
        # GPU/VRAM Chart
        self.gpu_vram_chart = QChart()
        self.gpu_vram_chart.setTitle("GPU & VRAM Usage (%)")
        self.gpu_vram_chart.setAnimationOptions(QChart.NoAnimation)
        
        self.gpu_series = QLineSeries()
        self.gpu_series.setName("GPU")
        self.gpu_series.setColor(QColor("#FF9800"))
        
        self.vram_series = QLineSeries()
        self.vram_series.setName("VRAM")
        self.vram_series.setColor(QColor("#9C27B0"))
        
        self.gpu_vram_chart.addSeries(self.gpu_series)
        self.gpu_vram_chart.addSeries(self.vram_series)
        
        # Set up axes
        self.gpu_vram_axis_x = QValueAxis()
        self.gpu_vram_axis_x.setRange(0, 60)  # 60 seconds
        self.gpu_vram_axis_x.setTitleText("Time (seconds)")
        
        self.gpu_vram_axis_y = QValueAxis()
        self.gpu_vram_axis_y.setRange(0, 100)
        self.gpu_vram_axis_y.setTitleText("Usage (%)")
        
        self.gpu_vram_chart.addAxis(self.gpu_vram_axis_x, Qt.AlignBottom)
        self.gpu_vram_chart.addAxis(self.gpu_vram_axis_y, Qt.AlignLeft)
        
        self.gpu_series.attachAxis(self.gpu_vram_axis_x)
        self.gpu_series.attachAxis(self.gpu_vram_axis_y)
        self.vram_series.attachAxis(self.gpu_vram_axis_x)
        self.vram_series.attachAxis(self.gpu_vram_axis_y)
        
        gpu_vram_view = QChartView(self.gpu_vram_chart)
        charts_layout.addWidget(gpu_vram_view)
        
        layout.addWidget(charts_group)
    
    def update_stats(self, stats: Dict):
        """Update system statistics"""
        current_time = time.time()
        
        # Update labels
        self.cpu_label.setText(f"{stats.get('cpu_usage', 0):.1f}%")
        self.ram_label.setText(f"{stats.get('ram_usage', 0):.1f}%")
        
        if 'gpu_usage' in stats:
            self.gpu_label.setText(f"{stats['gpu_usage']:.1f}%")
        if 'vram_usage' in stats:
            self.vram_label.setText(f"{stats['vram_usage']:.1f}%")
        
        # Add to data queues
        self.timestamp_data.append(current_time)
        self.cpu_data.append(stats.get('cpu_usage', 0))
        self.ram_data.append(stats.get('ram_usage', 0))
        self.gpu_data.append(stats.get('gpu_usage', 0))
        self.vram_data.append(stats.get('vram_usage', 0))
        
        # Update charts
        self.update_charts()
    
    def update_charts(self):
        """Update the performance charts"""
        if not self.timestamp_data:
            return
        
        # Calculate relative timestamps (seconds from start)
        start_time = self.timestamp_data[0]
        relative_times = [(t - start_time) for t in self.timestamp_data]
        
        # Update CPU/RAM chart
        self.cpu_series.clear()
        self.ram_series.clear()
        
        for i, time_val in enumerate(relative_times):
            self.cpu_series.append(time_val, self.cpu_data[i])
            self.ram_series.append(time_val, self.ram_data[i])
        
        # Update GPU/VRAM chart
        self.gpu_series.clear()
        self.vram_series.clear()
        
        for i, time_val in enumerate(relative_times):
            self.gpu_series.append(time_val, self.gpu_data[i])
            self.vram_series.append(time_val, self.vram_data[i])
        
        # Update axis ranges if needed
        if relative_times:
            max_time = max(relative_times)
            if max_time > 60:
                self.cpu_ram_axis_x.setRange(max_time - 60, max_time)
                self.gpu_vram_axis_x.setRange(max_time - 60, max_time)


class TrainingProgressWidget(QWidget):
    """Widget for displaying training progress"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Data storage for loss chart
        self.max_points = 500
        self.epochs = deque(maxlen=self.max_points)
        self.train_losses = deque(maxlen=self.max_points)
        self.eval_losses = deque(maxlen=self.max_points)
        self.psnr_values = deque(maxlen=self.max_points)
        self.ssim_values = deque(maxlen=self.max_points)
    
    def setup_ui(self):
        """Initialize the training progress UI"""
        layout = QVBoxLayout(self)
        
        # Progress bars and stats
        stats_group = QGroupBox("Training Progress")
        stats_layout = QGridLayout(stats_group)
        
        # Overall progress
        stats_layout.addWidget(QLabel("Overall Progress:"), 0, 0)
        self.overall_progress = QProgressBar()
        stats_layout.addWidget(self.overall_progress, 0, 1, 1, 2)
        
        # Current epoch
        stats_layout.addWidget(QLabel("Epoch:"), 1, 0)
        self.epoch_label = QLabel("0 / 0")
        self.epoch_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.epoch_label, 1, 1)
        
        # ETA
        stats_layout.addWidget(QLabel("ETA:"), 1, 2)
        self.eta_label = QLabel("Calculating...")
        self.eta_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.eta_label, 1, 3)
        
        # Training loss
        stats_layout.addWidget(QLabel("Training Loss:"), 2, 0)
        self.train_loss_label = QLabel("N/A")
        self.train_loss_label.setStyleSheet("color: #F44336; font-weight: bold;")
        stats_layout.addWidget(self.train_loss_label, 2, 1)
        
        # Eval loss
        stats_layout.addWidget(QLabel("Eval Loss:"), 2, 2)
        self.eval_loss_label = QLabel("N/A")
        self.eval_loss_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
        stats_layout.addWidget(self.eval_loss_label, 2, 3)
        
        # PSNR
        stats_layout.addWidget(QLabel("PSNR:"), 3, 0)
        self.psnr_label = QLabel("N/A")
        self.psnr_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        stats_layout.addWidget(self.psnr_label, 3, 1)
        
        # MS-SSIM
        stats_layout.addWidget(QLabel("MS-SSIM:"), 3, 2)
        self.ssim_label = QLabel("N/A")
        self.ssim_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        stats_layout.addWidget(self.ssim_label, 3, 3)
        
        layout.addWidget(stats_group)
        
        # Loss charts
        charts_group = QGroupBox("Training Charts")
        charts_layout = QVBoxLayout(charts_group)
        
        # Loss chart
        self.loss_chart = QChart()
        self.loss_chart.setTitle("Training & Validation Loss")
        self.loss_chart.setAnimationOptions(QChart.NoAnimation)
        
        self.train_loss_series = QLineSeries()
        self.train_loss_series.setName("Training Loss")
        self.train_loss_series.setColor(QColor("#F44336"))
        
        self.eval_loss_series = QLineSeries()
        self.eval_loss_series.setName("Validation Loss")
        self.eval_loss_series.setColor(QColor("#9C27B0"))
        
        self.loss_chart.addSeries(self.train_loss_series)
        self.loss_chart.addSeries(self.eval_loss_series)
        
        # Set up axes
        self.loss_axis_x = QValueAxis()
        self.loss_axis_x.setTitleText("Epoch")
        
        self.loss_axis_y = QValueAxis()
        self.loss_axis_y.setTitleText("Loss")
        
        self.loss_chart.addAxis(self.loss_axis_x, Qt.AlignBottom)
        self.loss_chart.addAxis(self.loss_axis_y, Qt.AlignLeft)
        
        self.train_loss_series.attachAxis(self.loss_axis_x)
        self.train_loss_series.attachAxis(self.loss_axis_y)
        self.eval_loss_series.attachAxis(self.loss_axis_x)
        self.eval_loss_series.attachAxis(self.loss_axis_y)
        
        loss_view = QChartView(self.loss_chart)
        charts_layout.addWidget(loss_view)
        
        # Metrics chart
        self.metrics_chart = QChart()
        self.metrics_chart.setTitle("Quality Metrics")
        self.metrics_chart.setAnimationOptions(QChart.NoAnimation)
        
        self.psnr_series = QLineSeries()
        self.psnr_series.setName("PSNR")
        self.psnr_series.setColor(QColor("#4CAF50"))
        
        self.ssim_series = QLineSeries()
        self.ssim_series.setName("MS-SSIM")
        self.ssim_series.setColor(QColor("#2196F3"))
        
        self.metrics_chart.addSeries(self.psnr_series)
        self.metrics_chart.addSeries(self.ssim_series)
        
        # Set up axes (dual Y-axis for different scales)
        self.metrics_axis_x = QValueAxis()
        self.metrics_axis_x.setTitleText("Epoch")
        
        self.psnr_axis_y = QValueAxis()
        self.psnr_axis_y.setTitleText("PSNR (dB)")
        
        self.ssim_axis_y = QValueAxis()
        self.ssim_axis_y.setTitleText("MS-SSIM")
        
        self.metrics_chart.addAxis(self.metrics_axis_x, Qt.AlignBottom)
        self.metrics_chart.addAxis(self.psnr_axis_y, Qt.AlignLeft)
        self.metrics_chart.addAxis(self.ssim_axis_y, Qt.AlignRight)
        
        self.psnr_series.attachAxis(self.metrics_axis_x)
        self.psnr_series.attachAxis(self.psnr_axis_y)
        self.ssim_series.attachAxis(self.metrics_axis_x)
        self.ssim_series.attachAxis(self.ssim_axis_y)
        
        metrics_view = QChartView(self.metrics_chart)
        charts_layout.addWidget(metrics_view)
        
        layout.addWidget(charts_group)
    
    def update_progress(self, progress_data: Dict):
        """Update training progress"""
        # Update progress bar
        if 'progress' in progress_data:
            self.overall_progress.setValue(int(progress_data['progress'] * 100))
        
        # Update epoch info
        if 'current_epoch' in progress_data and 'total_epochs' in progress_data:
            current = progress_data['current_epoch']
            total = progress_data['total_epochs']
            self.epoch_label.setText(f"{current} / {total}")
        
        # Update ETA
        if 'eta' in progress_data:
            self.eta_label.setText(format_duration(progress_data['eta']))
        
        # Update losses
        if 'train_loss' in progress_data:
            self.train_loss_label.setText(f"{progress_data['train_loss']:.6f}")
        
        if 'eval_loss' in progress_data:
            self.eval_loss_label.setText(f"{progress_data['eval_loss']:.6f}")
        
        # Update metrics
        if 'psnr' in progress_data:
            self.psnr_label.setText(f"{progress_data['psnr']:.2f} dB")
        
        if 'ms_ssim' in progress_data:
            self.ssim_label.setText(f"{progress_data['ms_ssim']:.4f}")
        
        # Add to data queues
        if 'current_epoch' in progress_data:
            epoch = progress_data['current_epoch']
            self.epochs.append(epoch)
            
            if 'train_loss' in progress_data:
                self.train_losses.append(progress_data['train_loss'])
            if 'eval_loss' in progress_data:
                self.eval_losses.append(progress_data['eval_loss'])
            if 'psnr' in progress_data:
                self.psnr_values.append(progress_data['psnr'])
            if 'ms_ssim' in progress_data:
                self.ssim_values.append(progress_data['ms_ssim'])
            
            # Update charts
            self.update_charts()
    
    def update_charts(self):
        """Update the training charts"""
        if not self.epochs:
            return
        
        # Update loss chart
        self.train_loss_series.clear()
        self.eval_loss_series.clear()
        
        for i, epoch in enumerate(self.epochs):
            if i < len(self.train_losses):
                self.train_loss_series.append(epoch, self.train_losses[i])
            if i < len(self.eval_losses):
                self.eval_loss_series.append(epoch, self.eval_losses[i])
        
        # Update metrics chart
        self.psnr_series.clear()
        self.ssim_series.clear()
        
        for i, epoch in enumerate(self.epochs):
            if i < len(self.psnr_values):
                self.psnr_series.append(epoch, self.psnr_values[i])
            if i < len(self.ssim_values):
                self.ssim_series.append(epoch, self.ssim_values[i])
        
        # Update axis ranges
        if self.epochs:
            min_epoch = min(self.epochs)
            max_epoch = max(self.epochs)
            
            self.loss_axis_x.setRange(min_epoch, max_epoch)
            self.metrics_axis_x.setRange(min_epoch, max_epoch)
            
            # Auto-scale Y axes
            if self.train_losses or self.eval_losses:
                all_losses = list(self.train_losses) + list(self.eval_losses)
                if all_losses:
                    min_loss = min(all_losses)
                    max_loss = max(all_losses)
                    self.loss_axis_y.setRange(min_loss * 0.9, max_loss * 1.1)
            
            if self.psnr_values:
                min_psnr = min(self.psnr_values)
                max_psnr = max(self.psnr_values)
                self.psnr_axis_y.setRange(min_psnr * 0.95, max_psnr * 1.05)
            
            if self.ssim_values:
                min_ssim = min(self.ssim_values)
                max_ssim = max(self.ssim_values)
                self.ssim_axis_y.setRange(min_ssim * 0.95, max_ssim * 1.05)


class LogViewerWidget(QWidget):
    """Widget for viewing training logs"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.log_path = None
        self.auto_scroll = True
        
        # Timer for checking log updates
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.check_log_updates)
        self.log_timer.start(1000)  # Check every second
        
        # Keep track of file position
        self.file_position = 0
    
    def setup_ui(self):
        """Initialize the log viewer UI"""
        layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.toggled.connect(self.toggle_auto_scroll)
        controls_layout.addWidget(self.auto_scroll_checkbox)
        
        controls_layout.addStretch()
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_logs)
        controls_layout.addWidget(self.clear_button)
        
        self.save_button = QPushButton("Save Logs")
        self.save_button.clicked.connect(self.save_logs)
        controls_layout.addWidget(self.save_button)
        
        layout.addLayout(controls_layout)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #f5f5f5;
                border: 1px solid #404040;
            }
        """)
        layout.addWidget(self.log_text)
    
    def set_log_path(self, log_path: str):
        """Set the path to the log file to monitor"""
        self.log_path = log_path
        self.file_position = 0
        self.log_text.clear()
    
    def check_log_updates(self):
        """Check for updates to the log file"""
        if not self.log_path or not os.path.exists(self.log_path):
            return
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                f.seek(self.file_position)
                new_content = f.read()
                
                if new_content:
                    self.log_text.insertPlainText(new_content)
                    self.file_position = f.tell()
                    
                    if self.auto_scroll:
                        scrollbar = self.log_text.verticalScrollBar()
                        scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    def toggle_auto_scroll(self, enabled: bool):
        """Toggle auto-scroll functionality"""
        self.auto_scroll = enabled
    
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.clear()
    
    def save_logs(self):
        """Save logs to a file"""
        from PySide6.QtWidgets import QFileDialog
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Logs", "training_log.txt", "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", f"Failed to save logs: {e}")


class ProgressWidget(QWidget):
    """Main progress widget combining all progress components"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.training_started = False
    
    # In ProgressWidget.setup_ui, replace the method:
    def setup_ui(self):
        """Initialize the progress widget UI"""
        layout = QVBoxLayout(self)
        
        # For non-dev mode, show simplified progress
        from main import DEV_MODE_ENABLED
        
        if not DEV_MODE_ENABLED:
            # Simple progress bar
            self.simple_progress_group = QGroupBox("Compression Progress")
            simple_layout = QVBoxLayout(self.simple_progress_group)
            
            self.status_label = QLabel("Waiting to start...")
            simple_layout.addWidget(self.status_label)
            
            self.overall_progress = QProgressBar()
            simple_layout.addWidget(self.overall_progress)
            
            self.time_label = QLabel("Elapsed: 0:00")
            simple_layout.addWidget(self.time_label)
            
            layout.addWidget(self.simple_progress_group)
            
            # Hide technical components
            self.training_progress = TrainingProgressWidget()
            self.training_progress.setVisible(False)
            self.system_stats = SystemStatsWidget()
            self.system_stats.setVisible(False)
            self.log_viewer = LogViewerWidget()
            self.log_viewer.setVisible(False)
        else:
            # Original implementation for dev mode
            # Create splitter for resizable panels
            splitter = QSplitter(Qt.Vertical)
            layout.addWidget(splitter)
            
            # Training progress
            self.training_progress = TrainingProgressWidget()
            splitter.addWidget(self.training_progress)
            
            # Tab widget for system stats and logs
            tabs = QTabWidget()
            
            # System stats tab
            self.system_stats = SystemStatsWidget()
            tabs.addTab(self.system_stats, "System Stats")
            
            # Log viewer tab
            self.log_viewer = LogViewerWidget()
            tabs.addTab(self.log_viewer, "Training Logs")
            
            splitter.addWidget(tabs)
            
            # Set splitter proportions
            splitter.setSizes([400, 300])

# Add this method to update simple progress:
    def update_simple_progress(self, progress_data: Dict):
        """Update simple progress display for non-dev mode"""
        if hasattr(self, 'simple_progress_group'):
            # Update status
            if 'status' in progress_data:
                self.status_label.setText(progress_data['status'])
            
            # Update progress bar
            if 'progress' in progress_data:
                self.overall_progress.setValue(int(progress_data['progress'] * 100))
            
            # Update time
            if 'elapsed_time' in progress_data:
                elapsed = progress_data['elapsed_time']
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                self.time_label.setText(f"Elapsed: {hours}:{minutes:02d}:{seconds:02d}")
        
    # def update_progress(self, progress_data: Dict):
    #     """Update training progress"""
    #     self.training_progress.update_progress(progress_data)
        
    #     if not self.training_started:
    #         self.training_started = True
    #         # Set up log monitoring if log path is provided
    #         if 'log_path' in progress_data:
    #             self.log_viewer.set_log_path(progress_data['log_path'])
    
    def update_system_stats(self, stats: Dict):
        """Update system statistics"""
        self.system_stats.update_stats(stats)
    
    def update_status(self, status_message: str):
        """Update status message"""
        # You can add a status label if needed
        pass