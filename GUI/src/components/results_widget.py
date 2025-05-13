"""
Results widget for displaying compression results
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QTextEdit, QScrollArea, QFrame,
    QProgressBar, QComboBox, QCheckBox, QDoubleSpinBox,
    QSlider, QFileDialog, QMessageBox, QTabWidget, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
)
from PySide6.QtCore import Qt, Signal, QThread, Slot, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon, QImage
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
from PySide6.QtGui import QPainter

from ..utils import format_duration, format_filesize, calculate_bitrate


class VideoPreviewPlayer(QThread):
    """Thread for playing video preview"""
    
    frame_ready = Signal(QPixmap)
    
    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self.playing = False
    
    def run(self):
        # Implement basic video preview using OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            
            while self.playing and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert frame to QPixmap and emit
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(q_image)
                    self.frame_ready.emit(pixmap)
                    self.msleep(33)  # ~30 FPS
                else:
                    break
            
            cap.release()
        except ImportError:
            pass
    
    def start_playback(self):
        self.playing = True
        self.start()
    
    def stop_playback(self):
        self.playing = False
        self.wait()


class ResultsWidget(QWidget):
    """Widget for displaying compression results"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.video_player = None
        
    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout(self)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Results summary
        left_panel = self.create_results_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Video preview and comparison
        right_panel = self.create_preview_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
    def create_results_panel(self) -> QWidget:
        """Create the results summary panel"""
        panel = QScrollArea()
        content = QWidget()
        panel.setWidget(content)
        panel.setWidgetResizable(True)
        
        layout = QVBoxLayout(content)
        
        # Compression summary group
        summary_group = QGroupBox("Compression Summary")
        summary_layout = QGridLayout(summary_group)
        
        # Metrics display
        self.original_size_label = QLabel("N/A")
        self.compressed_size_label = QLabel("N/A")
        self.compression_ratio_label = QLabel("N/A")
        self.bitrate_label = QLabel("N/A")
        self.psnr_label = QLabel("N/A")
        self.ssim_label = QLabel("N/A")
        self.ms_ssim_label = QLabel("N/A")
        self.lpips_label = QLabel("N/A")
        
        summary_layout.addWidget(QLabel("Original Size:"), 0, 0)
        summary_layout.addWidget(self.original_size_label, 0, 1)
        summary_layout.addWidget(QLabel("Compressed Size:"), 1, 0)
        summary_layout.addWidget(self.compressed_size_label, 1, 1)
        summary_layout.addWidget(QLabel("Compression Ratio:"), 2, 0)
        summary_layout.addWidget(self.compression_ratio_label, 2, 1)
        summary_layout.addWidget(QLabel("Bitrate:"), 3, 0)
        summary_layout.addWidget(self.bitrate_label, 3, 1)
        summary_layout.addWidget(QLabel("PSNR:"), 4, 0)
        summary_layout.addWidget(self.psnr_label, 4, 1)
        summary_layout.addWidget(QLabel("SSIM:"), 5, 0)
        summary_layout.addWidget(self.ssim_label, 5, 1)
        summary_layout.addWidget(QLabel("MS-SSIM:"), 6, 0)
        summary_layout.addWidget(self.ms_ssim_label, 6, 1)
        summary_layout.addWidget(QLabel("LPIPS:"), 7, 0)
        summary_layout.addWidget(self.lpips_label, 7, 1)
        
        layout.addWidget(summary_group)
        
        # Training details group
        details_group = QGroupBox("Training Details")
        details_layout = QGridLayout(details_group)
        
        self.model_used_label = QLabel()
        self.epochs_label = QLabel()
        self.training_time_label = QLabel()
        self.final_loss_label = QLabel()
        
        details_layout.addWidget(QLabel("Model:"), 0, 0)
        details_layout.addWidget(self.model_used_label, 0, 1)
        details_layout.addWidget(QLabel("Epochs:"), 1, 0)
        details_layout.addWidget(self.epochs_label, 1, 1)
        details_layout.addWidget(QLabel("Training Time:"), 2, 0)
        details_layout.addWidget(self.training_time_label, 2, 1)
        details_layout.addWidget(QLabel("Final Loss:"), 3, 0)
        details_layout.addWidget(self.final_loss_label, 3, 1)
        
        layout.addWidget(details_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        
        self.play_btn = QPushButton("Play Video")
        self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_btn.clicked.connect(self.play_video)
        actions_layout.addWidget(self.play_btn)
        
        self.open_location_btn = QPushButton("Open Location")
        self.open_location_btn.setIcon(QIcon.fromTheme("folder-open"))
        self.open_location_btn.clicked.connect(self.open_file_location)
        actions_layout.addWidget(self.open_location_btn)
        
        self.export_report_btn = QPushButton("Export Report")
        self.export_report_btn.setIcon(QIcon.fromTheme("document-save"))
        self.export_report_btn.clicked.connect(self.export_report)
        actions_layout.addWidget(self.export_report_btn)
        
        layout.addWidget(actions_group)
        
        # Stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def create_preview_panel(self) -> QWidget:
        """Create the video preview panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Video comparison tab
        comparison_tab = self.create_comparison_tab()
        tabs.addTab(comparison_tab, "Video Comparison")
        
        # Metrics charts tab
        charts_tab = self.create_charts_tab()
        tabs.addTab(charts_tab, "Metrics Charts")
        
        # Frame analysis tab
        frame_analysis_tab = self.create_frame_analysis_tab()
        tabs.addTab(frame_analysis_tab, "Frame Analysis")
        
        return panel
    
    def create_comparison_tab(self) -> QWidget:
        """Create the video comparison tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Control panel for comparison
        controls = QHBoxLayout()
        
        # Frame selection
        controls.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        controls.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        controls.addWidget(self.frame_label)
        
        # Playback controls
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_pause_btn)
        
        layout.addLayout(controls)
        
        # Video display
        video_layout = QHBoxLayout()
        
        # Original video side
        original_group = QGroupBox("Original")
        original_layout = QVBoxLayout(original_group)
        self.original_video_label = QLabel()
        self.original_video_label.setMinimumSize(400, 300)
        self.original_video_label.setStyleSheet("border: 1px solid gray;")
        self.original_video_label.setScaledContents(True)
        original_layout.addWidget(self.original_video_label)
        video_layout.addWidget(original_group)
        
        # Compressed video side
        compressed_group = QGroupBox("Compressed")
        compressed_layout = QVBoxLayout(compressed_group)
        self.compressed_video_label = QLabel()
        self.compressed_video_label.setMinimumSize(400, 300)
        self.compressed_video_label.setStyleSheet("border: 1px solid gray;")
        self.compressed_video_label.setScaledContents(True)
        compressed_layout.addWidget(self.compressed_video_label)
        video_layout.addWidget(compressed_group)
        
        layout.addLayout(video_layout)
        
        return widget
    
    def create_charts_tab(self) -> QWidget:
        """Create the metrics charts tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # PSNR chart
        self.psnr_chart = self.create_metric_chart("PSNR", "dB")
        layout.addWidget(self.psnr_chart, 0, 0)
        
        # SSIM chart
        self.ssim_chart = self.create_metric_chart("SSIM", "")
        layout.addWidget(self.ssim_chart, 0, 1)
        
        # MS-SSIM chart
        self.ms_ssim_chart = self.create_metric_chart("MS-SSIM", "")
        layout.addWidget(self.ms_ssim_chart, 1, 0)
        
        # LPIPS chart
        self.lpips_chart = self.create_metric_chart("LPIPS", "")
        layout.addWidget(self.lpips_chart, 1, 1)
        
        return widget
    
    def create_metric_chart(self, title: str, unit: str) -> QChartView:
        """Create a chart for a specific metric"""
        chart = QChart()
        chart.setTitle(title)
        chart.setAnimationOptions(QChart.SeriesAnimations)
        
        series = QLineSeries()
        chart.addSeries(series)
        
        # Configure axes
        axis_x = QValueAxis()
        axis_x.setTitleText("Frame")
        axis_x.setLabelFormat("%d")
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText(f"{title} ({unit})" if unit else title)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        
        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        
        return view
    
    def create_frame_analysis_tab(self) -> QWidget:
        """Create the frame analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems([
            "Pixel Difference",
            "Error Heatmap",
            "Frequency Analysis",
            "Motion Vectors"
        ])
        self.analysis_combo.currentTextChanged.connect(self.update_analysis)
        controls_layout.addWidget(self.analysis_combo)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Analysis display
        self.analysis_label = QLabel()
        self.analysis_label.setMinimumSize(800, 600)
        self.analysis_label.setStyleSheet("border: 1px solid gray;")
        self.analysis_label.setScaledContents(True)
        layout.addWidget(self.analysis_label)
        
        return widget
    
    def show_results(self, results: Dict):
        """Display compression results"""
        # Update summary labels
        self.original_size_label.setText(format_filesize(results.get('original_size', 0)))
        self.compressed_size_label.setText(format_filesize(results.get('compressed_size', 0)))
        
        # Calculate compression ratio
        if results.get('original_size', 0) > 0:
            ratio = results.get('original_size', 0) / results.get('compressed_size', 1)
            self.compression_ratio_label.setText(f"{ratio:.2f}x")
        
        # Calculate bitrate
        if 'duration' in results and results.get('compressed_size', 0) > 0:
            bitrate = calculate_bitrate(results['compressed_size'], results['duration'])
            self.bitrate_label.setText(f"{bitrate:.2f} Mbps")
        
        # Update metrics
        self.psnr_label.setText(f"{results.get('psnr', 0):.2f} dB")
        self.ssim_label.setText(f"{results.get('ssim', 0):.4f}")
        self.ms_ssim_label.setText(f"{results.get('ms_ssim', 0):.4f}")
        self.lpips_label.setText(f"{results.get('lpips', 0):.4f}")
        
        # Update training details
        self.model_used_label.setText(results.get('model_name', 'N/A'))
        self.epochs_label.setText(str(results.get('epochs', 0)))
        self.training_time_label.setText(format_duration(results.get('training_time', 0)))
        self.final_loss_label.setText(f"{results.get('final_loss', 0):.6f}")
        
        # Store results for later use
        self.results = results
        
        # Load video frames for comparison
        self.load_comparison_videos()
        
        # Update charts
        self.update_metric_charts()
    
    def load_comparison_videos(self):
        """Load original and compressed videos for comparison"""
        if not hasattr(self, 'results') or not self.results:
            return
        
        # Load video frames (simplified implementation)
        original_path = self.results.get('original_path')
        compressed_path = self.results.get('compressed_path')
        
        if original_path and compressed_path:
            # Set up frame slider
            total_frames = self.results.get('total_frames', 0)
            self.frame_slider.setMaximum(total_frames - 1)
            self.frame_label.setText(f"0 / {total_frames}")
            
            # Load first frames
            self.load_frame_at_index(0)
    
    def load_frame_at_index(self, index: int):
        """Load frames at specific index for comparison"""
        # This is a simplified implementation
        # In practice, you would extract frames from both videos
        pass
    
    def update_metric_charts(self):
        """Update metric charts with frame-by-frame data"""
        if not hasattr(self, 'results') or not self.results:
            return
        
        # Get frame-by-frame metrics if available
        frame_metrics = self.results.get('frame_metrics', {})
        
        if frame_metrics:
            self.update_chart(self.psnr_chart, frame_metrics.get('psnr', []))
            self.update_chart(self.ssim_chart, frame_metrics.get('ssim', []))
            self.update_chart(self.ms_ssim_chart, frame_metrics.get('ms_ssim', []))
            self.update_chart(self.lpips_chart, frame_metrics.get('lpips', []))
    
    def update_chart(self, chart_view: QChartView, data: List[float]):
        """Update a specific chart with data"""
        chart = chart_view.chart()
        series = chart.series()[0] if chart.series() else None
        
        if series and data:
            series.clear()
            for i, value in enumerate(data):
                series.append(i, value)
    
    def on_frame_changed(self, frame_index: int):
        """Handle frame slider change"""
        total_frames = self.frame_slider.maximum() + 1
        self.frame_label.setText(f"{frame_index} / {total_frames}")
        self.load_frame_at_index(frame_index)
    
    def toggle_playback(self):
        """Toggle video playback"""
        if self.video_player and self.video_player.playing:
            self.video_player.stop_playback()
            self.play_pause_btn.setText("Play")
            self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        else:
            if hasattr(self, 'results') and self.results.get('compressed_path'):
                self.video_player = VideoPreviewPlayer(self.results['compressed_path'])
                self.video_player.frame_ready.connect(self.update_video_frame)
                self.video_player.start_playback()
                self.play_pause_btn.setText("Pause")
                self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-pause"))
    
    def update_video_frame(self, pixmap: QPixmap):
        """Update video display with new frame"""
        self.compressed_video_label.setPixmap(pixmap)
    
    def update_analysis(self, analysis_type: str):
        """Update frame analysis display"""
        # Implement different analysis types
        if analysis_type == "Pixel Difference":
            self.show_pixel_difference()
        elif analysis_type == "Error Heatmap":
            self.show_error_heatmap()
        elif analysis_type == "Frequency Analysis":
            self.show_frequency_analysis()
        elif analysis_type == "Motion Vectors":
            self.show_motion_vectors()
    
    def show_pixel_difference(self):
        """Show pixel-wise difference between original and compressed"""
        # Implement pixel difference visualization
        pass
    
    def show_error_heatmap(self):
        """Show error heatmap"""
        # Implement error heatmap visualization
        pass
    
    def show_frequency_analysis(self):
        """Show frequency domain analysis"""
        # Implement frequency analysis
        pass
    
    def show_motion_vectors(self):
        """Show motion vector visualization"""
        # Implement motion vector visualization
        pass
    
    def play_video(self):
        """Play the compressed video in system default player"""
        if hasattr(self, 'results') and self.results.get('compressed_path'):
            video_path = self.results['compressed_path']
            
            # Open video with system default application
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(video_path)
                elif os.name == 'posix':  # Linux/Mac
                    subprocess.run(['xdg-open', video_path])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open video: {str(e)}")
    
    def open_file_location(self):
        """Open the output directory in file manager"""
        if hasattr(self, 'results') and self.results.get('output_dir'):
            output_dir = self.results['output_dir']
            
            # Open directory with system file manager
            try:
                if os.name == 'nt':  # Windows
                    subprocess.run(['explorer', output_dir])
                elif os.name == 'posix':  # Linux/Mac
                    subprocess.run(['xdg-open', output_dir])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open directory: {str(e)}")
    
    def export_report(self):
        """Export compression results to a report file"""
        if not hasattr(self, 'results') or not self.results:
            QMessageBox.warning(self, "No Results", "No results available to export.")
            return
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", 
            f"compression_report_{Path(self.results.get('original_path', 'video')).stem}.html",
            "HTML Files (*.html);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    # Export as JSON
                    with open(file_path, 'w') as f:
                        json.dump(self.results, f, indent=2)
                else:
                    # Export as HTML report
                    self.create_html_report(file_path)
                
                QMessageBox.information(self, "Export Complete", f"Report exported to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed to export report: {str(e)}")
    
    def create_html_report(self, file_path: str):
        """Create an HTML report of the compression results"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HiNeRV Compression Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; }}
                .metric {{ display: flex; justify-content: space-between; padding: 10px; margin: 5px 0; background-color: #f9f9f9; border-radius: 5px; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ color: #4CAF50; font-weight: bold; }}
                .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; text-align: center; color: #888; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>HiNeRV Compression Report</h1>
                
                <h2>Video Information</h2>
                <div class="metric">
                    <span class="metric-name">Original File:</span>
                    <span class="metric-value">{Path(self.results.get('original_path', 'N/A')).name}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Output File:</span>
                    <span class="metric-value">{Path(self.results.get('compressed_path', 'N/A')).name}</span>
                </div>
                
                <h2>Compression Statistics</h2>
                <div class="metric">
                    <span class="metric-name">Original Size:</span>
                    <span class="metric-value">{format_filesize(self.results.get('original_size', 0))}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Compressed Size:</span>
                    <span class="metric-value">{format_filesize(self.results.get('compressed_size', 0))}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Compression Ratio:</span>
                    <span class="metric-value">{self.results.get('original_size', 0) / self.results.get('compressed_size', 1):.2f}x</span>
                </div>
                
                <h2>Quality Metrics</h2>
                <div class="metric">
                    <span class="metric-name">PSNR:</span>
                    <span class="metric-value">{self.results.get('psnr', 0):.2f} dB</span>
                </div>
                <div class="metric">
                    <span class="metric-name">SSIM:</span>
                    <span class="metric-value">{self.results.get('ssim', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">MS-SSIM:</span>
                    <span class="metric-value">{self.results.get('ms_ssim', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">LPIPS:</span>
                    <span class="metric-value">{self.results.get('lpips', 0):.4f}</span>
                </div>
                
                <h2>Training Details</h2>
                <div class="metric">
                    <span class="metric-name">Model:</span>
                    <span class="metric-value">{self.results.get('model_name', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Epochs:</span>
                    <span class="metric-value">{self.results.get('epochs', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Training Time:</span>
                    <span class="metric-value">{format_duration(self.results.get('training_time', 0))}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Final Loss:</span>
                    <span class="metric-value">{self.results.get('final_loss', 0):.6f}</span>
                </div>
                
                <div class="footer">
                    <p>Generated by HiNeRV GUI</p>
                    <p>Report created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(file_path, 'w') as f:
            f.write(html_content)