# src/components/log_viewer.py
import os
from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTextEdit, QLabel, QFileDialog, QComboBox
)
from PySide6.QtCore import QTimer, Qt, Signal, QFile, QIODevice, QTextStream


class LogViewerWidget(QWidget):
    """Widget for viewing log files"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_file: Optional[str] = None
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_log)
        self.log_position = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the log viewer UI"""
        layout = QVBoxLayout(self)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        self.log_selector = QComboBox()
        self.log_selector.addItem("Application Log", "app")
        self.log_selector.addItem("Training Log", "training")
        self.log_selector.currentIndexChanged.connect(self.on_log_changed)
        header_layout.addWidget(self.log_selector)
        
        self.auto_scroll_btn = QPushButton("Auto-scroll")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setChecked(True)
        header_layout.addWidget(self.auto_scroll_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_log)
        header_layout.addWidget(self.clear_btn)
        
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self.copy_log)
        header_layout.addWidget(self.copy_btn)
        
        layout.addLayout(header_layout)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(self.monospace_font())
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.log_text)
        
        # Status bar
        self.status_label = QLabel("No log file selected")
        layout.addWidget(self.status_label)
        
    def monospace_font(self):
        """Get a monospace font for the log viewer"""
        font = self.font()
        font.setFamily("Monospace" if os.name != "nt" else "Consolas")
        return font
        
    def on_log_changed(self, index):
        """Handle log selection change"""
        log_type = self.log_selector.currentData()
        
        if log_type == "app":
            # Find application log file
            from PySide6.QtCore import QStandardPaths
            log_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
            log_file = log_dir / "hinerv_gui.log"
            if log_file.exists():
                self.set_log_file(str(log_file))
            else:
                self.log_text.setPlainText("Application log file not found.")
                self.stop_monitoring()
        
        elif log_type == "training":
            # Find most recent training log file
            if hasattr(self, 'current_output_dir') and self.current_output_dir:
                log_file = Path(self.current_output_dir) / "rank_0.txt"
                if log_file.exists():
                    self.set_log_file(str(log_file))
                else:
                    self.log_text.setPlainText("No active training log found.")
                    self.stop_monitoring()
            else:
                self.log_text.setPlainText("No active training session.")
                self.stop_monitoring()
    
    def set_log_file(self, file_path: str):
        """Set the log file to monitor"""
        self.log_file = file_path
        self.log_position = 0
        self.log_text.clear()
        
        # Initial load
        self.update_log()
        
        # Start timer for auto-updates
        if not self.log_timer.isActive():
            self.log_timer.start(1000)  # Update every second
            
        self.status_label.setText(f"Monitoring: {Path(file_path).name}")
    
    def update_log(self):
        """Update log contents from file"""
        if not self.log_file:
            return
            
        try:
            file = QFile(self.log_file)
            if file.open(QIODevice.ReadOnly | QIODevice.Text):
                file.seek(self.log_position)
                stream = QTextStream(file)
                new_text = stream.readAll()
                
                if new_text:
                    self.log_text.append(new_text)
                    self.log_position = file.pos()
                    
                    # Auto-scroll to bottom if enabled
                    if self.auto_scroll_btn.isChecked():
                        self.log_text.verticalScrollBar().setValue(
                            self.log_text.verticalScrollBar().maximum()
                        )
                        
                file.close()
        except Exception as e:
            self.log_text.append(f"Error reading log file: {str(e)}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring the log file"""
        if self.log_timer.isActive():
            self.log_timer.stop()
        self.status_label.setText("Log monitoring stopped")
    
    def clear_log(self):
        """Clear the log display"""
        self.log_text.clear()
    
    def copy_log(self):
        """Copy log contents to clipboard"""
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText(self.log_text.toPlainText())
        self.status_label.setText("Log copied to clipboard")
    
    def set_current_output_dir(self, output_dir: str):
        """Set the current output directory for training logs"""
        self.current_output_dir = output_dir
        
        # If currently viewing training logs, update
        if self.log_selector.currentData() == "training":
            self.on_log_changed(self.log_selector.currentIndex())