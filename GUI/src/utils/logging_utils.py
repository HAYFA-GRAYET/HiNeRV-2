"""
HiNeRV GUI - Logging Utilities
Provides functions for setting up logging
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
DEFAULT_LOG_BACKUP_COUNT = 3


def setup_logging(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Set up application logging with console and file handlers.
    
    Args:
        log_file: Path to log file (default: None)
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
    
    Returns:
        Logger: Root logger object
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Remove existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Ensure parent directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=DEFAULT_MAX_LOG_SIZE,
            backupCount=DEFAULT_LOG_BACKUP_COUNT
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Log startup information
    root_logger.info(f"Logging initialized at {datetime.now().isoformat()}")
    root_logger.info(f"Python version: {sys.version}")
    root_logger.info(f"Platform: {sys.platform}")
    
    if log_file:
        root_logger.info(f"Log file: {log_file}")
    
    return root_logger


def get_log_tail(log_file, lines=100):
    """
    Get the last N lines from a log file.
    
    Args:
        log_file: Path to log file
        lines: Number of lines to retrieve (default: 100)
    
    Returns:
        str: Last N lines of the log file
    """
    try:
        if not os.path.exists(log_file):
            return "Log file not found."
        
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            # Read all lines and get the last N
            all_lines = f.readlines()
            tail_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return ''.join(tail_lines)
    except Exception as e:
        return f"Error reading log file: {str(e)}"


class LogBuffer:
    """
    In-memory buffer for logs with callback functionality.
    Useful for showing logs in UI without reading the file.
    """
    
    def __init__(self, max_lines=1000):
        """
        Initialize the log buffer.
        
        Args:
            max_lines: Maximum number of lines to keep in buffer (default: 1000)
        """
        self.buffer = []
        self.max_lines = max_lines
        self.callbacks = []
    
    def add_line(self, line):
        """
        Add a line to the buffer.
        
        Args:
            line: Line to add
        """
        self.buffer.append(line)
        
        # Keep buffer size limited
        if len(self.buffer) > self.max_lines:
            self.buffer.pop(0)
        
        # Call all registered callbacks
        for callback in self.callbacks:
            callback(line)
    
    def get_content(self):
        """
        Get all content from the buffer.
        
        Returns:
            str: All buffered log content
        """
        return '\n'.join(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
    
    def register_callback(self, callback):
        """
        Register a callback for new log lines.
        
        Args:
            callback: Function that will be called with each new line
        """
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback):
        """
        Unregister a callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)


class LogBufferHandler(logging.Handler):
    """
    Custom logging handler to output logs to a LogBuffer.
    """
    
    def __init__(self, log_buffer):
        """
        Initialize the handler.
        
        Args:
            log_buffer: LogBuffer instance to write logs to
        """
        super().__init__()
        self.log_buffer = log_buffer
    
    def emit(self, record):
        """
        Process a log record.
        
        Args:
            record: Log record to process
        """
        try:
            log_entry = self.format(record)
            self.log_buffer.add_line(log_entry)
        except Exception:
            self.handleError(record)