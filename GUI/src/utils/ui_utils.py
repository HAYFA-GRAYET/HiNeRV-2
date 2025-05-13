"""
UI utility functions for HiNeRV GUI
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from PySide6.QtCore import Qt, QSize, QRect, QPoint, QEvent
from PySide6.QtGui import QIcon, QPixmap, QImage, QColor
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFormLayout, QFrame, QFileDialog, QMessageBox, QToolTip, QSizePolicy
)

# Define standard margins and spacing for consistent UI layout
UI_MARGINS = (16, 16, 16, 16)  # left, top, right, bottom
UI_SPACING = 8


def create_h_line() -> QFrame:
    """
    Create a horizontal separator line
    
    Returns:
        QFrame: Horizontal line widget
    """
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setObjectName("hSeparator")
    return line


def create_v_line() -> QFrame:
    """
    Create a vertical separator line
    
    Returns:
        QFrame: Vertical line widget
    """
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setObjectName("vSeparator")
    return line


def create_card_widget(
    title: Optional[str] = None,
    content_widget: Optional[QWidget] = None,
    layout_type: str = "vertical"
) -> QWidget:
    """
    Create a card-like container widget with optional title
    
    Args:
        title: Optional title for the card
        content_widget: Widget to add as content
        layout_type: Layout type ("vertical", "horizontal", or "grid")
        
    Returns:
        QWidget: Card widget
    """
    card = QFrame()
    card.setObjectName("card")
    card.setProperty("class", "card")
    
    # Set up the card layout
    if layout_type == "vertical":
        main_layout = QVBoxLayout(card)
    elif layout_type == "horizontal":
        main_layout = QHBoxLayout(card)
    else:  # grid
        main_layout = QGridLayout(card)
    
    main_layout.setContentsMargins(*UI_MARGINS)
    main_layout.setSpacing(UI_SPACING)
    
    # Add title if provided
    if title:
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        main_layout.addWidget(title_label)
        
        # Add separator after title
        main_layout.addWidget(create_h_line())
    
    # Add content widget if provided
    if content_widget:
        if isinstance(main_layout, QGridLayout):
            main_layout.addWidget(content_widget, 1, 0)
        else:
            main_layout.addWidget(content_widget)
    
    return card


def create_header(title: str, description: Optional[str] = None) -> QWidget:
    """
    Create a section header with optional description
    
    Args:
        title: Section title
        description: Optional description text
        
    Returns:
        QWidget: Header widget
    """
    header = QWidget()
    layout = QVBoxLayout(header)
    layout.setContentsMargins(0, 0, 0, UI_SPACING)
    layout.setSpacing(4)
    
    title_label = QLabel(title)
    title_label.setObjectName("sectionTitle")
    layout.addWidget(title_label)
    
    if description:
        desc_label = QLabel(description)
        desc_label.setObjectName("sectionDescription")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
    
    layout.addWidget(create_h_line())
    
    return header


def create_file_input(
    parent: QWidget,
    label: str,
    file_types: str,
    on_file_selected: Callable[[str], None],
    default_dir: str = ""
) -> QWidget:
    """
    Create a file input widget with label, text field and browse button
    
    Args:
        parent: Parent widget
        label: Label text
        file_types: File filter string (e.g. "Video Files (*.mp4 *.avi)")
        on_file_selected: Callback function when file is selected
        default_dir: Default directory
        
    Returns:
        QWidget: File input widget
    """
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    
    file_label = QLabel(label)
    file_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
    
    file_path = QLabel("No file selected")
    file_path.setObjectName("filePath")
    file_path.setFrameShape(QFrame.StyledPanel)
    file_path.setFrameShadow(QFrame.Sunken)
    file_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
    
    browse_btn = QPushButton("Browse...")
    browse_btn.setObjectName("browseButton")
    browse_btn.setMaximumWidth(100)
    
    layout.addWidget(file_label)
    layout.addWidget(file_path)
    layout.addWidget(browse_btn)
    
    def browse_file():
        file_dialog = QFileDialog(parent)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(file_types)
        
        if default_dir:
            file_dialog.setDirectory(default_dir)
        
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            file_path.setText(Path(selected_file).name)
            file_path.setToolTip(selected_file)
            on_file_selected(selected_file)
    
    browse_btn.clicked.connect(browse_file)
    
    return widget


def create_control_group(title: str, widgets: List[QWidget]) -> QWidget:
    """
    Create a grouped set of controls with title
    
    Args:
        title: Group title
        widgets: List of widgets to add to the group
        
    Returns:
        QWidget: Control group widget
    """
    group = QFrame()
    group.setObjectName("controlGroup")
    
    layout = QVBoxLayout(group)
    layout.setContentsMargins(*UI_MARGINS)
    layout.setSpacing(UI_SPACING)
    
    # Add title
    title_label = QLabel(title)
    title_label.setObjectName("groupTitle")
    layout.addWidget(title_label)
    
    # Add separator
    layout.addWidget(create_h_line())
    
    # Add widgets
    for widget in widgets:
        layout.addWidget(widget)
    
    return group


def show_confirmation_dialog(
    parent: QWidget,
    title: str,
    message: str,
    detail: Optional[str] = None
) -> bool:
    """
    Show a confirmation dialog with Yes/No buttons
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Dialog message
        detail: Optional detailed text
        
    Returns:
        bool: True if confirmed, False otherwise
    """
    dialog = QMessageBox(parent)
    dialog.setWindowTitle(title)
    dialog.setText(message)
    dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    dialog.setDefaultButton(QMessageBox.No)
    dialog.setIcon(QMessageBox.Question)
    
    if detail:
        dialog.setDetailedText(detail)
    
    return dialog.exec() == QMessageBox.Yes


def show_error_dialog(
    parent: QWidget,
    title: str,
    message: str,
    detail: Optional[str] = None,
    with_clipboard: bool = True
) -> None:
    """
    Show an error dialog with optional copy to clipboard button
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Dialog message
        detail: Optional detailed text
        with_clipboard: Whether to add "Copy to Clipboard" button
    """
    dialog = QMessageBox(parent)
    dialog.setWindowTitle(title)
    dialog.setText(message)
    dialog.setIcon(QMessageBox.Critical)
    
    if detail:
        dialog.setDetailedText(detail)
    
    if with_clipboard:
        copy_btn = dialog.addButton("Copy to Clipboard", QMessageBox.ActionRole)
        dialog.addButton(QMessageBox.Ok)
        
        result = dialog.exec()
        
        if dialog.clickedButton() == copy_btn:
            from PySide6.QtGui import QGuiApplication
            text = f"{message}\n\nDetails:\n{detail}" if detail else message
            QGuiApplication.clipboard().setText(text)
    else:
        dialog.exec_()


def format_file_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def format_time_duration(seconds: float) -> str:
    """
    Format seconds to human-readable duration
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"
    
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller
    
    Args:
        relative_path: Path relative to the application
        
    Returns:
        str: Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def create_separator():
    """Create a horizontal separator line"""
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    return separator

def create_tooltip(widget: QWidget, text: str) -> None:
    """
    Add a tooltip to a widget
    
    Args:
        widget: Widget to add tooltip to
        text: Tooltip text
    """
    widget.setToolTip(text)
    
    # Make tooltip stay longer
    widget.setProperty("hasCustomTooltip", True)
    original_event = widget.event
    
    def custom_event(event):
        if event.type() == QEvent.ToolTip:
            QToolTip.showText(event.globalPos(), text, widget)
            return True
        return original_event(event)
    
    widget.event = custom_event


def create_info_label(text: str, info_text: str) -> QWidget:
    """
    Create a label with an info icon that shows tooltip on hover
    
    Args:
        text: Label text
        info_text: Tooltip text
        
    Returns:
        QWidget: Label with info icon
    """
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    
    label = QLabel(text)
    layout.addWidget(label)
    
    info_icon = QLabel()
    info_icon.setObjectName("infoIcon")
    # You would need to include an info icon in your resources
    # info_icon.setPixmap(QIcon.fromTheme("dialog-information").pixmap(16, 16))
    layout.addWidget(info_icon)
    
    create_tooltip(widget, info_text)
    
    layout.addStretch()
    
    return widget


def set_form_field_width(widget: QWidget, width: int) -> None:
    """
    Set width for a form field widget
    
    Args:
        widget: Form field widget
        width: Desired width
    """
    widget.setFixedWidth(width)
    widget.setMinimumWidth(width)


def nv_memory_info() -> Dict[str, Any]:
    """
    Get NVIDIA GPU memory information
    
    Returns:
        Dict with keys: total_memory, free_memory, used_memory (in MB)
    """
    try:
        import nvidia_ml_py3 as nvml
        
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        
        total_mb = mem_info.total / 1024 / 1024
        used_mb = mem_info.used / 1024 / 1024
        free_mb = mem_info.free / 1024 / 1024
        
        nvml.nvmlShutdown()
        
        return {
            "total_memory": total_mb,
            "free_memory": free_mb,
            "used_memory": used_mb
        }
    except Exception as e:
        return {
            "total_memory": 0,
            "free_memory": 0,
            "used_memory": 0,
            "error": str(e)
        }


def estimate_batch_size(video_info: Dict, vram_limit_percent: int = 80) -> int:
    """
    Estimate safe batch size based on video dimensions and available VRAM
    
    Args:
        video_info: Video information dictionary
        vram_limit_percent: VRAM usage limit as percentage
        
    Returns:
        int: Estimated safe batch size
    """
    try:
        gpu_info = nv_memory_info()
        if "error" in gpu_info:
            return 4  # Conservative default
        
        # Get free VRAM and apply limit
        free_vram_mb = gpu_info["free_memory"]
        vram_limit_mb = free_vram_mb * (vram_limit_percent / 100)
        
        # Estimate memory requirement per frame based on resolution
        # This is a simplified heuristic that should be calibrated
        width = video_info["width"]
        height = video_info["height"]
        pixels = width * height
        
        # Estimated VRAM usage per frame (MB) - calibrate this based on actual usage
        # Currently using a simple heuristic based on resolution
        vram_per_frame_mb = (pixels / (1920 * 1080)) * 400  # ~400MB for 1080p
        
        # Factor in model size and training overhead
        training_overhead_mb = 2000  # ~2GB for model and optimizer state
        
        # Calculate max batch size
        available_for_frames = vram_limit_mb - training_overhead_mb
        max_batch_size = int(available_for_frames / vram_per_frame_mb)
        
        # Apply bounds
        max_batch_size = max(1, min(max_batch_size, 16))
        
        return max_batch_size
    
    except Exception as e:
        # Fall back to conservative default
        return 4

def human_readable_size(size_bytes: float) -> str:
    """
    Convert bytes to human readable string format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Human readable size string
    """
    if size_bytes == 0:
        return "0B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1
    
    return f"{size_bytes:.2f}{units[unit_index]}"

def create_icon_button(
    icon_name: str,
    tooltip: str = "",
    size: int = 24,
    clickable: bool = True
) -> QPushButton:
    """
    Create a button with an icon
    
    Args:
        icon_name: Name of the icon resource
        tooltip: Tooltip text for the button
        size: Size of the button in pixels
        clickable: Whether the button should be clickable
        
    Returns:
        QPushButton: Button with icon
    """
    button = QPushButton()
    button.setFixedSize(QSize(size, size))
    button.setIconSize(QSize(size - 8, size - 8))
    
    # Load icon from resources
    icon_path = resource_path(f"icons/{icon_name}")
    if os.path.exists(icon_path):
        button.setIcon(QIcon(icon_path))
    
    if tooltip:
        button.setToolTip(tooltip)
    
    if not clickable:
        button.setEnabled(False)
    
    button.setObjectName("iconButton")
    return button

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller
    
    Args:
        relative_path: Path relative to resources directory
        
    Returns:
        str: Absolute path to resource
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
    
    return os.path.join(base_path, relative_path)