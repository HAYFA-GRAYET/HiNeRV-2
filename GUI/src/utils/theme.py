"""
Theme and styling utilities for HiNeRV GUI
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QFont
from PySide6.QtWidgets import QApplication

# Color palette for dark theme
COLORS = {
    "background": "#1e1e2e",
    "background_alt": "#2a2a3a",
    "foreground": "#cdd6f4",
    "border": "#45475a",
    "accent": "#89b4fa",
    "accent_darker": "#5d88d5",
    "success": "#a6e3a1",
    "warning": "#f9e2af",
    "error": "#f38ba8",
    "info": "#74c7ec",
    "shadow": "#11111b80"
}

# Style sheets for different components
STYLE_SHEETS = {
    "main": f"""
        QMainWindow, QDialog {{
            background-color: {COLORS["background"]};
            color: {COLORS["foreground"]};
        }}
        
        QWidget {{
            background-color: {COLORS["background"]};
            color: {COLORS["foreground"]};
        }}
        
        QSplitter::handle {{
            background-color: {COLORS["border"]};
        }}
        
        QMenuBar, QMenu {{
            background-color: {COLORS["background_alt"]};
            color: {COLORS["foreground"]};
            border: none;
        }}
        
        QMenuBar::item:selected, QMenu::item:selected {{
            background-color: {COLORS["accent"]};
            color: {COLORS["background"]};
            border-radius: 4px;
        }}
        
        QStatusBar {{
            background-color: {COLORS["background_alt"]};
            color: {COLORS["foreground"]};
            border-top: 1px solid {COLORS["border"]};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            top: -1px;
        }}
        
        QTabBar::tab {{
            background-color: {COLORS["background_alt"]};
            color: {COLORS["foreground"]};
            border: 1px solid {COLORS["border"]};
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 6px 12px;
            margin-right: 2px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {COLORS["accent"]};
            color: {COLORS["background"]};
        }}
        
        QToolBar {{
            background-color: {COLORS["background_alt"]};
            border: none;
            spacing: 6px;
            padding: 3px;
        }}
        
        QToolButton {{
            background-color: {COLORS["background_alt"]};
            color: {COLORS["foreground"]};
            border: none;
            border-radius: 4px;
            padding: 4px;
        }}
        
        QToolButton:hover {{
            background-color: {COLORS["border"]};
        }}
        
        QGroupBox {{
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 12px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }}
    """,
    
    "form": f"""
        QLabel {{
            color: {COLORS["foreground"]};
        }}
        
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {COLORS["background_alt"]};
            color: {COLORS["foreground"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            padding: 4px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border: 1px solid {COLORS["accent"]};
        }}
        
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 16px;
            border-left: 1px solid {COLORS["border"]};
            border-bottom: 1px solid {COLORS["border"]};
            border-top-right-radius: 4px;
        }}
        
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 16px;
            border-left: 1px solid {COLORS["border"]};
            border-top: 1px solid {COLORS["border"]};
            border-bottom-right-radius: 4px;
        }}
        
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid {COLORS["border"]};
        }}
        
        QCheckBox {{
            spacing: 8px;
        }}
        
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {COLORS["border"]};
            border-radius: 3px;
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {COLORS["accent"]};
            border: 1px solid {COLORS["accent"]};
        }}
        
        QSlider::groove:horizontal {{
            height: 8px;
            background: {COLORS["background_alt"]};
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background: {COLORS["accent"]};
            border: none;
            width: 16px;
            margin: -4px 0;
            border-radius: 8px;
        }}
        
        QSlider::sub-page:horizontal {{
            background: {COLORS["accent"]};
            border-radius: 4px;
        }}
    """,
    
    "buttons": f"""
        QPushButton {{
            background-color: {COLORS["background_alt"]};
            color: {COLORS["foreground"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            padding: 6px 16px;
        }}
        
        QPushButton:hover {{
            background-color: {COLORS["border"]};
        }}
        
        QPushButton:pressed {{
            background-color: {COLORS["accent"]};
            color: {COLORS["background"]};
        }}
        
        QPushButton:disabled {{
            background-color: {COLORS["background_alt"]}80;
            color: {COLORS["foreground"]}80;
            border: 1px solid {COLORS["border"]}80;
        }}
        
        QPushButton#primaryButton {{
            background-color: {COLORS["accent"]};
            color: {COLORS["background"]};
            border: none;
            font-weight: bold;
        }}
        
        QPushButton#primaryButton:hover {{
            background-color: {COLORS["accent_darker"]};
        }}
        
        QPushButton#successButton {{
            background-color: {COLORS["success"]};
            color: {COLORS["background"]};
            border: none;
        }}
        
        QPushButton#warningButton {{
            background-color: {COLORS["warning"]};
            color: {COLORS["background"]};
            border: none;
        }}
        
        QPushButton#dangerButton {{
            background-color: {COLORS["error"]};
            color: {COLORS["background"]};
            border: none;
        }}
    """,
    
    "progress": f"""
        QProgressBar {{
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            text-align: center;
            background-color: {COLORS["background_alt"]};
        }}
        
        QProgressBar::chunk {{
            background-color: {COLORS["accent"]};
            border-radius: 3px;
        }}
    """,
    
    "lists": f"""
        QListWidget, QTreeWidget, QTableWidget {{
            background-color: {COLORS["background_alt"]};
            color: {COLORS["foreground"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
        }}
        
        QListWidget::item, QTreeWidget::item, QTableWidget::item {{
            padding: 4px;
            border-radius: 4px;
        }}
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
            background-color: {COLORS["accent"]};
            color: {COLORS["background"]};
        }}
        
        QTreeWidget::branch {{
            background-color: transparent;
        }}
    """,
    
    "scrollbars": f"""
        QScrollBar:vertical {{
            border: none;
            background: {COLORS["background_alt"]};
            width: 12px;
            margin: 12px 0 12px 0;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {COLORS["border"]};
            min-height: 20px;
            border-radius: 4px;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 12px;
        }}
        
        QScrollBar:horizontal {{
            border: none;
            background: {COLORS["background_alt"]};
            height: 12px;
            margin: 0 12px 0 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {COLORS["border"]};
            min-width: 20px;
            border-radius: 4px;
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            border: none;
            background: none;
            width: 12px;
        }}
    """
}


def create_dark_theme() -> str:
    """
    Create and return the complete dark theme stylesheet
    
    Returns:
        str: Complete CSS stylesheet for the dark theme
    """
    # Combine all stylesheets
    complete_style = ""
    for style in STYLE_SHEETS.values():
        complete_style += style
    
    return complete_style


def apply_dark_theme(app: QApplication) -> None:
    """
    Apply the dark theme to the entire application
    
    Args:
        app: QApplication instance
    """
    # Set the stylesheet
    app.setStyleSheet(create_dark_theme())
    
    # Set palette colors
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["background"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["foreground"]))
    palette.setColor(QPalette.Base, QColor(COLORS["background_alt"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["background"]))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS["background_alt"]))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS["foreground"]))
    palette.setColor(QPalette.Text, QColor(COLORS["foreground"]))
    palette.setColor(QPalette.Button, QColor(COLORS["background_alt"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["foreground"]))
    palette.setColor(QPalette.BrightText, QColor(COLORS["foreground"]))
    palette.setColor(QPalette.Link, QColor(COLORS["accent"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor(COLORS["background"]))
    
    app.setPalette(palette)
    
    # Set default font
    font = QFont("Segoe UI", 9)
    app.setFont(font)