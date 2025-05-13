"""
HiNeRV GUI Components
"""

from .video_preview import VideoPreviewWidget
from .model_presets import ModelPresetsWidget
from .training_options import TrainingOptionsWidget
from .resource_guard import ResourceGuardWidget
from .progress_widget import ProgressWidget
from .results_widget import ResultsWidget
from .history_widget import HistoryWidget
from .log_viewer import LogViewerWidget

__all__ = [
    'VideoPreviewWidget',
    'ModelPresetsWidget',
    'TrainingOptionsWidget',
    'ResourceGuardWidget',
    'ProgressWidget',
    'ResultsWidget',
    'HistoryWidget',
    'LogViewerWidget'
]