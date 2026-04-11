"""
Tab (VisionTab) — multimodal tab-completion training utilities.

Dataset preparation (AST-aware FIM, synthetic IDE screenshots) and helpers
for Gemma 4 E4B–style fine-tuning. Training recipes live under ``notebooks/``.
"""

__version__ = "0.1.0"

from .fim_transform import FIMTransformer
from .prepare_dataset import DatasetPipeline
from .screenshot_renderer import ScreenshotRenderer
from .multimodal_dataset import MultimodalPipeline

__all__ = [
    "FIMTransformer",
    "DatasetPipeline",
    "ScreenshotRenderer",
    "MultimodalPipeline",
    "__version__",
]
