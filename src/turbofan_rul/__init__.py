"""
turbofan_rul — shared data pipeline, model, loss, and inference code for the
NASA C-MAPSS Turbofan RUL project. Imported by both the training script
(training/train_v4.py) and the serving backend (backend/app/main.py) so
preprocessing, loss, and inference logic have exactly one implementation.
"""
from .config import CONFIG, PipelineConfig
from .losses import CUSTOM_OBJECTS, asymmetric_huber_loss

__all__ = ["CONFIG", "PipelineConfig", "CUSTOM_OBJECTS", "asymmetric_huber_loss"]
