"""
Vibration-Based Pump Fault Detection Package

This package provides tools for:
- Feature extraction from vibration signals
- Supervised fault classification
- Unsupervised anomaly detection
- Online monitoring and deployment
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .feature_extraction import (
    VibrationFeatureExtractor,
    preprocess_signal,
    segment_signal,
    extract_features_from_multiaxis
)

from .supervised_pipeline import PumpFaultClassifier
from .unsupervised_pipeline import AnomalyDetector, StatisticalProcessControl

__all__ = [
    'VibrationFeatureExtractor',
    'preprocess_signal',
    'segment_signal',
    'extract_features_from_multiaxis',
    'PumpFaultClassifier',
    'AnomalyDetector',
    'StatisticalProcessControl'
]
