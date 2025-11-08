"""
MLOps Ecosystem - Production-Ready ML Systems.

A comprehensive toolkit for building, deploying, and monitoring machine learning systems
with 2024-2025 best practices.
"""

__version__ = "1.0.0"
__author__ = "Umit Kacar"
__email__ = "umitkacar@example.com"

from mlops.core import ModelTrainer, PipelineOrchestrator
from mlops.monitoring import DriftDetector, ModelMonitor
from mlops.serving import ModelServer

__all__ = [
    "ModelTrainer",
    "PipelineOrchestrator",
    "DriftDetector",
    "ModelMonitor",
    "ModelServer",
    "__version__",
]
