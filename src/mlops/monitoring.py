"""Model monitoring and drift detection."""

from typing import Optional

import numpy as np
import pandas as pd


class DriftDetector:
    """Detect data drift in production ML systems.

    This class provides statistical methods to detect distribution shifts
    in input features and model predictions.

    Attributes:
        threshold: Statistical threshold for drift detection
        reference_data: Reference dataset for comparison

    Example:
        >>> detector = DriftDetector(threshold=0.05)
        >>> detector.fit(reference_data)
        >>> is_drift = detector.detect_drift(current_data)
    """

    def __init__(self, threshold: float = 0.05) -> None:
        """Initialize the DriftDetector.

        Args:
            threshold: P-value threshold for drift detection
        """
        self.threshold = threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(self, reference_data: pd.DataFrame) -> None:
        """Fit the drift detector on reference data.

        Args:
            reference_data: Reference dataset
        """
        self.reference_data = reference_data.copy()
        self._is_fitted = True

    def detect_drift(self, current_data: pd.DataFrame) -> bool:
        """Detect if drift has occurred.

        Args:
            current_data: Current production data

        Returns:
            True if drift is detected, False otherwise

        Raises:
            RuntimeError: If detector is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("DriftDetector must be fitted before detecting drift")

        # Simplified drift detection logic
        # In production, use statistical tests like KS-test
        if self.reference_data is None:
            return False

        ref_mean = self.reference_data.mean().values
        curr_mean = current_data.mean().values

        # Simple threshold-based detection
        drift_detected = bool(np.any(np.abs(ref_mean - curr_mean) > self.threshold))

        return drift_detected


class ModelMonitor:
    """Monitor ML model performance in production.

    This class tracks model predictions, calculates performance metrics,
    and triggers alerts when performance degrades.

    Example:
        >>> monitor = ModelMonitor()
        >>> monitor.log_prediction(features, prediction, actual)
        >>> metrics = monitor.get_metrics()
    """

    def __init__(self, alert_threshold: float = 0.8) -> None:
        """Initialize the ModelMonitor.

        Args:
            alert_threshold: Accuracy threshold for alerts
        """
        self.alert_threshold = alert_threshold
        self.predictions: list[float] = []
        self.actuals: list[float] = []

    def log_prediction(
        self,
        features: np.ndarray,
        prediction: float,
        actual: Optional[float] = None,
    ) -> None:
        """Log a model prediction.

        Args:
            features: Input features
            prediction: Model prediction
            actual: Actual ground truth (if available)
        """
        self.predictions.append(prediction)
        if actual is not None:
            self.actuals.append(actual)

    def get_metrics(self) -> dict[str, float]:
        """Calculate current performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.predictions:
            return {}

        metrics = {
            "total_predictions": len(self.predictions),
            "mean_prediction": float(np.mean(self.predictions)),
        }

        if self.actuals:
            accuracy = float(
                np.mean(
                    [
                        1 if p == a else 0
                        for p, a in zip(self.predictions, self.actuals)
                    ],
                ),
            )
            metrics["accuracy"] = accuracy

        return metrics

    def check_alerts(self) -> bool:
        """Check if alerts should be triggered.

        Returns:
            True if alert should be triggered
        """
        metrics = self.get_metrics()
        if "accuracy" in metrics:
            return bool(metrics["accuracy"] < self.alert_threshold)
        return False
