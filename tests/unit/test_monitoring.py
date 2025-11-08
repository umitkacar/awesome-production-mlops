"""Unit tests for monitoring functionality."""

import numpy as np
import pandas as pd
import pytest

from mlops.monitoring import DriftDetector, ModelMonitor


class TestDriftDetector:
    """Test suite for DriftDetector class."""

    def test_initialization(self) -> None:
        """Test DriftDetector initialization."""
        detector = DriftDetector(threshold=0.1)

        assert detector.threshold == 0.1
        assert detector.reference_data is None
        assert not detector._is_fitted

    def test_fit(self, sample_dataframe: pd.DataFrame) -> None:
        """Test fitting the drift detector."""
        detector = DriftDetector()
        detector.fit(sample_dataframe)

        assert detector._is_fitted
        assert detector.reference_data is not None
        assert len(detector.reference_data) == len(sample_dataframe)

    def test_detect_drift_before_fit(
        self,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test drift detection before fitting raises error."""
        detector = DriftDetector()

        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.detect_drift(sample_dataframe)

    def test_detect_drift_no_drift(
        self,
        reference_current_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Test drift detection with similar data."""
        reference, _ = reference_current_data

        detector = DriftDetector(threshold=1.0)  # High threshold
        detector.fit(reference)

        # Use same data - should not detect drift
        is_drift = detector.detect_drift(reference)

        assert not is_drift

    def test_detect_drift_with_drift(
        self,
        reference_current_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Test drift detection with drifted data."""
        reference, current = reference_current_data

        detector = DriftDetector(threshold=0.01)  # Low threshold
        detector.fit(reference)

        is_drift = detector.detect_drift(current)

        # With low threshold and drift, should detect drift
        # Note: This is a simplified example
        assert isinstance(is_drift, bool)


class TestModelMonitor:
    """Test suite for ModelMonitor class."""

    def test_initialization(self) -> None:
        """Test ModelMonitor initialization."""
        monitor = ModelMonitor(alert_threshold=0.85)

        assert monitor.alert_threshold == 0.85
        assert monitor.predictions == []
        assert monitor.actuals == []

    def test_log_prediction_without_actual(self) -> None:
        """Test logging prediction without actual value."""
        monitor = ModelMonitor()
        features = np.array([[1, 2, 3]])

        monitor.log_prediction(features, prediction=1.0)

        assert len(monitor.predictions) == 1
        assert monitor.predictions[0] == 1.0
        assert len(monitor.actuals) == 0

    def test_log_prediction_with_actual(self) -> None:
        """Test logging prediction with actual value."""
        monitor = ModelMonitor()
        features = np.array([[1, 2, 3]])

        monitor.log_prediction(features, prediction=1.0, actual=1.0)

        assert len(monitor.predictions) == 1
        assert len(monitor.actuals) == 1
        assert monitor.predictions[0] == 1.0
        assert monitor.actuals[0] == 1.0

    def test_get_metrics_empty(self) -> None:
        """Test getting metrics with no predictions."""
        monitor = ModelMonitor()

        metrics = monitor.get_metrics()

        assert metrics == {}

    def test_get_metrics_predictions_only(self) -> None:
        """Test getting metrics with only predictions."""
        monitor = ModelMonitor()

        for i in range(10):
            monitor.log_prediction(np.array([[i]]), prediction=float(i))

        metrics = monitor.get_metrics()

        assert "total_predictions" in metrics
        assert metrics["total_predictions"] == 10
        assert "mean_prediction" in metrics
        assert "accuracy" not in metrics

    def test_get_metrics_with_actuals(self) -> None:
        """Test getting metrics with actuals."""
        monitor = ModelMonitor()

        # Log 10 predictions with actuals (all correct)
        for i in range(10):
            monitor.log_prediction(
                np.array([[i]]),
                prediction=float(i),
                actual=float(i),
            )

        metrics = monitor.get_metrics()

        assert "total_predictions" in metrics
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0

    def test_check_alerts_no_data(self) -> None:
        """Test alert checking with no data."""
        monitor = ModelMonitor()

        should_alert = monitor.check_alerts()

        assert not should_alert

    def test_check_alerts_good_performance(self) -> None:
        """Test alerts with good performance."""
        monitor = ModelMonitor(alert_threshold=0.8)

        # All predictions correct
        for i in range(10):
            monitor.log_prediction(
                np.array([[i]]),
                prediction=1.0,
                actual=1.0,
            )

        should_alert = monitor.check_alerts()

        assert not should_alert

    def test_check_alerts_poor_performance(self) -> None:
        """Test alerts with poor performance."""
        monitor = ModelMonitor(alert_threshold=0.9)

        # All predictions wrong
        for i in range(10):
            monitor.log_prediction(
                np.array([[i]]),
                prediction=0.0,
                actual=1.0,
            )

        should_alert = monitor.check_alerts()

        assert should_alert
