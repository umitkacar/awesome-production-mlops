"""Unit tests for model serving functionality."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from mlops.serving import ModelServer


class TestModelServer:
    """Test suite for ModelServer class."""

    def test_initialization(self, trained_model: RandomForestClassifier) -> None:
        """Test ModelServer initialization."""
        metadata = {"model_type": "RandomForest", "features": 10}
        server = ModelServer(
            model=trained_model,
            version="2.0.0",
            metadata=metadata,
        )

        assert server.model == trained_model
        assert server.version == "2.0.0"
        assert server.metadata == metadata
        assert server.request_count == 0

    def test_initialization_default_values(
        self,
        trained_model: RandomForestClassifier,
    ) -> None:
        """Test ModelServer initialization with defaults."""
        server = ModelServer(model=trained_model)

        assert server.version == "1.0.0"
        assert server.metadata == {}

    def test_predict(
        self,
        trained_model: RandomForestClassifier,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test making predictions."""
        X, _ = sample_data
        server = ModelServer(model=trained_model, version="1.0.0")

        response = server.predict(X[:1])

        assert "prediction" in response
        assert "version" in response
        assert "request_id" in response
        assert response["version"] == "1.0.0"
        assert response["request_id"] == 1

    def test_predict_increments_counter(
        self,
        trained_model: RandomForestClassifier,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that predictions increment request counter."""
        X, _ = sample_data
        server = ModelServer(model=trained_model)

        initial_count = server.request_count
        server.predict(X[:1])
        server.predict(X[:1])

        assert server.request_count == initial_count + 2

    def test_predict_multiple_samples(
        self,
        trained_model: RandomForestClassifier,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test predictions with multiple samples."""
        X, _ = sample_data
        server = ModelServer(model=trained_model)

        response = server.predict(X[:5])

        assert "prediction" in response
        predictions = response["prediction"]
        assert len(predictions) == 5

    def test_health_check(self, trained_model: RandomForestClassifier) -> None:
        """Test health check endpoint."""
        metadata = {"deployed": "2024-11-08"}
        server = ModelServer(
            model=trained_model,
            version="1.5.0",
            metadata=metadata,
        )

        health = server.health_check()

        assert health["status"] == "healthy"
        assert health["version"] == "1.5.0"
        assert health["total_requests"] == 0
        assert health["metadata"] == metadata

    def test_health_check_after_requests(
        self,
        trained_model: RandomForestClassifier,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test health check after serving requests."""
        X, _ = sample_data
        server = ModelServer(model=trained_model)

        # Make some predictions
        server.predict(X[:1])
        server.predict(X[:1])
        server.predict(X[:1])

        health = server.health_check()

        assert health["total_requests"] == 3

    def test_request_count_property(
        self,
        trained_model: RandomForestClassifier,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test request_count property."""
        X, _ = sample_data
        server = ModelServer(model=trained_model)

        assert server.request_count == 0

        server.predict(X[:1])
        assert server.request_count == 1

        server.predict(X[:2])
        assert server.request_count == 2
