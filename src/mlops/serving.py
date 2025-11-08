"""Model serving functionality."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator


class ModelServer:
    """Serve ML models via REST API.

    This class provides a production-ready interface for serving ML models
    with features like versioning, logging, and monitoring.

    Attributes:
        model: The ML model to serve
        version: Model version
        metadata: Additional model metadata

    Example:
        >>> server = ModelServer(model, version="1.0.0")
        >>> prediction = server.predict(features)
    """

    def __init__(
        self,
        model: BaseEstimator,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the ModelServer.

        Args:
            model: Trained ML model
            version: Model version string
            metadata: Additional model metadata
        """
        self.model = model
        self.version = version
        self.metadata = metadata or {}
        self._request_count = 0

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make a prediction with the served model.

        Args:
            features: Input features for prediction

        Returns:
            Dictionary containing prediction and metadata
        """
        self._request_count += 1

        prediction = self.model.predict(features)

        response = {
            "prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction,
            "version": self.version,
            "request_id": self._request_count,
        }

        return response

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model server.

        Returns:
            Dictionary containing health status
        """
        return {
            "status": "healthy",
            "version": self.version,
            "total_requests": self._request_count,
            "metadata": self.metadata,
        }

    @property
    def request_count(self) -> int:
        """Get total number of requests served."""
        return self._request_count
