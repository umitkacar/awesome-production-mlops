"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sample classification data for testing.

    Returns:
        Tuple of (X, y) arrays
    """
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def train_test_split_data(
    sample_data: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate train/test split data.

    Args:
        sample_data: Sample data fixture

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X, y = sample_data
    split_idx = 80

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_model(
    train_test_split_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> RandomForestClassifier:
    """
    Get a trained Random Forest model.

    Args:
        train_test_split_data: Train/test data fixture

    Returns:
        Trained RandomForestClassifier
    """
    X_train, _, y_train, _ = train_test_split_data

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    return model


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Generate sample DataFrame for testing.

    Returns:
        Sample pandas DataFrame
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        }
    )


@pytest.fixture
def reference_current_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate reference and current data for drift detection.

    Returns:
        Tuple of (reference_df, current_df)
    """
    np.random.seed(42)
    reference = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        }
    )

    # Current data with slight drift
    current = pd.DataFrame(
        {
            "feature1": np.random.randn(100) + 0.1,
            "feature2": np.random.randn(100) + 0.1,
        }
    )

    return reference, current


@pytest.fixture
def mock_tracking_uri() -> str:
    """Get mock MLflow tracking URI."""
    return "http://localhost:5000"
