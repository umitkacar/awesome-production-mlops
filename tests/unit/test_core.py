"""Unit tests for core MLOps functionality."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from mlops.core import ModelTrainer, PipelineOrchestrator


class TestModelTrainer:
    """Test suite for ModelTrainer class."""

    def test_initialization(self, mock_tracking_uri: str) -> None:
        """Test ModelTrainer initialization."""
        model = RandomForestClassifier()
        trainer = ModelTrainer(
            model=model,
            tracking_uri=mock_tracking_uri,
            experiment_name="test-experiment",
        )

        assert trainer.model == model
        assert trainer.tracking_uri == mock_tracking_uri
        assert trainer.experiment_name == "test-experiment"
        assert not trainer.is_trained

    def test_train_basic(
        self,
        train_test_split_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        mock_tracking_uri: str,
    ) -> None:
        """Test basic model training."""
        X_train, _, y_train, _ = train_test_split_data

        trainer = ModelTrainer(
            model=RandomForestClassifier(random_state=42),
            tracking_uri=mock_tracking_uri,
        )

        metrics = trainer.train(X_train, y_train)

        assert trainer.is_trained
        assert "train_score" in metrics
        assert 0 <= metrics["train_score"] <= 1

    def test_train_with_validation(
        self,
        train_test_split_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        mock_tracking_uri: str,
    ) -> None:
        """Test model training with validation data."""
        X_train, X_test, y_train, y_test = train_test_split_data

        trainer = ModelTrainer(
            model=RandomForestClassifier(random_state=42),
            tracking_uri=mock_tracking_uri,
        )

        metrics = trainer.train(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
        )

        assert "train_score" in metrics
        assert "val_score" in metrics
        assert 0 <= metrics["val_score"] <= 1

    def test_train_invalid_data(self, mock_tracking_uri: str) -> None:
        """Test training with mismatched data shapes."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1, 2])  # Wrong shape

        trainer = ModelTrainer(
            model=RandomForestClassifier(),
            tracking_uri=mock_tracking_uri,
        )

        with pytest.raises(ValueError, match="same number of samples"):
            trainer.train(X, y)

    def test_predict_before_training(
        self,
        sample_data: tuple[np.ndarray, np.ndarray],
        mock_tracking_uri: str,
    ) -> None:
        """Test prediction before training raises error."""
        X, _ = sample_data

        trainer = ModelTrainer(
            model=RandomForestClassifier(),
            tracking_uri=mock_tracking_uri,
        )

        with pytest.raises(RuntimeError, match="must be trained"):
            trainer.predict(X)

    def test_predict_after_training(
        self,
        train_test_split_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        mock_tracking_uri: str,
    ) -> None:
        """Test prediction after training."""
        X_train, X_test, y_train, _ = train_test_split_data

        trainer = ModelTrainer(
            model=RandomForestClassifier(random_state=42),
            tracking_uri=mock_tracking_uri,
        )

        trainer.train(X_train, y_train)
        predictions = trainer.predict(X_test)

        assert len(predictions) == len(X_test)
        assert predictions.dtype in [np.int32, np.int64]


class TestPipelineOrchestrator:
    """Test suite for PipelineOrchestrator class."""

    def test_initialization(self) -> None:
        """Test PipelineOrchestrator initialization."""
        orchestrator = PipelineOrchestrator(pipeline_name="test-pipeline")

        assert orchestrator.pipeline_name == "test-pipeline"
        assert orchestrator.steps == []

    def test_add_step(self) -> None:
        """Test adding steps to pipeline."""
        orchestrator = PipelineOrchestrator()

        orchestrator.add_step("load_data")
        orchestrator.add_step("preprocess")
        orchestrator.add_step("train")

        assert len(orchestrator.steps) == 3
        assert orchestrator.steps == ["load_data", "preprocess", "train"]

    def test_run_pipeline(self) -> None:
        """Test running the pipeline."""
        orchestrator = PipelineOrchestrator(pipeline_name="ml-pipeline")

        orchestrator.add_step("step1")
        orchestrator.add_step("step2")

        result = orchestrator.run_pipeline(param1="value1")

        assert result["pipeline_name"] == "ml-pipeline"
        assert result["steps_executed"] == 2
        assert result["status"] == "success"
