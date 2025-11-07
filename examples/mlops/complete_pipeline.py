"""
📊 Complete MLOps Pipeline Example
End-to-end ML pipeline with:
- Data loading and preprocessing
- Experiment tracking with MLflow
- Model training and evaluation
- Model registry
- Workflow orchestration with Prefect
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from prefect import flow, task
from datetime import datetime


@task(name="load-data", retries=3)
def load_data(file_path: str = "data.csv"):
    """
    Load and validate data
    """
    print("📂 Loading data...")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} records")
    return df


@task(name="preprocess-data")
def preprocess_data(df: pd.DataFrame):
    """
    Preprocess and split data
    """
    print("🔧 Preprocessing data...")

    # Feature engineering
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"✅ Train set: {len(X_train)}, Test set: {len(X_test)}")

    return X_train, X_test, y_train, y_test


@task(name="train-model")
def train_model(X_train, y_train, params: dict = None):
    """
    Train ML model with experiment tracking
    """
    print("🎯 Training model...")

    # Default parameters
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }

    # Start MLflow run
    with mlflow.start_run(run_name=f"rf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "random_forest_model",
            registered_model_name="RandomForestClassifier"
        )

        print("✅ Model trained and logged")

        return model


@task(name="evaluate-model")
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and log metrics
    """
    print("📊 Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    # Log metrics to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_metrics(metrics)

    # Print metrics
    print("\n📈 Model Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    return metrics


@task(name="register-model")
def register_model(metrics: dict, threshold: float = 0.8):
    """
    Register model if it meets quality threshold
    """
    if metrics['accuracy'] >= threshold:
        print(f"✅ Model meets quality threshold ({threshold})")
        print("🎉 Model registered successfully!")
        return True
    else:
        print(f"❌ Model does not meet quality threshold ({threshold})")
        return False


@flow(name="ml-training-pipeline", log_prints=True)
def ml_pipeline(
    data_path: str = "data.csv",
    model_params: dict = None,
    quality_threshold: float = 0.8
):
    """
    Complete ML training pipeline orchestrated with Prefect
    """
    print("\n" + "="*80)
    print("🚀 Starting ML Pipeline")
    print("="*80 + "\n")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("ml-pipeline-experiment")

    # Pipeline steps
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train, model_params)
    metrics = evaluate_model(model, X_test, y_test)
    registered = register_model(metrics, quality_threshold)

    print("\n" + "="*80)
    print("✨ Pipeline completed successfully!")
    print("="*80 + "\n")

    return {
        'model': model,
        'metrics': metrics,
        'registered': registered
    }


if __name__ == "__main__":
    # Example usage
    result = ml_pipeline(
        data_path="data.csv",
        model_params={
            'n_estimators': 200,
            'max_depth': 15,
            'random_state': 42
        },
        quality_threshold=0.85
    )

    print(f"\n🎯 Final Results:")
    print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"  Model Registered: {result['registered']}")
