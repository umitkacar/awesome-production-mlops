# 🏆 MLOps Best Practices 2024-2025

## Overview

This guide covers the essential best practices for building production-ready ML systems in 2024-2025.

## 🔬 Development Phase

### 1. Experiment Tracking

**Always track your experiments!**

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)

    # Train model
    model = train_model(params)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")
```

**Benefits:**
- 📊 Compare experiments easily
- 🔄 Reproduce results
- 👥 Team collaboration
- 📈 Track progress over time

### 2. Data Versioning

**Use DVC or similar tools**

```bash
# Initialize DVC
dvc init

# Track data
dvc add data/raw/dataset.csv

# Create version
git add data/raw/dataset.csv.dvc .dvc/config
git commit -m "Add dataset v1.0"

# Push to remote storage
dvc push
```

### 3. Data Validation

**Validate data early and often**

```python
from great_expectations.dataset import PandasDataset

# Create expectations
df_ge = PandasDataset(df)

# Validate
df_ge.expect_column_values_to_not_be_null("feature1")
df_ge.expect_column_values_to_be_between("age", 0, 120)
df_ge.expect_column_values_to_be_in_set("category", ["A", "B", "C"])

# Save expectations
df_ge.save_expectation_suite("my_suite.json")
```

### 4. Feature Engineering

**Use Feature Stores for consistency**

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Get features for training
training_df = store.get_historical_features(
    entity_df=entities,
    features=[
        "user_features:age",
        "user_features:country",
        "product_features:category"
    ]
).to_df()

# Get features for inference (same feature definitions!)
features = store.get_online_features(
    features=["user_features:age"],
    entity_rows=[{"user_id": 1001}]
).to_dict()
```

## 🚀 Deployment Phase

### 1. Model Registry

**Always register production models**

```python
import mlflow

# Register model
mlflow.register_model(
    model_uri="runs:/{run_id}/model",
    name="ProductionModel"
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="ProductionModel",
    version=3,
    stage="Production"
)
```

### 2. Containerization

**Use Docker for consistency**

```dockerfile
# Use multi-stage builds
FROM python:3.11-slim as base
WORKDIR /app

FROM base as dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM dependencies as production
COPY . .
USER nonroot
CMD ["uvicorn", "main:app"]
```

### 3. CI/CD Pipeline

**Automate everything**

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Train model
        run: python train.py

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: ./deploy.sh
```

### 4. A/B Testing

**Test models in production safely**

```python
from random import random

def predict(request):
    # Route 10% to new model
    if random() < 0.1:
        return new_model.predict(request)
    else:
        return current_model.predict(request)
```

## 📊 Monitoring Phase

### 1. Performance Monitoring

**Track metrics in production**

```python
from evidently.metrics import DataDriftTable
from evidently.report import Report

# Create report
report = Report(metrics=[
    DataDriftTable(),
])

# Run report
report.run(
    reference_data=reference_df,
    current_data=current_df
)

# Save report
report.save_html("drift_report.html")
```

### 2. Data Drift Detection

**Monitor for distribution changes**

```python
import evidently

# Detect drift
drift_report = evidently.ColumnDriftMetric(column_name="feature1")

if drift_report.drift_detected:
    # Alert team
    send_alert("Data drift detected!")
    # Retrain model
    trigger_retraining()
```

### 3. Model Quality Monitoring

**Set up alerts**

```python
def monitor_predictions(predictions, actuals):
    accuracy = calculate_accuracy(predictions, actuals)

    # Alert if below threshold
    if accuracy < 0.85:
        send_alert(f"Model accuracy dropped to {accuracy}")
        trigger_investigation()

    # Log metrics
    mlflow.log_metric("production_accuracy", accuracy)
```

### 4. Logging

**Comprehensive logging**

```python
import logging

logger = logging.getLogger(__name__)

def predict(features):
    logger.info(f"Prediction request: {features}")

    try:
        prediction = model.predict(features)
        logger.info(f"Prediction: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
```

## 🔐 Security & Governance

### 1. Model Security

**Scan models for vulnerabilities**

```bash
# Use ModelScan
pip install modelscan
modelscan scan model.pkl
```

### 2. Access Control

**Implement proper permissions**

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Validate token
    user = validate_token(token)
    if not user:
        raise HTTPException(status_code=401)
    return user

@app.post("/predict")
async def predict(data: Data, user = Depends(get_current_user)):
    return model.predict(data)
```

### 3. Model Explainability

**Make models interpretable**

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model)

# Get explanations
shap_values = explainer.shap_values(X)

# Visualize
shap.summary_plot(shap_values, X)
```

### 4. Audit Logs

**Track all model operations**

```python
def log_prediction(user_id, features, prediction):
    audit_log.write({
        "timestamp": datetime.now(),
        "user_id": user_id,
        "model_version": model.version,
        "features": features,
        "prediction": prediction
    })
```

## 🎯 LLMOps Specific (2024-2025)

### 1. Prompt Versioning

```python
from langchain.prompts import PromptTemplate

# Version prompts like code
PROMPT_V1 = PromptTemplate(
    template="Answer the question: {question}",
    input_variables=["question"]
)

# Track in MLflow
mlflow.log_param("prompt_version", "v1")
mlflow.log_text(PROMPT_V1.template, "prompt.txt")
```

### 2. LLM Monitoring

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = chain.run(query)

    # Log costs and usage
    mlflow.log_metric("total_tokens", cb.total_tokens)
    mlflow.log_metric("total_cost", cb.total_cost)
```

### 3. RAG Quality

```python
# Evaluate retrieval quality
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("embedding_distance")

score = evaluator.evaluate_strings(
    prediction=retrieved_doc,
    reference=ground_truth
)
```

## ✅ Checklist

### Before Training
- [ ] Data validated with Great Expectations
- [ ] Data versioned with DVC
- [ ] Experiment tracking configured
- [ ] Baseline model established

### Before Deployment
- [ ] Model tested on validation set
- [ ] Model registered in registry
- [ ] Docker container built and tested
- [ ] CI/CD pipeline configured
- [ ] Rollback plan in place

### In Production
- [ ] Monitoring dashboard active
- [ ] Drift detection enabled
- [ ] Alerts configured
- [ ] Audit logging enabled
- [ ] Model explainability available

## 📚 Resources

- [MLOps Principles](https://ml-ops.org/content/mlops-principles)
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [AWS MLOps](https://aws.amazon.com/sagemaker/mlops/)

## 🎉 Conclusion

Following these best practices will help you build robust, scalable, and maintainable ML systems!

Remember: **The best MLOps is invisible MLOps** - it just works! 🚀
