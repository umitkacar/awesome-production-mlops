# 🚀 Quick Start Guide

## Welcome to MLOps Ecosystem 2024-2025!

This guide will help you get started with the most modern MLOps tools and practices.

## 📋 Prerequisites

- Python 3.8 or higher
- Docker Desktop (optional, for containerized deployment)
- Git

## ⚡ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/umitkacar/MLOps.git
cd MLOps
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv mlops-env

# Activate on Linux/Mac
source mlops-env/bin/activate

# Activate on Windows
mlops-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🎯 Quick Examples

### Example 1: Run Gradio Demo

```bash
python examples/ui/gradio_demo.py
```

Visit: http://localhost:7860

### Example 2: Launch Streamlit Dashboard

```bash
streamlit run examples/ui/streamlit_app.py
```

Visit: http://localhost:8501

### Example 3: Complete ML Pipeline

```bash
# Start MLflow tracking server first
mlflow server --host 0.0.0.0 --port 5000

# In another terminal, run the pipeline
python examples/mlops/complete_pipeline.py
```

### Example 4: LLM with RAG (Requires API keys)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Start Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant

# Run RAG example
python examples/llmops/rag_example.py
```

## 🐳 Docker Quick Start

### Launch Complete MLOps Stack

```bash
# Start all services
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f
```

### Access Services

- **MLflow**: http://localhost:5000
- **Qdrant**: http://localhost:6333/dashboard
- **Prefect**: http://localhost:4200
- **Gradio**: http://localhost:7860
- **Streamlit**: http://localhost:8501

### Stop Services

```bash
docker-compose down
```

## 📚 Next Steps

1. Explore the [examples](../examples/) directory
2. Read the [Best Practices](./BEST_PRACTICES.md) guide
3. Check out the [Architecture](./ARCHITECTURE.md) documentation
4. Join our [Community](../README.md#-community--resources)

## 🆘 Troubleshooting

### Common Issues

**Import errors?**
```bash
pip install -r requirements.txt --force-reinstall
```

**Port already in use?**
```bash
# Check what's using the port
lsof -i :5000  # or your port number

# Kill the process or use a different port
```

**Docker issues?**
```bash
# Clean up Docker
docker system prune -a

# Restart Docker Desktop
```

## 🎉 You're Ready!

Start building amazing ML systems with the latest tools and best practices!

Happy MLOps! 🚀
