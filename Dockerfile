# Multi-stage Dockerfile for MLOps Applications
# Optimized for production deployment

# Base stage with Python
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies stage
FROM base as dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM dependencies as development

COPY . .

# Expose ports for various services
EXPOSE 7860 8501 5000 8000

CMD ["bash"]

# Production stage
FROM dependencies as production

# Create non-root user
RUN useradd -m -u 1000 mlops && \
    chown -R mlops:mlops /app

USER mlops

# Copy application code
COPY --chown=mlops:mlops . .

# Health check (optional - uncomment if you have a health endpoint)
# HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Default command - starts a bash shell
# Override this command when running specific services
CMD ["bash"]
