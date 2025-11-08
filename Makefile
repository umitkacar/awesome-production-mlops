# Makefile for MLOps Ecosystem
# Modern development workflow automation

.PHONY: help install dev-install test test-cov lint format type-check clean docs docker pre-commit all

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo '$(BLUE)MLOps Ecosystem - Development Commands$(NC)'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Installation
# ============================================================================

install: ## Install package in production mode
	@echo '$(BLUE)Installing package...$(NC)'
	pip install -e .

dev-install: ## Install package with development dependencies
	@echo '$(BLUE)Installing development environment...$(NC)'
	pip install -e ".[complete]"
	pre-commit install
	@echo '$(GREEN)✓ Development environment ready!$(NC)'

# ============================================================================
# Testing
# ============================================================================

test: ## Run tests with pytest
	@echo '$(BLUE)Running tests...$(NC)'
	pytest tests/ -v

test-unit: ## Run only unit tests
	@echo '$(BLUE)Running unit tests...$(NC)'
	pytest tests/unit/ -v

test-integration: ## Run only integration tests
	@echo '$(BLUE)Running integration tests...$(NC)'
	pytest tests/integration/ -v -m integration

test-cov: ## Run tests with coverage report
	@echo '$(BLUE)Running tests with coverage...$(NC)'
	pytest --cov=src/mlops --cov-report=html --cov-report=term --cov-report=xml
	@echo '$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)'

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo '$(BLUE)Running tests in watch mode...$(NC)'
	ptw -- -v

cov-report: ## Open coverage report in browser
	@echo '$(BLUE)Opening coverage report...$(NC)'
	python -m http.server -d htmlcov 8080

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run all linters (ruff)
	@echo '$(BLUE)Running linters...$(NC)'
	ruff check .
	@echo '$(GREEN)✓ Linting complete!$(NC)'

lint-fix: ## Auto-fix linting issues
	@echo '$(BLUE)Auto-fixing linting issues...$(NC)'
	ruff check --fix .
	@echo '$(GREEN)✓ Auto-fixes applied!$(NC)'

format: ## Format code with black and ruff
	@echo '$(BLUE)Formatting code...$(NC)'
	black .
	ruff format .
	@echo '$(GREEN)✓ Code formatted!$(NC)'

format-check: ## Check code formatting without changes
	@echo '$(BLUE)Checking code format...$(NC)'
	black --check .
	ruff format --check .

type-check: ## Run type checking with mypy
	@echo '$(BLUE)Running type checker...$(NC)'
	mypy src/mlops
	@echo '$(GREEN)✓ Type checking complete!$(NC)'

security: ## Run security checks (bandit, safety)
	@echo '$(BLUE)Running security checks...$(NC)'
	bandit -r src/mlops
	safety check
	@echo '$(GREEN)✓ Security checks complete!$(NC)'

all-checks: lint type-check test ## Run all quality checks
	@echo '$(GREEN)✓ All checks passed!$(NC)'

# ============================================================================
# Pre-commit
# ============================================================================

pre-commit: ## Run pre-commit hooks on all files
	@echo '$(BLUE)Running pre-commit hooks...$(NC)'
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo '$(BLUE)Updating pre-commit hooks...$(NC)'
	pre-commit autoupdate

# ============================================================================
# Documentation
# ============================================================================

docs: ## Build documentation
	@echo '$(BLUE)Building documentation...$(NC)'
	cd docs && mkdocs build

docs-serve: ## Serve documentation locally
	@echo '$(BLUE)Serving documentation at http://localhost:8000$(NC)'
	cd docs && mkdocs serve

# ============================================================================
# Docker
# ============================================================================

docker-build: ## Build Docker image
	@echo '$(BLUE)Building Docker image...$(NC)'
	docker build -t mlops-ecosystem:latest .

docker-run: ## Run Docker container
	@echo '$(BLUE)Running Docker container...$(NC)'
	docker run -p 8000:8000 mlops-ecosystem:latest

docker-compose-up: ## Start all services with docker-compose
	@echo '$(BLUE)Starting services...$(NC)'
	docker-compose up -d
	@echo '$(GREEN)✓ Services started!$(NC)'
	@echo 'MLflow: http://localhost:5000'
	@echo 'Qdrant: http://localhost:6333'
	@echo 'Prefect: http://localhost:4200'

docker-compose-down: ## Stop all services
	@echo '$(BLUE)Stopping services...$(NC)'
	docker-compose down

docker-compose-logs: ## Show service logs
	docker-compose logs -f

# ============================================================================
# Cleaning
# ============================================================================

clean: ## Clean build artifacts and cache
	@echo '$(BLUE)Cleaning...$(NC)'
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	@echo '$(GREEN)✓ Cleaned!$(NC)'

clean-all: clean ## Clean everything including virtual environments
	rm -rf venv/
	rm -rf .venv/
	rm -rf .hatch/

# ============================================================================
# Building & Publishing
# ============================================================================

build: clean ## Build distribution packages
	@echo '$(BLUE)Building package...$(NC)'
	hatch build
	@echo '$(GREEN)✓ Build complete! Check dist/$(NC)'

publish-test: build ## Publish to TestPyPI
	@echo '$(BLUE)Publishing to TestPyPI...$(NC)'
	hatch publish -r test

publish: build ## Publish to PyPI
	@echo '$(RED)Publishing to PyPI...$(NC)'
	hatch publish

# ============================================================================
# Development Utilities
# ============================================================================

deps-update: ## Update all dependencies
	@echo '$(BLUE)Updating dependencies...$(NC)'
	pip install --upgrade pip
	pip install --upgrade -e ".[complete]"

deps-list: ## List installed dependencies
	pip list

deps-tree: ## Show dependency tree (requires pipdeptree)
	pip install pipdeptree
	pipdeptree

shell: ## Start IPython shell
	ipython

notebook: ## Start Jupyter notebook
	jupyter notebook

# ============================================================================
# Quick Commands
# ============================================================================

quick: lint-fix test ## Quick check: format and test
	@echo '$(GREEN)✓ Quick check complete!$(NC)'

ci: all-checks ## Run all CI checks locally
	@echo '$(GREEN)✓ All CI checks passed!$(NC)'

setup: dev-install ## Initial setup for new contributors
	@echo '$(GREEN)✓ Setup complete! Run "make help" to see available commands$(NC)'

# ============================================================================
# Examples
# ============================================================================

run-gradio: ## Run Gradio demo
	python examples/ui/gradio_demo.py

run-streamlit: ## Run Streamlit dashboard
	streamlit run examples/ui/streamlit_app.py

run-pipeline: ## Run example ML pipeline
	python examples/mlops/complete_pipeline.py

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help
