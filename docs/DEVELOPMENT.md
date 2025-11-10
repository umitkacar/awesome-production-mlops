# 🛠️ Development Guide

Welcome to the MLOps Ecosystem development guide! This document covers everything you need to know to contribute to the project.

## 🚀 Quick Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/umitkacar/MLOps.git
cd MLOps

# Install with hatch (recommended)
pip install hatch

# Or install all dev dependencies
pip install -e ".[complete]"
```

### 2. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

## 🔧 Development Tools

This project uses modern Python development tools:

### **Hatch** - Project Management

Hatch is used for project management, virtual environments, and build automation.

```bash
# Create/activate environment
hatch shell

# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# View coverage report
hatch run cov-html

# Format code
hatch run lint:fmt

# Type check
hatch run lint:typing

# Run all checks
hatch run lint:all
```

### **Ruff** - Fast Linter & Formatter

Ruff is an extremely fast Python linter and formatter.

```bash
# Lint code
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code (ruff format)
ruff format .
```

### **Black** - Code Formatter

Black ensures consistent code formatting.

```bash
# Format code
black .

# Check formatting
black --check .

# Format specific files
black src/mlops/core.py
```

### **MyPy** - Type Checking

MyPy provides static type checking.

```bash
# Type check
mypy src/mlops

# Type check tests too
mypy src/mlops tests
```

### **Pytest** - Testing Framework

Pytest is used for all testing.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/mlops

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test
pytest tests/unit/test_core.py::TestModelTrainer::test_train_basic

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run only integration tests

# Verbose output
pytest -v

# Show print statements
pytest -s

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

## 📝 Code Style Guidelines

### Type Annotations

All functions should have type annotations:

```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
) -> Dict[str, Any]:
    """Train a machine learning model."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def process_data(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Process and clean the input data.

    This function applies various preprocessing steps including handling
    missing values and removing outliers.

    Args:
        data: Input DataFrame to process
        threshold: Threshold for outlier detection (default: 0.5)

    Returns:
        Processed DataFrame with cleaned data

    Raises:
        ValueError: If data is empty
        TypeError: If threshold is not a float

    Example:
        >>> df = pd.DataFrame({'col1': [1, 2, 3]})
        >>> cleaned = process_data(df, threshold=0.8)
    """
    if data.empty:
        raise ValueError("Data cannot be empty")

    # Processing logic here
    return data
```

### Import Ordering

Imports should be organized as follows (handled by isort/ruff):

```python
# Standard library
import os
from typing import Any, Dict, Optional

# Third-party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Local
from mlops.core import ModelTrainer
from mlops.monitoring import DriftDetector
```

## 🧪 Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests
│   ├── test_core.py
│   ├── test_monitoring.py
│   └── test_serving.py
└── integration/         # Integration tests
    └── test_pipeline.py
```

### Writing Tests

```python
import pytest
from mlops.core import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    def test_initialization(self) -> None:
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(model=RandomForestClassifier())
        assert trainer.is_trained is False

    def test_train_raises_on_invalid_data(self) -> None:
        """Test that training raises error on invalid data."""
        trainer = ModelTrainer(model=RandomForestClassifier())

        with pytest.raises(ValueError, match="invalid data"):
            trainer.train(X=None, y=None)

    @pytest.mark.slow
    def test_train_large_dataset(self, large_dataset):
        """Test training on large dataset (marked as slow)."""
        ...

    @pytest.mark.integration
    def test_end_to_end_pipeline(self):
        """Integration test for complete pipeline."""
        ...
```

### Using Fixtures

```python
# In conftest.py
@pytest.fixture
def sample_data():
    """Generate sample data."""
    return np.random.randn(100, 10)

# In test file
def test_with_fixture(sample_data):
    """Test using fixture."""
    assert sample_data.shape == (100, 10)
```

### Coverage Requirements

- Minimum coverage: 80%
- Aim for 90%+ on critical modules
- Tests should cover:
  - Happy path (normal usage)
  - Edge cases
  - Error conditions
  - Different input types

## 🔍 Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

- **Ruff**: Linting and auto-fixes
- **Black**: Code formatting
- **MyPy**: Type checking
- **Pytest**: Run tests
- **Security checks**: Bandit, detect-secrets
- **File checks**: Trailing whitespace, EOF, etc.

### Skipping Hooks (Not Recommended)

```bash
# Skip all hooks (emergency only!)
git commit --no-verify

# Skip specific hooks
SKIP=mypy git commit
```

## 📦 Building & Publishing

### Build Package

```bash
# Build using hatch
hatch build

# Build using pip
pip install build
python -m build
```

### Version Management

Version is defined in `src/mlops/__init__.py`:

```python
__version__ = "1.0.0"
```

Update version for releases following [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

## 🐛 Debugging

### Using IPython Debugger

```python
# Add breakpoint in code
import ipdb; ipdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()
```

### Pytest Debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start
pytest --trace
```

## 📊 Performance Profiling

```bash
# Profile with pytest-profiling
pip install pytest-profiling
pytest --profile

# Profile specific code
python -m cProfile -o output.prof script.py
python -m pstats output.prof
```

## 🔄 Git Workflow

### Branch Naming

- `feature/amazing-feature` - New features
- `fix/bug-description` - Bug fixes
- `docs/update-readme` - Documentation
- `refactor/improve-code` - Refactoring
- `test/add-tests` - Adding tests

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat: add support for LLM monitoring
fix: resolve drift detection bug
docs: update development guide
test: add tests for ModelServer
refactor: improve code organization
chore: update dependencies
```

### Pull Request Process

1. Create feature branch
2. Make changes
3. Add tests
4. Run all checks locally
5. Push and create PR
6. Address review comments
7. Merge after approval

## 🎯 CI/CD

GitHub Actions runs automatically on:
- Push to main/develop
- Pull requests

CI pipeline includes:
- Linting (ruff, black)
- Type checking (mypy)
- Tests (pytest)
- Coverage reporting
- Security scanning
- Build verification

## 📚 Additional Resources

- [Hatch Documentation](https://hatch.pypa.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)

## 💡 Tips & Tricks

### Fast Development Loop

```bash
# Watch mode for tests (requires pytest-watch)
pip install pytest-watch
ptw

# Auto-format on save (configure your IDE)
# VS Code: settings.json
{
  "editor.formatOnSave": true,
  "python.formatting.provider": "black"
}
```

### IDE Configuration

#### VS Code

Install extensions:
- Python
- Pylance
- Ruff
- MyPy Type Checker

Settings (`.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true
}
```

#### PyCharm

- Enable Black integration
- Configure Ruff as external tool
- Enable pytest as test runner
- Enable MyPy integration

## 🆘 Getting Help

- Check existing issues
- Read the documentation
- Ask in discussions
- Join our community

Happy coding! 🚀
