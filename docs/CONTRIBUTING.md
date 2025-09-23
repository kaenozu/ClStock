# Contributing to ClStock

Thank you for your interest in contributing to ClStock! This document provides guidelines and procedures for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Requests](#pull-requests)
8. [Release Process](#release-process)

## Code of Conduct

This project follows a code of conduct focused on creating a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip for package management
- Git for version control

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/clstock.git
   cd clstock
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Project Structure

```
clstock/
├── api/              # API endpoints and security
├── app/              # Web application and dashboard
├── data/             # Data processing and providers
├── docs/             # Documentation files
├── models/           # Prediction models and algorithms
├── tests/            # Unit and integration tests
├── utils/            # Utility functions and helpers
├── config/           # Configuration files
└── monitoring/       # System monitoring tools
```

## Development Workflow

1. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Run tests** to ensure nothing is broken:
   ```bash
   pytest
   ```

4. **Commit your changes** with a descriptive message:
   ```bash
   git commit -m "Add feature: brief description of what was added"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a pull request** on GitHub

## Code Style

### Python Style Guide

We follow PEP 8 with some additional conventions:

1. **Line Length**: Maximum 88 characters (compatible with Black)
2. **Naming Conventions**:
   - Variables and functions: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_SNAKE_CASE`
   - Private members: prefixed with `_`

3. **Type Annotations**: All functions should have type annotations
   ```python
   def calculate_score(symbol: str) -> float:
       # Implementation
   ```

4. **Docstrings**: Use Google-style docstrings for all public functions
   ```python
   def predict_return_rate(symbol: str) -> float:
       """Predict the return rate for a given stock symbol.
       
       Args:
           symbol: Stock symbol to predict
            
       Returns:
           Predicted return rate as a decimal (e.g., 0.05 for 5%)
       """
   ```

### Code Formatting

We use Black for code formatting and Flake8 for linting:

```bash
# Format code
black .

# Check for linting issues
flake8 .
```

### Type Checking

We use MyPy for static type checking:

```bash
mypy .
```

## Testing

### Test Structure

Tests are organized in the `tests/` directory following the same structure as the source code.

### Writing Tests

1. Use pytest as the testing framework
2. Write both unit and integration tests
3. Follow the AAA pattern (Arrange, Act, Assert)
4. Use descriptive test names

```python
def test_calculate_score_positive_trend():
    # Arrange
    predictor = StockPredictor()
    
    # Act
    score = predictor.calculate_score("7203")
    
    # Assert
    assert score > 50
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_predictor.py
```

### Test Coverage

We aim for >90% test coverage. Check coverage with:

```bash
pytest --cov=. --cov-report=html
```

## Documentation

### Code Documentation

1. All public functions and classes should have docstrings
2. Complex algorithms should have inline comments
3. Configuration parameters should be documented

### User Documentation

1. Update README.md for user-facing changes
2. Add API documentation for new endpoints
3. Update architectural documentation for significant changes

## Pull Requests

### PR Requirements

1. All tests must pass
2. Code must follow style guidelines
3. Type checking must pass
4. Documentation must be updated
5. PR must have a clear, descriptive title
6. PR must include a summary of changes

### PR Review Process

1. PR is assigned to maintainers for review
2. At least one approval is required before merging
3. All CI checks must pass
4. Large changes may require multiple reviewers

### Merge Process

1. Squash and merge for feature branches
2. Maintain linear commit history
3. Delete branch after merging

## Release Process

### Versioning

We follow Semantic Versioning (SemVer):
- MAJOR version for incompatible API changes
- MINOR version for backward-compatible functionality
- PATCH version for backward-compatible bug fixes

### Release Steps

1. Update version in `config/settings.py`
2. Update CHANGELOG.md
3. Create and push a new tag
4. Create GitHub release
5. Update documentation if needed

## Getting Help

If you need help or have questions:

1. Check existing issues and pull requests
2. Open a new issue with your question
3. Tag maintainers if urgent

## Recognition

All contributors will be recognized in:
- Release notes
- CONTRIBUTORS file
- Project documentation