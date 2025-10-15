.PHONY: help install install-dev test test-cov lint format clean run setup check-env

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy

# Default target
help:
	@echo "Starship Horizons Learning AI - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup          - Complete initial setup"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make check-env      - Check environment configuration"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run all linters"
	@echo "  make format        - Format code with black and isort"
	@echo "  make clean         - Remove generated files"
	@echo ""
	@echo "Quality Checks:"
	@echo "  make check-all     - Run all checks (lint, test, type)"
	@echo "  make check-types   - Run type checking with mypy"
	@echo "  make check-style   - Check code style"
	@echo ""
	@echo "Running Commands:"
	@echo "  make run-record    - Start game recording"
	@echo "  make run-test-ws   - Test WebSocket connection"

# Setup commands
setup: check-env install-dev
	@echo "✅ Setup complete!"
	@echo "Edit .env file to set your GAME_HOST IP address"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

check-env:
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		cp .env.example .env; \
		echo "✅ .env created - Please edit it with your server settings"; \
	else \
		echo "✅ .env file exists"; \
	fi

# Testing commands
test:
	$(PYTEST) tests/

test-cov:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	$(PYTEST) tests/ -m "not integration"

test-integration:
	$(PYTEST) tests/ -m integration

# Code quality commands
lint: check-style check-types
	@echo "✅ All linting checks passed!"

check-all: lint test
	@echo "✅ All checks passed!"

check-style:
	$(BLACK) --check src/ tests/ scripts/
	$(ISORT) --check-only src/ tests/ scripts/
	$(FLAKE8) src/ tests/ scripts/

check-types:
	$(MYPY) src/

format:
	$(BLACK) src/ tests/ scripts/
	$(ISORT) src/ tests/ scripts/
	@echo "✅ Code formatted!"

# Cleaning commands
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	@echo "✅ Cleaned temporary files!"

clean-data:
	@echo "⚠️  This will delete recorded data. Are you sure? [y/N]"
	@read ans && [ $${ans:-N} = y ] && rm -rf data/* logs/* || echo "Cancelled"

# Running commands
run-record:
	$(PYTHON) scripts/record_unified_telemetry.py

run-test-ws:
	$(PYTHON) scripts/test_websocket_live_manual.py

run-test-browser:
	$(PYTHON) scripts/test_browser_mimic_manual.py

# Development shortcuts
dev: format lint test
	@echo "✅ Ready for commit!"

watch-test:
	$(PYTEST) tests/ --watch

# Docker commands (if using dev container)
container-build:
	docker-compose -f .devcontainer/docker-compose.yml build

container-up:
	docker-compose -f .devcontainer/docker-compose.yml up -d

container-down:
	docker-compose -f .devcontainer/docker-compose.yml down

# Documentation
docs:
	@echo "📚 Documentation files:"
	@ls -la docs/

# Show current configuration
show-config:
	@echo "Current Configuration:"
	@echo "====================="
	@if [ -f .env ]; then \
		grep -E "^[^#]" .env | sed 's/=.*$$/ = <configured>/'; \
	else \
		echo "No .env file found!"; \
	fi