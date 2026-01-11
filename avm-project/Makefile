# Makefile for AVM project

.PHONY: help install test clean train predict deploy lint format

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean artifacts and cache"
	@echo "  make train      - Train model"
	@echo "  make predict    - Make predictions"
	@echo "  make deploy     - Deploy to Vertex AI"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov black flake8 mypy

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/ -v -m "not integration"

test-integration:
	pytest tests/ -v -m integration

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build

train:
	python scripts/train.py

train-csv:
	python scripts/train.py --csv data/properties.csv

predict:
	python scripts/predict.py --input data/new_properties.csv --output predictions.csv

upload-gcs:
	python scripts/upload_to_gcs.py

deploy:
	cd deployment && ./deploy.sh

lint:
	flake8 src/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ scripts/ tests/ --line-length=100

docker-build:
	cd deployment && docker build -t avm-predictor .

docker-run:
	docker run -p 8080:8080 avm-predictor