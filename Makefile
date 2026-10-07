.PHONY: install install-dev validate-data train evaluate dashboard serve benchmark test lint docker-build docker-run clean

install:
	@echo "Install the correct torch/torchvision build first; see README.md."
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

validate-data:
	python -m src.download_data --root data

train:
	python -m src.train --config config.yaml

evaluate:
	python -m src.evaluate --config config.yaml --checkpoint checkpoints/best.pt

dashboard:
	python -m src.reporting

serve:
	uvicorn src.inference:app --host 0.0.0.0 --port 8000

benchmark:
	python -m src.benchmark --model unet --encoder resnet18

test:
	pytest

lint:
	ruff check src tests

docker-build:
	docker build -t deepcrack-segmentation-pipeline .

docker-run:
	docker run --rm -p 8000:8000 -v "$(PWD)/checkpoints:/app/checkpoints:ro" deepcrack-segmentation-pipeline

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ reports/test_metrics.json reports/experiment_dashboard.png
