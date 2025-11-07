# Car Price Prediction MLOps Platform

An industrial-grade MLOps system that predicts used car prices using machine learning. Features complete automation from data processing to production API with monitoring and auto-retraining.

## What This Project Does

- **Automated Data Pipeline**: Loads, validates, cleans, and processes car data
- **Machine Learning**: Trains multiple models (Linear, Ridge, Lasso, ElasticNet) and selects the best performer
- **Production API**: Serves predictions via REST API with proper input encoding
- **MLOps Automation**: Includes data validation, experiment tracking, drift detection, and auto-retraining
- **Containerized**: Ready for deployment with Docker and AWS

## Quick Start

### 1. Installation
```bash
git clone <your-repo>
cd car_price_mlops
python -m venv venv
source venv/bin/activate
pip install -e .
