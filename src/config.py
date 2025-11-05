import os
from pathlib import Path

#Project paths
PROJECT_ROOT=Path(__file__).parent.parent
SRC_DIR=PROJECT_ROOT/'src'
DATA_DIR=PROJECT_ROOT/'data'
MODELS_DIR = PROJECT_ROOT / "models"  # Asset directory - no __init__.py
LOGS_DIR = PROJECT_ROOT / "logs"


# Data files
RAW_DATA_PATH = DATA_DIR / "raw" / "used_cars_UK.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_cars.csv"

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# Features configuration  
TARGET_COLUMN = "Price"
DROP_COLUMNS = ["Unnamed: 0", "Emission Class", "Service history"]