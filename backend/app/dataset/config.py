from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Dataset location
RAW_DATA_PATH = PROJECT_ROOT / "backend" / "data" / "raw"


# Metadata output location
METADATA_PATH = PROJECT_ROOT / "backend" / "data" / "metadata"


# Create metadata folder if it does not exist
METADATA_PATH.mkdir(parents=True, exist_ok=True)