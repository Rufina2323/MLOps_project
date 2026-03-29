import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    Path(__file__).resolve().parent.parent
))

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "ridge_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
DVC_FILE = MODEL_DIR / "models.dvc"

MODEL_DVC_FILE = PROJECT_ROOT / "models.dvc"
