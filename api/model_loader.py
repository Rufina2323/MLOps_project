import subprocess
import pickle
import json
import logging
from typing import Optional
import numpy as np

from config import PROJECT_ROOT, MODEL_PATH, METADATA_PATH

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata: Optional[dict] = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def pull_from_dvc(self) -> None:
        logger.info("Pulling model from DVC...")

        result = subprocess.run(
            ["dvc", "pull", "-v"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        logger.info(f"DVC pull stdout: {result.stdout}")
        if result.returncode != 0:
            logger.warning(f"DVC pull stderr: {result.stderr}")

    def load(self) -> None:
        """Загружает модель: сначала pull из DVC, затем десериализация."""

        if not MODEL_PATH.exists():
            logger.info(f"Model not found at {MODEL_PATH}, pulling from DVC...")
            try:
                self.pull_from_dvc()
            except Exception as e:
                logger.error(f"DVC pull failed: {e}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Make sure the model is tracked by DVC and remote is configured."
            )

        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.model = data["model"]
            self.scaler = data.get("scaler")
        else:
            self.model = data
            self.scaler = None

        logger.info(f"Model loaded: {type(self.model).__name__}")

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Metadata loaded: {list(self.metadata.keys())}")
        else:
            logger.warning(f"Metadata file not found at {METADATA_PATH}")
            self.metadata = {
                "model_name": type(self.model).__name__,
                "model_params": self.model.get_params()
                if hasattr(self.model, "get_params")
                else {},
            }

        self._is_loaded = True
        logger.info("Model ready for predictions")

    def predict(self, features: list[float]) -> float:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded")

        X = np.array(features).reshape(1, -1)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        prediction = self.model.predict(X)
        return float(prediction[0])

    def get_info(self) -> dict:
        if not self._is_loaded or not self.metadata:
            return {"error": "Model not loaded"}

        return {
            "model_name": self.metadata.get("model_name", "unknown"),
            "model_params": self.metadata.get("model_params", {}),
            "training_date": self.metadata.get("training_date"),
            "metrics": self.metadata.get("metrics"),
            "feature_columns": self.metadata.get("data_columns"),
            "data_rows": self.metadata.get("data_rows"),
        }


model_manager = ModelManager()
