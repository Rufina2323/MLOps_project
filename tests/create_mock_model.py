import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def create_mock_model():
    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    np.random.seed(42)
    X = np.random.randn(100, 11)
    y = np.random.randint(3, 9, size=100).astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=0.1, fit_intercept=True)
    model.fit(X_scaled, y)

    model_path = model_dir / "ridge_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    metadata = {
        "model_name": "Ridge",
        "model_params": {"alpha": 0.1, "fit_intercept": True},
        "training_date": "2025-01-15T10:00:00",
        "data_rows": 1599,
        "data_columns": [
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
        ],
        "target_column": "quality",
        "metrics": {
            "test_rmse": 0.6451,
            "test_r2": 0.3572,
            "test_mae": 0.5123,
            "train_rmse": 0.6201,
            "train_r2": 0.3801,
        },
    }

    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Mock model saved to {model_path}")
    print(f"Mock metadata saved to {metadata_path}")


if __name__ == "__main__":
    create_mock_model()
