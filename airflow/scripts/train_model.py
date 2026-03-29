import pandas as pd
import numpy as np
import yaml
import pickle
import json
import os
import logging
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_config():
    """Загружает конфигурацию модели."""
    config_path = os.environ.get(
        "MODEL_CONFIG_PATH", "/opt/airflow/config/model_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}: {config}")
    return config


def train_model(**context):
    """
    Обучает модель Ridge на данных из DVC.
    """
    ti = context["ti"]

    # Получаем путь к данным из предыдущего таска
    data_path = ti.xcom_pull(task_ids="pull_data", key="data_path")
    if not data_path:
        raise ValueError("data_path not found in XCom")

    # Загружаем конфиг
    config = load_config()
    model_config = config["model"]
    data_config = config["data"]
    output_config = config["output"]

    logger.info(f"Training model: {model_config['name']}")
    logger.info(f"Model params: {model_config['params']}")

    # Загружаем данные
    df = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {df.shape}")

    # Разделяем на features и target
    target_col = data_config["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target: {target_col}, distribution: mean={y.mean():.3f}, std={y.std():.3f}")

    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
    )

    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Масштабируем признаки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Создаём и обучаем модель
    model = Ridge(**model_config["params"])
    model.fit(X_train_scaled, y_train)

    # Предсказания и метрики
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    metrics = {
        "train_mse": float(mean_squared_error(y_train, y_train_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_mse": float(mean_squared_error(y_test, y_test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
    }

    logger.info(f"Training metrics: {json.dumps(metrics, indent=2)}")

    # Сохраняем модель и scaler
    project_dir = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
    model_dir = os.path.join(project_dir, output_config["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    # Сохраняем модель
    model_path = os.path.join(model_dir, output_config["model_filename"])
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    logger.info(f"Model saved to {model_path}")

    # Формируем метаданные
    metadata = {
        "model_name": model_config["name"],
        "model_params": model_config["params"],
        "training_date": datetime.now().isoformat(),
        "data_path": data_path,
        "data_rows": int(df.shape[0]),
        "data_columns": list(X.columns),
        "target_column": target_col,
        "test_size": data_config["test_size"],
        "random_state": data_config["random_state"],
        "metrics": metrics,
        "feature_importances": dict(
            zip(X.columns, [float(c) for c in model.coef_])
        ),
        "intercept": float(model.intercept_),
    }

    # Сохраняем метаданные
    metadata_path = os.path.join(model_dir, output_config["metadata_filename"])
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {metadata_path}")

    # Передаём пути через XCom
    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="metadata_path", value=metadata_path)
    ti.xcom_push(key="model_dir", value=model_dir)
    ti.xcom_push(key="metrics", value=metrics)

    return model_path
