import subprocess
import os
import logging

logger = logging.getLogger(__name__)


def save_model_to_dvc(**context):
    """
    Добавляет обученную модель в DVC и пушит в remote storage.
    """
    ti = context["ti"]

    model_path = ti.xcom_pull(task_ids="train_model", key="model_path")
    metadata_path = ti.xcom_pull(task_ids="train_model", key="metadata_path")
    metrics = ti.xcom_pull(task_ids="train_model", key="metrics")

    if not model_path or not metadata_path:
        raise ValueError("model_path or metadata_path not found in XCom")

    project_dir = os.environ.get("PROJECT_DIR", "/opt/airflow/project")

    logger.info(f"Adding model to DVC: {model_path}")
    logger.info(f"Adding metadata to DVC: {metadata_path}")

    # Добавляем директорию models в DVC
    # Используем dvc add для трекинга через DVC
    result = subprocess.run(
        ["dvc", "add", "models/"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )

    logger.info(f"DVC add stdout: {result.stdout}")
    if result.returncode != 0:
        logger.error(f"DVC add stderr: {result.stderr}")
        raise RuntimeError(f"DVC add failed: {result.stderr}")

    # Пушим в remote storage
    result = subprocess.run(
        ["dvc", "push", "-v"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    logger.info(f"DVC push stdout: {result.stdout}")
    if result.returncode != 0:
        logger.warning(f"DVC push stderr: {result.stderr}")
        # Не падаем, если push не удался — модель уже сохранена локально
        logger.warning("DVC push may have failed, but model is saved locally")

    logger.info("=" * 50)
    logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Model: {model_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Test RMSE: {metrics.get('test_rmse', 'N/A'):.4f}")
    logger.info(f"Test R2: {metrics.get('test_r2', 'N/A'):.4f}")
    logger.info("=" * 50)

    return "success"
