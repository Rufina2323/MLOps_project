import subprocess
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def pull_data_from_dvc(**context):
    """
    Загружает данные из DVC remote storage.
    """
    project_dir = os.environ.get("PROJECT_DIR", "/opt/airflow/project")

    logger.info(f"Pulling data from DVC in {project_dir}")

    # Выполняем dvc pull для загрузки данных
    result = subprocess.run(
        ["dvc", "pull", "data/winequality-red.csv.dvc", "-v"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    logger.info(f"DVC pull stdout: {result.stdout}")

    if result.returncode != 0:
        logger.error(f"DVC pull stderr: {result.stderr}")
        # Проверяем, может файл уже существует
        data_path = os.path.join(project_dir, "data", "winequality-red.csv")
        if os.path.exists(data_path):
            logger.warning("DVC pull failed but data file exists, continuing...")
        else:
            raise RuntimeError(f"DVC pull failed: {result.stderr}")

    # Проверяем что файл скачался
    data_path = os.path.join(project_dir, "data", "winequality-red.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Валидируем данные
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # Передаём путь к данным через XCom
    context["ti"].xcom_push(key="data_path", value=data_path)
    context["ti"].xcom_push(key="data_rows", value=df.shape[0])
    context["ti"].xcom_push(key="data_cols", value=df.shape[1])

    return data_path