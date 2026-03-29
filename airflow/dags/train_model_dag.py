from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
import os

# Добавляем путь к скриптам
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from load_data import pull_data_from_dvc
from train_model import train_model
from save_model import save_model_to_dvc


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="train_ridge_model",
    default_args=default_args,
    description="Daily pipeline: pull data from DVC, train Ridge model, save model to DVC",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "training", "ridge"],
    max_active_runs=1,
) as dag:

    task_pull_data = PythonOperator(
        task_id="pull_data",
        python_callable=pull_data_from_dvc,
        provide_context=True,
    )

    task_train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
    )

    task_save_model = PythonOperator(
        task_id="save_model",
        python_callable=save_model_to_dvc,
        provide_context=True,
    )

    task_pull_data >> task_train_model >> task_save_model
