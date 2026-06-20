"""DAG 2: Model training — embed -> train -> evaluate -> register.

Triggered after DAG 1 completes or manually when new labelled data arrives.
The MLflow promotion gate (recall >= 0.90, precision >= 0.85) runs inside
mlflow_registry.py — a failed gate logs the run but does NOT promote to Production.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

with DAG(
    dag_id="dag2_model_training",
    description="Train harmful detector: embed -> train -> evaluate -> MLflow register",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["training", "mlflow"],
) as dag:

    embed = BashOperator(
        task_id="generate_embeddings",
        bash_command=f"cd {PROJECT} && python -m src.embedding",
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT} && python -m src.train_model",
    )

    register = BashOperator(
        task_id="evaluate_and_register",
        bash_command=f"cd {PROJECT} && python -m src.mlflow_registry",
    )

    build_faiss = BashOperator(
        task_id="build_faiss_index",
        bash_command=f"cd {PROJECT} && python -c \""
            "import pandas as pd, os;"
            "from src.matching.text_matching import TextMatcher;"
            "df = pd.read_parquet('datamart/gold/splits/train.parquet');"
            "seed = df[df['label']==1][['video_id','combined_text']];"
            "TextMatcher().build_index(seed).save_index()"
            "\"",
    )

    embed >> train >> register >> build_faiss
