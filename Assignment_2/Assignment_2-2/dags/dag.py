from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline: data pipeline + model train/inference/monitoring',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
) as dag:

    # ------------------------------------------------------------------
    # 0. Dependency checks (start markers)
    # ------------------------------------------------------------------
    dep_check_source_data = DummyOperator(task_id="dep_check_source_data")

    # ------------------------------------------------------------------
    # 1. Bronze layer — raw ingestion
    # ------------------------------------------------------------------
    bronze_clickstream = BashOperator(
        task_id='bronze_clickstream',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_feature_store.py --snapshotdate "{{ ds }}" --source clickstream'
        ),
    )

    bronze_financials = BashOperator(
        task_id='bronze_financials',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_feature_store.py --snapshotdate "{{ ds }}" --source financials'
        ),
    )

    bronze_attributes = BashOperator(
        task_id='bronze_attributes',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_feature_store.py --snapshotdate "{{ ds }}" --source attributes'
        ),
    )

    bronze_label_store = BashOperator(
        task_id='bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py --snapshotdate "{{ ds }}"'
        ),
    )

    # ------------------------------------------------------------------
    # 2. Silver layer — cleaning, schema enforcement, PII removal
    # ------------------------------------------------------------------
    silver_clickstream = BashOperator(
        task_id='silver_clickstream',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_feature_store.py --snapshotdate "{{ ds }}" --source clickstream'
        ),
    )

    silver_financials = BashOperator(
        task_id='silver_financials',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_feature_store.py --snapshotdate "{{ ds }}" --source financials'
        ),
    )

    silver_attributes = BashOperator(
        task_id='silver_attributes',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_feature_store.py --snapshotdate "{{ ds }}" --source attributes'
        ),
    )

    silver_label_store = BashOperator(
        task_id='silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_label_store.py --snapshotdate "{{ ds }}"'
        ),
    )

    # ------------------------------------------------------------------
    # 3. Gold layer — feature store & label store
    # ------------------------------------------------------------------
    gold_feature_store = BashOperator(
        task_id='gold_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_feature_store.py --snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id='gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_label_store.py --snapshotdate "{{ ds }}"'
        ),
    )

    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    label_store_completed = DummyOperator(task_id="label_store_completed")

    # ------------------------------------------------------------------
    # 4. ML pipeline — train, infer, monitor
    # ------------------------------------------------------------------
    MODEL_NAME = "credit_model_2024_09_01.pkl"

    model_train = BashOperator(
        task_id='model_train',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_train.py --snapshotdate "{{ ds }}"'
        ),
    )

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            f'python3 model_inference.py --snapshotdate "{{{{ ds }}}}" --modelname {MODEL_NAME}'
        ),
    )

    model_monitor = BashOperator(
        task_id='model_monitor',
        bash_command=(
            'cd /opt/airflow/scripts && '
            f'python3 model_monitor.py --snapshotdate "{{{{ ds }}}}" --modelname {MODEL_NAME}'
        ),
    )

    ml_pipeline_completed = DummyOperator(task_id="ml_pipeline_completed")

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    # bronze -> silver -> gold (feature store)
    dep_check_source_data >> bronze_clickstream >> silver_clickstream
    dep_check_source_data >> bronze_financials >> silver_financials
    dep_check_source_data >> bronze_attributes >> silver_attributes
    [silver_clickstream, silver_financials, silver_attributes] >> gold_feature_store >> feature_store_completed

    # bronze -> silver -> gold (label store)
    dep_check_source_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # ML pipeline: train (on schedule) -> inference -> monitor
    [feature_store_completed, label_store_completed] >> model_train
    model_train >> model_inference >> model_monitor >> ml_pipeline_completed
