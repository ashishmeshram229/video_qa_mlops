from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default settings for the DAG
default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the Airflow DAG
with DAG(
    'mvtec_data_engineering_pipeline',
    default_args=default_args,
    description='Extracts, transforms, and validates MVTec AD data for YOLO training.',
    schedule_interval=None, # Triggered manually for this project
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data_engineering', 'computer_vision'],
) as dag:

    # Task: Install dependencies and execute the ETL pipeline
    # Note: We run pip install here to ensure the lightweight Airflow container has OpenCV and Pydantic
    run_etl_pipeline = BashOperator(
        task_id='extract_transform_validate',
        bash_command='''
        pip install opencv-python-headless pydantic pydantic-settings && \
        cd /opt/airflow && \
        python -m src.pipeline.data_engineering
        '''
    )

    run_etl_pipeline