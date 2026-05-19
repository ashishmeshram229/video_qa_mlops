import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0, 
}

with DAG(
    'mvtec_advanced_segmentation_pipeline',
    default_args=default_args,
    description='Master ETL, Data Quality, and DVC Pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Check Data 
    check_raw_data = BashOperator(
        task_id='check_raw_data',
        bash_command='test -f /opt/airflow/data/raw/bottle.tar.xz || (echo "Raw data missing!" && exit 1)',
    )

    # Task 2: ETL Pipeline 
    main_etl_pipeline = BashOperator(
        task_id='extract_transform_validate',
        bash_command='''
            export PYTHONPATH=/opt/airflow
            cd /opt/airflow
            
            echo "Installing dependencies..."
            pip install pydantic-settings pydantic pyyaml opencv-python-headless pandas scikit-learn tqdm ultralytics
            
            echo "Running Data Engineering Script..."
            python -m src.pipeline.data_engineering
        ''',
    )

    # Task 3: Data Quality Metrics
    data_quality_check = BashOperator(
        task_id='data_quality_metrics',
        bash_command='''
            echo "Total Processed Images Generated:"
            find /opt/airflow/data/processed -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l || echo "0"
        ''',
    )



    # Execution Flow
    check_raw_data >> main_etl_pipeline >> data_quality_check >> trigger_dvc_pipeline