from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow import AirflowException
from datetime import datetime, timedelta
import subprocess
import logging

logging.basicConfig(   
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True
)

def check_data():
    pass

def update_dashboard():
    pass

with DAG(
    "push_data",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        'owner': "airflow",
        "depends_on_past": False,
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'on_skipped_callback': another_function, #or list of functions
        # 'trigger_rule': 'all_success'
    },
    description="A push data DAG  ",
    schedule=timedelta(minutes=10),
    start_date=datetime(2025, 8, 28),
    catchup=False,
    tags=["example"],
) as dag:
    # Define tasks/operators
    cehck_drift_data_task = BranchPythonOperator(
        task_id="check_data",
        python_callable=check_data
    )
    update_dashboard_task = PythonOperator(
        task_id="update_dashboard",
        python_callable=update_dashboard
    )
    end_task = EmptyOperator(
        task_id="end"
    )
    
    
    

