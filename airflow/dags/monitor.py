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

def monitor_drift():
    script_path = "C:\\Users\\mkrym\\OneDrive\\Documents\\My Folders\\my_main_portfolio\\airflow\\scripts\\monitoring.py"

    # Run the command using a list
    result = subprocess.run([script_path], capture_output=True, text=True)

    if(result.stdout.endswith("Data drift detected! Retraining required.\n")):
        return "trigger_retrain"
    else:
        return "no_retrain"
    
def retrain_model():
    script_path = "C:\\Users\\mkrym\\OneDrive\\Documents\\My Folders\\my_main_portfolio\\airflow\\scripts\\model_training.py"

    # Run the command using a list
    result = subprocess.run([script_path], capture_output=True, text=True)
    return "trigger_retrain"

def deploy_model():
    script_path = "C:\\Users\\mkrym\\OneDrive\\Documents\\My Folders\\my_main_portfolio\\serving\\model_serving.py"
    # Serve the BentoML service with reload
    subprocess.run(["bentoml", "serve", script_path, "--reload"])
    

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    #'retries': 1,
    #'retry_delay': timedelta(minutes=5),
}

with DAG(
    'check_drift_and_retrain',
    default_args=default_args,
    description='DAG to check data drift and retrain if required',
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    check_drift_task = BranchPythonOperator(
        task_id='check_data_drift',
        python_callable=monitor_drift
    )
    retrain_task = PythonOperator(
        task_id='trigger_retrain',
        python_callable=retrain_model
    )
    no_retrain_task = EmptyOperator(
        task_id='no_retrain'
    )
    deploy_model_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model
    )


check_drift_task >> [retrain_task, no_retrain_task]
retrain_task >> deploy_model_task