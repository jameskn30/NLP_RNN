from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': "james",
    'retries': 5,
    'retry_delay': timedelta(minutes = 5)
}

with DAG(
    dag_id = 'dag_with_catchup_backfill_v0',
    default_args=default_args,
    start_date = datetime(2024,10,15),
    schedule_interval="@daily",
    catchup=True, #this is default

) as dag:

    task1 = BashOperator(
        task_id="task1",
        bash_command="echo This is awesome"
    )

    task1