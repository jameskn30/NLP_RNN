from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': "james",
    'retries': 5,
    'retry_delay': timedelta(minutes = 5)
}

with DAG(
    default_args = default_args,
    dag_id = 'dag_with_cron',
    start_date = datetime(2024, 10,1),
    schedule_interval = '0 0 * * Tue' 

) as dag:
    task1 = BashOperator(
        task_id="task1",
        bash_command="echo hello world"
    )

    task1
    