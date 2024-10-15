from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = { 
    'owner': 'james',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="first-dag",
    default_args=default_args,
    description="First DAG ever",
    start_date = datetime(2021, 7, 29, 2),
    schedule_interval = '@daily'
) as dag:

    task1 = BashOperator(
        task_id='task1',
        bash_command='echo "Running task1"',
    )

    task2 = BashOperator(
        task_id='task2',
        bash_command='echo "Running task2"',
    ) 

    task3 = BashOperator(
        task_id='task3',
        bash_command='echo "Running task3"',
    ) 

    task1 >> [task2, task3]




