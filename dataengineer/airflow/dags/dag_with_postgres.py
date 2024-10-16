from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator

default_args = { 
    'owner': 'james',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="dag_with_postgres",
    default_args=default_args,
    description="DAG with postgres",
    start_date = datetime(2024, 10, 15),
    schedule_interval = '@daily',
    catchup = False
) as dag:

    task1 = PostgresOperator(
        task_id = 'create_table',
        postgres_conn_id = 'postgress_connection', #configure in UI > admin > connections
        sql = '''
        create table if not exists dag_runs(
            dt date,
            dag_id character varying,
            primary key (dt, dag_id)
        )
        '''
    )

    task1




