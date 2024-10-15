from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = { 
    'owner': 'james',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

def greet():
    print("hello world")

def greetv2(name, age):
    print(f"hello world, age = {age}, name = {name}")

def get_name(ti):
    ti.xcom_push(key="firstname", value="James")
    ti.xcom_push(key="lastname", value="Nguyen")
    ti.xcom_push(key="age", value=28)

# This function returns 1 value to the XCOM, then pull using xcom_pull(task_id='get_name')
# def get_name():
#       return 'James'

#NOTE: xcom variable size is 48kb

def greet_with_xcom(age, ti):
    firstname = ti.xcom_pull(task_ids='get_name', key = 'firstname')
    lastname = ti.xcom_pull(task_ids='get_name', key = 'lastname')
    print(f"from xcom: firstname={firstname}, lastname={lastname}, age={age}")

def greet_with_xcom(ti):
    firstname = ti.xcom_pull(task_ids='get_name', key = 'firstname')
    lastname = ti.xcom_pull(task_ids='get_name', key = 'lastname')
    age = ti.xcom_pull(task_ids='get_name', key = 'age')

    print(f"from xcom: firstname={firstname}, lastname={lastname}, age={age}")

with DAG(
    dag_id="python-dag",
    default_args=default_args,
    description="First DAG using PythonOperator",
    start_date = datetime(2024, 10, 12),
    schedule_interval = '@daily'
) as dag:

    # task1 = PythonOperator(
    #     task_id = "greet",
    #     python_callable=greet
    # )

    # task2 = PythonOperator(
    #     task_id = "greetv2",
    #     python_callable=greetv2,
    #     op_kwargs = {'name': "James", "age": 28}
    # )

    task3 = PythonOperator(
        task_id="get_name",
        python_callable=get_name,
    )

    # task4 = PythonOperator(
    #     task_id="greet_with_xcom",
    #     python_callable=greet_with_xcom,
    #     op_kwargs={'age': 28}
    # )

    task4_wihtout_args = PythonOperator(
        task_id="greet_with_xcom",
        python_callable=greet_with_xcom,
    )


    #get the name from XCom

    task3 >> task4_wihtout_args

    # task1 >> task2



   




