from datetime import datetime, timedelta
from airflow.decorators import dag, task

default_args = { 
    'owner': 'james',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

def greet():
    print("Hello world")

@dag(
        dag_id="dag_with_taskflow",
        default_args=default_args,
        start_date=datetime(2024,10,10),
        schedule_interval="@daily"
    )
def hello_world_etl():

    @task()
    def get_name():
        return "James1" #this data sends to xcome

    @task(multiple_outputs=True) #if function returns dict to xcom
    def get_first_last_name():
        return {'firstname': "James1", 'lastname': 'Nguyen'} # It has to be a dictionary
    
    @task()
    def get_age():
        return 100 #sends to xcom
    
    @task()
    def greet(name, age):
        print(f"hello, name={name}, age = {age}")

    @task()
    def greet_first_last(firstname, lastname, age):
        print(f"hello, name={firstname}, lastname={lastname}, age = {age}")
    
    # name = get_name() #retrieve from xcom
    names = get_first_last_name() #retrieve from xcom
    firstname = names['firstname']
    lastname = names['lastname']
    age = get_age() #retrieve from xcom

    # greet(name = name, age = age)
    greet_first_last(firstname=firstname, lastname=lastname, age = age)

greet_dag = hello_world_etl() #create instance of dag

    



  

   




