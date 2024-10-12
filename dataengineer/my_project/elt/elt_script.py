import subprocess
import time

def wait_for_prosgres(host, max_retries = 5, delay_seconds = 5):
    retries = 0 
    while retries < max_retries:
        try:
            result = subprocess.run(
                ['pg_isready', '-h', host],
                check = True,
                capture_output=True,
                text = True
            )
            print(result.stdout)

            if "accepting connections" in result.stdout:
                print("Connected to db")
                return True
        except subprocess.CalledProcessError as e:
            print(f'Error connecting to db {e}')
            retries += 1
            print(f'Retrying in {delay_seconds} seconds')
            time.sleep(delay_seconds)
        
    print("Max retries reached, exiting")
    return False

if not wait_for_prosgres(host = "source_postgres"): 
    exit(1)

print("Building ELT ...")

source_config = {
    'dbname': 'source_db',
    'user': 'postgres',
    'password': 'secret',
    'host': 'source_postgres',
}

destination_config = {
    'dbname': 'destination_db',
    'user': 'postgres',
    'password': 'secret',
    'host': 'destination_postgres',
}

#dump command: init source db
dump_command = [
    'pg_dump',
    '-h', source_config['host'],
    '-U', source_config['user'],
    '-d', source_config['dbname'],
    '-f', 'data_dump.sql',
    '-w'
]

# subprocess environment variables
try:
    subprocess_env = dict(PGPASSWORD=source_config['password'])
    subprocess.check_output(dump_command, env = subprocess_env)
except subprocess.CalledProcessError as e:
    print('error ', e.output)


load_command = [
    'psql',
    '-h', destination_config['host'],
    '-U', destination_config['user'],
    '-d', destination_config['dbname'],
    '-a', '-f', 'data_dump.sql'
]


subprocess_env = dict(PGPASSWORD=destination_config['password'])

try:
    subprocess_env = dict(PGPASSWORD=source_config['password'])
    subprocess.check_output(load_command, env = subprocess_env)
except subprocess.CalledProcessError as e:
    print('error ', e.output)

print('ending ELT script')

