# Source
https://www.youtube.com/watch?v=OLXkGB7krGo

steps: 

1. create python venv and install dbt-core, dbt-snowflake

!pip install dbt-core, dbt-snowflake


2. setup Snowflake account and follow his instructions to setup dbt_role, dbt_wh, dbt_db, ... 


3. Configure the dbt_project.yml

config-version: 2

models:
  dbt_tut:
    # Config indicated by + and applies to all files under models/example/
    staging:
      +materialized: view
      snowflake_warehouse: dbt_wh
    marts:
      +materialized: table
      snowflake_warehouse: dbt_wh


4. install 3rd party lib 
- create packages.yml
- add dbt-labels/dbt-utils
- run $dbt deps

5. configure staging sources
