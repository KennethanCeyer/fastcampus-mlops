from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.contrib.operators.bigquery_operator import BigQueryOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}
dag = DAG(
    "bigquery_airflow_example",
    default_args=default_args,
    description="A simple BigQuery Airflow example",
    schedule_interval="@daily",
)

bq_query_task = BigQueryOperator(
    task_id="bq_query_example",
    sql="""SELECT * FROM `fast-campus-machine-learning.introduction.wikipedia` LIMIT 100;""",
    use_legacy_sql=False,
    dag=dag,
)

