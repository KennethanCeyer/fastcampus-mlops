from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.providers.google.common.hooks.base_google import GoogleBaseHook
from airflow.providers.google.cloud.operators.dataflow import (
    DataflowCreatePythonJobOperator,
)


@dag(
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["example", "dataflow"],
)
def dataflow_example():
    @task
    def get_gcp_project() -> str:
        hook = GoogleBaseHook(gcp_conn_id="google_cloud_default")
        project_id = hook.project_id
        return project_id

    @task
    def start_dataflow_job(data_gcs_path: str, project_id: str):
        dataflow_job = DataflowCreatePythonJobOperator(
            task_id="start_dataflow_job",
            py_file=f"{data_gcs_path}/scripts/dataflow_gcs_to_bigquery.py",
            options={
                "input": f"{data_gcs_path}/sample_data/language.txt",
                "output": f"{project_id}:dataflow.languages",
                "project": project_id,
                "region": "us-central1",
                "temp_location": f"{data_gcs_path}/temp",
                "staging_location": f"{data_gcs_path}/staging",
            },
            py_requirements=["apache-beam"],
            py_interpreter="python3",
            py_system_site_packages=False,
            job_name="start_dataflow_job",
            location="us-central1",
        )
        dataflow_job.execute({})

    data_gcs_path = "gs://fast_campus_data_pipeline"
    project_id = get_gcp_project()
    start_dataflow_job(data_gcs_path, project_id)


dag_instance = dataflow_example()
