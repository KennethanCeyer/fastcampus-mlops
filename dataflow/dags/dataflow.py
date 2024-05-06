from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.providers.apache.beam.operators.beam import BeamRunPythonPipelineOperator
from airflow.providers.google.cloud.hooks.base import GoogleBaseHook

@dag(
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=['example', 'dataflow']
)
def dataflow_example():
    @task
    def get_gcp_project() -> str:
        hook = GoogleBaseHook(gcp_conn_id="google_cloud_default")
        project_id = hook.project_id
        return project_id

    @task
    def start_dataflow_job(project_id: str):
        start_python_job = BeamRunPythonPipelineOperator(
            runner=BeamRunnerType.DataflowRunner,
            task_id="start_python_job",
            py_file="gs://fast_campus_data_pipeline_example/scripts/dataflow_gcs_to_bigquery.py",
            py_options=[],
            pipeline_options={
                "input": "gs://fast_campus_data_pipeline_example/sample_data/language.txt",
                "output": f"{project_id}:dataflow.languages",
            },
            py_requirements=["apache-beam[gcp]==2.47.0"],
            py_interpreter="python3",
            py_system_site_packages=False,
            dataflow_config={"location": "us-central1", "job_name": "start_python_job"},
        )
        return start_python_job.execute({})

    project_id = get_gcp_project()
    start_dataflow_job(project_id)


dag_instance = dataflow_example()

