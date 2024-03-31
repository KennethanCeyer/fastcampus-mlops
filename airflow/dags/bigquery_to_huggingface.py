import io
import uuid
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from huggingface_hub import HfApi, HfFolder
import pandas as pd


@dag(
    dag_id="bigquery_to_huggingface",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    description="BigQuery to GCS to HuggingFace dataset registration using TaskFlow API",
)
def dag_bigquery_to_huggingface():
    @task
    def bigquery_to_gcs(
        dataset_id: str,
        table_id: str,
        gcs_bucket_name: str,
        gcs_object_prefix: str
    ):
        bq_hook = BigQueryHook(use_legacy_sql=False)
        bq_client = bq_hook.get_client(project_id=bq_hook.project_id)

        dataset_ref = bq_client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)

        temp_name = str(uuid.uuid4())
        destination_uri = f"gs://{gcs_bucket_name}/{gcs_object_prefix}_{temp_name}_*.csv"
        extract_job = bq_client.extract_table(
            table_ref,
            destination_uri,
            location="US",
        )
        extract_job.result()

        return destination_uri

    @task
    def register_dataset_to_huggingface(
        gcs_bucket_name: str,
        gcs_object_prefix: str,
        gcs_path: str,
        dataset_repo_name: str,
    ):
        hf_username = Variable.get("HF_USERNAME")
        hf_token = Variable.get("HF_API_TOKEN")

        HfFolder.save_token(hf_token)
        api = HfApi()
        api.create_repo(
            repo_id=dataset_repo_name,
            token=hf_token,
            repo_type="dataset",
            exist_ok=True,
        )

        gcs_hook = GCSHook()
        blobs = gcs_hook.list(gcs_bucket_name, prefix=gcs_object_prefix)

        dfs = []
        for blob_name in blobs:
            blob_content = gcs_hook.download(gcs_bucket_name, blob_name)
            dfs.append(pd.read_csv(io.BytesIO(blob_content), sep=","))
        merged_df = pd.concat(dfs, ignore_index=True)

        with io.BytesIO() as byte_stream:
            merged_df.to_csv(byte_stream)
            byte_stream.seek(0)

            api.upload_file(
                path_or_fileobj=byte_stream,
                path_in_repo="train.csv",
                repo_id=f"{hf_username}/{dataset_repo_name}",
                repo_type="dataset",
                token=hf_token
            )

    gcs_bucket_name = "fast_campus_airflow"
    gcs_object_prefix = "wikipedia_dataset_"

    gcs_path = bigquery_to_gcs(
        dataset_id="introduction",
        table_id="wikipedia",
        gcs_bucket_name=gcs_bucket_name,
        gcs_object_prefix=gcs_object_prefix
    )
    register_dataset_to_huggingface(
        gcs_bucket_name=gcs_bucket_name,
        gcs_object_prefix=gcs_object_prefix,
        gcs_path=gcs_path,
        dataset_repo_name="fast_campus_wikipedia",
    )

dag_instance = dag_bigquery_to_huggingface()

