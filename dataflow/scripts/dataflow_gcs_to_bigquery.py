import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


class JobPipelineOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument(
            "--input", type=str, help="Input file path in GCS"
        )
        parser.add_value_provider_argument(
            "--output", type=str, help="Output BigQuery table"
        )


def run():
    pipeline_options = PipelineOptions()
    job_options = pipeline_options.view_as(JobPipelineOptions)

    with beam.Pipeline(options=pipeline_options) as p:
        data = (
            p
            | "Read from GCS" >> beam.io.ReadFromText(job_options.input)
            | "Convert to dictionary" >> beam.Map(lambda s: {"data": s})
        )

        _ = data | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            job_options.output,
            schema="data:STRING",
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        )


if __name__ == "__main__":
    run()
