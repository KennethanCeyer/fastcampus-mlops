import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromText
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition


class JobPipelineOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument(
            "--input", type=str, help="Input GCS directory containing TXT files"
        )
        parser.add_value_provider_argument(
            "--output", type=str, help="Output BigQuery table"
        )


def parse_netflix_data(element, movie_id=[None]):
    try:
        if ":" in element:
            movie_id[0] = element.split(":")[0]
        else:
            user_id, rating, date = element.split(",")
            return {
                "Movie_Id": movie_id[0],
                "User_Id": user_id,
                "Rating": rating,
                "Date": date,
            }
    except:
        pass
    return None


def run():
    pipeline_options = PipelineOptions()
    job_options = pipeline_options.view_as(JobPipelineOptions)

    with beam.Pipeline(options=pipeline_options) as p:
        lines = p | "Read TXT Files" >> ReadFromText(job_options.input.get() + "*.txt")
        parsed_data = lines | "Parse Netflix Data" >> beam.FlatMap(parse_netflix_data)
        valid_data = parsed_data | "Filter None" >> beam.Filter(lambda x: x is not None)
        _ = valid_data | "Write to BigQuery" >> WriteToBigQuery(
            job_options.output.get(),
            schema="Movie_Id:STRING, User_Id:STRING, Rating:STRING, Date:STRING",
            create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
            write_disposition=BigQueryDisposition.WRITE_APPEND,
        )


if __name__ == "__main__":
    run()
