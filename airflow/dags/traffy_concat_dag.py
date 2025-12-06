from airflow.sdk import dag, task  # type: ignore
from jobs.traffy_concat import concat_traffy_raw_to_processed
from pendulum import datetime

@task
def concat_task():
    return concat_traffy_raw_to_processed()

@dag(
    dag_id="traffy_concat_daily_1am",
    schedule="0 1 * * *",  # run 01:00 UTC every day
    start_date=datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["traffy", "concat"],
)
def concat_dag():
    concat_task()

concat_dag()
