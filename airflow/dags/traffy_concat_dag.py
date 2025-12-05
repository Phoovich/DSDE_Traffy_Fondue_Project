from airflow.sdk import dag, task  # type: ignore

# import ฟังก์ชันจาก jobs/
from jobs.traffy_concat import concat_traffy_raw_to_processed
from pendulum import datetime


@task
def concat_task():
    return concat_traffy_raw_to_processed()


@dag(
    dag_id="traffy_concat_every_30min",
    schedule="*/30 * * * *",  # ← ทุก 30 นาที
    start_date=datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["traffy", "concat"],
)
def concat_dag():
    concat_task()


concat_dag()
