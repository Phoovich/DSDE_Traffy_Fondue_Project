from airflow.decorators import dag, task  # type: ignore
from pendulum import datetime

# import ฟังก์ชันจากไฟล์ jobs/traffy_fetch.py
from jobs.traffy_fetch import fetch_traffy_and_save_raw


@task
def call_api_and_save():
    # แค่เรียกฟังก์ชันที่เราเขียนไว้
    return fetch_traffy_and_save_raw()

@dag(
    dag_id="traffy_api_every_10min",
    schedule="*/10 * * * *",   # ทุก 10 นาที
    start_date=datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["call_api", "traffy"],
)
def api_dag():
    call_api_and_save()

api_dag()
