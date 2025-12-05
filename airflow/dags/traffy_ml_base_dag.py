# dags/traffy_ml_base_dag.py
from airflow.decorators import dag, task  # type: ignore
from pendulum import datetime
from jobs.traffy_ml_base import build_traffy_ml_base


@task
def build_ml_base_task():
    return build_traffy_ml_base()


@dag(
    dag_id="traffy_ml_base_daily",
    schedule="0 2 * * *",  # ตัวอย่าง: รันทุกวัน 02:00 UTC
    start_date=datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["traffy", "ml_base", "clean"],
)
def ml_base_dag():
    build_ml_base_task()


ml_base_dag()
