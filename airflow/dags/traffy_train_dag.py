from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.task.trigger_rule import TriggerRule

from pendulum import datetime
from pathlib import Path
import shutil

# Import functions
from jobs.traffy_model_jobs import prepare_data_task, train_model

DATA_DIR = Path("/Users/phoovich/airflow/data")
MODELS_DIR = DATA_DIR / "models"
BEST_MODEL_PATH = DATA_DIR / "best_model.pkl" 

# --- Wrapper Function ---
def _prepare_data_wrapper(**kwargs):
    # Airflow 2.x ส่ง context มาใน kwargs ให้อัตโนมัติครับ
    dag_run = kwargs.get('dag_run')
    sample_size = None
    
    if dag_run and dag_run.conf:
        sample_size = dag_run.conf.get('sample_size')
        if sample_size:
            print(f"Running with custom sample_size: {sample_size}")
        
    return prepare_data_task(sample_size=sample_size)

def _train_svc(**kwargs):
    return train_model("svc")

def _train_sgd(**kwargs):
    return train_model("sgd")

def _train_nb(**kwargs):
    return train_model("nb")

def _choose_best_model(**kwargs):
    ti = kwargs['ti']
    score_svc = ti.xcom_pull(task_ids='train_svc')
    score_sgd = ti.xcom_pull(task_ids='train_sgd')
    score_nb = ti.xcom_pull(task_ids='train_nb')
    
    scores = {
        'finalize_svc': score_svc,
        'finalize_sgd': score_sgd,
        'finalize_nb': score_nb
    }
    
    best_task = max(scores, key=scores.get)
    best_score = scores[best_task]
    
    print(f"Scores -> SVC:{score_svc}, SGD:{score_sgd}, NB:{score_nb}")
    print(f"Winner: {best_task} with score {best_score}")
    
    return best_task

def _finalize_model(model_name):
    source = MODELS_DIR / f"model_{model_name}.pkl"
    shutil.copy(source, BEST_MODEL_PATH)
    print(f"Promoted {model_name} to {BEST_MODEL_PATH}")

with DAG(
    dag_id="traffy_model_training_selection",
    schedule="@monthly",
    start_date=datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["traffy", "training"],
    params={"sample_size": 0} 
) as dag:

    # 1. Prepare Data
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=_prepare_data_wrapper,
        # provide_context=True  <--- ลบทิ้งครับ ไม่ต้องใช้แล้วใน Airflow 2
    )

    # 2. Parallel Training
    t_svc = PythonOperator(task_id="train_svc", python_callable=_train_svc)
    t_sgd = PythonOperator(task_id="train_sgd", python_callable=_train_sgd)
    t_nb  = PythonOperator(task_id="train_nb",  python_callable=_train_nb)

    # 3. Branching
    choose_best = BranchPythonOperator(
        task_id="choose_best_model",
        python_callable=_choose_best_model,
        # provide_context=True <--- ลบทิ้งเช่นกัน
    )

    # 4. Finalize
    f_svc = PythonOperator(task_id="finalize_svc", python_callable=_finalize_model, op_args=["svc"])
    f_sgd = PythonOperator(task_id="finalize_sgd", python_callable=_finalize_model, op_args=["sgd"])
    f_nb  = PythonOperator(task_id="finalize_nb",  python_callable=_finalize_model, op_args=["nb"])

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ONE_SUCCESS)

    # Flow
    prepare >> [t_svc, t_sgd, t_nb] >> choose_best
    
    choose_best >> f_svc >> end
    choose_best >> f_sgd >> end
    choose_best >> f_nb  >> end
