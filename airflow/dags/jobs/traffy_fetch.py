from pathlib import Path
from datetime import datetime
import requests
import pandas as pd
from io import StringIO

# โฟลเดอร์นี้คือ .../airflow/dags/jobs/traffy_fetch.py
THIS_FILE = Path(__file__).resolve()

# ถอยขึ้นไป 2 ชั้น → .../airflow
PROJECT_ROOT = THIS_FILE.parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "traffy"

def fetch_traffy_and_save_raw():
    url = "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1"
    params = {
        "output_format": "csv",
        "name": "a",
        "org": "a",
        "purpose": "a",
        "email": "a@a.com",
        "limit": 10000,
        "start": "2025-12-04",
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RAW_DIR / f"traffy_{ts}.csv"

    df.to_csv(file_path, index=False)

    print(f"Saved raw data to: {file_path}")
    print(f"Rows: {len(df)}")
    return str(file_path)
