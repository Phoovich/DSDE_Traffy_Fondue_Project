# jobs/traffy_fetch.py
import os
from datetime import datetime
import requests
import pandas as pd
from io import StringIO

def fetch_traffy_and_save_raw():
    url = "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1"
    params = {
        "output_format": "csv",
        "name": "a",
        "org": "a",
        "purpose": "a",
        "email": "a@a.com",
        "limit": 10000,
        # "text": "น้าท่วม",
        "start": "2025-12-04",   # แนะนำใช้ format YYYY-MM-DD ให้ชัด
        # "end": "2025-10-31",
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()

    # แปลง CSV text → DataFrame
    df = pd.read_csv(StringIO(resp.text))

    # สร้างโฟลเดอร์เก็บ raw ถ้ายังไม่มี
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(BASE_DIR, "data/raw/traffy")
    os.makedirs(raw_dir, exist_ok=True)

    # ใช้ timestamp เป็นชื่อไฟล์ เพื่อเก็บทุก 10 นาทีแยกกัน
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # เช่น 20251204_101000
    file_path = os.path.join(raw_dir, f"traffy_{ts}.csv")

    df.to_csv(file_path, index=False)

    print(f"Saved raw data to: {file_path}")
    print(f"Rows: {len(df)}")
    return file_path  # เผื่ออยากใช้ต่อใน Airflow (XCom)
