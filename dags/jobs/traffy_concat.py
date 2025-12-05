# jobs/traffy_concat.py
import glob
import os
import pandas as pd

def concat_traffy_raw_to_processed():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # root ของ airflow/
    raw_dir = os.path.join(BASE_DIR, "data/raw/traffy")
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "traffy_*.csv")))

    if not raw_files:
        print("No raw files found.")
        return None

    dfs = [pd.read_csv(f) for f in raw_files]
    df_all = pd.concat(dfs, ignore_index=True)

    # ถ้า API มี ticket_id → ใช้อันนี้ dedup (สำคัญมาก!)
    if "ticket_id" in df_all.columns:
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=["ticket_id"])
        after = len(df_all)
        print(f"Dedup: {before} → {after} rows")

    processed_dir = os.path.join(BASE_DIR, "data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "traffy_all.parquet")
    
    df_all.to_parquet(output_path, index=False)

    print(f"Saved processed data to: {output_path}")
    return output_path
