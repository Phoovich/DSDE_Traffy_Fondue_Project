from pathlib import Path
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw" / "traffy"
PROCESSED_DIR = DATA_DIR / "processed"

def concat_traffy_raw_to_processed():
    raw_files = sorted(RAW_DIR.glob("traffy_*.csv"))

    if not raw_files:
        print("No raw files found.")
        return None

    dfs = [pd.read_csv(f) for f in raw_files]
    df_all = pd.concat(dfs, ignore_index=True)

    if "ticket_id" in df_all.columns:
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=["ticket_id"])
        after = len(df_all)
        print(f"Dedup: {before} → {after} rows")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "traffy_bronze_all.parquet"
    df_all.to_parquet(output_path, index=False)

    print(f"Saved processed data to: {output_path}")
    return str(output_path)
