# dags/jobs/traffy_ml_base.py
from pathlib import Path
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# old + new
SOURCE_PATH = DATA_DIR / "source" / "bangkok_traffy.csv"         # 700k เดิม
BRONZE_PATH = DATA_DIR / "processed" / "traffy_bronze_all.csv"   # จาก API ทุก 30 นาที

# output (ใช้ทั้ง viz + ML)
ML_BASE_DIR = DATA_DIR / "ml_base"
ML_BASE_PATH = ML_BASE_DIR / "traffy_ml_base.parquet"


def parse_brace_list(s):
    """
    แปลง string แบบ '{น้ำท่วม, ร้องเรียน}' → ['น้ำท่วม', 'ร้องเรียน']
    """
    s = str(s).strip("{}")
    parts = [x.strip() for x in s.split(",")]
    return [p for p in parts if p]


def clean_traffy(df: pd.DataFrame) -> pd.DataFrame:
    """
    รวม logic ทำความสะอาด:
      - drop column ที่ null > 50% (รวม photo, photo_after)
      - แยก coords → longitude, latitude
      - แปลงคอลัมน์ type ให้เป็น list
    """
    print(f"original shape: {df.shape}")

    # ---------- 1) drop column null > 50% ----------
    half_len = df.shape[0] / 2
    drop_col = ["photo", "photo_after"]

    for col in df.columns:
        null_count = df[col].isna().sum()
        print(f"null in {col} is {null_count}")
        if null_count > half_len:
            drop_col.append(col)

    print("columns have null > 50% (including fixed ones):", drop_col)
    df = df.drop(columns=drop_col, errors="ignore")

    # ---------- 2) แยก coords → longitude, latitude ----------
    if "coords" in df.columns:
        headers = df.columns.to_list()
        coords_index = headers.index("coords")
        print(f"Original 'coords' index: {coords_index}")

        # แทนที่ตำแหน่งเดิมของ coords ด้วย 2 คอลัมน์ใหม่
        headers[coords_index : coords_index + 1] = ["longitude", "latitude"]

        # split string "long,lat"
        split_data = df["coords"].astype(str).str.split(",", expand=True)

        df["longitude"] = split_data[0].astype(float)
        df["latitude"] = split_data[1].astype(float)

        # เอา df ให้เรียงคอลัมน์ตาม headers ใหม่ (ไม่มี coords แล้ว)
        # ถ้า coords ยังอยู่ใน df.columns ให้ drop
        if "coords" in headers:
            headers.remove("coords")
        df = df[headers]
    else:
        print("WARNING: no 'coords' column found, skip split to longitude/latitude")

    # ---------- 3) แปลง type ให้เป็น list ----------
    if "type" in df.columns:
        df["type"] = df["type"].apply(parse_brace_list)
    else:
        print("WARNING: no 'type' column found, skip parse_brace_list")

    print(f"cleaned shape: {df.shape}")

        # ---------- 4) Normalize province ----------
    if "province" in df.columns:
        print("Normalizing province column...")

        province_map = {
            'จังหวัดกรุงเทพมหานคร': 'กรุงเทพมหานคร',
            'จังหวัดจังหวัด กรุงเทพมหานคร': 'กรุงเทพมหานคร',
            'จังหวัดBangkok': 'กรุงเทพมหานคร',
            'จังหวัดกรุงเทพฯ': 'กรุงเทพมหานคร'
        }

        df["province"] = df["province"].replace(province_map)

        # Keep only rows within Bangkok
        df = df.loc[df["province"] == "กรุงเทพมหานคร"]

        print("Province normalized. Unique values:", df["province"].unique())
    else:
        print("WARNING: province column not found. Skip normalization.")

    return df


def build_traffy_ml_base():
    """
    รวม old data + bronze แล้ว clean → เซฟเป็น parquet ใช้ต่อใน viz + ML
    """
    # 1) โหลด old data
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Old data not found: {SOURCE_PATH}")
    df_old = pd.read_csv(SOURCE_PATH)
    print(f"Loaded old data: {len(df_old)} rows")

    # 2) โหลด bronze ใหม่
    if BRONZE_PATH.exists():
        df_new = pd.read_csv(BRONZE_PATH)
        print(f"Loaded new bronze data: {len(df_new)} rows")
    else:
        print(f"Bronze file not found: {BRONZE_PATH}")
        df_new = pd.DataFrame()

    # 3) รวมเก่า + ใหม่
    if df_new.empty:
        df_all = df_old.copy()
    else:
        df_all = pd.concat([df_old, df_new], ignore_index=True)

    # 4) dedup ด้วย ticket_id
    if "ticket_id" in df_all.columns:
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=["ticket_id"])
        after = len(df_all)
        print(f"Dedup: {before} → {after} rows")
    else:
        print("WARNING: 'ticket_id' not found, skip dedup")

    # 5) clean ด้วย logic ที่คุณเขียน
    df_clean = clean_traffy(df_all)

    # 6) เซฟ parquet (แทน `data_perped.parquet` ด้วย path ที่เป็นมาตรฐานในโปรเจกต์)
    ML_BASE_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(ML_BASE_PATH, index=False)

    print(f"Saved ML base to: {ML_BASE_PATH}")
    print(f"Total clean rows: {len(df_clean)}")

    return str(ML_BASE_PATH)
