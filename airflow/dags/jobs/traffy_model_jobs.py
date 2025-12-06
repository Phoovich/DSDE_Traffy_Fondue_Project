import pandas as pd
import joblib
import re
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

# Selection & Metrics
from sklearn.model_selection import RandomizedSearchCV
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

# Thai NLP Imports
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from pythainlp.corpus import thai_stopwords

stop_words = set(thai_stopwords())

# Import clean function
def clean_and_tokenize(text):
    if not isinstance(text, str):
        return ""
    text = normalize(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
MODELS_DIR = DATA_DIR / "models"

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data_task(sample_size=None):
    """
    โหลดข้อมูล -> Clean -> MLB -> Split -> Save
    """
    input_path = DATA_DIR / "ml_base" / "traffy_ml_base.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_parquet(input_path)

    df = df.dropna(subset=['type'])
    df = df.dropna(subset=['comment'])

    # 1.3 ลบแถวที่หลังจากกรองแล้วกลายเป็น List ว่าง []
    df = df[df['type'].map(len) > 0]


    # --- Logic Test Mode ---
    if sample_size:
        print(f"⚠️ TESTING MODE: Using only {sample_size} rows.")
        if len(df) > int(sample_size):
            df = df.sample(n=int(sample_size), random_state=42)

    # 1. Clean & Tokenize
    print("Tokenizing...")
    df['processed_comment'] = df['comment'].apply(clean_and_tokenize)
    
    # 2. MLB
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['type'])
    X = df['processed_comment'].to_numpy().reshape(-1, 1)
    
    # 3. Split (แก้ไข: ลบส่วนที่ซ้ำออก เหลือแค่ Try-Except)
    print("Splitting...")
    try:
        # พยายามใช้ Iterative Split (ดีสำหรับ Multi-label)
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.3)
    except ValueError as e:
        # ถ้าข้อมูลน้อยเกินไปจน Split ไม่ได้ ให้ใช้ Simple Split
        print(f"Iterative split failed (likely due to small sample size). Falling back to simple split. Error: {e}")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = X_train.ravel()
    X_test = X_test.ravel()
    
    # 4. Save
    data_bundle = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "mlb": mlb
    }
    dump_path = INTERMEDIATE_DIR / "train_data.pkl"
    joblib.dump(data_bundle, dump_path)
    print(f"Data prepared and saved at {dump_path}")
    return str(dump_path)

def train_model(model_name):
    # (โค้ดส่วนนี้ของคุณถูกต้องแล้ว ไม่ต้องแก้ครับ)
    # ... Copy Code เดิมของคุณในส่วน train_model มาใส่ได้เลย ...
    # เพื่อความกระชับ ผมละไว้ในฐานที่เข้าใจนะครับ
    
    # 1. Load Data
    data_path = INTERMEDIATE_DIR / "train_data.pkl"
    data = joblib.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    pipeline = None
    param_dist = {}
    common_tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

    if model_name == "svc":
        pipeline = Pipeline([
            ("tfidf", common_tfidf),
            ("clf", OneVsRestClassifier(LinearSVC(class_weight='balanced', random_state=42, dual='auto')))
        ])
        param_dist = {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__estimator__C': [0.1, 1, 10],
        }

    elif model_name == "sgd":
        pipeline = Pipeline([
            ("tfidf", common_tfidf),
            ("clf", OneVsRestClassifier(SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42)))
        ])
        param_dist = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__estimator__alpha': [1e-4, 1e-3, 1e-2],
            'clf__estimator__penalty': ['l2', 'l1', 'elasticnet'],
        }

    elif model_name == "nb":
        pipeline = Pipeline([
            ("tfidf", common_tfidf),
            ("clf", OneVsRestClassifier(ComplementNB()))
        ])
        param_dist = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__estimator__alpha': [0.1, 0.5, 1.0],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Starting RandomizedSearchCV for {model_name}...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=5,
        scoring='f1_micro',
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    final_score = f1_score(y_test, y_pred, average='micro')
    
    print(f"[{model_name}] Final Test Set F1-Score: {final_score:.4f}")
    
    model_save_path = MODELS_DIR / f"model_{model_name}.pkl"
    joblib.dump(best_model, model_save_path)
    
    return final_score
