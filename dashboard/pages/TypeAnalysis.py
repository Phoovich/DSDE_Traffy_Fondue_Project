import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

# --- NLP Imports (ต้องมีเพราะเราต้องตัดคำก่อนส่งเข้า Model) ---
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from pythainlp.corpus import thai_stopwords

# ---------------------------------------------------------
# Config & Setup
# ---------------------------------------------------------
# ตั้งค่าฟอนต์ภาษาไทยสำหรับ Mac
plt.rcParams['font.family'] = 'Thonburi'

MODEL_PATH = "best_model.pkl"
TRAIN_DATA_PATH = "train_data.pkl"

stop_words = set(thai_stopwords())

# ---------------------------------------------------------
# 1. NLP Function (ต้องเหมือนใน Airflow เป๊ะๆ)
# ---------------------------------------------------------
def clean_and_tokenize(text):
    """ฟังก์ชันตัดคำและทำความสะอาดข้อมูล"""
    if not isinstance(text, str):
        return ""
    
    # 1. Normalize
    text = normalize(text)
    
    # 2. Remove URL & Numbers
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    
    # 4. Remove Stopwords
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    # 5. Join กลับด้วยช่องว่าง (เพื่อให้ TfidfVectorizer ใน Pipeline ทำงานต่อได้)
    return " ".join(tokens)

# ---------------------------------------------------------
# 2. Load Resources
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    # Load Model (Pipeline: Tfidf -> Classifier)

    model = joblib.load(MODEL_PATH)
    
    # Load MLB & Test Data

    data_bundle = joblib.load(TRAIN_DATA_PATH)
    
    return model, data_bundle

# Load
model, data_bundle = load_resources()

# แยกส่วนประกอบจาก Bundle
mlb = data_bundle['mlb'] if data_bundle else None
X_test = data_bundle['X_test'] if data_bundle else None
y_test = data_bundle['y_test'] if data_bundle else None

# ---------------------------------------------------------
# 3. Main UI
# ---------------------------------------------------------
st.set_page_config(layout="wide")
st.title("🚦 Bangkok Traffy Fondue AI (TF-IDF Version)")
st.caption("Model: TF-IDF + Machine Learning (LinearSVC/SGD)")

if not model or not mlb:
    st.error(f"ไม่พบไฟล์โมเดลที่ {MODEL_PATH} หรือไฟล์ข้อมูลที่ {TRAIN_DATA_PATH}")
    st.info("กรุณารัน Airflow DAG: 'traffy_model_training_selection' ให้เสร็จสมบูรณ์ก่อน")
    st.stop()

# --- Section: Prediction ---
st.subheader("🤖 ทดสอบแจ้งปัญหา")
user_input = st.text_area("ข้อความร้องเรียน:", "น้ำท่วมขังรอการระบาย ที่เขตจตุจักร")

if st.button("ทำนายผล (Predict)"):
    # 1. Preprocess (สำคัญมาก! ต้องตัดคำก่อน)
    processed_text = clean_and_tokenize(user_input)
    
    # 2. Predict
    # ส่งเป็น list เพราะ model คาดหวัง array-like
    pred_binary = model.predict([processed_text])
    
    # 3. แปลงกลับเป็นชื่อ Class
    pred_labels = mlb.inverse_transform(pred_binary)
    final_labels = pred_labels[0]
    
    # 4. แสดงผล
    if final_labels:
        st.success(f"ผลลัพธ์: **{', '.join(final_labels)}**")
    else:
        st.warning("ไม่เข้าข่ายประเภทใดเลย (หรือ Model ไม่มั่นใจ)")

    # 5. Show Confidence (ถ้า Model รองรับ)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([processed_text])[0]
        prob_df = pd.DataFrame({"Category": mlb.classes_, "Confidence": probs})
        prob_df = prob_df.sort_values(by="Confidence", ascending=False).head(5)
        st.write("ระดับความมั่นใจ (Top 5):")
        st.dataframe(prob_df.style.format({"Confidence": "{:.2%}"}).background_gradient(cmap='Blues'))
    elif hasattr(model, "decision_function"):
        st.info("Model นี้ (LinearSVC) ใช้ Decision Function (ระยะห่างจากเส้นแบ่ง) แทนค่า %")

# ---------------------------------------------------------
# 4. Evaluation Section
# ---------------------------------------------------------
st.divider()
st.header("📈 Model Evaluation")

# ใช้ Session State จำค่าปุ่มกด
if 'run_eval' not in st.session_state:
    st.session_state['run_eval'] = False

if st.button("🚀 Calculate Metrics"):
    st.session_state['run_eval'] = True

if st.session_state['run_eval']:
    if st.button("❌ Close Metrics"):
        st.session_state['run_eval'] = False
        st.rerun()

    if X_test is not None and y_test is not None:
        with st.spinner("Evaluating on Test Set..."):
            y_pred = model.predict(X_test)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Report", "🔲 Confusion Matrix", "🔍 Prediction Check", "🔑 Keywords", "📊 Label Distribution"])            
            # --- TAB 1: Report ---
            with tab1:
                # 1. หา Index ของ Class ที่ไม่ใช่ 'nan'
                valid_indices = [i for i, c in enumerate(mlb.classes_) if str(c) != 'nan']
                
                # 2. สร้าง List ชื่อ Class ใหม่ที่ไม่มี nan
                clean_classes = [mlb.classes_[i] for i in valid_indices]
                
                # 3. ตัดคอลัมน์ข้อมูลที่เป็น nan ทิ้งไป (Slicing)
                # y_test และ y_pred ต้องเป็น numpy array (ซึ่งปกติมาจาก model มันเป็นอยู่แล้ว)
                y_test_clean = y_test[:, valid_indices]
                y_pred_clean = y_pred[:, valid_indices]
                
                # 4. ส่งเข้า classification_report (ตอนนี้จำนวนคอลัมน์กับชื่อจะเท่ากันแล้ว)
                report = classification_report(
                    y_test_clean, 
                    y_pred_clean, 
                    target_names=clean_classes, 
                    output_dict=True, 
                    zero_division=0
                )
                
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.background_gradient(cmap='Greens', subset=['f1-score']))

            # --- TAB 2: Confusion Matrix (Plotly Version) ---
            with tab2:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### Select Class")
                    clean_classes = [c for c in mlb.classes_ if str(c) != 'nan']
                    sel_class = st.selectbox("เลือก Class ที่ต้องการดู:", clean_classes)
                    
                with col2:
                    # 1. Calculate Confusion Matrix
                    idx = list(mlb.classes_).index(sel_class)
                    cm = confusion_matrix(y_test[:, idx], y_pred[:, idx])

                    # 2. Plotly Heatmap (px.imshow)
                    # We pass the matrix directly. text_auto=True shows the numbers.
                    fig = px.imshow(
                        cm,
                        text_auto=True, 
                        labels=dict(x="Prediction", y="Actual", color="Count"),
                        x=['Pred: NO', 'Pred: YES'],
                        y=['Actual: NO', 'Actual: YES'],
                        color_continuous_scale='Blues',
                        title=f"Confusion Matrix: {sel_class}"
                    )

                    # 3. Formatting
                    fig.update_layout(
                        width=600,
                        height=500,
                        font=dict(size=14)
                    )
                    
                    # Ensure the text contrast is good (white text on dark blue)
                    fig.update_traces(texttemplate="%{z}") 

                    st.plotly_chart(fig, use_container_width=False)

            # --- TAB 3: Sample Check ---
            with tab3:
                # X_test ใน TF-IDF version เป็น String ที่ตัดคำแล้ว (อ่านรู้เรื่องอยู่)
                limit = 10
                results = []
                true_lbls = mlb.inverse_transform(y_test[:limit])
                pred_lbls = mlb.inverse_transform(y_pred[:limit])
                
                for i in range(limit):
                    is_correct = set(true_lbls[i]) == set(pred_lbls[i])
                    results.append({
                        "Processed Text": X_test[i],
                        "Answer": ", ".join(true_lbls[i]),
                        "Prediction": ", ".join(pred_lbls[i]),
                        "Status": "✅" if is_correct else "❌",
                    })
                    st.dataframe(pd.DataFrame(results))

            # --- TAB 4: Keywords (TF-IDF Feature Importance - Plotly Version) ---
            with tab4:
                st.subheader("คำศัพท์ที่มีอิทธิพลต่อแต่ละ Class")
                try:
                    # 1. Get Feature Names
                    tfidf = model.named_steps['tfidf']
                    feat_names = tfidf.get_feature_names_out()
                    
                    # 2. Select Class

                    clean_classes = [c for c in mlb.classes_ if str(c) != 'nan']
                    target_cls = st.selectbox("เลือก Class เพื่อดู Keywords:", clean_classes, key='fi_select')
                    cls_idx = list(mlb.classes_).index(target_cls)
                    
                    # 3. Get Weights from Model
                    clf = model.named_steps['clf']
                    estimator = clf.estimators_[cls_idx]
                    
                    weights = None
                    if hasattr(estimator, 'coef_'): # LinearSVC, SGD, Logistic
                        weights = estimator.coef_.ravel()
                    elif hasattr(estimator, 'feature_log_prob_'): # NaiveBayes
                        weights = estimator.feature_log_prob_[1]
                        
                    if weights is not None:
                        # 4. Prepare Data
                        df_imp = pd.DataFrame({'Word': feat_names, 'Weight': weights})
                        # Sort and take Top 15
                        df_top = df_imp.sort_values(by='Weight', ascending=False).head(15)
                        
                        # 5. Plotly Horizontal Bar Chart
                        fig = px.bar(
                            df_top, 
                            x='Weight', 
                            y='Word', 
                            orientation='h', # Horizontal Bar
                            title=f"Top 15 Keywords for '{target_cls}'",
                            color='Weight',
                            color_continuous_scale='Viridis', # Green/Purple gradient
                            text='Weight' # Show values on bars
                        )
                        
                        # 6. Formatting
                        fig.update_traces(
                            texttemplate='%{text:.4f}', # Format decimals
                            textposition='outside'
                        )
                        fig.update_layout(
                            yaxis=dict(autorange="reversed"), # Put the highest value at the top
                            xaxis_title="Weight / Importance",
                            yaxis_title="Keyword",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.warning("Model นี้ไม่รองรับ Feature Importance")
                except Exception as e:
                    st.error(f"ไม่สามารถดึง Keywords ได้: {e}")
                # --- TAB 5: Label Distribution (Adapted from your snippet) ---
# --- TAB 5: Label Distribution (Plotly Version) ---
            with tab5:
                st.subheader("จำนวนข้อมูลในแต่ละ Class (Test Set)")

                # 1. Prepare Data
                counts = y_test.sum(axis=0)
                labels = mlb.classes_
                
                # Create DataFrame for Plotly
                dist_df = pd.DataFrame({'Label': labels, 'Count': counts})
                dist_df = dist_df.sort_values(by='Count', ascending=False)

                # 2. Create Plotly Bar Chart
                fig = px.bar(
                    dist_df, 
                    x='Label', 
                    y='Count',
                    text='Count',  # Show values on bars
                    title='Label Distribution',
                    labels={'Label': 'Category', 'Count': 'Number of Records'},
                    color='Count',  # Optional: Color gradient based on value
                    color_continuous_scale='Blues'
                )

                # 3. Customize Layout
                fig.update_traces(
                    texttemplate='%{text}', 
                    textposition='outside' # Put numbers above the bars
                )
                fig.update_layout(
                    xaxis_tickangle=-45, # Rotate labels 45 degrees
                    height=500,          # Set chart height
                    margin=dict(t=50, b=100) # Add margin for labels
                )

                # 4. Show in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("ไม่พบข้อมูล Test Set")
