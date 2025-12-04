import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

st.set_page_config(layout="wide")

# ---------- Font Thai ----------
font_path = r"C:\Windows\Fonts\angsana.ttc"  # แก้ path ได้ตามเครื่องคุณ
font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ---------- Load data ----------
@st.cache_data
def load_data():
    return pd.read_csv("output.csv")

pattern = load_data()

# ---------- Sidebar filter ----------
districts = sorted(pattern['district'].dropna().unique())
selected_district = st.sidebar.selectbox(
    "เลือกเขต (district)", 
    districts
)

st.title("Correlation Heatmap ตามเขต")

st.write(f"เขตที่เลือก: **{selected_district}**")

# ---------- Filter ตามเขต ----------
df_d = pattern[pattern['district'] == selected_district]

if df_d.empty:
    st.warning("ไม่มีข้อมูลในเขตนี้")
else:
    # Pivot: type → next_type matrix
    pivot = df_d.pivot_table(
        index='type',
        columns='next_type',
        values='count',
        fill_value=0
    )

    # Normalize rows
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    # วาดกราฟ
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_norm, cmap="Blues", annot=False, ax=ax)
    ax.set_title(f"Correlation Heatmap — District: {selected_district}", fontsize=16)
    ax.set_xlabel("Next Type")
    ax.set_ylabel("Type (Unresolved)")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)
