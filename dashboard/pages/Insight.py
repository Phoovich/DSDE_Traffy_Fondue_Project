import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

st.set_page_config(layout="wide")

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_parquet('data_perped.parquet')
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['type'])
    data = data.dropna(subset=['district'])
    df2 = data[['timestamp', 'type', 'state', 'district']].copy()
    df2['type'] = df2['type'].apply(
        lambda arr: [str(x).strip() for x in arr] if isinstance(arr, np.ndarray) else []
    )
    
    df2 = df2.explode('type').reset_index(drop=True)
    df_sorted = df2.sort_values(['district', 'timestamp'])
    df_sorted['next_type'] = df_sorted.groupby('district')['type'].shift(-1)
    df_sorted['next_state'] = df_sorted.groupby('district')['state'].shift(-1)
    chain = df_sorted[df_sorted['state'].isin(['รอรับเรื่อง', 'กำลังดำเนินการ'])]
    pattern = chain.groupby(['district', 'type', 'next_type']).size().reset_index(name='count')
    pattern = pattern.sort_values('count', ascending=False)
    return pattern

# ---------- Font Thai ----------
font_path = "/System/Library/Fonts/Supplemental/SukhumvitSet.ttc"  # แก้ path ได้ตามเครื่องคุณ
font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False


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
pattern = pattern.dropna()
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
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(pivot_norm, cmap="Blues", annot=False, ax=ax)
    ax.set_title(f"Correlation Heatmap — District: {selected_district}", fontsize=16)
    ax.set_xlabel("Next Type")
    ax.set_ylabel("Type (Unresolved)")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)

    with st.expander("View Top10 Consequence Type", expanded=False):
        st.subheader('Top 10 Consequence')
        top_density = pattern.nlargest(10, 'count')[
            ['district', 'type', 'next_type', 'count']
        ].reset_index(drop=True)
        st.dataframe(top_density, use_container_width=True)
