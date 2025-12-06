import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Traffy Impact Analysis")

# ---------------------------------------------------------
# 1. LOAD & PREPARE DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # โหลดไฟล์ (ตรวจสอบ path ให้ถูกต้อง)
    try:
        data = pd.read_parquet('./traffy_ml_base.parquet')
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ 'traffy_ml_base.parquet' กรุณาตรวจสอบตำแหน่งไฟล์")
        return pd.DataFrame()

    # แปลง timestamp และลบข้อมูลที่ไม่สมบูรณ์
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['type', 'district'])
    
    # เลือกเฉพาะ column ที่จำเป็น
    df = data[['timestamp', 'type', 'state', 'district']].copy()
    
    # แปลง type ที่เป็น string/list ให้เป็น list เสมอ แล้ว Explode (กระจายแถว)
    # เพื่อให้ 1 แถว มี 1 ปัญหา (เพื่อง่ายต่อการนับคู่ลำดับ)
    df['type'] = df['type'].apply(
        lambda arr: [str(x).strip() for x in arr] if isinstance(arr, np.ndarray) else []
    )
    df = df.explode('type')
    
    # --- CORE LOGIC: Sequential Pattern ---
    # 1. เรียงตาม เขต -> เวลา (เพื่อให้รู้ว่าอะไรเกิดก่อนเกิดหลังในพื้นที่เดียวกัน)
    df = df.sort_values(['district', 'timestamp'])
    
    # 2. Shift(-1) เพื่อดึง "ประเภทปัญหาถัดไป" ขึ้นมาอยู่บรรทัดเดียวกัน
    df['next_type'] = df.groupby('district')['type'].shift(-1)
    
    # 3. กรองเฉพาะ "ปัญหาต้นทาง" ที่ยังแก้ไขไม่เสร็จ (Unresolved)
    # สมมติสถานะที่ยังไม่จบคือ: 'รอรับเรื่อง', 'กำลังดำเนินการ'
    unresolved_states = ['รอรับเรื่อง', 'กำลังดำเนินการ']
    df_chain = df[df['state'].isin(unresolved_states)].copy()
    
    # 4. ลบแถวที่ไม่มีเหตุการณ์ถัดไป (ตัวสุดท้ายของเขต)
    df_chain = df_chain.dropna(subset=['next_type'])
    
    return df_chain

# โหลดข้อมูล
df_main = load_data()

# ---------------------------------------------------------
# 2. SIDEBAR & INTERFACE
# ---------------------------------------------------------
st.title("🔎 Sequential Impact Analysis")
st.markdown("""
**วิเคราะห์ความต่อเนื่องของปัญหา:** ตารางนี้แสดงความน่าจะเป็นว่า **"หากปัญหา A ยังไม่ถูกแก้ไข (Unresolved) -> มีโอกาสแค่ไหนที่จะเกิดปัญหา B ตามมา"**
""")

if not df_main.empty:
    # Sidebar เลือกเขต
    all_districts = sorted(df_main['district'].unique())
    selected_district = st.sidebar.selectbox("📍 เลือกเขต (District)", all_districts)
    
    # กรองข้อมูลตามเขตที่เลือก
    df_district = df_main[df_main['district'] == selected_district]
    
    if df_district.empty:
        st.warning(f"ไม่พบข้อมูลลำดับเหตุการณ์ในเขต {selected_district}")
    else:
        # ---------------------------------------------------------
        # 3. CALCULATION (Pivot & Normalize)
        # ---------------------------------------------------------
        # นับจำนวนคู่ (Start Type -> Next Type)
        pivot_count = pd.crosstab(
            index=df_district['type'], 
            columns=df_district['next_type']
        )
        
        # Normalize เป็น % (Row-wise): หารด้วยจำนวนรวมของแถวนั้นๆ
        # ความหมาย: ถ้าเกิดปัญหาแถว i แล้ว โอกาสเกิดคอลัมน์ j เป็นกี่ %
        pivot_prob = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100

        # ---------------------------------------------------------
        # 4. PLOTLY HEATMAP
        # ---------------------------------------------------------
        fig = px.imshow(
            pivot_prob,
            labels=dict(x="ปัญหาที่ตามมา (Consequence)", y="ปัญหาที่ค้างอยู่ (Unresolved Cause)", color="โอกาสเกิด (%)"),
            x=pivot_prob.columns,
            y=pivot_prob.index,
            color_continuous_scale="Reds",  # สีแดงสื่อถึงความรุนแรง/ความเสี่ยง
            text_auto='.1f',                # แสดงตัวเลขทศนิยม 1 ตำแหน่งในช่อง
            aspect="auto",
            title=f"Impact Heatmap: เขต{selected_district}"
        )

        fig.update_layout(
            height=700,
            xaxis_title="<b>Next Issue</b> (ปัญหาถัดไปที่แจ้งเข้ามา)",
            yaxis_title="<b>Current Issue</b> (ปัญหาที่ยังแก้ไม่เสร็จ)",
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------
        # 5. DATA TABLE (Top Sequences)
        # ---------------------------------------------------------
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Top 10 Most Common Sequences")
            st.caption("คู่ปัญหาที่พบบ่อยที่สุด (นับตามจำนวนครั้ง)")
            
            # จัดกลุ่มนับจำนวน
            top_counts = df_district.groupby(['type', 'next_type']).size().reset_index(name='count')
            top_counts = top_counts.sort_values('count', ascending=False).head(10)
            top_counts.columns = ['Cause (ต้นเหตุ)', 'Effect (ผลที่ตามมา)', 'Count (จำนวนครั้ง)']
            
            st.dataframe(top_counts, use_container_width=True, hide_index=True)

else:
    st.info("กรุณารอโหลดข้อมูล หรือตรวจสอบไฟล์ข้อมูล...")
st.markdown("---") # เส้นคั่นสวยๆ

# ---------------------------------------------------------
# 6. ADDITIONAL GRAPH: Statistical Correlation (Ignore State)
# ---------------------------------------------------------
st.header("🔗 Statistical Correlation Matrix")
st.markdown("""
**ความสัมพันธ์เชิงสถิติ:** กราฟนี้วิเคราะห์ข้อมูลรายวันโดย **ไม่สนสถานะ (State)** ว่าแก้หรือยัง
* **สีน้ำเงินเข้ม (เข้าใกล้ 1):** ปัญหามักจะ **เกิดพร้อมกัน** หรือมาในช่วงเวลาเดียวกัน
* **สีแดง/ขาว (เข้าใกล้ 0 หรือติดลบ):** ไม่มีความสัมพันธ์กัน หรือมักจะไม่เกิดพร้อมกัน
""")

if not df_district.empty:
    # 1. เตรียมข้อมูล: เปลี่ยนเป็น Time Series รายวัน
    # เราต้องนับว่า "ในแต่ละวัน มีปัญหาแต่ละประเภทกี่เรื่อง"
    df_district['date'] = df_district['timestamp'].dt.date
    
    # Crosstab: แถว=วันที่, คอลัมน์=ประเภทปัญหา, ค่า=จำนวนเรื่อง
    daily_counts = pd.crosstab(df_district['date'], df_district['type'])

    # 2. คำนวณ Correlation (Pearson)
    # ดูว่ากราฟเส้นของปัญหาแต่ละคู่ วิ่งไปทิศทางเดียวกันไหม
    corr_matrix = daily_counts.corr(method='pearson')

    # 3. สร้าง Heatmap
    # จัดการค่า NaN (กรณี type นั้นมีข้อมูลเดียวคำนวณ corr ไม่ได้)
    corr_matrix = corr_matrix.fillna(0)

    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(x="Problem Type A", y="Problem Type B", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale="RdBu", # แดง-ขาว-น้ำเงิน (มาตรฐาน Correlation)
        zmin=-1, zmax=1,               # บังคับสเกลให้อยู่ช่วง -1 ถึง 1
        aspect="auto",
        title=f"Correlation Matrix: เขต{selected_district}"
    )

    fig_corr.update_layout(
        height=700,
        xaxis_title="",
        yaxis_title="",
    )

    st.plotly_chart(fig_corr, use_container_width=True)
    
    # (Optional) คำอธิบายเพิ่มเติม
    st.info(f"💡 **Tip:** ข้อมูลนี้คำนวณจากจำนวนเรื่องร้องเรียนรายวันของเขต **{selected_district}** "
            "ถ้าคู่ไหนเป็นสีน้ำเงินเข้ม แปลว่าช่วงไหนที่มีเรื่องหนึ่งแจ้งเข้ามา อีกเรื่องมักจะแจ้งเข้ามาเยอะด้วยเหมือนกัน")

else:
    st.write("ไม่มีข้อมูลเพียงพอสำหรับคำนวณ Correlation")
