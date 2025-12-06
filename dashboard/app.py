import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import ast

st.set_page_config(layout="wide", page_title="Traffy Fondue Debug Mode")

# ---------------------------------------------------------
# 1. ROBUST DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('./traffy_ml_base.parquet')
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return pd.DataFrame()

    cols = ['longitude', 'latitude', 'district', 'subdistrict', 'timestamp', 'comment', 'type']
    cols = [c for c in cols if c in df.columns] 
    df = df[cols].copy()

    # --- THE FIX: Updated Safe Converter ---
    def safe_convert_to_list(val):
        # 1. Handle NumPy Arrays (The cause of your error)
        if isinstance(val, np.ndarray):
            return val.tolist()
            
        # 2. Handle Lists
        if isinstance(val, list):
            return val
        
        # 3. Handle None/NaN (Safe now because arrays are handled above)
        if val is None or pd.isna(val):
            return []
            
        # 4. Handle Strings
        if isinstance(val, str):
            val = val.strip()
            if val == "" or val == "[]": 
                return []
            try:
                # Try standard python syntax "['a', 'b']"
                return ast.literal_eval(val)
            except:
                try:
                    # Fallback cleanup
                    clean_str = val.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
                    return [item.strip() for item in clean_str.split(',') if item.strip()]
                except:
                    return []
        
        # 5. Anything else
        return []

    if 'type' in df.columns:
        df['type'] = df['type'].apply(safe_convert_to_list)

    if len(df) > 100000:
        df = df.sample(n=100000, random_state=108)
        
    return df

# Run Load Data
df = load_data()

if df.empty:
    st.stop() # Stop if data load failed

st.title("Traffy Fondue Report (Fixed Version)")

# ---------------------------------------------------------
# 2. DEBUG SECTION (Check your data here!)
# ---------------------------------------------------------
with st.expander("🛠️ DEBUG: Check Data Format", expanded=False):
    st.write("First 5 rows of 'type' column after cleaning:")
    st.write(df['type'].head(5))
    st.write(f"Total rows: {len(df)}")
    
    # Check all unique tags found
    all_tags = [item for sublist in df['type'] for item in sublist]
    unique_tags = sorted(list(set(all_tags)))
    st.write(f"Unique Tags Found ({len(unique_tags)}):", unique_tags)

# ---------------------------------------------------------
# 3. SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Filter Options")

# --- Filter A: Type of Problem ---
# 1. Explode to get all individual tags
raw_types = df['type'].explode().unique()

# 2. Clean up: Remove None, empty strings, and 'nan'
unique_types = []
for t in raw_types:
    # Convert to string and strip whitespace
    s = str(t).strip()
    # Check if it is a valid string and not 'nan'
    if s and s.lower() != 'nan' and s.lower() != 'none':
        unique_types.append(s)

unique_types = sorted(unique_types)

selected_types = st.sidebar.multiselect("Select Problem Type", unique_types)

# --- Filter B: District ---
all_districts = sorted(df['district'].dropna().unique().astype(str))
selected_districts = st.sidebar.multiselect("Select District", all_districts)

# --- Filter C: Subdistrict ---
if selected_districts:
    sub_options = sorted(df[df['district'].isin(selected_districts)]['subdistrict'].dropna().unique().astype(str))
else:
    sub_options = sorted(df['subdistrict'].dropna().unique().astype(str))
    
selected_subdistricts = st.sidebar.multiselect("Select Subdistrict", sub_options)

# ---------------------------------------------------------
# 4. APPLY FILTERS
# ---------------------------------------------------------
filtered_df = df.copy()

# Filter Logic
if selected_types:
    target_set = set(selected_types)
    # Check if intersection is valid
    filtered_df = filtered_df[filtered_df['type'].apply(lambda x: bool(set(x) & target_set))]

if selected_districts:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]

if selected_subdistricts:
    filtered_df = filtered_df[filtered_df['subdistrict'].isin(selected_subdistricts)]

st.sidebar.markdown("---")
st.sidebar.write(f"Found: **{len(filtered_df)}** items")

if filtered_df.empty:
    st.warning("No data matches these filters.")
    st.stop()

# ---------------------------------------------------------
# 5. MAPS
# ---------------------------------------------------------
# Recalculate center
view_lat = filtered_df['latitude'].mean()
view_lon = filtered_df['longitude'].mean()
zoom_level = 11 if (selected_districts or selected_types) else 10

heat_layer = pdk.Layer(
    "HeatmapLayer",
    filtered_df,
    opacity=0.8,
    get_position=["longitude", "latitude"],
    threshold=0.1,
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    filtered_df,
    get_position=["longitude", "latitude"],
    get_radius=80,
    get_fill_color=[255, 140, 0],
    opacity=0.6,
    pickable=True
)

col1, col2 = st.columns([100, 1])
with col1:
    st.subheader("Scatter Plot")
    st.pydeck_chart(
        pdk.Deck(
            layers=[scatter_layer],
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=zoom_level),
            tooltip={"text": "Type: {type}\nLoc: {subdistrict}, {district}\n{comment}"}
        ),
        use_container_width=True
    )

    st.subheader("Heat Map")
    st.pydeck_chart(
        pdk.Deck(
            layers=[heat_layer],
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=zoom_level),
        ),
        use_container_width=True
    )
