import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import ast
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Traffy Fondue Dashboard")

# ---------------------------------------------------------
# 1. ROBUST DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('./data_perped.parquet')
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return pd.DataFrame()

    cols = ['longitude', 'latitude', 'district', 'subdistrict', 'timestamp', 'comment', 'type', 'state', 'ticket_id']
    cols = [c for c in cols if c in df.columns] 
    df = df[cols].copy()

    def safe_convert_to_list(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, list):
            return val
        if val is None or pd.isna(val):
            return []
        if isinstance(val, str):
            val = val.strip()
            if val == "" or val == "[]": 
                return []
            try:
                return ast.literal_eval(val)
            except:
                try:
                    clean_str = val.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
                    return [item.strip() for item in clean_str.split(',') if item.strip()]
                except:
                    return []
        return []

    if 'type' in df.columns:
        df['type'] = df['type'].apply(safe_convert_to_list)

    if len(df) > 100000:
        df = df.sample(n=100000, random_state=108)
        
    return df


@st.cache_data
def run_dbscan(coords_array, eps, min_samples):
    """Run DBSCAN clustering on coordinates"""
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords_array)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(coords_scaled)
    return clusters


@st.cache_data
def compute_kde(lat, lon, grid_size=100, bw_method=None):
    """Compute 2D KDE for geographic coordinates"""
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    
    lat_pad = (lat_max - lat_min) * 0.05
    lon_pad = (lon_max - lon_min) * 0.05
    
    lat_grid = np.linspace(lat_min - lat_pad, lat_max + lat_pad, grid_size)
    lon_grid = np.linspace(lon_min - lon_pad, lon_max + lon_pad, grid_size)
    
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    positions = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()])
    
    values = np.vstack([lat, lon])
    kernel = stats.gaussian_kde(values, bw_method=bw_method)
    density = np.reshape(kernel(positions).T, lat_mesh.shape)
    
    return lat_grid, lon_grid, density, lat_mesh, lon_mesh


# Run Load Data
df = load_data()

if df.empty:
    st.stop()

st.title("Traffy Fondue")
st.caption("Spatial Clustering & Density Analysis")

# ---------------------------------------------------------
# 2. SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")

# --- Filter A: Type of Problem ---
raw_types = df['type'].explode().unique()
unique_types = []
for t in raw_types:
    s = str(t).strip()
    if s and s.lower() != 'nan' and s.lower() != 'none':
        unique_types.append(s)
unique_types = sorted(unique_types)

selected_types = st.sidebar.multiselect("Problem Type", unique_types)

# --- Filter B: District ---
all_districts = sorted(df['district'].dropna().unique().astype(str))
selected_districts = st.sidebar.multiselect("District", all_districts)

# --- Filter C: Subdistrict ---
if selected_districts:
    sub_options = sorted(df[df['district'].isin(selected_districts)]['subdistrict'].dropna().unique().astype(str))
else:
    sub_options = sorted(df['subdistrict'].dropna().unique().astype(str))
selected_subdistricts = st.sidebar.multiselect("Subdistrict", sub_options)

# --- Filter D: State ---
if 'state' in df.columns:
    all_states = sorted(df['state'].dropna().unique().astype(str))
    selected_states = st.sidebar.multiselect("State", all_states)
else:
    selected_states = []

st.sidebar.markdown("---")

# ---------------------------------------------------------
# 3. DBSCAN & KDE PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("DBSCAN Parameters")

eps = st.sidebar.slider(
    "Epsilon (eps)",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="Maximum distance between two samples to be considered in the same neighborhood"
)

min_samples = st.sidebar.slider(
    "Minimum Samples",
    min_value=2,
    max_value=100,
    value=10,
    step=1,
    help="Minimum number of samples in a neighborhood to form a cluster"
)

st.sidebar.header("KDE Parameters")

kde_bandwidth = st.sidebar.slider(
    "Bandwidth",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Controls smoothness of KDE"
)

kde_grid_size = st.sidebar.slider(
    "Grid Resolution",
    min_value=50,
    max_value=200,
    value=100,
    step=10,
    help="Resolution of the density grid"
)

# ---------------------------------------------------------
# 4. APPLY FILTERS
# ---------------------------------------------------------
filtered_df = df.copy()

if selected_types:
    target_set = set(selected_types)
    filtered_df = filtered_df[filtered_df['type'].apply(lambda x: bool(set(x) & target_set))]

if selected_districts:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]

if selected_subdistricts:
    filtered_df = filtered_df[filtered_df['subdistrict'].isin(selected_subdistricts)]

if selected_states and 'state' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]

# Remove rows without coordinates
filtered_df = filtered_df.dropna(subset=['latitude', 'longitude'])

st.sidebar.markdown(f"**Filtered records:** {len(filtered_df):,}")

if filtered_df.empty:
    st.warning("No data matches these filters.")
    st.stop()

if len(filtered_df) < min_samples:
    st.warning(f"Not enough data points ({len(filtered_df)}) for clustering with min_samples={min_samples}")
    st.stop()

# ---------------------------------------------------------
# 5. RUN DBSCAN
# ---------------------------------------------------------
with st.spinner("Running DBSCAN clustering..."):
    coords_array = filtered_df[["latitude", "longitude"]].values
    clusters = run_dbscan(coords_array, eps, min_samples)
    filtered_df = filtered_df.copy()
    filtered_df["cluster"] = clusters

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Points", f"{len(filtered_df):,}")
with col2:
    st.metric("Clusters Found", n_clusters)
with col3:
    st.metric("Noise Points", f"{n_noise:,}")
with col4:
    st.metric("Clustered Points", f"{len(filtered_df) - n_noise:,}")

# Color palette
colors = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#6b6ecf", "#9c9ede", "#b5cf6b", "#cedb9c",
]

# ---------------------------------------------------------
# 6. DBSCAN CLUSTERING MAP
# ---------------------------------------------------------
st.markdown("---")
st.header("DBSCAN Clustering")
st.markdown("*Geographic distribution of identified clusters*")

df_plot = filtered_df.copy()
df_plot["cluster_label"] = df_plot["cluster"].apply(
    lambda x: "Noise" if x == -1 else f"Cluster {x}"
)

unique_clusters = sorted(df_plot["cluster"].unique())
color_map = {}
for i, c in enumerate(unique_clusters):
    if c == -1:
        color_map["Noise"] = "#404040"
    else:
        color_map[f"Cluster {c}"] = colors[i % len(colors)]

# Convert type list to string for display
df_plot["type_str"] = df_plot["type"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

fig = px.scatter_mapbox(
    df_plot,
    lat="latitude",
    lon="longitude",
    color="cluster_label",
    color_discrete_map=color_map,
    hover_data=["type_str", "district", "subdistrict"],
    zoom=10,
    height=600,
    title="DBSCAN Clustering Results"
)

fig.update_layout(
    mapbox_style="carto-darkmatter",
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 7. KDE HEATMAP
# ---------------------------------------------------------
st.markdown("---")
st.header("Kernel Density Estimation")
st.markdown("*Spatial density distribution of complaints*")

with st.spinner("Computing KDE..."):
    lat_vals = filtered_df["latitude"].values
    lon_vals = filtered_df["longitude"].values
    
    lat_grid, lon_grid, density, lat_mesh, lon_mesh = compute_kde(
        lat_vals, lon_vals, 
        grid_size=kde_grid_size, 
        bw_method=kde_bandwidth
    )

fig_kde = go.Figure()

fig_kde.add_trace(go.Densitymapbox(
    lat=lat_vals,
    lon=lon_vals,
    z=np.ones(len(lat_vals)),
    radius=10 + int(kde_bandwidth * 10),
    colorscale=[
        [0, "rgba(0,0,0,0)"],
        [0.1, "#440154"],
        [0.3, "#3b528b"],
        [0.5, "#21918c"],
        [0.7, "#5ec962"],
        [0.9, "#fde725"],
        [1.0, "#ffffff"]
    ],
    showscale=True,
    colorbar=dict(
        title=dict(text="Density", font=dict(color="white")),
        tickfont=dict(color="white")
    )
))

fig_kde.update_layout(
    mapbox_style="carto-darkmatter",
    mapbox=dict(
        center=dict(lat=lat_vals.mean(), lon=lon_vals.mean()),
        zoom=10
    ),
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    height=600,
    title="Complaint Density Heatmap",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font_color="white"
)

st.plotly_chart(fig_kde, use_container_width=True)

# 2D contour plot
st.markdown("**Density Contour Plot**")

fig_contour = go.Figure()

fig_contour.add_trace(go.Contour(
    x=lon_grid,
    y=lat_grid,
    z=density,
    colorscale="Viridis",
    contours=dict(
        showlabels=True,
        labelfont=dict(size=10, color="white")
    ),
    colorbar=dict(
        title=dict(text="Density", font=dict(color="white")),
        tickfont=dict(color="white")
    )
))

scatter_sample = min(2000, len(filtered_df))
if len(filtered_df) > scatter_sample:
    scatter_df = filtered_df.sample(n=scatter_sample, random_state=42)
else:
    scatter_df = filtered_df

fig_contour.add_trace(go.Scatter(
    x=scatter_df["longitude"],
    y=scatter_df["latitude"],
    mode="markers",
    marker=dict(size=3, color="white", opacity=0.3),
    name="Data Points",
    hoverinfo="skip"
))

fig_contour.update_layout(
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    height=500,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1a1a2e",
    font_color="white",
    xaxis=dict(gridcolor="#2d2d44", zerolinecolor="#2d2d44"),
    yaxis=dict(gridcolor="#2d2d44", zerolinecolor="#2d2d44", scaleanchor="x", scaleratio=1)
)

st.plotly_chart(fig_contour, use_container_width=True)

# KDE statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Peak Density (Lat)", f"{lat_grid[np.unravel_index(density.argmax(), density.shape)[0]]:.4f}")
with col2:
    st.metric("Peak Density (Lon)", f"{lon_grid[np.unravel_index(density.argmax(), density.shape)[1]]:.4f}")
with col3:
    st.metric("Max Density Value", f"{density.max():.2e}")

# ---------------------------------------------------------
# 8. CLUSTER ANALYSIS
# ---------------------------------------------------------
st.markdown("---")
st.header("Cluster Analysis")

col1, col2 = st.columns(2)

with col1:
    cluster_counts = filtered_df[filtered_df["cluster"] != -1]["cluster"].value_counts().sort_index()
    
    if len(cluster_counts) > 0:
        bar_colors = [colors[i % len(colors)] for i in cluster_counts.index]
        fig_bar = px.bar(
            x=[f"Cluster {i}" for i in cluster_counts.index],
            y=cluster_counts.values,
            title="Cluster Sizes",
            labels={"x": "Cluster", "y": "Number of Points"},
            color_discrete_sequence=bar_colors
        )
        fig_bar.update_traces(marker_color=bar_colors)
        fig_bar.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1a2e",
            font_color="white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No clusters found with current parameters")

with col2:
    fig_pie = px.pie(
        values=[len(filtered_df) - n_noise, n_noise],
        names=["Clustered", "Noise"],
        title="Clustered vs Noise Points",
        color_discrete_sequence=["#1f77b4", "#404040"]
    )
    fig_pie.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="white"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Cluster details
if n_clusters > 0:
    st.markdown("**Cluster Details**")
    
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_data = filtered_df[filtered_df["cluster"] == cluster_id]
        stats_dict = {
            "Cluster": f"Cluster {cluster_id}",
            "Size": len(cluster_data),
            "Avg Latitude": cluster_data["latitude"].mean(),
            "Avg Longitude": cluster_data["longitude"].mean(),
        }
        
        if "district" in cluster_data.columns:
            top_district = cluster_data["district"].mode()
            stats_dict["Top District"] = top_district.iloc[0] if len(top_district) > 0 else "N/A"
        
        cluster_stats.append(stats_dict)
    
    st.dataframe(pd.DataFrame(cluster_stats), use_container_width=True)

# ---------------------------------------------------------
# 9. DATA TABLE
# ---------------------------------------------------------
st.markdown("---")
st.header("Data Table")

cluster_filter = st.selectbox(
    "Filter by Cluster",
    options=["All"] + [f"Cluster {i}" for i in range(n_clusters)] + ["Noise"],
)

if cluster_filter == "All":
    display_df = filtered_df
elif cluster_filter == "Noise":
    display_df = filtered_df[filtered_df["cluster"] == -1]
else:
    cluster_num = int(cluster_filter.split()[-1])
    display_df = filtered_df[filtered_df["cluster"] == cluster_num]

display_cols = ["ticket_id", "type", "district", "subdistrict", "state", "latitude", "longitude", "cluster"]
display_cols = [col for col in display_cols if col in display_df.columns]

st.dataframe(display_df[display_cols], use_container_width=True, height=400)

csv = display_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="traffy_dbscan_results.csv",
    mime="text/csv"
)
