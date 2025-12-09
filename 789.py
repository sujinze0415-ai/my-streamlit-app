import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import math


# ----------------------------------
# Haversine distance (km)
# ----------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


# ------------------------------
# 1. Connect to SQLite & Load Data
# ------------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect("project.db")

    # City-level aggregated Starbucks data for California
    df_city = pd.read_sql_query("SELECT * FROM starbucks_ca_city;", conn)

    # State-level census info (single row, background use)
    df_census_ca = pd.read_sql_query("SELECT * FROM census_ca;", conn)

    # Store-level Starbucks data for California
    df_store = pd.read_sql_query(
        """
        SELECT store_number,
               store_name,
               city,
               state,
               latitude  AS lat,
               longitude AS lon,
               ownership_type
        FROM starbucks_ca;
        """,
        conn,
    )

    # City + county + demographics (created by build_income_tables.py)
    df_city_income = pd.read_sql_query("SELECT * FROM city_income;", conn)

    conn.close()

    # Clean coordinates for store table
    df_store["lat"] = pd.to_numeric(df_store["lat"], errors="coerce")
    df_store["lon"] = pd.to_numeric(df_store["lon"], errors="coerce")
    df_store = df_store.dropna(subset=["lat", "lon"])

    # Clean coordinates for city table
    df_city = df_city.dropna(subset=["mean_latitude", "mean_longitude"])

    return df_city, df_census_ca, df_store, df_city_income


df_city, df_census_ca, df_store, df_city_income = load_data()

# ------------------------------
# 2. Page Title & Introduction
# ------------------------------
st.title("Starbucks California Location Analysis App")

st.write(
    "This application visualizes Starbucks store distribution across California cities, "
    "supported by state-level census data and county-level demographics. "
    "It provides exploratory insights for potential site-selection discussions."
)

# ------------------------------
# 3. Sidebar Filters (city-level)
# ------------------------------
st.sidebar.header("Filters")

min_stores = int(df_city["num_stores"].min())
max_stores = int(df_city["num_stores"].max())

store_range = st.sidebar.slider(
    "Range of Starbucks store counts (per city)",
    min_value=min_stores,
    max_value=max_stores,
    value=(min_stores, max_stores),
)

selected_cities = st.sidebar.multiselect(
    "Filter by specific cities (leave empty for all cities)",
    options=sorted(df_city["city"].unique()),
)

# Apply filters to city-level table
filtered = df_city[
    (df_city["num_stores"] >= store_range[0])
    & (df_city["num_stores"] <= store_range[1])
]

if selected_cities:
    filtered = filtered[filtered["city"].isin(selected_cities)]

# ------------------------------
# 4. California Summary Metrics
# ------------------------------
st.subheader("California Overview")

census_row = df_census_ca.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Population", f"{int(census_row.total_pop):,}")
col2.metric("Average Household Income", f"${int(census_row.avg_income):,}")
col3.metric("Total Starbucks Stores", int(df_city["num_stores"].sum()))
col4.metric("Number of Cities with Starbucks", df_city["city"].nunique())

st.markdown("---")

# ------------------------------
# 5. City-Level Store Distribution Table
# ------------------------------
st.subheader("Starbucks Store Distribution Across California Cities (Filtered)")

filtered_sorted = filtered.sort_values("num_stores", ascending=False)
st.dataframe(filtered_sorted)

st.markdown("---")

# ------------------------------
# 6A. Store Count vs County Population (2025)
# ------------------------------
st.subheader("📊 Store Count vs County Population (2025)")

df_pop = df_city_income.dropna(subset=["population_2025"]).copy()

county_options = ["All counties"] + sorted(
    df_pop["county"].dropna().unique().tolist()
)
selected_county = st.selectbox(
    "Filter by county (optional)",
    county_options,
    index=0,
    key="pop_county_filter",
)

if selected_county != "All counties":
    df_pop = df_pop[df_pop["county"] == selected_county]

fig_pop = px.scatter(
    df_pop,
    x="population_2025",
    y="num_stores",
    color="county",
    size="num_stores",
    hover_name="city",
    hover_data={
        "county": True,
        "population_2025": ":,",
        "median_income": ":$,",
        "num_stores": True,
    },
    labels={
        "population_2025": "County Population (2025)",
        "num_stores": "Number of Stores",
        "county": "County",
    },
    title="Starbucks Store Count vs County Population (2025)",
)

fig_pop.update_traces(opacity=0.85)
st.plotly_chart(fig_pop, use_container_width=True)

st.markdown("---")

# ------------------------------
# 6B. Store Count vs County Median Income
# ------------------------------
st.subheader("💵 Store Count vs County Median Income")

df_income = df_city_income.dropna(subset=["median_income"]).copy()

county_options2 = ["All counties"] + sorted(
    df_income["county"].dropna().unique().tolist()
)
selected_county2 = st.selectbox(
    "Filter by county (optional)",
    county_options2,
    index=0,
    key="income_county_filter",
)

if selected_county2 != "All counties":
    df_income = df_income[df_income["county"] == selected_county2]

fig_income = px.scatter(
    df_income,
    x="median_income",
    y="num_stores",
    color="county",
    size="num_stores",
    hover_name="city",
    hover_data={
        "county": True,
        "median_income": ":$,",
        "population_2025": ":,",
        "num_stores": True,
    },
    labels={
        "median_income": "County Median Income (USD)",
        "num_stores": "Number of Stores",
        "county": "County",
    },
    title="Starbucks Store Count vs County Median Income",
)

fig_income.update_traces(opacity=0.85)
st.plotly_chart(fig_income, use_container_width=True)

st.markdown("---")

# ------------------------------
# 6C. Store Density vs Distance to Major Metro
# ------------------------------
st.subheader("📍 Store Density vs Distance to Nearest Major Metro")

metros = {
    "Los Angeles": (34.0522, -118.2437),
    "San Francisco": (37.7749, -122.4194),
    "San Diego": (32.7157, -117.1611),
}

distances = []
nearest_metros = []

for _, row in df_city_income.iterrows():
    city_lat = row["mean_latitude"]
    city_lon = row["mean_longitude"]

    min_dist = float("inf")
    closest_name = None
    for name, (m_lat, m_lon) in metros.items():
        dist = haversine(city_lat, city_lon, m_lat, m_lon)
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    distances.append(min_dist)
    nearest_metros.append(closest_name)

df_city_income["dist_to_metro"] = distances
df_city_income["nearest_metro"] = nearest_metros

max_dist = float(df_city_income["dist_to_metro"].max())
dist_limit = st.slider(
    "Maximum distance to nearest metro (km)",
    min_value=0.0,
    max_value=round(max_dist, 1),
    value=round(min(200.0, max_dist), 1),
)

df_dist = df_city_income[df_city_income["dist_to_metro"] <= dist_limit].copy()

fig_dist = px.scatter(
    df_dist,
    x="dist_to_metro",
    y="num_stores",
    color="nearest_metro",
    size="num_stores",
    hover_name="city",
    hover_data={
        "nearest_metro": True,
        "dist_to_metro": ":.1f",
        "population_2025": ":,",
        "median_income": ":$,",
        "num_stores": True,
    },
    labels={
        "dist_to_metro": "Distance to Nearest Metro (km)",
        "num_stores": "Store Count",
        "nearest_metro": "Nearest Metro",
    },
    title="City Store Count vs Distance to Nearest Major Metro Area",
)

fig_dist.update_traces(opacity=0.85)
st.plotly_chart(fig_dist, use_container_width=True)

st.caption(
    "Bubbles closer to the left are nearer to large metropolitan areas "
    "(Los Angeles, San Francisco, San Diego). Bubble size represents store count."
)

st.markdown("---")

# ------------------------------
# 7. Store-Level Scatter Map
# ------------------------------
st.subheader("Store-Level Starbucks Distribution Across California")

fig_map = px.scatter_mapbox(
    df_store,
    lat="lat",
    lon="lon",
    hover_name="store_name",
    hover_data={"city": True, "ownership_type": True, "lat": False, "lon": False},
    zoom=4.5,
    height=550,
    color="city",
)

fig_map.update_layout(mapbox_style="open-street-map")
fig_map.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})

st.plotly_chart(fig_map, use_container_width=True)

st.caption(
    "Each point represents a Starbucks store. Color indicates the city; clusters are visible "
    "around major metropolitan areas such as the Bay Area and Los Angeles."
)

st.markdown("---")

# ------------------------------
# 8. Donut: Store Share by County
# ------------------------------
st.subheader("🍩 Store Share by County")

df_donut = (
    df_city_income.groupby("county", as_index=False)["num_stores"]
    .sum()
    .sort_values("num_stores", ascending=False)
)

top_counties = st.slider(
    "Number of top counties to show in the donut chart",
    min_value=5,
    max_value=min(20, len(df_donut)),
    value=10,
)

df_donut_top = df_donut.head(top_counties)

fig_donut = px.pie(
    df_donut_top,
    names="county",
    values="num_stores",
    hole=0.55,
    labels={"county": "County", "num_stores": "Store Count"},
    title=f"Share of Starbucks Stores – Top {top_counties} Counties",
)

fig_donut.update_traces(textposition="inside", textinfo="percent+label")

st.plotly_chart(fig_donut, use_container_width=True)

st.caption(
    "The donut chart shows how Starbucks stores are distributed across the top counties "
    "in California. Each slice represents the share of total stores in that county."
)

st.markdown("---")

# ------------------------------
# 9. Composite Potential Score (Population + Income + Distance + Stores)
# ------------------------------
st.subheader("⭐ Composite City Potential Score")

df_score = df_city_income.copy()
required_cols = ["population_2025", "median_income", "dist_to_metro", "num_stores"]
df_score = df_score.dropna(subset=required_cols).copy()

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

df_score["pop_norm"] = normalize(df_score["population_2025"])
df_score["income_norm"] = normalize(df_score["median_income"])
df_score["dist_norm"] = 1 - normalize(df_score["dist_to_metro"])   # closer → higher
df_score["store_norm"] = 1 - normalize(df_score["num_stores"])     # fewer stores → higher

st.write("Adjust component weights:")

w_pop = st.slider("Weight: Population", 0.0, 1.0, 0.35, 0.05)
w_income = st.slider("Weight: Median Income", 0.0, 1.0, 0.35, 0.05)
w_dist = st.slider("Weight: Metro Proximity (Closer = Higher)", 0.0, 1.0, 0.20, 0.05)
w_store = st.slider("Weight: Store Scarcity (Fewer Stores = Higher)", 0.0, 1.0, 0.10, 0.05)

weight_sum = w_pop + w_income + w_dist + w_store
if weight_sum == 0:
    weight_sum = 1

df_score["potential_score"] = (
    w_pop * df_score["pop_norm"] +
    w_income * df_score["income_norm"] +
    w_dist * df_score["dist_norm"] +
    w_store * df_score["store_norm"]
) / weight_sum

df_score_ranked = df_score.sort_values("potential_score", ascending=False)

top_n_score = st.slider("Number of top cities to show in ranking", 5, 30, 15)

fig_comp = px.bar(
    df_score_ranked.head(top_n_score),
    x="potential_score",
    y="city",
    orientation="h",
    labels={
        "potential_score": "Composite Potential Score",
        "city": "City"
    },
    title="Top Cities by Composite Potential Score",
    color="potential_score",
    color_continuous_scale="Blues"
)

fig_comp.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_comp, use_container_width=True)

st.caption(
    "This composite score combines population, income, metro proximity, and existing store counts. "
    "Higher values indicate greater potential for new Starbucks locations."
)

st.markdown("---")

# ------------------------------
# 10. Expandable City Details (based on composite score)
# ------------------------------
st.subheader("City Details (Based on Composite Potential Score)")

top_detail_n = st.slider(
    "Number of high-potential cities to display", 5, 20, 10
)

top_detail_cities = df_score_ranked.head(top_detail_n).copy()

for _, row in top_detail_cities.iterrows():
    city_name = row["city"]
    with st.expander(
        f"{city_name} — Stores: {int(row['num_stores'])}, "
        f"Composite Score: {row['potential_score']:.3f}"
    ):
        st.write(f"**City:** {city_name}")
        st.write(f"**County:** {row['county']}")
        st.write(f"**Store Count:** {int(row['num_stores'])}")
        st.write(
            f"**Mean Coordinates:** ({row['mean_latitude']:.4f}, "
            f"{row['mean_longitude']:.4f})"
        )
        st.write(
            f"**County Population (2025):** {int(row['population_2025']):,}"
        )
        st.write(
            f"**County Median Income:** ${int(row['median_income']):,}"
        )
        st.write(
            f"**Distance to Nearest Metro:** {row['dist_to_metro']:.1f} km "
            f"({row['nearest_metro']})"
        )

        city_stores = df_store[df_store["city"] == city_name][
            ["store_number", "store_name", "ownership_type", "lat", "lon"]
        ]
        st.write("**Store Details:**")
        st.dataframe(city_stores)

        if not city_stores.empty:
            fig_city_map = px.scatter_mapbox(
                city_stores,
                lat="lat",
                lon="lon",
                hover_name="store_name",
                zoom=9,
                height=350,
                color_discrete_sequence=["green"],
            )
            fig_city_map.update_layout(mapbox_style="open-street-map")
            fig_city_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig_city_map, use_container_width=True)

st.caption(
    "Cities above are ranked by the composite potential score to help quickly identify "
    "locations that may have more room for expansion.")