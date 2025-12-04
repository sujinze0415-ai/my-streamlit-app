import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------
# 1. Start dashboard app
# -----------------------
st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

st.title("📦 Supply Chain Performance Dashboard")
st.caption("Explore key metrics, product performance, and quality patterns across the supply chain.")

# -----------------------
# 2. Load and clean data
# -----------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()

    numeric_cols = [
        "PRICE", "REVENUE_GENERATED", "COSTS", "DEFECT_RATES", "NUMBER_OF_PRODUCTS_SOLD",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data("supply_chain_data_cleaned.csv")

required_cols = [
    "PRODUCT_TYPE", "LOCATION", "TRANSPORTATION_MODES", "INSPECTION_RESULTS",
    "CUSTOMER_DEMOGRAPHICS", "PRICE", "REVENUE_GENERATED", "COSTS",
    "DEFECT_RATES", "NUMBER_OF_PRODUCTS_SOLD",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

# -------------------------------
# 3. Global filters
# -------------------------------
st.sidebar.header("🔍 Global Filters")
st.sidebar.info("Use these filters to slice the entire dashboard. All visuals update together.")

product_options = sorted(df["PRODUCT_TYPE"].dropna().unique())
selected_products = st.sidebar.multiselect("Product Type", product_options, product_options)

location_options = sorted(df["LOCATION"].dropna().unique())
selected_locations = st.sidebar.multiselect("Location", location_options, location_options)

transport_options = sorted(df["TRANSPORTATION_MODES"].dropna().unique())
selected_transports = st.sidebar.multiselect("Transportation Modes", transport_options, transport_options)

inspect_options = sorted(df["INSPECTION_RESULTS"].dropna().unique())
selected_inspections = st.sidebar.multiselect("Inspection Results", inspect_options, inspect_options)

min_price = float(df["PRICE"].min())
max_price = float(df["PRICE"].max())
price_min, price_max = st.sidebar.slider(
    "Price range", min_value=min_price, max_value=max_price, value=(min_price, max_price)
)

min_sold = st.sidebar.number_input("Min NUMBER_OF_PRODUCTS_SOLD (scatter)", min_value=0, value=0)

donut_category_initial = st.sidebar.selectbox(
    "Default donut category", ["INSPECTION_RESULTS", "CUSTOMER_DEMOGRAPHICS"]
)

filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["PRODUCT_TYPE"].isin(selected_products)]
filtered_df = filtered_df[filtered_df["LOCATION"].isin(selected_locations)]
filtered_df = filtered_df[filtered_df["TRANSPORTATION_MODES"].isin(selected_transports)]
filtered_df = filtered_df[filtered_df["INSPECTION_RESULTS"].isin(selected_inspections)]
filtered_df = filtered_df[(filtered_df["PRICE"] >= price_min) & (filtered_df["PRICE"] <= price_max)]

if filtered_df.empty:
    st.warning("No data after applying filters. Please adjust filters.")
    st.stop()

st.markdown(f"✅ Showing **{len(filtered_df):,}** filtered records.")

# -------------------------------
# 4. KPIs
# -------------------------------
st.subheader("Key Performance Indicators")

total_revenue = filtered_df["REVENUE_GENERATED"].sum()
total_costs = filtered_df["COSTS"].sum()
total_profit = total_revenue - total_costs
avg_defect_rate = filtered_df["DEFECT_RATES"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue 💰", f"${total_revenue:,.0f}")
k2.metric("Total Costs 💸", f"${total_costs:,.0f}")
k3.metric("Total Profit 📈", f"${total_profit:,.0f}")
k4.metric("Avg Defect Rate ⚠️", f"{avg_defect_rate*100:.2f}%")

st.caption("KPIs update dynamically based on filters.")
st.markdown("---")

# ===============================
# 5. Dashboard visual layout
# ===============================
# -------- Row 1: Bar + Scatter --------
col1, col2 = st.columns(2)

# BAR CHART
with col1:
    st.markdown("### Product Performance (Bar Chart)")
    metric_option = st.selectbox(
        "Metric", ["Total Revenue", "Total Costs", "Total Profit"], key="bar_metric"
    )

    grouped = filtered_df.groupby("PRODUCT_TYPE")

    if metric_option == "Total Revenue":
        bar_data = grouped["REVENUE_GENERATED"].sum().reset_index(name="VALUE")
    elif metric_option == "Total Costs":
        bar_data = grouped["COSTS"].sum().reset_index(name="VALUE")
    else:
        bar_data = (
            grouped["REVENUE_GENERATED"].sum() - grouped["COSTS"].sum()
        ).reset_index(name="VALUE")

    bar_fig = px.bar(
        bar_data,
        x="PRODUCT_TYPE",
        y="VALUE",
        color="PRODUCT_TYPE",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Product Performance",
    )

    bar_fig.update_traces(text=bar_data["VALUE"].round(0), textposition="outside")
    bar_fig.update_layout(height=300, margin=dict(t=40, b=40), showlegend=False)

    st.plotly_chart(bar_fig, use_container_width=True)

# SCATTER
with col2:
    st.markdown("### Cost vs Revenue (Scatter)")

    scatter_df = filtered_df[filtered_df["NUMBER_OF_PRODUCTS_SOLD"] >= min_sold]

    if scatter_df.empty:
        st.info("No scatter data under current filters.")
    else:
        color_choice = st.selectbox("Color by", ["PRODUCT_TYPE", "LOCATION"], key="scatter_color")

        scatter_fig = px.scatter(
            scatter_df,
            x="COSTS",
            y="REVENUE_GENERATED",
            color=color_choice,
            size="NUMBER_OF_PRODUCTS_SOLD",
            labels={"COSTS": "Costs (USD)", "REVENUE_GENERATED": "Revenue (USD)"},
            color_discrete_sequence=px.colors.qualitative.Set1,
            title="Costs vs Revenue",
        )

        scatter_fig.update_layout(height=300, margin=dict(t=40, b=40))
        st.plotly_chart(scatter_fig, use_container_width=True)

# -------- Row 2: Heatmap + Donut --------
col3, col4 = st.columns(2)

# HEATMAP
with col3:
    st.markdown("### Defect Rate Heatmap")

    heat = (
        filtered_df.groupby(["LOCATION", "TRANSPORTATION_MODES"])["DEFECT_RATES"]
        .mean()
        .reset_index()
    )
    heat_pivot = heat.pivot(index="LOCATION", columns="TRANSPORTATION_MODES", values="DEFECT_RATES")

    heat_fig = px.imshow(
        heat_pivot,
        labels=dict(x="Transport Mode", y="Location", color="Avg Defect Rate"),
        color_continuous_scale="Reds",
        title="Defect Rate by Location & Transport",
    )

    heat_fig.update_traces(
        text=(heat_pivot.values * 100).round(1),
        texttemplate="%{text}%",
        textfont=dict(size=10),
    )
    heat_fig.update_layout(height=300, margin=dict(t=40, b=40))

    st.plotly_chart(heat_fig, use_container_width=True)

# DONUT
with col4:
    st.markdown("### Category Distribution (Donut)")

    donut_category = st.selectbox(
        "Category",
        ["INSPECTION_RESULTS", "CUSTOMER_DEMOGRAPHICS"],
        index=["INSPECTION_RESULTS", "CUSTOMER_DEMOGRAPHICS"].index(donut_category_initial),
        key="donut_category_tab",
    )

    donut_counts = (
        filtered_df[donut_category]
        .value_counts()
        .reset_index(name="COUNT")
        .rename(columns={"index": donut_category})
    )

    donut_fig = px.pie(
        donut_counts,
        names=donut_category,
        values="COUNT",
        hole=0.5,
        title=f"{donut_category} Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    donut_fig.update_layout(height=300, margin=dict(t=40, b=40))
    donut_fig.update_traces(textinfo="percent+label", textposition="inside")

    st.plotly_chart(donut_fig, use_container_width=True)
