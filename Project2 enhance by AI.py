import streamlit as st
import pandas as pd
import numpy as np  # AI-enhanced: used for jitter in scatter
import plotly.express as px

# -----------------------
# 1. Start dashboard app
# -----------------------
st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")
st.title("Supply Chain Dashboard")


# -----------------------
# 2. Load and clean data
# -----------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Convert key numeric columns to numeric types
    numeric_cols = [
        "PRICE",
        "REVENUE_GENERATED",
        "COSTS",
        "DEFECT_RATES",
        "NUMBER_OF_PRODUCTS_SOLD",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df = load_data("supply_chain_data_cleaned.csv")

required_cols = [
    "PRODUCT_TYPE",
    "LOCATION",
    "TRANSPORTATION_MODES",
    "INSPECTION_RESULTS",
    "CUSTOMER_DEMOGRAPHICS",
    "PRICE",
    "REVENUE_GENERATED",
    "COSTS",
    "DEFECT_RATES",
    "NUMBER_OF_PRODUCTS_SOLD",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()


# -------------------------------
# 3. Create global filters
# -------------------------------
st.sidebar.header("Filters")

# PRODUCT_TYPE filter
product_options = sorted(df["PRODUCT_TYPE"].dropna().unique())
selected_products = st.sidebar.multiselect(
    "Product Type", options=product_options, default=product_options
)

# LOCATION filter
location_options = sorted(df["LOCATION"].dropna().unique())
selected_locations = st.sidebar.multiselect(
    "Location", options=location_options, default=location_options
)

# TRANSPORTATION_MODES filter
transport_options = sorted(df["TRANSPORTATION_MODES"].dropna().unique())
selected_transports = st.sidebar.multiselect(
    "Transportation Modes", options=transport_options, default=transport_options
)

# INSPECTION_RESULTS filter
inspect_options = sorted(df["INSPECTION_RESULTS"].dropna().unique())
selected_inspections = st.sidebar.multiselect(
    "Inspection Results", options=inspect_options, default=inspect_options
)

# Price range slider
min_price = float(df["PRICE"].min())
max_price = float(df["PRICE"].max())
price_min, price_max = st.sidebar.slider(
    "Price range",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
)

# Min number of products sold (for scatter plot)
min_sold = st.sidebar.number_input(
    "Minimum NUMBER_OF_PRODUCTS_SOLD (for scatter plot)",
    min_value=0,
    value=0,
)

# Donut chart category choice
donut_category = st.sidebar.selectbox(
    "Donut category",
    options=["INSPECTION_RESULTS", "CUSTOMER_DEMOGRAPHICS"],
)


# Apply all filters to create filtered_df
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["PRODUCT_TYPE"].isin(selected_products)]
filtered_df = filtered_df[filtered_df["LOCATION"].isin(selected_locations)]
filtered_df = filtered_df[filtered_df["TRANSPORTATION_MODES"].isin(selected_transports)]
filtered_df = filtered_df[filtered_df["INSPECTION_RESULTS"].isin(selected_inspections)]
filtered_df = filtered_df[
    (filtered_df["PRICE"] >= price_min) & (filtered_df["PRICE"] <= price_max)
]

# If no rows after filtering, show warning and stop
if filtered_df.empty:
    st.warning("No data after applying filters. Please relax the filter settings.")
    st.stop()


# -------------------------------
# 4. Show KPI summary
# -------------------------------
total_revenue = filtered_df["REVENUE_GENERATED"].sum()
total_costs = filtered_df["COSTS"].sum()
total_profit = total_revenue - total_costs
avg_defect_rate = filtered_df["DEFECT_RATES"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
# AI-enhanced: show delta vs the full (unfiltered) dataset to give context
base_total_revenue = df["REVENUE_GENERATED"].sum()
base_total_costs = df["COSTS"].sum()
base_total_profit = base_total_revenue - base_total_costs
base_avg_defect = df["DEFECT_RATES"].mean()

def _format_delta(curr, base):
    try:
        pct = (curr - base) / base
    except Exception:
        pct = 0
    return f"{pct:+.1%}"

kpi1.metric("Total Revenue", f"{total_revenue:,.2f}", delta=_format_delta(total_revenue, base_total_revenue))
kpi2.metric("Total Costs", f"{total_costs:,.2f}", delta=_format_delta(total_costs, base_total_costs))
kpi3.metric("Total Profit", f"{total_profit:,.2f}", delta=_format_delta(total_profit, base_total_profit))
kpi4.metric("Average Defect Rate", f"{avg_defect_rate * 100:.2f}%", delta=_format_delta(avg_defect_rate, base_avg_defect))

st.markdown("---")


# -------------------------------
# 5. Bar chart by PRODUCT_TYPE
# -------------------------------
st.subheader("Bar Chart by Product Type")

# AI-enhanced: allow sorting bars and show value labels
metric_option = st.selectbox(
    "Select metric for bar chart",
    options=["Total Revenue", "Total Costs", "Total Profit"],
)
sort_bars = st.checkbox("Sort bars by value (descending)", value=True)

grouped = filtered_df.groupby("PRODUCT_TYPE")

if metric_option == "Total Revenue":
    bar_data = grouped["REVENUE_GENERATED"].sum().reset_index(name="VALUE")
elif metric_option == "Total Costs":
    bar_data = grouped["COSTS"].sum().reset_index(name="VALUE")
else:  # Total Profit
    bar_data = (
        grouped["REVENUE_GENERATED"].sum()
        - grouped["COSTS"].sum()
    ).reset_index(name="VALUE")

if sort_bars:
    bar_data = bar_data.sort_values("VALUE", ascending=False)

bar_fig = px.bar(
    bar_data,
    x="PRODUCT_TYPE",
    y="VALUE",
    labels={"PRODUCT_TYPE": "Product Type", "VALUE": metric_option},
    text_auto=True,  # AI-enhanced: show values on bars
)
bar_fig.update_layout(margin=dict(t=30, b=30))
st.plotly_chart(bar_fig, use_container_width=True)


# -------------------------------
# 6. Scatter plot: costs vs revenue
# -------------------------------
st.subheader("Scatter Plot: Costs vs Revenue")

# AI-enhanced: provide jitter controls to reduce overplotting
scatter_df = filtered_df.copy()
scatter_df = scatter_df[scatter_df["NUMBER_OF_PRODUCTS_SOLD"] >= min_sold]

if scatter_df.empty:
    st.info("No data for scatter plot after applying minimum NUMBER_OF_PRODUCTS_SOLD.")
else:
    jitter_apply = st.checkbox("Apply jitter to points to reduce overplotting", value=False)
    jitter_frac = st.slider("Jitter fraction (fraction of axis range)", min_value=0.0, max_value=0.2, value=0.02, step=0.005)

    x_col = "COSTS"
    y_col = "REVENUE_GENERATED"

    # prepare hover columns (show original values even if jitter applied)
    hover_cols = ["SKU", "LOCATION"]
    hover_cols = [c for c in hover_cols if c in scatter_df.columns]

    # Apply jitter reproducibly
    if jitter_apply and not scatter_df[[x_col, y_col]].dropna().empty:
        rng = np.random.default_rng(42)
        x_range = scatter_df[x_col].max() - scatter_df[x_col].min()
        y_range = scatter_df[y_col].max() - scatter_df[y_col].min()
        x_noise = rng.normal(0, jitter_frac * (x_range if x_range != 0 else 1), size=len(scatter_df))
        y_noise = rng.normal(0, jitter_frac * (y_range if y_range != 0 else 1), size=len(scatter_df))
        scatter_df["_x_plot"] = scatter_df[x_col] + x_noise
        scatter_df["_y_plot"] = scatter_df[y_col] + y_noise
    else:
        scatter_df["_x_plot"] = scatter_df[x_col]
        scatter_df["_y_plot"] = scatter_df[y_col]

    scatter_fig = px.scatter(
        scatter_df,
        x="_x_plot",
        y="_y_plot",
        color="PRODUCT_TYPE",
        size="NUMBER_OF_PRODUCTS_SOLD",
        hover_data=hover_cols + [x_col, y_col],  # AI-enhanced: include original numeric values in hover
        labels={"_x_plot": "Costs (possibly jittered)", "_y_plot": "Revenue (possibly jittered)"},
    )
    scatter_fig.update_layout(margin=dict(t=30, b=30))
    st.plotly_chart(scatter_fig, use_container_width=True)


# -------------------------------
# 7. Heatmap of defect rates
# -------------------------------
st.subheader("Heatmap of Average Defect Rates")

# AI-enhanced: allow choosing colormap and annotate cells (text_auto)
heat = (
    filtered_df.groupby(["LOCATION", "TRANSPORTATION_MODES"])["DEFECT_RATES"]
    .mean()
    .reset_index()
)
heat_pivot = heat.pivot(
    index="LOCATION",
    columns="TRANSPORTATION_MODES",
    values="DEFECT_RATES",
)

colormap = st.selectbox("Heatmap color scale", options=["Viridis", "Plasma", "Cividis", "Reds"], index=0)
heat_fig = px.imshow(
    heat_pivot,
    labels=dict(x="Transportation Modes", y="Location", color="Avg Defect Rate"),
    color_continuous_scale=colormap.lower(),
    text_auto=True,  # AI-enhanced: annotate each cell with the defect rate
)
heat_fig.update_layout(margin=dict(t=30, b=30))
st.plotly_chart(heat_fig, use_container_width=True)


# -------------------------------
# 8. Donut chart
# -------------------------------
st.subheader("Donut Chart")

# AI-enhanced: group small slices into 'Other' via a user-controlled threshold
small_pct = st.slider("Group categories below this percent into 'Other'", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

donut_series = filtered_df[donut_category].fillna("(missing)")
donut_counts = (
    donut_series
    .value_counts()
    .reset_index(name="COUNT")
    .rename(columns={"index": donut_category})
)

total = donut_counts["COUNT"].sum() if not donut_counts.empty else 0
if total > 0 and small_pct > 0:
    donut_counts["PCT"] = donut_counts["COUNT"] / total * 100
    large = donut_counts[donut_counts["PCT"] >= small_pct].copy()
    small = donut_counts[donut_counts["PCT"] < small_pct]
    if not small.empty:
        other_row = pd.DataFrame([{donut_category: "Other", "COUNT": small["COUNT"].sum()}])
        donut_counts = pd.concat([large[[donut_category, "COUNT"]], other_row], ignore_index=True)
    else:
        donut_counts = large[[donut_category, "COUNT"]]

donut_fig = px.pie(
    donut_counts,
    names=donut_category,
    values="COUNT",
    hole=0.5,
)
donut_fig.update_traces(textposition="inside", textinfo="percent+label")
donut_fig.update_layout(margin=dict(t=30, b=30))
st.plotly_chart(donut_fig, use_container_width=True)

