import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------
# 1. Start dashboard app
# -----------------------
st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

# Fun title with emoji
st.title("📦 Supply Chain Performance Dashboard")
st.caption(
    "Explore key metrics, product performance, and quality patterns across the supply chain."
)

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
st.sidebar.header("🔍 Global Filters")

# Sidebar mini-intro
st.sidebar.info(
    "Use these filters to slice the entire dashboard.\n\n"
    "All KPIs and visualizations are linked to the same filtered data."
)

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

# Minimum units sold (used in scatter plot)
min_sold = st.sidebar.number_input(
    "Minimum NUMBER_OF_PRODUCTS_SOLD (for scatter plot)",
    min_value=0,
    value=0,
)

# Default donut category (can be overridden in the Donut tab)
donut_category_initial = st.sidebar.selectbox(
    "Default donut category",
    options=["INSPECTION_RESULTS", "CUSTOMER_DEMOGRAPHICS"],
)


# Apply all filters
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["PRODUCT_TYPE"].isin(selected_products)]
filtered_df = filtered_df[filtered_df["LOCATION"].isin(selected_locations)]
filtered_df = filtered_df[filtered_df["TRANSPORTATION_MODES"].isin(selected_transports)]
filtered_df = filtered_df[filtered_df["INSPECTION_RESULTS"].isin(selected_inspections)]
filtered_df = filtered_df[
    (filtered_df["PRICE"] >= price_min) & (filtered_df["PRICE"] <= price_max)
]

# If no rows remain, stop execution
if filtered_df.empty:
    st.warning("No data after applying filters. Please relax the filter settings.")
    st.stop()

# Add a small status line under the title
st.markdown(
    f"✅ Showing **{len(filtered_df):,}** records after applying current filters."
)

# -------------------------------
# 4. Tabs layout
# -------------------------------
overview_tab, product_tab, performance_tab, quality_tab = st.tabs(
    ["📊 Overview", "🧷 Product View", "⚙️ Performance", "✅ Quality & Segments"]
)

# ===========================
# TAB 1: Overview (KPI + Bar)
# ===========================
with overview_tab:
    st.subheader("Key Performance Indicators")

    total_revenue = filtered_df["REVENUE_GENERATED"].sum()
    total_costs = filtered_df["COSTS"].sum()
    total_profit = total_revenue - total_costs
    avg_defect_rate = filtered_df["DEFECT_RATES"].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # Student-enhanced: clearer formatting with currency and percentages
    kpi1.metric("Total Revenue (USD) 💰", f"${total_revenue:,.0f}")
    kpi2.metric("Total Costs (USD) 💸", f"${total_costs:,.0f}")
    kpi3.metric("Total Profit (USD) 📈", f"${total_profit:,.0f}")
    kpi4.metric("Average Defect Rate ⚠️", f"{avg_defect_rate * 100:.2f}%")

    st.caption(
        "These KPIs are calculated from the filtered dataset and update when you change the filters on the left."
    )

    st.markdown("---")

    # Bar chart by product type
    st.subheader("Product Performance Overview")

    metric_option = st.selectbox(
        "Select metric for bar chart",
        options=["Total Revenue", "Total Costs", "Total Profit"],
        key="bar_metric",
    )

    grouped = filtered_df.groupby("PRODUCT_TYPE")

    if metric_option == "Total Revenue":
        bar_data = grouped["REVENUE_GENERATED"].sum().reset_index(name="VALUE")
    elif metric_option == "Total Costs":
        bar_data = grouped["COSTS"].sum().reset_index(name="VALUE")
    else:
        bar_data = (
            grouped["REVENUE_GENERATED"].sum()
            - grouped["COSTS"].sum()
        ).reset_index(name="VALUE")

    bar_fig = px.bar(
        bar_data,
        x="PRODUCT_TYPE",
        y="VALUE",
        color="PRODUCT_TYPE",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"VALUE": metric_option},
        title=f"{metric_option} by Product Type (Student-enhanced)",
    )

    bar_fig.update_traces(
        text=bar_data["VALUE"].round(0),
        textposition="outside",
    )

    bar_fig.update_layout(
        xaxis_tickangle=-30,
        margin=dict(t=60, b=80),
        showlegend=False,
    )

    st.plotly_chart(bar_fig, use_container_width=True)


# ===========================
# TAB 2: Product View (Bar only, optional extras later)
# ===========================
with product_tab:
    st.subheader("Product Mix and Revenue Focus")

    st.write(
        "This view helps us compare how different product types contribute to overall performance."
    )

    # Reuse same bar chart for now; you can later add another chart here if needed
    product_bar_fig = px.bar(
        bar_data,
        x="PRODUCT_TYPE",
        y="VALUE",
        color="PRODUCT_TYPE",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"VALUE": metric_option},
        title=f"{metric_option} by Product Type (same as Overview, shown separately here)",
    )
    product_bar_fig.update_traces(
        text=bar_data["VALUE"].round(0),
        textposition="outside",
    )
    product_bar_fig.update_layout(
        xaxis_tickangle=-30,
        margin=dict(t=60, b=80),
        showlegend=False,
    )

    st.plotly_chart(product_bar_fig, use_container_width=True)


# ===========================
# TAB 3: Performance (Scatter)
# ===========================
with performance_tab:
    st.subheader("Cost vs Revenue by Product / Location")

    scatter_df = filtered_df.copy()
    scatter_df = scatter_df[scatter_df["NUMBER_OF_PRODUCTS_SOLD"] >= min_sold]

    if scatter_df.empty:
        st.info(
            "No data for the scatter plot under the current filters and minimum units sold."
        )
    else:
        color_choice = st.selectbox(
            "Color points by",
            options=["PRODUCT_TYPE", "LOCATION"],
            index=0,
            key="scatter_color",
        )

        hover_cols = [
            c for c in ["SKU", "LOCATION", "PRODUCT_TYPE"] if c in scatter_df.columns
        ]

        scatter_fig = px.scatter(
            scatter_df,
            x="COSTS",
            y="REVENUE_GENERATED",
            color=color_choice,
            size="NUMBER_OF_PRODUCTS_SOLD",
            hover_data=hover_cols,
            labels={"COSTS": "Costs (USD)", "REVENUE_GENERATED": "Revenue (USD)"},
            title="Costs vs Revenue with Number Sold as Point Size",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )

        scatter_fig.update_layout(
            xaxis_title="Costs (USD)",
            yaxis_title="Revenue (USD)",
            margin=dict(t=80, b=40),
        )

        st.plotly_chart(scatter_fig, use_container_width=True)

        st.caption(
            "Each bubble shows one record in the filtered dataset. "
            "Higher points indicate higher revenue; larger bubbles represent more units sold."
        )


# ===========================
# TAB 4: Quality & Segments (Heatmap + Donut)
# ===========================
with quality_tab:
    st.subheader("Defect Rates by Location and Transport Mode")

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

    heat_fig = px.imshow(
        heat_pivot,
        labels=dict(
            x="Transportation Modes",
            y="Location",
            color="Avg Defect Rate",
        ),
        title="Average Defect Rates by Location and Transport (Student-enhanced)",
        color_continuous_scale="Reds",
        aspect="auto",
    )

    heat_fig.update_traces(
        text=(heat_pivot.values * 100).round(1),
        texttemplate="%{text}%",
        textfont=dict(size=10),
    )

    heat_fig.update_layout(
        margin=dict(t=80, b=40),
    )

    st.plotly_chart(heat_fig, use_container_width=True)

    st.caption("Darker cells indicate higher defect rates. Values are displayed as percentages.")

    st.markdown("---")

    st.subheader("Category Distribution (Donut Chart)")

    donut_category = st.selectbox(
        "Select category for donut chart",
        options=["INSPECTION_RESULTS", "CUSTOMER_DEMOGRAPHICS"],
        index=["INSPECTION_RESULTS", "CUSTOMER_DEMOGRAPHICS"].index(
            donut_category_initial
        )
        if donut_category_initial in ["INSPECTION_RESULTS", "CUSTOMER_DEMOGRAPHICS"]
        else 0,
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
        title=f"Distribution of {donut_category}",
        color=donut_category,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    donut_fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        hovertemplate=f"{donut_category}: %{{label}}<br>Count: %{{value}}<extra></extra>",
    )

    donut_fig.update_layout(
        margin=dict(t=80, b=40),
    )

    st.plotly_chart(donut_fig, use_container_width=True)

    st.caption(
        "Use this donut chart to understand how records are distributed across "
        "inspection outcomes or customer segments."
    )

