"""
Supply Chain Dashboard App

This Streamlit app builds an interactive, multi-visual business dashboard
based on the cleaned supply_chain_data_cleaned.csv dataset (Project 1 output).

Features:
- Global filters (product type, location, transportation mode, inspection result)
- 4+ different visualization types (bar, scatter, heatmap, donut)
- All visualizations are connected via the same filtered DataFrame
- Clear separation between:
  - AI-generated enhancements (to be filled using GitHub Copilot)
  - Student-generated enhancements (written manually by the team)
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# =========================
# 1. DATA LOADING
# =========================

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load cleaned supply chain data from CSV.

    Parameters
    ----------
    csv_path : str
        Path to the cleaned CSV file (Project 1 output).

    Returns
    -------
    pd.DataFrame
        Cleaned supply chain dataset.
    """
    df = pd.read_csv(csv_path)
    return df


# =========================
# 2. GLOBAL FILTER FUNCTION
# =========================

def filter_data(
    df: pd.DataFrame,
    product_types: list,
    locations: list,
    transport_modes: list,
    inspection_results: list,
) -> pd.DataFrame:
    """
    Apply global filters and return filtered DataFrame.
    All visualizations will be based on this filtered_df,
    so that they are automatically linked and synchronized.

    Parameters
    ----------
    df : pd.DataFrame
        Original full dataset.
    product_types : list
        Selected product types from sidebar.
    locations : list
        Selected locations from sidebar.
    transport_modes : list
        Selected transportation modes from sidebar.
    inspection_results : list
        Selected inspection results from sidebar.

    Returns
    -------
    pd.DataFrame
        Filtered dataset.
    """
    filtered = df.copy()

    if product_types:
        filtered = filtered[filtered["PRODUCT_TYPE"].isin(product_types)]

    if locations:
        filtered = filtered[filtered["LOCATION"].isin(locations)]

    if transport_modes:
        filtered = filtered[filtered["TRANSPORTATION_MODES"].isin(transport_modes)]

    if inspection_results:
        filtered = filtered[filtered["INSPECTION_RESULTS"].isin(inspection_results)]

    return filtered


# =========================
# 3. VISUALIZATION 1
#    Bar Chart: Revenue by Product Type
# =========================

def viz1_base_revenue_by_product_type(df: pd.DataFrame):
    """
    Base bar chart for total revenue by product type.

    Visualization Type: Bar Chart
    Business Question:
        Which product types generate the highest revenue?
    """
    revenue_by_type = (
        df.groupby("PRODUCT_TYPE", as_index=False)["REVENUE_GENERATED"].sum()
    )

    fig = px.bar(
        revenue_by_type,
        x="PRODUCT_TYPE",
        y="REVENUE_GENERATED",
        title="Total Revenue by Product Type",
        labels={"REVENUE_GENERATED": "Total Revenue", "PRODUCT_TYPE": "Product Type"},
    )

    return fig


def viz1_student_enhancement(fig, df: pd.DataFrame):
    """
    STUDENT-GENERATED ENHANCEMENT for Visualization 1.

    Interactive feature:
    - Let the user choose between:
        * Absolute revenue
        * Percentage of total revenue
    - Optionally sort bars by value.

    This satisfies the requirement:
    "At least two visualizations must include interactive elements
    in the student-generated features."
    """
    st.subheader("Visualization 1: Revenue by Product Type")

    # Student interactive controls
    display_mode = st.radio(
        "Display mode (Student enhancement):",
        ["Absolute revenue", "Percentage of total revenue"],
        key="viz1_display_mode",
    )

    sort_option = st.checkbox(
        "Sort by value (descending)", key="viz1_sort_desc"
    )

    # Recompute aggregation according to mode
    revenue_by_type = (
        df.groupby("PRODUCT_TYPE", as_index=False)["REVENUE_GENERATED"].sum()
    )

    if display_mode == "Percentage of total revenue":
        total = revenue_by_type["REVENUE_GENERATED"].sum()
        revenue_by_type["REVENUE_PERCENT"] = (
            revenue_by_type["REVENUE_GENERATED"] / total * 100
        )
        y_col = "REVENUE_PERCENT"
        y_label = "Revenue (%)"
        title_suffix = " (Percentage)"
    else:
        y_col = "REVENUE_GENERATED"
        y_label = "Total Revenue"
        title_suffix = " (Absolute)"

    if sort_option:
        revenue_by_type = revenue_by_type.sort_values(by=y_col, ascending=False)

    fig = px.bar(
        revenue_by_type,
        x="PRODUCT_TYPE",
        y=y_col,
        title=f"Total Revenue by Product Type{title_suffix}",
        labels={y_col: y_label, "PRODUCT_TYPE": "Product Type"},
    )

    st.plotly_chart(fig, use_container_width=True)

    return fig


def viz1_ai_enhancement_placeholder(fig, df: pd.DataFrame):
    """
    AI-GENERATED ENHANCEMENT PLACEHOLDER for Visualization 1.

    IMPORTANT:
    - In your real project, ask GitHub Copilot inside VS Code, e.g.:

        # Prompt example (do NOT include in final code):
        # "Copilot, add an automatic highlight for the top 1 product type
        #  and display the exact revenue in the hover tooltip."

    - Replace this placeholder with the code suggested by Copilot.
    - Clearly label the final block as "AI-generated" in your report/notebook.

    Here we just keep a simple version to show where AI code should go.
    """
    st.markdown("**(AI enhancement for Viz 1 – to be implemented using Copilot.)**")
    # Example minimal display to keep app running:
    st.plotly_chart(fig, use_container_width=True)


# =========================
# 4. VISUALIZATION 2
#    Heatmap: Delivery Efficiency by Location and Transport Mode
# =========================

def viz2_base_heatmap_efficiency(df: pd.DataFrame):
    """
    Base heatmap of average delivery efficiency by (LOCATION, TRANSPORTATION_MODES).

    Visualization Type: Heatmap
    Business Question:
        Which combinations of location and transportation mode are most efficient?
    """
    if df.empty:
        return None

    pivot_df = (
        df.groupby(["LOCATION", "TRANSPORTATION_MODES"], as_index=False)[
            "DELIVERY_EFFICIENCY"
        ]
        .mean()
    )

    fig = px.density_heatmap(
        pivot_df,
        x="TRANSPORTATION_MODES",
        y="LOCATION",
        z="DELIVERY_EFFICIENCY",
        color_continuous_scale="Viridis",
        title="Average Delivery Efficiency by Location and Transportation Mode",
        labels={
            "TRANSPORTATION_MODES": "Transportation Mode",
            "LOCATION": "Location",
            "DELIVERY_EFFICIENCY": "Delivery Efficiency",
        },
    )

    return fig


def viz2_student_enhancement(fig, df: pd.DataFrame):
    """
    STUDENT-GENERATED ENHANCEMENT for Visualization 2.

    Interactive feature:
    - Add a slider to filter by minimum delivery efficiency.
    - Only show cells where DELIVERY_EFFICIENCY >= threshold.
    """
    st.subheader("Visualization 2: Delivery Efficiency Heatmap")

    if df.empty:
        st.warning("No data available under current filters.")
        return None

    min_eff = float(df["DELIVERY_EFFICIENCY"].min())
    max_eff = float(df["DELIVERY_EFFICIENCY"].max())

    threshold = st.slider(
        "Minimum delivery efficiency to display (Student enhancement):",
        min_value=min_eff,
        max_value=max_eff,
        value=min_eff,
        step=(max_eff - min_eff) / 20 if max_eff > min_eff else 1.0,
        key="viz2_threshold",
    )

    pivot_df = (
        df.groupby(["LOCATION", "TRANSPORTATION_MODES"], as_index=False)[
            "DELIVERY_EFFICIENCY"
        ]
        .mean()
    )

    pivot_df = pivot_df[pivot_df["DELIVERY_EFFICIENCY"] >= threshold]

    if pivot_df.empty:
        st.warning("No combinations meet the selected efficiency threshold.")
        return None

    fig = px.density_heatmap(
        pivot_df,
        x="TRANSPORTATION_MODES",
        y="LOCATION",
        z="DELIVERY_EFFICIENCY",
        color_continuous_scale="Viridis",
        title="Average Delivery Efficiency by Location and Transportation Mode",
        labels={
            "TRANSPORTATION_MODES": "Transportation Mode",
            "LOCATION": "Location",
            "DELIVERY_EFFICIENCY": "Delivery Efficiency",
        },
    )

    st.plotly_chart(fig, use_container_width=True)
    return fig


def viz2_ai_enhancement_placeholder(fig, df: pd.DataFrame):
    """
    AI-GENERATED ENHANCEMENT PLACEHOLDER for Visualization 2.

    Use Copilot later to add features such as:
    - Dynamic annotations for best/worst cells.
    - A toggle to switch between DELIVERY_EFFICIENCY and DEFECT_RATES.
    """
    st.markdown("**(AI enhancement for Viz 2 – to be implemented using Copilot.)**")
    # Keep a simple version:
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


# =========================
# 5. VISUALIZATION 3
#    Scatter: Price vs Units Sold, colored by Product Type
# =========================

def viz3_base_scatter_price_vs_sold(df: pd.DataFrame):
    """
    Base scatter plot: Price vs Number of Products Sold, colored by product type.

    Visualization Type: Scatter Plot
    Business Question:
        How does price relate to sales volume across product types?
    """
    if df.empty:
        return None

    fig = px.scatter(
        df,
        x="PRICE",
        y="NUMBER_OF_PRODUCTS_SOLD",
        color="PRODUCT_TYPE",
        hover_data=["SKU"],
        title="Price vs Units Sold by Product Type",
        labels={
            "PRICE": "Price",
            "NUMBER_OF_PRODUCTS_SOLD": "Number of Products Sold",
            "PRODUCT_TYPE": "Product Type",
        },
    )
    return fig


def viz3_student_enhancement(fig, df: pd.DataFrame):
    """
    STUDENT-GENERATED ENHANCEMENT for Visualization 3.

    Interactive feature:
    - Allow user to filter by a price range using a range slider.
    """
    st.subheader("Visualization 3: Price vs Units Sold")

    if df.empty:
        st.warning("No data available under current filters.")
        return None

    min_price = float(df["PRICE"].min())
    max_price = float(df["PRICE"].max())

    price_range = st.slider(
        "Select price range (Student enhancement):",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        key="viz3_price_range",
    )

    mask = df["PRICE"].between(price_range[0], price_range[1])
    filtered_df = df[mask]

    if filtered_df.empty:
        st.warning("No products within the selected price range.")
        return None

    fig = px.scatter(
        filtered_df,
        x="PRICE",
        y="NUMBER_OF_PRODUCTS_SOLD",
        color="PRODUCT_TYPE",
        hover_data=["SKU"],
        title="Price vs Units Sold by Product Type (Filtered by Price Range)",
        labels={
            "PRICE": "Price",
            "NUMBER_OF_PRODUCTS_SOLD": "Number of Products Sold",
            "PRODUCT_TYPE": "Product Type",
        },
    )

    st.plotly_chart(fig, use_container_width=True)
    return fig


def viz3_ai_enhancement_placeholder(fig, df: pd.DataFrame):
    """
    AI-GENERATED ENHANCEMENT PLACEHOLDER for Visualization 3.

    Example ideas for Copilot:
    - Add a regression trend line.
    - Calculate and display correlation between PRICE and NUMBER_OF_PRODUCTS_SOLD.
    """
    st.markdown("**(AI enhancement for Viz 3 – to be implemented using Copilot.)**")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


# =========================
# 6. VISUALIZATION 4
#    Donut Chart: Transportation Cost by Mode
# =========================

def viz4_base_donut_cost_by_transport_mode(df: pd.DataFrame):
    """
    Base donut chart of total transportation costs by transportation mode.

    Visualization Type: Donut (Pie) Chart
    Business Question:
        How are transportation costs distributed across different modes?
    """
    if df.empty:
        return None

    cost_by_mode = (
        df.groupby("TRANSPORTATION_MODES", as_index=False)["COSTS"].sum()
    )

    fig = px.pie(
        cost_by_mode,
        names="TRANSPORTATION_MODES",
        values="COSTS",
        hole=0.4,
        title="Total Transportation Costs by Mode",
    )

    return fig


def viz4_student_enhancement(fig, df: pd.DataFrame):
    """
    STUDENT-GENERATED ENHANCEMENT for Visualization 4.

    Feature:
    - Simple toggle to switch between viewing COSTS and SHIPPING_COSTS.
    (This one is less interactive than sliders, but still demonstrates a student choice
    of business metric.)
    """
    st.subheader("Visualization 4: Transportation Cost Breakdown")

    if df.empty:
        st.warning("No data available under current filters.")
        return None

    metric = st.radio(
        "Select cost metric (Student enhancement):",
        ["COSTS", "SHIPPING_COSTS"],
        key="viz4_metric",
    )

    cost_by_mode = (
        df.groupby("TRANSPORTATION_MODES", as_index=False)[metric].sum()
    )

    fig = px.pie(
        cost_by_mode,
        names="TRANSPORTATION_MODES",
        values=metric,
        hole=0.4,
        title=f"Total {metric} by Transportation Mode",
    )

    st.plotly_chart(fig, use_container_width=True)
    return fig


def viz4_ai_enhancement_placeholder(fig, df: pd.DataFrame):
    """
    AI-GENERATED ENHANCEMENT PLACEHOLDER for Visualization 4.

    Possible Copilot ideas:
    - Automatically highlight the most expensive mode.
    - Add an annotation with the percentage of total cost.
    """
    st.markdown("**(AI enhancement for Viz 4 – to be implemented using Copilot.)**")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


# =========================
# 7. MAIN APP LAYOUT
# =========================

def main():
    st.set_page_config(
        page_title="Supply Chain Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Supply Chain Performance Dashboard")
    st.caption(
        "Project 2 – Interactive, multi-visual dashboard based on cleaned supply chain data."
    )

    # --- Load data ---
    # You can adjust this path depending on where you put the file
    csv_path = Path("supply_chain_data_cleaned.csv")
    df = load_data(str(csv_path))

    # --- Sidebar filters (global) ---
    st.sidebar.header("Global Filters")

    product_types = st.sidebar.multiselect(
        "Product Type",
        options=sorted(df["PRODUCT_TYPE"].dropna().unique()),
        default=sorted(df["PRODUCT_TYPE"].dropna().unique()),
    )

    locations = st.sidebar.multiselect(
        "Location",
        options=sorted(df["LOCATION"].dropna().unique()),
        default=sorted(df["LOCATION"].dropna().unique()),
    )

    transport_modes = st.sidebar.multiselect(
        "Transportation Modes",
        options=sorted(df["TRANSPORTATION_MODES"].dropna().unique()),
        default=sorted(df["TRANSPORTATION_MODES"].dropna().unique()),
    )

    inspection_results = st.sidebar.multiselect(
        "Inspection Results",
        options=sorted(df["INSPECTION_RESULTS"].dropna().unique()),
        default=sorted(df["INSPECTION_RESULTS"].dropna().unique()),
    )

    # Apply global filters
    filtered_df = filter_data(
        df,
        product_types=product_types,
        locations=locations,
        transport_modes=transport_modes,
        inspection_results=inspection_results,
    )

    if filtered_df.empty:
        st.warning("No data available with the current filter selections.")
        st.stop()

    # ======================
    # Layout: 2x2 grid of visualizations
    # ======================

    col1, col2 = st.columns(2)

    with col1:
        # Visualization 1 – student version
        viz1_student_enhancement(
            fig=None,  # base fig not required; function recomputes internally
            df=filtered_df,
        )

    with col2:
        # Visualization 2 – student version
        viz2_student_enhancement(
            fig=None,
            df=filtered_df,
        )

    col3, col4 = st.columns(2)

    with col3:
        # Visualization 3 – student version
        viz3_student_enhancement(
            fig=None,
            df=filtered_df,
        )

    with col4:
        # Visualization 4 – student version
        viz4_student_enhancement(
            fig=None,
            df=filtered_df,
        )

    # You can optionally add a separate section below for the AI-enhanced versions
    # when you have Copilot code ready, e.g.:
    st.markdown("---")
    st.header("AI-generated Enhancements (Placeholders)")

    # Example: call base figs and then pass to AI placeholders
    base_fig1 = viz1_base_revenue_by_product_type(filtered_df)
    viz1_ai_enhancement_placeholder(base_fig1, filtered_df)

    base_fig2 = viz2_base_heatmap_efficiency(filtered_df)
    viz2_ai_enhancement_placeholder(base_fig2, filtered_df)

    base_fig3 = viz3_base_scatter_price_vs_sold(filtered_df)
    viz3_ai_enhancement_placeholder(base_fig3, filtered_df)

    base_fig4 = viz4_base_donut_cost_by_transport_mode(filtered_df)
    viz4_ai_enhancement_placeholder(base_fig4, filtered_df)


if __name__ == "__main__":
    main()
