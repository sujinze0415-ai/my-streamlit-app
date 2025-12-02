import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")


@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
	"""Load CSV into DataFrame and do initial cleaning.

	Assumptions: The CSV is named `supply_chain_data_cleaned.csv` and sits next to this file
	or in the working directory. Numeric columns may be named in a variety of ways; the
	loader will try to coerce commonly named numeric columns to numeric types.
	"""
	df = pd.read_csv(csv_path)

	# Drop exact duplicate rows
	df = df.drop_duplicates()

	# Normalize column names to uppercase stripped for mapping convenience
	cols_upper = {c: c for c in df.columns}

	def find_col(substrings):
		"""Return first matching column name by checking substrings (case-insensitive)."""
		s = [c for c in df.columns]
		for substr in substrings:
			for c in s:
				if substr.lower() in c.lower():
					return c
		return None

	# Attempt to find key numeric columns
	price_col = find_col(["price", "unit_price", "unit price"]) 
	revenue_col = find_col(["revenue", "sales", "total_revenue", "total revenue"]) 
	costs_col = find_col(["cost", "costs", "expense", "total_cost"]) 
	defect_col = find_col(["defect_rate", "defect", "defect rate", "defects_pct"]) 
	qty_col = find_col(["number_of_products_sold", "units_sold", "quantity", "qty", "sold"]) 

	# Try to coerce found columns to numeric, if present
	for c in [price_col, revenue_col, costs_col, defect_col, qty_col]:
		if c is not None:
			df[c] = pd.to_numeric(df[c], errors="coerce")

	# If defect rate found but it's in fraction (0-1), allow average presentation as percent later
	return df


def main():
	# Locate CSV next to this script or in current working directory
	base = Path(__file__).resolve().parent
	csv_path = base / "supply_chain_data_cleaned.csv"
	if not csv_path.exists():
		# fallback to working directory
		csv_path = Path("supply_chain_data_cleaned.csv")

	if not csv_path.exists():
		st.error(f"Could not find supply_chain_data_cleaned.csv at {csv_path}. Put the file in the app folder.")
		return

	df = load_data(csv_path)

	st.title("Supply Chain Dashboard")

	# Identify commonly used categorical columns
	def find_col(df, substrings):
		for substr in substrings:
			for c in df.columns:
				if substr.lower() in c.lower():
					return c
		return None

	product_col = find_col(df, ["product_type", "product type", "product"]) or "PRODUCT_TYPE"
	location_col = find_col(df, ["location", "site", "warehouse"]) or "LOCATION"
	transport_col = find_col(df, ["transportation_modes", "transportation mode", "transport"]) or "TRANSPORTATION_MODES"
	inspect_col = find_col(df, ["inspection_results", "inspection result", "inspection"]) or "INSPECTION_RESULTS"
	customer_demo_col = find_col(df, ["customer_demographics", "customer demo", "demographics"]) or "CUSTOMER_DEMOGRAPHICS"

	# Provide sidebar filters
	st.sidebar.header("Filters")

	# Multiselect helpers that tolerate missing columns
	def multiselect_for(col, label):
		if col in df.columns:
			opts = sorted(df[col].dropna().unique().tolist())
			return st.sidebar.multiselect(label, options=opts, default=opts)
		else:
			return None

	selected_products = multiselect_for(product_col, "Product Type")
	selected_locations = multiselect_for(location_col, "Location")
	selected_transports = multiselect_for(transport_col, "Transportation Modes")
	selected_inspections = multiselect_for(inspect_col, "Inspection Results")

	# Price range slider
	# Try to find a price column
	price_candidates = [c for c in df.columns if "price" in c.lower()]
	price_col = price_candidates[0] if price_candidates else None
	if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
		min_p = float(df[price_col].min(skipna=True))
		max_p = float(df[price_col].max(skipna=True))
		price_min, price_max = st.sidebar.slider("Price range", min_value=min_p, max_value=max_p, value=(min_p, max_p))
	else:
		price_min, price_max = None, None

	# Bar chart metric selection
	metric_choice = st.sidebar.selectbox("Bar chart metric", ["Total Revenue", "Total Costs", "Total Profit"]) 

	# Scatter: min products sold
	qty_candidates = [c for c in df.columns if any(x in c.lower() for x in ["sold", "quantity", "units", "number_of_products"])]
	qty_col = qty_candidates[0] if qty_candidates else None
	min_sold = int(st.sidebar.number_input("Minimum number of products sold (for scatter)", min_value=0, value=0))

	# Donut category choice
	donut_options = [opt for opt in [inspect_col, customer_demo_col] if opt in df.columns]
	if not donut_options:
		donut_options = []
	donut_choice = st.sidebar.selectbox("Donut category", options=donut_options if donut_options else ["None"])

	# Apply filters to create filtered_df
	filtered = df.copy()
	# Apply product filter
	if selected_products is not None and product_col in filtered.columns:
		filtered = filtered[filtered[product_col].isin(selected_products)]
	if selected_locations is not None and location_col in filtered.columns:
		filtered = filtered[filtered[location_col].isin(selected_locations)]
	if selected_transports is not None and transport_col in filtered.columns:
		filtered = filtered[filtered[transport_col].isin(selected_transports)]
	if selected_inspections is not None and inspect_col in filtered.columns:
		filtered = filtered[filtered[inspect_col].isin(selected_inspections)]
	if price_col and price_min is not None:
		filtered = filtered[(filtered[price_col] >= price_min) & (filtered[price_col] <= price_max)]

	# If no rows after filtering, show warning and stop
	if filtered.empty:
		st.warning("No data after applying filters. Try expanding the filters.")
		st.stop()

	# KPIs
	# Try to find revenue and costs columns
	rev_candidates = [c for c in df.columns if "revenue" in c.lower() or "sales" in c.lower()]
	rev_col = rev_candidates[0] if rev_candidates else None
	cost_candidates = [c for c in df.columns if "cost" in c.lower()]
	cost_col = cost_candidates[0] if cost_candidates else None
	defect_candidates = [c for c in df.columns if "defect" in c.lower()]
	defect_col = defect_candidates[0] if defect_candidates else None

	total_rev = filtered[rev_col].sum() if rev_col in filtered.columns else np.nan
	total_costs = filtered[cost_col].sum() if cost_col in filtered.columns else np.nan
	total_profit = (total_rev - total_costs) if (not np.isnan(total_rev) and not np.isnan(total_costs)) else np.nan
	avg_defect = filtered[defect_col].mean() if defect_col in filtered.columns else np.nan

	k1, k2, k3, k4 = st.columns(4)
	k1.metric("Total Revenue", f"{total_rev:,.2f}" if not np.isnan(total_rev) else "N/A")
	k2.metric("Total Costs", f"{total_costs:,.2f}" if not np.isnan(total_costs) else "N/A")
	k3.metric("Total Profit", f"{total_profit:,.2f}" if not np.isnan(total_profit) else "N/A")
	# Format defect as percent if looks like 0-1
	if not np.isnan(avg_defect):
		if avg_defect <= 1:
			k4.metric("Avg Defect Rate", f"{avg_defect*100:.2f}%")
		else:
			k4.metric("Avg Defect Rate", f"{avg_defect:.2f}")
	else:
		k4.metric("Avg Defect Rate", "N/A")

	st.markdown("---")

	# Layout: left column charts and right column heatmap/donut
	left, right = st.columns((2, 1))

	# Bar chart by PRODUCT_TYPE
	with left:
		st.subheader("Bar chart by Product Type")
		if product_col in filtered.columns:
			grp = filtered.groupby(product_col)
			if metric_choice == "Total Revenue":
				series = grp[rev_col].sum() if rev_col in filtered.columns else pd.Series([])
				y_label = "Revenue"
			elif metric_choice == "Total Costs":
				series = grp[cost_col].sum() if cost_col in filtered.columns else pd.Series([])
				y_label = "Costs"
			else:
				# profit
				if rev_col in filtered.columns and cost_col in filtered.columns:
					series = grp[rev_col].sum() - grp[cost_col].sum()
				else:
					series = pd.Series([])
				y_label = "Profit"

			bar_df = series.reset_index()
			bar_df.columns = [product_col, "value"]
			bar_df = bar_df.sort_values("value", ascending=False)
			fig_bar = px.bar(bar_df, x=product_col, y="value", labels={"value": y_label, product_col: "Product Type"})
			st.plotly_chart(fig_bar, use_container_width=True)
		else:
			st.info("No product type column found for bar chart.")

		st.subheader("Costs vs Revenue (scatter)")
		if (cost_col in filtered.columns) and (rev_col in filtered.columns):
			scatter_df = filtered.copy()
			if qty_col and qty_col in scatter_df.columns:
				scatter_df = scatter_df[scatter_df[qty_col] >= min_sold]

			if scatter_df.empty:
				st.info("No data for scatter after applying minimum products sold filter.")
			else:
				size = scatter_df[qty_col] if (qty_col and qty_col in scatter_df.columns) else None
				fig_scatter = px.scatter(
					scatter_df,
					x=cost_col,
					y=rev_col,
					color=product_col if product_col in scatter_df.columns else None,
					size=size,
					hover_data=[product_col, location_col] if product_col in scatter_df.columns and location_col in scatter_df.columns else None,
					labels={cost_col: "Costs", rev_col: "Revenue"},
				)
				st.plotly_chart(fig_scatter, use_container_width=True)
		else:
			st.info("Revenue or Costs column not found for scatter plot.")

	# Right column: heatmap and donut
	with right:
		st.subheader("Heatmap: Avg Defect Rate")
		if (location_col in filtered.columns) and (transport_col in filtered.columns) and (defect_col in filtered.columns):
			heat = filtered.groupby([location_col, transport_col])[defect_col].mean().reset_index()
			pivot = heat.pivot(index=location_col, columns=transport_col, values=defect_col)
			# Use plotly heatmap
			fig_heat = go.Figure(data=go.Heatmap(
				z=pivot.values,
				x=pivot.columns.astype(str),
				y=pivot.index.astype(str),
				colorscale="Viridis",
				colorbar=dict(title="Avg Defect")
			))
			fig_heat.update_layout(xaxis_title=transport_col, yaxis_title=location_col, height=400)
			st.plotly_chart(fig_heat, use_container_width=True)
		else:
			st.info("Require LOCATION, TRANSPORTATION_MODES and a defect rate column for heatmap.")

		st.subheader("Donut chart")
		if donut_choice and donut_choice in filtered.columns:
			donut_df = filtered[donut_choice].value_counts().reset_index()
			donut_df.columns = [donut_choice, "count"]
			fig_donut = go.Figure(data=[go.Pie(labels=donut_df[donut_choice], values=donut_df["count"], hole=0.5)])
			fig_donut.update_traces(textinfo='percent+label')
			st.plotly_chart(fig_donut, use_container_width=True)
		else:
			st.info("No category selected or column missing for donut chart.")


if __name__ == "__main__":
	main()