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

	# Donut category choice
	donut_options = [opt for opt in [inspect_col, customer_demo_col] if opt in df.columns]
	if not donut_options:
		donut_options = []
	donut_choice = st.sidebar.selectbox("Donut category", options=donut_options if donut_options else ["None"])

	# Quantity/units column detection (used for scatter point size)
	qty_candidates = [c for c in df.columns if any(x in c.lower() for x in ["sold", "quantity", "units", "number_of_products"])]
	qty_col = qty_candidates[0] if qty_candidates else None

	# Helper to apply a set of filter selections to a dataframe
	def apply_filters(df_to_filter, prod_sel, loc_sel, trans_sel, insp_sel, price_range_tuple):
		d = df_to_filter.copy()
		if prod_sel is not None and product_col in d.columns:
			d = d[d[product_col].isin(prod_sel)]
		if loc_sel is not None and location_col in d.columns:
			d = d[d[location_col].isin(loc_sel)]
		if trans_sel is not None and transport_col in d.columns:
			d = d[d[transport_col].isin(trans_sel)]
		if insp_sel is not None and inspect_col in d.columns:
			d = d[d[inspect_col].isin(insp_sel)]
		if price_col and price_range_tuple[0] is not None:
			pmin, pmax = price_range_tuple
			d = d[(d[price_col] >= pmin) & (d[price_col] <= pmax)]
		return d

	# Per-chart filters: allow each visual to opt into using the global filters or a custom set
	st.sidebar.markdown("---")
	st.sidebar.header("Per-chart filters")

	# Local multiselect helper to render inside expanders/containers (not sidebar)
	def local_multiselect_for(col, label, container, default_all=True, key_suffix=""):
		if col in df.columns:
			opts = sorted(df[col].dropna().unique().tolist())
			default = opts if default_all else None
			return container.multiselect(label, options=opts, default=default, key=f"{label}_{key_suffix}")
		else:
			return None

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

	# Inline KPI filters next to KPI cards
	exp_kpi = st.expander("KPIs filters (local)", expanded=False)
	use_global_kpi = exp_kpi.checkbox("Use global filters", value=True, key="use_global_kpi")
	prod_sel_kpi = loc_sel_kpi = trans_sel_kpi = insp_sel_kpi = None
	price_range_kpi = (price_min, price_max)
	if not use_global_kpi:
		prod_sel_kpi = local_multiselect_for(product_col, "KPIs: Product Type", exp_kpi, key_suffix="kpi_prod")
		loc_sel_kpi = local_multiselect_for(location_col, "KPIs: Location", exp_kpi, key_suffix="kpi_loc")
		trans_sel_kpi = local_multiselect_for(transport_col, "KPIs: Transportation Modes", exp_kpi, key_suffix="kpi_trans")
		insp_sel_kpi = local_multiselect_for(inspect_col, "KPIs: Inspection Results", exp_kpi, key_suffix="kpi_insp")
		if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
			pmin = float(df[price_col].min(skipna=True))
			pmax = float(df[price_col].max(skipna=True))
			price_range_kpi = exp_kpi.slider("KPIs: Price range", min_value=pmin, max_value=pmax, value=(pmin, pmax), key="price_kpi")

	kpi_df = filtered if use_global_kpi else apply_filters(df, prod_sel_kpi, loc_sel_kpi, trans_sel_kpi, insp_sel_kpi, price_range_kpi)

	total_rev = kpi_df[rev_col].sum() if rev_col in kpi_df.columns else np.nan
	total_costs = kpi_df[cost_col].sum() if cost_col in kpi_df.columns else np.nan
	total_profit = (total_rev - total_costs) if (not np.isnan(total_rev) and not np.isnan(total_costs)) else np.nan
	avg_defect = kpi_df[defect_col].mean() if defect_col in kpi_df.columns else np.nan

	# Compute deltas relative to the full dataset totals (for quick context)
	overall_rev = df[rev_col].sum() if rev_col in df.columns else np.nan
	overall_costs = df[cost_col].sum() if cost_col in df.columns else np.nan
	overall_profit = (overall_rev - overall_costs) if (not np.isnan(overall_rev) and not np.isnan(overall_costs)) else np.nan

	def pct_delta(current, baseline):
		try:
			if np.isnan(current) or np.isnan(baseline) or baseline == 0:
				return None
			return (current - baseline) / abs(baseline)
		except Exception:
			return None

	rev_delta = pct_delta(total_rev, overall_rev)
	costs_delta = pct_delta(total_costs, overall_costs)
	profit_delta = pct_delta(total_profit, overall_profit)

	k1, k2, k3, k4 = st.columns(4)
	k1.metric("Total Revenue", f"${total_rev:,.0f}" if not np.isnan(total_rev) else "N/A", delta=f"{rev_delta:+.1%}" if rev_delta is not None else None)
	k2.metric("Total Costs", f"${total_costs:,.0f}" if not np.isnan(total_costs) else "N/A", delta=f"{costs_delta:+.1%}" if costs_delta is not None else None)
	k3.metric("Total Profit", f"${total_profit:,.0f}" if not np.isnan(total_profit) else "N/A", delta=f"{profit_delta:+.1%}" if profit_delta is not None else None)
	# Format defect as percent if looks like 0-1
	if not np.isnan(avg_defect):
		if avg_defect <= 1:
			k4.metric("Avg Defect Rate", f"{avg_defect*100:.2f}%", delta=None)
		else:
			k4.metric("Avg Defect Rate", f"{avg_defect:.2f}", delta=None)
	else:
		k4.metric("Avg Defect Rate", "N/A")

	# small caption about row count
	st.caption(f"Showing {len(kpi_df):,} rows for KPIs — global dataset total {len(df):,} rows")

	st.markdown("---")

	# Layout: left column charts and right column heatmap/donut
	left, right = st.columns((2, 1))

	# Bar chart by PRODUCT_TYPE — improved with labels and sorting
	with left:
		st.subheader("Bar chart by Product Type")
		# Bar chart: inline local filters
		exp_bar = st.expander("Bar filters", expanded=False)
		use_global_bar = exp_bar.checkbox("Use global filters", value=True, key="use_global_bar")
		prod_sel_bar = loc_sel_bar = trans_sel_bar = insp_sel_bar = None
		price_range_bar = (price_min, price_max)
		if not use_global_bar:
			prod_sel_bar = local_multiselect_for(product_col, "Bar: Product Type", exp_bar, key_suffix="bar_prod")
			loc_sel_bar = local_multiselect_for(location_col, "Bar: Location", exp_bar, key_suffix="bar_loc")
			trans_sel_bar = local_multiselect_for(transport_col, "Bar: Transportation Modes", exp_bar, key_suffix="bar_trans")
			insp_sel_bar = local_multiselect_for(inspect_col, "Bar: Inspection Results", exp_bar, key_suffix="bar_insp")
			if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
				pmin = float(df[price_col].min(skipna=True))
				pmax = float(df[price_col].max(skipna=True))
				price_range_bar = exp_bar.slider("Bar: Price range", min_value=pmin, max_value=pmax, value=(pmin, pmax), key="price_bar")

		bar_source = filtered if use_global_bar else apply_filters(df, prod_sel_bar, loc_sel_bar, trans_sel_bar, insp_sel_bar, price_range_bar)
		if product_col in bar_source.columns:
			grp = bar_source.groupby(product_col)
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
			# Add count per product for hover
			counts = bar_source[product_col].value_counts().reindex(bar_df[product_col]).fillna(0).values
			bar_df["count"] = counts
			fig_bar = px.bar(
				bar_df,
				x=product_col,
				y="value",
				text="value",
				hover_data={product_col: True, "value": ":,.0f", "count": True},
				labels={"value": y_label, product_col: "Product Type"},
				color=product_col,
				color_discrete_sequence=px.colors.qualitative.Set2,
			)
			fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside', showlegend=False)
			fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis_title=y_label, xaxis_tickangle=-45, height=450)
			st.plotly_chart(fig_bar, use_container_width=True)
		else:
			st.info("No product type column found for bar chart.")

		st.subheader("Costs vs Revenue (scatter)")
		# Scatter: inline local filters and min_sold control
		exp_scatter = st.expander("Scatter filters", expanded=False)
		use_global_scatter = exp_scatter.checkbox("Use global filters", value=True, key="use_global_scatter")
		prod_sel_scat = loc_sel_scat = trans_sel_scat = insp_sel_scat = None
		price_range_scat = (price_min, price_max)
		min_sold_local = 0
		if not use_global_scatter:
			prod_sel_scat = local_multiselect_for(product_col, "Scatter: Product Type", exp_scatter, key_suffix="scat_prod")
			loc_sel_scat = local_multiselect_for(location_col, "Scatter: Location", exp_scatter, key_suffix="scat_loc")
			trans_sel_scat = local_multiselect_for(transport_col, "Scatter: Transportation Modes", exp_scatter, key_suffix="scat_trans")
			insp_sel_scat = local_multiselect_for(inspect_col, "Scatter: Inspection Results", exp_scatter, key_suffix="scat_insp")
			if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
				pmin = float(df[price_col].min(skipna=True))
				pmax = float(df[price_col].max(skipna=True))
				price_range_scat = exp_scatter.slider("Scatter: Price range", min_value=pmin, max_value=pmax, value=(pmin, pmax), key="price_scatter")
			min_sold_local = int(exp_scatter.number_input("Minimum number of products sold (for scatter)", min_value=0, value=0, key="min_sold_scatter"))

		scatter_source = filtered if use_global_scatter else apply_filters(df, prod_sel_scat, loc_sel_scat, trans_sel_scat, insp_sel_scat, price_range_scat)
		if (cost_col in scatter_source.columns) and (rev_col in scatter_source.columns):
			scatter_df = scatter_source.copy()
			if qty_col and qty_col in scatter_df.columns:
				scatter_df = scatter_df[scatter_df[qty_col] >= min_sold_local]

			if scatter_df.empty:
				st.info("No data for scatter after applying minimum products sold filter.")
			else:
				# detect SKU column for richer hover
				sku_col = find_col(df, ["sku", "product_sku", "product code", "item_id", "product_code"]) or "SKU"
				hover_cols = []
				if product_col in scatter_df.columns:
					hover_cols.append(product_col)
				if location_col in scatter_df.columns:
					hover_cols.append(location_col)
				if sku_col in scatter_df.columns:
					hover_cols.append(sku_col)

				size = scatter_df[qty_col] if (qty_col and qty_col in scatter_df.columns) else None
				fig_scatter = px.scatter(
					scatter_df,
					x=cost_col,
					y=rev_col,
					color=product_col if product_col in scatter_df.columns else None,
					size=size,
					size_max=18,
					hover_data=hover_cols if hover_cols else None,
					labels={cost_col: "Costs", rev_col: "Revenue"},
					color_discrete_sequence=px.colors.qualitative.Dark24,
				)
				# Add 1:1 reference line
				min_xy = min(scatter_df[cost_col].min(), scatter_df[rev_col].min())
				max_xy = max(scatter_df[cost_col].max(), scatter_df[rev_col].max())
				fig_scatter.add_shape(type="line", x0=min_xy, y0=min_xy, x1=max_xy, y1=max_xy, line=dict(dash='dash', color='gray'))
				fig_scatter.update_layout(height=450)
				st.plotly_chart(fig_scatter, use_container_width=True)
		else:
			st.info("Revenue or Costs column not found for scatter plot.")

	# Right column: heatmap and donut
	with right:
		st.subheader("Heatmap: Avg Defect Rate")
		# Heatmap: inline local filters
		exp_heat = st.expander("Heatmap filters", expanded=False)
		use_global_heat = exp_heat.checkbox("Use global filters", value=True, key="use_global_heat")
		prod_sel_heat = loc_sel_heat = trans_sel_heat = insp_sel_heat = None
		price_range_heat = (price_min, price_max)
		if not use_global_heat:
			prod_sel_heat = local_multiselect_for(product_col, "Heatmap: Product Type", exp_heat, key_suffix="heat_prod")
			loc_sel_heat = local_multiselect_for(location_col, "Heatmap: Location", exp_heat, key_suffix="heat_loc")
			trans_sel_heat = local_multiselect_for(transport_col, "Heatmap: Transportation Modes", exp_heat, key_suffix="heat_trans")
			insp_sel_heat = local_multiselect_for(inspect_col, "Heatmap: Inspection Results", exp_heat, key_suffix="heat_insp")
			if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
				pmin = float(df[price_col].min(skipna=True))
				pmax = float(df[price_col].max(skipna=True))
				price_range_heat = exp_heat.slider("Heatmap: Price range", min_value=pmin, max_value=pmax, value=(pmin, pmax), key="price_heat")

		heat_source = filtered if use_global_heat else apply_filters(df, prod_sel_heat, loc_sel_heat, trans_sel_heat, insp_sel_heat, price_range_heat)
		if (location_col in heat_source.columns) and (transport_col in heat_source.columns) and (defect_col in heat_source.columns):
			heat = heat_source.groupby([location_col, transport_col])[defect_col].mean().reset_index()
			pivot = heat.pivot(index=location_col, columns=transport_col, values=defect_col).fillna(0)
			z = pivot.values
			x = pivot.columns.astype(str)
			y = pivot.index.astype(str)
			fig_heat = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale="Viridis", colorbar=dict(title="Avg Defect")))
			# Add annotations with formatted percent when 0-1
			annotations = []
			for i, yi in enumerate(y):
				for j, xj in enumerate(x):
					val = z[i][j]
					if np.isnan(val):
						txt = "-"
					else:
						txt = f"{val*100:.1f}%" if val <= 1 else f"{val:.2f}"
					annotations.append(dict(x=xj, y=yi, text=txt, showarrow=False, font=dict(color='white' if val > z.max()/2 else 'black')))
			fig_heat.update_layout(annotations=annotations, xaxis_title=transport_col, yaxis_title=location_col, height=420)
			st.plotly_chart(fig_heat, use_container_width=True)
		else:
			st.info("Require LOCATION, TRANSPORTATION_MODES and a defect rate column for heatmap.")

		st.subheader("Donut chart")
		if donut_choice and donut_choice in filtered.columns:
			vc = filtered[donut_choice].value_counts(dropna=False)
			donut_df = vc.reset_index()
			donut_df.columns = [donut_choice, "count"]
			donut_df["pct"] = donut_df["count"] / donut_df["count"].sum()
			# Group small slices into 'Other' (less than 3%)
			thresh = 0.03
			large = donut_df[donut_df["pct"] >= thresh].copy()
			small = donut_df[donut_df["pct"] < thresh]
			if not small.empty:
				other = pd.DataFrame({donut_choice: ["Other"], "count": [small["count"].sum()]})
				other["pct"] = other["count"] / donut_df["count"].sum()
				plot_df = pd.concat([large, other], ignore_index=True)
			else:
				plot_df = large

			fig_donut = go.Figure(data=[go.Pie(labels=plot_df[donut_choice], values=plot_df["count"], hole=0.55)])
			fig_donut.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{value} (%{percent})')
			fig_donut.update_layout(height=400)
			st.plotly_chart(fig_donut, use_container_width=True)
		else:
			st.info("No category selected or column missing for donut chart.")


if __name__ == "__main__":
	main()
