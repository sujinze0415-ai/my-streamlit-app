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

	# Generic helper to render multiselects within an expander
	def chart_filters_ui(name, use_default=True):
		exp = st.sidebar.expander(f"{name} filters", expanded=False)
		use_global = exp.checkbox("Use global filters", value=use_default, key=f"use_global_{name}")
		prod_sel = loc_sel = trans_sel = insp_sel = None
		price_range = (price_min, price_max)
		if not use_global:
			with exp:
				prod_sel = multiselect_for(product_col, f"{name}: Product Type")
				loc_sel = multiselect_for(location_col, f"{name}: Location")
				trans_sel = multiselect_for(transport_col, f"{name}: Transportation Modes")
				insp_sel = multiselect_for(inspect_col, f"{name}: Inspection Results")
				# price range only if price column exists
				if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
					pmin = float(df[price_col].min(skipna=True))
					pmax = float(df[price_col].max(skipna=True))
					price_range = exp.slider(f"{name}: Price range", min_value=pmin, max_value=pmax, value=(pmin, pmax), key=f"price_{name}")
		return dict(use_global=use_global, prod_sel=prod_sel, loc_sel=loc_sel, trans_sel=trans_sel, insp_sel=insp_sel, price_range=price_range)

	# Create per-chart filter selections for KPIs, Bar, Scatter, Heatmap
	kpi_filters = chart_filters_ui("KPIs", use_default=True)
	bar_filters = chart_filters_ui("Bar", use_default=True)
	scatter_filters = chart_filters_ui("Scatter", use_default=True)
	heatmap_filters = chart_filters_ui("Heatmap", use_default=True)

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

	# Determine which dataframe to use for KPIs (global filtered or its own filters)
	if kpi_filters.get("use_global", True):
		kpi_df = filtered
	else:
		kpi_df = apply_filters(df, kpi_filters.get("prod_sel"), kpi_filters.get("loc_sel"), kpi_filters.get("trans_sel"), kpi_filters.get("insp_sel"), kpi_filters.get("price_range"))

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
		# bar chart may use its own filters
		bar_source = filtered if bar_filters.get("use_global", True) else apply_filters(df, bar_filters.get("prod_sel"), bar_filters.get("loc_sel"), bar_filters.get("trans_sel"), bar_filters.get("insp_sel"), bar_filters.get("price_range"))
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
		# scatter may use its own filters
		scatter_source = filtered if scatter_filters.get("use_global", True) else apply_filters(df, scatter_filters.get("prod_sel"), scatter_filters.get("loc_sel"), scatter_filters.get("trans_sel"), scatter_filters.get("insp_sel"), scatter_filters.get("price_range"))
		if (cost_col in scatter_source.columns) and (rev_col in scatter_source.columns):
			scatter_df = scatter_source.copy()
			if qty_col and qty_col in scatter_df.columns:
				scatter_df = scatter_df[scatter_df[qty_col] >= min_sold]

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
		# heatmap may use its own filters
		heat_source = filtered if heatmap_filters.get("use_global", True) else apply_filters(df, heatmap_filters.get("prod_sel"), heatmap_filters.get("loc_sel"), heatmap_filters.get("trans_sel"), heatmap_filters.get("insp_sel"), heatmap_filters.get("price_range"))
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
