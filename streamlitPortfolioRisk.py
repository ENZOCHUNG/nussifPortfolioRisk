import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt
import plotly.graph_objects as go

# === Streamlit page config ===
st.set_page_config(layout="wide")

# ==============================
# Load data
# ==============================
risk_df = pd.read_parquet("global_avg_metric.parquet")

# ==============================
# Sidebar controls
# ==============================
st.sidebar.title("Settings")

# Map friendly labels → internal keys
view_map = {
    "Value at Risk": "Var",
    "Expected Shortfall": "Es",
    "Weight & Volatility": "Weights vs Volatility",
    "Correlation Matrix": "Corr",
    "Portfolio Performance" : "PnL"
}

view_option_label = st.sidebar.radio(
    "Select View",
    list(view_map.keys())
)

view_option = view_map[view_option_label]

# Confidence only matters for Var/Es
confidence = None
if view_option in ["Var", "Es"]:
    confidence = st.sidebar.radio("Select Confidence Level", ["99", "95", "90"])

# ==============================
# Section 1: Portfolio Risk (VaR / ES)
# ==============================
if view_option in ["Var", "Es"]:
    metric_type = view_option
    st.title(f"Portfolio Risk Dashboard — {view_option_label} {confidence}%")

    # Pick columns dynamically
    cols = [f"avg{metric_type}{confidence}_{k}d" for k in [1, 5, 21]]

    # Melt dataframe for plotting
    plot_df = risk_df.melt(
        id_vars="date",
        value_vars=cols,
        var_name="Horizon",
        value_name="Value"
    )
    plot_df["Horizon"] = plot_df["Horizon"].str.extract(r"_(\d+d)$")

    # Plot risk chart
    fig_risk = px.line(
        plot_df,
        x="date",
        y="Value",
        color="Horizon",
        markers=True,
        title=f"Average {view_option_label} {confidence}% across horizons"
    )
    fig_risk.update_layout(
        xaxis_title="Date",
        yaxis_title="SGD",
        legend_title="Horizon",
        height=600,
        width=900
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# ==============================
# Section 2: Portfolio Weights vs Volatility
# ==============================
elif view_option == "Weights vs Volatility":
    st.title("Portfolio Weights vs Volatility")

    # Load weights & vol data
    weights_df = pd.read_parquet("weights.parquet")
    vol_df = pd.read_parquet("vol.parquet")

    # Merge on symbol + date
    merged_df = pd.merge(weights_df, vol_df, on=["symbol", "date"], how="inner")

    # Rename CASH_SGD → CCY_SGD
    merged_df["symbol"] = merged_df["symbol"].replace("CASH_SGD", "CCY_SGD")

    # ✅ Keep only the latest snapshot
    latest_date = merged_df["date"].max()
    merged_latest = merged_df[merged_df["date"] == latest_date]

    # ✅ Drop duplicates just in case
    merged_latest = merged_latest.drop_duplicates(subset=["symbol"], keep="last")

    # Split into currencies vs stocks
    df_currencies = merged_latest[merged_latest["symbol"].str.startswith("CCY_")].sort_values("weights", ascending=False)
    df_stocks     = merged_latest[~merged_latest["symbol"].str.startswith("CCY_")].sort_values("weights", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Currencies")
        fig_currencies = px.bar(
            df_currencies,
            x="symbol",
            y="weights",
            color="volatility",
            color_continuous_scale="Viridis",
            text="weights"
        )
        fig_currencies.update_traces(
            texttemplate='%{y:.2%}',  # show as %
            textposition="outside"
        )
        fig_currencies.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title="Portfolio Weight",
            height=500
        )
        st.plotly_chart(fig_currencies, use_container_width=True)

    with col2:
        st.subheader("Stocks / ETFs")
        fig_stocks = px.bar(
            df_stocks,
            x="symbol",
            y="weights",
            color="volatility",
            color_continuous_scale="Viridis",
            text="weights"
        )
        fig_stocks.update_traces(
            texttemplate='%{y:.2%}',
            textposition="outside"
        )
        fig_stocks.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title="Portfolio Weight",
            height=500
        )
        st.plotly_chart(fig_stocks, use_container_width=True)

# ==============================
# Section 3: Correlation Matrix
# ==============================
elif view_option == "Corr":
    st.title("Correlation Matrix")

    raw_dt = dt.datetime.now()
    today = pd.Timestamp(raw_dt).round('h')

    corr_file = "corr.parquet"
    df_corr = pd.read_parquet(corr_file)

    # Normalize date format
    df_corr["date"] = pd.to_datetime(df_corr["date"])

    if df_corr.empty:
        st.error(f"No correlation data found for {today}")
    else:
        corr_matrix = df_corr.pivot(index="asset1", columns="asset2", values="corr")

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            aspect="equal",  # square cells
            title="Correlation Heatmap (Today)"
        )
        fig_corr.update_layout(
            xaxis_title="Assets",
            yaxis_title="Assets",
            xaxis=dict(tickangle=0),
            margin=dict(l=40, r=40, t=60, b=40),
            height=900   # 🔹 make it taller/bigger
        )

        st.plotly_chart(fig_corr, use_container_width=True)

# ==============================
# Section 4: PnL Section
# ==============================
elif view_option == "PnL":
    st.title("Portfolio Performance & PnL Attribution")

    # --- KPI Summary Row ---
    try:
        df_pnl = pd.read_parquet("pnl.parquet")
        df_pnl["date"] = pd.to_datetime(df_pnl["date"])
        # Sort by date to ensure lines connect correctly
        df_pnl = df_pnl.sort_values("date")
    except FileNotFoundError:
        st.error("pnl.parquet not found. Please run your processor script first.")
        st.stop()
    
    latest = df_pnl.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current NAV", f"${latest['nav']:,.2f}")
    c2.metric("Total PnL", f"${latest['totalPnL']:,.2f}")
    c3.metric("Realised", f"${latest['realisedPnL']:,.2f}")
    c4.metric("Return", f"{latest['percentage_change']}%")
    # Load PnL data
    # Path handles the directory structure we discussed (../app/pnl.parquet)
    try:
        df_pnl = pd.read_parquet("pnl.parquet")
        df_pnl["date"] = pd.to_datetime(df_pnl["date"])
        # Sort by date to ensure lines connect correctly
        df_pnl = df_pnl.sort_values("date")
    except FileNotFoundError:
        st.error("pnl.parquet not found. Please run your processor script first.")
        st.stop()

    # --- Chart 1: NAV Against Time ---
    st.subheader("1. Net Asset Value (Equity Curve)")
    fig_nav = px.line(
        df_pnl, 
        x="date", 
        y="nav", 
        title="Total Portfolio Value (SGD)",
        markers=True,
        line_shape="linear"
    )
    fig_nav.update_layout(yaxis_title="NAV (SGD)", height=400)
    # Don't start Y-axis at 0 to see the hourly movements clearly
    fig_nav.update_yaxes(autorange=True, fixedrange=False)
    st.plotly_chart(fig_nav, use_container_width=True)

    # --- Chart 2: PnL Attribution (Stacked Area) ---
    st.subheader("2. PnL Attribution: Realised vs Unrealised")
    
    fig_attrib = go.Figure()

    # Realised PnL Area
    fig_attrib.add_trace(go.Scatter(
        x=df_pnl["date"], 
        y=df_pnl["realisedPnL"],
        mode='lines',
        name='Realised PnL',
        stackgroup='one', # Groups the areas
        fillcolor='rgba(26, 150, 65, 0.5)', # Greenish
        line=dict(width=0.5, color='rgb(26, 150, 65)')
    ))

    # Unrealised PnL Area
    fig_attrib.add_trace(go.Scatter(
        x=df_pnl["date"], 
        y=df_pnl["unrealisedPnL"],
        mode='lines',
        name='Unrealised PnL',
        stackgroup='one',
        fillcolor='rgba(0, 176, 246, 0.5)', # Bluish
        line=dict(width=0.5, color='rgb(0, 176, 246)')
    ))

    # Total PnL Line (The sum of the above)
    fig_attrib.add_trace(go.Scatter(
        x=df_pnl["date"], 
        y=df_pnl["totalPnL"],
        mode='lines+markers',
        name='Total PnL',
        line=dict(color='black', width=3, dash='dot')
    ))

    fig_attrib.update_layout(
        title="Total PnL Composition",
        xaxis_title="Date",
        yaxis_title="PnL (SGD)",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig_attrib, use_container_width=True)

    # --- Chart 3: Cumulative Percentage Return ---
    st.subheader("3. Overall Portfolio Return (%)")
    
    # Using a line chart because this is a cumulative metric (Total Return)
    fig_pct = px.line(
        df_pnl,
        x="date",
        y="percentage_change",
        title="Cumulative Return Since Inception",
        markers=True,
        line_shape="linear"
    )
    
    # Professional touch: Add a horizontal line at 0.0 to clearly show profit vs loss
    fig_pct.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Inception")

    fig_pct.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Return (%)",
        yaxis_ticksuffix="%",
        height=400
    )
    
    # Improve the tooltip to show the exact percentage
    fig_pct.update_traces(hovertemplate="Date: %{x}<br>Total Return: %{y}%")
    
    st.plotly_chart(fig_pct, use_container_width=True)