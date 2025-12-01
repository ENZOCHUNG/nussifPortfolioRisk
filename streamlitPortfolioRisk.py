import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt

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
    "Correlation Matrix": "Corr"
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
