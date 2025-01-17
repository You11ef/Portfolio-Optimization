import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns  # For correlation heatmap
import financedatabase as fd

# -------------------------------------------------
# 1. LOAD & PREPARE A COMPREHENSIVE TICKER LIST
# -------------------------------------------------
@st.cache_data
def load_ticker_list():
    """
    Loads a large list of possible tickers (ETFs and Equities) from financedatabase,
    merges symbol and name into a single string for a better user search experience.
    """
    # Fetch ETFs and Equities
    df_etfs = fd.ETFs().select().reset_index()[['symbol', 'name']]
    df_equities = fd.Equities().select().reset_index()[['symbol', 'name']]

    # Combine them
    df_all = pd.concat([df_etfs, df_equities], ignore_index=True)
    df_all.dropna(subset=['symbol'], inplace=True)

    # Filter out tickers that start with '^'
    df_all = df_all[~df_all['symbol'].str.startswith('^')]

    # Create 'symbol_name' = "AAPL - Apple Inc."
    df_all['symbol_name'] = df_all['symbol'] + " - " + df_all['name'].fillna('')
    df_all.drop_duplicates(subset='symbol', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all

ticker_list = load_ticker_list()

# -------------------------------------------------
# 2. STREAMLIT APP LAYOUT
# -------------------------------------------------
st.set_page_config(page_title="Interactive Portfolio Optimization", layout="wide")

st.title("Interactive Portfolio Optimization")

# --- SIDEBAR: Portfolio Settings
st.sidebar.header("Portfolio Settings")

# --- A) Ticker Search & Selection
st.sidebar.subheader("Choose Tickers")
# Use the combined "symbol_name" as selectable options
sel_tickers = st.sidebar.multiselect(
    "Search and Select Tickers",
    options=ticker_list["symbol_name"],
    default=[]
)

# Convert user-friendly "symbol_name" back to the raw symbol
sel_symbol_list = ticker_list.loc[ticker_list.symbol_name.isin(sel_tickers), 'symbol'].tolist()

# --- B) Date Range Selection
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# --- C) Risk-free Rate Selection
st.sidebar.subheader("Risk-Free Rate")
use_auto_risk_free = st.sidebar.checkbox("Auto-fetch Rate from FRED", value=True)
if not use_auto_risk_free:
    risk_free_rate_input = st.sidebar.number_input("Enter Risk-Free Rate (%)", value=1.5)
    risk_free_rate = risk_free_rate_input / 100
else:
    fred_api_key = st.secrets.get("FRED_API_KEY", "FRED_API_KEY")  # Use your real key or store in secrets
    fred = Fred(api_key=fred_api_key)
    try:
        ten_year_rate_series = fred.get_series_latest_release("GS10") / 100
        risk_free_rate = ten_year_rate_series.iloc[-1]
        st.sidebar.write(f"Using Fetched 10Y Rate: {risk_free_rate:.4f}")
    except Exception as e:
        st.sidebar.error("Error fetching from FRED. Using default 1.5%.")
        risk_free_rate = 0.015

# --- D) Weight Bounds
st.sidebar.subheader("Weight Bounds per Asset")
min_weight = st.sidebar.slider("Minimum Weight", 0.0, 0.5, 0.0, 0.05)
max_weight = st.sidebar.slider("Maximum Weight", 0.0, 1.0, 0.4, 0.05)

# -------------------------------------------------
# 3. DISPLAY SELECTED TICKERS & PRICE CHART
# -------------------------------------------------
if len(sel_symbol_list) > 0:
    st.markdown(f"**You have selected {len(sel_symbol_list)} tickers**: {', '.join(sel_symbol_list)}")

    # 3A) Fetch & display line chart
    st.subheader("Price History")
    data_raw = yf.download(
        sel_symbol_list, 
        start=start_date, 
        end=end_date, 
        auto_adjust=True
    )

    if 'Close' in data_raw:
        # If multiple tickers -> 'Close' is a DataFrame; if one ticker -> Series
        price_data = data_raw['Close']
    else:
        # If we only have one ticker, data_raw might already be the 'Close' series
        price_data = data_raw

    # Forward-fill or handle missing data
    price_data = price_data.ffill().dropna()

    # Streamlit line chart expects a DataFrame
    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame()

    st.line_chart(price_data)

    # 3B) Display correlation heatmap
    st.subheader("Correlation Heatmap")
    if len(sel_symbol_list) > 1:
        corr_matrix = price_data.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.info("Correlation Heatmap is available when 2 or more tickers are selected.")
else:
    st.info("Select at least one ticker to see data and enable optimization.")

# -------------------------------------------------
# 4. PORTFOLIO OPTIMIZATION
# -------------------------------------------------
if st.sidebar.button("Optimize Portfolio"):
    if len(sel_symbol_list) == 0:
        st.warning("No tickers selected. Please select tickers first.")
    else:
        st.subheader("Portfolio Optimization Results")
        # Re-fetch the data to ensure correct shape
        df_prices = yf.download(sel_symbol_list, start=start_date, end=end_date, auto_adjust=True)['Close']
        df_prices.fillna(method='ffill', inplace=True)
        df_prices.dropna(axis=0, how='all', inplace=True)

        # If there's only one ticker, make sure df_prices is a DataFrame
        if isinstance(df_prices, pd.Series):
            df_prices = df_prices.to_frame()

        # Calculate log returns
        log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
        cov_matrix = log_returns.cov() * 252

        # Define portfolio functions
        def portfolio_std(weights, cov):
            return np.sqrt(weights.T @ cov @ weights)

        def portfolio_return(weights, returns):
            return np.sum(returns.mean() * weights) * 252

        def sharpe_ratio(weights, returns, cov, rf):
            return (portfolio_return(weights, returns) - rf) / portfolio_std(weights, cov)

        def neg_sharpe_ratio(weights, returns, cov, rf):
            return -sharpe_ratio(weights, returns, cov, rf)

        # Prepare optimization
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(min_weight, max_weight) for _ in sel_symbol_list]
        initial_weights = np.array([1 / len(sel_symbol_list)] * len(sel_symbol_list))

        # Run optimization
        optimized = minimize(
            neg_sharpe_ratio,
            initial_weights,
            args=(log_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds
        )

        if optimized.success:
            opt_weights = optimized.x

            # Display weights
            st.markdown("### Optimal Portfolio Weights")
            w_df = pd.DataFrame({
                "Ticker": sel_symbol_list,
                "Weight": opt_weights
            })
            st.dataframe(w_df.set_index("Ticker"), use_container_width=True)

            # Calculate metrics
            opt_ret = portfolio_return(opt_weights, log_returns)
            opt_vol = portfolio_std(opt_weights, cov_matrix)
            opt_sharpe = sharpe_ratio(opt_weights, log_returns, cov_matrix, risk_free_rate)

            # 4A) Display Portfolio Metrics in a table
            st.markdown("### Portfolio Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Expected Annual Return", "Expected Annual Volatility", "Sharpe Ratio"],
                "Value": [f"{opt_ret:.4f}", f"{opt_vol:.4f}", f"{opt_sharpe:.4f}"]
            })
            st.table(metrics_df)

            # 4B) Allocation pie chart with a custom color palette
            # Filter near-zero weights
            epsilon = 1e-6
            nz_idx = [i for i, w in enumerate(opt_weights) if abs(w) > epsilon]
            filtered_tickers = [sel_symbol_list[i] for i in nz_idx]
            filtered_weights = [opt_weights[i] for i in nz_idx]

            # Example: using "tab20" palette from matplotlib
            # We pick len(filtered_weights) distinct colors from this palette
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i) for i in range(len(filtered_weights))]

            fig, ax = plt.subplots()
            ax.pie(filtered_weights, labels=filtered_tickers, autopct='%1.1f%%', startangle=90,
                   colors=colors)
            ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle

            st.markdown("### Optimal Allocation")
            st.pyplot(fig)

        else:
            st.error("Optimization failed. Please adjust your inputs and try again.")
