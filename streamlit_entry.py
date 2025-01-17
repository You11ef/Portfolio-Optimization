import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
import financedatabase as fd

# -------------------------------------------------
# 1. PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Interactive Portfolio Optimization", layout="wide")

# -------------------------------------------------
# 2. HELPER FUNCTIONS & CACHING
# -------------------------------------------------
@st.cache_data
def load_ticker_list():
    """Load & prepare a comprehensive ticker list from financedatabase."""
    df_etfs = fd.ETFs().select().reset_index()[['symbol', 'name']]
    df_equities = fd.Equities().select().reset_index()[['symbol', 'name']]
    df_all = pd.concat([df_etfs, df_equities], ignore_index=True)
    df_all.dropna(subset=['symbol'], inplace=True)
    df_all = df_all[~df_all['symbol'].str.startswith('^')]
    df_all['symbol_name'] = df_all['symbol'] + " - " + df_all['name'].fillna('')
    df_all.drop_duplicates(subset='symbol', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all

@st.cache_data
def fetch_prices(tickers, start, end):
    """Download price data once, to avoid re-downloading for every small parameter change."""
    return yf.download(tickers, start=start, end=end, auto_adjust=True)

def resample_prices(df_prices, freq):
    """
    Resample prices to user-selected frequency.
    freq = "Daily", "Weekly", or "Monthly".
    """
    if freq == "Daily":
        return df_prices  # no resampling
    elif freq == "Weekly":
        # Use 'W-FRI' so data lines up with last available price on Friday
        return df_prices.resample('W-FRI').last().dropna(how='all')
    elif freq == "Monthly":
        return df_prices.resample('M').last().dropna(how='all')

def portfolio_return(weights, returns, ann_factor):
    """Annualized portfolio return."""
    return np.sum(returns.mean() * weights) * ann_factor

def portfolio_std(weights, cov, ann_factor):
    """Annualized portfolio volatility."""
    return np.sqrt(weights.T @ cov @ weights) * (ann_factor ** 0.5)

def sharpe_ratio(weights, returns, cov, rf, ann_factor):
    """Annualized Sharpe ratio."""
    ret = portfolio_return(weights, returns, ann_factor)
    vol = portfolio_std(weights, cov, ann_factor)
    return (ret - rf) / vol if vol != 0 else 0

def sortino_ratio(weights, returns, rf, ann_factor):
    """
    Annualized Sortino ratio:
    (Return - RF) / Downside Deviation
    Using 0 as the 'target' for downside risk, or you can use rf if you prefer.
    """
    portfolio_daily = returns @ weights  # daily returns of the portfolio
    avg_return = portfolio_daily.mean() * ann_factor
    # Downside deviation relative to 0:
    negative_excess = portfolio_daily[portfolio_daily < 0]
    if len(negative_excess) == 0:
        # no negative returns => effectively infinite sortino
        return float('inf')
    downside_std = np.sqrt((negative_excess ** 2).mean()) * (ann_factor ** 0.5)
    if downside_std == 0:
        return float('inf')
    return (avg_return - rf) / downside_std

def objective_function(weights, returns, cov, rf, ann_factor, objective):
    """
    Unified objective function for:
      - Max Sharpe => minimize negative Sharpe
      - Min Vol    => minimize volatility
      - Max Return => minimize negative return
      - Max Sortino => minimize negative sortino
    """
    if objective == "Max Sharpe Ratio":
        return -sharpe_ratio(weights, returns, cov, rf, ann_factor)
    elif objective == "Min Volatility":
        return portfolio_std(weights, cov, ann_factor)
    elif objective == "Max Return":
        return -portfolio_return(weights, returns, ann_factor)
    elif objective == "Max Sortino Ratio":
        return -sortino_ratio(weights, returns, rf, ann_factor)

@st.cache_data
def generate_random_portfolios(num_portfolios, n_assets, min_weight, max_weight, 
                               returns, cov, rf, ann_factor):
    """Generate random feasible portfolios, compute metrics for plotting the efficient frontier."""
    results = []
    for _ in range(num_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()  # ensure sum to 1
        # enforce bounds
        if np.any(w < min_weight) or np.any(w > max_weight):
            continue
        ret = portfolio_return(w, returns, ann_factor)
        vol = portfolio_std(w, cov, ann_factor)
        sr = sharpe_ratio(w, returns, cov, rf, ann_factor)
        results.append((ret, vol, sr, w))
    # Convert to DataFrame
    columns = ['Return', 'Volatility', 'Sharpe', 'Weights']
    df_rand = pd.DataFrame(results, columns=columns)
    return df_rand


# -------------------------------------------------
# 3. MAIN APP LAYOUT
# -------------------------------------------------
ticker_list = load_ticker_list()

st.title("Interactive Portfolio Optimization")
st.sidebar.header("Portfolio Settings")

# 3.1 Ticker selection
st.sidebar.subheader("Choose Tickers")
sel_tickers = st.sidebar.multiselect(
    "Search and Select Tickers",
    options=ticker_list["symbol_name"],
    default=[]
)
sel_symbol_list = ticker_list.loc[ticker_list.symbol_name.isin(sel_tickers), 'symbol'].tolist()

# 3.2 Date Range
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# 3.3 Frequency selection
st.sidebar.subheader("Price Frequency")
freq_choice = st.sidebar.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
freq_map = {"Daily": 252, "Weekly": 52, "Monthly": 12}
ann_factor = freq_map[freq_choice]

# 3.4 Risk-free Rate
st.sidebar.subheader("Risk-Free Rate")
use_auto_risk_free = st.sidebar.checkbox("Auto-fetch Rate from FRED", value=True)
if not use_auto_risk_free:
    risk_free_rate_input = st.sidebar.number_input("Enter Risk-Free Rate (%)", value=1.5)
    risk_free_rate = risk_free_rate_input / 100
else:
    fred_api_key = st.secrets.get("FRED_API_KEY", "FRED_API_KEY")
    fred = Fred(api_key=fred_api_key)
    try:
        ten_year_rate_series = fred.get_series_latest_release("GS10") / 100
        risk_free_rate = ten_year_rate_series.iloc[-1]
        st.sidebar.write(f"Using Fetched 10Y Rate: {risk_free_rate:.4f}")
    except Exception:
        st.sidebar.error("Error fetching from FRED. Using default 1.5%.")
        risk_free_rate = 0.015

# 3.5 Weight Bounds
st.sidebar.subheader("Weight Bounds per Asset")
min_weight = st.sidebar.slider("Minimum Weight", 0.0, 0.5, 0.0, 0.05)
max_weight = st.sidebar.slider("Maximum Weight", 0.0, 1.0, 0.4, 0.05)

# 3.6 Optimization Objective
st.sidebar.subheader("Objective Function")
opt_objective = st.sidebar.selectbox(
    "Optimization Objective",
    ["Max Sharpe Ratio", "Min Volatility", "Max Return", "Max Sortino Ratio"]
)

# -------------------------------------------------
# 4. PRICE DATA & VISUALIZATION
# -------------------------------------------------
if sel_symbol_list:
    st.markdown(f"**You have selected {len(sel_symbol_list)} tickers**: {', '.join(sel_symbol_list)}")
    st.subheader("Price History")

    data_raw = fetch_prices(sel_symbol_list, start=start_date, end=end_date)
    if data_raw.empty:
        st.warning("No data available for the selected tickers and date range.")
    else:
        # Use Close prices if available
        if 'Close' in data_raw:
            price_data_raw = data_raw['Close']
        else:
            price_data_raw = data_raw

        # Forward-fill & drop rows entirely NaN
        price_data_raw = price_data_raw.ffill().dropna(how='all')

        # Identify which tickers truly have data
        unavailable_tickers = []
        valid_tickers = []
        for t in sel_symbol_list:
            if (t not in price_data_raw.columns) or (price_data_raw[t].isnull().all()):
                unavailable_tickers.append(t)
            else:
                valid_tickers.append(t)

        if unavailable_tickers:
            st.info(
                "The following tickers returned no data in the chosen range "
                f"and will be excluded from computations: {', '.join(unavailable_tickers)}"
            )

        if not valid_tickers:
            st.warning("No valid data available for the selected tickers.")
        else:
            full_price_data = price_data_raw[valid_tickers]

            # Show line chart of the full user-chosen date range
            st.line_chart(full_price_data)

            # Resample to chosen frequency
            # (We do this AFTER showing the full daily chart so user sees all data visually.)
            freq_price_data = resample_prices(full_price_data, freq_choice)

            # Build table of earliest & latest date for each ticker
            first_valid = full_price_data.apply(lambda col: col.first_valid_index())
            last_valid  = full_price_data.apply(lambda col: col.last_valid_index())

            table_data = []
            for t in valid_tickers:
                earliest = first_valid[t]
                latest = last_valid[t]
                table_data.append({
                    "Ticker": t,
                    "Earliest Available": earliest.strftime("%Y-%m-%d") if earliest else "N/A",
                    "Latest Available": latest.strftime("%Y-%m-%d") if latest else "N/A"
                })

            st.subheader("Data Availability Table")
            st.write("Below is each ticker’s earliest & latest date _within your chosen date range_.")
            st.dataframe(pd.DataFrame(table_data))

            # Determine intersection range
            non_null_starts = [d for d in first_valid if d is not None]
            non_null_ends = [d for d in last_valid if d is not None]
            max_start = None
            min_end = None
            if non_null_starts and non_null_ends:
                max_start = max(non_null_starts)
                min_end = min(non_null_ends)
                if max_start >= min_end:
                    st.warning("No overlapping date range found among these tickers.")
                else:
                    st.write(
                        f"**Intersection date range (for correlation & optimization):** "
                        f"{max_start.strftime('%Y-%m-%d')} → {min_end.strftime('%Y-%m-%d')}"
                    )
            else:
                st.warning("No valid intersection found (some tickers have no data at all).")

            # -------------------------------------------------
            # 4.1 CORRELATION HEATMAP (on intersection range, but at user-chosen frequency)
            # -------------------------------------------------
            st.subheader("Correlation Heatmap")
            if len(valid_tickers) > 1 and (max_start and min_end and max_start < min_end):
                # Slice freq_price_data by intersection
                freq_price_data = freq_price_data.loc[max_start:min_end].dropna(how='all')
                if freq_price_data.empty:
                    st.warning("No valid data in the intersection date range for correlation.")
                else:
                    corr_matrix = freq_price_data.corr()
                    cmap = sns.diverging_palette(240, 10, as_cmap=True)
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        cmap=cmap,
                        vmin=-1, vmax=1, center=0,
                        ax=ax_corr
                    )
                    st.pyplot(fig_corr)
            else:
                st.info("Correlation Heatmap is shown when ≥2 valid tickers & a valid intersection range exist.")
else:
    st.info("Select at least one ticker to see data and enable optimization.")

# -------------------------------------------------
# 5. OPTIMIZATION
# -------------------------------------------------
if st.sidebar.button("Optimize Portfolio"):
    if not sel_symbol_list:
        st.warning("No tickers selected. Please select tickers first.")
    else:
        if 'full_price_data' not in locals() or full_price_data.empty:
            st.warning("Cannot perform optimization due to lack of valid price data.")
        else:
            valid_tickers = full_price_data.columns.tolist()
            st.subheader("Portfolio Optimization Results")

            if ('max_start' not in locals() or 'min_end' not in locals() 
                or max_start is None or min_end is None or max_start >= min_end):
                st.warning("Cannot optimize because there is no valid intersection date range.")
            else:
                # Resample & slice to intersection
                freq_price_data = resample_prices(full_price_data, freq_choice)
                restricted_data = freq_price_data.loc[max_start:min_end].dropna(how='all')

                if restricted_data.empty:
                    st.warning("No data in the intersection date range to optimize on.")
                else:
                    # Compute log-returns
                    log_returns = np.log(restricted_data / restricted_data.shift(1)).dropna()
                    cov_matrix = log_returns.cov()
                    # Adjust for annualization in the objective functions
                    # (We'll multiply stdev by sqrt(ann_factor) and returns by ann_factor inside the objective.)

                    # SLSQP constraints
                    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
                    bounds = [(min_weight, max_weight) for _ in valid_tickers]
                    initial_weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))

                    # Run optimization
                    optimized = minimize(
                        objective_function,
                        initial_weights,
                        args=(log_returns, cov_matrix, risk_free_rate, ann_factor, opt_objective),
                        method="SLSQP",
                        constraints=constraints,
                        bounds=bounds
                    )

                    if optimized.success:
                        opt_weights = optimized.x
                        w_df = pd.DataFrame({"Ticker": valid_tickers, "Weight": opt_weights})

                        # ---- Display weights & metrics
                        st.markdown("### Optimal Portfolio Weights")
                        st.dataframe(w_df.set_index("Ticker"), use_container_width=True)

                        # Key metrics
                        ret_val = portfolio_return(opt_weights, log_returns, ann_factor)
                        vol_val = portfolio_std(opt_weights, cov_matrix, ann_factor)
                        sr_val = sharpe_ratio(opt_weights, log_returns, cov_matrix, risk_free_rate, ann_factor)
                        so_val = sortino_ratio(opt_weights, log_returns, risk_free_rate, ann_factor)

                        st.markdown("### Portfolio Metrics")
                        metrics_df = pd.DataFrame({
                            "Metric": [
                                "Expected Annual Return",
                                "Expected Annual Volatility",
                                "Sharpe Ratio",
                                "Sortino Ratio"
                            ],
                            "Value": [
                                f"{ret_val:.4f}",
                                f"{vol_val:.4f}",
                                f"{sr_val:.4f}",
                                f"{so_val:.4f}"
                            ]
                        })
                        st.table(metrics_df)

                        # Pie Chart of Optimal Weights
                        cmap = plt.get_cmap("tab20b")
                        colors = [cmap(i) for i in range(len(opt_weights))]

                        fig, ax = plt.subplots()
                        ax.pie(
                            opt_weights,
                            labels=w_df["Ticker"],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors
                        )
                        ax.axis("equal")

                        st.markdown("### Optimal Allocation")
                        st.pyplot(fig)

                        # ---- Download CSV
                        csv_data = w_df.to_csv(index=False)
                        st.download_button(
                            label="Download Weights CSV",
                            data=csv_data.encode('utf-8'),
                            file_name="optimized_weights.csv",
                            mime="text/csv",
                        )

                        # ---- Efficient Frontier Visualization
                        st.markdown("### Efficient Frontier (Random Portfolios)")
                        # Generate random portfolios
                        df_rand = generate_random_portfolios(
                            num_portfolios=2000,
                            n_assets=len(valid_tickers),
                            min_weight=min_weight,
                            max_weight=max_weight,
                            returns=log_returns,
                            cov=cov_matrix,
                            rf=risk_free_rate,
                            ann_factor=ann_factor
                        )
                        if not df_rand.empty:
                            fig_ef, ax_ef = plt.subplots()
                            scatter = ax_ef.scatter(
                                df_rand["Volatility"],
                                df_rand["Return"],
                                c=df_rand["Sharpe"],  # color by Sharpe
                                cmap="viridis",
                                alpha=0.6
                            )
                            cbar = fig_ef.colorbar(scatter, ax=ax_ef)
                            cbar.set_label("Sharpe Ratio")

                            # Plot the optimized portfolio
                            ax_ef.scatter(
                                vol_val, ret_val,
                                c="red", s=80, edgecolors="black",
                                label="Optimized"
                            )
                            ax_ef.set_xlabel("Annual Volatility")
                            ax_ef.set_ylabel("Annual Return")
                            ax_ef.set_title("Random Portfolios & Optimized Portfolio")
                            ax_ef.legend()
                            st.pyplot(fig_ef)
                        else:
                            st.warning("Could not generate random portfolios with the given constraints.")
                    else:
                        st.error("Optimization failed. Please adjust your inputs and try again.")
