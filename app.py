# =========================================================
# Australia Macro & Markets Dashboard (Streamlit)
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from openai import OpenAI
import yfinance as yf
import numpy as np
import plotly.express as px
from datetime import timedelta

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AU Macro & Markets Dashboard", layout="wide")

# ---------- OpenAI client ---------- #
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def explain_with_gpt(indicator_stats, umbrella_name):
    """Generate short analytical summary via GPT."""
    if not indicator_stats:
        return "No data available to summarize."
    prompt = f"""
    You are an Australian economic analyst. Based on the following indicators for {umbrella_name}, 
    write a concise analytical summary (2–3 sentences) focusing on key trends:

    {indicator_stats}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(AI summary unavailable: {e})"


# =========================================================
# 1) CoreLogic Housing Section
# =========================================================
def housing_market_section():
    st.header("🏠 CoreLogic Daily Home Value Index")

    try:
        df = pd.read_excel("data/corelogic_daily_index.xlsx", sheet_name=0)
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return

    if 'Date' not in df.columns:
        st.error("❌ 'Date' column not found in Excel file.")
        return

    df['Date'] = pd.to_datetime(df['Date'])

    cities = [
        'Sydney (SYDD)',
        'Melbourne (MELD)',
        'Brisbane (inc Gold Coast) (BRID)',
        'Adelaide (ADED)',
        'Perth (PERD)',
        '5 capital city aggregate (AUSD)'
    ]
    available_cities = [c for c in cities if c in df.columns]

    selected_cities = st.multiselect(
        "Select cities",
        available_cities,
        default=['5 capital city aggregate (AUSD)'] if '5 capital city aggregate (AUSD)' in available_cities else available_cities[:1]
    )

    if not selected_cities:
        st.warning("Please select at least one city to view data.")
        return

    # --- Line chart ---
    fig = px.line(
        df,
        x='Date',
        y=selected_cities,
        labels={'value': 'Home Value Index', 'variable': 'City'},
        title="Daily Home Value Index Trends"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Summary table ---
    st.subheader("📊 Summary Statistics (Last 12 Months)")
    summary = df[selected_cities].describe().T[['mean', 'min', 'max']]
    summary['% Change'] = (
        (df[selected_cities].iloc[-1].values / df[selected_cities].iloc[0].values - 1) * 100
    )
    st.dataframe(
        summary.style.format({
            'mean': '{:.2f}', 'min': '{:.2f}', 'max': '{:.2f}', '% Change': '{:.2f}%'
        })
    )

    # --- AI Summary ---
    try:
        one_year_ago = df['Date'].max() - timedelta(days=365)
        past_df = df[df['Date'] >= one_year_ago]

        yoy_stats = []
        for city in selected_cities:
            start_val = past_df[city].iloc[0]
            end_val = past_df[city].iloc[-1]
            change = ((end_val / start_val) - 1) * 100
            yoy_stats.append(f"{city}: {change:.2f}% change over past year")

        st.markdown("**AI Summary (YoY Changes):** " + explain_with_gpt("\n".join(yoy_stats), "YoY Index Changes"))
    except Exception as e:
        st.warning(f"AI summary unavailable: {e}")


# =========================================================
# 2) RBA + Market Dashboard Functions
# =========================================================
RBA_URLS = {
    "H3": "https://www.rba.gov.au/statistics/tables/xls/h03hist.xlsx",
    "H1": "https://www.rba.gov.au/statistics/tables/xls/h01hist.xlsx",
    "G1": "https://www.rba.gov.au/statistics/tables/xls/g01hist.xlsx",
    "G2": "https://www.rba.gov.au/statistics/tables/xls/g02hist.xlsx",
    "H5": "https://www.rba.gov.au/statistics/tables/xls/h05hist.xlsx",
    "H4": "https://www.rba.gov.au/statistics/tables/xls/h04hist.xlsx",
    "E1": "https://www.rba.gov.au/statistics/tables/xls/e01hist.xlsx",
    "E2": "https://www.rba.gov.au/statistics/tables/xls/e02hist.xlsx",
    "E13":"https://www.rba.gov.au/statistics/tables/xls/e13hist.xlsx",
    "F1": "https://www.rba.gov.au/statistics/tables/xls/f01hist.xlsx",
    "F6": "https://www.rba.gov.au/statistics/tables/xls/f06hist.xlsx",
    "D1": "https://www.rba.gov.au/statistics/tables/xls/d01hist.xlsx",
}

@st.cache_data(ttl=3600)
def load_rba_table(code: str) -> pd.DataFrame:
    url = RBA_URLS[code]
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_excel(BytesIO(r.content), skiprows=10)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df

def line_fig(df: pd.DataFrame, ycol: str, title: str, ylabel: str = "Percent / Index"):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(df["Date"], df[ycol])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    fig.tight_layout()
    return fig

def calc_mom_yoy(df: pd.DataFrame, col: str, label: str) -> str:
    series = df.dropna(subset=[col]).set_index("Date")[col]
    if len(series) < 2:
        return f"{label}: insufficient data"
    latest_date = series.index[-1].strftime("%b %Y")
    latest = series.iloc[-1]
    prev = series.iloc[-2]
    yoy = series.iloc[-13] if len(series) > 12 else None
    mom_change = (latest - prev) if prev is not None else None
    yoy_change = (latest - yoy) if yoy is not None else None
    return f"{label} ({latest_date}) — MoM: {mom_change:+.2f}, YoY: {yoy_change:+.2f}"

# =========================================================
# 3) App Layout with Tabs
# =========================================================
st.sidebar.title("Economic Dashboard")
tab1, tab2, tab3 = st.tabs(["Macro", "Markets", "Housing Market"])

# ---------------- MACRO TAB ---------------- #
with tab1:
    st.title("Australia Macro Dashboard")

    # Example macro section (RBA data)
    h3 = load_rba_table("H3")
    activity_map = {
        "GISSRTCYP": "Year-ended retail sales growth",
        "GISPSDA":   "Private dwelling approvals",
        "GISPSNBA":  "Private non-residential building approvals",
    }

    st.header("Monthly Activity Levels")
    stats = []
    for i, (col, label) in enumerate(activity_map.items()):
        if col in h3.columns:
            st.pyplot(line_fig(h3, col, label))
            text = calc_mom_yoy(h3, col, label)
            st.markdown(text)
            stats.append(text)

    st.markdown("**AI Summary:** " + explain_with_gpt("\n".join(stats), "Monthly Activity Levels"))

# ---------------- MARKETS TAB ---------------- #
with tab2:
    st.header("Markets Dashboard (Yahoo Finance)")

    def plot_yf(ticker, title, period="5y", freq="1mo"):
        data = yf.download(ticker, period=period, interval=freq)["Close"].dropna()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(data.index, data, label=title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        if len(data) > 13:
            latest, prev, yoy = float(data.iloc[-1]), float(data.iloc[-2]), float(data.iloc[-13])
            return f"{title} — MoM: {latest-prev:+.2f}, YoY: {latest-yoy:+.2f}"
        return f"{title}: insufficient data"

    fx_stats = []
    col1, col2 = st.columns(2)
    with col1: fx_stats.append(plot_yf("AUDUSD=X", "AUD/USD (FX rate)"))
    with col2: fx_stats.append(plot_yf("AUDGBP=X", "AUD/GBP (FX rate)"))
    st.markdown("**AI Summary (FX):** " + explain_with_gpt("\n".join(fx_stats), "Exchange Rates"))

    eq_stats = []
    col1, col2 = st.columns(2)
    with col1: eq_stats.append(plot_yf("^AXJO", "ASX200 Index"))
    with col2: eq_stats.append(plot_yf("^GSPC", "S&P500 Index"))
    st.markdown("**AI Summary (Equities):** " + explain_with_gpt("\n".join(eq_stats), "Equity Indices"))

# ---------------- HOUSING TAB ---------------- #
with tab3:
    housing_market_section()

# ---------------- FOOTER ---------------- #
st.caption("Data source: RBA Statistical Tables, Yahoo Finance, CoreLogic RP Data.")
