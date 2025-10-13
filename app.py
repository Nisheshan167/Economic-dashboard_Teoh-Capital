import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from openai import OpenAI
import yfinance as yf
import numpy as np

st.set_page_config(page_title="AU Macro & Markets Dashboard", layout="wide")

# ---------- OpenAI client ----------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def explain_with_gpt(indicator_stats, umbrella_name):
    if not indicator_stats:
        return "No data available to summarize."
    prompt = f"""
    You are an economic analyst. Based on the following indicators for {umbrella_name}, 
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

# ---------- RBA Links ----------
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

# ---------- Chart Helpers ----------
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
    prev = series.iloc[-2] if len(series) > 1 else None
    yoy = series.iloc[-13] if len(series) > 12 else None
    mom_change = (latest - prev) if prev is not None else None
    yoy_change = (latest - yoy) if yoy is not None else None
    return f"{label} ({latest_date}) — MoM: {mom_change:+.2f}, YoY: {yoy_change:+.2f}"

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)

# ---------- Sidebar ----------
st.sidebar.header("Filters")
default_start = pd.to_datetime("2015-01-01")
default_end = pd.to_datetime("today")

start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", default_end)

def clamp_period(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))].copy()

st.title("Australia Macro Dashboard")

# =========================================================
# 1) Monthly activity levels (H3)
# =========================================================
st.header("Monthly activity levels")

h3 = clamp_period(load_rba_table("H3"))
activity_map = {
    "GISSRTCYP": "Year-ended retail sales growth",
    "GISPSDA":   "Private dwelling approvals",
    "GISPSNBA":  "Private non-residential building approvals",
    "GICWMICS":  "Consumer sentiment",
    "GICNBC":    "Business conditions",
}

activity_stats = []
codes = [c for c in activity_map if c in h3.columns]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        with cols[j]:
            st.pyplot(line_fig(h3, code, activity_map[code]))
            change = calc_mom_yoy(h3, code, activity_map[code])
            st.markdown(change)
            activity_stats.append(change)

st.markdown("**AI Summary:** " + explain_with_gpt("\n".join(activity_stats), "Monthly Activity Levels"))

# =========================================================
# 2) Key macro metrics (H1, G1, F1)
# =========================================================
st.header("Key macro metrics")

h1 = clamp_period(load_rba_table("H1"))
g1 = clamp_period(load_rba_table("G1"))
f1 = clamp_period(load_rba_table("F1"))
cash_col = pick_first_existing(f1, ["FIRMMCRTD","FIRMMCRT","FIRMMCR","FIRMMCRTDV"])

macro_map = {
    "GGDPCVGDPY": "Year-ended real GDP growth",
    "GCPIAGYP":   "Year-ended inflation (CPI)",
    cash_col:     "Cash Rate Target" if cash_col else None,
}

macro_stats = []
codes = [c for c in macro_map if c]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        df = h1 if code in h1.columns else g1 if code in g1.columns else f1
        with cols[j]:
            st.pyplot(line_fig(df, code, macro_map[code]))
            change = calc_mom_yoy(df, code, macro_map[code])
            st.markdown(change)
            macro_stats.append(change)

st.markdown("**AI Summary:** " + explain_with_gpt("\n".join(macro_stats), "Key Macro Metrics"))

# =========================================================
# 3) Inflation detail (G2)
# =========================================================
st.header("Inflation (CPI components)")

g2 = clamp_period(load_rba_table("G2"))
g2_map = {
    "GCPIFYP": "Food & non-alcoholic beverages",
    "GCPIATYP": "Alcohol & tobacco",
    "GCPICFYP": "Clothing & footwear",
    "GCPIHOYP": "Housing",
    "GCPIHCSYP":"Furnishings, hh equipment & services",
    "GCPIHEYP": "Health",
    "GCPITYP": "Transport",
    "GCPICYP": "Communication",
    "GCPIRYP": "Recreation & culture",
    "GCPIEYP": "Education",
    "GCPIFISYP":"Insurance & financial services",
}

valid_cols = [c for c in g2_map if c in g2.columns]
g2_nonan = g2.dropna(subset=valid_cols)

inflation_stats = []
if not g2_nonan.empty:
    latest = g2_nonan.iloc[-1]
    date_str = latest["Date"].strftime("%b %Y")
    values = [float(latest[c]) for c in valid_cols]
    labels = [g2_map[c] for c in valid_cols]

    # --- One bar chart for all components ---
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(labels, values)
    ax.set_title(f"CPI Components (YoY %) — {date_str}")
    ax.set_ylabel("Percent")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    st.pyplot(fig)

    # Build stats for AI summary
    for c in valid_cols:
        inflation_stats.append(calc_mom_yoy(g2, c, g2_map[c]))

st.markdown("**AI Summary:** " + explain_with_gpt("\n".join(inflation_stats), "Inflation Components"))

# =========================================================
# 4) Labour market (H5, H4)
# =========================================================
st.header("Labour market")

h5 = clamp_period(load_rba_table("H5"))
h4 = clamp_period(load_rba_table("H4"))

labour_map = {
    "GLFSURSA": "Unemployment rate",
    "GWPIYP":   "Year-ended wage growth",
    "GLFSEPTSYP":"Year-ended employment growth",
}

labour_stats = []
codes = [c for c in labour_map if c in (h5.columns.tolist() + h4.columns.tolist())]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        df = h5 if code in h5.columns else h4
        with cols[j]:
            st.pyplot(line_fig(df, code, labour_map[code]))
            change = calc_mom_yoy(df, code, labour_map[code])
            st.markdown(change)
            labour_stats.append(change)

st.markdown("**AI Summary:** " + explain_with_gpt("\n".join(labour_stats), "Labour Market"))

# =========================================================
# 5) Household finance
# =========================================================
st.header("Household finance")

e1 = clamp_period(load_rba_table("E1"))
e2 = clamp_period(load_rba_table("E2"))
e13 = clamp_period(load_rba_table("E13"))
f6 = clamp_period(load_rba_table("F6"))
d1 = clamp_period(load_rba_table("D1"))

merged = pd.merge(e1, e2, on="Date", how="outer").sort_values("Date")
for col in ["BSPNSHUFAD","BSPNSHUA","BSPNSHUL"]:
    if col not in merged.columns:
        merged[col] = pd.NA
merged["Savings_%_Assets"] = (merged["BSPNSHUFAD"] / merged["BSPNSHUA"]) * 100
merged["Savings_%_Liabilities"] = (merged["BSPNSHUFAD"] / merged["BSPNSHUL"]) * 100
merged = clamp_period(merged)

finance_stats = []

# Savings
st.subheader("Savings")
cols = st.columns(2)
if merged["Savings_%_Assets"].notna().any():
    cols[0].pyplot(line_fig(merged, "Savings_%_Assets", "Household savings as % of assets", "Percent"))
    finance_stats.append(calc_mom_yoy(merged, "Savings_%_Assets", "Household savings as % of assets"))
if merged["Savings_%_Liabilities"].notna().any():
    cols[1].pyplot(line_fig(merged, "Savings_%_Liabilities", "Household savings as % of liabilities", "Percent"))
    finance_stats.append(calc_mom_yoy(merged, "Savings_%_Liabilities", "Household savings as % of liabilities"))

# Debt
st.subheader("Debt")
codes = [("BHFDDIH","Housing debt to income"),("BHFDA","Household debt to assets"),("LPHTSPRI","Loan repayments to income")]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, (code, label) in enumerate(codes[i:i+2]):
        df = e2 if code in e2.columns else e13
        if code in df.columns:
            cols[j].pyplot(line_fig(df, code, label))
            finance_stats.append(calc_mom_yoy(df, code, label))

# Lending Rates
st.subheader("Lending Rates")
lending_codes = [
    ("FLRHOOTA","Lending rates (all rates)"),
    ("FLRHOOVA","Lending rates (variable rates)"),
    ("FLRHOLA","Lending rates (LVR ≤81%)"),
    ("FLRHOLB","Lending rates (LVR >81%)"),
    ("FLRHOVA","Lending rates (≤600k)"),
    ("FLRHOVB","Lending rates (600–1m)"),
    ("FLRHOVC","Lending rates (1m+)"),
]
for i in range(0, len(lending_codes), 2):
    cols = st.columns(2)
    for j, (code,label) in enumerate(lending_codes[i:i+2]):
        if code in f6.columns:
            cols[j].pyplot(line_fig(f6, code, label))
            finance_stats.append(calc_mom_yoy(f6, code, label))

# Credit Growth
st.subheader("Credit Growth")
cols = st.columns(2)
if "DGFACOHM" in d1.columns:
    cols[0].pyplot(line_fig(d1, "DGFACOHM", "12-month housing credit growth", "Percent"))
    finance_stats.append(calc_mom_yoy(d1, "DGFACOHM", "12-month housing credit growth"))
if "DGFACBNF12" in d1.columns:
    cols[1].pyplot(line_fig(d1, "DGFACBNF12", "12-month business credit growth", "Percent"))
    finance_stats.append(calc_mom_yoy(d1, "DGFACBNF12", "12-month business credit growth"))

st.markdown("**AI Summary:** " + explain_with_gpt("\n".join(finance_stats), "Household Finance"))

# =========================================================
# 6) Markets (Yahoo Finance)
# =========================================================
st.header("Markets Dashboard (Yahoo Finance)")

def plot_yf(ticker, title, period="5y", freq="1mo"):
    data = yf.download(ticker, period=period, interval=freq)["Close"].dropna()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(data.index, data, label=title)
    ax.set_title(title)
    ax.set_ylabel("Index / FX")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Ensure we only work with scalar floats
    if len(data) > 13:
        latest, prev, yoy = float(data.iloc[-1]), float(data.iloc[-2]), float(data.iloc[-13])
        if not np.isnan(latest):
            return f"{title} ({data.index[-1].strftime('%b %Y')}) — MoM: {latest-prev:+.2f}, YoY: {latest-yoy:+.2f}"
    return f"{title}: insufficient data"


# FX
st.subheader("Exchange Rates")
fx_stats = []
col1, col2 = st.columns(2)
with col1: fx_stats.append(plot_yf("AUDUSD=X", "AUD/USD (FX rate)"))
with col2: fx_stats.append(plot_yf("AUDGBP=X", "AUD/GBP (FX rate)"))
st.markdown("**AI Summary (FX):** " + explain_with_gpt("\n".join(fx_stats), "Exchange Rates"))

# Equities
st.subheader("Equity Indices")
eq_stats = []
col1, col2 = st.columns(2)
with col1: eq_stats.append(plot_yf("^AXJO", "ASX200 Index"))
with col2: eq_stats.append(plot_yf("^GSPC", "S&P500 Index"))
st.markdown("**AI Summary (Equities):** " + explain_with_gpt("\n".join(eq_stats), "Equity Indices"))

# YoY Change side by side
st.subheader("YoY Change in Equity Indices")
yoy_stats = []
col1, col2 = st.columns(2)

with col1:
    data = yf.download("^AXJO", period="5y", interval="1mo")["Close"].dropna()
    yoy = data.pct_change(periods=12) * 100
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(yoy.index, yoy, label="ASX200 YoY Change (%)")
    ax.set_title("ASX200 YoY Change")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    if len(yoy) > 0:
        latest_val = float(yoy.iloc[-1])  # force to scalar
        if not np.isnan(latest_val):
            yoy_stats.append(f"ASX200 latest YoY change: {latest_val:+.2f}%")

with col2:
    data = yf.download("^GSPC", period="5y", interval="1mo")["Close"].dropna()
    yoy = data.pct_change(periods=12) * 100
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(yoy.index, yoy, label="S&P500 YoY Change (%)")
    ax.set_title("S&P500 YoY Change")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    if len(yoy) > 0:
        latest_val = float(yoy.iloc[-1])  # force to scalar
        if not np.isnan(latest_val):
            yoy_stats.append(f"S&P500 latest YoY change: {latest_val:+.2f}%")

st.markdown("**AI Summary (YoY Changes):** " + explain_with_gpt("\n".join(yoy_stats), "YoY Index Changes"))

st.markdown("**AI Summary (YoY Changes):** " + explain_with_gpt("\n".join(yoy_stats), "YoY Index Changes"))

# =========================================================
# 🏠 CoreLogic Daily Home Value Index
# =========================================================
st.header("🏠 CoreLogic Daily Home Value Index")

try:
    # Load Excel directly
    df = pd.read_excel("corelogic_daily_index.xlsx")

    # Ensure proper datetime format
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # List the city columns exactly as in your file
    cities = [
        "Sydney (SYDD)",
        "Melbourne (MELD)",
        "Brisbane (inc Gold Coast) (BRID)",
        "Adelaide (ADED)",
        "Perth (PERD)",
        "5 capital city aggregate (AUSD)"
    ]

    # Filter to available columns
    available = [c for c in cities if c in df.columns]

    # Plot the line chart
    fig = px.line(df, x="Date", y=available, title="Daily Home Value Index Trends")
    st.plotly_chart(fig, use_container_width=True)

    # Year-over-year (or total period) change summary
    yoy_stats = []
    for c in available:
        if len(df[c].dropna()) > 1:
            change = (df[c].iloc[-1] / df[c].iloc[0] - 1) * 100
            yoy_stats.append(f"{c}: {change:.2f}% change over period")

    # Display stats + AI summary
    st.markdown("**AI Summary (CoreLogic):** " + explain_with_gpt("\n".join(yoy_stats), "CoreLogic Home Value Index"))

except Exception as e:
    st.warning(f"Unable to load CoreLogic data: {e}")


st.caption("Data source: RBA Statistical Tables, Yahoo Finance. Figures computed from public APIs and XLSX files at run-time.")
