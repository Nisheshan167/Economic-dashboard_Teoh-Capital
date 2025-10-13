import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from openai import OpenAI
import yfinance as yf
import numpy as np
import plotly.express as px

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
if valid_cols:
    latest = g2.iloc[-1]
    date_str = latest["Date"].strftime("%b %Y")
    values = [float(latest[c]) for c in valid_cols]
    labels = [g2_map[c] for c in valid_cols]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(labels, values)
    ax.set_title(f"CPI Components (YoY %) — {date_str}")
    ax.set_ylabel("Percent")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    st.pyplot(fig)
st.markdown("**AI Summary:** " + explain_with_gpt("CPI component trends", "Inflation Components"))

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
for code, label in labour_map.items():
    df = h5 if code in h5.columns else h4
    if code in df.columns:
        st.pyplot(line_fig(df, code, label))
st.markdown("**AI Summary:** " + explain_with_gpt("Unemployment, wage growth, employment trends", "Labour Market"))

# =========================================================
# 5) CoreLogic Daily Home Value Index
# =========================================================
st.header("🏠 CoreLogic Daily Home Value Index")
try:
    df_cl = pd.read_excel("data/corelogic_daily_index.xlsx")
    date_col = next((c for c in df_cl.columns if "date" in c.lower()), None)
    df_cl[date_col] = pd.to_datetime(df_cl[date_col], errors="coerce")
    cities = [c for c in df_cl.columns if "(" in c]
    fig = px.line(df_cl, x=date_col, y=cities, title="Daily Home Value Index Trends")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Unable to load CoreLogic data: {e}")

# =========================================================
# 6) Population & Migration (ABS)
# =========================================================
st.header("👥 Population Growth & Net Overseas Migration (ABS)")

@st.cache_data(ttl=86400)
def load_abs_population():
    base = "https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/latest-release/downloads/"
    total_pop = base + "31010do002_202406.csv"   # Table 2: Estimated Resident Population
    nom = base + "31010do004_202406.csv"         # Table 4: Net Overseas Migration
    pop_df = pd.read_csv(total_pop, skiprows=8)
    pop_df = pop_df.rename(columns={pop_df.columns[0]: "Region"}).dropna(subset=["Region"])
    nom_df = pd.read_csv(nom, skiprows=8)
    nom_df = nom_df.rename(columns={nom_df.columns[0]: "Region"}).dropna(subset=["Region"])
    return pop_df, nom_df

try:
    pop_df, nom_df = load_abs_population()
    st.subheader("Estimated Resident Population by State")
    st.dataframe(pop_df.head(10))
    st.subheader("Net Overseas Migration by State")
    st.dataframe(nom_df.head(10))
    st.markdown("**AI Summary:** " + explain_with_gpt("Population and migration updates from ABS", "ABS Population Data"))
except Exception as e:
    st.warning(f"ABS data unavailable: {e}")

# =========================================================
# 7) Global Policy Rates (US, EU, Japan, UK, Canada)
# =========================================================
st.header("🌍 Global Central Bank Policy Rates")

@st.cache_data(ttl=86400)
def load_global_rates():
    urls = {
        "United States (Fed Funds Rate)": "https://fred.stlouisfed.org/data/FEDFUNDS.csv",
        "Euro Area (ECB Main Refinancing Rate)": "https://fred.stlouisfed.org/data/ECBMAIN.csv",
        "Japan (BOJ Policy Rate)": "https://fred.stlouisfed.org/data/IRJPAN.csv",
        "United Kingdom (BOE Bank Rate)": "https://fred.stlouisfed.org/data/IRGBUS.csv",
        "Canada (BoC Overnight Rate)": "https://fred.stlouisfed.org/data/INTDSRCANM193N.csv"
    }
    dfs = {}
    for name, url in urls.items():
        try:
            df = pd.read_csv(url)
            df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "Rate"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
            df = df.dropna(subset=["Rate"])
            df = df[df["Date"] >= pd.to_datetime("2010-01-01")]
            dfs[name] = df
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    return dfs

global_rates = load_global_rates()
for name, df in global_rates.items():
    st.pyplot(line_fig(df, "Rate", name, "%"))
st.markdown("**AI Summary:** " + explain_with_gpt("Global interest rate movements", "Global Policy Rates"))

# =========================================================
# 8) Vanguard Global Market ETFs
# =========================================================
st.header("📈 Vanguard Global Market Indices")

def plot_vanguard(ticker, title, period="5y", freq="1mo"):
    data = yf.download(ticker, period=period, interval=freq)["Close"].dropna()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(data.index, data, label=title)
    ax.set_title(title)
    ax.set_ylabel("Price (USD/AUD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

vanguard_tickers = {
    "VTI (US Total Market)": "VTI",
    "VGK (Europe ETF)": "VGK",
    "EWJ (Japan ETF)": "EWJ",
    "VAS (Australia ETF)": "VAS.AX",
    "BNDX (Global Bonds ex-US)": "BNDX",
}
for label, ticker in vanguard_tickers.items():
    plot_vanguard(ticker, label)

st.caption("Data sources: RBA, CoreLogic, ABS, FRED, Yahoo Finance. Figures computed automatically at run-time.")
