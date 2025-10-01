import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from openai import OpenAI

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

st.markdown("**Summary:** " + explain_with_gpt("\n".join(activity_stats), "Monthly Activity Levels"))

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

st.markdown("**Summary:** " + explain_with_gpt("\n".join(macro_stats), "Key Macro Metrics"))

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

inflation_stats = []
codes = [c for c in g2_map if c in g2.columns]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        with cols[j]:
            st.pyplot(line_fig(g2, code, g2_map[code]))
            change = calc_mom_yoy(g2, code, g2_map[code])
            st.markdown(change)
            inflation_stats.append(change)

st.markdown("**Summary:** " + explain_with_gpt("\n".join(inflation_stats), "Inflation Components"))

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

st.markdown("**Summary:** " + explain_with_gpt("\n".join(labour_stats), "Labour Market"))

# =========================================================
# 5) Household finance
# =========================================================
st.header("Household finance")

e1 = clamp_period(load_rba_table("E1"))
e2 = clamp_period(load_rba_table("E2"))
e13 = clamp_period(load_rba_table("E13"))
f6 = clamp_period(load_rba_table("F6"))
d1 = clamp_period(load_rba_table("D1"))

# Derived ratios
merged = pd.merge(e1, e2, on="Date", how="outer").sort_values("Date")
for col in ["BSPNSHUFAD","BSPNSHUA","BSPNSHUL"]:
    if col not in merged.columns:
        merged[col] = pd.NA
merged["Savings_%_Assets"] = (merged["BSPNSHUFAD"] / merged["BSPNSHUA"]) * 100
merged["Savings_%_Liabilities"] = (merged["BSPNSHUFAD"] / merged["BSPNSHUL"]) * 100
merged = clamp_period(merged)

finance_map = {
    "Savings_%_Assets": "Household savings as % of assets",
    "Savings_%_Liabilities": "Household savings as % of liabilities",
    "BHFDDIH": "Housing debt to income",
    "BHFDA": "Household debt to assets",
    "LPHTSPRI": "Loan repayments to income",
    "FLRHOOTA": "Lending rates (all rates)",
    "FLRHOOVA": "Lending rates (variable rates)",
    "FLRHOLA": "Lending rates (LVR ≤81%)",
    "FLRHOLB": "Lending rates (LVR >81%)",
    "FLRHOVA": "Lending rates (≤600k)",
    "FLRHOVB": "Lending rates (600–1m)",
    "FLRHOVC": "Lending rates (1m+)",
    "DGFACOHM": "12-month housing credit growth",
    "DGFACBNF12": "12-month business credit growth",
}

finance_stats = []

# --- Savings ---
st.subheader("Savings")
for code in ["Savings_%_Assets","Savings_%_Liabilities"]:
    if code in merged.columns:
        cols = st.columns(2)
        with cols[0]:
            st.pyplot(line_fig(merged, code, finance_map[code]))
        with cols[1]:
            change = calc_mom_yoy(merged, code, finance_map[code])
            st.markdown(change)
            finance_stats.append(change)

# --- Debt ---
st.subheader("Debt")
for code in ["BHFDDIH","BHFDA","LPHTSPRI"]:
    df = e2 if code in e2.columns else e13
    if code in df.columns:
        cols = st.columns(2)
        with cols[0]:
            st.pyplot(line_fig(df, code, finance_map[code]))
        with cols[1]:
            change = calc_mom_yoy(df, code, finance_map[code])
            st.markdown(change)
            finance_stats.append(change)

# --- Lending Rates ---
st.subheader("Lending Rates")
for code in ["FLRHOOTA","FLRHOOVA","FLRHOLA","FLRHOLB","FLRHOVA","FLRHOVB","FLRHOVC"]:
    if code in f6.columns:
        cols = st.columns(2)
        with cols[0]:
            st.pyplot(line_fig(f6, code, finance_map[code]))
        with cols[1]:
            change = calc_mom_yoy(f6, code, finance_map[code])
            st.markdown(change)
            finance_stats.append(change)

# --- Credit Growth ---
st.subheader("Credit Growth")
for code in ["DGFACOHM","DGFACBNF12"]:
    if code in d1.columns:
        cols = st.columns(2)
        with cols[0]:
            st.pyplot(line_fig(d1, code, finance_map[code]))
        with cols[1]:
            change = calc_mom_yoy(d1, code, finance_map[code])
            st.markdown(change)
            finance_stats.append(change)

st.markdown("**Summary:** " + explain_with_gpt("\n".join(finance_stats), "Household Finance"))

st.caption("Data source: Reserve Bank of Australia Statistical Tables. Figures computed from public XLSX files at run-time.")
