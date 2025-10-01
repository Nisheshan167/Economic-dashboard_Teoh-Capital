import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

st.set_page_config(page_title="AU Macro & Markets Dashboard", layout="wide")

# ---------- Helpers ----------
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
    fig, ax = plt.subplots(figsize=(8,4))
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

# ---------- Sidebar filters ----------
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

codes = [c for c in activity_map if c in h3.columns]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        with cols[j]:
            st.pyplot(line_fig(h3, code, activity_map[code]))
            st.markdown(calc_mom_yoy(h3, code, activity_map[code]))

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
codes = [c for c in macro_map if c and c in (h1.columns.tolist() + g1.columns.tolist() + f1.columns.tolist())]

dfs = {**h1, **g1, **f1}  # merge dict of frames for lookup
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        df = h1 if code in h1.columns else g1 if code in g1.columns else f1
        with cols[j]:
            st.pyplot(line_fig(df, code, macro_map[code]))
            st.markdown(calc_mom_yoy(df, code, macro_map[code]))

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

codes = [c for c in g2_map if c in g2.columns]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        with cols[j]:
            st.pyplot(line_fig(g2, code, g2_map[code]))
            st.markdown(calc_mom_yoy(g2, code, g2_map[code]))

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

codes = [c for c in labour_map if c in (h5.columns.tolist() + h4.columns.tolist())]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        df = h5 if code in h5.columns else h4
        with cols[j]:
            st.pyplot(line_fig(df, code, labour_map[code]))
            st.markdown(calc_mom_yoy(df, code, labour_map[code]))

# =========================================================
# 5) Household finance (E1, E2, E13, F6, D1)
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

# Plot household finance in pairs
dfs = {"Savings_%_Assets": merged, "Savings_%_Liabilities": merged,
       "BHFDDIH": e2, "BHFDA": e2, "LPHTSPRI": e13,
       "DGFACOHM": d1, "DGFACBNF12": d1}
for code in ["Savings_%_Assets","Savings_%_Liabilities","BHFDDIH","BHFDA","LPHTSPRI",
             "FLRHOOTA","FLRHOOVA","FLRHOLA","FLRHOLB","FLRHOVA","FLRHOVB","FLRHOVC",
             "DGFACOHM","DGFACBNF12"]:
    df = dfs.get(code, f6 if code in f6.columns else None)
    if df is not None and code in df.columns:
        st.pyplot(line_fig(df, code, finance_map[code]))
        st.markdown(calc_mom_yoy(df, code, finance_map[code]))

st.caption("Data source: Reserve Bank of Australia Statistical Tables. Figures computed from public XLSX files at run-time.")
