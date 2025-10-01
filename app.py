import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

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

@st.cache_data(ttl=60*60)
def load_rba_table(code: str) -> pd.DataFrame:
    url = RBA_URLS[code]
    # Simple sanity fetch (lets Streamlit cache the bytes)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_excel(pd.io.common.BytesIO(r.content), skiprows=10)
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

def latest_change_str(series: pd.Series) -> str:
    series = series.dropna()
    if len(series) < 2:
        return "Insufficient data."
    start, end = series.iloc[0], series.iloc[-1]
    pct = ((end / start) - 1) * 100 if start not in [0, None] else float("nan")
    return f"{start:,.2f} → {end:,.2f} ({pct:+.2f}%)."

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
min_year = 1990
max_year = pd.Timestamp.today().year
start_year, end_year = st.sidebar.slider(
    "Time range",
    min_value=min_year, max_value=max_year,
    value=(2015, max_year)
)

def clamp_period(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)].copy()

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

# Row 1: 3 charts side-by-side
colsA = st.columns(3)
for i, code in enumerate(["GISSRTCYP","GISPSDA","GISPSNBA"]):
    if code in h3.columns:
        with colsA[i]:
            st.pyplot(line_fig(h3, code, activity_map[code], "Percent / Index"))

# Row 2: 2 charts side-by-side
colsB = st.columns(2)
for i, code in enumerate(["GICWMICS","GICNBC"]):
    if code in h3.columns:
        with colsB[i]:
            st.pyplot(line_fig(h3, code, activity_map[code], "Index / Dev."))

# Insight
parts = []
for code in activity_map:
    if code in h3.columns:
        parts.append(f"**{activity_map[code]}** {latest_change_str(h3[code])}")
st.markdown(
    f"**Analytical statement:** {', '.join(parts)} "
    f"({start_year}–{end_year})."
)

# =========================================================
# 2) Key macro metrics (H1, G1, F1)
# =========================================================
st.header("Key macro metrics")

h1 = clamp_period(load_rba_table("H1"))
g1 = clamp_period(load_rba_table("G1"))
f1 = clamp_period(load_rba_table("F1"))

# Cash Rate column may vary across vintages
cash_col = pick_first_existing(f1, ["FIRMMCRTD","FIRMMCRT","FIRMMCR","FIRMMCRTDV"])

cols = st.columns(3)
if "GGDPCVGDPY" in h1.columns:
    cols[0].pyplot(line_fig(h1, "GGDPCVGDPY", "Year-ended real GDP growth", "Percent"))
if "GCPIAGYP" in g1.columns:
    cols[1].pyplot(line_fig(g1, "GCPIAGYP", "Year-ended inflation (CPI)", "Percent"))
if cash_col:
    cols[2].pyplot(line_fig(f1, cash_col, "Cash Rate Target", "Percent"))

st.markdown(
    "**Analytical statement:** "
    f"GDP {latest_change_str(h1.get('GGDPCVGDPY', pd.Series(dtype=float)))}; "
    f"Inflation {latest_change_str(g1.get('GCPIAGYP', pd.Series(dtype=float)))}; "
    f"Cash Rate {latest_change_str(f1.get(cash_col, pd.Series(dtype=float)))} "
    f"({start_year}–{end_year})."
)

# =========================================================
# 3) Inflation detail (G2) — bar snapshot + option time series
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

# Latest valid row snapshot bar
valid_cols = [c for c in g2_map if c in g2.columns]
g2_nonan = g2.dropna(subset=valid_cols)
if not g2_nonan.empty:
    latest = g2_nonan.iloc[-1]
    date_str = latest["Date"].strftime("%b %Y")
    values = [latest[c] for c in valid_cols]
    labels = [g2_map[c] for c in valid_cols]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(labels, values)
    ax.set_title(f"CPI components (YoY %) — {date_str}")
    ax.set_ylabel("Percent")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    st.pyplot(fig)

st.markdown(
    "**Analytical statement:** Components with highest YoY in the latest print lead the inflation profile; "
    "note relative contributions from Housing and Transport where applicable."
)

# =========================================================
# 4) Labour market (H5, H4)
# =========================================================
st.header("Labour market")

h5 = clamp_period(load_rba_table("H5"))
h4 = clamp_period(load_rba_table("H4"))

cols = st.columns(3)
if "GLFSURSA" in h5.columns:
    cols[0].pyplot(line_fig(h5, "GLFSURSA", "Unemployment rate", "Percent"))
if "GWPIYP" in h4.columns:
    cols[1].pyplot(line_fig(h4, "GWPIYP", "Year-ended wage growth", "Percent"))
if "GLFSEPTSYP" in h5.columns:
    cols[2].pyplot(line_fig(h5, "GLFSEPTSYP", "Year-ended employment growth", "Percent"))

st.markdown(
    "**Analytical statement:** "
    f"Unemployment {latest_change_str(h5.get('GLFSURSA', pd.Series(dtype=float)))}; "
    f"Wage growth {latest_change_str(h4.get('GWPIYP', pd.Series(dtype=float)))}; "
    f"Employment growth {latest_change_str(h5.get('GLFSEPTSYP', pd.Series(dtype=float)))} "
    f"({start_year}–{end_year})."
)

# =========================================================
# 5) Household finance (E1, E2, E13, F6, D1)
# =========================================================
st.header("Household finance")

e1 = clamp_period(load_rba_table("E1"))
e2 = clamp_period(load_rba_table("E2"))
e13 = clamp_period(load_rba_table("E13"))
f6 = clamp_period(load_rba_table("F6"))
d1 = clamp_period(load_rba_table("D1"))

# Merge E1+E2 for derived ratios
merged = pd.merge(e1, e2, on="Date", how="outer").sort_values("Date")
for col in ["BSPNSHUFAD","BSPNSHUA","BSPNSHUL"]:
    if col not in merged.columns:
        merged[col] = pd.NA
merged["Savings_%_Assets"] = (merged["BSPNSHUFAD"] / merged["BSPNSHUA"]) * 100
merged["Savings_%_Liabilities"] = (merged["BSPNSHUFAD"] / merged["BSPNSHUL"]) * 100
merged = clamp_period(merged)

# Ratios: two separate charts
c1, c2 = st.columns(2)
if merged["Savings_%_Assets"].notna().any():
    c1.pyplot(line_fig(merged, "Savings_%_Assets", "Household savings as % of assets", "Percent"))
if merged["Savings_%_Liabilities"].notna().any():
    c2.pyplot(line_fig(merged, "Savings_%_Liabilities", "Household savings as % of liabilities", "Percent"))

# Debt metrics (E2, E13)
row = st.columns(3)
if "BHFDDIH" in e2.columns:
    row[0].pyplot(line_fig(e2, "BHFDDIH", "Housing debt to income", "Ratio / Index"))
if "BHFDA" in e2.columns:
    row[1].pyplot(line_fig(e2, "BHFDA", "Household debt to assets", "Ratio / Percent"))
if "LPHTSPRI" in e13.columns:
    row[2].pyplot(line_fig(e13, "LPHTSPRI", "Loan repayments to income", "Percent"))

# Lending rates (each separate)
lending_series = {
    "FLRHOOTA": "Lending rates (all rates)",
    "FLRHOOVA": "Lending rates (variable rates)",
    "FLRHOLA":  "Lending rates (LVR ≤81%)",
    "FLRHOLB":  "Lending rates (LVR >81%)",
    "FLRHOVA":  "Lending rates (≤600k)",
    "FLRHOVB":  "Lending rates (600–1m)",
    "FLRHOVC":  "Lending rates (1m+)",
}
# Grid them 3 per row
codes = [c for c in lending_series if c in f6.columns]
for i in range(0, len(codes), 3):
    cols = st.columns(3)
    for j, code in enumerate(codes[i:i+3]):
        cols[j].pyplot(line_fig(f6, code, lending_series[code], "Percent"))

# Credit growth (two separate)
row = st.columns(2)
if "DGFACOHM" in d1.columns:
    row[0].pyplot(line_fig(d1, "DGFACOHM", "12-month housing credit growth", "Percent"))
if "DGFACBNF12" in d1.columns:
    row[1].pyplot(line_fig(d1, "DGFACBNF12", "12-month business credit growth", "Percent"))

# Insight
parts = []
for code, name in [("BHFDDIH","Housing debt to income"),
                   ("BHFDA","Household debt to assets"),
                   ("LPHTSPRI","Loan repayments to income")]:
    series = (e2 if code in e2.columns else e13).get(code, pd.Series(dtype=float))
    if not series.empty:
        parts.append(f"**{name}** {latest_change_str(series.dropna())}")
st.markdown(
    "**Analytical statement:** "
    f"Household savings ratios and debt metrics indicate evolving balance-sheet resilience; "
    + ", ".join(parts) + f" ({start_year}–{end_year})."
)

st.caption("Data source: Reserve Bank of Australia Statistical Tables. Figures computed from public XLSX files at run-time.")
