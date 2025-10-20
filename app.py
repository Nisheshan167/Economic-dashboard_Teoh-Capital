import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from openai import OpenAI
import yfinance as yf
import numpy as np
import plotly.express as px

from fpdf import FPDF
import tempfile
import os
report_sections = []

def plotly_to_matplotlib(fig_px):
    """Convert a simple Plotly line figure to Matplotlib (for PDF export without Kaleido)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for trace in fig_px.data:
        # Some traces might not be lines or may lack a name
        x = getattr(trace, "x", None)
        y = getattr(trace, "y", None)
        if x is None or y is None:
            continue
        name = getattr(trace, "name", None) or ""
        ax.plot(x, y, label=name if name is not None else "")

    title = getattr(fig_px.layout, "title", None)
    if title and getattr(title, "text", None):
        ax.set_title(title.text)

    # Only show legend if we actually added labels
    handles, labels = ax.get_legend_handles_labels()
    if any(lbl for lbl in labels):
        ax.legend()

    ax.grid(True)
    fig.tight_layout()
    return fig


def generate_pdf(report_title: str, sections: list[dict]) -> bytes:
    """
    sections: list of dicts containing {
        'header': str,
        'text': str,
        'figs': list[plt.Figure | plotly.Figure]
    }
    """
    from fpdf import FPDF
    import tempfile, os

    def ascii_sanitize(s: str) -> str:
        repl = {
            "–": "-", "—": "-", "−": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
            "…": "...", "•": "-", "°": " deg", "×": "x", "€": "EUR", "£": "GBP",
            "→": "->", "←": "<-", "↑": "^", "↓": "v", "±": "+/-",
            "≤": "<=", "≥": ">=",
            "🏠": "Home ", "🌍": "Global ", "🇦🇺": "Australia ",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        return s.encode("latin-1", "ignore").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Font setup (Unicode safe if DejaVuSans.ttf present)
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    unicode_font = False
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        unicode_font = True

    # --- Title ---
    if unicode_font:
        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 8, report_title, ln=True, align="C")
    else:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, ascii_sanitize(report_title), ln=True, align="C")
    pdf.ln(5)

    # --- Sections ---
    for section in sections:
        header = section.get("header", "")
        body = section.get("text", "")
        figs = section.get("figs", [])

        # Header
        if unicode_font:
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 7, header, ln=True)
        else:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, ascii_sanitize(header), ln=True)

        # Compact body (smaller font & tighter spacing)
        if unicode_font:
            pdf.set_font("DejaVu", "", 9)
            pdf.multi_cell(0, 5, body)
        else:
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 5, ascii_sanitize(body))
        pdf.ln(4)

        # Figures
        for fig in figs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                ok = False
                try:
                    fig.savefig(tmpfile.name, bbox_inches="tight")
                    ok = True
                except AttributeError:
                    try:
                        fig.write_image(tmpfile.name, format="png")
                        ok = True
                    except Exception:
                        ok = False

                if ok:
                    try:
                        pdf.image(tmpfile.name, w=170)
                    except Exception:
                        pdf.set_font("Arial", "I", 9)
                        pdf.multi_cell(0, 5, "(Image could not be embedded)")
                os.remove(tmpfile.name)
            pdf.ln(6)

    return pdf.output(dest="S").encode("latin-1")


    # Important: with legacy FPDF (latin-1 only),
    # internal pages are encoded as latin-1. We've sanitized text above when unicode font isn't available.
    return pdf.output(dest="S").encode("latin-1" if not unicode_font_available else "latin-1")
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def combine_side_by_side(fig1, fig2, title="Comparison"):
    """
    Combine two Matplotlib figures side by side and keep correct datetime scaling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left chart ---
    for line in fig1.axes[0].get_lines():
        x, y = line.get_xdata(), line.get_ydata()
        axes[0].plot(x, y, label=line.get_label(), linewidth=1.8)
    axes[0].set_title(fig1.axes[0].get_title())
    axes[0].set_ylabel(fig1.axes[0].get_ylabel() or "")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Right chart ---
    for line in fig2.axes[0].get_lines():
        x, y = line.get_xdata(), line.get_ydata()
        axes[1].plot(x, y, label=line.get_label(), linewidth=1.8)
    axes[1].set_title(fig2.axes[0].get_title())
    axes[1].set_ylabel(fig2.axes[0].get_ylabel() or "")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(title, fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


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

# keys are the canonical “codes”, values are the human labels
activity_map = {
    "GISSRTCYP": "Year-ended retail sales growth",
    "GISPSDA":   "Private dwelling approvals",
    "GISPSNBA":  "Private non-residential building approvals",
    "GICWMICS":  "Consumer sentiment",
    "GICNBC":    "Business conditions",
}

activity_stats = []
activity_figs = []

# Build a resolved list of (actual_column_name, label) using fuzzy matching
resolved = []
for key, label in activity_map.items():
    match = next((col for col in h3.columns if key.lower() in str(col).lower()), None)
    if match:
        resolved.append((match, label))

if not resolved:
    st.warning("No matching H3 columns were found in the downloaded table.")
else:
    # Show charts in a 2-column grid
    for i in range(0, len(resolved), 2):
        cols = st.columns(2)
        for j, (colname, label) in enumerate(resolved[i:i+2]):
            with cols[j]:
                fig = line_fig(h3, colname, label)
                st.pyplot(fig)
                change = calc_mom_yoy(h3, colname, label)
                st.markdown(change)
                activity_stats.append(change)
                activity_figs.append(fig)

    activity_summary = explain_with_gpt("\n".join(activity_stats), "Monthly Activity Levels")
    st.markdown("**AI Summary:** " + activity_summary)

    # --- Combine figures pairwise for PDF export ---
    combined_activity_figs = []
    for i in range(0, len(activity_figs), 2):
        if i + 1 < len(activity_figs):
            # use the resolved labels for the title
            left_label  = resolved[i][1]
            right_label = resolved[i+1][1]
            combined_fig = combine_side_by_side(
                activity_figs[i],
                activity_figs[i + 1],
                title=f"Comparison: {left_label} vs {right_label}"
            )
            combined_activity_figs.append(combined_fig)
        else:
            combined_activity_figs.append(activity_figs[i])

    # Add combined pairs to report
    report_sections.append({
        "header": "Monthly Activity Levels",
        "text": "\n".join(activity_stats) + "\n\nAI Summary: " + activity_summary,
        "figs": combined_activity_figs
    })



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
macro_figs = []

codes = [c for c in macro_map if c]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, code in enumerate(codes[i:i+2]):
        df = h1 if code in h1.columns else g1 if code in g1.columns else f1
        with cols[j]:
            fig = line_fig(df, code, macro_map[code])
            st.pyplot(fig)
            change = calc_mom_yoy(df, code, macro_map[code])
            st.markdown(change)
            macro_stats.append(change)
            macro_figs.append(fig)

macro_summary = explain_with_gpt("\n".join(macro_stats), "Key Macro Metrics")
st.markdown("**AI Summary:** " + macro_summary)

# Add to report_sections
combined_macro_figs = []
for i in range(0, len(macro_figs), 2):
    if i + 1 < len(macro_figs):
        combined_macro_figs.append(
            combine_side_by_side(macro_figs[i], macro_figs[i + 1], title="Key Macro Comparison")
        )
    else:
        combined_macro_figs.append(macro_figs[i])

report_sections.append({
    "header": "Key Macro Metrics",
    "text": "\n".join(macro_stats) + "\n\nAI Summary: " + macro_summary,
    "figs": combined_macro_figs
})

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
finance_figs = []

# ---------- Savings ----------
st.subheader("Savings")
cols = st.columns(2)
if merged["Savings_%_Assets"].notna().any():
    fig = line_fig(merged, "Savings_%_Assets", "Household savings as % of assets", "Percent")
    cols[0].pyplot(fig)
    finance_figs.append(fig)
    finance_stats.append(calc_mom_yoy(merged, "Savings_%_Assets", "Household savings as % of assets"))

if merged["Savings_%_Liabilities"].notna().any():
    fig = line_fig(merged, "Savings_%_Liabilities", "Household savings as % of liabilities", "Percent")
    cols[1].pyplot(fig)
    finance_figs.append(fig)
    finance_stats.append(calc_mom_yoy(merged, "Savings_%_Liabilities", "Household savings as % of liabilities"))

# ---------- Debt ----------
st.subheader("Debt")
codes = [("BHFDDIH","Housing debt to income"),("BHFDA","Household debt to assets"),("LPHTSPRI","Loan repayments to income")]
for i in range(0, len(codes), 2):
    cols = st.columns(2)
    for j, (code, label) in enumerate(codes[i:i+2]):
        df = e2 if code in e2.columns else e13
        if code in df.columns:
            fig = line_fig(df, code, label)
            cols[j].pyplot(fig)
            finance_figs.append(fig)
            finance_stats.append(calc_mom_yoy(df, code, label))

# ---------- Lending Rates ----------
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
    for j, (code, label) in enumerate(lending_codes[i:i+2]):
        if code in f6.columns:
            fig = line_fig(f6, code, label)
            cols[j].pyplot(fig)
            finance_figs.append(fig)
            finance_stats.append(calc_mom_yoy(f6, code, label))

# ---------- Credit Growth ----------
st.subheader("Credit Growth")
cols = st.columns(2)
if "DGFACOHM" in d1.columns:
    fig = line_fig(d1, "DGFACOHM", "12-month housing credit growth", "Percent")
    cols[0].pyplot(fig)
    finance_figs.append(fig)
    finance_stats.append(calc_mom_yoy(d1, "DGFACOHM", "12-month housing credit growth"))

if "DGFACBNF12" in d1.columns:
    fig = line_fig(d1, "DGFACBNF12", "12-month business credit growth", "Percent")
    cols[1].pyplot(fig)
    finance_figs.append(fig)
    finance_stats.append(calc_mom_yoy(d1, "DGFACBNF12", "12-month business credit growth"))

# ---------- AI Summary ----------
finance_summary = explain_with_gpt("\n".join(finance_stats), "Household Finance")
st.markdown("**AI Summary:** " + finance_summary)

# ---------- Add to PDF export ----------
combined_finance_figs = []
for i in range(0, len(finance_figs), 2):
    if i + 1 < len(finance_figs):
        combined_finance_figs.append(
            combine_side_by_side(finance_figs[i], finance_figs[i + 1], title="Household Finance Comparison")
        )
    else:
        combined_finance_figs.append(finance_figs[i])

report_sections.append({
    "header": "Household Finance",
    "text": "\n".join(finance_stats) + "\n\nAI Summary: " + finance_summary,
    "figs": combined_finance_figs
})



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
            return f"{title} ({data.index[-1].strftime('%b %Y')}) — MoM: {latest-prev:+.2f}, YoY: {latest-yoy:+.2f}", fig
    return f"{title}: insufficient data", fig


markets_figs = []
markets_stats = []

# ---------- FX ----------
st.subheader("Exchange Rates")
fx_stats = []
col1, col2 = st.columns(2)
with col1:
    stat, fig = plot_yf("AUDUSD=X", "AUD/USD (FX rate)")
    fx_stats.append(stat)
    markets_figs.append(fig)
with col2:
    stat, fig = plot_yf("AUDGBP=X", "AUD/GBP (FX rate)")
    fx_stats.append(stat)
    markets_figs.append(fig)

fx_summary = explain_with_gpt("\n".join(fx_stats), "Exchange Rates")
st.markdown("**AI Summary (FX):** " + fx_summary)
markets_stats.extend(fx_stats)

# ---------- Equities ----------
st.subheader("Equity Indices")
eq_stats = []
col1, col2 = st.columns(2)
with col1:
    stat, fig = plot_yf("^AXJO", "ASX200 Index")
    eq_stats.append(stat)
    markets_figs.append(fig)
with col2:
    stat, fig = plot_yf("^GSPC", "S&P500 Index")
    eq_stats.append(stat)
    markets_figs.append(fig)

eq_summary = explain_with_gpt("\n".join(eq_stats), "Equity Indices")
st.markdown("**AI Summary (Equities):** " + eq_summary)
markets_stats.extend(eq_stats)

# ---------- YoY Change side-by-side ----------
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
    markets_figs.append(fig)

    if len(yoy) > 0:
        latest_val = float(yoy.iloc[-1])
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
    markets_figs.append(fig)

    if len(yoy) > 0:
        latest_val = float(yoy.iloc[-1])
        if not np.isnan(latest_val):
            yoy_stats.append(f"S&P500 latest YoY change: {latest_val:+.2f}%")

yoy_summary = explain_with_gpt("\n".join(yoy_stats), "YoY Index Changes")
st.markdown("**AI Summary (YoY Changes):** " + yoy_summary)
markets_stats.extend(yoy_stats)

# ---------- Add to PDF export ----------
combined_market_figs = []
for i in range(0, len(markets_figs), 2):
    if i + 1 < len(markets_figs):
        combined_market_figs.append(
            combine_side_by_side(markets_figs[i], markets_figs[i + 1], title="Markets Comparison")
        )
    else:
        combined_market_figs.append(markets_figs[i])

report_sections.append({
    "header": "Markets Dashboard (Yahoo Finance)",
    "text": "\n".join(markets_stats)
            + "\n\nAI Summary (FX): " + fx_summary
            + "\n\nAI Summary (Equities): " + eq_summary
            + "\n\nAI Summary (YoY Changes): " + yoy_summary,
    "figs": combined_market_figs
})


# =========================================================
# 🏠 CoreLogic Daily Home Value Index
# =========================================================
st.header("🏠 CoreLogic Daily Home Value Index")

corelogic_figs = []
corelogic_stats = []
corelogic_summary = ""

try:
    # Read the Excel file and detect the real header row
    df_raw = pd.read_excel("corelogic_daily_index.xlsx", header=None)

    # Find the row that contains the actual column headers (starts with 'Date')
    header_row = None
    for i in range(len(df_raw)):
        if df_raw.iloc[i].astype(str).str.contains("Date", case=False).any():
            header_row = i
            break

    if header_row is None:
        st.error("Couldn't find a header row containing 'Date'.")
    else:
        # Re-read the file with the detected header row
        df = pd.read_excel("corelogic_daily_index.xlsx", header=header_row)

        # Clean up column names
        df.columns = df.columns.str.strip()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])

        cities = [
            "Sydney (SYDD)",
            "Melbourne (MELD)",
            "Brisbane (inc Gold Coast) (BRID)",
            "Adelaide (ADED)",
            "Perth (PERD)",
            "5 capital city aggregate (AUSD)"
        ]
        available = [c for c in cities if c in df.columns]

        if not available:
            st.error(f"No valid city columns found. Columns detected: {list(df.columns)}")
        else:
            fig = px.line(df, x="Date", y=available,
                          title="CoreLogic Daily Home Value Index Trends")
            st.plotly_chart(fig, use_container_width=True)
            corelogic_figs.append(fig)

            for c in available:
                if len(df[c].dropna()) > 1:
                    change = (df[c].iloc[-1] / df[c].iloc[0] - 1) * 100
                    corelogic_stats.append(f"{c}: {change:.2f}% change over period")

            corelogic_summary = explain_with_gpt("\n".join(corelogic_stats),
                                                 "CoreLogic Home Value Index")
            st.markdown("**AI Summary (CoreLogic):** " + corelogic_summary)

except Exception as e:
    st.warning(f"Unable to load CoreLogic data: {e}")

# ---------- Add to PDF export ----------
converted_corelogic_figs = []
for f in corelogic_figs:
    try:
        converted_corelogic_figs.append(plotly_to_matplotlib(f))
    except Exception:
        pass

report_sections.append({
    "header": "CoreLogic Daily Home Value Index",
    "text": "\n".join(corelogic_stats) + "\n\nAI Summary: " + corelogic_summary,
    "figs": converted_corelogic_figs
})


st.caption("Data source: RBA Statistical Tables, Yahoo Finance, and CoreLogic. Figures computed from public APIs and XLSX files at run-time.")

# =========================================================
# 🇦🇺 Australian Population Growth by State
# =========================================================
st.header("🇦🇺 Australian Net Migration by State")

population_figs = []
population_stats = []
population_summary = ""

try:
    # Load Excel and find header row containing 'Period'
    df_raw = pd.read_excel("Population.xlsx", sheet_name="New.WorkingSheet", header=None)
    header_row = None
    for i in range(len(df_raw)):
        if df_raw.iloc[i].astype(str).str.contains("Period", case=False).any():
            header_row = i
            break

    if header_row is None:
        st.error(f"Could not find header row with 'Period'. Rows detected: {len(df_raw)}")
    else:
        # Re-read with correct header row
        df_pop = pd.read_excel("Population.xlsx", sheet_name="New.WorkingSheet", header=header_row)
        df_pop.columns = df_pop.columns.astype(str).str.strip()

        # Rename & clean
        period_col = next((c for c in df_pop.columns if "period" in c.lower()), None)
        df_pop = df_pop.rename(columns={period_col: "Period"})
        df_pop["Period"] = pd.to_datetime(df_pop["Period"], errors="coerce", format="%Y")
        df_pop = df_pop.dropna(subset=["Period"])

        # Select remaining columns as states
        states = [c for c in df_pop.columns if c != "Period"]

        # --- Plot population trends ---
        fig, ax = plt.subplots(figsize=(10,5))
        for s in states:
            ax.plot(df_pop["Period"], df_pop[s], label=s)
        ax.set_title("Population by State (ABS)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population ('000s)")
        ax.legend(loc="upper left", ncol=2)
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
        population_figs.append(fig)

        # --- Calculate change over period for each state ---
        for s in states:
            if df_pop[s].dropna().shape[0] > 1:
                change = (df_pop[s].iloc[-1] / df_pop[s].iloc[0] - 1) * 100
                population_stats.append(f"{s}: {change:.2f}% change over period")

        population_summary = explain_with_gpt("\n".join(population_stats),
                                              "Australian Population by State")
        st.markdown("**AI Summary (Australia):** " + population_summary)

except Exception as e:
    st.warning(f"Unable to load population data: {e}")


# =========================================================
# 🌐 Global Total Population
# =========================================================
st.header("🌐 Global Total Population Over Time")

global_figs = []
global_stats = []
global_summary = ""

try:
    url = "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    json_data = resp.json()

    records = json_data[1]
    df = pd.json_normalize(records)[["date", "value"]].dropna()
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("date")

    # Plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["date"], df["value"] / 1e9, color="tab:blue")  # scale to billions
    ax.set_title("World Population (total) — Billions")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population (Billions)")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)
    global_figs.append(fig)

    # Latest value text
    latest = df.iloc[-1]
    latest_text = f"Latest world population (year {int(latest['date'])}): {latest['value'] / 1e9:.3f} billion"
    st.markdown(f"**{latest_text}**")
    global_stats.append(latest_text)

    global_summary = explain_with_gpt("\n".join(global_stats), "Global Population Trends")
    st.markdown("**AI Summary (Global):** " + global_summary)

except Exception as e:
    st.warning(f"Unable to load global population data: {e}")

# ---------- Add to PDF export ----------
report_sections.append({
    "header": "Population Trends",
    "text": "\n".join(population_stats + global_stats)
            + "\n\nAI Summary (Australia): " + population_summary
            + "\n\nAI Summary (Global): " + global_summary,
    "figs": population_figs + global_figs
})




# =========================================================
# 🇦🇺 Vanguard Australian Shares Index ETF (VAS.AX)
# =========================================================
st.header("🇦🇺 Vanguard Australian Shares Index ETF (VAS.AX) — 5-Year Indexed Performance")

vas_figs = []
vas_stats = []
vas_summary = ""

try:
    ticker = "VAS.AX"
    data = yf.download(ticker, period="5y", interval="1mo")["Close"].dropna()
    data = data / data.iloc[0] * 100  # Normalize to 100 for indexed performance

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(data.index, data, color="tab:blue", linewidth=2)
    ax.set_title("Vanguard Australian Shares Index ETF (VAS.AX) — 5-Year Indexed Performance")
    ax.set_ylabel("Indexed Price (Base = 100)")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)
    vas_figs.append(fig)

    # Summary with explicit float conversion
    change_1y = (
        (float(data.iloc[-1]) / float(data.iloc[-12]) - 1) * 100
        if len(data) > 12 else np.nan
    )
    change_5y = (float(data.iloc[-1]) / float(data.iloc[0]) - 1) * 100
    perf_text = f"5-Year Change: {change_5y:+.2f}% | 1-Year Change: {change_1y:+.2f}%"
    st.markdown(f"**{perf_text}**")

    vas_stats.append(perf_text)
    vas_summary = explain_with_gpt("\n".join(vas_stats),
                                   "Vanguard Australian Shares ETF Performance")
    st.markdown("**AI Summary:** " + vas_summary)

except Exception as e:
    st.warning(f"Unable to load Vanguard ETF data: {e}")

# Add to report_sections
report_sections.append({
    "header": "Vanguard Australian Shares Index ETF (VAS.AX)",
    "text": "\n".join(vas_stats) + "\n\nAI Summary: " + vas_summary,
    "figs": vas_figs
})


# =========================================================
# 🌍 Global Central Bank Policy Rates
# =========================================================
st.header("🌍 Global Central Bank Policy Rates")

rates_figs = []
rates_stats = []
rates_summary = ""

try:
    # Load and clean
    df_rates = pd.read_excel("global_interest_rates.xlsx")
    df_rates.columns = [c.strip() for c in df_rates.columns]
    df_rates["Date"] = pd.to_datetime(df_rates["Date"], errors="coerce", format="%b-%Y")
    df_rates = df_rates.dropna(subset=["Date"]).sort_values("Date")

    countries = [c for c in df_rates.columns if c != "Date"]

    # --- Line chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in countries:
        ax.plot(df_rates["Date"], df_rates[c], label=c)
    ax.set_title("Global Central Bank Policy Rates (Monthly)")
    ax.set_ylabel("Policy Rate (%)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    rates_figs.append(fig)

    # --- Summary table ---
    latest = df_rates.iloc[-1]
    summary = pd.DataFrame({
        "Country": countries,
        "Latest Rate (%)": [latest[c] for c in countries]
    })

    # Compare vs Australia
    if "Australia" in summary["Country"].values:
        au_rate = float(summary.loc[summary["Country"] == "Australia", "Latest Rate (%)"].values[0])
        summary["Δ vs Australia (bps)"] = ((summary["Latest Rate (%)"] - au_rate) * 100).round(0)

    st.subheader("Latest Policy Rates")
    st.dataframe(summary.style.format({"Latest Rate (%)": "{:.2f}", "Δ vs Australia (bps)": "{:+.0f}"}))

    # --- AI Summary ---
    lines = [f"{r['Country']}: {r['Latest Rate (%)']:.2f}%" for _, r in summary.iterrows()]
    rates_summary = explain_with_gpt("\n".join(lines), "Global Central Bank Policy Rates")
    st.markdown("**AI Summary:** " + rates_summary)
    rates_stats.extend(lines)

except Exception as e:
    st.warning(f"Unable to load interest rate data: {e}")

# Add to report_sections
report_sections.append({
    "header": "Global Central Bank Policy Rates",
    "text": "\n".join(rates_stats) + "\n\nAI Summary: " + rates_summary,
    "figs": rates_figs
})

# =========================================================
# 📄 Generate Full PDF Report
# =========================================================
if st.button("📄 Generate Full PDF Report"):
    pdf_bytes = generate_pdf(
        f"AU Macro & Markets Dashboard ({start_date} to {end_date})",
        report_sections
    )
    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_bytes,
        file_name=f"AU_Macro_Dashboard_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )






