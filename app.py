# =========================================================
# 🏠 CoreLogic Daily Home Value Index
# =========================================================
st.header("🏠 CoreLogic Daily Home Value Index")

try:
    # Load file from data folder (adjust if it's elsewhere)
    df = pd.read_excel("data/corelogic_daily_index.xlsx")

    # Normalize and auto-detect date column
    df.columns = df.columns.str.strip().str.lower()
    date_col = next((c for c in df.columns if "date" in c.lower()), None)

    if not date_col:
        st.error("❌ Could not find a 'Date' column in CoreLogic file.")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        # Detect city columns dynamically (anything except the date)
        city_cols = [c for c in df.columns if c != date_col]
        if not city_cols:
            st.error("❌ No city columns found in CoreLogic file.")
        else:
            # Plot interactive multi-city line chart
            fig = px.line(
                df,
                x=date_col,
                y=city_cols,
                title="CoreLogic Daily Home Value Index Trends",
                labels={"value": "Index", "variable": "City"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # YoY change summary
            yoy_stats = []
            for c in city_cols:
                if len(df[c].dropna()) > 1:
                    change = (df[c].iloc[-1] / df[c].iloc[0] - 1) * 100
                    yoy_stats.append(f"{c}: {change:.2f}% change over period")

            if yoy_stats:
                st.markdown(
                    "**AI Summary (CoreLogic):** "
                    + explain_with_gpt("\n".join(yoy_stats), "CoreLogic Home Value Index")
                )
            else:
                st.info("No valid data for YoY calculation.")
except Exception as e:
    st.warning(f"Unable to load CoreLogic data: {e}")
