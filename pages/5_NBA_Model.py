import os
import subprocess
import pandas as pd
import streamlit as st

st.title("NBA Betting Model")

st.write("Run the model to generate today's NBA moneyline and spread edges.")

if st.button("Run NBA Model"):
    with st.spinner("Running NBA model... this can take a minute."):
        result = subprocess.run(
            ["python", "nba_model_runner.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            st.success("NBA model finished running.")
        else:
            st.error("NBA model failed.")
            st.text(result.stderr)

# look for latest saved files
moneyline_files = sorted(
    [f for f in os.listdir(".") if f.startswith("nba_best_moneyline_edges_") and f.endswith(".csv")]
)
spread_files = sorted(
    [f for f in os.listdir(".") if f.startswith("nba_best_spread_edges_") and f.endswith(".csv")]
)

if moneyline_files:
    latest_ml = moneyline_files[-1]
    st.subheader("Top Moneyline Edges")
    st.dataframe(pd.read_csv(latest_ml), use_container_width=True)
else:
    st.info("No moneyline results found yet. Click 'Run NBA Model' first.")

if spread_files:
    latest_spread = spread_files[-1]
    st.subheader("Top Spread Edges")
    st.dataframe(pd.read_csv(latest_spread), use_container_width=True)
else:
    st.info("No spread results found yet. Click 'Run NBA Model' first.")
