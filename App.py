import requests
import pandas as pd
import numpy as np
import streamlit as st
import time

st.set_page_config(page_title="All-Sports PrizePicks Simulator", page_icon="📊")

st.title("📊 All-Sports PrizePicks Simulator (auto baselines, fixed parsing)")
st.caption("Automatically pulls PrizePicks data for any league and runs over/under simulations")

leagues = ["NFL", "NBA", "MLB", "CFB", "Soccer", "NHL"]
selected_league = st.selectbox("League", leagues)

PRIZEPICKS_URL = "https://api.prizepicks.com/projections"

# API request function with retries and correct headers
def fetch_prizepicks_data(league):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://app.prizepicks.com",
        "Origin": "https://app.prizepicks.com"
    }
    params = {"per_page": 250, "single_stat": "true", "league": league.lower()}
    for _ in range(5):
        response = requests.get(PRIZEPICKS_URL, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code in [403, 429]:
            st.warning("⏳ Waiting before retrying (PrizePicks rate limit)...")
            time.sleep(3)
        else:
            st.error(f"PrizePicks fetch failed ({response.status_code}): {response.text}")
            return None
    return None

# Parse PrizePicks API response
def parse_prizepicks_data(data):
    if not data or "data" not in data:
        return pd.DataFrame()

    projections = []
    for item in data["data"]:
        try:
            player = item["relationships"]["new_player"]["data"]["id"]
            attr = item["attributes"]
            projections.append({
                "player": attr["display_name"],
                "team": attr.get("team", "N/A"),
                "market": attr.get("stat_type", "N/A"),
                "line": attr.get("line_score", np.nan),
                "league": attr.get("league", "N/A")
            })
        except Exception:
            continue
    return pd.DataFrame(projections)

# Run simulation for each player
def simulate_probs(df, sims=10000):
    np.random.seed(42)
    results = []
    for _, row in df.iterrows():
        mean = float(row["line"])
        sd = mean * 0.15  # basic variance estimate (adjust later with real stats)
        samples = np.random.normal(mean, sd, sims)
        p_over = np.mean(samples > mean)
        p_under = 1 - p_over
        results.append({"player": row["player"], "market": row["market"],
                        "line": mean, "P(over)": p_over, "P(under)": p_under})
    return pd.DataFrame(results)

# Main flow
data = fetch_prizepicks_data(selected_league)

if data:
    df = parse_prizepicks_data(data)
    if not df.empty:
        st.success(f"✅ Pulled {len(df)} {selected_league} player lines from PrizePicks.")
        st.dataframe(df.head(15))

        st.subheader("Simulated edges (normal model)")
        sim_df = simulate_probs(df)
        st.dataframe(sim_df)

    else:
        st.warning("⚠️ No valid player data parsed from PrizePicks.")
else:
    st.error("❌ Could not fetch PrizePicks data. Check network or retry later.")
