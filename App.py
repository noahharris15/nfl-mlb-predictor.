import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

# -----------------------------
# SETUP
# -----------------------------
st.set_page_config(page_title="🏈 College Football — Auto from CFBD", layout="centered")

st.title("🏈 College Football — 2025 (auto from CFBD)")

# Get CFBD API key from Streamlit secrets
CFBD_API_KEY = st.secrets.get("CFBD_API_KEY", None)

if not CFBD_API_KEY:
    st.error("⚠️ Missing CFBD API Key in Streamlit secrets.")
    st.stop()

headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}

# -----------------------------
# DATA FETCHING FUNCTIONS
# -----------------------------

@st.cache_data(ttl=86400)
def get_cfb_team_stats(year: int = 2024):
    """Fetch team stats (points for/against) from CFBD."""
    url = f"https://api.collegefootballdata.com/team/stats?year={year}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"❌ API request failed with status {response.status_code}")
        return None
    try:
        data = response.json()
        if not data:
            st.warning("⚠️ No data returned from CFBD API.")
            return None
        df = pd.json_normalize(data)
        # Extract useful stats (points per game for and against)
        team_ppg = []
        for team in data:
            name = team.get("team")
            points = next((x["stat"] for x in team["stats"] if x["category"] == "pointsPerGame"), None)
            points_against = next((x["stat"] for x in team["stats"] if x["category"] == "opponentPointsPerGame"), None)
            if points and points_against:
                team_ppg.append({"team": name, "off_ppg": float(points), "def_ppg": float(points_against)})
        return pd.DataFrame(team_ppg)
    except json.JSONDecodeError:
        st.error("❌ Could not decode JSON. Check API key or endpoint.")
        return None


# -----------------------------
# SIMULATION FUNCTION
# -----------------------------

def simulate_matchup(team1, team2, df, sims=10000):
    """Simulate a matchup between two teams using Poisson distribution."""
    t1 = df[df["team"] == team1].iloc[0]
    t2 = df[df["team"] == team2].iloc[0]

    # Expected points = offensive avg vs opponent's defensive avg
    t1_exp = (t1["off_ppg"] + t2["def_ppg"]) / 2
    t2_exp = (t2["off_ppg"] + t1["def_ppg"]) / 2

    t1_scores = np.random.poisson(t1_exp, sims)
    t2_scores = np.random.poisson(t2_exp, sims)

    t1_wins = np.sum(t1_scores > t2_scores)
    t2_wins = np.sum(t2_scores > t1_scores)
    ties = sims - t1_wins - t2_wins

    return {
        "team1_exp": t1_exp,
        "team2_exp": t2_exp,
        "team1_win_prob": round(t1_wins / sims * 100, 2),
        "team2_win_prob": round(t2_wins / sims * 100, 2),
        "tie_prob": round(ties / sims * 100, 2),
    }

# -----------------------------
# MAIN APP UI
# -----------------------------

st.subheader("Select Season and Teams")

season = st.number_input("Season", min_value=2023, max_value=2025, value=2024, step=1)

# Fetch stats
with st.spinner("Fetching College Football stats..."):
    df = get_cfb_team_stats(season)

if df is None or df.empty:
    st.error("❌ No CFB stats available. Check your API key or try again later.")
    st.stop()

teams = df["team"].sort_values().unique()
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Home Team", teams, index=0)
with col2:
    away_team = st.selectbox("Away Team", teams, index=1)

if home_team and away_team:
    st.write(f"Comparing: **{home_team}** vs **{away_team}**")

    results = simulate_matchup(home_team, away_team, df)

    st.markdown(f"""
    **🏠 {home_team} Expected Points:** {results['team1_exp']:.1f}  
    **✈️ {away_team} Expected Points:** {results['team2_exp']:.1f}  
    **📊 Win Probabilities:**  
    - {home_team}: **{results['team1_win_prob']}%**  
    - {away_team}: **{results['team2_win_prob']}%**  
    - Tie: **{results['tie_prob']}%**
    """)

st.divider()
with st.expander("Show Raw Team Stats Table"):
    st.dataframe(df.sort_values("off_ppg", ascending=False))
