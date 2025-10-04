import streamlit as st
import requests
import pandas as pd
import numpy as np

st.title("🏈 College Football — Auto from CFBD")

API_KEY = st.secrets["CFBD_API_KEY"]
BASE_URL = "https://api.collegefootballdata.com"

headers = {"Authorization": f"Bearer {API_KEY}"}

# --- Function to fetch team stats ---
@st.cache_data
def get_cfb_team_stats(year=2024):  # default 2024 since 2025 may not exist yet
    url = f"{BASE_URL}/stats/season?year={year}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"CFBD API error: {response.status_code} - {response.text}")
        return pd.DataFrame()

    try:
        data = response.json()
    except:
        st.error("❌ Could not decode JSON. Check API key or endpoint.")
        return pd.DataFrame()

    teams = []
    for team in data:
        team_name = team.get("team")
        off_ppg = team.get("pointsPerGame")
        def_ppg = team.get("opponentPointsPerGame")

        teams.append({
            "team": team_name,
            "off_ppg": float(off_ppg) if off_ppg else np.nan,
            "def_ppg": float(def_ppg) if def_ppg else np.nan
        })

    return pd.DataFrame(teams)

# --- Load Data ---
year = st.selectbox("Select Year", [2023, 2024, 2025], index=1)
df = get_cfb_team_stats(year)

if df.empty:
    st.warning("⚠️ No CFB stats available. Try a different year or check API key.")
    st.stop()

# Team selection
home_team = st.selectbox("Home team", df["team"].unique())
away_team = st.selectbox("Away team", df["team"].unique())

home_stats = df[df["team"] == home_team].iloc[0]
away_stats = df[df["team"] == away_team].iloc[0]

# --- Simulation ---
def simulate_game(home, away, sims=10000):
    home_score_exp = (home["off_ppg"] + away["def_ppg"]) / 2
    away_score_exp = (away["off_ppg"] + home["def_ppg"]) / 2

    home_scores = np.random.poisson(home_score_exp, sims)
    away_scores = np.random.poisson(away_score_exp, sims)

    return home_score_exp, away_score_exp, np.mean(home_scores > away_scores), np.mean(away_scores > home_scores)

home_exp, away_exp, p_home, p_away = simulate_game(home_stats, away_stats)

st.subheader(f"{home_team} vs {away_team}")
st.write(f"Expected points: {home_team} {home_exp:.1f} – {away_team} {away_exp:.1f}")
st.write(f"P({home_team} win) = {p_home:.1%}")
st.write(f"P({away_team} win) = {p_away:.1%}")

with st.expander("📊 Show full team stats table"):
    st.dataframe(df)
