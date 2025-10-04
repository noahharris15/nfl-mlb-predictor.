import streamlit as st
import pandas as pd
import numpy as np
import requests

# --------------------------
# 🔑 CFBD API Setup
# --------------------------
API_KEY = "YOUR_CFBD_API_KEY"  # replace with your real key
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(page_title="College Football Predictor", layout="centered")
st.title("🏈 College Football Predictor — Auto from CFBD")

# --------------------------
# ⚙️ Function: Fetch team stats from CFBD
# --------------------------
@st.cache_data
def get_cfb_team_ppg(season):
    url = f"https://api.collegefootballdata.com/team/seasonStats?year={season}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None

        # Build dataframe with PPG
        records = []
        for team_data in data:
            team = team_data["team"]
            total_points = 0
            total_games = 0
            for stat in team_data["stats"]:
                if stat["statName"] == "points":
                    total_points = stat["statValue"]
                if stat["statName"] == "games":
                    total_games = stat["statValue"]
            if total_games > 0:
                ppg = total_points / total_games
                records.append({"team": team, "off_ppg": round(ppg, 2)})

        return pd.DataFrame(records)

    except Exception as e:
        st.error(f"CFBD API Error: {e}")
        return None


# --------------------------
# 🏈 Simulation Function
# --------------------------
def simulate_game(home_ppg, away_ppg, sims=10000):
    home_scores = np.random.poisson(home_ppg, sims)
    away_scores = np.random.poisson(away_ppg, sims)
    home_win_rate = np.mean(home_scores > away_scores)
    away_win_rate = np.mean(away_scores > home_scores)
    return home_win_rate, away_win_rate


# --------------------------
# 🎮 UI
# --------------------------
season = st.number_input("Season", min_value=2000, max_value=2025, value=2024, step=1)
st.info("Fetching real offensive PPG stats from CFBD API...")

df = get_cfb_team_ppg(season)

if df is not None and not df.empty:
    st.success(f"Loaded {len(df)} teams for {season} season!")

    home_team = st.selectbox("Home Team", df["team"].unique())
    away_team = st.selectbox("Away Team", df["team"].unique())

    if home_team != away_team:
        home_ppg = df.loc[df["team"] == home_team, "off_ppg"].values[0]
        away_ppg = df.loc[df["team"] == away_team, "off_ppg"].values[0]

        home_win, away_win = simulate_game(home_ppg, away_ppg)
        st.metric(f"{home_team} Win %", f"{home_win*100:.1f}%")
        st.metric(f"{away_team} Win %", f"{away_win*100:.1f}%")

    else:
        st.warning("Select two different teams to simulate a game.")
else:
    st.error("⚠️ No CFBD data found. You can upload a fallback CSV (columns: team, off_ppg).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("CSV loaded successfully.")
