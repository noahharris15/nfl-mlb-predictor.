import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- Function to pull College Football stats from CFBD ---
def cfb_team_stats(year=2025):
    api_key = st.secrets["CFBD_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}"}

    url = f"https://api.collegefootballdata.com/teams/season?year={year}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"CFBD request failed: {response.text}")
        return pd.DataFrame()

    data = response.json()

    # Extract team-level offense and defense points per game
    teams = []
    for team in data:
        try:
            teams.append({
                "team": team["team"],
                "off_ppg": team.get("pointsPerGame", np.nan),
                "def_ppg": team.get("opponentPointsPerGame", np.nan)
            })
        except:
            continue

    return pd.DataFrame(teams)


# --- Monte Carlo Simulation for CFB games ---
def simulate_game(home_off, away_def, away_off, home_def, sims=10000):
    home_scores = np.random.normal(home_off, 7, sims) - np.random.normal(away_def, 7, sims)
    away_scores = np.random.normal(away_off, 7, sims) - np.random.normal(home_def, 7, sims)

    home_avg = np.mean(home_scores)
    away_avg = np.mean(away_scores)
    home_win_prob = np.mean(home_scores > away_scores)
    away_win_prob = 1 - home_win_prob

    return home_avg, away_avg, home_win_prob, away_win_prob


# --- Streamlit Page ---
st.title("🏈 College Football — 2025 (auto from CFBD)")

df = cfb_team_stats(2025)

if df.empty:
    st.warning("No CFB stats available. Double-check API key or CFBD limits.")
else:
    st.dataframe(df)

    home_team = st.selectbox("Home team", df["team"].unique())
    away_team = st.selectbox("Away team", df["team"].unique())

    if st.button("Simulate Game"):
        home = df[df["team"] == home_team].iloc[0]
        away = df[df["team"] == away_team].iloc[0]

        home_avg, away_avg, home_prob, away_prob = simulate_game(
            home["off_ppg"], away["def_ppg"], away["off_ppg"], home["def_ppg"]
        )

        st.write(f"**{home_team} vs {away_team}**")
        st.write(f"Expected points: {home_avg:.1f} – {away_avg:.1f}")
        st.write(f"P({home_team} win) = {home_prob*100:.1f}%")
        st.write(f"P({away_team} win) = {away_prob*100:.1f}%")
