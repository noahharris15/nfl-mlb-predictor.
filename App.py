import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="🏈 College Football Predictor", layout="centered")
st.title("🏈 College Football — 2025 (auto from CFBD)")

# --- API Key ---
CFBD_API_KEY = st.secrets.get("CFBD_API_KEY", None)
if not CFBD_API_KEY:
    st.error("⚠️ Missing CFBD API key in Streamlit secrets.")
    st.stop()

headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}

# --- Pull team stats (using teams/season endpoint for reliability) ---
@st.cache_data(ttl=3600)
def get_cfb_team_stats(year=2024):
    """Pulls team-level stats (points for/against per game)"""
    url = f"https://api.collegefootballdata.com/teams/season?year={year}"
    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        st.error(f"❌ CFBD API error: {r.status_code}")
        return pd.DataFrame()

    try:
        data = r.json()
    except Exception as e:
        st.error(f"JSON error: {e}")
        return pd.DataFrame()

    if not data:
        st.warning("⚠️ No data returned for this year.")
        return pd.DataFrame()

    # Convert to DataFrame
    teams = []
    for team in data:
        try:
            pts_for = team.get("pointsFor", 0)
            pts_against = team.get("pointsAgainst", 0)
            games = team.get("games", 1)
            off_ppg = pts_for / games if games > 0 else 0
            def_ppg = pts_against / games if games > 0 else 0
            teams.append({
                "team": team["team"]["school"],
                "conference": team.get("conference", ""),
                "off_ppg": round(off_ppg, 2),
                "def_ppg": round(def_ppg, 2)
            })
        except Exception:
            continue

    df = pd.DataFrame(teams)
    if df.empty:
        st.error("❌ Could not extract scoring stats. Check CFBD API format.")
    return df

# --- Game simulation ---
def simulate_game(home, away, df, sims=10000):
    """Simulate a game using Poisson scoring model"""
    h = df[df["team"] == home].iloc[0]
    a = df[df["team"] == away].iloc[0]

    home_exp = (h["off_ppg"] + a["def_ppg"]) / 2
    away_exp = (a["off_ppg"] + h["def_ppg"]) / 2

    home_scores = np.random.poisson(home_exp, sims)
    away_scores = np.random.poisson(away_exp, sims)

    home_win = np.mean(home_scores > away_scores) * 100
    away_win = np.mean(away_scores > home_scores) * 100

    return home_exp, away_exp, home_win, away_win

# --- Streamlit UI ---
year = st.number_input("Season", min_value=2023, max_value=2025, value=2024)
st.info("Using CFBD `/teams/season` API to load real offensive & defensive stats per game.")

with st.spinner("Loading team stats..."):
    df = get_cfb_team_stats(year)

if df.empty:
    st.stop()

teams = sorted(df["team"].unique())
col1, col2 = st.columns(2)
home_team = col1.selectbox("🏠 Home Team", teams)
away_team = col2.selectbox("✈️ Away Team", teams)

if st.button("Run Simulation"):
    home_exp, away_exp, home_win, away_win = simulate_game(home_team, away_team, df)
    st.markdown(f"""
    ## 🏈 {home_team} vs {away_team}
    **Expected Points:** {home_team} {home_exp:.1f} — {away_team} {away_exp:.1f}  
    **Win Probabilities:**  
    - {home_team}: {home_win:.2f}%  
    - {away_team}: {away_win:.2f}%
    """)
    st.dataframe(df.sort_values("off_ppg", ascending=False).reset_index(drop=True))
