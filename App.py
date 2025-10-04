import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="College Football Predictor", layout="centered")
st.title("🏈 College Football — Predictor (CFBD)")

# --- Load API Key ---
CFBD_API_KEY = st.secrets.get("CFBD_API_KEY")
if not CFBD_API_KEY:
    st.error("⚠️ Missing CFBD API key in Streamlit secrets.")
    st.stop()

headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}

# --- Fetch team stats via /teams/season, fallback logic ---
@st.cache_data(ttl=3600)
def get_cfb_team_stats(season: int) -> pd.DataFrame:
    """
    Fetch team stats (points for / points against) from CFBD /teams/season.
    If the chosen season returns no data, fallback to 2024.
    """
    def fetch(year):
        url = f"https://api.collegefootballdata.com/teams/season?year={year}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except:
            return None

    data = fetch(season)
    used = season
    if not data:
        st.warning(f"No data for {season}, falling back to 2024.")
        data = fetch(2024)
        used = 2024
    if not data:
        st.error("No CFBD data available (even fallback).")
        return pd.DataFrame()

    rows = []
    for team in data:
        # team["team"] is dict with school & team info
        school = team.get("team", {}).get("school") or team.get("team", "")
        games = team.get("games", 0)
        pts_for = team.get("pointsFor", 0)
        pts_against = team.get("pointsAgainst", 0)
        if games > 0:
            off_ppg = pts_for / games
            def_ppg = pts_against / games
        else:
            off_ppg = None
            def_ppg = None
        rows.append({
            "team": school,
            "off_ppg": off_ppg,
            "def_ppg": def_ppg
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["off_ppg", "def_ppg"])
    return df

# --- Simulation logic ---
def simulate_matchup(home: str, away: str, df: pd.DataFrame, trials: int = 10000):
    h = df[df["team"] == home].iloc[0]
    a = df[df["team"] == away].iloc[0]

    mu_home = (h["off_ppg"] + a["def_ppg"]) / 2.0
    mu_away = (a["off_ppg"] + h["def_ppg"]) / 2.0

    # safety floor
    mu_home = max(mu_home, 0.1)
    mu_away = max(mu_away, 0.1)

    home_scores = np.random.poisson(mu_home, trials)
    away_scores = np.random.poisson(mu_away, trials)

    home_win = float((home_scores > away_scores).mean())
    away_win = float((away_scores > home_scores).mean())

    avg_home = float(home_scores.mean())
    avg_away = float(away_scores.mean())

    return avg_home, avg_away, home_win, away_win

# --- UI ---
st.subheader("Select Season & Teams")

season = st.number_input("Season", min_value=2023, max_value=2025, value=2025, step=1)

with st.spinner("Loading team stats..."):
    df = get_cfb_team_stats(int(season))

if df.empty:
    st.stop()

teams = sorted(df["team"].unique())
home = st.selectbox("Home team", teams)
away = st.selectbox("Away team", [t for t in teams if t != home], index=0)

if st.button("Run Simulation"):
    avg_h, avg_a, p_h, p_a = simulate_matchup(home, away, df)
    st.markdown(f"""
    **{home} vs {away}**  
    Expected points: **{home} {avg_h:.1f} – {away} {avg_a:.1f}**  
    Win probabilities: **{home} {100 * p_h:.1f}%**, **{away} {100 * p_a:.1f}%**
    """)

with st.expander("Team stats table"):
    st.dataframe(df.sort_values("off_ppg", ascending=False))
