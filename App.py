import streamlit as st
import requests
import pandas as pd
import numpy as np

st.title("🏈 College Football — 2025 (auto from CFBD)")

API_KEY = st.secrets["CFBD_API_KEY"]
BASE_URL = "https://api.collegefootballdata.com"
headers = {"Authorization": f"Bearer {API_KEY}"}

# ------------------------------
# Pull Team Stats from CFBD API
# ------------------------------
@st.cache_data
def get_cfb_team_stats(year=2024):
    url = f"{BASE_URL}/stats/season?year={year}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"CFBD request failed: {e}")
        return pd.DataFrame()

    teams = []
    for item in data:
        team_name = item.get("team")
        off_ppg = item.get("pointsPerGame")
        def_ppg = item.get("opponentPointsPerGame")

        # Skip teams missing stats
        if off_ppg is None or def_ppg is None:
            continue

        teams.append({
            "team": team_name,
            "off_ppg": float(off_ppg),
            "def_ppg": float(def_ppg)
        })

    df = pd.DataFrame(teams)
    return df

# ------------------------------
# Load Data and Fallback
# ------------------------------
year = st.selectbox("Select Year", [2023, 2024, 2025], index=1)
df = get_cfb_team_stats(year)

if df.empty:
    st.warning("⚠️ No data found for this year — try 2024 or upload fallback CSV.")
    uploaded = st.file_uploader("Upload CFB CSV (team, off_ppg, def_ppg)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

if df.empty:
    st.stop()

# ------------------------------
# Team Comparison
# ------------------------------
home_team = st.selectbox("Home team", df["team"].unique())
away_team = st.selectbox("Away team", df["team"].unique())

home_stats = df[df["team"] == home_team].iloc[0]
away_stats = df[df["team"] == away_team].iloc[0]

# ------------------------------
# Simulation Function
# ------------------------------
def simulate_game(home, away, sims=10000):
    # Calculate expected points safely
    home_exp = np.nanmean([(home["off_ppg"] + away["def_ppg"]) / 2])
    away_exp = np.nanmean([(away["off_ppg"] + home["def_ppg"]) / 2])

    # Prevent invalid (negative or nan) values
    home_exp = max(home_exp, 0.1)
    away_exp = max(away_exp, 0.1)

    home_scores = np.random.poisson(home_exp, sims)
    away_scores = np.random.poisson(away_exp, sims)

    return {
        "home_exp": home_exp,
        "away_exp": away_exp,
        "p_home_win": np.mean(home_scores > away_scores),
        "p_away_win": np.mean(away_scores > home_scores)
    }

# ------------------------------
# Run Simulation
# ------------------------------
result = simulate_game(home_stats, away_stats)

st.subheader(f"{home_team} vs {away_team}")
st.write(
    f"Expected points: **{home_team} {result['home_exp']:.1f} – {away_team} {result['away_exp']:.1f}**"
)
st.write(
    f"P({home_team} win) = **{result['p_home_win']:.1%}**, "
    f"P({away_team} win) = **{result['p_away_win']:.1%}**"
)

with st.expander("📊 Full team stats table"):
    st.dataframe(df)
