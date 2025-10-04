import streamlit as st
import requests
import pandas as pd
import numpy as np

st.title("🏈 College Football — 2025 (auto from CFBD)")

API_KEY = st.secrets["CFBD_API_KEY"]
BASE_URL = "https://api.collegefootballdata.com"
headers = {"Authorization": f"Bearer {API_KEY}"}

# ------------------------------
# Function to load stats
# ------------------------------
@st.cache_data
def get_cfb_team_stats(year):
    url = f"{BASE_URL}/stats/season?year={year}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"CFBD request failed for {year}: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    teams = []
    for item in data:
        team_name = item.get("team")
        off_ppg = item.get("pointsPerGame")
        def_ppg = item.get("opponentPointsPerGame")

        if off_ppg and def_ppg:
            teams.append({
                "team": team_name,
                "off_ppg": float(off_ppg),
                "def_ppg": float(def_ppg)
            })

    return pd.DataFrame(teams)

# ------------------------------
# Auto fallback years
# ------------------------------
year_order = [2025, 2024, 2023]
df = pd.DataFrame()
used_year = None

for yr in year_order:
    df = get_cfb_team_stats(yr)
    if not df.empty:
        used_year = yr
        break

if df.empty:
    st.error("❌ No CFBD data found for 2023–2025. Try uploading a CSV fallback.")
    uploaded = st.file_uploader("Upload CFB CSV (team, off_ppg, def_ppg)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        used_year = "CSV upload"
    else:
        st.stop()

st.success(f"✅ Using {used_year} season data from CollegeFootballData.com")

# ------------------------------
# Simulation setup
# ------------------------------
home_team = st.selectbox("Home team", df["team"].unique())
away_team = st.selectbox("Away team", df["team"].unique())

home_stats = df[df["team"] == home_team].iloc[0]
away_stats = df[df["team"] == away_team].iloc[0]

def simulate_game(home, away, sims=10000):
    home_exp = np.nanmean([(home["off_ppg"] + away["def_ppg"]) / 2])
    away_exp = np.nanmean([(away["off_ppg"] + home["def_ppg"]) / 2])

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

with st.expander("📊 Show full team stats"):
    st.dataframe(df)
