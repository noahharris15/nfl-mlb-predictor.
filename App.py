import streamlit as st
import pandas as pd
import numpy as np
import cfbd
from cfbd.rest import ApiException
from cfbd.configuration import Configuration

# --------------------------
# 🔑 CFBD API Setup
# --------------------------
API_KEY = "YOUR_CFBD_API_KEY"  # Replace this with your real key
configuration = Configuration()
configuration.api_key['Authorization'] = API_KEY
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_instance = cfbd.TeamStatsApi(cfbd.ApiClient(configuration))

st.set_page_config(page_title="College Football Predictor (CFBD)", layout="centered")
st.title("🏈 College Football Predictor — Auto from CFBD")

# --------------------------
# 🎯 Helper: Get CFB Stats
# --------------------------
@st.cache_data
def get_cfb_team_ppg(season):
    try:
        stats = api_instance.get_team_season_stats(year=season)
        team_data = {}
        for entry in stats:
            team = entry.team
            for stat in entry.stats:
                if stat.stat_name == "points":
                    points = stat.stat_value
                elif stat.stat_name == "games":
                    games = stat.stat_value
            if team and games > 0:
                ppg = points / games
                team_data[team] = ppg

        df = pd.DataFrame(list(team_data.items()), columns=["team", "off_ppg"])
        return df

    except ApiException as e:
        st.warning(f"CFBD API error: {e}")
        return None


# --------------------------
# 🏈 Simulation Function
# --------------------------
def simulate_cfb_game(home_ppg, away_ppg, sims=10000):
    home_scores = np.random.poisson(home_ppg, sims)
    away_scores = np.random.poisson(away_ppg, sims)
    home_wins = np.sum(home_scores > away_scores)
    away_wins = np.sum(away_scores > home_scores)
    return home_wins / sims, away_wins / sims


# --------------------------
# 🧠 Streamlit UI
# --------------------------
season = st.number_input("Season", min_value=2000, max_value=2025, value=2024, step=1)

st.info("Using CFBD API to load real offensive PPG stats.")
df = get_cfb_team_ppg(season)

if df is not None and not df.empty:
    st.success(f"Loaded {len(df)} teams for {season} season!")

    home_team = st.selectbox("Select Home Team", df['team'].unique())
    away_team = st.selectbox("Select Away Team", df['team'].unique())

    if home_team != away_team:
        home_ppg = df.loc[df['team'] == home_team, 'off_ppg'].values[0]
        away_ppg = df.loc[df['team'] == away_team, 'off_ppg'].values[0]

        home_win_prob, away_win_prob = simulate_cfb_game(home_ppg, away_ppg)

        st.metric(label=f"{home_team} Win Probability", value=f"{home_win_prob*100:.2f}%")
        st.metric(label=f"{away_team} Win Probability", value=f"{away_win_prob*100:.2f}%")
    else:
        st.warning("Select two different teams.")
else:
    st.error("⚠️ No CFBD data available. Upload fallback CSV below.")
    uploaded = st.file_uploader("Upload fallback CSV (columns: team, off_ppg)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Fallback CSV loaded!")
