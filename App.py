import streamlit as st
import pandas as pd
import numpy as np
import requests

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="🏈 College Football — Auto from CFBD", layout="centered")
st.title("🏈 College Football — 2025 (auto from CFBD)")

# -----------------------------
# API KEY SETUP
# -----------------------------
CFBD_API_KEY = st.secrets.get("CFBD_API_KEY", None)
if not CFBD_API_KEY:
    st.error("⚠️ Missing CFBD API key in Streamlit secrets.")
    st.stop()

headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}

# -----------------------------
# FETCH TEAM STATS
# -----------------------------
@st.cache_data(ttl=3600)
def get_cfb_stats(year=2024):
    """Fetch team stats from the new CFBD endpoint."""
    url = f"https://api.collegefootballdata.com/stats/season?year={year}&seasonType=regular"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        st.error(f"❌ API request failed with status {resp.status_code}")
        return None

    try:
        data = resp.json()
    except Exception as e:
        st.error(f"❌ Could not decode JSON: {e}")
        return None

    if not data:
        st.warning("⚠️ No CFB data returned for that year.")
        return None

    df = pd.DataFrame(data)
    # Extract relevant stats
    grouped = df.groupby("team").agg(
        off_ppg=("stat", lambda x: pd.to_numeric(df.loc[x.index][df.loc[x.index]["statName"]=="pointsPerGame"], errors="coerce").mean()),
        def_ppg=("stat", lambda x: pd.to_numeric(df.loc[x.index][df.loc[x.index]["statName"]=="opponentPointsPerGame"], errors="coerce").mean())
    ).reset_index()

    # Drop missing data
    grouped = grouped.dropna(subset=["off_ppg", "def_ppg"])
    return grouped

# -----------------------------
# SIMULATE MATCHUP
# -----------------------------
def simulate_game(home, away, df, sims=10000):
    """Simulate expected scores and win probabilities."""
    h = df[df["team"] == home].iloc[0]
    a = df[df["team"] == away].iloc[0]

    home_exp = (h["off_ppg"] + a["def_ppg"]) / 2
    away_exp = (a["off_ppg"] + h["def_ppg"]) / 2

    home_scores = np.random.poisson(home_exp, sims)
    away_scores = np.random.poisson(away_exp, sims)

    home_win = np.sum(home_scores > away_scores) / sims * 100
    away_win = np.sum(away_scores > home_scores) / sims * 100
    tie = 100 - home_win - away_win

    return {
        "home_exp": home_exp,
        "away_exp": away_exp,
        "home_win": home_win,
        "away_win": away_win,
        "tie": tie
    }

# -----------------------------
# MAIN UI
# -----------------------------
st.subheader("Select Season and Teams")

year = st.number_input("Season", min_value=2023, max_value=2025, value=2024)

with st.spinner("Loading college football data..."):
    df = get_cfb_stats(year)

if df is None or df.empty:
    st.error("❌ No valid data found. Try 2024 or verify your API key.")
    st.stop()

teams = sorted(df["team"].unique())

col1, col2 = st.columns(2)
home = col1.selectbox("Home Team", teams, index=0)
away = col2.selectbox("Away Team", teams, index=1)

if home and away:
    st.write(f"Comparing **{home} vs {away}**")
    results = simulate_game(home, away, df)
    st.markdown(f"""
    - 🏠 **{home} Expected Points:** {results['home_exp']:.1f}  
    - ✈️ **{away} Expected Points:** {results['away_exp']:.1f}  
    - 📊 **Win Probability:**  
        - {home}: {results['home_win']:.2f}%  
        - {away}: {results['away_win']:.2f}%  
        - Tie: {results['tie']:.2f}%
    """)

st.divider()
with st.expander("Show Raw Team Stats Table"):
    st.dataframe(df.sort_values("off_ppg", ascending=False))
