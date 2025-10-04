import streamlit as st
import pandas as pd
import numpy as np
import requests

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="🏈 College Football — 2025 (auto from CFBD)", layout="centered")
st.title("🏈 College Football — 2025 (auto from CFBD)")

# -----------------------------
# API Key setup
# -----------------------------
CFBD_API_KEY = st.secrets.get("CFBD_API_KEY", None)
if not CFBD_API_KEY:
    st.error("⚠️ Missing CFBD API key in Streamlit secrets.")
    st.stop()

headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}

# -----------------------------
# Pull and clean CFB stats
# -----------------------------
@st.cache_data(ttl=3600)
def get_cfb_stats(year=2024):
    """Fetch and flatten team stats from CFBD API"""
    url = f"https://api.collegefootballdata.com/stats/season?year={year}&seasonType=regular"
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        st.error(f"❌ CFBD request failed: {resp.status_code}")
        return pd.DataFrame()

    try:
        data = resp.json()
    except Exception as e:
        st.error(f"❌ Could not decode JSON: {e}")
        return pd.DataFrame()

    if not data:
        st.warning("⚠️ No data returned for this season.")
        return pd.DataFrame()

    # Convert JSON to DataFrame and keep only relevant columns
    df = pd.json_normalize(data)[["team", "statName", "statValue"]]

    # Pivot to get each stat as a column
    df_pivot = df.pivot_table(index="team", columns="statName", values="statValue", aggfunc="first").reset_index()

    # Convert numeric columns
    for col in df_pivot.columns:
        if col != "team":
            df_pivot[col] = pd.to_numeric(df_pivot[col], errors="coerce")

    # Rename key stats
    df_pivot = df_pivot.rename(columns={
        "pointsPerGame": "off_ppg",
        "opponentPointsPerGame": "def_ppg"
    })

    df_pivot = df_pivot.dropna(subset=["off_ppg", "def_ppg"], how="any")
    return df_pivot

# -----------------------------
# Simulate Game
# -----------------------------
def simulate_game(home, away, df, sims=10000):
    """Monte Carlo sim for expected scores & win probability"""
    h = df[df["team"] == home].iloc[0]
    a = df[df["team"] == away].iloc[0]

    home_exp = (h["off_ppg"] + a["def_ppg"]) / 2
    away_exp = (a["off_ppg"] + h["def_ppg"]) / 2

    home_scores = np.random.poisson(home_exp, sims)
    away_scores = np.random.poisson(away_exp, sims)

    home_win = np.mean(home_scores > away_scores) * 100
    away_win = np.mean(away_scores > home_scores) * 100
    tie = 100 - home_win - away_win

    return {
        "home_exp": home_exp,
        "away_exp": away_exp,
        "home_win": home_win,
        "away_win": away_win,
        "tie": tie
    }

# -----------------------------
# User Interface
# -----------------------------
year = st.number_input("Season", min_value=2023, max_value=2025, value=2024, step=1)
st.write("📊 Using CFBD API to load real offensive & defensive PPG stats")

with st.spinner("Fetching live college football data..."):
    df = get_cfb_stats(year)

if df.empty:
    st.error("❌ No stats found. Try a different season or check API key.")
    st.stop()

teams = sorted(df["team"].unique())

col1, col2 = st.columns(2)
home = col1.selectbox("🏠 Home Team", teams, index=0)
away = col2.selectbox("✈️ Away Team", teams, index=1)

if home and away:
    st.subheader(f"**{home} vs {away}** — Simulation Results")

    results = simulate_game(home, away, df)
    st.markdown(f"""
    - 🏠 **{home} Expected Points:** {results['home_exp']:.1f}  
    - ✈️ **{away} Expected Points:** {results['away_exp']:.1f}  
    - 📈 **Win Probability:**  
        - {home}: {results['home_win']:.2f}%  
        - {away}: {results['away_win']:.2f}%  
        - Tie: {results['tie']:.2f}%
    """)

st.divider()
with st.expander("📋 Show Full Team Stats Table"):
    st.dataframe(df.sort_values("off_ppg", ascending=False))
