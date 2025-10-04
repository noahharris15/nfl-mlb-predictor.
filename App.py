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

# --- Pull & clean CFBD stats ---
@st.cache_data(ttl=3600)
def get_cfb_stats(year=2024):
    """Fetches and formats team-level stats from CFBD"""
    url = f"https://api.collegefootballdata.com/stats/season?year={year}&seasonType=regular"
    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        st.error(f"CFBD API error: {r.status_code}")
        return pd.DataFrame()

    try:
        data = r.json()
    except Exception as e:
        st.error(f"JSON parse error: {e}")
        return pd.DataFrame()

    if not data:
        st.warning("⚠️ No data found for this season.")
        return pd.DataFrame()

    df = pd.json_normalize(data)
    df = df[["team", "statName", "statValue"]]

    # Pivot stats wide
    df = df.pivot_table(index="team", columns="statName", values="statValue", aggfunc="first").reset_index()

    # Convert everything numeric that can be
    for c in df.columns:
        if c != "team":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Detect available scoring columns
    possible_off = [c for c in df.columns if "points" in c.lower() and "pergame" in c.lower() and "opponent" not in c.lower()]
    possible_def = [c for c in df.columns if "opponentpoints" in c.lower() or "opp_points" in c.lower()]

    if not possible_off or not possible_def:
        st.error("❌ Could not find points per game columns in CFBD data. Check API format.")
        return pd.DataFrame()

    df["off_ppg"] = df[possible_off[0]]
    df["def_ppg"] = df[possible_def[0]]

    df = df.dropna(subset=["off_ppg", "def_ppg"])
    return df

# --- Simulation function ---
def simulate_game(home, away, df, sims=10000):
    h = df[df["team"] == home].iloc[0]
    a = df[df["team"] == away].iloc[0]

    home_exp = (h["off_ppg"] + a["def_ppg"]) / 2
    away_exp = (a["off_ppg"] + h["def_ppg"]) / 2

    home_scores = np.random.poisson(home_exp, sims)
    away_scores = np.random.poisson(away_exp, sims)

    home_win = np.mean(home_scores > away_scores) * 100
    away_win = np.mean(away_scores > home_scores) * 100

    return home_exp, away_exp, home_win, away_win

# --- UI ---
year = st.number_input("Season", min_value=2023, max_value=2025, value=2024)
st.info("Using CFBD API to load real offensive & defensive PPG stats.")

with st.spinner("Loading data..."):
    df = get_cfb_stats(year)

if df.empty:
    st.stop()

teams = sorted(df["team"].unique())
col1, col2 = st.columns(2)
home_team = col1.selectbox("🏠 Home Team", teams)
away_team = col2.selectbox("✈️ Away Team", teams)

if st.button("Simulate Game"):
    home_exp, away_exp, home_win, away_win = simulate_game(home_team, away_team, df)
    st.markdown(f"""
    ### 🏈 {home_team} vs {away_team}
    - **Expected Points:** {home_team} {home_exp:.1f} — {away_team} {away_exp:.1f}
    - **Win Probability:**
        - {home_team}: {home_win:.2f}%
        - {away_team}: {away_win:.2f}%
    """)
    st.dataframe(df[["team", "off_ppg", "def_ppg"]].sort_values("off_ppg", ascending=False))
