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

# --- Fetch team stats from /stats/season endpoint ---
@st.cache_data(ttl=3600)
def get_cfb_team_stats(season: int) -> pd.DataFrame:
    """
    Fetches offensive and defensive points per game using /stats/season CFBD API.
    """
    url = f"https://api.collegefootballdata.com/stats/season?year={season}"
    r = requests.get(url, headers=headers, timeout=30)

    if r.status_code != 200:
        st.error(f"CFBD API error: {r.status_code}")
        return pd.DataFrame()

    try:
        data = r.json()
    except:
        st.error("❌ Could not decode JSON response from CFBD.")
        return pd.DataFrame()

    if not data:
        st.warning(f"No CFBD data found for {season}. Try another year.")
        return pd.DataFrame()

    # Extract points per game stats
    off_data = [d for d in data if d["statName"] == "pointsPerGame" and d["team"]]
    def_data = [d for d in data if d["statName"] == "opponentPointsPerGame" and d["team"]]

    df_off = pd.DataFrame(off_data)[["team", "statValue"]].rename(columns={"statValue": "off_ppg"})
    df_def = pd.DataFrame(def_data)[["team", "statValue"]].rename(columns={"statValue": "def_ppg"})

    df = pd.merge(df_off, df_def, on="team", how="inner")
    df["off_ppg"] = df["off_ppg"].astype(float)
    df["def_ppg"] = df["def_ppg"].astype(float)

    return df


# --- Simulation logic ---
def simulate_matchup(home: str, away: str, df: pd.DataFrame, trials: int = 10000):
    h = df[df["team"] == home].iloc[0]
    a = df[df["team"] == away].iloc[0]

    mu_home = (h["off_ppg"] + a["def_ppg"]) / 2.0
    mu_away = (a["off_ppg"] + h["def_ppg"]) / 2.0

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
season = st.number_input("Season", min_value=2018, max_value=2025, value=2024, step=1)

with st.spinner("Loading CFBD team data..."):
    df = get_cfb_team_stats(int(season))

if df.empty:
    st.stop()

teams = sorted(df["team"].unique())
home = st.selectbox("🏠 Home Team", teams)
away = st.selectbox("✈️ Away Team", [t for t in teams if t != home])

if st.button("Run Simulation"):
    avg_h, avg_a, p_h, p_a = simulate_matchup(home, away, df)
    st.markdown(f"""
    ### {home} vs {away}
    **Expected Points:** {home} {avg_h:.1f} – {away} {avg_a:.1f}  
    **Win Probabilities:** {home} {100 * p_h:.1f}% | {away} {100 * p_a:.1f}%
    """)

with st.expander("📊 View Full Team Stats"):
    st.dataframe(df.sort_values("off_ppg", ascending=False))
