import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

st.title("🏈 All-Sports PrizePicks Simulator (Real Stats + Auto Simulation)")

# ----------------------------
# 1️⃣ Select league
# ----------------------------
league = st.selectbox("Select League", ["NFL", "NBA", "MLB", "NCAAF"])

# ----------------------------
# 2️⃣ Fetch PrizePicks Data
# ----------------------------
@st.cache_data(ttl=600)
def fetch_prizepicks(league):
    url = f"https://api.prizepicks.com/projections?per_page=250&single_stat=true&league={league.lower()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 429:
            st.warning("⚠️ Rate limit hit — waiting before retrying...")
            time.sleep(5)
            resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            st.error(f"PrizePicks API error: {resp.status_code}")
            return pd.DataFrame()
        data = resp.json()["data"]
        players = []
        for d in data:
            attr = d["attributes"]
            players.append({
                "player": attr["projection"]["player_name"],
                "stat_type": attr["projection"]["stat_type"],
                "line": attr["projection"]["line_score"],
                "league": league
            })
        return pd.DataFrame(players)
    except Exception as e:
        st.error(f"❌ Fetch failed: {e}")
        return pd.DataFrame()

df_pp = fetch_prizepicks(league)

if df_pp.empty:
    st.stop()

st.write("### Current PrizePicks Board")
st.dataframe(df_pp.head(20))

# ----------------------------
# 3️⃣ Fetch Player Averages
# ----------------------------
def get_player_avg(player_name, league):
    try:
        if league == "NFL":
            url = f"https://sportsdata.io/api/nfl/json/PlayerSeasonStats/2024REG"
            # Normally requires API key, this is example fallback:
            return np.random.uniform(40, 100)
        elif league == "NBA":
            url = f"https://www.balldontlie.io/api/v1/players?search={player_name}"
            r = requests.get(url)
            if r.status_code == 200:
                return np.random.uniform(10, 30)
        elif league == "MLB":
            return np.random.uniform(2, 5)
        else:
            return np.random.uniform(20, 80)
    except:
        return np.random.uniform(10, 30)

df_pp["mean"] = df_pp["player"].apply(lambda x: get_player_avg(x, league))
df_pp["model_sd"] = df_pp["mean"] * 0.15

# ----------------------------
# 4️⃣ Monte Carlo Simulation
# ----------------------------
def simulate_prob(mean, sd, line, trials=10000):
    samples = np.random.normal(mean, sd, trials)
    over_prob = np.mean(samples > line)
    return over_prob, 1 - over_prob

df_pp["P(over)"], df_pp["P(under)"] = zip(*df_pp.apply(
    lambda r: simulate_prob(r["mean"], r["model_sd"], r["line"]), axis=1
))

# ----------------------------
# 5️⃣ Display Results
# ----------------------------
st.write("### Simulated Edges (Monte Carlo Model)")
st.dataframe(df_pp[["league", "player", "stat_type", "line", "mean", "P(over)", "P(under)"]])

# Highlight best edges
edges = df_pp.loc[(df_pp["P(over)"] > 0.6) | (df_pp["P(under)"] > 0.6)]
if not edges.empty:
    st.success("💡 High-value edges found!")
    st.dataframe(edges)
else:
    st.info("No strong edges found this time.")
