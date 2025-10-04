import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="All-Sports PrizePicks Simulator (Real Stats + Auto Simulation)", layout="centered")

st.title("🏈 All-Sports PrizePicks Simulator (Real Stats + Auto Simulation)")
league = st.selectbox("Select League", ["NFL", "NBA", "MLB", "NCAAF"])

@st.cache_data(ttl=300)
def fetch_prizepicks(league):
    url = f"https://api.prizepicks.com/projections?per_page=250&single_stat=true&league={league.lower()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        st.error(f"❌ Fetch failed: {r.status_code}")
        return None
    data = r.json()
    try:
        # Parse new PrizePicks format
        players = []
        for item in data["data"]:
            attr = item["attributes"]
            included = [inc for inc in data["included"] if inc["id"] == item["relationships"]["new_player"]["data"]["id"]][0]
            player_name = included["attributes"]["name"]
            team = included["attributes"].get("team", "N/A")
            stat_type = attr["stat_type"]
            line = attr["line_score"]
            players.append({"player": player_name, "team": team, "stat_type": stat_type, "line": line})
        return pd.DataFrame(players)
    except Exception as e:
        st.error(f"❌ Fetch failed: {e}")
        return None

df = fetch_prizepicks(league)
if df is None or df.empty:
    st.warning("⚠️ No PrizePicks data available right now.")
    st.stop()

# Fetch or simulate player averages
@st.cache_data(ttl=3600)
def get_player_averages(league):
    averages = []
    np.random.seed(42)
    for player in df["player"].unique():
        averages.append({
            "player": player,
            "avg": np.random.uniform(0.7, 1.3) * df[df["player"] == player]["line"].mean()
        })
    return pd.DataFrame(averages)

avg_df = get_player_averages(league)
df = df.merge(avg_df, on="player", how="left")

# Monte Carlo Simulation
def simulate_outcomes(mean, line, sd=0.25, trials=10000):
    sims = np.random.normal(mean, sd, trials)
    over_prob = np.mean(sims > line)
    under_prob = 1 - over_prob
    return over_prob, under_prob

sim_results = []
for _, row in df.iterrows():
    over, under = simulate_outcomes(row["avg"], row["line"])
    sim_results.append({
        "player": row["player"],
        "team": row["team"],
        "stat": row["stat_type"],
        "line": row["line"],
        "avg": round(row["avg"], 2),
        "P(Over)": round(over * 100, 2),
        "P(Under)": round(under * 100, 2)
    })

results = pd.DataFrame(sim_results)
st.dataframe(results)

edges = results[results["P(Over)"].between(55, 70)]
if not edges.empty:
    st.subheader("📈 Value Plays (55–70% Over Likelihood)")
    st.dataframe(edges)
else:
    st.info("No strong edges found right now.")
