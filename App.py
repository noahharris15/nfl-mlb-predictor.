# app.py — NFL Player Props (ESPN + Odds API + Defense Scaling)
# Run: streamlit run app.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO
from typing import List, Dict, Optional
from rapidfuzz import process, fuzz

st.set_page_config(page_title="NFL Player Props — ESPN + Odds API", layout="wide")
st.title("🏈 NFL Player Props — ESPN Stats + Odds API (2025)")

SIM_TRIALS = 10_000
EPS = 1e-9
VALID_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_receptions",
    "player_rec_yds"
]

# ---------------- 2025 Defense EPA ----------------
DEFENSE_EPA_2025 = """Team,EPA_Pass,EPA_Rush,Comp_Pct
Minnesota Vikings,-0.37,0.06,0.6762
Jacksonville Jaguars,-0.17,-0.05,0.5962
Denver Broncos,-0.10,-0.12,0.5746
Los Angeles Chargers,-0.17,0.01,0.5938
Detroit Lions,0.00,-0.22,0.6271
Philadelphia Eagles,-0.11,-0.04,0.5693
Houston Texans,-0.16,0.04,0.5714
Los Angeles Rams,-0.12,0.00,0.6640
Seattle Seahawks,0.00,-0.19,0.6645
San Francisco 49ers,-0.09,-0.03,0.6829
Tampa Bay Buccaneers,-0.02,-0.11,0.6429
Atlanta Falcons,-0.13,0.05,0.5769
Cleveland Browns,0.06,-0.17,0.6442
Indianapolis Colts,-0.04,-0.05,0.6643
Kansas City Chiefs,-0.09,0.09,0.6694
Arizona Cardinals,0.06,-0.14,0.6369
Las Vegas Raiders,0.14,-0.22,0.6565
Green Bay Packers,0.03,-0.07,0.6815
Chicago Bears,0.01,0.00,0.7368
Buffalo Bills,-0.06,0.10,0.6214
Carolina Panthers,0.03,0.05,0.6239
Pittsburgh Steelers,0.11,-0.05,0.6957
Washington Commanders,0.18,-0.12,0.6098
New England Patriots,0.19,-0.15,0.7120
New York Giants,-0.01,0.19,0.6375
New Orleans Saints,0.20,-0.06,0.7117
Cincinnati Bengals,0.13,0.04,0.6536
New York Jets,0.23,-0.03,0.6577
Tennessee Titans,0.16,0.07,0.6984
Baltimore Ravens,0.14,0.12,0.6667
Dallas Cowboys,0.40,0.06,0.7333
Miami Dolphins,0.34,0.12,0.7757
"""

@st.cache_data(show_spinner=False)
def load_defense_table():
    df = pd.read_csv(StringIO(DEFENSE_EPA_2025))
    for c in ["EPA_Pass", "EPA_Rush", "Comp_Pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    def adj(s, scale):
        out = 1.0 - scale * s.fillna(0)
        return out.clip(0.7, 1.3)
    pass_adj = adj(df["EPA_Pass"], 0.8)
    rush_adj = adj(df["EPA_Rush"], 0.8)
    comp = df["Comp_Pct"].clip(0.45, 0.8)
    comp_adj = (1 + (comp - comp.mean()) * 0.6).clip(0.7, 1.3)
    recv_adj = (0.7 * pass_adj + 0.3 * comp_adj).clip(0.7, 1.3)
    return pd.DataFrame({"Team": df["Team"], "pass_adj": pass_adj, "rush_adj": rush_adj, "recv_adj": recv_adj})

DEF_TABLE = load_defense_table()

# ---------------- ESPN Stats ----------------
st.header("1️⃣ Pulling Live ESPN Player Stats")

@st.cache_data(show_spinner=True)
def load_espn_2025_stats() -> pd.DataFrame:
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/statistics"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("ESPN API unavailable.")
        return pd.DataFrame()
    js = r.json()
    tables = js.get("stats", [])
    rows = []
    for t in tables:
        for cat in t.get("categories", []):
            for stat in cat.get("stats", []):
                name = stat.get("athlete", {}).get("displayName")
                if not name:
                    continue
                values = stat.get("stats", {})
                rows.append({
                    "Player": name,
                    "team": stat.get("team", {}).get("displayName"),
                    "rush_yds": float(values.get("rushingYards", 0) or 0),
                    "rec_yds": float(values.get("receivingYards", 0) or 0),
                    "receptions": float(values.get("receptions", 0) or 0),
                    "pass_yds": float(values.get("passingYards", 0) or 0),
                    "pass_tds": float(values.get("passingTouchdowns", 0) or 0),
                    "games": float(values.get("gamesPlayed", 1) or 1)
                })
    df = pd.DataFrame(rows)
    df["rush_ypg"] = df["rush_yds"] / df["games"]
    df["rec_ypg"] = df["rec_yds"] / df["games"]
    df["receptions_pg"] = df["receptions"] / df["games"]
    df["pass_ypg"] = df["pass_yds"] / df["games"]
    df["pass_tdp"] = df["pass_tds"] / df["games"]
    return df

stats_df = load_espn_2025_stats()
if stats_df.empty:
    st.stop()

# ---------------- Opponent Scaling ----------------
st.header("2️⃣ Choose Opponent Defense Scaling")
opp_team = st.selectbox("Opponent Defense", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()

# ---------------- Priors ----------------
def trimmed_mean(s, trim=0.2):
    s = s.dropna().sort_values()
    if s.empty: return np.nan
    n = len(s); k = int(n * trim)
    return float(s.iloc[k:n - k].mean() if n > 2 * k else s.mean())

rushers = stats_df[stats_df["rush_ypg"] >= 15]
receivers = stats_df[stats_df["receptions_pg"] >= 1.5]
passers = stats_df[stats_df["pass_ypg"] >= 100]

lg_mu_rush_yds = trimmed_mean(rushers["rush_ypg"])
lg_mu_rec_yds = trimmed_mean(receivers["rec_ypg"])
lg_mu_receptions = trimmed_mean(receivers["receptions_pg"])
lg_mu_pass_yds = trimmed_mean(passers["pass_ypg"])
lg_mu_pass_tds = trimmed_mean(passers["pass_tdp"])

def shrink(mu, prior, w=0.15):
    if pd.isna(mu) or pd.isna(prior): return mu
    return (1 - w) * mu + w * prior

for col in [
    "rush_ypg", "rec_ypg", "receptions_pg", "pass_ypg", "pass_tdp"
]:
    stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")

stats_df["rush_ypg_adj"] = stats_df["rush_ypg"].apply(lambda x: shrink(x, lg_mu_rush_yds)) * scalers["rush_adj"]
stats_df["rec_ypg_adj"] = stats_df["rec_ypg"].apply(lambda x: shrink(x, lg_mu_rec_yds)) * scalers["recv_adj"]
stats_df["receptions_pg_adj"] = stats_df["receptions_pg"].apply(lambda x: shrink(x, lg_mu_receptions)) * scalers["recv_adj"]
stats_df["pass_ypg_adj"] = stats_df["pass_ypg"].apply(lambda x: shrink(x, lg_mu_pass_yds)) * scalers["pass_adj"]
stats_df["pass_tdp_adj"] = stats_df["pass_tdp"].apply(lambda x: shrink(x, lg_mu_pass_tds)) * scalers["pass_adj"]

# ---------------- SD Estimates ----------------
stats_df["sd_rush_yds"] = (stats_df["rush_ypg_adj"] * 0.25).clip(lower=5)
stats_df["sd_rec_yds"] = (stats_df["rec_ypg_adj"] * 0.35).clip(lower=5)
stats_df["sd_receptions"] = (stats_df["receptions_pg_adj"] * 0.35).clip(lower=0.5)
stats_df["sd_pass_yds"] = (stats_df["pass_ypg_adj"] * 0.20).clip(lower=25)
stats_df["sd_pass_tds"] = (stats_df["pass_tdp_adj"] * 0.60).clip(lower=0.25)

# ---------------- Odds API ----------------
st.header("3️⃣ Fetch Props & Run Simulations (10,000 trials)")
api_key = st.text_input("Odds API Key", value="", type="password")
region = st.selectbox("Region", ["us", "us2", "eu", "uk"], index=0)
lookahead = st.slider("Lookahead (days)", 0, 7, value=1)
if not api_key:
    st.stop()

def odds_get(url, params):
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_nfl_events(api_key, lookahead, region):
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
    params = {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead, "regions": region}
    return odds_get(base, params)

def fetch_event_props(api_key, event_id, region):
    base = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": ",".join(VALID_MARKETS), "oddsFormat": "american"}
    return odds_get(base, params)

try:
    events = list_nfl_events(api_key, lookahead, region)
except Exception as e:
    st.error(f"Event fetch error: {e}")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]}' for e in events]
event_pick = st.selectbox("Choose Game", event_labels)
event_id = events[event_labels.index(event_pick)]["id"]

if st.button("Fetch & Simulate"):
    try:
        data = fetch_event_props(api_key, event_id, region)
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            key = m.get("key")
            for o in m.get("outcomes", []):
                player = o.get("description")
                side = o.get("name")
                point = o.get("point")
                if key not in VALID_MARKETS or not player or point is None:
                    continue
                rows.append({"market": key, "player": player, "side": side, "line": float(point)})

    if not rows:
        st.warning("No player props found.")
        st.stop()

    odds_df = pd.DataFrame(rows).drop_duplicates()

    def simulate(mu, sd, line):
        sd = max(sd, 1.0)
        return round((np.random.normal(mu, sd, SIM_TRIALS) > line).mean() * 100, 2)

    results = []
    for _, row in odds_df.iterrows():
        name = row["player"]; line = row["line"]; market = row["market"]; side = row["side"]
        col_map = {
            "player_pass_yds": ("pass_ypg_adj", "sd_pass_yds"),
            "player_pass_tds": ("pass_tdp_adj", "sd_pass_tds"),
            "player_rush_yds": ("rush_ypg_adj", "sd_rush_yds"),
            "player_receptions": ("receptions_pg_adj", "sd_receptions"),
            "player_rec_yds": ("rec_ypg_adj", "sd_rec_yds")
        }
        if market not in col_map: continue
        mu_col, sd_col = col_map[market]
        match = process.extractOne(name, stats_df["Player"].tolist(), scorer=fuzz.token_sort_ratio)
        if not match or match[1] < 85:
            continue
        player = match[0]
        p = stats_df[stats_df["Player"] == player].iloc[0]
        mu, sd = p[mu_col], p[sd_col]
        p_over = simulate(mu, sd, line)
        prob = p_over if side == "Over" else 100 - p_over
        results.append({
            "Player": player,
            "Market": market,
            "Side": side,
            "Line": round(line, 2),
            "μ": round(mu, 2),
            "σ": round(sd, 2),
            "Win Prob %": round(prob, 2)
        })

    if not results:
        st.warning("No matched player projections.")
    else:
        res = pd.DataFrame(results).sort_values("Win Prob %", ascending=False)
        st.dataframe(res, use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            res.to_csv(index=False).encode("utf-8"),
            file_name="props_sim_results.csv",
            mime="text/csv"
        )
