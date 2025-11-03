# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 30k sims

import re, unicodedata, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- NBA Stats ----------
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Player Props — Odds API + NBA Stats (live)")

# ✅ 30,000 simulations
SIM_TRIALS = 30_000

# Defense ratings (2025 season, lower = better defense)
defense_2025 = {
    "Oklahoma City": 1.031,"San Antonio": 1.053,"Portland": 1.073,"Miami": 1.073,
    "Denver": 1.074,"Detroit": 1.076,"Cleveland": 1.084,"Dallas": 1.093,
    "Boston": 1.097,"Orlando": 1.100,"Houston": 1.106,"Golden State": 1.109,
    "Indiana": 1.112,"Philadelphia": 1.116,"Chicago": 1.122,"Atlanta": 1.123,
    "LA Lakers": 1.127,"Milwaukee": 1.133,"Minnesota": 1.135,"Phoenix": 1.137,
    "New York": 1.138,"LA Clippers": 1.141,"Memphis": 1.147,"Charlotte": 1.149,
    "Utah": 1.150,"Toronto": 1.152,"Sacramento": 1.153,"Washington": 1.167,
    "New Orleans": 1.226,"Brooklyn": 1.249
}

VALID_MARKETS = [
    "player_points","player_rebounds","player_assists","player_threes",
    "player_blocks","player_steals","player_blocks_steals","player_turnovers",
    "player_points_rebounds_assists","player_points_rebounds",
    "player_points_assists","player_rebounds_assists",
    "player_field_goals","player_frees_made","player_frees_attempts",
]

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket","player_first_team_basket","player_double_double",
    "player_triple_double","player_points_q1","player_rebounds_q1","player_assists_q1",
}

ODDS_SPORT = "basketball_nba"

# ------------------ Utilities ------------------
def normalize_name(n: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", re.sub(r"[.,']", " ", str(n).split("(")[0]).replace("-", " ")) if not unicodedata.combining(c)).lower().strip()

def t_over_prob(mu, sd, line, trials):
    sd = max(1e-6, sd)
    return float(((mu + sd * np.random.standard_t(df=5, size=trials)) > line).mean())

def sample_sd(s, s2, g):
    if g <= 1: return float("nan")
    m = s/g
    v = (s2/g) - m**2
    return float(max(np.sqrt(max(v * (g/(g-1)), 1e-9)), 0.0))

@st.cache_data(show_spinner=False)
def _players_index():
    df = pd.DataFrame(nba_players.get_players())
    df["norm"] = df["full_name"].apply(normalize_name)
    return df[["id","full_name","norm"]]

def find_player_id_by_name(name):
    df = _players_index()
    name = normalize_name(name)
    hit = df[df["norm"]==name]
    if not hit.empty: return int(hit.iloc[0]["id"])
    part = name.split()
    if len(part)==2:
        cand = df[df["norm"].str.contains(part[-1])]
        if not cand.empty: return int(cand.iloc[0]["id"])
    return None

def fetch_player_gamelog_df(pid, season):
    time.sleep(0.30)
    df = playergamelog.PlayerGameLog(pid, season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def agg_full_season(df):
    g = len(df)
    if g == 0: return {"g":0}
    stats={}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        arr = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        stats["mu_"+col] = arr.mean()
        stats["sd_"+col] = sample_sd(arr.sum(), (arr**2).sum(), g)
    stats["g"]=g
    return stats

# ------------------ UI ------------------
st.markdown("### 1) Season Locked to 2025-26")
season_locked = "2025-26"

st.markdown("### 2) Odds API Inputs")
api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets = st.multiselect("Markets to fetch", VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE)), VALID_MARKETS)

def odds_get(url, p): return requests.get(url, params=p, timeout=25).json()

def list_events():
    return odds_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey":api_key,"daysFrom":0,"daysTo":lookahead,"regions":region})

def fetch_event_props(eid, mkts):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
        {"apiKey":api_key,"regions":region,"markets":",".join(mkts),"oddsFormat":"american"})

if not api_key:
    st.info("Enter your Odds API key")
    st.stop()

events = list_events()
if not events: st.stop()

ev_opts = [f'{e["away_team"]} @ {e["home_team"]} — {e["commence_time"]}' for e in events]
pick = st.selectbox("Select Game", ev_opts)
event = events[ev_opts.index(pick)]

# ------------------ Build Player Stats ------------------
st.markdown("### 3) Load Player Stats")
if st.button("📥 Build NBA projections"):

    preview = fetch_event_props(event["id"], markets)

    names=set()
    for bk in preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m["outcomes"]:
                names.add(normalize_name(o["description"]))

    rows=[]
    for n in sorted(names):
        pid = find_player_id_by_name(n)
        if not pid: continue
        df = fetch_player_gamelog_df(pid, season_locked)
        stats = agg_full_season(df)
        if stats["g"]==0:
            df = fetch_player_gamelog_df(pid, "2024-25")
            stats = agg_full_season(df)
        if stats["g"]==0: continue
        rows.append({"Player":n, **stats})

    st.session_state["proj"]=pd.DataFrame(rows)
    st.dataframe(st.session_state["proj"])

# ------------------ Simulate ------------------
st.markdown("### 4) Simulate Props")
if st.button("Run 30,000-game simulation"):

    proj = st.session_state.get("proj")
    if proj is None or proj.empty: st.stop()

    proj["norm"]=proj["Player"].apply(normalize_name)
    idx = proj.set_index("norm")

    data = fetch_event_props(event["id"], markets)

    rows=[]
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE: continue
            if m["key"] not in VALID_MARKETS: continue
            for o in m["outcomes"]:
                rows.append({
                    "market":m["key"],
                    "player":normalize_name(o["description"]),
                    "side":o["name"],
                    "line":float(o["point"]),
                })

    results=[]
    for r in rows:
        if r["player"] not in idx.index: continue
        row = idx.loc[r["player"]]

        # choose stat (same as your original logic)
        mkt=r["market"]; side=r["side"]; line=r["line"]

        stat_map = {
            "player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
            "player_threes":"FG3M","player_blocks":"BLK","player_steals":"STL",
            "player_turnovers":"TOV"
        }

        if mkt in stat_map:
            mu=row["mu_"+stat_map[mkt]]
            sd=row["sd_"+stat_map[mkt]]
        elif mkt=="player_points_rebounds_assists":
            mu=row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
            sd=np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2+row["sd_AST"]**2)
        elif mkt=="player_points_rebounds":
            mu=row["mu_PTS"]+row["mu_REB"]
            sd=np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2)
        elif mkt=="player_points_assists":
            mu=row["mu_PTS"]+row["mu_AST"]
            sd=np.sqrt(row["sd_PTS"]**2+row["sd_AST"]**2)
        elif mkt=="player_rebounds_assists":
            mu=row["mu_REB"]+row["mu_AST"]
            sd=np.sqrt(row["sd_REB"]**2+row["sd_AST"]**2)
        else:
            continue

        # ✅ defense adjustment — multiply by defense factor
        opponent = event["home_team"] if row["Player"] in event["away_team"].lower() else event["away_team"]
        if opponent in defense_2025:
            mu = mu * defense_2025[opponent]

        # ✅ simulation
        p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
        win = p_over if side=="Over" else 1-p_over

        sim_draws = mu + sd*np.random.standard_t(df=5, size=SIM_TRIALS)
        proj_med = float(np.median(sim_draws))

        results.append({
            "Player":row["Player"],
            "Market":mkt,
            "Side":side,
            "Line":round(line,2),
            "Model Projection":round(proj_med,2),
            "Win %":round(win*100,2),
        })

    res = pd.DataFrame(results).sort_values(["Market","Win %"],ascending=[True,False])
    st.dataframe(res, use_container_width=True)
    st.download_button("Download results CSV", res.to_csv(index=False), "nba_results.csv")
