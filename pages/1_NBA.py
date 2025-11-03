# NBA Player Props — Odds API + NBA Stats — 30,000 sims + Defense Model

import re, unicodedata, time, random
from typing import Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog
import nba_api.stats.static.teams as nba_teams
from nba_api.stats.endpoints import commonteamroster

st.title("🏀 NBA Player Props — Model + 30k Sims + Defense Adjustment")

SIM_TRIALS = 30_000

# ✅ Defense ratings table (2025, lower = better defense)
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


# -------- Utility functions --------

def normalize_name(n):
    n = str(n or "").split("(")[0]
    n = re.sub(r"[.,']", " ", n).replace("-", " ").strip()
    return "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining()).lower()

def sample_sd(sum_x, sum_x2, g):
    if g <= 1: return 0
    mean = sum_x / g
    var = (sum_x2/g) - (mean**2)
    var = var * (g/(g-1))
    return max(np.sqrt(max(var,1e-9)), 0)

def t_over_prob(mu, sd, line, trials):
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())


# -------- NBA player ID index --------

@st.cache_data(show_spinner=False)
def _players_index():
    df = pd.DataFrame(nba_players.get_players())
    df["norm"] = df["full_name"].apply(normalize_name)
    return df[["id","full_name","norm"]]

def find_player_id(name):
    df = _players_index()
    name = normalize_name(name)
    hit = df[df["norm"]==name]
    if not hit.empty: return int(hit.iloc[0]["id"])
    parts = name.split()
    if len(parts)==2:
        cand = df[df["norm"].str.contains(parts[-1])]
        if not cand.empty: return int(cand.iloc[0]["id"])
    return None


# -------- Team roster map (Player → Team) --------

@st.cache_data(show_spinner=False)
def build_player_team_map():
    teams = nba_teams.get_teams()
    player_team = {}
    for t in teams:
        tid = t["id"]
        try:
            df = commonteamroster.CommonTeamRoster(team_id=tid, season="2025-26").get_data_frames()[0]
        except:
            df = commonteamroster.CommonTeamRoster(team_id=tid, season="2024-25").get_data_frames()[0]
        for _, row in df.iterrows():
            player_team[normalize_name(row["PLAYER"])] = t["full_name"]
    return player_team


# -------- Fetch gamelog & compute stats --------

def fetch_gamelog(pid, season):
    time.sleep(0.3)
    gl = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season")
    df = gl.get_data_frames()[0]
    return df

def agg_season(df):
    g = len(df)
    if g == 0: return {"g":0}
    stats = {"g":g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        stats["mu_"+col] = x.mean()
        stats["sd_"+col] = sample_sd(x.sum(), (x**2).sum(), g)
    return stats


# -------- UI --------

st.markdown("### 1) Season fixed to 2025-26 (falls back to 24-25)")

api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"])
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets = st.multiselect("Markets", VALID_MARKETS + sorted(UNSUPPORTED_MARKETS_HIDE), VALID_MARKETS)

def api_get(url, p):
    return requests.get(url, params=p, timeout=30).json()

def get_events():
    return api_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey":api_key,"daysFrom":0,"daysTo":lookahead,"regions":region})

def get_props(eid, mkts):
    return api_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
        {"apiKey":api_key,"regions":region,"markets":",".join(mkts),"oddsFormat":"american"})

if not api_key:
    st.info("Enter Odds API key to continue")
    st.stop()

events = get_events()
if not events:
    st.error("No events found")
    st.stop()

event_options = [f'{e["away_team"]} @ {e["home_team"]} — {e["commence_time"]}' for e in events]
pick = st.selectbox("Select Game", event_options)
event = events[event_options.index(pick)]


# -------- Step 3: Build NBA stats --------

if st.button("📥 Build Player Averages"):
    props = get_props(event["id"], markets)
    player_names=set()

    for bk in props.get("bookmakers",[]):
        for m in bk.get("markets",[]):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes",[]):
                player_names.add(normalize_name(o["description"]))

    rows=[]
    for pn in sorted(player_names):
        pid = find_player_id(pn)
        if not pid: continue

        df = fetch_gamelog(pid, "2025-26")
        stats = agg_season(df)
        if stats["g"] == 0:
            df = fetch_gamelog(pid, "2024-25")
            stats = agg_season(df)
        if stats["g"] == 0: continue

        rows.append({"Player":pn, **stats})

    proj = pd.DataFrame(rows)
    proj["norm"] = proj["Player"].apply(normalize_name)
    st.session_state["proj"] = proj
    st.session_state["player_team"] = build_player_team_map()

    st.success("✅ Player averages loaded")
    st.dataframe(proj)


# -------- Step 4: Simulate --------

if st.button("▶️ Run 30,000 Sims"):

    proj = st.session_state.get("proj")
    player_team_map = st.session_state.get("player_team")

    if proj is None or proj.empty:
        st.warning("Build projections first")
        st.stop()

    data = get_props(event["id"], markets)

    proj = proj.set_index("norm")

    sim_rows=[]
    for bk in data.get("bookmakers",[]):
        for m in bk.get("markets",[]):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE or m["key"] not in VALID_MARKETS:
                continue
            for o in m["outcomes"]:
                name = normalize_name(o["description"])
                if name not in proj.index: continue
                sim_rows.append({
                    "market":m["key"],
                    "player":name,
                    "side":o["name"],
                    "line":float(o["point"])
                })

    results=[]
    for r in sim_rows:
        row = proj.loc[r["player"]]

        # select stats
        mkt = r["market"]
        side = r["side"]
        line = r["line"]

        stat_map = {
            "player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
            "player_threes":"FG3M","player_blocks":"BLK","player_steals":"STL",
            "player_turnovers":"TOV"
        }

        if mkt in stat_map:
            mu = row["mu_"+stat_map[mkt]]
            sd = row["sd_"+stat_map[mkt]]
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

        # ✅ correct opponent lookup via NBA roster
        player_team = player_team_map.get(r["player"])
        home, away = event["home_team"], event["away_team"]

        if player_team == home:
            opponent = away
        elif player_team == away:
            opponent = home
        else:
            opponent = home  # fallback

        # ✅ apply defense multiplier
        if opponent in defense_2025:
            mu = mu * defense_2025[opponent]

        # ✅ simulate
        p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
        win = p_over if side=="Over" else 1-p_over

        draws = mu + sd*np.random.standard_t(df=5, size=SIM_TRIALS)
        proj_med = float(np.median(draws))

        results.append({
            "Player":row["Player"].title(),
            "Market":mkt,
            "Side":side,
            "Line":line,
            "Model Projection":round(proj_med,2),
            "Win %":round(win*100,2),
            "Opponent":opponent
        })

    res = pd.DataFrame(results).sort_values(["Market","Win %"], ascending=[True,False])
    st.dataframe(res, use_container_width=True)
    st.download_button("Download CSV", res.to_csv(index=False), "nba_model_results.csv")
