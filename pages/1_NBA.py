# pages/1_NBA.py
# NBA Player Props — 30,000 Sims + Defense + Team Map

import re, unicodedata, time, random
from typing import Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st

# NBA API imports
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog
import nba_api.stats.static.teams as nba_teams
from nba_api.stats.endpoints import commonteamroster

st.title("🏀 NBA Player Prop Model — 30,000 Sims + Defense")

SIM_TRIALS = 30_000

# ✅ Defense ratings table (your list)
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

# -------- Helper functions --------

def normalize_name(n):
    if not n: return ""
    n = str(n).split("(")[0]
    n = re.sub(r"[.,']", " ", n).replace("-", " ").strip()
    return "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining()).lower()

def t_over_prob(mu, sd, line, trials):
    sd = max(sd, 1e-6)
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def sample_sd(s, s2, g):
    if g <= 1: return 0
    mean = s / g
    var = (s2/g) - mean**2
    return max(np.sqrt(max(var * (g/(g-1)), 1e-9)), 0)


# -------- NBA player index --------

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


# -------- Build player→team map --------

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


# -------- Stats functions --------

def fetch_gamelog(pid, season):
    time.sleep(0.25)
    gl = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season")
    return gl.get_data_frames()[0]

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

st.subheader("Season: 2025-26 (Fallback: 2024-25)")

api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"])
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets = st.multiselect("Markets", VALID_MARKETS + sorted(UNSUPPORTED_MARKETS_HIDE), VALID_MARKETS)

def api_get(url, params):
    return requests.get(url, params=params, timeout=20).json()

def get_events():
    return api_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey":api_key,"daysFrom":0,"daysTo":lookahead,"regions":region})

def get_props(eid, mkts):
    return api_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
        {"apiKey":api_key,"regions":region,"markets":",".join(mkts),"oddsFormat":"american"})

if not api_key:
    st.stop()

events = get_events()
if not events: st.stop()

ev_str = [f'{e["away_team"]} @ {e["home_team"]} — {e["commence_time"]}' for e in events]
pick = st.selectbox("Select Game", ev_str)
event = events[ev_str.index(pick)]


# -------- Build averages --------

if st.button("📥 Build Player Averages"):

    props = get_props(event["id"], markets)

    player_names = set()

for bk in props.get("bookmakers", []) or []:
    for m in bk.get("markets", []) or []:
        if m.get("key") in UNSUPPORTED_MARKETS_HIDE:
            continue
        for o in m.get("outcomes", []) or []:
            desc = o.get("description", "")
            # ✅ Skip anything that is not a real player name string
            if not isinstance(desc, str):
                continue
            nm = normalize_name(desc)
            if nm and nm not in ["", "team", "yes", "no"]:
                player_names.add(nm)

# ✅ stop if nothing valid came through
if not player_names:
    st.error("No valid player props found — try selecting points / rebounds / assists markets.")
    st.stop()

    if not player_names:
        st.error("⚠️ No player props found. Add core markets like points/reb/ast.")
        st.stop()

    rows=[]
    for pn in sorted(player_names):
        pid = find_player_id(pn)
        if not pid: continue
        df = fetch_gamelog(pid, "2025-26")
        stats = agg_season(df)
        if stats["g"]==0:
            df = fetch_gamelog(pid, "2024-25")
            stats = agg_season(df)
        if stats["g"]==0: continue
        rows.append({"Player":pn, **stats})

    proj = pd.DataFrame(rows)
    proj["norm"] = proj["Player"].apply(normalize_name)
    st.session_state["proj"] = proj
    st.session_state["player_team"] = build_player_team_map()

    st.success("✅ Player averages ready")
    st.dataframe(proj)


# -------- Run sims --------

if st.button("▶️ Run 30,000 Sims"):

    proj = st.session_state.get("proj")
    team_map = st.session_state.get("player_team")

    if proj is None or proj.empty: st.stop()

    data = get_props(event["id"], markets)
    proj = proj.set_index("norm")

    sim_rows=[]
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE or m["key"] not in VALID_MARKETS:
                continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm in proj.index:
                    sim_rows.append({
                        "market":m["key"],
                        "player":nm,
                        "side":o["name"],
                        "line":float(o["point"])
                    })

    results=[]
    for r in sim_rows:
        row = proj.loc[r["player"]]
        mkt, side, line = r["market"], r["side"], r["line"]

        smap = {
            "player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
            "player_threes":"FG3M","player_blocks":"BLK","player_steals":"STL",
            "player_turnovers":"TOV"
        }

        if mkt in smap:
            mu = row["mu_"+smap[mkt]]
            sd = row["sd_"+smap[mkt]]
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

        # ✅ correct opponent from roster map
        player_team = team_map.get(r["player"])
        home, away = event["home_team"], event["away_team"]
        opponent = away if player_team == home else home

        # ✅ apply defense multiplier
        if opponent in defense_2025:
            mu = mu * defense_2025[opponent]

        # ✅ simulate
        p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
        win = p_over if side=="Over" else 1-p_over

        draws = mu + sd*np.random.standard_t(df=5, size=SIM_TRIALS)
        proj = float(np.median(draws))

        results.append({
            "Player":row["Player"],
            "Market":mkt,
            "Side":side,
            "Line":line,
            "Model Projection":round(proj,2),
            "Win %":round(win*100,2),
            "Opponent":opponent
        })

    df = pd.DataFrame(results).sort_values(["Market","Win %"], ascending=[True, False])
    st.dataframe(df, use_container_width=True)

    st.download_button("Download CSV", df.to_csv(index=False), file_name="nba_model_results.csv")
