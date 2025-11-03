# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 30k sims
# IMPORTANT: Do NOT call st.set_page_config here (it's in your main app).

import re, unicodedata, datetime as dt, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- NBA Stats ----------
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Player Props — Odds API + NBA Stats (live)")

# ✅ changed to 30K sims
SIM_TRIALS = 30_000  

# -------------------- VALID MARKETS (simulated) --------------------
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
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n).lower()

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def sample_sd(sum_x: float, sum_x2: float, g: int, floor: float = 0.0) -> float:
    g = int(g)
    if g <= 1: return float("nan")
    mean = sum_x / g
    var  = (sum_x2 / g) - (mean**2)
    var  = var * (g / (g - 1))
    return float(max(np.sqrt(max(var, 1e-9)), floor))

# ------------------ NBA Stats helpers ------------------
@st.cache_data(show_spinner=False)
def _players_index() -> pd.DataFrame:
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    df["name_norm"]  = df["full_name"].apply(normalize_name)
    return df[["id","full_name","name_norm"]]

def find_player_id_by_name(name: str) -> Optional[int]:
    df = _players_index()
    n = normalize_name(name)
    hit = df.loc[df["name_norm"] == n]
    if not hit.empty:
        return int(hit.iloc[0]["id"])
    parts = n.split()
    if len(parts) == 2:
        last = parts[-1]
        cand = df[df["name_norm"].str.contains(last)]
        if not cand.empty:
            return int(cand.iloc[0]["id"])
    return None

def fetch_player_gamelog_df(player_id: int, season: str, season_type: str) -> pd.DataFrame:
    time.sleep(0.25 + random.random()*0.15)
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star=season_type)
    df = gl.get_data_frames()[0]
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def agg_full_season(df: pd.DataFrame) -> Dict[str, float]:
    g = int(df.shape[0])
    if g == 0:
        return {"g": 0}

    def s(col: str):
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        return float(x.sum()), float((x ** 2).sum())

    sums = {}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        sums[col], sums["sq_"+col] = s(col)

    out = {"g": g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        out["mu_"+col] = sums[col] / g
        out["sd_"+col] = sample_sd(sums[col], sums["sq_"+col], g, floor=0.0)
    return out

# ------------------ UI: season choice ------------------
st.markdown("### 1) Season Locked to 2025-26")
season_locked = "2025-26"

# ------------------ Odds API UI ------------------
st.markdown("### 2) Pick Game & Markets")
api_key = st.text_input("Odds API Key", value="", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)

markets = st.multiselect("Markets to fetch", VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE)), default=VALID_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25); return r.json()

def list_events(api_key, lookahead_days, region):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
                    {"apiKey":api_key,"daysFrom":0,"daysTo":lookahead_days,"regions":region})

def fetch_event_props(api_key, event_id, region, markets):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
                    {"apiKey":api_key,"regions":region,"markets":",".join(markets),"oddsFormat":"american"})

events = []
if api_key:
    events = list_events(api_key, lookahead, region)

if not events:
    st.info("Enter your Odds API key to continue.")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]

# ------------------ Build projections ------------------
st.markdown("### 3) Build Season Averages")
build = st.button("📥 Build NBA projections")

if build:
    props_preview = fetch_event_props(api_key, event["id"], region, list(set(markets)))

    players=set()
    for bk in props_preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o["description"])
                if nm: players.add(nm)

    rows=[]
    for pn in sorted(players):
        pid=find_player_id_by_name(pn)
        if not pid: continue
        df=fetch_player_gamelog_df(pid, season_locked, "Regular Season")
        stats=agg_full_season(df)
        if stats["g"]==0:
            df=fetch_player_gamelog_df(pid, "2024-25", "Regular Season")
            stats=agg_full_season(df)
        if stats["g"]==0: continue
        rows.append({"Player":pn, **stats})

    proj=pd.DataFrame(rows)
    st.session_state["nba_proj"]=proj
    st.success("✅ Season averages loaded")
    st.dataframe(proj)

# ------------------ SIMULATE ------------------
st.markdown("### 4) Simulate Props")
go = st.button("Fetch lines & simulate (NBA)")

if go:
    proj = st.session_state.get("nba_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build projections first")
        st.stop()

    data = fetch_event_props(api_key, event["id"], region, list(set(markets)))

    proj["norm"] = proj["Player"].apply(normalize_name)
    idx = proj.set_index("norm")

    rows=[]
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []):
                name=normalize_name(o["description"])
                if name not in idx.index: continue
                if m["key"] not in VALID_MARKETS: continue
                if o["point"] is None: continue
                rows.append({"market":m["key"],"player":name,"side":o["name"],"line":float(o["point"])})

    results=[]
    for r in rows:
        row=idx.loc[r["player"]]
        mkt=r["market"]; side=r["side"]; line=r["line"]

        mu=sd=None
        if mkt=="player_points":
            mu=row["mu_PTS"]; sd=row["sd_PTS"]
        elif mkt=="player_rebounds":
            mu=row["mu_REB"]; sd=row["sd_REB"]
        elif mkt=="player_assists":
            mu=row["mu_AST"]; sd=row["sd_AST"]
        elif mkt=="player_threes":
            mu=row["mu_FG3M"]; sd=row["sd_FG3M"]
        elif mkt=="player_blocks":
            mu=row["mu_BLK"]; sd=row["sd_BLK"]
        elif mkt=="player_steals":
            mu=row["mu_STL"]; sd=row["sd_STL"]
        elif mkt=="player_turnovers":
            mu=row["mu_TOV"]; sd=row["sd_TOV"]
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

        # win probability
        p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
        p = p_over if side=="Over" else 1 - p_over

        # ✅ **NEW — MODEL PROJECTION = mean of 30k sims**
        sim_draws = mu + sd * np.random.standard_t(df=5, size=SIM_TRIALS)
        proj_avg = float(np.mean(sim_draws))

        results.append({
            "market":mkt,
            "player":row["Player"],
            "side":side,
            "line":round(line,2),
            "Model Projection":round(proj_avg,2),  # ✅ added
            "Win Prob %":round(100*p,2),
        })

    results = pd.DataFrame(results).sort_values(["market","Win Prob %"], ascending=[True,False])
    st.dataframe(results)
    st.download_button("⬇️ Download CSV", results.to_csv(index=False), "nba_props_sim_results.csv", "text/csv")
