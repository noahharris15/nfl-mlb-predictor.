# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 10k sims
# Place this file at: pages/1_NBA.py
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

SIM_TRIALS = 10_000

# -------------------- VALID MARKETS (simulated) --------------------
VALID_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_blocks_steals",
    "player_turnovers",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_rebounds_assists",
    "player_field_goals",
    "player_frees_made",
    "player_frees_attempts",
]

# We will fetch these if selected, but hide from output until modeled:
UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket",
    "player_first_team_basket",
    "player_double_double",
    "player_triple_double",
    "player_points_q1",
    "player_rebounds_q1",
    "player_assists_q1",
}

ODDS_SPORT = "basketball_nba"  # Odds API sport key

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
    df["first_norm"] = df["first_name"].apply(normalize_name)
    df["last_norm"]  = df["last_name"].apply(normalize_name)
    return df[["id","full_name","name_norm","first_norm","last_norm"]]

def find_player_id_by_name(name: str) -> Optional[int]:
    df = _players_index()
    n = normalize_name(name)
    parts = n.split()

    hit = df.loc[df["name_norm"] == n]
    if not hit.empty:
        return int(hit.iloc[0]["id"])

    if len(parts) == 2:
        first, last = parts
        cand = df.loc[df["last_norm"].str.startswith(last)]
        if cand.empty: cand = df.loc[df["last_norm"].str.contains(last)]
        if not cand.empty:
            cand = cand.loc[cand["first_norm"].str.startswith(first[:1]) | cand["first_norm"].str.contains(first)]
            if not cand.empty: return int(cand.iloc[0]["id"])

    last = parts[-1] if parts else n
    cand = df.loc[df["name_norm"].str.contains(last)]
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

    def s(col: str) -> Tuple[float, float]:
        if col not in df.columns: return 0.0, 0.0
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

# ------------------ UI ------------------
st.markdown("### 1) Season Locked to 2025-26")
season_locked = "2025-26"
st.caption(f"Using NBA season: **{season_locked}** only")

# ------------------ Odds API ------------------
st.markdown("### 2) Pick a game & markets (Odds API)")
api_key = st.text_input("Odds API Key", value="", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets_pickable = VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE))
markets = st.multiselect("Markets", markets_pickable, default=VALID_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200: raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()

def list_events():
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "daysFrom":0, "daysTo":lookahead, "regions":region}
    )

def fetch_event_props(eid, mkts):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
        {"apiKey": api_key, "regions":region, "markets":",".join(mkts)}
    )

if not api_key: st.stop()
events = list_events()
if not events: st.stop()

# ✅ FIXED LABELS (no emojis/commence time)
event_labels = [f'{e["away_team"]} @ {e["home_team"]}' for e in events]
pick = st.selectbox("Game", event_labels)
event = next(e for e in events if f'{e["away_team"]} @ {e["home_team"]}' == pick)
event_id = event["id"]

# ------------------ Build projections ------------------
st.markdown("### 3) Build per-player projections from stats")
build = st.button("📥 Build NBA projections")

if build:
    preview = fetch_event_props(event_id, list(set(markets)))
    player_names = set()

    for bk in preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []) or []:
                nm = normalize_name(o.get("description"))
                if nm: player_names.add(nm)

    rows = []
    missing = {}

    for pn in sorted(player_names):
        pid = find_player_id_by_name(pn)
        if not pid: 
            missing[pn] = "player_not_found"
            continue

        df = fetch_player_gamelog_df(pid, season_locked, "Regular Season")
        stats = agg_full_season(df)

        if stats["g"] == 0:
            df = fetch_player_gamelog_df(pid, "2024-25", "Regular Season")
            stats = agg_full_season(df)

        if stats["g"] == 0:
            missing[pn] = "no_games"
            continue

        rows.append({"Player":pn, **stats})

    if missing:
        with st.expander("Missing players info"):
            st.json(missing)

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    st.session_state["nba_proj"] = proj
    st.success("✅ Projections built")
    st.dataframe(proj.head(25))

# ------------------ Simulate ------------------
st.markdown("### 4) Simulate Props (10k draws)")
go = st.button("Run Simulation")

if go:
    proj = st.session_state.get("nba_proj", pd.DataFrame())
    if proj.empty: st.stop()

    data = fetch_event_props(event_id, list(set(markets)))
    proj = proj.set_index("player_norm")

    out_rows = []
    rows = []

    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                if not name or name not in proj.index or mkey not in VALID_MARKETS: continue
                rows.append({"market":mkey, "player_norm":name, "side":o["name"], "point":float(o["point"])})

    props_df = pd.DataFrame(rows).groupby(["market","player_norm","side"], as_index=False).agg(line=("point","median"))

    for _, r in props_df.iterrows():
        mkt, name, side, line = r["market"], r["player_norm"], r["side"], r["line"]
        row = proj.loc[name]

        if mkt == "player_points": mu = row["mu_PTS"]; sd = row["sd_PTS"]
        elif mkt == "player_rebounds": mu = row["mu_REB"]; sd = row["sd_REB"]
        elif mkt == "player_assists": mu = row["mu_AST"]; sd = row["sd_AST"]
        elif mkt == "player_threes": mu = row["mu_FG3M"]; sd = row["sd_FG3M"]
        elif mkt == "player_blocks": mu = row["mu_BLK"]; sd = row["sd_BLK"]
        elif mkt == "player_steals": mu = row["mu_STL"]; sd = row["sd_STL"]
        elif mkt == "player_turnovers": mu = row["mu_TOV"]; sd = row["sd_TOV"]
        elif mkt == "player_field_goals": mu = row["mu_FGM"]; sd = row["sd_FGM"]
        elif mkt == "player_frees_made": mu = row["mu_FTM"]; sd = row["sd_FTM"]
        elif mkt == "player_frees_attempts": mu = row["mu_FTA"]; sd = row["sd_FTA"]
        elif mkt == "player_points_rebounds_assists":
            mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
            sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)
        elif mkt == "player_points_rebounds":
            mu = row["mu_PTS"]+row["mu_REB"]; sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2)
        elif mkt == "player_points_assists":
            mu = row["mu_PTS"]+row["mu_AST"]; sd = np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2)
        elif mkt == "player_rebounds_assists":
            mu = row["mu_REB"]+row["mu_AST"]; sd = np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2)
        elif mkt == "player_blocks_steals":
            mu = row["mu_BLK"]+row["mu_STL"]; sd = np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2)
        else:
            continue

        p_over = t_over_prob(float(mu), float(sd), float(line), SIM_TRIALS)
        p = p_over if side == "Over" else 1-p_over

        out_rows.append({
            "market":mkt,
            "player":row["Player"],
            "side":side,
            "line":round(float(line), 2),
            "Avg":round(float(mu),2),
            "SD":round(float(sd),2),
            "Win %":round(p*100, 2),
        })

    results = pd.DataFrame(out_rows).sort_values(["market","Win %"], ascending=[True,False])
    st.dataframe(results, use_container_width=True)
    st.download_button("Download", results.to_csv(index=False).encode(), "nba_sim_results.csv")
