# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 10k sims
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
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_blocks", "player_steals", "player_blocks_steals", "player_turnovers",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
    "player_field_goals", "player_frees_made", "player_frees_attempts",
]

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket","player_first_team_basket","player_double_double",
    "player_triple_double","player_points_q1","player_rebounds_q1","player_assists_q1"
}

ODDS_SPORT = "basketball_nba"

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
        if cand.empty:
            cand = df.loc[df["last_norm"].str.contains(last)]
        if not cand.empty:
            cand = cand.loc[cand["first_norm"].str.startswith(first[:1]) | cand["first_norm"].str.contains(first)]
            if not cand.empty:
                return int(cand.iloc[0]["id"])

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
    if g == 0: return {"g": 0}
    
    def s(col: str) -> Tuple[float, float]:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        return float(x.sum()), float((x**2).sum())

    sums, out = {}, {"g": g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        sums[col], sums["sq_"+col] = s(col)
        out["mu_"+col] = sums[col] / g
        out["sd_"+col] = sample_sd(sums[col], sums["sq_"+col], g)
    return out

st.markdown("### 1) Season Locked to 2025-26")
season_locked = "2025-26"

# ------------------ Odds API ------------------
st.markdown("### 2) Pick a game & markets (Odds API)")
api_key = st.text_input("Odds API Key", value="", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, 1)
markets_pickable = VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE))
markets = st.multiselect("Markets to fetch", markets_pickable, default=VALID_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    return r.json()

def list_events(api_key: str, lookahead: int, region: str):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "daysFrom":0, "daysTo":lookahead, "regions":region}
    )

if api_key:
    events = list_events(api_key, lookahead, region)
else:
    events = []

if not events:
    st.info("Enter API key & lookahead to load games")
    st.stop()

### ✅ FIXED — removed f-string in list loop (was causing your error)
event_labels = []
for e in events:
    away = e["away_team"]
    home = e["home_team"]
    event_labels.append(f"{away} @ {home}")

pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

def fetch_event_props(api_key, event_id, region, mkts):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions":region, "markets":",".join(mkts), "oddsFormat":"american"}
    )

# ------------------ Build projections ------------------
st.markdown("### 3) Build per-player projections")
build = st.button("📥 Build NBA projections")

if build:
    preview = fetch_event_props(api_key, event_id, region, list(set(markets)))
    player_names = set()

    for bk in preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm: player_names.add(nm)

    rows = []
    for pn in sorted(player_names):
        pid = find_player_id_by_name(pn)
        if not pid: continue

        df = fetch_player_gamelog_df(pid, season_locked, "Regular Season")
        ag = agg_full_season(df)

        if ag["g"] == 0:
            df2 = fetch_player_gamelog_df(pid, "2024-25", "Regular Season")
            ag = agg_full_season(df2)

        if ag["g"] > 0:
            rows.append({"Player":pn, **ag})

    st.session_state["nba_proj"] = pd.DataFrame(rows)
    st.success("✅ Player averages loaded")
    st.dataframe(st.session_state["nba_proj"], use_container_width=True)

# ------------------ Run Simulation ------------------
st.markdown("### 4) Fetch lines & simulate (sharper)")
go = st.button("Fetch lines & simulate (NBA)")

if go:
    proj = st.session_state.get("nba_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build projections first")
        st.stop()

    props = fetch_event_props(api_key, event_id, region, list(set(markets)))
    proj = proj.copy()
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    idx = proj.set_index("player_norm")

    rows = []
    for bk in props.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey in UNSUPPORTED_MARKETS_HIDE or mkey not in VALID_MARKETS: continue

            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm not in idx.index: continue

                side = o["name"]
                line = float(o["point"])
                row = idx.loc[nm]

                # ---- Stat mapping (unchanged) ----
                def get(col):
                    return row[f"mu_{col}"], row[f"sd_{col}"]

                if mkey=="player_points":   mu,sd = get("PTS")
                elif mkey=="player_rebounds": mu,sd = get("REB")
                elif mkey=="player_assists":  mu,sd = get("AST")
                elif mkey=="player_threes":   mu,sd = get("FG3M")
                elif mkey=="player_blocks":   mu,sd = get("BLK")
                elif mkey=="player_steals":   mu,sd = get("STL")
                elif mkey=="player_turnovers": mu,sd = get("TOV")
                elif mkey=="player_field_goals": mu,sd = get("FGM")
                elif mkey=="player_frees_made": mu,sd = get("FTM")
                elif mkey=="player_frees_attempts": mu,sd = get("FTA")
                elif mkey=="player_points_rebounds_assists":
                    mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2+row["sd_AST"]**2)
                elif mkey=="player_points_rebounds":
                    mu = row["mu_PTS"]+row["mu_REB"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2)
                elif mkey=="player_points_assists":
                    mu = row["mu_PTS"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_AST"]**2)
                elif mkey=="player_rebounds_assists":
                    mu = row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_REB"]**2+row["sd_AST"]**2)
                elif mkey=="player_blocks_steals":
                    mu = row["mu_BLK"]+row["mu_STL"]
                    sd = np.sqrt(row["sd_BLK"]**2+row["sd_STL"]**2)
                else:
                    continue

                # ---- ✅ SHARPENED MODEL PROJECTION ----
                draws = mu + sd * np.random.standard_t(df=5, size=SIM_TRIALS)
                proj_val = float(np.median(draws))  # 🔥 median more accurate
                p_over = float((draws > line).mean())
                win = p_over if side=="Over" else (1-p_over)

                rows.append({
                    "market": mkey,
                    "player": row["Player"],
                    "side": side,
                    "line": round(line,2),
                    "Avg (raw)": round(mu,2),
                    "Model Projection": round(proj_val,2),  # ✅ output projection
                    "Win Prob %": round(win*100,2),
                })

    results = pd.DataFrame(rows).sort_values(["market","Win Prob %"], ascending=[True, False])
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="nba_props_sim_results_sharp.csv",
        mime="text/csv",
    )
