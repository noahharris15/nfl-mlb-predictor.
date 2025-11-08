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

# ✅ DEFENSE MULTIPLIERS (Converted from your rankings)
DEFENSE_MULTIPLIERS = {
    "Oklahoma City Thunder": 1.031,
    "San Antonio Spurs": 1.053,
    "Portland Trail Blazers": 1.073,
    "Miami Heat": 1.073,
    "Denver Nuggets": 1.074,
    "Detroit Pistons": 1.076,
    "Cleveland Cavaliers": 1.084,
    "Dallas Mavericks": 1.093,
    "Boston Celtics": 1.097,
    "Orlando Magic": 1.100,
    "Houston Rockets": 1.106,
    "Golden State Warriors": 1.109,
    "Indiana Pacers": 1.112,
    "Philadelphia 76ers": 1.116,
    "Chicago Bulls": 1.122,
    "Atlanta Hawks": 1.123,
    "Los Angeles Lakers": 1.127,
    "Milwaukee Bucks": 1.133,
    "Minnesota Timberwolves": 1.135,
    "Phoenix Suns": 1.137,
    "New York Knicks": 1.138,
    "Los Angeles Clippers": 1.141,
    "Memphis Grizzlies": 1.147,
    "Charlotte Hornets": 1.149,
    "Utah Jazz": 1.150,
    "Toronto Raptors": 1.152,
    "Sacramento Kings": 1.153,
    "Washington Wizards": 1.167,
    "New Orleans Pelicans": 1.226,
    "Brooklyn Nets": 1.249,
}

# -------------------- VALID MARKETS --------------------
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

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket",
    "player_first_team_basket",
    "player_double_double",
    "player_triple_double",
    "player_points_q1",
    "player_rebounds_q1",
    "player_assists_q1",
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

def t_distribution_draws(mu: float, sd: float, trials: int = SIM_TRIALS):
    sd = max(1e-6, float(sd))
    return mu + sd * np.random.standard_t(df=5, size=trials)

def sample_sd(sum_x: float, sum_x2: float, g: int, floor: float = 0.0) -> float:
    if g <= 1:
        return float("nan")
    mean = sum_x / g
    var = (sum_x2 / g) - (mean**2)
    var = var * (g / (g - 1))
    return float(max(np.sqrt(max(var, 1e-9)), floor))

# ------------------ NBA Stats helpers ------------------
@st.cache_data(show_spinner=False)
def _players_index() -> pd.DataFrame:
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    df["name_norm"] = df["full_name"].apply(normalize_name)
    df["first_norm"] = df["first_name"].apply(normalize_name)
    df["last_norm"] = df["last_name"].apply(normalize_name)
    return df[["id", "full_name", "name_norm", "first_norm", "last_norm"]]

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
            cand = cand.loc[cand["first_norm"].str.startswith(first[:1]) |
                            cand["first_norm"].str.contains(first)]
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
    if "GAME_DATE" in df:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def agg_full_season(df: pd.DataFrame) -> Dict[str, float]:
    g = len(df)
    if g == 0:
        return {"g": 0}

    def s(col):
        arr = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        return float(arr.sum()), float((arr**2).sum())

    sums = {}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        sums[col], sums["sq_"+col] = s(col)

    out = {"g": g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        out["mu_"+col] = sums[col]/g
        out["sd_"+col] = sample_sd(sums[col], sums["sq_"+col], g)
    return out

# ------------------ Odds API ------------------
def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    return r.json()

def list_events(api_key, lookahead, region):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "regions": region, "daysFrom": 0, "daysTo": lookahead or 1},
    )

def fetch_event_props(api_key, event_id, region, markets):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {
            "apiKey": api_key,
            "regions": region,
            "markets": ",".join(markets),
            "oddsFormat": "american",
        },
    )

# ------------------ UI ------------------
st.markdown("### 1) Season Locked to 2025-26")
season_locked = "2025-26"

api_key = st.text_input("Odds API Key", "", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"])
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets_pickable = VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE))
markets = st.multiselect("Markets", markets_pickable, default=VALID_MARKETS)

events = list_events(api_key, lookahead, region) if api_key else []

if not events:
    st.info("Enter API key.")
    st.stop()

event_labels = []
for e in events:
    away = e.get("away_team") or (e.get("teams") or ["Away","Home"])[0]
    home = e.get("home_team") or (e.get("teams") or ["Away","Home"])[1]
    date = e.get("commence_time", "")
    event_labels.append(f"{away} @ {home} — {date}")

pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ------------------ Build Projections ------------------
st.markdown("### 3) Build Projections")
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
        if not pid:
            continue

        df = fetch_player_gamelog_df(pid, season_locked, "Regular Season")
        stats = agg_full_season(df)

        if stats["g"] == 0:
            df2 = fetch_player_gamelog_df(pid, "2024-25", "Regular Season")
            stats2 = agg_full_season(df2)
            if stats2["g"] == 0:
                continue
            df, stats = df2, stats2

        rows.append({
            "Player": pn,
            **stats,
            "game_log": df
        })

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    st.session_state["nba_proj"] = proj

    st.dataframe(proj)

# ------------------ Simulation ------------------
st.markdown("### 4) Fetch lines & simulate")
go = st.button("Run Simulation")

if go:

    proj = st.session_state.get("nba_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build projections first.")
        st.stop()

    props = fetch_event_props(api_key, event_id, region, list(set(markets)))
    proj = proj.set_index("player_norm")

    rows = []

    # ✅ Get opponent team
    home_team = event.get("home_team")
    away_team = event.get("away_team")

    for bk in props.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey in UNSUPPORTED_MARKETS_HIDE:
                continue

            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                if name not in proj.index:
                    continue
                if mkey not in VALID_MARKETS:
                    continue

                row = proj.loc[name]
                df = row["game_log"]
                side = o["name"]
                line = float(o["point"])

                # -----------------------------------------
                # Determine stat column
                # -----------------------------------------
                def grab(col):
                    return row[f"mu_{col}"], row[f"sd_{col}"], col

                if mkey=="player_points":   mu,sd,stat="PTS"
                elif mkey=="player_rebounds": mu,sd,stat="REB"
                elif mkey=="player_assists":  mu,sd,stat="AST"
                elif mkey=="player_threes":   mu,sd,stat="FG3M"
                elif mkey=="player_blocks":   mu,sd,stat="BLK"
                elif mkey=="player_steals":   mu,sd,stat="STL"
                elif mkey=="player_turnovers": mu,sd,stat="TOV"
                elif mkey=="player_field_goals": mu,sd,stat="FGM"
                elif mkey=="player_frees_made": mu,sd,stat="FTM"
                elif mkey=="player_frees_attempts": mu,sd,stat="FTA"

                elif mkey=="player_points_rebounds_assists":
                    stat=None
                    mu=row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                    sd=(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)**0.5

                elif mkey=="player_points_rebounds":
                    stat=None
                    mu=row["mu_PTS"]+row["mu_REB"]
                    sd=(row["sd_PTS"]**2 + row["sd_REB"]**2)**0.5

                elif mkey=="player_points_assists":
                    stat=None
                    mu=row["mu_PTS"]+row["mu_AST"]
                    sd=(row["sd_PTS"]**2 + row["sd_AST"]**2)**0.5

                elif mkey=="player_rebounds_assists":
                    stat=None
                    mu=row["mu_REB"]+row["mu_AST"]
                    sd=(row["sd_REB"]**2 + row["sd_AST"]**2)**0.5

                elif mkey=="player_blocks_steals":
                    stat=None
                    mu=row["mu_BLK"]+row["mu_STL"]
                    sd=(row["sd_BLK"]**2 + row["sd_STL"]**2)**0.5

                else:
                    continue

                # ✅ APPLY DEFENSE ADJUSTMENT
                opp_team = home_team if side=="Over" else away_team
                def_mult = DEFENSE_MULTIPLIERS.get(opp_team, 1.0)
                mu = mu / def_mult
                sd = sd / def_mult

                # ✅ SHARPER SIMULATION
                if stat and stat in df.columns and df.shape[0] >= 6:
                    draws = np.random.choice(df[stat].astype(float).to_numpy(), SIM_TRIALS, True)
                else:
                    draws = t_distribution_draws(mu, sd)

                projection = float(np.median(draws))
                win_prob = float((draws > line).mean()) if side=="Over" else float((draws <= line).mean())

                rows.append({
                    "market": mkey,
                    "player": row["Player"],
                    "side": side,
                    "line": round(line,2),
                    "Avg (raw)": round(mu,2),
                    "Model Projection": round(projection,2),
                    "Win Prob %": round(win_prob*100,2),
                })

    results = pd.DataFrame(rows).sort_values(["market","Win Prob %"], ascending=[True,False])
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download CSV",
        results.to_csv(index=False).encode(),
        "nba_props_sim_results.csv",
        "text/csv",
    )
