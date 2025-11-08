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

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> Tuple[float, np.ndarray]:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean()), draws

def sample_sd(sum_x: float, sum_x2: float, g: int, floor: float = 0.0) -> float:
    g = int(g)
    if g <= 1: return float("nan")
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
            cand = cand.loc[cand["first_norm"].str.startswith(first[:1]) |
                            cand["first_norm"].str.contains(first)]
            if not cand.empty:
                return int(cand.iloc[0]["id"])

    last = parts[-1]
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

    def s(col: str):
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        return float(x.sum()), float((x**2).sum())

    sums = {}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        sums[col], sums["sq_"+col] = s(col)

    out = {"g": g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        out["mu_"+col] = sums[col] / g
        out["sd_"+col] = sample_sd(sums[col], sums["sq_"+col], g)
    return out

# ------------------ UI ------------------
st.markdown("### 1) Season Locked to 2025-26")
season_locked = "2025-26"
st.caption(f"Using NBA season: **{season_locked}**")

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
        {"apiKey": api_key, "regions": region, "daysFrom": 0, "daysTo": lookahead if lookahead > 0 else 1}
    )

# ✅ RESTORED MISSING FUNCTION
def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    base = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds"
    return odds_get(
        base,
        {
            "apiKey": api_key,
            "regions": region,
            "markets": ",".join(markets),
            "oddsFormat": "american"
        }
    )

if api_key:
    events = list_events(api_key, lookahead, region)
else:
    events = []

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list games.")
    st.stop()

# ✅ Safe event parsing
event_labels = []
for e in events:
    away = e.get("away_team") or e.get("teams",[None,None])[0] or "Away"
    home = e.get("home_team") or e.get("teams",[None,None])[1] or "Home"
    date = e.get("commence_time", "")
    event_labels.append(f"{away} @ {home} — {date}")

pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ------------------ Build projections ------------------
st.markdown("### 3) Build per-player projections")
build = st.button("📥 Build NBA projections")

if build:
    data_preview = fetch_event_props(api_key, event_id, region, list(set(markets)))
    player_names = set()

    for bk in data_preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE:
                continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm:
                    player_names.add(nm)

    rows = []
    for pn in sorted(player_names):
        pid = find_player_id_by_name(pn)
        if not pid:
            continue

        df = fetch_player_gamelog_df(pid, season_locked, "Regular Season")
        stats = agg_full_season(df)

        if stats["g"] == 0:
            df = fetch_player_gamelog_df(pid, "2024-25", "Regular Season")
            stats = agg_full_season(df)

        if stats["g"] == 0:
            continue

        rows.append({"Player": pn, **stats})

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    st.session_state["nba_proj"] = proj

    st.dataframe(proj, use_container_width=True)

# ------------------ Single Game Simulation ------------------
st.markdown("### 4) Fetch lines & simulate (10k draws)")
go = st.button("Fetch lines & simulate (NBA)")

if go:

    proj = st.session_state.get("nba_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build projections first")
        st.stop()

    props = fetch_event_props(api_key, event_id, region, list(set(markets)))

    proj = proj.set_index("player_norm")
    rows = []

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
                side = o["name"]
                line = float(o["point"])

                def grab(col):
                    return row[f"mu_{col}"], row[f"sd_{col}"]

                if mkey=="player_points": mu,sd = grab("PTS")
                elif mkey=="player_rebounds": mu,sd = grab("REB")
                elif mkey=="player_assists": mu,sd = grab("AST")
                elif mkey=="player_threes": mu,sd = grab("FG3M")
                elif mkey=="player_blocks": mu,sd = grab("BLK")
                elif mkey=="player_steals": mu,sd = grab("STL")
                elif mkey=="player_turnovers": mu,sd = grab("TOV")
                elif mkey=="player_field_goals": mu,sd = grab("FGM")
                elif mkey=="player_frees_made": mu,sd = grab("FTM")
                elif mkey=="player_frees_attempts": mu,sd = grab("FTA")

                elif mkey=="player_points_rebounds_assists":
                    mu = row["mu_PTS"] + row["mu_REB"] + row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)

                elif mkey=="player_points_rebounds":
                    mu = row["mu_PTS"] + row["mu_REB"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2)

                elif mkey=="player_points_assists":
                    mu = row["mu_PTS"] + row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2)

                elif mkey=="player_rebounds_assists":
                    mu = row["mu_REB"] + row["mu_AST"]
                    sd = np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2)

                elif mkey=="player_blocks_steals":
                    mu = row["mu_BLK"] + row["mu_STL"]
                    sd = np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2)

                else:
                    continue

                p_over, draws = t_over_prob(mu, sd, line, SIM_TRIALS)
                projection = float(np.median(draws))
                win_prob = p_over if side == "Over" else (1 - p_over)

                rows.append({
                    "market": mkey,
                    "player": row["Player"],
                    "side": side,
                    "line": round(line,2),
                    "Avg (raw)": round(mu,2),
                    "Model Projection": round(projection,2),
                    "Win Prob %": round(win_prob*100,2),
                })

    results = pd.DataFrame(rows).sort_values(["market","Win Prob %"], ascending=[True, False])
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="nba_props_sim_results.csv",
        mime="text/csv",
    )

# ------------------ FULL SLATE MODE ------------------
st.markdown("### 5) Full Slate: Biggest Value Plays (All Games)")
full_slate = st.button("📊 Run Full-Slate Value Report")

if full_slate:

    if not api_key:
        st.warning("Enter API Key")
        st.stop()

    all_events = list_events(api_key, lookahead, region)
    if not all_events:
        st.error("No games found.")
        st.stop()

    master_rows = []

    for ev in all_events:
        eid = ev["id"]

        try:
            props = fetch_event_props(api_key, eid, region, list(set(markets)))
        except:
            continue

        player_names = set()
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") in UNSUPPORTED_MARKETS_HIDE:
                    continue
                for o in m.get("outcomes", []):
                    nm = normalize_name(o.get("description"))
                    if nm: player_names.add(nm)

        rows = []
        for pn in sorted(player_names):
            pid = find_player_id_by_name(pn)
            if not pid: continue

            df = fetch_player_gamelog_df(pid, season_locked, "Regular Season")
            stats = agg_full_season(df)

            if stats["g"] == 0:
                df = fetch_player_gamelog_df(pid, "2024-25", "Regular Season")
                stats = agg_full_season(df)

            if stats["g"] == 0:
                continue

            rows.append({"Player": pn, **stats})

        if not rows:
            continue

        proj = pd.DataFrame(rows)
        proj["player_norm"] = proj["Player"].apply(normalize_name)
        proj = proj.set_index("player_norm")

        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                mkey = m.get("key")
                if mkey not in VALID_MARKETS:
                    continue

                for o in m.get("outcomes", []):
                    name = normalize_name(o.get("description"))
                    if name not in proj.index:
                        continue
                    if pd.isna(o.get("point")):
                        continue

                    row = proj.loc[name]
                    side = o["name"]
                    line = float(o["point"])

                    def grab(col):
                        return row[f"mu_{col}"], row[f"sd_{col}"]

                    if mkey=="player_points": mu,sd = grab("PTS")
                    elif mkey=="player_rebounds": mu,sd = grab("REB")
                    elif mkey=="player_assists": mu,sd = grab("AST")
                    elif mkey=="player_threes": mu,sd = grab("FG3M")
                    elif mkey=="player_blocks": mu,sd = grab("BLK")
                    elif mkey=="player_steals": mu,sd = grab("STL")
                    elif mkey=="player_turnovers": mu,sd = grab("TOV")
                    elif mkey=="player_field_goals": mu,sd = grab("FGM")
                    elif mkey=="player_frees_made": mu,sd = grab("FTM")
                    elif mkey=="player_frees_attempts": mu,sd = grab("FTA")

                    elif mkey=="player_points_rebounds_assists":
                        mu = row["mu_PTS"] + row["mu_REB"] + row["mu_AST"]
                        sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)

                    elif mkey=="player_points_rebounds":
                        mu = row["mu_PTS"] + row["mu_REB"]
                        sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2)

                    elif mkey=="player_points_assists":
                        mu = row["mu_PTS"] + row["mu_AST"]
                        sd = np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2)

                    elif mkey=="player_rebounds_assists":
                        mu = row["mu_REB"] + row["mu_AST"]
                        sd = np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2)

                    elif mkey=="player_blocks_steals":
                        mu = row["mu_BLK"] + row["mu_STL"]
                        sd = np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2)

                    else:
                        continue

                    p_over, draws = t_over_prob(mu, sd, line, SIM_TRIALS)
                    proj_val = float(np.median(draws))
                    edge = proj_val - line

                    master_rows.append({
                        "Game": f"{ev.get('away_team')} @ {ev.get('home_team')}",
                        "Player": row["Player"],
                        "Market": mkey,
                        "Side": side,
                        "Line": round(line,2),
                        "Model Projection": round(proj_val,2),
                        "Projection Edge": round(edge,2),
                    })

    if not master_rows:
        st.error("No props found.")
        st.stop()

    full_df = pd.DataFrame(master_rows)
    full_df = full_df.sort_values("Projection Edge", ascending=False)

    st.markdown("## 🔥 TOP VALUE PROJECTION EDGES (FULL SLATE)")
    st.dataframe(full_df.head(50), use_container_width=True)

    st.markdown("### 📈 Biggest Edges Chart")
    chart_df = full_df.head(20)[["Player","Projection Edge"]].set_index("Player")
    st.bar_chart(chart_df)
