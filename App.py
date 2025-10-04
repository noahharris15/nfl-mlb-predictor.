# all_sports_auto.py
# Multi-sport PrizePicks simulator with automatic baselines (no CSVs).

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------- UI --------------------------------------------
st.set_page_config(page_title="All-Sports PrizePicks Simulator", layout="wide")
st.title("📊 All-Sports PrizePicks Simulator (no CSV needed)")

# ----------------------------- PrizePicks -------------------------------------
PP_URL = "https://api.prizepicks.com/projections"
PP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; Streamlit/1.30)",
    "Origin": "https://www.prizepicks.com",
    "Referer": "https://www.prizepicks.com/",
}

LEAGUE_SLUGS = {
    "NFL": "nfl",
    "NBA": "nba",
    "MLB": "mlb",
    "CFB (NCAA-F)": "college-football",
}

@st.cache_data(ttl=60)
def fetch_pp_board(league_slug: str):
    r = requests.get(
        PP_URL,
        params={"per_page": 1000, "single_stat": "true", "league": league_slug},
        headers=PP_HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()

def parse_pp_board(payload, market_map_fn):
    # Collect players from "included"
    players = {}
    for inc in payload.get("included", []):
        if inc.get("type") in ("new_player", "players"):
            a = inc.get("attributes", {}) or {}
            pid = str(inc.get("id"))
            name = a.get("name") or f"{a.get('first_name','')} {a.get('last_name','')}".strip()
            players[pid] = {
                "name": (name or "").strip(),
                "team": a.get("team") or a.get("team_name"),
                "pos": a.get("position"),
            }
    # Projections
    rows = []
    for d in payload.get("data", []):
        a = d.get("attributes", {}) or {}
        rels = d.get("relationships", {}) or {}
        raw = (a.get("projection_type") or a.get("stat_type") or "").lower().strip()
        line = a.get("line_score") or a.get("value") or a.get("line")
        # link to player
        pid = None
        for k in ("new_player", "player", "athlete"):
            if k in rels and isinstance(rels[k].get("data"), dict):
                pid = str(rels[k]["data"]["id"])
                break
        pl = players.get(pid, {})
        rows.append({
            "player": pl.get("name"),
            "team": pl.get("team"),
            "pos": pl.get("pos"),
            "raw_market": raw,
            "market": market_map_fn(raw),
            "line": float(line) if line not in (None, "") else np.nan,
        })
    df = pd.DataFrame(rows).dropna(subset=["player", "market", "line"]).reset_index(drop=True)
    # Clean names a little
    df["player"] = df["player"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

# ----------------------- Distributions & Simulation ---------------------------
def normal_over_prob(mean, sd, line):
    sd = max(1e-6, float(sd))
    z = (line - mean) / sd
    # P(X>line) for Normal
    return float(0.5 * (1 - math.erf(z / math.sqrt(2))))

def poisson_over_prob(rate, line):
    # For integer counts; P(X > line). If line is not integer, use floor.
    k = int(math.floor(line))
    # Poisson CDF up to k
    # Use survival 1 - CDF
    from math import exp
    # sum_{i=0}^{k} e^-λ λ^i / i!
    # Use scipy? Not allowed. We'll approximate via loop for small k; for big use Normal approx
    lam = max(1e-6, float(rate))
    if k < 60:
        term = math.exp(-lam)
        cdf = term
        for i in range(1, k + 1):
            term *= lam / i
            cdf += term
        return max(0.0, min(1.0, 1.0 - cdf))
    # Normal approx for large k
    mu = lam
    sd = math.sqrt(lam)
    # continuity correction
    return normal_over_prob(mu, sd, line + 0.5)

@dataclass
class Baseline:
    mean: float
    sd: float
    dist: str  # "normal" or "poisson"

# ----------------------- Sport-specific baselines -----------------------------
# Each sport exposes:
# - market_map(raw_market_str) -> standardized market label (or None to ignore)
# - get_player_baselines(df_pp) -> dict[(player, market)] = Baseline

### NFL (nfl_data_py)
def nfl_market_map(raw: str) -> Optional[str]:
    m = raw.lower()
    if "passing yards" in m or m == "pass yards": return "pass_yards"
    if "rushing yards" in m or m == "rush yards": return "rush_yards"
    if "receiving yards" in m or m == "rec yards": return "rec_yards"
    if "receptions" in m: return "receptions"
    if "passing tds" in m or "pass tds" in m: return "pass_tds"
    if "rushing tds" in m: return "rush_tds"
    if "receiving tds" in m or "rec tds" in m: return "rec_tds"
    return None

@st.cache_data(ttl=600)
def nfl_player_game_logs() -> pd.DataFrame:
    import nfl_data_py as nfl
    # Get 2025 weekly player stats (regular season weeks to date)
    df = nfl.import_weekly_data([2025])
    # Make name and unify simple columns
    # passing_yards, rushing_yards, receiving_yards already exist
    # receptions, passing_tds, rushing_tds, receiving_tds
    df["player"] = (df["player_name"].fillna("") + "").str.strip()
    return df

def nfl_baselines(df_pp: pd.DataFrame) -> Dict[Tuple[str, str], Baseline]:
    df = nfl_player_game_logs()
    out = {}
    for (player, market), grp in df_pp.groupby(["player", "market"]):
        name_mask = df["player"].str.lower() == str(player).lower()
        d = df.loc[name_mask]
        if d.empty:
            # fallback priors by market
            out[(player, market)] = fallback_prior("NFL", market)
            continue
        if market == "pass_yards":
            x = pd.to_numeric(d["passing_yards"], errors="coerce").dropna()
            mean, sd = float(x.mean()), float(x.std(ddof=1) if len(x)>1 else 0.6*max(x.mean(), 50))
            out[(player, market)] = Baseline(mean, max(sd, 25.0), "normal")
        elif market == "rush_yards":
            x = pd.to_numeric(d["rushing_yards"], errors="coerce").dropna()
            mean, sd = float(x.mean()), float(x.std(ddof=1) if len(x)>1 else 0.8*max(x.mean(), 20))
            out[(player, market)] = Baseline(mean, max(sd, 15.0), "normal")
        elif market == "rec_yards":
            x = pd.to_numeric(d["receiving_yards"], errors="coerce").dropna()
            mean, sd = float(x.mean()), float(x.std(ddof=1) if len(x)>1 else 0.9*max(x.mean(), 18))
            out[(player, market)] = Baseline(mean, max(sd, 14.0), "normal")
        elif market == "receptions":
            x = pd.to_numeric(d["receptions"], errors="coerce").dropna()
            mean, sd = float(x.mean()), float(x.std(ddof=1) if len(x)>1 else max(1.8, 0.6*max(x.mean(), 2)))
            out[(player, market)] = Baseline(mean, max(sd, 1.0), "normal")
        elif market in ("pass_tds","rush_tds","rec_tds"):
            col = {"pass_tds":"passing_tds","rush_tds":"rushing_tds","rec_tds":"receiving_tds"}[market]
            x = pd.to_numeric(d[col], errors="coerce").dropna()
            lam = float(x.mean()) if len(x) else 0.2
            out[(player, market)] = Baseline(lam, math.sqrt(max(lam,1e-6)), "poisson")
        else:
            out[(player, market)] = fallback_prior("NFL", market)
    return out

### NBA (nba_api) – recent 10 games
def nba_market_map(raw: str) -> Optional[str]:
    m = raw.lower()
    if "points" in m and "reb" not in m and "ast" not in m: return "points"
    if "reb" in m: return "rebounds"
    if "ast" in m: return "assists"
    if "pra" in m: return "pra"
    if "3pt" in m or "three" in m: return "threes_made"
    return None

@st.cache_data(ttl=600)
def nba_glogs_last10() -> pd.DataFrame:
    # Uses nba_api and may need a stable network. Falls back silently in caller.
    from nba_api.stats.endpoints import leaguegamefinder, playergamelog
    from nba_api.stats.library.parameters import Season
    # Detect current season string quickly (e.g., "2024-25")
    season_year = pd.Timestamp.today().year
    if pd.Timestamp.today().month < 9:
        season = f"{season_year-1}-{str(season_year)[-2:]}"
    else:
        season = f"{season_year}-{str(season_year+1)[-2:]}"
    # Pull a lot of player logs may be heavy—do per-player on demand
    return pd.DataFrame()  # we fetch on demand below

def nba_on_demand_player(player_name: str) -> pd.DataFrame:
    # Try exact search by name using a players list endpoint
    try:
        from nba_api.stats.static import players
        plist = players.find_players_by_full_name(player_name)
        if not plist:
            return pd.DataFrame()
        pid = plist[0]["id"]
        from nba_api.stats.endpoints import playergamelog
        season_year = pd.Timestamp.today().year
        season = f"{season_year-1}-{str(season_year)[-2:]}" if pd.Timestamp.today().month < 9 else f"{season_year}-{str(season_year+1)[-2:]}"
        gl = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]
        return gl.head(10).copy()  # last 10
    except Exception:
        return pd.DataFrame()

def nba_baselines(df_pp: pd.DataFrame) -> Dict[Tuple[str, str], Baseline]:
    out = {}
    cache: Dict[str, pd.DataFrame] = {}
    for (player, market), grp in df_pp.groupby(["player","market"]):
        if player not in cache:
            cache[player] = nba_on_demand_player(player)
        d = cache[player]
        if d.empty:
            out[(player, market)] = fallback_prior("NBA", market)
            continue
        if market in ("points","rebounds","assists","pra","threes_made"):
            # Columns: PTS, REB, AST, FG3M
            if market == "points":
                x = pd.to_numeric(d["PTS"], errors="coerce").dropna()
            elif market == "rebounds":
                x = pd.to_numeric(d["REB"], errors="coerce").dropna()
            elif market == "assists":
                x = pd.to_numeric(d["AST"], errors="coerce").dropna()
            elif market == "pra":
                x = (pd.to_numeric(d["PTS"], errors="coerce").fillna(0)
                     + pd.to_numeric(d["REB"], errors="coerce").fillna(0)
                     + pd.to_numeric(d["AST"], errors="coerce").fillna(0))
            else:
                x = pd.to_numeric(d["FG3M"], errors="coerce").dropna()
            if len(x) == 0:
                out[(player, market)] = fallback_prior("NBA", market)
            else:
                m = float(x.mean())
                s = float(x.std(ddof=1) if len(x)>1 else 0.6*max(m,1))
                out[(player, market)] = Baseline(m, max(s, 0.8), "normal")
        else:
            out[(player, market)] = fallback_prior("NBA", market)
    return out

### MLB (pybaseball) – season to date per game
def mlb_market_map(raw: str) -> Optional[str]:
    m = raw.lower()
    if "hits" in m: return "hits"
    if "total bases" in m or "tb" == m: return "total_bases"
    if "strikeouts" in m and "pitch" in m or m == "pitcher strikeouts": return "pitcher_ks"
    if "runs" in m and "allowed" in m: return "runs_allowed"
    if "home runs" in m or "hr" == m: return "home_runs"
    return None

@st.cache_data(ttl=1200)
def mlb_batting_tod() -> pd.DataFrame:
    from pybaseball import batting_stats
    y = pd.Timestamp.today().year
    df = batting_stats(y, qual=0)
    # per game versions
    df["H_pg"] = df["H"] / df["G"].replace(0, np.nan)
    df["TB_pg"] = df["TB"] / df["G"].replace(0, np.nan)
    df["HR_pg"] = df["HR"] / df["G"].replace(0, np.nan)
    # names
    df["player"] = df["Name"].astype(str).str.strip()
    return df

@st.cache_data(ttl=1200)
def mlb_pitching_tod() -> pd.DataFrame:
    from pybaseball import pitching_stats
    y = pd.Timestamp.today().year
    df = pitching_stats(y, qual=0)
    df["SO_per_g"] = df["SO"] / df["G"].replace(0, np.nan)
    df["R_per_g"]  = df["R"]  / df["G"].replace(0, np.nan)
    df["player"] = df["Name"].astype(str).str.strip()
    return df

def mlb_baselines(df_pp: pd.DataFrame) -> Dict[Tuple[str, str], Baseline]:
    bat = mlb_batting_tod()
    pit = mlb_pitching_tod()
    out = {}
    for (player, market), grp in df_pp.groupby(["player","market"]):
        if market in ("hits","total_bases","home_runs"):
            d = bat.loc[bat["player"].str.lower()==str(player).lower()]
            if d.empty:
                out[(player, market)] = fallback_prior("MLB", market)
                continue
            if market == "hits":
                m = float(d["H_pg"].fillna(0).iloc[0])
                s = max(0.6, 0.9*m)
            elif market == "total_bases":
                m = float(d["TB_pg"].fillna(0).iloc[0])
                s = max(0.8, 0.8*m)
            else:
                m = float(d["HR_pg"].fillna(0).iloc[0])
                s = max(0.3, max(1.0, 2.0*m))
            out[(player, market)] = Baseline(m, s, "normal")
        elif market in ("pitcher_ks","runs_allowed"):
            d = pit.loc[pit["player"].str.lower()==str(player).lower()]
            if d.empty:
                out[(player, market)] = fallback_prior("MLB", market)
                continue
            if market == "pitcher_ks":
                lam = float(d["SO_per_g"].fillna(0).iloc[0])
            else:
                lam = float(d["R_per_g"].fillna(0).iloc[0])
            out[(player, market)] = Baseline(lam, math.sqrt(max(lam, 0.5)), "poisson")
        else:
            out[(player, market)] = fallback_prior("MLB", market)
    return out

### CFB (cfbd) – season player stats (needs API key in secrets)
def cfb_market_map(raw: str) -> Optional[str]:
    m = raw.lower()
    if "passing yards" in m: return "pass_yards"
    if "rushing yards" in m: return "rush_yards"
    if "receiving yards" in m: return "rec_yards"
    if "receptions" in m: return "receptions"
    if "passing tds" in m or "pass tds" in m: return "pass_tds"
    if "rushing tds" in m: return "rush_tds"
    if "receiving tds" in m: return "rec_tds"
    return None

@st.cache_data(ttl=900)
def cfb_player_stats() -> pd.DataFrame:
    try:
        import cfbd
        key = st.secrets.get("cfb", {}).get("api_key")
        if not key:
            return pd.DataFrame()
        configuration = cfbd.Configuration()
        configuration.api_key["Authorization"] = key
        configuration.api_key_prefix["Authorization"] = "Bearer"
        api = cfbd.PlayersApi(cfbd.ApiClient(configuration))
        y = pd.Timestamp.today().year
        stats = api.player_season_stats(year=y)
        # Normalize to DataFrame
        recs = []
        for s in stats:
            recs.append({
                "player": s.player,
                "team": s.team,
                "games": s.games if s.games else np.nan,
                "pass_yards": s.passing_yards or 0,
                "rush_yards": s.rushing_yards or 0,
                "rec_yards": s.receiving_yards or 0,
                "receptions": s.receptions or 0,
                "pass_tds": s.passing_tds or 0,
                "rush_tds": s.rushing_tds or 0,
                "rec_tds": s.receiving_tds or 0,
            })
        df = pd.DataFrame.from_records(recs)
        # Per game
        g = df["games"].replace(0, np.nan)
        for col in ["pass_yards","rush_yards","rec_yards","receptions","pass_tds","rush_tds","rec_tds"]:
            df[col] = df[col] / g
        df["player"] = df["player"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def cfb_baselines(df_pp: pd.DataFrame) -> Dict[Tuple[str, str], Baseline]:
    base = cfb_player_stats()
    out = {}
    for (player, market), grp in df_pp.groupby(["player","market"]):
        d = base.loc[base["player"].str.lower()==str(player).lower()]
        if d.empty:
            out[(player, market)] = fallback_prior("CFB", market)
            continue
        m = float(d[market].fillna(0).iloc[0]) if market in d.columns else 0.0
        # Conservative variance for college
        if market.endswith("tds"):
            out[(player, market)] = Baseline(max(m, 0.1), math.sqrt(max(m, 0.1)), "poisson")
        else:
            sd = max(5.0, 0.9*m + 6.0)
            out[(player, market)] = Baseline(m, sd, "normal")
    return out

# ------------------------- Priors / Fallbacks ---------------------------------
def fallback_prior(sport: str, market: str) -> Baseline:
    # Used when a player’s logs/stats can’t be found quickly.
    priors = {
        "NFL": {
            "pass_yards": (220, 55, "normal"),
            "rush_yards": (48, 22, "normal"),
            "rec_yards":  (52, 20, "normal"),
            "receptions": (4.0, 1.6, "normal"),
            "pass_tds":   (1.4, math.sqrt(1.4), "poisson"),
            "rush_tds":   (0.3, math.sqrt(0.3), "poisson"),
            "rec_tds":    (0.3, math.sqrt(0.3), "poisson"),
        },
        "NBA": {
            "points": (18, 6.5, "normal"),
            "rebounds": (6.0, 3.0, "normal"),
            "assists": (4.5, 2.5, "normal"),
            "pra": (28.0, 8.0, "normal"),
            "threes_made": (2.0, 1.5, "normal"),
        },
        "MLB": {
            "hits": (1.0, 0.9, "normal"),
            "total_bases": (1.8, 1.4, "normal"),
            "home_runs": (0.18, 0.6, "normal"),
            "pitcher_ks": (5.6, math.sqrt(5.6), "poisson"),
            "runs_allowed": (2.8, math.sqrt(2.8), "poisson"),
        },
        "CFB": {
            "pass_yards": (210, 65, "normal"),
            "rush_yards": (58, 28, "normal"),
            "rec_yards":  (52, 26, "normal"),
            "receptions": (3.6, 1.8, "normal"),
            "pass_tds":   (1.3, math.sqrt(1.3), "poisson"),
            "rush_tds":   (0.4, math.sqrt(0.4), "poisson"),
            "rec_tds":    (0.4, math.sqrt(0.4), "poisson"),
        },
    }
    m = priors.get(sport, {}).get(market, (10.0, 5.0, "normal"))
    return Baseline(m[0], m[1], m[2])

# ----------------------------- Router -----------------------------------------
SPORTS = {
    "NFL":   (nfl_market_map, nfl_baselines),
    "NBA":   (nba_market_map, nba_baselines),
    "MLB":   (mlb_market_map, mlb_baselines),
    "CFB (NCAA-F)": (cfb_market_map, cfb_baselines),
}

sport_name = st.selectbox("League", list(SPORTS.keys()), index=0)
market_map_fn, baselines_fn = SPORTS[sport_name]

slug = LEAGUE_SLUGS[sport_name]
with st.spinner("Fetching PrizePicks board…"):
    try:
        pp = fetch_pp_board(slug)
    except Exception as e:
        st.error(f"PrizePicks fetch failed: {e}")
        st.stop()

df_pp = parse_pp_board(pp, market_map_fn)
if df_pp.empty:
    st.warning("No PrizePicks markets parsed for this league right now.")
    st.stop()

st.write(f"Loaded **{len(df_pp)}** lines from PrizePicks.")

with st.spinner("Building per-player baselines (auto)…"):
    try:
        baselines = baselines_fn(df_pp)
    except Exception as e:
        st.error(f"Baseline build failed: {e}")
        baselines = {}

def compute_prob(row: pd.Series) -> float:
    base = baselines.get((row["player"], row["market"]))
    if not base:
        base = fallback_prior(sport_name, row["market"])
    if base.dist == "poisson":
        return poisson_over_prob(base.mean, row["line"])
    else:
        return normal_over_prob(base.mean, base.sd, row["line"])

df_pp["p_over"] = df_pp.apply(compute_prob, axis=1)
df_pp["p_under"] = 1.0 - df_pp["p_over"]
df_pp["edge_over"] = df_pp["p_over"] - 0.5
df_pp["edge_under"] = df_pp["p_under"] - 0.5

# Filters & sorters
left, right = st.columns([2,1])
with right:
    show_market = st.multiselect("Filter markets", sorted(df_pp["market"].unique()),
                                 default=list(sorted(df_pp["market"].unique())))
    sort_by = st.selectbox("Sort by", ["edge_over","edge_under","p_over","p_under"])
    top_n = st.number_input("Top N", 5, 200, 30, step=5)

view = df_pp[df_pp["market"].isin(show_market)].copy()
view = view.sort_values(sort_by, ascending=False).head(int(top_n))

st.dataframe(
    view[["player","team","pos","market","line","p_over","p_under","edge_over","edge_under"]]
      .style.format({"line":"{:.2f}","p_over":"{:.1%}","p_under":"{:.1%}","edge_over":"{:.1%}","edge_under":"{:.1%}"})
)

st.caption("Notes: baselines come from public season/log data. If a player log fails or is missing, we fall back to conservative priors to avoid blank rows. NBA endpoints can occasionally throttle; try again if you see many priors.")
