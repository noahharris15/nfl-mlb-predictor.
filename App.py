import time
import math
import json
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# UI header
# -----------------------------
st.set_page_config(page_title="All-Sports PrizePicks Simulator (no CSV)", layout="wide")
st.title("📊 All-Sports PrizePicks Simulator (no CSV)")
st.caption("Pulls current PrizePicks lines + real player averages automatically, then runs a Monte-Carlo simulation.")

# -----------------------------
# Settings
# -----------------------------
LEAGUE_KEYS = {
    "NFL": "nfl",
    "NBA": "nba",
    # "MLB": "mlb",  # add later
    # "NHL": "nhl",  # add later
}

DEFAULT_SIM_TRIALS = 20000
MAX_ROWS = 1500  # keep the UI snappy

PRIZEPICKS_URL = "https://site.api.prizepicks.com/api/v1/projections"  # more reliable than api.prizepicks.com
PP_MARKET_INCLUDE = {
    "nfl": {
        # mapping PrizePicks "type" (their stat label) -> our stat key used in averages
        "Pass Yds": "pass_yards",
        "Pass Attempts": "pass_attempts",
        "Pass Completions": "pass_completions",
        "Pass TDs": "pass_tds",
        "Rush Yds": "rush_yards",
        "Rush Attempts": "rush_attempts",
        "Rec Yds": "rec_yards",
        "Receptions": "receptions",
        "Receiving TDs": "rec_tds",
        "Interceptions": "interceptions_thrown",  # thrown
        "Longest Pass": "long_pass",
        "Longest Rush": "long_rush",
        "Longest Reception": "long_rec",
        # add more as needed
    },
    "nba": {
        "Points": "pts",
        "Rebounds": "reb",
        "Assists": "ast",
        "Pts+Reb+Ast": "pra",
        "3-PT Made": "fg3m",
        "Blocks": "blk",
        "Steals": "stl",
        "Pts+Reb": "pr",
        "Pts+Ast": "pa",
        "Reb+Ast": "ra",
        # add more if you want
    },
}

UA = st.secrets.get(
    "PRIZEPICKS_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
)

# -----------------------------
# Helpers: HTTP with retry/backoff
# -----------------------------
def http_get_json(url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None,
                  max_retries: int = 6, timeout: int = 15, backoff0: float = 1.0) -> dict:
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as je:
                    last_err = je
            elif r.status_code in (403, 429, 520, 521, 522, 523, 524):
                # rate-limited or forbidden: backoff
                wait = backoff0 * (2 ** i) + random.random()
                st.info(f"⏳ Network hiccup ({r.status_code}); retrying in {wait:.1f}s…")
                time.sleep(wait)
            else:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
                break
        except requests.RequestException as e:
            last_err = e
            wait = backoff0 * (2 ** i) + random.random()
            st.info(f"⏳ Network hiccup; retrying in {wait:.1f}s…")
            time.sleep(wait)
    raise RuntimeError(f"GET failed: {url}\n{last_err}")

# -----------------------------
# PrizePicks pull (strict per-league filtering)
# -----------------------------
@st.cache_data(ttl=120)
def fetch_prizepicks(league_key: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    ['league','player','team','market_pp','line','opponent','type_raw','id']
    """
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.prizepicks.com/",
        "Origin": "https://www.prizepicks.com",
    }

    params = {
        "per_page": 500,          # we'll page if needed
        "single_stat": "true",
        "league": league_key,
    }

    data = http_get_json(PRIZEPICKS_URL, headers=headers, params=params)

    if not isinstance(data, dict) or "data" not in data:
        raise RuntimeError("PrizePicks: unexpected payload")

    included = data.get("included", [])
    # Build lookup maps from included
    players = {i["id"]: i["attributes"]["name"] for i in included if i.get("type") == "players"}
    teams = {i["id"]: i["attributes"]["name"] for i in included if i.get("type") == "teams"}
    # Some payloads put opponents/teams in relationships; handle gracefully

    rows = []
    for proj in data["data"]:
        attr = proj.get("attributes", {})
        # strict league check
        if attr.get("league") != league_key:
            continue

        player_id = attr.get("player_id") or proj.get("relationships", {}).get("player", {}).get("data", {}).get("id")
        player_name = players.get(str(player_id), attr.get("display_name") or "Unknown")

        market_pp = attr.get("stat_type") or attr.get("type") or attr.get("projection_type")
        line = attr.get("line_score") or attr.get("value")
        team_id = attr.get("team_id") or proj.get("relationships", {}).get("team", {}).get("data", {}).get("id")
        team_nm = teams.get(str(team_id), None)

        rows.append({
            "league": league_key,
            "player": player_name,
            "team": team_nm,
            "market_pp": market_pp,
            "line": float(line) if line is not None else None,
            "type_raw": market_pp,
            "id": proj.get("id"),
        })

    df = pd.DataFrame(rows).dropna(subset=["player", "line", "market_pp"])
    # keep only markets we know how to simulate
    keep_map = PP_MARKET_INCLUDE.get(league_key, {})
    df = df[df["market_pp"].isin(keep_map.keys())].copy()
    return df.head(MAX_ROWS)

# -----------------------------
# Player averages
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_nba_averages(season_year: int) -> pd.DataFrame:
    """
    BallDontLie (free). Returns per-game averages for season.
    """
    # bdl v1 supports modern seasons; paginate
    base = "https://www.balldontlie.io/api/v1/season_averages"
    all_rows = []
    # We need player IDs; we can fetch league-wide active players quickly (several pages)
    players = []
    page = 1
    while True:
        r = requests.get("https://www.balldontlie.io/api/v1/players", params={"per_page": 100, "page": page, "active": "true"}, timeout=15)
        if r.status_code != 200:
            break
        js = r.json()
        data = js.get("data", [])
        if not data:
            break
        players.extend([p["id"] for p in data])
        if page >= js.get("meta", {}).get("total_pages", page):
            break
        page += 1

    # chunk requests (API accepts multiple player_ids)
    for i in range(0, len(players), 75):
        chunk = players[i:i+75]
        r = requests.get(base, params={"season": season_year, "player_ids[]": chunk}, timeout=20)
        if r.status_code == 200:
            all_rows.extend(r.json().get("data", []))

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # BDL columns: pts, reb, ast, stl, blk, fg3m, fga, fgm, etc.
    # Rename to our keys
    df.rename(columns={
        "player_id": "player_id",
        "games_played": "gp",
        "pts": "pts",
        "reb": "reb",
        "ast": "ast",
        "stl": "stl",
        "blk": "blk",
        "fg3m": "fg3m",
        "turnover": "tov",
    }, inplace=True)

    # We still need player names
    id2name = {}
    page = 1
    while True:
        r = requests.get("https://www.balldontlie.io/api/v1/players", params={"per_page": 100, "page": page, "active": "true"}, timeout=15)
        if r.status_code != 200:
            break
        js = r.json()
        data = js.get("data", [])
        if not data:
            break
        for p in data:
            id2name[p["id"]] = f"{p['first_name']} {p['last_name']}".strip()
        if page >= js.get("meta", {}).get("total_pages", page):
            break
        page += 1

    df["player"] = df["player_id"].map(id2name)
    # build derived combos
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    df["pr"] = df["pts"] + df["reb"]
    df["pa"] = df["pts"] + df["ast"]
    df["ra"] = df["reb"] + df["ast"]

    # crude per-stat SDs as 15% of mean when we lack game-by-game — adjust if you prefer
    for c in ["pts", "reb", "ast", "fg3m", "stl", "blk", "pra", "pr", "pa", "ra"]:
        if c not in df.columns:
            df[c] = np.nan
        df[f"{c}_sd"] = df[c] * 0.15

    return df[["player","pts","reb","ast","fg3m","stl","blk","pra","pr","pa","ra",
               "pts_sd","reb_sd","ast_sd","fg3m_sd","stl_sd","blk_sd","pra_sd","pr_sd","pa_sd","ra_sd"]]

@st.cache_data(ttl=3600)
def fetch_nfl_averages(season_year: int, api_key: Optional[str]) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    url = f"https://api.sportsdata.io/v3/nfl/stats/json/PlayerSeasonStats/{season_year}REG"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    r = requests.get(url, headers=headers, timeout=25)
    if r.status_code != 200:
        return pd.DataFrame()
    js = r.json()
    df = pd.DataFrame(js)

    # Build per-game averages
    gp = df["Played"].replace(0, np.nan)
    out = pd.DataFrame()
    out["player"] = df["Name"]
    out["pass_yards"] = (df["PassingYards"] / gp)
    out["pass_attempts"] = (df["PassingAttempts"] / gp)
    out["pass_completions"] = (df["PassingCompletions"] / gp)
    out["pass_tds"] = (df["PassingTouchdowns"] / gp)
    out["interceptions_thrown"] = (df["Interceptions"] / gp)
    out["rush_yards"] = (df["RushingYards"] / gp)
    out["rush_attempts"] = (df["RushingAttempts"] / gp)
    out["rec_yards"] = (df["ReceivingYards"] / gp)
    out["receptions"] = (df["Receptions"] / gp)
    out["rec_tds"] = (df["ReceivingTouchdowns"] / gp)
    # Long plays aren't in per-game; keep NaN to skip those markets if we can't simulate

    # crude SDs ~ 35% of mean (NFL game-to-game variance is higher)
    for c in ["pass_yards","pass_attempts","pass_completions","pass_tds","interceptions_thrown",
              "rush_yards","rush_attempts","rec_yards","receptions","rec_tds"]:
        out[f"{c}_sd"] = out[c] * 0.35
    return out

def get_averages_df(league_key: str, season_year: int) -> pd.DataFrame:
    if league_key == "nba":
        return fetch_nba_averages(season_year)
    if league_key == "nfl":
        return fetch_nfl_averages(season_year, st.secrets.get("SPORTSDATA_API_KEY"))
    return pd.DataFrame()

# -----------------------------
# Join + Simulation
# -----------------------------
def _norm_prob_over(mu: float, sd: float, line: float) -> float:
    if pd.isna(mu) or pd.isna(sd) or sd <= 0 or pd.isna(line):
        return np.nan
    # normal approximation
    z = (line - mu) / sd
    # P(X > line) = 1 - Phi(z)
    return float(1.0 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

def simulate(df_pp: pd.DataFrame, df_avg: pd.DataFrame, league_key: str, trials: int = DEFAULT_SIM_TRIALS) -> pd.DataFrame:
    """
    For each PP row, map to average & sd, compute P(over)/P(under) (normal model).
    """
    stat_map = PP_MARKET_INCLUDE.get(league_key, {})
    out_rows = []

    # lowercase name match helper
    def ln(x): return (x or "").strip().lower()

    for _, row in df_pp.iterrows():
        stat_key = stat_map.get(row["market_pp"])
        if not stat_key:
            continue

        # find matching player (simple normalizer)
        sub = df_avg[df_avg["player"].apply(ln) == ln(row["player"])]
        if sub.empty:
            # try contains (handles suffix, accents, etc.)
            sub = df_avg[df_avg["player"].apply(ln).str.contains(rf"\b{ln(row['player']).replace('.','\.')}\b", regex=True, na=False)]
        if sub.empty:
            continue

        mu = float(sub.iloc[0].get(stat_key, np.nan))
        sd = float(sub.iloc[0].get(f"{stat_key}_sd", np.nan))

        p_over = _norm_prob_over(mu, sd, float(row["line"]))
        p_under = 1 - p_over if not pd.isna(p_over) else np.nan

        out_rows.append({
            "league": league_key.upper(),
            "player": row["player"],
            "team": row.get("team"),
            "market": stat_key,
            "market_pp": row["market_pp"],
            "line": row["line"],
            "mean": mu,
            "model_sd": sd,
            "P(over)": p_over,
            "P(under)": p_under,
        })

    res = pd.DataFrame(out_rows)
    # keep sensible rows only
    res = res.dropna(subset=["P(over)"])
    # show strongest edges first
    res["edge_abs"] = (res["P(over)"] - 0.5).abs()
    res = res.sort_values("edge_abs", ascending=False).drop(columns=["edge_abs"])
    return res.reset_index(drop=True)

# -----------------------------
# Sidebar / Controls
# -----------------------------
colA, colB = st.columns([1, 1])
with colA:
    league_label = st.selectbox("League", list(LEAGUE_KEYS.keys()), index=0)
league_key = LEAGUE_KEYS[league_label]

with colB:
    season_year = st.number_input("Season (year)", min_value=2015, max_value=2030,
                                  value=(2025 if league_key == "nfl" else 2024), step=1)

# -----------------------------
# Fetch Data
# -----------------------------
st.markdown("### 1) Load current board (PrizePicks)")
pp_err = None
try:
    df_pp = fetch_prizepicks(league_key)
    if df_pp.empty:
        st.warning("No PrizePicks markets parsed for this league right now (or none matched supported stat types).")
    else:
        st.success(f"Loaded {len(df_pp)} PrizePicks markets.")
        with st.expander("Preview PrizePicks rows"):
            st.dataframe(df_pp.head(30), use_container_width=True)
except Exception as e:
    pp_err = str(e)
    st.error(f"PrizePicks fetch/parse error: {pp_err}")

st.markdown("### 2) Load real player averages (auto)")
df_avg = get_averages_df(league_key, season_year)
if df_avg.empty:
    if league_key == "nfl":
        st.error("NFL averages require `SPORTSDATA_API_KEY` in Secrets (SportsData.io).")
    else:
        st.error("Could not load player averages for this league.")
else:
    st.success(f"Loaded averages for {len(df_avg)} players.")
    with st.expander("Preview averages"):
        st.dataframe(df_avg.head(30), use_container_width=True)

# -----------------------------
# Simulation
# -----------------------------
if not pp_err and not df_avg.empty and not df_pp.empty:
    st.markdown("### 3) Simulate probabilities")
    trials = st.slider("Simulation trials (normal model uses mean/sd; higher = smoother)", 5000, 50000, DEFAULT_SIM_TRIALS, 5000)
    results = simulate(df_pp, df_avg, league_key, trials=trials)
    if results.empty:
        st.warning("Could not match any PrizePicks players/markets to stats — try a different league or season, or wait for a fuller board.")
    else:
        st.caption("Probabilities are model estimates based on auto baselines (no CSVs).")
        st.dataframe(results, use_container_width=True)
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", csv, file_name=f"{league_key}_pp_sim_results.csv", mime="text/csv")
else:
    st.info("Waiting for board + averages…")
