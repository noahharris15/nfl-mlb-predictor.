# app_prizepicks_auto.py
# Streamlit PrizePicks simulator (clean league filtering + per-market simulations)

import time
import math
import json
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm, poisson, nbinom

st.set_page_config(page_title="All-Sports PrizePicks Simulator", layout="wide")

# ---------------------------
# Config / heuristics
# ---------------------------

PRIZEPICKS_URL = "https://api.prizepicks.com/projections?per_page=1000&single_stat=true&league={league}"

# Reasonable, conservative SD heuristics by market (used if we can't infer better).
# Normal for continuous-ish totals; Poisson for whole-number counting stats.
SD_HEURISTICS_NORMAL = {
    # NBA
    "points": 7.0, "rebounds": 3.2, "assists": 3.0, "pra": 9.5, "pr": 8.0, "pa": 8.0, "ra": 4.8,
    "threes_made": 1.7,
    # NFL
    "passing_yards": 45.0, "rushing_yards": 28.0, "receiving_yards": 30.0,
    "receptions": 2.1, "pass_attempts": 5.5, "pass_completions": 4.5,
    # MLB (game-to-game variance is high; totals are small so Poisson sometimes better)
    "hits": 0.8, "total_bases": 1.2, "runs": 0.7, "rbi": 0.8,
    # NHL
    "shots_on_goal": 1.6,
}

POISSON_MARKETS = set([
    # NBA
    "blocks", "steals", "blocks_steals",
    # Soccer
    "goals", "shots", "shots_on_target", "clearances", "tackles",
    # MLB-ish (optional): "home_runs"
])

# Mappings from PrizePicks market names to our canonical market keys per league
MARKET_MAP_BY_LEAGUE: Dict[str, Dict[str, str]] = {
    "nba": {
        "Points": "points", "Rebounds": "rebounds", "Assists": "assists",
        "Pts+Reb+Ast": "pra", "Pts+Reb": "pr", "Pts+Ast": "pa", "Reb+Ast": "ra",
        "3-PT Made": "threes_made", "Blocks": "blocks", "Steals": "steals",
        "Blk+Stl": "blocks_steals",
    },
    "nfl": {
        "Pass Yds": "passing_yards", "Rush Yds": "rushing_yards",
        "Rec Yds": "receiving_yards", "Receptions": "receptions",
        "Pass Att": "pass_attempts", "Completions": "pass_completions",
    },
    "mlb": {
        "Hits": "hits", "Total Bases": "total_bases", "Runs": "runs", "RBI": "rbi",
        # "Home Runs": "home_runs",
    },
    "nhl": {"Shots On Goal": "shots_on_goal"},
    # Some soccer examples PrizePicks uses:
    "soccer": {"Goals": "goals", "Shots": "shots", "Shots On Target": "shots_on_target",
               "Clearances": "clearances", "Tackles": "tackles"},
}

SUPPORTED_LEAGUES = ["NFL", "NBA", "MLB", "NHL", "Soccer"]

# ---------------------------
# Helpers
# ---------------------------

def pp_fetch(league_slug: str, retries: int = 3, backoff: float = 1.25) -> dict:
    """
    Fetch PrizePicks JSON for the specific league *from the API*.
    We do NOT fake a league label; we filter by the league within the payload.
    """
    url = PRIZEPICKS_URL.format(league=league_slug.lower())
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 429:
                time.sleep(backoff * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"PrizePicks fetch failed for {league_slug}: {last_err}")

def pp_parse_rows(payload: dict) -> pd.DataFrame:
    """
    Convert PrizePicks payload into a flat DataFrame with columns:
    ['league','player','team','market_pp','market','line','opponent','game_date','projection_id']
    Only rows with a numeric line and recognized league in payload are returned.
    """
    included = {obj.get("id"): obj for obj in payload.get("included", [])}
    rows = []
    for item in payload.get("data", []):
        attr = item.get("attributes", {})
        relationships = item.get("relationships", {})
        proj_type = attr.get("stat_type") or attr.get("projection_type")
        line = attr.get("line_score")
        try:
            line = float(line)
        except (TypeError, ValueError):
            continue

        # Pull athlete & league safely
        athlete_rel = relationships.get("new_player") or relationships.get("player")
        athlete = included.get(athlete_rel and athlete_rel.get("data", {}).get("id"))
        player_name = (athlete or {}).get("attributes", {}).get("name") or ""

        league_rel = relationships.get("league")
        league_obj = included.get(league_rel and league_rel.get("data", {}).get("id"))
        league_slug = (league_obj or {}).get("attributes", {}).get("name") or ""
        league_slug = league_slug.lower()

        team = (athlete or {}).get("attributes", {}).get("team", "") or (athlete or {}).get("attributes", {}).get("team_name", "")
        opp = attr.get("opponent") or ""
        game_date = attr.get("start_time") or attr.get("start_time_tbd")

        rows.append({
            "league": league_slug,
            "player": player_name,
            "team": team,
            "market_pp": proj_type,         # PrizePicks label
            "line": line,
            "opponent": opp,
            "game_date": game_date,
            "projection_id": item.get("id"),
        })

    df = pd.DataFrame(rows)
    # Drop junk / empties
    if not df.empty:
        df = df.dropna(subset=["league", "player", "market_pp", "line"])
        # Normalize league text a bit
        df["league"] = df["league"].str.lower()
    return df

def normalize_market(league_slug: str, market_pp: str) -> str:
    m = MARKET_MAP_BY_LEAGUE.get(league_slug, {})
    return m.get(market_pp, "").strip().lower()

def infer_sd(market_key: str, line: float, board_stats: Dict[str, float]) -> float:
    """
    Pick an SD for the simulation:
    1) Use empirical SD from board if available (board_stats has per-market MAD->sd).
    2) Fallback to a heuristic SD table.
    3) As a last resort, use a % of the line.
    """
    if market_key in board_stats and board_stats[market_key] > 0:
        return board_stats[market_key]
    if market_key in SD_HEURISTICS_NORMAL:
        return SD_HEURISTICS_NORMAL[market_key]
    # generic fallback
    return max(1.0, 0.25 * max(1.0, float(line)))

def build_board_sd(df: pd.DataFrame, market_col: str = "market") -> Dict[str, float]:
    """
    Estimate per-market SD from the current board lines using a robust MAD-based estimate.
    sd ≈ 1.4826 * MAD(lines)
    """
    out = {}
    for mk, grp in df.groupby(market_col):
        arr = grp["line"].astype(float).values
        if len(arr) >= 6:
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            out[mk] = float(1.4826 * mad) if mad > 0 else 0.0
        else:
            out[mk] = 0.0
    return out

def simulate_row(mean: float, line: float, market_key: str, dist_hint: str, sd: float, n_sims: int = 20000) -> Tuple[float, float]:
    """
    Run Monte Carlo for one row & return (P_over, P_under).
    dist_hint: "normal" or "poisson" (or "auto")
    """
    if dist_hint == "poisson" or market_key in POISSON_MARKETS:
        lam = max(0.01, mean)
        draws = np.random.poisson(lam=lam, size=n_sims)
    else:
        # Clamp SD
        sd = max(0.5, float(sd))
        draws = np.random.normal(loc=mean, scale=sd, size=n_sims)
    pover = float(np.mean(draws > line))
    punder = 1.0 - pover
    return pover, punder

def market_distribution_hint(market_key: str) -> str:
    return "poisson" if market_key in POISSON_MARKETS else "normal"

# ---------------------------
# UI
# ---------------------------

st.title("📊 All-Sports PrizePicks Simulator (auto baselines, fixed parsing)")
league = st.selectbox("League", SUPPORTED_LEAGUES, index=0)

league_slug = league.lower()
st.caption("We pull only the chosen league from PrizePicks and filter strictly to that league in the payload.")

# Fetch & parse
try:
    payload = pp_fetch(league_slug)
    df_raw = pp_parse_rows(payload)
except Exception as e:
    st.error(f"PrizePicks fetch/parse error: {e}")
    st.stop()

if df_raw.empty:
    st.warning("No markets returned right now for this league.")
    st.stop()

# STRICT filter to the league that came from PrizePicks payload
df = df_raw[df_raw["league"] == league_slug].copy()
if df.empty:
    st.warning(f"PrizePicks returned no rows tagged with league='{league_slug}'. Try later.")
    st.stop()

# Normalize markets per-league mapping, drop rows we don't understand
df["market"] = df["market_pp"].apply(lambda m: normalize_market(league_slug, str(m)))
df = df[df["market"].astype(str) != ""].copy()

# Optional: view raw list
with st.expander("See parsed board (after strict filtering)"):
    st.dataframe(df[["league","player","team","market_pp","market","line","opponent","game_date"]].sort_values(["market","player"]).reset_index(drop=True), use_container_width=True)

# Baselines:
st.subheader("Simulation settings")
col_a, col_b, col_c = st.columns(3)
with col_a:
    n_sims = st.slider("Monte-Carlo simulations", 5000, 50000, 20000, step=5000)
with col_b:
    baseline_choice = st.selectbox("Baseline mean", ["Use line as mean", "Use market average line"], index=0)
with col_c:
    min_edge = st.slider("Minimum absolute edge to show", 0.00, 0.20, 0.02, step=0.01)

# Build market SDs from board
board_sd = build_board_sd(df)

# Choose baseline mean per row
if baseline_choice == "Use market average line":
    mk_means = df.groupby("market")["line"].mean().to_dict()
    df["mean_est"] = df.apply(lambda r: float(mk_means.get(r["market"], r["line"])), axis=1)
else:
    df["mean_est"] = df["line"].astype(float)

# Simulate per row with per-market distribution
P_over = []
P_under = []
for r in df.itertuples(index=False):
    mk = r.market
    line = float(r.line)
    mean = float(r.mean_est)
    sd = infer_sd(mk, line, board_sd)
    hint = market_distribution_hint(mk)
    pover, punder = simulate_row(mean, line, mk, hint, sd, n_sims=n_sims)
    P_over.append(pover)
    P_under.append(punder)

df["P(over)"] = P_over
df["P(under)"] = P_under
df["edge_over"] = df["P(over)"] - 0.5
df["edge_under"] = df["P(under)"] - 0.5
df["edge"] = df[["edge_over","edge_under"]].abs().max(axis=1)
df["pick"] = np.where(df["edge_over"] >= df["edge_under"], "OVER", "UNDER")
df["prob"] = np.where(df["pick"]=="OVER", df["P(over)"], df["P(under)"])
df["model_sd_used"] = df["market"].apply(lambda m: infer_sd(m, 0, board_sd))

# Show results (filtered by edge)
st.subheader("Simulated edges (by market, properly filtered)")
out_cols = ["league","player","team","market","market_pp","line","pick","prob","P(over)","P(under)","edge","model_sd_used"]
view = df[out_cols].sort_values("edge", ascending=False)
view = view[view["edge"] >= min_edge].reset_index(drop=True)

if view.empty:
    st.info("No props meet the current edge filter. Lower the threshold or try later.")
else:
    st.dataframe(view, use_container_width=True)

st.caption("Note: Without historical player stat baselines, means are derived from the current board (line or market average), and SDs are inferred from board dispersion + heuristics per market. This keeps leagues separated and avoids cross-market mixups.")
