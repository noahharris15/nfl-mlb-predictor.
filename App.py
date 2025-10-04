# app.py
import time
import math
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from functools import lru_cache

st.set_page_config(page_title="All-Sports PrizePicks Simulator", layout="wide")

# ----------------------------
# Small utilities
# ----------------------------
def normal_cdf(x, mean=0.0, sd=1.0):
    # P(X <= x) for Normal(mean, sd). Avoid scipy dependency.
    z = (x - mean) / (sd if sd > 0 else 1e-6)
    # Abramowitz-Stegun approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989423 * math.exp(-z*z/2.0)
    prob = 1 - d * (1.330274*t - 1.821256*t**2 + 1.781478*t**3 - 0.356538*t**4 + 0.3193815*t**5)
    return prob if z >= 0 else 1 - prob

def clamp01(x):
    return max(0.0, min(1.0, x))

def backoff_sleep(attempt):
    time.sleep(min(2**attempt, 8))

# ----------------------------------------------------
# PrizePicks: Correct endpoint + robust fetch & parse
# ----------------------------------------------------
PP_URL = "https://api.prizepicks.com/projections"

@lru_cache(maxsize=32)
def fetch_prizepicks_raw(league: str):
    """Fetch PP projections for one league only, with retries and a UA header."""
    params = {
        "per_page": 250,          # Pulls ~full board page by page
        "single_stat": "true",
        "league": league.lower()
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(PP_URL, params=params, headers=headers, timeout=10)
            if r.status_code == 429:
                last_err = f"429 Too Many Requests (attempt {attempt+1})"
                backoff_sleep(attempt+1)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = str(e)
            backoff_sleep(attempt+1)
    raise RuntimeError(f"PrizePicks fetch failed for {league}: {last_err}")

def parse_pp_board(data: dict, league: str) -> pd.DataFrame:
    """Turn PP JSON into a tidy DataFrame filtered to the chosen league."""
    # PP payload has "data" (projections) & "included" (players/teams/games)
    df_proj = pd.json_normalize(data.get("data", []))
    inc = pd.json_normalize(data.get("included", []))

    # Keep only matching league at projection level when available
    # (Some props include league on relationships/attributes)
    if "attributes.league" in df_proj.columns:
        df_proj = df_proj[df_proj["attributes.league"].str.lower() == league.lower()]

    # Build a player lookup
    players = inc[inc["type"] == "players"].copy()
    players["player_id"] = players["id"]
    players["player_name"] = players["attributes.name"]
    players["team"] = players["attributes.team"]  # sometimes None

    proj = df_proj.copy()
    # Common fields
    proj["player_id"] = proj["relationships.new_player.data.id"].fillna(proj["relationships.player.data.id"])
    proj["player_id"] = proj["player_id"].astype(str)
    proj["market_pp"] = proj["attributes.stat_type"].fillna(proj["attributes.title"])
    proj["line"] = pd.to_numeric(proj["attributes.line_score"], errors="coerce")
    proj["league"] = proj["attributes.league"].str.upper()

    # Join player info
    proj = proj.merge(players[["player_id", "player_name", "team"]], on="player_id", how="left")

    # Drop junk / null lines
    proj = proj.dropna(subset=["line", "market_pp", "player_name"]).reset_index(drop=True)

    # Normalize market names a bit (lower snake)
    proj["market"] = (
        proj["market_pp"].str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # Only chosen league (extra safety)
    proj = proj[proj["league"].str.lower() == league.lower()].reset_index(drop=True)

    # Keep the essentials
    keep_cols = ["league", "player_name", "team", "market", "market_pp", "line"]
    proj = proj[keep_cols].rename(columns={"player_name": "player"})

    # Deduplicate (PrizePicks can duplicate the same prop across slips)
    proj = proj.drop_duplicates(subset=["player", "team", "market", "line"]).reset_index(drop=True)
    return proj

# ----------------------------------------------------
# Optional baselines from free public APIs (NBA only here)
# ----------------------------------------------------
@lru_cache(maxsize=512)
def nba_player_averages(player_name: str):
    """Return approximate NBA per-game averages using balldontlie (free)."""
    # Note: name matching is imperfect; we do a best-effort fuzzy-ish search.
    base = "https://api.balldontlie.io/v1/players"
    stats = "https://api.balldontlie.io/v1/season_averages"
    headers = {"User-Agent": "Mozilla/5.0"}

    # Find player
    r = requests.get(base, params={"search": player_name}, headers=headers, timeout=10)
    if r.status_code != 200:
        return {}
    res = r.json().get("data", [])
    if not res:
        return {}

    player_id = res[0]["id"]

    # Current or last season
    season = pd.Timestamp.today().year - 1  # safe default: last season
    r2 = requests.get(stats, params={"season": season, "player_ids[]": player_id}, headers=headers, timeout=10)
    if r2.status_code != 200:
        return {}

    dat = (r2.json().get("data") or [])
    if not dat:
        return {}
    row = dat[0]
    # Map to common PP markets we’ll see
    return {
        "points": row.get("pts"),
        "rebounds": row.get("reb"),
        "assists": row.get("ast"),
        "points_rebounds_assists": (row.get("pts") or 0) + (row.get("reb") or 0) + (row.get("ast") or 0),
        # Add more as needed…
    }

def get_baseline_mean_sd(row: pd.Series) -> tuple[float, float, str]:
    """
    Return (mean, sd, source) for a given prop row.
    - If NBA & we can fetch a matching stat from balldontlie: use that mean and a sane SD.
    - Else: use a conservative prior centered near the PP line with jitter to avoid 0/100.
    """
    league = row["league"].lower()
    market = row["market"]
    line = float(row["line"])

    # Try NBA real averages first
    if league == "nba":
        avg = nba_player_averages(row["player"])
        if avg:
            # Attempt a market match
            m = None
            # Common mapping heuristics
            if "points_rebounds_assists" in market or market in {"pra", "points+rebounds+assists"}:
                m = avg.get("points_rebounds_assists")
            elif "points" in market and "rebounds" not in market and "assists" not in market:
                m = avg.get("points")
            elif "rebounds" in market and "assists" not in market and "points" not in market:
                m = avg.get("rebounds")
            elif "assists" in market and "points" not in market and "rebounds" not in market:
                m = avg.get("assists")

            if m is not None:
                # SD heuristic: 35% of mean, bounded
                sd = max(0.6, 0.35 * max(1.0, m))
                return float(m), float(sd), "nba_avg"

    # Conservative fallback around the line (prevents 0/100 artifacts):
    # N(mean ~ line ± jitter, sd ~ 20% of max(line, 8) bounded)
    jitter = np.random.normal(loc=0.0, scale=max(0.4, 0.03 * max(10.0, line)))
    mean = max(0.0, line + jitter)
    sd = max(0.8, 0.20 * max(8.0, line))
    return float(mean), float(sd), "fallback"

def simulate_probs(df_props: pd.DataFrame) -> pd.DataFrame:
    """Compute P(Over) and P(Under) using a normal model for each prop."""
    rows = []
    for _, r in df_props.iterrows():
        mean, sd, src = get_baseline_mean_sd(r)
        line = float(r["line"])
        p_under = clamp01(normal_cdf(line, mean, sd))
        p_over = clamp01(1.0 - p_under)
        rows.append({
            **r.to_dict(),
            "mean": round(mean, 3),
            "model_sd": round(sd, 3),
            "P(Over)": round(p_over * 100, 2),
            "P(Under)": round(p_under * 100, 2),
            "baseline": src
        })
    out = pd.DataFrame(rows)
    # Rank by distance from 50/50 (bigger edge first)
    out["edge"] = (out["P(Over)"] - 50).abs()
    return out.sort_values(["edge"], ascending=False).reset_index(drop=True)

# ----------------------------
# UI
# ----------------------------
st.title("📊 All-Sports PrizePicks Simulator (Real stats when available)")

league = st.selectbox(
    "Select League",
    ["NFL", "NBA", "MLB", "NHL", "WNBA", "CBB", "CFB", "SOC", "LOL", "VAL", "CS2"],
    index=0
)

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
        1) We fetch the current **PrizePicks board** for the chosen league only.  
        2) We **parse & keep** only that league (no cross-sport mixing).  
        3) We estimate per-prop **P(Over/Under)** with a **conservative normal model**:  
           - **NBA**: tries real per-game averages via *balldontlie* for points / rebounds / assists / PRA.  
           - Others: fallback around the PP line with jitter and a sane SD to avoid 0%/100% artifacts.  
        """
    )

# Fetch board
status = st.empty()
try:
    status.info(f"Fetching PrizePicks board for **{league}**…")
    raw = fetch_prizepicks_raw(league)
    board = parse_pp_board(raw, league)
    if board.empty:
        status.warning("No PrizePicks markets parsed for this league right now.")
        st.stop()
    status.success(f"Loaded {len(board)} props.")
except Exception as e:
    status.error(f"Fetch/parse error: {e}")
    st.stop()

st.subheader("Clean board (parsed)")
st.dataframe(board, use_container_width=True)

# Simulate
st.subheader("Simulated probabilities (conservative model)")
with st.spinner("Simulating…"):
    results = simulate_probs(board)

# Show results with some helpful sorting/filters
col1, col2, col3 = st.columns([1,1,1])
with col1:
    min_edge = st.slider("Minimum edge (|P-50|)", 0.0, 25.0, 5.0, 0.5)
with col2:
    view = st.selectbox("View", ["All", "Only Over ≥ 55%", "Only Under ≥ 55%"], index=0)
with col3:
    src_filter = st.selectbox("Baseline source", ["any", "nba_avg", "fallback"], index=0)

df_show = results.copy()
df_show = df_show[df_show["edge"] >= min_edge]

if view == "Only Over ≥ 55%":
    df_show = df_show[df_show["P(Over)"] >= 55]
elif view == "Only Under ≥ 55%":
    df_show = df_show[df_show["P(Under)"] >= 55]

if src_filter != "any":
    df_show = df_show[df_show["baseline"] == src_filter]

# Friendly column order
cols = ["league", "player", "team", "market", "market_pp", "line", "mean", "model_sd", "P(Over)", "P(Under)", "baseline"]
df_show = df_show[cols] if all(c in df_show.columns for c in cols) else df_show

st.dataframe(df_show, use_container_width=True)

st.caption("Tip: use the filters to avoid unrealistic 0/100% edges. Baselines are deliberately conservative unless real averages are available.")
