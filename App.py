# app.py  — All-Sports PrizePicks Simulator (robust parsing + sane probabilities)

import math
import time
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ------------------------------------------------------------
# UI SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="All-Sports PrizePicks Simulator", page_icon="📊", layout="wide")
st.title("📊 All-Sports PrizePicks Simulator (Realistic, league-clean, robust parsing)")

# ------------------------------------------------------------
# Robust PrizePicks fetch/parse (handles old/new payloads)
# ------------------------------------------------------------
def fetch_prizepicks_board(league: str, per_page: int = 250, max_retries: int = 4) -> pd.DataFrame:
    """
    Fetch PrizePicks projections for a single league and return a tidy DataFrame.
    Handles both:
      - relationships.player.data.id            (old)
      - relationships.new_player.data.id        (new)
    Also builds player details from `included` (type 'player' or 'new_player').
    """
    league = (league or "").lower().strip()
    base = "https://api.prizepicks.com/projections"
    params = {
        "per_page": per_page,
        "single_stat": "true",
        "league": league,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PP-Streamlit/1.0)",
        "Accept": "application/json",
    }

    last_err: Optional[Exception] = None
    for i in range(max_retries):
        try:
            r = requests.get(base, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            payload: Dict[str, Any] = r.json() if r.content else {"data": [], "included": []}
            data = payload.get("data", []) or []
            included = payload.get("included", []) or []

            # Build player lookup
            player_by_id: Dict[str, Dict[str, Any]] = {}
            for inc in included:
                typ = inc.get("type")
                if typ not in ("player", "new_player"):
                    continue
                pid = inc.get("id")
                attrs = inc.get("attributes", {}) or {}
                name = (
                    attrs.get("name")
                    or attrs.get("display_name")
                    or attrs.get("full_name")
                    or attrs.get("nickname")
                    or ""
                )
                team = (
                    attrs.get("team")
                    or attrs.get("team_name")
                    or attrs.get("pro_team")
                    or attrs.get("league_team")
                    or ""
                )
                player_by_id[pid] = {"player": name, "team": team}

            rows: List[Dict[str, Any]] = []
            for item in data:
                attrs = item.get("attributes", {}) or {}

                item_league = (attrs.get("league") or "").lower()
                # Hard filter: keep only chosen league
                if league and item_league and item_league != league:
                    continue

                line = (
                    attrs.get("line_score")
                    or attrs.get("line_score_decimal")
                    or attrs.get("value")
                )
                market_pp = (
                    attrs.get("stat_type")
                    or attrs.get("stat")
                    or attrs.get("display_stat")
                )

                rel = item.get("relationships", {}) or {}
                player_rel = rel.get("new_player") or rel.get("player") or {}
                player_id = (player_rel.get("data") or {}).get("id")

                pinfo = player_by_id.get(player_id, {})
                rows.append(
                    {
                        "league": (item_league.upper() or league.upper()),
                        "player_id": player_id,
                        "player": pinfo.get("player", ""),
                        "team": pinfo.get("team", ""),
                        "market_pp": market_pp,
                        "line_raw": line,
                    }
                )

            df = pd.DataFrame(rows)
            if df.empty:
                return df

            # Clean columns
            df["line"] = pd.to_numeric(df["line_raw"], errors="coerce")
            df = df.dropna(subset=["player", "market_pp", "line"])
            return df.reset_index(drop=True)

        except Exception as e:
            last_err = e
            # polite backoff
            time.sleep(1.5 * (2 ** i))

    # If we got here, we failed
    raise RuntimeError(f"PrizePicks fetch/parse error: {last_err}")

# ------------------------------------------------------------
# Market normalization (keeps names tidy & comparable)
# ------------------------------------------------------------
MARKET_MAP = {
    # Basketball
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "rebounds_assists": "rebounds_assists",
    "pts_rebs_asts": "pra",
    "threes_made": "threes",
    "fantasy_score": "fantasy",

    # Football (NFL)
    "pass_yards": "pass_yards",
    "rush_yards": "rush_yards",
    "receiving_yards": "rec_yards",
    "rec_yds": "rec_yards",
    "receptions": "receptions",
    "rush_attempts": "rush_att",
    "pass_completions": "pass_cmp",

    # Baseball (examples)
    "hits": "hits",
    "total_bases": "tb",
    "strikeouts": "ks",
}

def normalize_market(s: str) -> str:
    if not s:
        return ""
    key = s.lower().strip().replace(" ", "_")
    return MARKET_MAP.get(key, key)

# ------------------------------------------------------------
# Conservative model: estimate mean/sd and compute P(Over/Under)
# (We avoid 0/100 artifacts by using sensible floors/caps.)
# ------------------------------------------------------------
# Default per-market SDs if we have no real variance (tuned conservatively)
DEFAULT_SD = {
    # NBA-ish
    "points": 6.0,
    "rebounds": 3.0,
    "assists": 2.5,
    "pra": 7.5,
    "threes": 1.5,

    # NFL-ish
    "pass_yards": 40.0,
    "rush_yards": 20.0,
    "rec_yards": 22.0,
    "receptions": 1.6,
    "rush_att": 3.0,
    "pass_cmp": 4.0,

    # MLB-ish
    "hits": 0.6,
    "tb": 1.0,
    "ks": 1.8,

    # Fallback
    "fantasy": 8.0,
}

VAR_FLOOR = 1e-6        # never allow zero variance
Z_CAP = 5.25            # keeps probs away from 0/100 extremes

def normal_cdf(x: float, mean: float, sd: float) -> float:
    sd = max(sd, math.sqrt(VAR_FLOOR))
    z = (x - mean) / sd
    # cap z to avoid perfect 0/1
    z = max(min(z, Z_CAP), -Z_CAP)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def estimate_market_mean(row: pd.Series) -> float:
    """
    Simple neutral baseline: use the line as a central estimate
    (this is conservative if you don't have real per-player averages).
    If you have real averages in your pipeline, plug them in here.
    """
    return float(row["line"])

def estimate_market_sd(market: str, line: float) -> float:
    sd = DEFAULT_SD.get(market, None)
    if sd is None:
        # generic heuristic if a new market sneaks in
        sd = max(0.08 * float(line), 1.0)
    return float(sd)

def compute_probs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["market"] = df["market_pp"].map(normalize_market)

    # Keep only rows we can model
    df = df.dropna(subset=["line", "market"])

    # Estimate mean/sd
    df["avg"] = df.apply(estimate_market_mean, axis=1)
    df["model_sd"] = df.apply(lambda r: estimate_market_sd(r["market"], r["line"]), axis=1)

    # Probabilities
    # P(Over) = 1 - CDF(line)
    df["P(Over)"] = 1.0 - df.apply(lambda r: normal_cdf(r["line"], r["avg"], r["model_sd"]), axis=1)
    df["P(Under)"] = 1.0 - df["P(Over)"]

    # Keep tidy columns
    keep = ["league", "player", "team", "market", "market_pp", "line", "avg", "model_sd", "P(Over)", "P(Under)"]
    return df[keep].reset_index(drop=True)

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
LEAGUES = ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB"]

col_a, col_b = st.columns([1, 1])
with col_a:
    league = st.selectbox("Select League", LEAGUES, index=0)

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
- We fetch the **current** PrizePicks board for the league you select and **filter strictly** to that league.
- Market names are normalized (e.g., `rec_yds` → `rec_yards`) so you don’t get cross-sport or odd label artifacts.
- We use a **conservative normal model** with variance floors to avoid silly **0%/100%** results.
- If you later add real player averages/variance, plug them into `estimate_market_mean` / `estimate_market_sd`.
        """
    )

debug = st.checkbox("Show debug logs", value=False)

# Fetch & simulate
try:
    with st.spinner("Fetching PrizePicks board…"):
        raw = fetch_prizepicks_board(league)

    if raw.empty:
        st.warning("No markets parsed for this league right now.")
    else:
        modeled = compute_probs(raw)

        # Display results
        st.subheader("Simulated edges (conservative normal model)")
        st.caption("Probabilities are model estimates based on a neutral baseline from the current board.")
        st.dataframe(
            modeled.sort_values(["P(Over)"], ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

        # Download
        csv = modeled.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=f"pp_{league.lower()}_sims.csv", mime="text/csv")

        if debug:
            st.divider()
            st.subheader("Debug: raw parsed rows")
            st.dataframe(raw.head(200), use_container_width=True)

except Exception as e:
    st.error(f"Fetch/parse error: {e}")
    if debug:
        st.exception(e)
