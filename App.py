# app_prizepicks_auto.py
# All-Sports PrizePicks Simulator (auto baselines; no CSVs)

import math
import time
import json
from typing import Dict, Tuple, Optional, List

import requests
import pandas as pd
import numpy as np
import streamlit as st


# ------------------------------ Utilities

def normal_cdf(x: float) -> float:
    """Standard normal CDF with math.erf (no SciPy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def prob_over_normal(mean: float, sd: float, line: float) -> float:
    sd = max(sd, 1e-6)
    z = (line - mean) / sd
    return 1.0 - normal_cdf(z)


# ------------------------------ Market mappers (very forgiving)

def market_map_generic(raw: str) -> Optional[str]:
    if not raw:
        return None
    r = raw.lower().replace("(", " ").replace(")", " ").replace("-", " ").replace("_", " ")
    r = " ".join(r.split())

    # NFL/NBA common
    if ("pass" in r or "passing" in r) and "yard" in r:      return "passing_yards"
    if ("rush" in r or "rushing" in r) and "yard" in r:       return "rushing_yards"
    if ("rec" in r or "receiving" in r) and "yard" in r:      return "receiving_yards"
    if "reception" in r and "yard" not in r:                  return "receptions"
    if "pass" in r and "td" in r:                             return "passing_tds"
    if ("rush" in r or "rushing" in r) and "td" in r:         return "rushing_tds"
    if ("rec" in r or "receiving" in r) and "td" in r:        return "receiving_tds"
    if "fg made" in r or "field goal made" in r:              return "fg_made"

    # NBA
    if "points" in r or "pts" in r:                           return "points"
    if "reb" in r or "rebounds" in r:                         return "rebounds"
    if "ast" in r or "assists" in r:                          return "assists"
    if "pr" in r and "pra" not in r:                          return "points_rebounds"
    if "pa" in r and "pra" not in r:                          return "points_assists"
    if "ra" in r and "pra" not in r:                          return "rebounds_assists"
    if "pra" in r:                                            return "points_rebounds_assists"
    if "3pt" in r or "three" in r:                            return "threes_made"
    if "blk" in r or "blocks" in r:                           return "blocks"
    if "stl" in r or "steals" in r:                           return "steals"
    if "stocks" in r or ("stl" in r and "blk" in r):          return "stocks"

    # MLB (lite)
    if "total bases" in r or "tb" == r:                       return "total_bases"
    if "hits allowed" in r:                                   return "hits_allowed"
    if "strikeouts" in r or "ks" in r:                        return "strikeouts"
    if "hits runs rbis" in r or "h+r+rbis" in r:              return "hrr"
    if "walks" in r and "pitcher" in r:                       return "walks"
    if "outs" in r and ("pitcher" in r or "pit" in r):        return "outs_recorded"
    if "home runs" in r and "allowed" not in r:               return "home_runs"

    # Fallback: one-word basics
    if "yard" in r:                                           return "yards"
    if "td" in r:                                             return "tds"
    return None


# ------------------------------ Baselines (auto from current board)

def baselines_from_board(df: pd.DataFrame) -> Dict[Tuple[str], Dict[str, float]]:
    """
    Compute per-market normal baselines from the current board lines:
    mean = average line on the board; sd = std of lines on the board (with guard).
    """
    baselines = {}
    for mkt, grp in df.groupby("market"):
        vals = grp["line"].astype(float)
        if len(vals) >= 2:
            mu = float(vals.mean())
            sd = float(vals.std(ddof=1))
            if not np.isfinite(sd) or sd <= 1e-6:
                sd = max(0.12 * mu, 1.5)  # guard
        else:
            mu = float(vals.iloc[0])
            sd = max(0.15 * mu, 2.0)
        baselines[(mkt,)] = {"mean": mu, "sd": sd, "dist": "normal"}
    return baselines


def fallback_prior(market: str) -> Dict[str, float]:
    """If a market is missing on the board, use a simple prior so we still compute."""
    m = market
    # chosen to be conservative; sd wide to avoid overconfidence
    if m in {"points"}:                      return {"mean": 15.0, "sd": 6.0, "dist": "normal"}
    if m in {"rebounds"}:                    return {"mean": 7.0,  "sd": 3.0, "dist": "normal"}
    if m in {"assists"}:                     return {"mean": 5.0,  "sd": 2.5, "dist": "normal"}
    if "yards" in m:                         return {"mean": 55.0, "sd": 22.0, "dist": "normal"}
    if "receptions" in m:                    return {"mean": 4.0,  "sd": 2.0, "dist": "normal"}
    if "td" in m:                            return {"mean": 0.5,  "sd": 0.6, "dist": "normal"}
    if m in {"total_bases"}:                 return {"mean": 1.7,  "sd": 0.9, "dist": "normal"}
    if m in {"strikeouts"}:                  return {"mean": 5.5,  "sd": 2.0, "dist": "normal"}
    return {"mean": 8.0, "sd": 4.0, "dist": "normal"}


# ------------------------------ Sport registry (FIX for your NameError)

SPORTS: Dict[str, Tuple] = {
    # You can add more leagues later; these three are enabled now.
    "NFL": (market_map_generic, baselines_from_board),
    "NBA": (market_map_generic, baselines_from_board),
    "MLB": (market_map_generic, baselines_from_board),
}


# ------------------------------ PrizePicks fetch (with retry + pagination)

@st.cache_data(show_spinner=False, ttl=60)
def fetch_prizepicks(league: str, max_pages: int = 4) -> pd.DataFrame:
    """
    Pull PrizePicks projections for a given league.
    Returns a DataFrame with columns: player, team, market_raw, line, league.
    """
    league_param = league.lower()
    base = "https://api.prizepicks.com/projections"
    headers = {
        "User-Agent": "Mozilla/5.0 (PP-Research/1.0; +streamlit-app)",
        "Accept": "application/json",
    }

    all_rows: List[dict] = []
    for page in range(1, max_pages + 1):
        url = f"{base}?per_page=1000&single_stat=true&league={league_param}&page={page}"
        for attempt in range(4):
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                # Handle rate limiting
                if resp.status_code == 429:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.JSONDecodeError:
                # Sometimes CF/CDN responses are empty for a tick
                time.sleep(0.8)
                continue
            except requests.RequestException:
                time.sleep(0.8 * (attempt + 1))
                continue
        else:
            # failed all attempts; continue to next page
            continue

        items = data.get("data", [])
        included = data.get("included", [])
        if not items:
            break

        # map included players by id -> name/team
        name_by_id = {}
        team_by_id = {}
        for inc in included:
            if inc.get("type") == "new_player":
                pid = inc.get("id")
                attrs = inc.get("attributes", {})
                name_by_id[pid] = attrs.get("name") or attrs.get("display_name") or "Unknown"
                team_by_id[pid] = (attrs.get("team") or attrs.get("team_name") or "").strip()

        # rows
        for it in items:
            attrs = it.get("attributes", {})
            rel = it.get("relationships", {}) or {}
            player_id = (rel.get("new_player", {}).get("data") or {}).get("id")
            player = name_by_id.get(player_id, "Unknown")
            team = team_by_id.get(player_id, "")

            market_raw = (
                attrs.get("stat_type")
                or attrs.get("display_stat")  # sometimes present
                or attrs.get("description")
                or ""
            )
            try:
                line = float(attrs.get("line_score"))
            except Exception:
                continue

            all_rows.append(
                {
                    "league": league.upper(),
                    "player": player,
                    "team": team,
                    "market_raw": market_raw,
                    "line": line,
                }
            )

        # small pause to be kind to their API
        time.sleep(0.25)

    df = pd.DataFrame(all_rows)
    return df


# ------------------------------ Player-level distribution params

def attach_params(df: pd.DataFrame, sport: str,
                  market_map_fn, baselines_fn) -> pd.DataFrame:
    """Map markets, compute market baselines from board, attach mean/sd per row."""
    if df.empty:
        return df

    df = df.copy()
    df["market"] = df["market_raw"].apply(market_map_fn)
    df = df[df["market"].notna()].reset_index(drop=True)
    if df.empty:
        return df

    # Baselines per market from current board lines
    mk_baselines = baselines_fn(df)

    # Player params: shrink toward market mean so we don't overfit the single line
    means = []
    sds = []
    for _, row in df.iterrows():
        mkt = row["market"]
        line = float(row["line"])
        base = mk_baselines.get((mkt,), None) or fallback_prior(mkt)
        mu = 0.55 * line + 0.45 * base["mean"]     # light shrinkage to market
        sd = max(base["sd"], 1e-6)
        means.append(mu)
        sds.append(sd)

    df["mu"] = means
    df["sd"] = sds
    return df


# ------------------------------ Simulation (analytic normal)

def evaluate_over_under(df: pd.DataFrame) -> pd.DataFrame:
    """Add probabilities & edges based on normal model."""
    if df.empty:
        return df
    p_over = []
    p_under = []
    for _, r in df.iterrows():
        po = prob_over_normal(r["mu"], r["sd"], r["line"])
        p_over.append(po)
        p_under.append(1.0 - po)
    df = df.copy()
    df["P(over)"] = p_over
    df["P(under)"] = p_under
    df["edge"] = (df["P(over)"] - 0.5).abs()
    return df


# ------------------------------ Streamlit UI

st.set_page_config(page_title="All-Sports PrizePicks Simulator", page_icon="📊", layout="wide")
st.title("📊 All-Sports PrizePicks Simulator (auto baselines)")

league = st.selectbox("League", list(SPORTS.keys()), index=0)

market_map_fn, baselines_fn = SPORTS[league]   # <- this was the NameError before; now fixed.

with st.status("Fetching PrizePicks board…", expanded=False):
    board = fetch_prizepicks(league)
    st.write(f"Rows fetched: **{len(board)}**")
    if st.checkbox("Show raw board sample"):
        st.dataframe(board.head(20), use_container_width=True)

if board.empty:
    st.warning("No PrizePicks markets parsed for this league right now.")
    st.stop()

with st.status("Building baselines and evaluating…", expanded=False):
    modeled = attach_params(board, league, market_map_fn, baselines_fn)
    if modeled.empty:
        st.error("Could not map any PrizePicks markets to supported stats.")
        st.stop()
    results = evaluate_over_under(modeled)
    st.write("Markets mapped:", int(results["market"].nunique()))

# Controls
left, right = st.columns([3, 2])
with left:
    q = st.text_input("Filter by player / team / market", "")
with right:
    min_edge = st.slider("Minimum edge (|P-0.5|)", 0.00, 0.30, 0.06, 0.01)

view = results.copy()
if q:
    ql = q.lower()
    view = view[
        view["player"].str.lower().str.contains(ql)
        | view["team"].str.lower().str.contains(ql)
        | view["market"].str.lower().str.contains(ql)
        | view["market_raw"].str.lower().str.contains(ql)
    ]

view = view[(view["edge"] >= min_edge)].sort_values("edge", ascending=False)

pretty = view[[
    "league", "player", "team", "market", "market_raw", "line", "mu", "sd", "P(over)", "P(under)", "edge"
]].rename(columns={
    "market_raw": "market_pp",
    "mu": "model_mean",
    "sd": "model_sd",
    "edge": "edge_vs_50%"
})
st.subheader("Simulated edges (normal model)")
st.caption("Probabilities are model estimates based on auto baselines from the current board (no historical CSVs).")
st.dataframe(pretty.reset_index(drop=True), use_container_width=True)

# Download
csv = pretty.to_csv(index=False).encode("utf-8")
st.download_button("Download results (.csv)", data=csv, file_name=f"{league}_prizepicks_sim.csv", mime="text/csv")

st.info(
    "Tip: If the page sometimes shows a 429 error from PrizePicks, just wait ~30–60 seconds and click **Rerun** "
    "(we backoff/retry and cache for 60s to be gentle on the API)."
)
