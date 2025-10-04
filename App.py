# all_sports_auto_with_retries.py

import time
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="All-Sports PrizePicks Simulator (robust)", layout="wide")
st.title("🏀🏈 All-Sports PrizePicks Simulator (auto baselines)")

# PrizePicks API config
PP_URL = "https://api.prizepicks.com/projections"
PP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Origin": "https://www.prizepicks.com",
    "Referer": "https://www.prizepicks.com",
}

LEAGUE_SLUGS = {
    "NFL": "nfl",
    "NBA": "nba",
    "MLB": "mlb",
    "CFB (NCAA-F)": "college-football",
}

# Utility: fetch with retries on 429/HTTP errors
def fetch_pp_board_with_retries(slug: str, max_retries=3, backoff_factor=1.5):
    params = {"per_page": 1000, "single_stat": "true", "league": slug}
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.get(PP_URL, params=params, headers=PP_HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429:
                st.warning(f"Rate limit hit (429). Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= backoff_factor
                continue
            else:
                raise
        except Exception as e:
            st.error(f"Unexpected fetch error: {e}")
            raise
    st.error("Failed to fetch PrizePicks after retries.")
    return None

# Market mapping & baseline skeletons (reuse from earlier)
# You can copy your existing market_map, baseline builders, fallback_prior, etc.

def parse_pp_board(payload, market_map_fn):
    if payload is None:
        return pd.DataFrame()
    players = {}
    for inc in payload.get("included", []):
        if inc.get("type") in ("new_player", "players"):
            a = inc.get("attributes", {}) or {}
            pid = str(inc.get("id"))
            name = a.get("name") or (f"{a.get('first_name','')} {a.get('last_name','')}").strip()
            players[pid] = {"name": name or "", "team": a.get("team"), "pos": a.get("position")}
    rows = []
    for d in payload.get("data", []):
        a = d.get("attributes", {}) or {}
        rels = d.get("relationships", {}) or {}
        raw = (a.get("projection_type") or a.get("stat_type") or "").lower().strip()
        try:
            line = a.get("line_score") or a.get("value") or a.get("line")
            line = float(line)
        except Exception:
            continue
        pid = None
        for key in ("new_player","player","athlete"):
            if key in rels and isinstance(rels[key].get("data"), dict):
                pid = str(rels[key]["data"].get("id"))
                break
        pl = players.get(pid, {})
        m = market_map_fn(raw)
        if not m:
            continue
        rows.append({
            "player": pl.get("name","").strip(),
            "team": pl.get("team"),
            "pos": pl.get("pos"),
            "raw_market": raw,
            "market": m,
            "line": line,
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["player","market","line"])
    df = df.reset_index(drop=True)
    return df

# (Here, insert your sport configs, market_map, baseline builders, and fallback_prior
#  from the earlier “all_sports_auto.py” sample. For brevity I omit them here.)

# --- App logic starts here ---
sport_name = st.selectbox("League", list(LEAGUE_SLUGS.keys()), index=0)
market_map_fn, baselines_fn = SPORTS[sport_name]  # from prior code block
slug = LEAGUE_SLUGS[sport_name]

with st.spinner("Fetching PrizePicks board…"):
    payload = fetch_pp_board_with_retries(slug)
df_pp = parse_pp_board(payload, market_map_fn)

if df_pp.empty:
    st.warning("No PrizePicks markets parsed for this league right now.")
    # Optionally show raw payload for debugging
    if payload is not None:
        st.json(payload.get("data", [])[:5])
    st.stop()

st.write(f"Loaded **{len(df_pp)}** PrizePicks lines.")

with st.spinner("Computing baselines…"):
    try:
        baselines = baselines_fn(df_pp)
    except Exception as e:
        st.error(f"Baseline construction error: {e}")
        baselines = {}

def compute_p_over(row):
    key = (row["player"], row["market"])
    base = baselines.get(key)
    if base is None:
        base = fallback_prior(sport_name, row["market"])
    if base.dist == "poisson":
        return poisson_over_prob(base.mean, row["line"])
    else:
        return normal_over_prob(base.mean, base.sd, row["line"])

df_pp["p_over"] = df_pp.apply(compute_p_over, axis=1)
df_pp["p_under"] = 1.0 - df_pp["p_over"]
df_pp["edge_over"] = df_pp["p_over"] - 0.5
df_pp["edge_under"] = df_pp["p_under"] - 0.5

# Display & filtering
markets = sorted(df_pp["market"].unique())
pick_market = st.selectbox("Filter market", ["All"] + markets)
q = st.text_input("Search player/team")

view = df_pp.copy()
if pick_market != "All":
    view = view[view["market"] == pick_market]
if q.strip():
    ql = q.lower()
    view = view[
        view["player"].str.lower().str.contains(ql, na=False) |
        view["team"].astype(str).str.lower().str.contains(ql, na=False)
    ]

sort_opt = st.selectbox("Sort by", ["edge_over","edge_under","p_over","p_under"])
top_n = st.number_input("Top N rows", min_value=5, max_value=len(view), value=20)

view = view.sort_values(sort_opt, ascending=False).head(int(top_n))

st.dataframe(
    view[["player","team","pos","market","line","p_over","p_under","edge_over","edge_under"]]
      .style.format({"line":"{:.2f}", "p_over":"{:.1%}", "p_under":"{:.1%}", "edge_over":"{:.1%}", "edge_under":"{:.1%}"})
)

st.success("Simulation done.")
