# app.py  —  All-Sports PrizePicks Simulator (clean parsing, sane sims)
import math
import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="All-Sports PrizePicks Simulator", layout="wide")

# -----------------------------
# PrizePicks fetch w/ backoff
# -----------------------------
PP_URL = "https://site.api.prizepicks.com/api/v1/projections"
PP_HEADERS = {
    # A very plain UA works fine; rotate if you get 403s.
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}

# Canonical league codes PrizePicks uses in the projections payload
LEAGUE_CODES = {
    "NFL": "nfl",
    "NBA": "nba",
    "MLB": "mlb",
    "NHL": "nhl",
    "NCAAF": "college-football",
    "NCAAB": "college-basketball",
}

# Map PrizePicks market labels -> canonical stat keys you want to model
# (Add more as you need. Unknown markets will be dropped.)
MARKET_MAP = {
    # Football
    "Pass Yards": "pass_yards",
    "Pass Attempts": "pass_att",
    "Pass Completions": "pass_comp",
    "Pass Touchdowns": "pass_tds",
    "Interceptions": "interceptions",
    "Rush Yards": "rush_yards",
    "Rush Attempts": "rush_att",
    "Receptions": "receptions",
    "Receiving Yards": "rec_yards",
    "Receiving + Rush Yds": "rush_rec_yards",
    "Fantasy Score": "fantasy",
    # Basketball
    "Points": "points",
    "Rebounds": "rebounds",
    "Assists": "assists",
    "Pts+Reb+Ast": "pra",
    "3-PT Made": "threes",
    "Blocks": "blocks",
    "Steals": "steals",
    "Blocks+Steals": "stocks",
    # Baseball
    "Hitter Fantasy Score": "fantasy",
    "Hits": "hits",
    "TB": "total_bases",
    "Runs": "runs",
    "RBI": "rbi",
    "Walks": "walks",
    "Hits + Runs + RBIs": "hrr",
    "Pitcher Strikeouts": "ks",
    "Pitcher Outs": "outs",
    "Pitcher Walks": "bb",
    "Pitcher Fantasy Score": "fantasy_p",
    # Hockey
    "Shots On Goal": "sog",
    "Goalie Saves": "saves",
    "Points (NHL)": "points_nhl",
}

# Conservative fallback SDs to avoid 0/100s (tuned by sport & stat family)
# If a stat isn't present, we’ll use 0.18 * line (floor to 1.0)
SD_FALLBACK = {
    "NFL": {
        "pass_yards": 55, "rush_yards": 28, "rec_yards": 32,
        "receptions": 1.2, "pass_tds": 0.9, "interceptions": 0.7,
        "fantasy": 7.5, "rush_rec_yards": 45,
    },
    "NCAAF": {  # CFB is noisier
        "pass_yards": 75, "rush_yards": 35, "rec_yards": 38,
        "receptions": 1.5, "pass_tds": 1.1, "interceptions": 0.8,
        "fantasy": 9.0, "rush_rec_yards": 55,
    },
    "NBA": {
        "points": 7.5, "rebounds": 3.2, "assists": 2.9, "pra": 9.5,
        "threes": 1.4, "stocks": 1.4, "blocks": 1.1, "steals": 1.1,
        "fantasy": 8.5,
    },
    "MLB": {
        "ks": 1.8, "outs": 2.9, "bb": 0.8, "fantasy_p": 6.0,
        "hits": 0.7, "walks": 0.6, "rbi": 0.7, "runs": 0.7,
        "total_bases": 1.1, "hrr": 1.4, "fantasy": 6.0,
    },
    "NHL": {"sog": 1.6, "saves": 5.5, "points_nhl": 0.9},
    "NCAAB": {"points": 6.0, "rebounds": 3.0, "assists": 2.4, "pra": 8.0, "threes": 1.2},
}

def z_to_prob_over(z: float) -> float:
    # standard normal cdf complement for "over"
    return float(0.5 * (1 - math.erf(-z / math.sqrt(2))))

def prob_over_normal(line, mean, sd):
    sd = max(1e-6, sd)  # avoid div by zero
    z = (mean - line) / sd
    p_over = z_to_prob_over(z)
    # clamp extreme certainty to avoid 0/1 due to rough SDs
    return float(np.clip(p_over, 0.01, 0.99))

@st.cache_data(show_spinner=False, ttl=90)
def fetch_prizepicks(league_code: str) -> dict:
    # Pull one league at a time and only single-stat markets to keep payloads small
    params = {"per_page": 250, "single_stat": "true", "league": league_code}
    tries, wait = 6, 1.0
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(PP_URL, headers=PP_HEADERS, params=params, timeout=15)
            if r.status_code == 429:  # rate limited
                time.sleep(wait); wait *= 1.7; continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(wait); wait *= 1.7
    raise RuntimeError(f"PrizePicks fetch failed for {league_code}: {last_err}")

def parse_pp(payload: dict, want_league_code: str) -> pd.DataFrame:
    """
    Build a clean dataframe with ONLY the chosen league.
    Columns: league, player, team, market, stat_key, line
    """
    incl = {obj["id"]: obj for obj in payload.get("included", [])}
    rows = []
    for proj in payload.get("data", []):
        attr = proj.get("attributes", {})
        # 1) Strict league filter
        if (attr.get("league") or "") != want_league_code:
            continue

        rel = proj.get("relationships", {})
        # Pull linked player object for clean name/team if present
        player_name, team_abbr = None, None
        plink = rel.get("new_player")
        if plink and isinstance(plink.get("data"), dict):
            pid = plink["data"].get("id")
            pobj = incl.get(pid, {}).get("attributes", {})
            player_name = pobj.get("name") or pobj.get("display_name")
            team_abbr = pobj.get("team") or pobj.get("team_name")

        stat_label = attr.get("stat_type") or attr.get("display_stat")
        market_pp = attr.get("market_type") or attr.get("label") or stat_label
        line_val = attr.get("line_score") or attr.get("projection")
        # Use MARKET_MAP to keep only supported markets
        stat_key = MARKET_MAP.get(stat_label) or MARKET_MAP.get(market_pp)
        if stat_key is None:
            continue  # unknown/complex market -> ignore

        try:
            line = float(line_val)
        except Exception:
            continue

        rows.append({
            "league": want_league_code,
            "player": player_name or attr.get("new_player_name") or "Unknown",
            "team": team_abbr or attr.get("team") or "",
            "market_pp": stat_label or market_pp,
            "stat_key": stat_key,
            "line": line,
        })

    df = pd.DataFrame(rows)
    # De-dup identical (player, stat_key, line)
    if not df.empty:
        df = df.drop_duplicates(subset=["league", "player", "stat_key", "line"]).reset_index(drop=True)
    return df

def choose_sd(row) -> float:
    # Use league/stat SD fallback first; else a line-proportional fallback
    league = row["league_disp"]
    stat = row["stat_key"]
    line = float(row["line"])
    sd = SD_FALLBACK.get(league, {}).get(stat)
    if sd is None:
        sd = max(1.0, 0.18 * abs(line))
    return float(sd)

def estimate_mean(row) -> float:
    """
    Without a free, stable, multi-sport stats API, we use a conservative mean:
    baseline = line * 0.98 for unders/overs to not be trivially 50/50.
    You can replace this with true player averages per league later.
    """
    return float(row["line"]) * 0.98

# -----------------------------
# UI
# -----------------------------
st.title("📊 All-Sports PrizePicks Simulator (clean parsing + sane sims)")
colA, colB = st.columns([1, 2])
with colA:
    league_disp = st.selectbox("Select League", list(LEAGUE_CODES.keys()), index=0)
with colB:
    st.caption("We fetch PrizePicks → keep only this league → normalize markets → "
               "estimate P(Over/Under) using a conservative normal model to avoid 0/100% artifacts.")

league_code = LEAGUE_CODES[league_disp]

# -----------------------------
# Fetch + Parse
# -----------------------------
with st.status("Fetching PrizePicks board…", expanded=False):
    try:
        raw = fetch_prizepicks(league_code)
    except Exception as e:
        st.error(f"Fetch error: {e}")
        st.stop()

df = parse_pp(raw, league_code)

if df.empty:
    st.warning("No PrizePicks markets parsed for this league right now.")
    st.stop()

df["league_disp"] = league_disp  # for SD fallback dict
# Conservative mean & SD; you can wire in real averages per sport later
df["mean"] = df.apply(estimate_mean, axis=1)
df["model_sd"] = df.apply(choose_sd, axis=1)

# Probabilities
df["P(over)"] = df.apply(lambda r: prob_over_normal(r["line"], r["mean"], r["model_sd"]), axis=1)
df["P(under)"] = 1.0 - df["P(over)"]
df["edge_over"] = df["P(over)"] - 0.5
df["edge_under"] = df["P(under)"] - 0.5

# Present
show = df[[
    "player", "team", "market_pp", "stat_key",
    "line", "mean", "model_sd",
    "P(over)", "P(under)", "edge_over", "edge_under"
]].copy()

# Sort by biggest edge in either direction
show["abs_edge"] = show[["edge_over", "edge_under"]].abs().max(axis=1)
show = show.sort_values("abs_edge", ascending=False).drop(columns=["abs_edge"]).reset_index(drop=True)

st.subheader("Simulated edges (league-filtered, conservative model)")
st.caption("Probabilities come from a normal model with sport/stat-specific fallback variance. "
           "Values are clamped between 1% and 99% to avoid false certainty.")
st.dataframe(show, use_container_width=True)

# Download
st.download_button(
    "Download as CSV",
    data=show.to_csv(index=False).encode("utf-8"),
    file_name=f"pp_sim_{league_disp.lower()}.csv",
    mime="text/csv",
)
