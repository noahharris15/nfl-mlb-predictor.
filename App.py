# streamlit_app.py
# One script with 3 pages: NFL Sim, MLB Sim, Player Props (CSV-driven).
# - NFL/MLB: simple team-vs-team Poisson simulation using uploaded team stats
# - Player Props: upload props CSV (lines) + optional player-averages CSV, simulate P(Over/Under)

import io
import math
import random
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm, poisson
from rapidfuzz import fuzz

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="NFL/MLB Sims + Player Props", layout="wide")
st.title("🏈⚾️ NFL / MLB Sims + Player Props (CSV)")

page = st.sidebar.radio("Pages", ["NFL Sim", "MLB Sim", "Player Props"])

# --------------------------------------------------
# Utilities (shared)
# --------------------------------------------------
def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").replace("'", "").strip().lower()

def best_name_match(name: str, candidates: List[str], score_cut=82) -> Optional[str]:
    nm = clean_name(name); best=None; best_sc=-1
    for c in candidates:
        sc = fuzz.token_sort_ratio(nm, clean_name(c))
        if sc > best_sc:
            best, best_sc = c, sc
    return best if best_sc >= score_cut else None

def conservative_sd(avg, minimum=0.75, frac=0.30):
    if pd.isna(avg): return 1.25
    if avg <= 0:     return 1.0
    return max(max(frac * float(avg), minimum), 0.5)

def simulate_prob(avg, line):
    sd = conservative_sd(avg)
    p_over  = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over*100, 2), round(p_under*100, 2), round(sd, 3)

def poisson_match_sim(lambda_home: float, lambda_away: float, sims: int = 20000, home_edge: float = 0.0):
    """
    Simple Poisson match simulator. Optionally add a small home_edge (in runs/points) to home team lambda.
    Returns arrays of (home_score, away_score).
    """
    lam_h = max(lambda_home + home_edge, 0.01)
    lam_a = max(lambda_away, 0.01)
    home = np.random.poisson(lam=lam_h, size=sims)
    away = np.random.poisson(lam=lam_a, size=sims)
    return home, away

def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# --------------------------------------------------
# NFL SIM PAGE
# --------------------------------------------------
if page == "NFL Sim":
    st.header("🏈 NFL Team Simulator (CSV)")
    st.caption(
        "Upload a team stats CSV with columns at minimum: **team, off_ppg, def_ppg**. "
        "We estimate team scoring rates and simulate a game (Poisson)."
    )

    nfl_csv = st.file_uploader("Upload NFL team stats CSV", type=["csv"])
    if nfl_csv is None:
        st.info("No file uploaded yet. Example columns:\n\nteam,off_ppg,def_ppg")
        st.stop()

    df = pd.read_csv(nfl_csv)
    required = ["team", "off_ppg", "def_ppg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in NFL CSV: {missing}")
        st.stop()

    df = ensure_numeric(df, ["off_ppg","def_ppg"])
    teams = sorted(df["team"].dropna().astype(str).unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        team_home = st.selectbox("Home Team", teams, index=0)
    with c2:
        team_away = st.selectbox("Away Team", teams, index=min(1, len(teams)-1))

    sims = st.slider("Number of simulations", 2000, 100000, 20000, step=2000)
    home_edge = st.slider("Home edge (points added to home λ)", 0.0, 3.0, 0.5, 0.1)

    th = df.loc[df["team"] == team_home].iloc[0]
    ta = df.loc[df["team"] == team_away].iloc[0]

    # naive expected scoring rates blending offense vs opponent defense
    lam_home = (float(th["off_ppg"]) + float(ta["def_ppg"])) / 2.0
    lam_away = (float(ta["off_ppg"]) + float(th["def_ppg"])) / 2.0

    home_scores, away_scores = poisson_match_sim(lam_home, lam_away, sims=sims, home_edge=home_edge)

    total = home_scores + away_scores
    margin = home_scores - away_scores

    st.subheader("Results")
    st.metric("Mean Home Points", f"{np.mean(home_scores):.2f}")
    st.metric("Mean Away Points", f"{np.mean(away_scores):.2f}")
    st.metric("Mean Total", f"{np.mean(total):.2f}")
    st.metric("Home Win %", f"{np.mean(margin>0)*100:.1f}%")

    # quick total/spread calculator
    colA, colB = st.columns(2)
    with colA:
        user_total = st.number_input("Total line", 20.0, 100.0, value=float(np.mean(total).round(1)))
        p_over = float(1 - norm.cdf(user_total, loc=np.mean(total), scale=max(np.std(total), 1.5)))
        st.write(f"**P(Total Over)** ≈ {p_over*100:.1f}%")
    with colB:
        spread = st.number_input("Home spread (negative = favorite)", -30.0, 30.0, value=float(-np.mean(margin).round(1)))
        p_cover = float(1 - norm.cdf(spread, loc=np.mean(margin), scale=max(np.std(margin), 1.5)))
        st.write(f"**P(Home covers)** ≈ {p_cover*100:.1f}%")

# --------------------------------------------------
# MLB SIM PAGE
# --------------------------------------------------
elif page == "MLB Sim":
    st.header("⚾️ MLB Team Simulator (CSV)")
    st.caption(
        "Upload a team stats CSV with columns at minimum: **team, runs_for_pg, runs_against_pg**. "
        "We estimate team run rates and simulate a game (Poisson)."
    )

    mlb_csv = st.file_uploader("Upload MLB team stats CSV", type=["csv"])
    if mlb_csv is None:
        st.info("No file uploaded yet. Example columns:\n\nteam,runs_for_pg,runs_against_pg")
        st.stop()

    df = pd.read_csv(mlb_csv)
    required = ["team", "runs_for_pg", "runs_against_pg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in MLB CSV: {missing}")
        st.stop()

    df = ensure_numeric(df, ["runs_for_pg","runs_against_pg"])
    teams = sorted(df["team"].dropna().astype(str).unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        team_home = st.selectbox("Home Team", teams, index=0)
    with c2:
        team_away = st.selectbox("Away Team", teams, index=min(1, len(teams)-1))

    sims = st.slider("Number of simulations", 2000, 100000, 20000, step=2000)
    home_edge = st.slider("Home edge (runs added to home λ)", 0.0, 0.8, 0.15, 0.05)

    th = df.loc[df["team"] == team_home].iloc[0]
    ta = df.loc[df["team"] == team_away].iloc[0]

    lam_home = (float(th["runs_for_pg"]) + float(ta["runs_against_pg"])) / 2.0
    lam_away = (float(ta["runs_for_pg"]) + float(th["runs_against_pg"])) / 2.0

    home_runs, away_runs = poisson_match_sim(lam_home, lam_away, sims=sims, home_edge=home_edge)

    total = home_runs + away_runs
    margin = home_runs - away_runs

    st.subheader("Results")
    st.metric("Mean Home Runs", f"{np.mean(home_runs):.2f}")
    st.metric("Mean Away Runs", f"{np.mean(away_runs):.2f}")
    st.metric("Mean Total", f"{np.mean(total):.2f}")
    st.metric("Home Win %", f"{np.mean(margin>0)*100:.1f}%")

    colA, colB = st.columns(2)
    with colA:
        user_total = st.number_input("Total line (runs)", 3.0, 16.0, value=float(np.mean(total).round(1)))
        p_over = float(1 - norm.cdf(user_total, loc=np.mean(total), scale=max(np.std(total), 0.9)))
        st.write(f"**P(Total Over)** ≈ {p_over*100:.1f}%")
    with colB:
        spread = st.number_input("Home runline (negative = favorite)", -5.0, 5.0, value=float(-np.mean(margin).round(1)))
        p_cover = float(1 - norm.cdf(spread, loc=np.mean(margin), scale=max(np.std(margin), 0.9)))
        st.write(f"**P(Home covers)** ≈ {p_cover*100:.1f}%")

# --------------------------------------------------
# PLAYER PROPS PAGE (CSV)
# --------------------------------------------------
else:
    st.header("📄 Player Props (CSV) — Over/Under Simulator")
    st.caption(
        "Upload a **Props CSV** with at least: **player, stat, line**. "
        "Optionally upload a **Player Averages CSV** with per-game columns (examples below). "
        "We fuzzy-match names, map stats, and estimate P(Over/Under) with a conservative normal model."
    )

    st.markdown("**Props CSV required columns:** `player, stat, line`  \n"
                "Examples for `stat`: `Pass Yards`, `Rush Yards`, `Receiving Yards`, `Receptions`, "
                "`Hits`, `Home Runs`, `RBIs`, `Pitcher Strikeouts`, etc.")

    props_file = st.file_uploader("Upload Props CSV (required)", type=["csv"])
    stats_file = st.file_uploader("Upload Player Averages CSV (optional)", type=["csv"])

    if props_file is None:
        st.stop()

    props = pd.read_csv(props_file)
    if not set(["player","stat","line"]).issubset(props.columns):
        st.error("Props CSV must include columns: player, stat, line")
        st.stop()
    props["line"] = pd.to_numeric(props["line"], errors="coerce")
    props = props.dropna(subset=["player","stat","line"]).reset_index(drop=True)

    # Mapping: common market names -> expected player-averages columns
    # You can extend/change these to match your averages CSV.
    MARKET_MAP = {
        # NFL-like
        "pass yards": ["pass_yards","passing_yards","passyds","py"],
        "rushing yards": ["rush_yards","rushing_yards","ry"],
        "receiving yards": ["rec_yards","receiving_yards","recy"],
        "receptions": ["receptions","recs","rec"],
        "rush+rec yds": ["rush_yards","rec_yards"],
        "pass+rush+rec yds": ["pass_yards","rush_yards","rec_yards"],
        # MLB-like
        "hits": ["hits","H"],
        "home runs": ["hr","home_runs"],
        "rbis": ["rbi","RBIs","RBIs_pg","rbi_pg"],
        "stolen bases": ["sb","stolen_bases"],
        "pitcher strikeouts": ["pitch_strikeouts","k","Ks","SO"],
        "total bases": ["total_bases","TB"],  # if you compute it in your averages
        # NBA-like (if you ever use)
        "points": ["points","pts"],
        "assists": ["assists","ast"],
        "rebounds": ["rebounds","reb"],
        "3-pt made": ["threes","fg3m"],
        "pts+reb+ast": ["points","rebounds","assists"],
        "pts+reb": ["points","rebounds"],
        "pts+ast": ["points","assists"],
        "reb+ast": ["rebounds","assists"],
    }

    def find_stat_columns_for_label(label: str, available_cols: List[str]) -> Optional[List[str]]:
        lbl = clean_name(label)
        best_key = None
        best_score = -1
        for k in MARKET_MAP.keys():
            sc = fuzz.token_set_ratio(lbl, k)
            if sc > best_score:
                best_key, best_score = k, sc
        if best_key is None: 
            return None

        # Filter to columns that actually exist in the uploaded averages CSV
        cand = MARKET_MAP[best_key]
        if isinstance(cand, list):
            # keep those present
            cols = [c for c in cand if c in available_cols]
            return cols if cols else None
        return [cand] if cand in available_cols else None

    if stats_file is None:
        st.warning("No player averages CSV uploaded — you’ll need to provide one to run the sim.")
        st.stop()

    stats = pd.read_csv(stats_file)
    if "player" not in stats.columns:
        st.error("Player averages CSV must include a 'player' column.")
        st.stop()

    # ensure numeric for all non-player columns
    for c in stats.columns:
        if c == "player":
            continue
        stats[c] = pd.to_numeric(stats[c], errors="coerce")

    players = stats["player"].dropna().astype(str).unique().tolist()

    rows=[]
    prog = st.progress(0.0)
    for i, r in props.iterrows():
        prog.progress((i+1)/len(props))
        p_name = str(r["player"])
        stat_label = str(r["stat"])
        line_val = r["line"]
        if pd.isna(line_val): 
            continue

        # fuzzy name match
        match = best_name_match(p_name, players, score_cut=82)
        if match is None:
            continue

        # stat mapping -> actual column names present in stats file
        cols = find_stat_columns_for_label(stat_label, list(stats.columns))
        if cols is None or not cols:
            continue

        row_stats = stats.loc[stats["player"] == match].iloc[0]
        vals = [row_stats.get(c) for c in cols if c in row_stats.index]
        vals = [v for v in vals if pd.notna(v)]
        if not vals: 
            continue
        avg_val = float(np.sum(vals))

        try:
            line_val = float(line_val)
        except:
            continue

        p_over, p_under, sd_used = simulate_prob(avg_val, line_val)
        rows.append({
            "player": match,
            "prop_player": p_name,
            "market": stat_label,
            "line": round(line_val, 3),
            "avg": round(avg_val, 3),
            "model_sd": sd_used,
            "P(Over)": p_over,
            "P(Under)": p_under,
        })

    prog.empty()
    results = pd.DataFrame(rows).sort_values(["P(Over)","P(Under)"], ascending=[False, True])

    if results.empty:
        st.warning("No matched rows. Check player names and stat labels vs your averages CSV.")
        st.stop()

    st.subheader("Simulated props (conservative normal model)")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="props_sim_results.csv",
        mime="text/csv",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 Overs**")
        st.dataframe(results.nlargest(10, "P(Over)")[["player","market","line","avg","P(Over)","P(Under)"]], use_container_width=True)
    with c2:
        st.markdown("**Top 10 Unders**")
        st.dataframe(results.nlargest(10, "P(Under)")[["player","market","line","avg","P(Over)","P(Under)"]], use_container_width=True)
