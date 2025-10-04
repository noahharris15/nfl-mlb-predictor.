# app.py — NFL + MLB Predictor + Player Props (Odds API lines + your CSV stats)
# - NFL page: team PF/PA Poisson (kept same)
# - MLB page: team RS/RA Poisson (kept same)
# - Player Props page: fetch lines from Odds API, use your CSV per-game averages, simulate vs the exact line

from __future__ import annotations

import io
import math
import time
import json
import random
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz
from scipy.stats import norm

# ----------------------------- Page config ------------------------------------
st.set_page_config(page_title="NFL + MLB + Player Props (Odds API)", layout="wide")
st.title("🏈⚾ NFL + MLB + 🎯 Player Props (Odds API lines + your stats)")

# ----------------------------- Constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6
EPS = 1e-9

# Odds API key: secrets override, else default (you can edit below)
ODDS_API_KEY_DEFAULT = "9ede18ca5b55fa2afc180a2b375367e2"
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", ODDS_API_KEY_DEFAULT)

# ---------------------------------------------------------------------
#                       COMMON SMALL HELPERS
# ---------------------------------------------------------------------
def conservative_sd(avg: float, floor: float = 0.75, frac: float = 0.30) -> float:
    """Conservative SD so we don't get 0%/100% artifacts."""
    try:
        avg = float(avg)
    except Exception:
        return 1.25
    if avg <= 0:
        return 1.0
    return max(frac * avg, floor, 0.5)

def simulate_over_under_prob(avg: float, line: float) -> Tuple[float,float,float]:
    """Normal-model: P(over), P(under), sd used."""
    sd = conservative_sd(avg)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    # keep away from exactly 0/1 for readability
    p_over = float(np.clip(p_over, 0.005, 0.995))
    p_under = float(np.clip(p_under, 0.005, 0.995))
    return round(100*p_over,2), round(100*p_under,2), round(sd,3)

def best_name_match(name: str, candidates: List[str], score_cut: int = 86) -> Optional[str]:
    """Fuzzy match a player name to your CSV."""
    name_c = (name or "").replace(".", "").replace("-", " ").strip().lower()
    best, best_score = None, -1
    for c in candidates:
        s = fuzz.token_sort_ratio(name_c, (c or "").replace(".", "").replace("-", " ").strip().lower())
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= score_cut else None

# ---------------------------------------------------------------------
#                   NFL TEAM MODEL (unchanged behavior)
# ---------------------------------------------------------------------
import nfl_data_py as nfl

@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    date_col: Optional[str] = None
    for c in ("gameday", "game_date", "start_time"):
        if c in sched.columns:
            date_col = c
            break

    played = sched.dropna(subset=["home_score", "away_score"])
    home = played.rename(columns={
        "home_team": "team", "away_team": "opp", "home_score": "pf", "away_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    away = played.rename(columns={
        "away_team": "team", "home_team": "opp", "away_score": "pf", "home_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 45.0 / 2.0
        teams32 = [
            "Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills",
            "Carolina Panthers","Chicago Bears","Cincinnati Bengals","Cleveland Browns",
            "Dallas Cowboys","Denver Broncos","Detroit Lions","Green Bay Packers",
            "Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
            "Las Vegas Raiders","Los Angeles Chargers","Los Angeles Rams","Miami Dolphins",
            "Minnesota Vikings","New England Patriots","New Orleans Saints","New York Giants",
            "New York Jets","Philadelphia Eagles","Pittsburgh Steelers","San Francisco 49ers",
            "Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders",
        ]
        rates = pd.DataFrame({"team": teams32, "PF_pg": per, "PA_pg": per})
    else:
        team = long.groupby("team", as_index=False).agg(
            games=("pf","size"), PF=("pf","sum"), PA=("pa","sum")
        )
        rates = pd.DataFrame({
            "team": team["team"],
            "PF_pg": team["PF"] / team["games"],
            "PA_pg": team["PA"] / team["games"],
        })
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"]/4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink) * rates["PF_pg"] + shrink * prior
        rates["PA_pg"] = (1 - shrink) * rates["PA_pg"] + shrink * prior

    # upcoming list
    if {"home_team","away_team"}.issubset(set(sched.columns)):
        filt = sched["home_score"].isna() & sched["away_score"].isna()
        upcoming = sched.loc[filt, ["home_team","away_team"]].copy()
        upcoming["date"] = sched.loc[filt, date_col].astype(str) if date_col else ""
    else:
        upcoming = pd.DataFrame(columns=["home_team","away_team","date"])

    for c in ["home_team","away_team"]:
        if c in upcoming.columns:
            upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+", " ", regex=True)

    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"])/2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"])/2.0)
    return mu_home, mu_away

def poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

# ---------------------------------------------------------------------
#                   MLB TEAM MODEL (unchanged behavior)
# ---------------------------------------------------------------------
from pybaseball import schedule_and_record

MLB_TEAMS_2025: Dict[str, str] = {
    "ARI": "Arizona Diamondbacks","ATL": "Atlanta Braves","BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox","CHC": "Chicago Cubs","CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds","CLE": "Cleveland Guardians","COL": "Colorado Rockies",
    "DET": "Detroit Tigers","HOU": "Houston Astros","KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels","LAD": "Los Angeles Dodgers","MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers","MIN": "Minnesota Twins","NYM": "New York Mets",
    "NYY": "New York Yankees","OAK": "Oakland Athletics","PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates","SDP": "San Diego Padres","SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants","STL": "St. Louis Cardinals","TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers","TOR": "Toronto Blue Jays","WSN": "Washington Nationals",
}

@st.cache_data(show_spinner=False)
def mlb_team_rates_2025() -> pd.DataFrame:
    rows = []
    for br, name in MLB_TEAMS_2025.items():
        try:
            sar = schedule_and_record(2025, br)
            sar = sar[pd.to_numeric(sar.get("R"), errors="coerce").notna()]
            sar = sar[pd.to_numeric(sar.get("RA"), errors="coerce").notna()]
            if sar.empty:
                RS_pg = RA_pg = 4.5
            else:
                sar["R"] = sar["R"].astype(float)
                sar["RA"] = sar["RA"].astype(float)
                games = int(len(sar))
                RS_pg = float(sar["R"].sum() / games)
                RA_pg = float(sar["RA"].sum() / games)
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean())
        league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9 * df["RS_pg"] + 0.1 * league_rs
        df["RA_pg"] = 0.9 * df["RA_pg"] + 0.1 * league_ra
    return df

# ---------------------------------------------------------------------
#                   ODDS API — player lines fetch
# ---------------------------------------------------------------------
SPORT_KEY = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    # If you later want Soccer props, add proper sport key & markets here
}

# Map our user-friendly market names -> Odds API player prop market keys (per league)
ODDS_MARKETS = {
    "NFL": {
        "Passing Yards": "player_pass_yards",
        "Rush Yards": "player_rush_yards",
        "Receiving Yards": "player_receiving_yards",
        "Receptions": "player_receptions",
        "Pass+Rush Yds": "player_pass_plus_rush_yards",
        "Rush+Rec Yds": "player_rush_plus_receiving_yards",
    },
    "NBA": {
        "Points": "player_points",
        "Rebounds": "player_rebounds",
        "Assists": "player_assists",
        "3-PT Made": "player_threes",
        "Pts+Reb+Ast": "player_points_plus_rebounds_plus_assists",
        "Pts+Reb": "player_points_plus_rebounds",
        "Pts+Ast": "player_points_plus_assists",
        "Reb+Ast": "player_rebounds_plus_assists",
    },
    "MLB": {
        "Hits": "player_hits",
        "Home Runs": "player_home_runs",
        "RBIs": "player_rbis",
        "Stolen Bases": "player_stolen_bases",
        "Pitcher Strikeouts": "player_strikeouts",
    },
}

BOOKMAKERS = "draftkings,fanduel,betmgm,caesars,pointsbet"  # pick the books you want
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

def fetch_player_prop_lines(league: str, wanted_markets: List[str]) -> pd.DataFrame:
    """Return DataFrame with columns: player, market, line (average of available), team, opponent, game_time."""
    sport = SPORT_KEY.get(league)
    if not sport:
        return pd.DataFrame(columns=["player","market","line","team","opponent","game_time"])

    # translate wanted display markets -> odds markets that API expects
    api_markets = list({ODDS_MARKETS.get(league, {}).get(m) for m in wanted_markets if ODDS_MARKETS.get(league, {}).get(m)})
    if not api_markets:
        return pd.DataFrame(columns=["player","market","line","team","opponent","game_time"])

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "markets": ",".join(api_markets),
        "bookmakers": BOOKMAKERS,
    }

    try:
        r = requests.get(ODDS_BASE.format(sport=sport), params=params, timeout=20)
        r.raise_for_status()
        data = r.json()  # list of games
    except Exception as e:
        st.error(f"Odds API error: {e}")
        return pd.DataFrame(columns=["player","market","line","team","opponent","game_time"])

    rows = []
    for game in data:
        commence = game.get("commence_time")
        home = (game.get("home_team") or "").strip()
        away = (game.get("away_team") or "").strip()
        for book in (game.get("bookmakers") or []):
            for market in (book.get("markets") or []):
                mkey = market.get("key")  # e.g., player_pass_yards
                outcomes = market.get("outcomes") or []
                for o in outcomes:
                    player_name = o.get("description") or o.get("name") or ""  # Odds API uses 'description' for players
                    line = o.get("point")
                    if player_name and line is not None:
                        # map back to our display market
                        disp_market = next((k for k,v in ODDS_MARKETS.get(league,{}).items() if v == mkey), mkey)
                        rows.append({
                            "player": player_name,
                            "market": disp_market,
                            "line": float(line),
                            "team": home if (o.get("team") == home) else (away if (o.get("team") == away) else None),
                            "opponent": away if (o.get("team") == home) else (home if (o.get("team") == away) else None),
                            "game_time": commence
                        })

    if not rows:
        return pd.DataFrame(columns=["player","market","line","team","opponent","game_time"])

    df = pd.DataFrame(rows)
    # group duplicates across books -> average line
    agg = df.groupby(["player","market","game_time","team","opponent"], as_index=False)["line"].mean()
    return agg.reset_index(drop=True)

# ---------------------------------------------------------------------
#           PLAYER CSV READING + COLUMN GUESSING (simple & robust)
# ---------------------------------------------------------------------
def read_any_table(upload) -> pd.DataFrame:
    if upload is None: return pd.DataFrame()
    name = (upload.name or "").lower()
    if name.endswith(".csv"):  return pd.read_csv(upload)
    if name.endswith(".xlsx") or name.endswith(".xls"): return pd.read_excel(upload)
    # default try CSV
    return pd.read_csv(upload)

def guess_player_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in ("player","name","player_name","player full name","fullname"):
            return c
    return df.columns[0]

# A minimal mapping of common stat column names for each league
CSV_STAT_ALIASES = {
    "NFL": {
        "Passing Yards": ["pass_yards","passing_yards","py/g","pass yds","pass yards","pass_yds"],
        "Rush Yards": ["rush_yards","rushing_yards","ry/g","rush yds","rush yards","rush_yds"],
        "Receiving Yards": ["rec_yards","receiving_yards","rec yds","rec yards","receiving yds"],
        "Receptions": ["receptions","rec","rec/g","receptions/g"],
        "Pass+Rush Yds": ["pass_plus_rush_yards","pass+rush yds","pass rush yds"],
        "Rush+Rec Yds": ["rush_plus_rec_yards","rush+rec yds","rush rec yds"],
    },
    "NBA": {
        "Points": ["points","pts","pts/g"],
        "Rebounds": ["rebounds","rebs","reb","reb/g"],
        "Assists": ["assists","ast","ast/g"],
        "3-PT Made": ["threes","3pm","3p made","fg3m"],
        "Pts+Reb+Ast": ["pra","pts+reb+ast"],
        "Pts+Reb": ["pr","pts+reb"],
        "Pts+Ast": ["pa","pts+ast"],
        "Reb+Ast": ["ra","reb+ast"],
    },
    "MLB": {
        "Hits": ["hits","h"],
        "Home Runs": ["home_runs","hr"],
        "RBIs": ["rbis","rbi"],
        "Stolen Bases": ["sb","stolen_bases"],
        "Pitcher Strikeouts": ["pitch_strikeouts","k","so","strikeouts"],
    }
}

def pick_stat_from_csv(row_stats: pd.Series, league: str, market: str) -> Optional[float]:
    aliases = CSV_STAT_ALIASES.get(league, {}).get(market, [])
    for a in aliases:
        if a in row_stats.index and pd.notna(row_stats[a]):
            return float(row_stats[a])
    # final fallback: try exact (if user already named it exactly)
    if market in row_stats.index and pd.notna(row_stats[market]):
        return float(row_stats[market])
    return None

# ---------------------------------------------------------------------
#                                   UI
# ---------------------------------------------------------------------
page = st.radio("Choose a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 regular season (team scoring model)")
    rates, upcoming = nfl_team_rates_2025()

    if not upcoming.empty and {"home_team","away_team"}.issubset(set(upcoming.columns)):
        labels = [f"{r['home_team']} vs {r['away_team']} — {r.get('date','')}" for _, r in upcoming.iterrows()]
        sel = st.selectbox("Upcoming game", labels) if labels else None
        if sel:
            try:
                teams_part = sel.split(" — ")[0]
                home, away = [t.strip() for t in teams_part.split(" vs ")]
            except Exception:
                home = away = None
        else:
            home = away = None
    else:
        st.info("No upcoming list available — pick any two teams:")
        home = st.selectbox("Home team", rates["team"].tolist())
        away = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != home])

    if home and away:
        try:
            mu_h, mu_a = nfl_matchup_mu(rates, home, away)
            pH, pA, mH, mA = poisson_game(mu_h, mu_a)
            st.markdown(
                f"**{home}** vs **{away}** — "
                f"Expected points: {mH:.1f}–{mA:.1f} · "
                f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
            )
        except Exception as e:
            st.error(str(e))

# -------------------------- MLB page --------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 regular season (team scoring model)")
    try:
        rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't load MLB team rates: {e}")
        st.stop()

    t1 = st.selectbox("Home team", rates["team"].tolist())
    t2 = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != t1])

    H = rates.loc[rates["team"] == t1].iloc[0]
    A = rates.loc[rates["team"] == t2].iloc[0]
    mu_home = (H["RS_pg"] + A["RA_pg"]) / 2.0
    mu_away = (A["RS_pg"] + H["RA_pg"]) / 2.0
    pH, pA, mH, mA = poisson_game(mu_home, mu_away)
    st.markdown(
        f"**{t1}** vs **{t2}** — "
        f"Expected runs: {mH:.1f}–{mA:.1f} · "
        f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**"
    )

# -------------------------- Player Props page --------------------------
else:
    st.subheader("🎯 Player Props — Odds API lines + your CSV per-game stats")
    league = st.selectbox("League for props", ["NFL","NBA","MLB"], index=0, help="We can add more later.")
    st.caption(f"Using Odds API key: {'secrets' if 'ODDS_API_KEY' in st.secrets else 'default in code'}")

    # Let you constrain markets you care about (defaults per league)
    default_markets = list(ODDS_MARKETS.get(league, {}).keys())
    markets = st.multiselect("Markets to pull", default_markets, default=default_markets)

    # 1) Fetch live lines from Odds API
    with st.spinner("Fetching live player lines from Odds API…"):
        lines_df = fetch_player_prop_lines(league, markets)

    if lines_df.empty:
        st.error("No player lines received. Try different markets or check API key/quota.")
        st.stop()

    st.success(f"Got {len(lines_df)} player lines.")
    if st.checkbox("Show raw lines", value=False):
        st.dataframe(lines_df, use_container_width=True)

    # 2) Upload your player averages CSV for this league
    up = st.file_uploader(f"Upload your {league} player averages (CSV / Excel)", type=["csv","xlsx","xls"])
    if up is None:
        st.info("Upload a CSV with columns: player (name) + stat columns (e.g., pass_yards, receptions, points, etc.)")
        st.stop()

    df_stats = read_any_table(up).copy()
    if df_stats.empty:
        st.error("Stats CSV appears empty.")
        st.stop()

    player_col = guess_player_col(df_stats)
    df_stats[player_col] = df_stats[player_col].astype(str)
    csv_players = df_stats[player_col].dropna().astype(str).unique().tolist()

    # 3) Match each line to your CSV and simulate
    rows = []
    prog = st.progress(0)
    for i, r in lines_df.iterrows():
        prog.progress((i+1)/len(lines_df))
        pp_player = r["player"]
        market = r["market"]
        line = r["line"]

        match = best_name_match(pp_player, csv_players, score_cut=86)
        if not match:
            continue

        row_stats = df_stats.loc[df_stats[player_col] == match].iloc[0]
        avg_val = pick_stat_from_csv(row_stats, league, market)

        # Support simple combos if the CSV already provides them
        if avg_val is None:
            # try a couple common combos if user provided separate columns
            if league == "NFL" and market == "Pass+Rush Yds":
                a = pick_stat_from_csv(row_stats, "NFL", "Passing Yards")
                b = pick_stat_from_csv(row_stats, "NFL", "Rush Yards")
                if a is not None and b is not None:
                    avg_val = a + b
            if league == "NFL" and market == "Rush+Rec Yds":
                a = pick_stat_from_csv(row_stats, "NFL", "Rush Yards")
                b = pick_stat_from_csv(row_stats, "NFL", "Receiving Yards")
                if a is not None and b is not None:
                    avg_val = a + b
            if league == "NBA" and market == "Pts+Reb+Ast":
                a = pick_stat_from_csv(row_stats, "NBA", "Points")
                b = pick_stat_from_csv(row_stats, "NBA", "Rebounds")
                c = pick_stat_from_csv(row_stats, "NBA", "Assists")
                if None not in (a,b,c):
                    avg_val = a + b + c
            if league == "NBA" and market == "Pts+Reb":
                a = pick_stat_from_csv(row_stats, "NBA", "Points")
                b = pick_stat_from_csv(row_stats, "NBA", "Rebounds")
                if None not in (a,b):
                    avg_val = a + b
            if league == "NBA" and market == "Pts+Ast":
                a = pick_stat_from_csv(row_stats, "NBA", "Points")
                b = pick_stat_from_csv(row_stats, "NBA", "Assists")
                if None not in (a,b):
                    avg_val = a + b
            if league == "NBA" and market == "Reb+Ast":
                a = pick_stat_from_csv(row_stats, "NBA", "Rebounds")
                b = pick_stat_from_csv(row_stats, "NBA", "Assists")
                if None not in (a,b):
                    avg_val = a + b

        if avg_val is None:
            continue

        try:
            line_val = float(line)
        except Exception:
            continue

        p_over, p_under, used_sd = simulate_over_under_prob(float(avg_val), line_val)
        rows.append({
            "league": league,
            "player": match,
            "market": market,
            "line": round(line_val, 2),
            "avg": round(float(avg_val), 2),
            "P(Over)%": p_over,
            "P(Under)%": p_under,
            "sd_used": used_sd,
            "opp": r.get("opponent"),
            "game_time": r.get("game_time"),
        })

    prog.empty()
    results = pd.DataFrame(rows)

    if results.empty:
        st.warning("Nothing matched. Check your CSV column names vs the markets you selected.")
        st.stop()

    # sort with most positive edge (P(Over) far from 50) on top
    results["edge_abs"] = (results["P(Over)%"] - 50).abs()
    results = results.sort_values(["edge_abs","P(Over)%"], ascending=[False, False]).drop(columns=["edge_abs"])

    st.subheader("Simulated results (using Odds API line + your CSV averages)")
    st.caption("SD is conservative to avoid unrealistic 0%/100% outputs.")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name=f"props_{league.lower()}_sim.csv",
        mime="text/csv",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 10 — Overs")
        st.dataframe(results.nlargest(10, "P(Over)%")[["player","market","line","avg","P(Over)%","P(Under)%","opp","game_time"]], use_container_width=True)
    with col2:
        st.markdown("#### Top 10 — Unders")
        st.dataframe(results.nlargest(10, "P(Under)%")[["player","market","line","avg","P(Over)%","P(Under)%","opp","game_time"]], use_container_width=True)
