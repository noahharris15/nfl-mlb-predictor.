# streamlit_app.py
# One app, 3 pages (NFL / MLB / Soccer). Pull player lines automatically and simulate O/U.

import os
import time
import json
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz

# ==============================
# ---- CONFIG / API KEYS -------
# ==============================
# Your Odds API key (kept because you asked to redo with your new key)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "7401399bd14e8778312da073b621094f")

# >>> To actually get player-prop lines, add one of these optional keys <<<
PROPODDS_API_KEY = os.getenv("PROPODDS_API_KEY", "")       # https://propodds.io (recommended free tier)
SPORTSDATA_API_KEY = os.getenv("SPORTSDATA_API_KEY", "")   # https://sportsdata.io (trial works)

# Pick which source to try first for PLAYER PROPS.
# choices: "propodds", "sportsdata", "oddsapi"
PRIMARY_PROP_SOURCE = os.getenv("PRIMARY_PROP_SOURCE", "propodds")

# ==============================
# ----------- UI ---------------
# ==============================
st.set_page_config(page_title="All-Sports Props Simulator", layout="wide")
st.title("📊 All-Sports Player Props — Odds + Real Stats + Simulator")

page = st.sidebar.radio("Pages", ["NFL", "MLB", "Soccer"])
season_default = {"NFL": 2025, "MLB": 2024, "Soccer": 2025}[page]
season = st.sidebar.number_input("Season", 2018, 2026, value=season_default, step=1)

st.caption(
    "We **auto-fetch player lines** (tries your Odds API, then falls back to a props provider), "
    "load **real per-game player averages**, fuzzy-match names & stat types, and estimate **P(Over/Under)** "
    "using a conservative normal model."
)

# Optional: pick books to average across (only used when the provider returns multi-book data)
books_input = st.sidebar.text_input(
    "Bookmakers (comma-separated; leave empty to average all returned)",
    value="draftkings,fanduel,betmgm,caesars"
).strip()
BOOKS = [b.strip().lower() for b in books_input.split(",") if b.strip()] if books_input else []

# ==============================
# ------- Utility / Model -------
# ==============================
def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").replace("'", "").strip().lower()

def best_name_match(name: str, candidates: List[str], score_cut=82) -> Optional[str]:
    nm = clean_name(name)
    best = None
    best_sc = -1
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

def value_from_mapping(row: pd.Series, mapping):
    if isinstance(mapping, list):
        vals = [row.get(c) for c in mapping if c in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan
    return row.get(mapping)

# ==============================
# ---- Real stats providers -----
# ==============================
@st.cache_data(ttl=3600)
def load_nfl_stats(season: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    import nfl_data_py as nfl
    st.info("Loading NFL season data…")
    df = nfl.import_seasonal_data([season])
    g = df["games"].replace(0, np.nan) if "games" in df.columns else np.nan
    out = pd.DataFrame({
        "player": df.get("player_display_name"),
        "team": df.get("recent_team"),
        "pass_yards": df.get("passing_yards") / g if "games" in df else df.get("passing_yards"),
        "rush_yards": df.get("rushing_yards") / g if "games" in df else df.get("rushing_yards"),
        "rec_yards":  df.get("receiving_yards") / g if "games" in df else df.get("receiving_yards"),
        "receptions": df.get("receptions") / g if "games" in df else df.get("receptions"),
        "pass_tds":   df.get("passing_tds") / g if "games" in df else df.get("passing_tds"),
        "rush_tds":   df.get("rushing_tds") / g if "games" in df else df.get("rushing_tds"),
        "rec_tds":    df.get("receiving_tds") / g if "games" in df else df.get("receiving_tds"),
        "ints":       df.get("interceptions") / g if "games" in df else df.get("interceptions"),
    }).dropna(subset=["player"])
    mapping = {
        # display → column(s)
        "Pass Yards": "pass_yards", "Passing Yards": "pass_yards",
        "Rush Yards": "rush_yards", "Rushing Yards": "rush_yards",
        "Receiving Yards": "rec_yards",
        "Receptions": "receptions",
        "Pass+Rush Yards": ["pass_yards", "rush_yards"],
        "Rush+Rec Yards": ["rush_yards", "rec_yards"],
        "Pass+Rush+Rec Yards": ["pass_yards", "rush_yards", "rec_yards"],
    }
    return out, mapping

@st.cache_data(ttl=3600)
def load_mlb_stats(season: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    st.info("Loading MLB season stats (pybaseball)…")
    from pybaseball import batting_stats, pitching_stats
    bat = batting_stats(season); pit = pitching_stats(season)
    bat_out = pd.DataFrame({
        "player": bat.get("Name"),
        "hits": bat.get("H"), "hr": bat.get("HR"), "rbi": bat.get("RBI"),
        "sb": bat.get("SB"), "bb": bat.get("BB"), "so": bat.get("SO"),
        "pa": bat.get("PA"),
    }).dropna(subset=["player"])
    bat_out["pa"] = bat_out["pa"].replace(0, np.nan)
    for c in ["hits","hr","rbi","sb"]:
        bat_out[c] = bat_out[c] / bat_out["pa"] * 4.2
    pit_out = pd.DataFrame({
        "player": pit.get("Name"),
        "pitch_strikeouts": pit.get("SO"),
    }).dropna(subset=["player"])
    out = pd.merge(bat_out.drop(columns=["pa"]), pit_out, on="player", how="outer")
    mapping = {
        "Hits": "hits", "Home Runs": "hr", "RBIs": "rbi", "Stolen Bases": "sb",
        "Pitcher Strikeouts": "pitch_strikeouts",
        "Hitter Fantasy (proxy)": ["hits","hr","rbi","sb"],
    }
    return out, mapping

@st.cache_data(ttl=3600)
def load_soccer_stats(season: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    st.info("Loading Soccer player stats (FBref via soccerdata)…")
    import soccerdata as sd
    fb_season = season if season <= 2024 else 2024
    leagues = ["ENG-Premier League","ESP-La Liga","ITA-Serie A","GER-Bundesliga","FRA-Ligue 1"]
    frames = []
    for lg in leagues:
        try:
            fb = sd.FBref(leagues=lg, seasons=fb_season)
            df = fb.read_player_season_stats(stat_type="standard").reset_index()
            name_col = "player" if "player" in df.columns else "Player"
            team_col = "team" if "team" in df.columns else ("Squad" if "Squad" in df.columns else None)
            frames.append(pd.DataFrame({
                "player": df[name_col],
                "team": df[team_col] if team_col and team_col in df.columns else None,
                "goals": df["Gls"] if "Gls" in df.columns else np.nan,
                "shots": df["Sh"] if "Sh" in df.columns else np.nan,
                "sog":   df["SoT"] if "SoT" in df.columns else np.nan,
                "assists": df["Ast"] if "Ast" in df.columns else np.nan,
                "key_passes": df["KP"] if "KP" in df.columns else np.nan,
            }))
        except Exception:
            continue
    if not frames:
        raise ValueError("No soccer frames returned.")
    out = pd.concat(frames, ignore_index=True)
    mapping = {
        "Goals": "goals", "Shots": "shots", "Shots On Goal": "sog",
        "Assists": "assists", "Shots+SOG": ["shots","sog"], "G+A": ["goals","assists"],
    }
    return out, mapping

def load_stats_and_map(which: str, season: int):
    if which == "NFL":   return load_nfl_stats(season)
    if which == "MLB":   return load_mlb_stats(season)
    if which == "Soccer":return load_soccer_stats(season)
    raise ValueError("Unknown league page")

# ==============================
# --- PLAYER LINES PROVIDERS ---
# ==============================
def _avg_from_books(offers: List[dict], books: List[str]) -> Optional[float]:
    vals = []
    for o in offers:
        bk = (o.get("bookmaker") or o.get("sportsbook") or "").lower()
        if books and bk not in books: 
            continue
        line = o.get("line") or o.get("odds") or o.get("price") or o.get("value")
        try:
            vals.append(float(line))
        except Exception:
            continue
    if not vals:
        return None
    return float(np.mean(vals))

def fetch_props_propodds(league: str) -> pd.DataFrame:
    if not PROPODDS_API_KEY:
        raise RuntimeError("PropOdds key not set.")
    sport = {"NFL":"nfl","MLB":"mlb","Soccer":"soccer"}[league]
    url = f"https://api.propodds.io/api/v1/props?sport={sport}"
    r = requests.get(url, headers={"x-api-key": PROPODDS_API_KEY}, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for it in data.get("data", []):
        player = it.get("playerName") or it.get("player")
        market = it.get("market") or it.get("stat")
        offers = it.get("offers") or it.get("books") or []
        line = _avg_from_books(offers, BOOKS)
        if player and market and line is not None:
            rows.append({"player": player, "market": market, "line": line})
    return pd.DataFrame(rows)

def fetch_props_sportsdata_nfl() -> pd.DataFrame:
    if not SPORTSDATA_API_KEY:
        raise RuntimeError("SportsData.io key not set.")
    # Example NFL player props by week — you can refine with week if you like.
    # We'll aggregate most recent week returned.
    url = f"https://api.sportsdata.io/v4/nfl/odds/json/PlayerPropsByWeek/2024/1"
    r = requests.get(url, headers={"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data:
        player = d.get("PlayerName")
        market = d.get("BetName") or d.get("StatType")
        line   = d.get("Value")
        if player and market and line is not None:
            rows.append({"player": player, "market": market, "line": float(line)})
    return pd.DataFrame(rows)

def fetch_props_oddsapi_game_only(league: str) -> pd.DataFrame:
    # The Odds API does not expose player props. We return empty but log why.
    raise RuntimeError("The Odds API /v4 odds endpoint does not support player prop markets (only game lines).")

def fetch_player_lines(league: str) -> Tuple[pd.DataFrame, str]:
    """
    Returns (board_df, source_used) with columns: player, market, line
    Tries PRIMARY_PROP_SOURCE first, then other fallbacks if available.
    """
    providers = []
    if PRIMARY_PROP_SOURCE == "propodds":
        providers = ["propodds", "sportsdata", "oddsapi"]
    elif PRIMARY_PROP_SOURCE == "sportsdata":
        providers = ["sportsdata", "propodds", "oddsapi"]
    else:
        providers = ["oddsapi", "propodds", "sportsdata"]

    last_err = None
    for p in providers:
        try:
            if p == "propodds":
                df = fetch_props_propodds(league)
                if not df.empty:
                    return df, "PropOdds"
            elif p == "sportsdata":
                if league != "NFL":
                    continue  # demo path only implemented for NFL example
                df = fetch_props_sportsdata_nfl()
                if not df.empty:
                    return df, "SportsData.io"
            elif p == "oddsapi":
                df = fetch_props_oddsapi_game_only(league)  # will raise
            else:
                continue
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(last_err or "No provider returned data.")

# ==============================
# ----- MARKET NORMALIZER ------
# ==============================
def normalize_market(league: str, mkt: str) -> Optional[str]:
    m = (mkt or "").strip().lower()
    if league == "NFL":
        mapping = {
            "pass yards":"Pass Yards", "passing yards":"Pass Yards", "player pass yards":"Pass Yards",
            "rush yards":"Rush Yards", "rushing yards":"Rush Yards",
            "receiving yards":"Receiving Yards", "rec yards":"Receiving Yards",
            "receptions":"Receptions",
            "pass+rush yards":"Pass+Rush Yards", "rush+rec yards":"Rush+Rec Yards",
            "pass+rush+rec yards":"Pass+Rush+Rec Yards",
        }
    elif league == "MLB":
        mapping = {
            "hits":"Hits", "home runs":"Home Runs", "rbi":"RBIs", "stolen bases":"Stolen Bases",
            "strikeouts":"Pitcher Strikeouts", "pitcher strikeouts":"Pitcher Strikeouts",
            "fantasy score":"Hitter Fantasy (proxy)",
        }
    else:  # Soccer
        mapping = {
            "goals":"Goals", "shots":"Shots", "shots on goal":"Shots On Goal",
            "assists":"Assists", "shots+sog":"Shots+SOG", "goals+assists":"G+A",
        }
    # best fuzzy key
    best_key = None; best_sc = -1
    for k, v in mapping.items():
        sc = fuzz.token_set_ratio(m, k)
        if sc > best_sc:
            best_key, best_sc = v, sc
    return best_key if best_sc >= 70 else None

# ==============================
# --------- PAGE BODY ----------
# ==============================
def run_page(league: str, season: int):
    st.header(league)

    # 1) Fetch props board
    props_df, src = None, None
    err_box = st.empty()
    with st.spinner("Fetching player prop lines…"):
        try:
            props_df, src = fetch_player_lines(league)
        except Exception as e:
            err_box.error(f"Could not load player props: {e}")
            st.stop()

    st.success(f"Loaded {len(props_df)} player lines from **{src}**.")
    st.dataframe(props_df.head(20), use_container_width=True)

    # 2) Load real stats
    try:
        stats_df, market_map = load_stats_and_map(league, season)
    except Exception as e:
        st.error(f"Failed to load {league} stats: {e}")
        st.stop()

    stats_df = stats_df.dropna(subset=["player"]).copy()
    players = list(stats_df["player"].unique())

    # 3) Simulate
    rows = []
    prog = st.progress(0.0)
    total = len(props_df)
    for i, r in props_df.iterrows():
        prog.progress((i+1)/max(total,1))
        player = r.get("player"); mkt = r.get("market"); line = r.get("line")
        if pd.isna(player) or pd.isna(mkt) or pd.isna(line): 
            continue

        std_mkt = normalize_market(league, str(mkt))
        if std_mkt is None:
            continue

        match = best_name_match(player, players, score_cut=82)
        if match is None:
            continue

        row_stats = stats_df.loc[stats_df["player"] == match].iloc[0]
        stat_cols = market_map.get(std_mkt)
        if stat_cols is None:
            continue

        avg_val = value_from_mapping(row_stats, stat_cols)
        if pd.isna(avg_val):
            continue

        try:
            line_val = float(line)
        except:
            continue

        p_over, p_under, sd_used = simulate_prob(avg_val, line_val)
        rows.append({
            "player": match,
            "prop_player": player,
            "market": std_mkt,
            "line": round(line_val, 3),
            "avg": round(float(avg_val), 3),
            "model_sd": sd_used,
            "P(Over)": p_over,
            "P(Under)": p_under,
            "source": src,
        })

    prog.empty()
    results = pd.DataFrame(rows).sort_values(["P(Over)","P(Under)"], ascending=[False, True])
    if results.empty:
        st.warning("No matches (market not supported or names didn’t match).")
        st.stop()

    st.subheader("Simulated edges")
    st.caption("Conservative normal model off real per-game averages (avoids 0/100 artifacts).")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name=f"{league.lower()}_{season}_props_sim.csv",
        mime="text/csv",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 Overs**")
        st.dataframe(results.nlargest(10, "P(Over)")[["player","market","line","avg","P(Over)","P(Under)"]], use_container_width=True)
    with c2:
        st.markdown("**Top 10 Unders**")
        st.dataframe(results.nlargest(10, "P(Under)")[["player","market","line","avg","P(Over)","P(Under)"]], use_container_width=True)

# Run the chosen page
if page == "NFL":
    run_page("NFL", season)
elif page == "MLB":
    run_page("MLB", season)
else:
    run_page("Soccer", season)
