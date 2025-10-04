# streamlit_app.py
# One script, 3 pages (NFL/MLB/Soccer). Pull player props from The Odds API ONLY.
# Then simulate P(Over/Under) using conservative normal model based on real per-game stats.

import os, time, json, math, random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz

# --------------------- CONFIG ---------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()  # <<< put your key in env/secrets
ODDS_BASE = "https://api.the-odds-api.com/v4"

# Sport keys for The Odds API
SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "MLB": "baseball_mlb",
}

SOCCER_COMP_KEYS = {
    "EPL (England)": "soccer_epl",
    "La Liga (Spain)": "soccer_spain_la_liga",
    "Serie A (Italy)": "soccer_italy_serie_a",
    "Bundesliga (Germany)": "soccer_germany_bundesliga",
    "MLS (USA)": "soccer_usa_mls",
}

# Valid player-prop market codes by league for The Odds API
ODDS_MARKETS = {
    "NFL": [
        "player_pass_yds",
        "player_rush_yds",
        "player_receiving_yds",
        "player_receptions",
        "player_rush_rec_yds",      # rush+rec
        "player_pass_rush_rec_yds", # pass+rush+rec (some books offer)
    ],
    "MLB": [
        "player_hits",
        "player_home_runs",
        "player_rbis",
        "player_stolen_bases",
        "player_total_bases",
        "player_strikeouts",   # pitcher strikeouts
    ],
    # Soccer coverage varies by comp/book; these are commonly seen:
    "Soccer": [
        "player_goals",
        "player_assists",
        "player_shots",
        "player_shots_on_target",
        "player_goals_assists",  # G+A (when available)
    ],
}

# Friendly -> Odds API market code mapping for UI
UI_TO_ODDS = {
    "NFL": {
        "Pass Yards": "player_pass_yds",
        "Rush Yards": "player_rush_yds",
        "Receiving Yards": "player_receiving_yds",
        "Receptions": "player_receptions",
        "Rush+Rec Yards": "player_rush_rec_yds",
        "Pass+Rush+Rec Yards": "player_pass_rush_rec_yds",
    },
    "MLB": {
        "Hits": "player_hits",
        "Home Runs": "player_home_runs",
        "RBIs": "player_rbis",
        "Stolen Bases": "player_stolen_bases",
        "Total Bases": "player_total_bases",
        "Pitcher Strikeouts": "player_strikeouts",
    },
    "Soccer": {
        "Goals": "player_goals",
        "Assists": "player_assists",
        "Shots": "player_shots",
        "Shots On Target": "player_shots_on_target",
        "G+A": "player_goals_assists",
    },
}

# --------------------- UI ---------------------
st.set_page_config(page_title="All-Sports Props Simulator", layout="wide")
st.title("📊 All-Sports Player Props — Odds API + Real Stats + Simulator (No fallbacks)")

page = st.sidebar.radio("Pages", ["NFL", "MLB", "Soccer"])
season_default = {"NFL": 2025, "MLB": 2024, "Soccer": 2025}[page]
season = st.sidebar.number_input("Season", 2018, 2026, value=season_default, step=1)

if page == "Soccer":
    comp = st.sidebar.selectbox("Competition", list(SOCCER_COMP_KEYS.keys()), index=0)
else:
    comp = None

books_input = st.sidebar.text_input(
    "Bookmakers (optional, comma-separated – average across selected; leave empty for all)",
    value="draftkings,fanduel,betmgm,caesars"
).strip()
BOOKS = [b.strip().lower() for b in books_input.split(",") if b.strip()] if books_input else []

region = st.sidebar.selectbox("Region", ["us", "us2", "eu", "uk"], index=0)
st.caption(
    "We fetch **player prop lines** from **The Odds API** ➜ load **real per-game stats** ➜ "
    "fuzzy-match players & markets ➜ estimate **P(Over/Under)** with a conservative normal model."
)

# ----------------- Utils/Model ------------------
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

def value_from_mapping(row: pd.Series, mapping):
    if isinstance(mapping, list):
        vals = [row.get(c) for c in mapping if c in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan
    return row.get(mapping)

# --------- Real stats loaders ----------
@st.cache_data(ttl=3600)
def load_nfl_stats(season: int):
    import nfl_data_py as nfl
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
        "player_pass_yds": "pass_yards",
        "player_rush_yds": "rush_yards",
        "player_receiving_yds": "rec_yards",
        "player_receptions": "receptions",
        "player_rush_rec_yds": ["rush_yards","rec_yards"],
        "player_pass_rush_rec_yds": ["pass_yards","rush_yards","rec_yards"],
    }
    return out, mapping

@st.cache_data(ttl=3600)
def load_mlb_stats(season: int):
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
        "player_hits": "hits",
        "player_home_runs": "hr",
        "player_rbis": "rbi",
        "player_stolen_bases": "sb",
        "player_total_bases": ["hits","hr"],  # coarse proxy
        "player_strikeouts": "pitch_strikeouts",
    }
    return out, mapping

@st.cache_data(ttl=3600)
def load_soccer_stats(season: int):
    import soccerdata as sd
    fb_season = season if season <= 2024 else 2024
    leagues = ["ENG-Premier League","ESP-La Liga","ITA-Serie A","GER-Bundesliga","FRA-Ligue 1"]
    frames=[]
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
            }))
        except Exception:
            continue
    if not frames:
        raise ValueError("No soccer frames returned.")
    out = pd.concat(frames, ignore_index=True)
    mapping = {
        "player_goals": "goals",
        "player_assists": "assists",
        "player_shots": "shots",
        "player_shots_on_target": "sog",
        "player_goals_assists": ["goals","assists"],
    }
    return out, mapping

def load_stats_map_for_page(page: str, season: int):
    if page == "NFL": return load_nfl_stats(season)
    if page == "MLB": return load_mlb_stats(season)
    return load_soccer_stats(season)

# ----------- Odds API: Player Props -------------
def _avg_from_books(offers: List[dict], books: List[str]) -> Optional[float]:
    vals=[]
    for off in offers:
        bk = (off.get("bookmaker") or off.get("key") or off.get("title") or "").lower()
        if books and bk not in books:
            continue
        mkts = off.get("markets") or []
        for m in mkts:
            # For player props, m['key'] is already our market; outcomes have one entry per player
            outcomes = m.get("outcomes") or []
            # We'll extract at higher level; this helper is unused in new parse path.
    return None

def fetch_oddsapi_props(sport_key: str, markets: List[str], books: List[str], region: str) -> pd.DataFrame:
    """
    Calls /v4/sports/{sport}/odds with markets=... and parses player prop lines.
    Returns DataFrame with columns: player, market, line
    """
    if not ODDS_API_KEY:
        raise RuntimeError("Set ODDS_API_KEY in environment/secrets.")

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "markets": ",".join(markets),
    }
    if books:
        params["bookmakers"] = ",".join(books)

    url = f"{ODDS_BASE}/sports/{sport_key}/odds"
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text}")

    events = r.json()
    rows = []

    # Structure: events[] → bookmakers[] → markets[] (by our requested keys)
    # For player props, each market has outcomes[] where each outcome belongs to a player.
    for ev in events:
        bks = ev.get("bookmakers") or []
        # group by market key -> list of lines by player
        market_player_points: Dict[str, Dict[str, List[float]]] = {}
        for bk in bks:
            bk_key = (bk.get("key") or bk.get("title") or "").lower()
            if books and bk_key not in [b.lower() for b in books]:
                continue
            mkts = bk.get("markets") or []
            for m in mkts:
                mkey = m.get("key")
                if not mkey or mkey not in markets:
                    continue
                outs = m.get("outcomes") or []
                for o in outs:
                    # Common fields: name (player), point (line)
                    player = o.get("name") or o.get("description")
                    point = o.get("point")
                    if player is None or point is None:
                        continue
                    try:
                        val = float(point)
                    except:
                        continue
                    market_player_points.setdefault(mkey, {}).setdefault(player, []).append(val)

        # after looping bookmakers, average line per player/market across selected books
        for mkey, players in market_player_points.items():
            for player, pts in players.items():
                if not pts: 
                    continue
                rows.append({"player": player, "market": mkey, "line": float(np.mean(pts))})

    return pd.DataFrame(rows).dropna(subset=["player","market","line"]).reset_index(drop=True)

# ------------- Market → Stats mapper -------------
def market_to_stats_columns(page: str, market_key: str, mapping_dict: Dict[str, object]) -> Optional[object]:
    """Return stats column(s) for a given Odds API market code."""
    return mapping_dict.get(market_key)

# ------------- Page runner -----------------------
def run_page(page: str, season: int, comp_key: Optional[str] = None):
    st.header(page)

    # Market picker (friendly labels mapped to Odds API market keys)
    ui_map = UI_TO_ODDS[page]
    chosen = st.multiselect(
        "Markets to pull", list(ui_map.keys()),
        default=list(ui_map.keys())[:4]
    )
    markets = [ui_map[k] for k in chosen]

    # 1) Fetch prop lines (Odds API only)
    sport_key = SPORT_KEYS.get(page) if page != "Soccer" else SOCCER_COMP_KEYS[comp]
    try:
        with st.spinner("Fetching player props from The Odds API…"):
            board = fetch_oddsapi_props(sport_key, markets, BOOKS, region)
    except Exception as e:
        st.error(f"Odds API error: {e}")
        st.stop()

    if board.empty:
        st.warning("No player props returned by The Odds API for these settings.")
        st.stop()

    st.success(f"Loaded {len(board)} player props.")
    st.dataframe(board.head(20), use_container_width=True)

    # 2) Load real stats + mapping (per page)
    try:
        stats_df, market_map = load_stats_map_for_page(page, season)
    except Exception as e:
        st.error(f"Could not load {page} stats: {e}")
        st.stop()

    stats_df = stats_df.dropna(subset=["player"]).copy()
    players = list(stats_df["player"].unique())

    # 3) Simulate
    rows=[]
    prog = st.progress(0.0)
    total = len(board)
    for i, r in board.iterrows():
        prog.progress((i+1)/max(total,1))
        pp_player, mkt_key, line_val = r["player"], r["market"], r["line"]

        # name match
        match = best_name_match(pp_player, players, score_cut=82)
        if match is None:
            continue

        # map Odds API market code -> stats columns for this page
        stat_cols = market_to_stats_columns(page, mkt_key, market_map)
        if stat_cols is None:
            continue

        row_stats = stats_df.loc[stats_df["player"] == match].iloc[0]
        avg_val = value_from_mapping(row_stats, stat_cols)
        if pd.isna(avg_val):
            continue

        try:
            line_val = float(line_val)
        except:
            continue

        p_over, p_under, sd_used = simulate_prob(avg_val, line_val)
        rows.append({
            "player": match,
            "prop_player": pp_player,
            "market": mkt_key,
            "line": round(line_val, 3),
            "avg": round(float(avg_val), 3),
            "model_sd": sd_used,
            "P(Over)": p_over,
            "P(Under)": p_under,
        })

    prog.empty()
    results = pd.DataFrame(rows).sort_values(["P(Over)","P(Under)"], ascending=[False, True])

    if results.empty:
        st.warning("No matched rows (names/markets didn’t align). Try different markets or books.")
        st.stop()

    st.subheader("Simulated edges (conservative normal model)")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name=f"{page.lower()}_{season}_oddsapi_props_sim.csv",
        mime="text/csv",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 Overs**")
        st.dataframe(results.nlargest(10, "P(Over)")[["player","market","line","avg","P(Over)","P(Under)"]], use_container_width=True)
    with c2:
        st.markdown("**Top 10 Unders**")
        st.dataframe(results.nlargest(10, "P(Under)")[["player","market","line","avg","P(Over)","P(Under)"]], use_container_width=True)

# ------------- Run the selected page -------------
if page == "NFL":
    run_page("NFL", season)
elif page == "MLB":
    run_page("MLB", season)
else:
    run_page("Soccer", season, comp_key=SOCCER_COMP_KEYS.get(comp))
