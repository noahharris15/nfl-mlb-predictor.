# ================================
# All-Sports PrizePicks Simulator
# (Real stats + automatic simulation)
# ================================
import time
import math
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz, process

# ------- Optional providers (each imported lazily inside functions) -------
# nfl_data_py  -> NFL season/game stats
# pybaseball   -> MLB batting/pitching stats
# soccerdata   -> FBref player-season stats for top Euro leagues
# nba: use balldontlie public API (no key needed)

st.set_page_config(page_title="All-Sports PrizePicks Simulator", layout="wide")
st.title("📊 All-Sports PrizePicks Simulator (Real stats + Auto Simulation)")

# -----------------------------
# UI controls
# -----------------------------
league = st.selectbox("League", ["NFL", "NBA", "MLB", "Soccer"])
season_default = {"NFL": 2024, "NBA": 2024, "MLB": 2024, "Soccer": 2024}[league]
season = st.number_input("Season", min_value=2018, max_value=2025, value=season_default, step=1)

st.caption(
    "We fetch the live PrizePicks board ➜ filter to the selected league ➜ "
    "fetch real per-game averages ➜ fuzzy-match players/stat ➜ "
    "simulate using a conservative normal model to avoid 0%/100% artifacts."
)

# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0"}

def backoff_fetch(url, headers=None, tries=4, start_wait=1.0):
    wait = start_wait
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, headers=headers or HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            last_err = Exception(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
        time.sleep(wait)
        wait *= 2
    raise last_err

def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").strip().lower()

def conservative_sd(avg, minimum=0.75, frac=0.30):
    """
    Conservative SD:
      - If avg is 0 → small floor SD to avoid 0/100
      - Otherwise max( frac*avg, minimum ) but scaled for very large counts
    """
    if pd.isna(avg): 
        return 1.25
    if avg <= 0:
        return 1.0
    sd = max(frac * float(avg), minimum)
    return sd

def simulate_prob(avg, line):
    sd = conservative_sd(avg)
    # clamp sd (avoid insane 0/100)
    sd = max(sd, 0.5)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over * 100, 2), round(p_under * 100, 2), round(sd, 3)

# ------------------------------------------------------------------------
# PrizePicks fetch & parse (single-stat projections only)
# ------------------------------------------------------------------------
PP_URL = "https://api.prizepicks.com/projections?per_page=1000&single_stat=true"

def fetch_prizepicks_board() -> pd.DataFrame:
    r = backoff_fetch(PP_URL)
    data = r.json()
    df = pd.json_normalize(data.get("data", []))

    # Guard against structure changes
    keep = []
    for col in df.columns:
        if col in [
            "attributes.league",
            "attributes.display_stat",
            "attributes.stat_type",
            "attributes.line_score",
            "attributes.player_name",
            "attributes.team"
        ]:
            keep.append(col)
    if not keep:
        raise ValueError("PrizePicks schema changed: expected attributes.* fields not found.")

    out = df[keep].rename(columns={
        "attributes.league": "league",
        "attributes.display_stat": "stat",
        "attributes.stat_type": "stat_type",
        "attributes.line_score": "line",
        "attributes.player_name": "player",
        "attributes.team": "pp_team"
    })
    out["league"] = out["league"].astype(str)
    out["player"] = out["player"].astype(str)
    out["stat"] = out["stat"].astype(str)
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    return out.dropna(subset=["line"])

# Normalize PP league label to our UI
PP_LEAGUE_MAP = {
    "nfl": "NFL", "NFL": "NFL",
    "nba": "NBA", "NBA": "NBA",
    "mlb": "MLB", "MLB": "MLB",
    "soc": "Soccer", "soccer": "Soccer", "Soccer": "Soccer", "EPL": "Soccer"
}

def filter_league(df: pd.DataFrame, league: str) -> pd.DataFrame:
    mask = df["league"].map(lambda x: PP_LEAGUE_MAP.get(str(x), str(x)))
    return df[mask == league].reset_index(drop=True)

# ------------------------------------------------------------------------
# Real stats providers (per league) ➜ return (df, schema dict)
# df must have columns: player, team(optional), stat columns we map to PP stats
# ------------------------------------------------------------------------

# ---- NFL ----
def load_nfl_stats(season: int) -> tuple[pd.DataFrame, dict]:
    import nfl_data_py as nfl
    st.info("Loading NFL player season data…")
    df = nfl.import_seasonal_data([season])
    # Per-game where possible (nfl_data_py returns totals; divide by games if present)
    if "games" in df.columns:
        g = df["games"].replace(0, np.nan)
    else:
        g = np.nan

    out = pd.DataFrame({
        "player": df.get("player_display_name"),
        "team": df.get("recent_team"),
        "pass_yards": df.get("passing_yards") / g if "passing_yards" in df and "games" in df else df.get("passing_yards"),
        "rush_yards": df.get("rushing_yards") / g if "rushing_yards" in df and "games" in df else df.get("rushing_yards"),
        "rec_yards": df.get("receiving_yards") / g if "receiving_yards" in df and "games" in df else df.get("receiving_yards"),
        "receptions": df.get("receptions") / g if "receptions" in df and "games" in df else df.get("receptions"),
        "pass_tds": df.get("passing_tds") / g if "passing_tds" in df and "games" in df else df.get("passing_tds"),
        "rush_tds": df.get("rushing_tds") / g if "rushing_tds" in df and "games" in df else df.get("rushing_tds"),
        "rec_tds": df.get("receiving_tds") / g if "receiving_tds" in df and "games" in df else df.get("receiving_tds"),
        "ints": df.get("interceptions") / g if "interceptions" in df and "games" in df else df.get("interceptions"),
    })
    out = out.dropna(subset=["player"])
    # Mapping PP market names to our columns (most common)
    mapping = {
        "Pass Yards": "pass_yards",
        "Passing Yards": "pass_yards",
        "Rush Yards": "rush_yards",
        "Rushing Yards": "rush_yards",
        "Receiving Yards": "rec_yards",
        "Receptions": "receptions",
        "Pass TDs": "pass_tds",
        "Rush TDs": "rush_tds",
        "Receiving TDs": "rec_tds",
        "Interceptions": "ints",
        "Pass+Rush Yds": ["pass_yards", "rush_yards"],
        "Rush+Rec Yds": ["rush_yards", "rec_yards"],
        "Pass+Rush+Rec Yds": ["pass_yards", "rush_yards", "rec_yards"],
    }
    return out, mapping

# ---- NBA (balldontlie) ----
def load_nba_stats(season: int) -> tuple[pd.DataFrame, dict]:
    st.info("Loading NBA per-game stats (balldontlie)…")
    # Aggregate across all game logs
    per_page = 100
    page = 1
    rows = []
    while True:
        url = f"https://www.balldontlie.io/api/v1/season_averages?season={season}&per_page={per_page}&page={page}"
        r = backoff_fetch(url)
        js = r.json()
        data = js.get("data", [])
        if not data:
            break
        rows.extend(data)
        page += 1
        if page > 40:  # safety
            break
    if not rows:
        raise ValueError("No NBA averages returned.")

    df = pd.DataFrame(rows)
    # Map columns
    out = pd.DataFrame({
        "player": df["player_id"].astype(str),  # placeholder; remap to names
        "points": df.get("pts"),
        "assists": df.get("ast"),
        "rebounds": df.get("reb"),
        "threes": df.get("fg3m"),
    })
    # Get id->name
    if not out.empty:
        ids = list(df["player_id"].unique())
        id_to_name = {}
        # fetch in chunks
        for chunk in [ids[i:i+50] for i in range(0, len(ids), 50)]:
            url = "https://www.balldontlie.io/api/v1/players?per_page=100&" + "&".join([f"ids[]={x}" for x in chunk])
            rr = backoff_fetch(url)
            for p in rr.json().get("data", []):
                id_to_name[str(p["id"])] = f"{p['first_name']} {p['last_name']}"
        out["player"] = out["player"].map(id_to_name)
    out = out.dropna(subset=["player"])
    mapping = {
        "Points": "points",
        "Assists": "assists",
        "Rebounds": "rebounds",
        "3-PT Made": "threes",
        "Pts+Reb+Ast": ["points", "rebounds", "assists"],
        "Pts+Reb": ["points", "rebounds"],
        "Pts+Ast": ["points", "assists"],
        "Reb+Ast": ["rebounds", "assists"],
    }
    return out, mapping

# ---- MLB (pybaseball) ----
def load_mlb_stats(season: int) -> tuple[pd.DataFrame, dict]:
    st.info("Loading MLB season stats (pybaseball)…")
    from pybaseball import batting_stats, pitching_stats
    bat = batting_stats(season)
    pit = pitching_stats(season)

    bat_out = pd.DataFrame({
        "player": bat.get("Name"),
        "hits": bat.get("H"),
        "hr": bat.get("HR"),
        "rbi": bat.get("RBI"),
        "sb": bat.get("SB"),
        "bb": bat.get("BB"),
        "so": bat.get("SO"),
        "ab": bat.get("AB")
    }).dropna(subset=["player"])

    # Convert to per-game-ish rates using PA proxy if available
    pa = bat.get("PA") if "PA" in bat.columns else None
    if pa is not None:
        pa = pa.replace(0, np.nan)
        bat_out["hits"] = bat_out["hits"] / pa * 4.2  # approx per-game scale
        bat_out["hr"]   = bat_out["hr"] / pa * 4.2
        bat_out["rbi"]  = bat_out["rbi"] / pa * 4.2
        bat_out["sb"]   = bat_out["sb"] / pa * 4.2

    pit_out = pd.DataFrame({
        "player": pit.get("Name"),
        "pitch_strikeouts": pit.get("SO"),
        "innings": pit.get("IP")
    }).dropna(subset=["player"])

    out = pd.merge(bat_out, pit_out, on="player", how="outer")
    mapping = {
        "Hitter Fantasy Score": ["hits", "hr", "rbi", "sb"],  # rough proxy
        "Hits": "hits",
        "Home Runs": "hr",
        "RBIs": "rbi",
        "Stolen Bases": "sb",
        "Pitcher Strikeouts": "pitch_strikeouts",
    }
    return out, mapping

# ---- Soccer (soccerdata/FBref top 5 leagues) ----
def load_soccer_stats(season: int) -> tuple[pd.DataFrame, dict]:
    st.info("Loading Soccer player stats (FBref via soccerdata) …")
    # leagues FBref uses: 'ENG-Premier League', 'ESP-La Liga', etc.
    import soccerdata as sd
    leagues = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
    frames = []
    for lg in leagues:
        try:
            fb = sd.FBref(leagues=lg, seasons=season)
            df = fb.read_player_season_stats(stat_type="standard")
            # df index may be multi; reset
            df = df.reset_index()
            # Basic per-90 metrics if present
            name_col = "player" if "player" in df.columns else "Player"
            team_col = "team" if "team" in df.columns else "Squad" if "Squad" in df.columns else None
            goals = df["Gls"] if "Gls" in df.columns else (df["Gls/90"]*df["90s"] if "Gls/90" in df and "90s" in df.columns else np.nan)
            shots = df["Sh"] if "Sh" in df.columns else np.nan
            sog   = df["SoT"] if "SoT" in df.columns else np.nan
            ast   = df["Ast"] if "Ast" in df.columns else np.nan
            kp    = df["KP"] if "KP" in df.columns else np.nan

            frames.append(pd.DataFrame({
                "player": df[name_col],
                "team": df[team_col] if team_col and team_col in df.columns else None,
                "goals": goals,
                "shots": shots,
                "sog": sog,
                "assists": ast,
                "key_passes": kp
            }))
        except Exception:
            continue
    if not frames:
        raise ValueError("No soccer frames returned from FBref.")
    out = pd.concat(frames, ignore_index=True)
    # crude per-game scale if 90s present (not guaranteed here), otherwise leave as totals
    mapping = {
        "Goals": "goals",
        "Shots": "shots",
        "Shots On Goal": "sog",
        "Assists": "assists",
        "Passes Created": "key_passes",
        "Shots+SOG": ["shots", "sog"],
        "G+A": ["goals", "assists"]
    }
    return out, mapping

# ------------------------------------------------------------------------
# Choose provider
# ------------------------------------------------------------------------
def get_provider(league: str):
    if league == "NFL":
        return load_nfl_stats
    if league == "NBA":
        return load_nba_stats
    if league == "MLB":
        return load_mlb_stats
    if league == "Soccer":
        return load_soccer_stats
    raise ValueError("Unsupported league")

# ------------------------------------------------------------------------
# Fuzzy match utilities
# ------------------------------------------------------------------------
def best_name_match(name, candidates, score_cut=80):
    name_c = clean_name(name)
    choices = [(c, clean_name(c)) for c in candidates]
    # RapidFuzz: return best match
    best = None
    best_score = -1
    for raw, c in choices:
        s = fuzz.token_sort_ratio(name_c, c)
        if s > best_score:
            best = raw
            best_score = s
    return best if best_score >= score_cut else None

# Sum columns if mapping is a list
def value_from_mapping(row_stats: pd.Series, stat_map):
    if isinstance(stat_map, list):
        vals = [row_stats.get(col) for col in stat_map if col in row_stats.index]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan
    return row_stats.get(stat_map)

# ------------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------------
# 1) PrizePicks board
with st.expander("Fetch PrizePicks board (debug)", expanded=False):
    st.write("GET", PP_URL)

try:
    board = fetch_prizepicks_board()
except Exception as e:
    st.error(f"PrizePicks fetch/parse error: {e}")
    st.stop()

board = filter_league(board, league)
if board.empty:
    st.warning("No PrizePicks markets found for this league right now.")
    st.stop()

st.success(f"Loaded {len(board)} PrizePicks projections for {league}.")

# 2) Real stats
provider = get_provider(league)
try:
    stats_df, market_map = provider(season)
except Exception as e:
    st.error(f"Could not load {league} stats: {e}")
    st.stop()

stats_df = stats_df.dropna(subset=["player"]).copy()
stats_df["player_clean"] = stats_df["player"].map(clean_name)

# 3) Build per-row matches + simulation
rows = []
players_set = list(stats_df["player"].unique())

progress = st.progress(0)
for i, r in board.iterrows():
    progress.progress((i+1)/len(board))
    pp_player = r["player"]
    pp_stat   = r["stat"]
    pp_line   = r["line"]

    # Choose market mapping
    stat_map = None
    # exact match first
    if pp_stat in market_map:
        stat_map = market_map[pp_stat]
    else:
        # some PP labels vary slightly; try loose keys
        # example: "Pass Yards" vs "Passing Yards"
        for k in market_map.keys():
            if fuzz.token_set_ratio(pp_stat.lower(), k.lower()) >= 90:
                stat_map = market_map[k]
                break
    if stat_map is None:
        continue  # unsupported market → skip

    # Fuzzy player match
    best = best_name_match(pp_player, players_set, score_cut=82)
    if best is None:
        continue

    row_stats = stats_df.loc[stats_df["player"] == best].iloc[0]
    avg_val = value_from_mapping(row_stats, stat_map)
    try:
        line_val = float(pp_line)
    except:
        continue

    if pd.isna(avg_val):
        continue

    p_over, p_under, used_sd = simulate_prob(avg_val, line_val)

    rows.append({
        "league": league,
        "player": best,
        "pp_player": pp_player,
        "team": row_stats.get("team"),
        "market": pp_stat,
        "line": line_val,
        "avg": round(float(avg_val), 3),
        "model_sd": used_sd,
        "P(Over)": p_over,
        "P(Under)": p_under
    })

progress.empty()

results = pd.DataFrame(rows).sort_values(["P(Over)"], ascending=False)
if results.empty:
    st.warning("No rows could be matched (either unsupported markets or player names didn’t match).")
    st.stop()

st.subheader("Simulated edges (conservative normal model)")
st.caption("Probabilities are model estimates based on real per-game averages. SD is conservative to avoid 0/100 artifacts.")
st.dataframe(results, use_container_width=True)

st.download_button(
    "Download CSV",
    results.to_csv(index=False).encode("utf-8"),
    file_name=f"{league.lower()}_{season}_prizepicks_sim.csv",
    mime="text/csv",
)

# Nice small summary
st.markdown("#### Top 10 Overs")
st.dataframe(results.nlargest(10, "P(Over)")[["player","market","line","avg","P(Over)","P(Under)"]], use_container_width=True)
st.markdown("#### Top 10 Unders")
st.dataframe(results.nlargest(10, "P(Under)")[["player","market","line","avg","P(Over)","P(Under)"]], use_container_width=True)
