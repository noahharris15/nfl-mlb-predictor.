# app.py
# One-file Streamlit app: Odds API player props + real stats + Monte Carlo (conservative normal) simulation
# Leagues: NFL, NCAAF (College Football), MLB, Soccer

import os, time, random, json, math
import pandas as pd
import numpy as np
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# ------------------------ UI / CONFIG ------------------------
st.set_page_config(page_title="All-Sports Simulator (Odds API + Auto Stats)", layout="wide")
st.title("📊 All-Sports Player-Props Simulator (Odds API + Real Stats)")

with st.sidebar:
    st.markdown("### 🔑 API Keys")
    ODDS_API_KEY = st.text_input("The Odds API Key", value="9ede18ca5b55fa2afc180a2b375367e2", type="password")
    CFBD_API_KEY = st.text_input("CollegeFootballData (CFBD) API Key (optional, needed for CFB)", type="password")
    st.caption("CFB player stats require a free CFBD key. NFL/MLB/Soccer do not.")

# Top controls
league = st.selectbox("League", ["NFL", "College Football", "MLB", "Soccer"])
season_default = {"NFL": 2025, "College Football": 2024, "MLB": 2024, "Soccer": 2024}[league]
season = st.number_input("Season", min_value=2018, max_value=2025, value=season_default, step=1)

st.caption(
    "We fetch **player lines** from **The Odds API**, auto-load **real per-game player averages**, "
    "fuzzy-match names and stat types, and estimate **P(Over/Under)** with a conservative normal model."
)

# ------------------------ HELPERS ------------------------
def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").replace("'", "").strip().lower()

def conservative_sd(avg, minimum=0.75, frac=0.30):
    """Conservative SD to avoid 0%/100% artifacts."""
    if pd.isna(avg): return 1.25
    if avg <= 0:     return 1.0
    sd = max(frac * float(avg), minimum)
    return max(sd, 0.5)

def simulate_prob(avg, line):
    sd = conservative_sd(avg)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over * 100, 2), round(p_under * 100, 2), round(sd, 3)

def best_name_match(name, candidates, score_cut=82):
    name_c = clean_name(name)
    best, best_score = None, -1
    for c in candidates:
        s = fuzz.token_sort_ratio(name_c, clean_name(c))
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= score_cut else None

def sum_or_nan(values):
    vals = [v for v in values if pd.notna(v)]
    return float(np.sum(vals)) if vals else np.nan

def value_from_mapping(row_stats: pd.Series, stat_map):
    if isinstance(stat_map, list):
        return sum_or_nan([row_stats.get(c) for c in stat_map if c in row_stats.index])
    return row_stats.get(stat_map)

# ------------------------ LEAGUE MAPS ------------------------
SPORT_KEY = {
    "NFL": "americanfootball_nfl",
    "College Football": "americanfootball_ncaaf",
    "MLB": "baseball_mlb",
    "Soccer": "soccer_usa_mls"  # Odds API key for a default soccer league (MLS). You can add others if you like.
}

# Preferred markets we try (names differ by sport/book; we try multiple aliases)
MARKET_ALIAS = {
    "NFL": [
        "player_props",
        "player_passing_yards,player_rushing_yards,player_receiving_yards,player_receptions",
        "player_pass_yds,player_rush_yds,player_receiving_yds,player_receptions",
    ],
    "College Football": [
        "player_props",
        "player_passing_yards,player_rushing_yards,player_receiving_yards,player_receptions",
        "player_pass_yds,player_rush_yds,player_receiving_yds,player_receptions",
    ],
    "MLB": [
        "player_props",
        "player_hits,player_home_runs,player_rbis,player_stolen_bases,player_strikeouts",
        "player_total_bases,player_hits_runs_rbis,player_walks,player_strikeouts",  # many books expose K's as pitcher SO
    ],
    "Soccer": [
        "player_props",
        "player_goals,player_shots,player_shots_on_target,player_assists",
    ],
}

# Display name -> our stat map (per league)
STAT_MAP = {
    "NFL": {
        "Passing Yards": "pass_yards",
        "Rushing Yards": "rush_yards",
        "Receiving Yards": "rec_yards",
        "Receptions": "receptions",
        "Pass+Rush Yds": ["pass_yards", "rush_yards"],
        "Rush+Rec Yds": ["rush_yards", "rec_yards"],
        "Pass+Rush+Rec Yds": ["pass_yards", "rush_yards", "rec_yards"],
    },
    "College Football": {
        "Passing Yards": "pass_yards",
        "Rushing Yards": "rush_yards",
        "Receiving Yards": "rec_yards",
        "Receptions": "receptions",
        "Pass+Rush Yds": ["pass_yards", "rush_yards"],
        "Rush+Rec Yds": ["rush_yards", "rec_yards"],
        "Pass+Rush+Rec Yds": ["pass_yards", "rush_yards", "rec_yards"],
    },
    "MLB": {
        "Hits": "hits",
        "Home Runs": "hr",
        "RBIs": "rbi",
        "Stolen Bases": "sb",
        "Pitcher Strikeouts": "pitch_strikeouts",
        "Hitter Fantasy (proxy)": ["hits", "hr", "rbi", "sb"],
    },
    "Soccer": {
        "Goals": "goals",
        "Shots": "shots",
        "Shots On Goal": "sog",
        "Assists": "assists",
        "Shots+SOG": ["shots", "sog"],
        "G+A": ["goals", "assists"],
    },
}

# Text in Odds API -> our normalized market names to hit the above STAT_MAP
ODDS_MARKET_NORMALIZER = {
    # NFL/NCAAF
    "player_passing_yards": ("Passing Yards", "pass_yards"),
    "player_pass_yds": ("Passing Yards", "pass_yards"),
    "player_rushing_yards": ("Rushing Yards", "rush_yards"),
    "player_rush_yds": ("Rushing Yards", "rush_yards"),
    "player_receiving_yards": ("Receiving Yards", "rec_yards"),
    "player_receiving_yds": ("Receiving Yards", "rec_yards"),
    "player_receptions": ("Receptions", "receptions"),

    # MLB
    "player_hits": ("Hits", "hits"),
    "player_home_runs": ("Home Runs", "hr"),
    "player_rbis": ("RBIs", "rbi"),
    "player_stolen_bases": ("Stolen Bases", "sb"),
    "player_strikeouts": ("Pitcher Strikeouts", "pitch_strikeouts"),
    "pitcher_strikeouts": ("Pitcher Strikeouts", "pitch_strikeouts"),
    "player_total_bases": ("Hitter Fantasy (proxy)", ["hits", "hr", "rbi", "sb"]),  # proxy mapping
    "player_hits_runs_rbis": ("Hitter Fantasy (proxy)", ["hits", "hr", "rbi", "sb"]),

    # Soccer
    "player_goals": ("Goals", "goals"),
    "player_shots": ("Shots", "shots"),
    "player_shots_on_target": ("Shots On Goal", "sog"),
    "player_assists": ("Assists", "assists"),
}

# ------------------------ ODDS API FETCH ------------------------
ODDS_BASE = "https://api.the-odds-api.com/v4"

def build_odds_url(sport_key: str, markets: str, regions="us", bookmakers=None):
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "oddsFormat": "decimal",
        "includeLinks": "false",
        "markets": markets
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    return f"{ODDS_BASE}/sports/{sport_key}/odds", params

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=5))
def _http_get(url, params):
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
    return r.json()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_player_lines(league: str, bookmakers_csv: str | None) -> pd.DataFrame:
    """Try multiple market patterns until we get player props; normalize to (player, team, market_key, market_name, line)."""
    sport_key = SPORT_KEY[league]
    trials = MARKET_ALIAS.get(league, ["player_props"])
    last_err = None
    for markets in trials:
        try:
            url, params = build_odds_url(sport_key, markets, bookmakers=bookmakers_csv)
            js = _http_get(url, params)
            rows = []
            for game in js or []:
                book_list = game.get("bookmakers", [])
                for bk in book_list:
                    for mk in bk.get("markets", []):
                        mkey = mk.get("key", "")
                        # only keep markets we can normalize
                        if mkey not in ODDS_MARKET_NORMALIZER:
                            continue
                        norm_name, _ = ODDS_MARKET_NORMALIZER[mkey]
                        for outcome in mk.get("outcomes", []):
                            player = outcome.get("description") or outcome.get("name")
                            line = outcome.get("point")
                            team = outcome.get("team")
                            if player and line is not None:
                                rows.append({
                                    "bookmaker": bk.get("key"),
                                    "market_key": mkey,
                                    "market": norm_name,
                                    "player": player,
                                    "team": team,
                                    "line": float(line),
                                })
            if not rows:
                # try next market pattern
                last_err = f"No usable rows for markets='{markets}'"
                continue
            df = pd.DataFrame(rows)
            # combine across books: average line per player & market
            agg = df.groupby(["player", "market"], as_index=False).agg(
                line=("line", "mean"),
                n_books=("bookmaker", "nunique")
            )
            return agg.sort_values(["market","player"]).reset_index(drop=True)
        except Exception as e:
            last_err = str(e)
            time.sleep(0.4 + random.random())
            continue
    raise RuntimeError(f"Odds API fetch failed: {last_err}")

# ------------------------ REAL STATS LOADERS ------------------------
@st.cache_data(ttl=3600, show_spinner=False)
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
    }).dropna(subset=["player"])
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def load_cfb_stats(season: int, cfbd_key: str | None):
    if not cfbd_key:
        raise RuntimeError("CFBD key missing. Add it in the sidebar to enable College Football.")
    # CFBD player season stats (per game) – offense
    headers = {"Authorization": f"Bearer {cfbd_key}"}
    url = f"https://api.collegefootballdata.com/player/season?year={season}"
    r = requests.get(url, headers=headers, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"CFBD {r.status_code}: {r.text[:200]}")
    js = r.json()
    rows = []
    for p in js:
        name = p.get("player")
        team = p.get("team")
        gp = p.get("games") or p.get("gp") or np.nan
        def per_game(v): 
            try:
                v = float(v)
                return v / gp if gp and gp > 0 else np.nan
            except: 
                return np.nan
        rows.append({
            "player": name, "team": team,
            "pass_yards": per_game(p.get("passingYards")),
            "rush_yards": per_game(p.get("rushingYards")),
            "rec_yards":  per_game(p.get("receivingYards")),
            "receptions": per_game(p.get("receptions")),
        })
    df = pd.DataFrame(rows).dropna(subset=["player"])
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_mlb_stats(season: int):
    from pybaseball import batting_stats, pitching_stats
    bat = batting_stats(season); pit = pitching_stats(season)
    bat_out = pd.DataFrame({
        "player": bat.get("Name"),
        "hits": bat.get("H"), "hr": bat.get("HR"), "rbi": bat.get("RBI"),
        "sb": bat.get("SB"), "pa": bat.get("PA")
    }).dropna(subset=["player"])
    bat_out["pa"] = bat_out["pa"].replace(0, np.nan)
    for c in ["hits","hr","rbi","sb"]:
        bat_out[c] = bat_out[c] / bat_out["pa"] * 4.2  # per-game-ish proxy using PA
    bat_out = bat_out.drop(columns=["pa"])
    pit_out = pd.DataFrame({
        "player": pit.get("Name"),
        "pitch_strikeouts": pit.get("SO"),
        "games": pit.get("G")
    }).dropna(subset=["player"])
    pit_out["pitch_strikeouts"] = pit_out["pitch_strikeouts"] / pit_out["games"].replace({0:np.nan})
    pit_out = pit_out.drop(columns=["games"])
    out = pd.merge(bat_out, pit_out, on="player", how="outer")
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def load_soccer_stats(season: int):
    import soccerdata as sd
    fb_season = season if season <= 2024 else 2024
    leagues = ["ENG-Premier League","ESP-La Liga","ITA-Serie A","GER-Bundesliga","FRA-Ligue 1","USA-MLS"]
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
            }))
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No soccer stats returned.")
    return pd.concat(frames, ignore_index=True)

def load_stats_for_league(league: str, season: int, cfbd_key: str | None):
    if league == "NFL": return load_nfl_stats(season), STAT_MAP["NFL"]
    if league == "College Football": return load_cfb_stats(season, cfbd_key), STAT_MAP["College Football"]
    if league == "MLB": return load_mlb_stats(season), STAT_MAP["MLB"]
    if league == "Soccer": return load_soccer_stats(season), STAT_MAP["Soccer"]
    raise ValueError("Unsupported league")

# ------------------------ MAIN FLOW ------------------------
bookmakers = st.multiselect(
    "Bookmakers (optional – average line across selected; leave empty to include all returned)",
    ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "williamhill", "barstool", "betrivers"],
)
bookmakers_csv = ",".join(bookmakers) if bookmakers else None

try:
    board = fetch_player_lines(league, bookmakers_csv)
    st.success(f"Fetched {len(board)} player lines from Odds API for {league}.")
except Exception as e:
    st.error(f"Odds API error: {e}")
    st.stop()

# Load stats
try:
    stats_df, market_map = load_stats_for_league(league, season, CFBD_API_KEY)
except Exception as e:
    st.error(f"Could not load {league} stats: {e}")
    st.stop()

stats_df = stats_df.dropna(subset=["player"]).copy()
players_list = list(stats_df["player"].unique())

# Map Odd API market_key text to our normalized key (PPR friendly string already set in board.market)
def normalize_market_name(market_text: str) -> str:
    # already normalized in fetch; just return as-is
    return market_text

# ------------------------ SIMULATION ------------------------
rows = []
prog = st.progress(0)
for i, r in board.iterrows():
    prog.progress((i+1)/len(board))
    pp_player, pp_market, pp_line = r["player"], normalize_market_name(r["market"]), r["line"]
    if pd.isna(pp_player) or pd.isna(pp_market) or pd.isna(pp_line):
        continue

    # map Odds market to our stat columns (via STAT_MAP)
    stat_map = STAT_MAP[league].get(pp_market)
    if stat_map is None:
        # attempt loose match
        for k in STAT_MAP[league].keys():
            if fuzz.token_set_ratio(pp_market.lower(), k.lower()) >= 90:
                stat_map = STAT_MAP[league][k]; break
    if stat_map is None:
        continue

    match = best_name_match(pp_player, players_list, score_cut=82)
    if not match:
        continue

    row_stats = stats_df.loc[stats_df["player"] == match].iloc[0]
    avg_val = value_from_mapping(row_stats, stat_map)
    if pd.isna(avg_val):
        continue

    try:
        line_val = float(pp_line)
    except:
        continue

    p_over, p_under, used_sd = simulate_prob(avg_val, line_val)
    rows.append({
        "league": league,
        "player": match,
        "api_player": pp_player,
        "team": row_stats.get("team"),
        "market": pp_market,
        "line": round(line_val, 3),
        "avg": round(float(avg_val), 3),
        "model_sd": used_sd,
        "P(Over)": p_over,
        "P(Under)": p_under,
        "books_used": r.get("n_books", np.nan),
    })

prog.empty()
results = pd.DataFrame(rows).sort_values(["P(Over)","P(Under)"], ascending=[False, True])

if results.empty:
    st.warning("No matches between Odds API players/markets and our stat providers. Try different markets or season.")
    st.stop()

st.subheader("Simulated edges (conservative normal model)")
st.caption("Probabilities are model estimates using real per-game averages. SD is conservative to avoid 0%/100% artifacts.")
st.dataframe(results, use_container_width=True)

st.download_button(
    "⬇️ Download results (CSV)",
    results.to_csv(index=False).encode("utf-8"),
    file_name=f"{league.replace(' ','_').lower()}_{season}_oddsapi_sim.csv",
    mime="text/csv",
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Top 15 — Overs")
    st.dataframe(results.nlargest(15, "P(Over)")[["player","market","line","avg","P(Over)","P(Under)","books_used"]], use_container_width=True)
with col2:
    st.markdown("#### Top 15 — Unders")
    st.dataframe(results.nlargest(15, "P(Under)")[["player","market","line","avg","P(Over)","P(Under)","books_used"]], use_container_width=True)

with st.expander("ℹ️ How this works / Debug"):
    st.write("Sport key:", SPORT_KEY[league])
    st.write("Tried markets:", MARKET_ALIAS.get(league))
    st.write("Sample of fetched board:")
    st.dataframe(board.head(20), use_container_width=True)
    st.write("Sample of stats:")
    st.dataframe(stats_df.head(20), use_container_width=True)
