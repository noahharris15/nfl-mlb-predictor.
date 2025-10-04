# app.py  —  ONE SCRIPT, multi-section via sidebar
import time, json, random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz
from scipy.stats import norm
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# ---------------------------- App chrome ----------------------------
st.set_page_config(page_title="Player Prop Simulator (Odds API + Real Stats)", layout="wide")
st.title("📈 Player Prop Simulator (Odds API + Real Stats)")

st.caption(
    "We fetch **live player lines** from **The Odds API**, auto-load **real per-game player "
    "averages**, fuzzy-match names & market types, and estimate **P(Over/Under)** with a "
    "conservative normal model."
)

# ---------------------------- Utilities ----------------------------
DEFAULT_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]

def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").replace("'", "").strip().lower()

def best_name_match(name: str, candidates: List[str], score_cut: int = 82) -> Optional[str]:
    name_c = clean_name(name)
    best, best_score = None, -1
    for c in candidates:
        s = fuzz.token_sort_ratio(name_c, clean_name(c))
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= score_cut else None

def conservative_sd(avg: float, minimum: float = 0.75, frac: float = 0.30) -> float:
    if pd.isna(avg): return 1.25
    if avg <= 0:     return 1.0
    return max(max(frac * float(avg), minimum), 0.5)

def simulate_prob(avg: float, line: float) -> Tuple[float, float, float]:
    sd = conservative_sd(avg)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over * 100, 2), round(p_under * 100, 2), round(sd, 3)

def _markets_for_league(league: str) -> Dict[str, str]:
    # friendly label -> Odds API market code (league-specific)
    if league == "americanfootball_nfl":
        return {
            "Pass Yards": "player_pass_yds",
            "Rush Yards": "player_rush_yds",
            "Receiving Yards": "player_reception_yds",
            "Receptions": "player_receptions",
            "Rush+Rec Yds": "player_rush_rec_yds",
            "Pass+Rush Yds": "player_pass_rush_yds",
            "Rush Attempts": "player_rush_att",
            "Anytime TD": "player_td_anytime",
        }
    if league == "americanfootball_ncaaf":
        return {
            "Rush Yards": "player_rush_yds",
            "Receiving Yards": "player_reception_yds",
            "Receptions": "player_receptions",
            "Rush+Rec Yds": "player_rush_rec_yds",
            "Anytime TD": "player_td_anytime",
        }
    if league == "baseball_mlb":
        return {
            "Hits": "player_hits",
            "Home Runs": "player_home_runs",
            "RBIs": "player_rbis",
            "Total Bases": "player_total_bases",
            "Pitcher Strikeouts": "player_strikeouts",
        }
    if league == "soccer_epl":
        return {
            "Goals": "player_goals",
            "Shots": "player_shots",
            "Shots On Target": "player_shots_on_target",
            "Assists": "player_assists",
        }
    return {}

def friendly_to_market_code(friendly: str, league_key: str) -> Optional[str]:
    m = _markets_for_league(league_key)
    if friendly in m:
        return m[friendly]
    best, best_score = None, -1
    for k, v in m.items():
        s = fuzz.token_set_ratio(friendly.lower(), k.lower())
        if s > best_score:
            best, best_score = v, s
    return best

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=6))
def _odds_get(url, params):
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:400]}")
    return r

@st.cache_data(ttl=120, show_spinner=False)
def fetch_odds_player_props(
    api_key: str,
    sport_key: str,
    markets: List[str],
    bookmakers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return df: player, market_code, line (avg across chosen books)."""
    base = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "oddsFormat": "decimal",
        "markets": ",".join(markets),
        "includeLinks": "false",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    r = _odds_get(base, params)
    events = r.json()

    rows = []
    for ev in events or []:
        for bk in ev.get("bookmakers", []):
            bk_key = bk.get("key")
            for m in bk.get("markets", []):
                m_key = m.get("key")
                for out in m.get("outcomes", []):
                    player_name = out.get("description")
                    line = out.get("point")
                    if player_name is None or line is None:
                        continue
                    rows.append({
                        "player": player_name,
                        "market_code": m_key,
                        "bookmaker": bk_key,
                        "line": float(line),
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.groupby(["player", "market_code"], as_index=False)["line"].mean()

# ---------------------------- Real-stat loaders ----------------------------
def load_nfl_stats(season: int) -> pd.DataFrame:
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
        "pass_rush_yds": (df.get("passing_yards")+df.get("rushing_yards"))/g if "games" in df else (df.get("passing_yards")+df.get("rushing_yards")),
        "rush_rec_yds": (df.get("rushing_yards")+df.get("receiving_yards"))/g if "games" in df else (df.get("rushing_yards")+df.get("receiving_yards")),
    }).dropna(subset=["player"])
    return out

def load_ncaaf_stats(season: int, cfbd_key: Optional[str]) -> Optional[pd.DataFrame]:
    if not cfbd_key:
        return None
    import cfbd
    configuration = cfbd.Configuration()
    configuration.api_key["Authorization"] = cfbd_key
    configuration.api_key_prefix["Authorization"] = "Bearer"
    api = cfbd.PlayersApi(cfbd.ApiClient(configuration))

    rush = api.get_player_season_rushing_stats(year=season)
    rec  = api.get_player_season_receiving_stats(year=season)
    rush_df = pd.DataFrame([{
        "player": f"{r.player.first_name} {r.player.last_name}".strip(),
        "team": r.team,
        "games": r.games if r.games and r.games>0 else np.nan,
        "rush_yards": r.yards
    } for r in rush])
    rec_df = pd.DataFrame([{
        "player": f"{r.player.first_name} {r.player.last_name}".strip(),
        "team": r.team,
        "games": r.games if r.games and r.games>0 else np.nan,
        "rec_yards": r.yards,
        "receptions": r.receptions
    } for r in rec])
    stats = pd.merge(rush_df, rec_df, on=["player","team","games"], how="outer")
    stats["rush_yards"] = stats["rush_yards"] / stats["games"]
    stats["rec_yards"]  = stats["rec_yards"]  / stats["games"]
    stats["receptions"] = stats["receptions"] / stats["games"]
    stats["rush_rec_yds"] = stats["rush_yards"].fillna(0) + stats["rec_yards"].fillna(0)
    return stats.dropna(subset=["player"])

def load_mlb_stats(season: int) -> pd.DataFrame:
    from pybaseball import batting_stats, pitching_stats
    bat = batting_stats(season); pit = pitching_stats(season)
    bat_df = pd.DataFrame({
        "player": bat.get("Name"),
        "games": bat.get("G"),
        "H": bat.get("H"),
        "HR": bat.get("HR"),
        "RBI": bat.get("RBI"),
        "TB": bat.get("TB"),
    }).dropna(subset=["player"])
    bat_df["games"] = bat_df["games"].replace(0, np.nan)
    for col,new in [("H","hits"),("HR","hr"),("RBI","rbi"),("TB","tb")]:
        bat_df[new] = bat_df[col] / bat_df["games"]

    pit_df = pd.DataFrame({
        "player": pit.get("Name"),
        "G": pit.get("G"),
        "SO": pit.get("SO"),
    }).dropna(subset=["player"])
    pit_df["G"] = pit_df["G"].replace(0, np.nan)
    pit_df["pitcher_strikeouts"] = pit_df["SO"] / pit_df["G"]

    return pd.merge(
        bat_df[["player","hits","hr","rbi","tb"]],
        pit_df[["player","pitcher_strikeouts"]],
        on="player", how="outer"
    ).dropna(subset=["player"])

def load_soccer_stats(season_first_year: int) -> pd.DataFrame:
    import soccerdata as sd
    fb = sd.FBref(leagues="ENG-Premier League", seasons=season_first_year)
    df = fb.read_player_season_stats(stat_type="standard").reset_index()
    name_col = "player" if "player" in df.columns else "Player"
    stats_df = pd.DataFrame({
        "player": df[name_col],
        "games": df["MP"] if "MP" in df.columns else np.nan,
        "goals": df["Gls"] if "Gls" in df.columns else np.nan,
        "shots": df["Sh"] if "Sh" in df.columns else np.nan,
        "sog": df["SoT"] if "SoT" in df.columns else np.nan,
        "assists": df["Ast"] if "Ast" in df.columns else np.nan,
    }).dropna(subset=["player"])
    stats_df["games"] = stats_df["games"].replace(0, np.nan)
    for col in ["goals","shots","sog","assists"]:
        stats_df[col] = stats_df[col] / stats_df["games"]
    return stats_df

# ---------------------------- Page renderers ----------------------------
def render_sim_page(
    league_label: str,
    sport_key: str,
    season_default: int,
    stats_loader,                      # function(season, *optional_key) -> DataFrame
    market_map_to_statcols: Dict[str, str],
    score_cut: int = 82,
    extra_key_label: Optional[str] = None,
):
    st.subheader(league_label)

    # Keys
    odds_key = st.secrets.get("ODDS_API_KEY", "")
    odds_key = st.text_input("The Odds API key", value=odds_key or "", type="password")

    # Season & books
    season = st.number_input("Season", min_value=2018, max_value=2025, value=season_default, step=1)
    books = st.multiselect("Bookmakers (optional – average across selected; leave empty for all)",
                           DEFAULT_BOOKS, default=["draftkings","fanduel","betmgm","caesars"])

    # Markets UI
    friendly = list(_markets_for_league(sport_key).keys())
    default_pick = friendly[:4] if len(friendly) >= 4 else friendly
    chosen = st.multiselect("Markets to pull", friendly, default=default_pick)

    # Optional extra key (CFBD for NCAAF)
    extra_key_val = None
    if extra_key_label:
        extra_key_val = st.secrets.get(extra_key_label, "")
        extra_key_val = st.text_input(extra_key_label, value=extra_key_val or "", type="password")

    if not odds_key:
        st.warning("Enter your Odds API key to pull live lines.")
        return

    # Fetch Board
    codes = [friendly_to_market_code(x, sport_key) for x in chosen]
    codes = [c for c in codes if c]
    try:
        board = fetch_odds_player_props(odds_key, sport_key, codes, books)
    except Exception as e:
        st.error(f"Odds API error: {e}")
        return

    if board.empty:
        st.warning("No player lines received. Try fewer markets or bookmakers.")
        return

    # Load Stats
    try:
        if extra_key_label:
            stats_df = stats_loader(season, extra_key_val)
            if stats_df is None:
                st.warning("Add the extra API key above to compute real-player averages.")
                st.dataframe(board, use_container_width=True)
                return
        else:
            stats_df = stats_loader(season)
    except Exception as e:
        st.error(f"Could not load real stats: {e}")
        return

    if stats_df is None or stats_df.empty:
        st.warning("Stats loader returned no rows.")
        return

    players = list(stats_df["player"].unique())
    rows = []
    prog = st.progress(0)
    for i, r in board.iterrows():
        prog.progress((i+1)/len(board))
        stat_col = market_map_to_statcols.get(r["market_code"])
        if not stat_col:
            continue
        match = best_name_match(r["player"], players, score_cut)
        if not match:
            continue
        row = stats_df.loc[stats_df["player"] == match].iloc[0]
        avg_val = row.get(stat_col)
        if pd.isna(avg_val):
            continue
        line_val = float(r["line"])
        p_over, p_under, sd = simulate_prob(avg_val, line_val)
        rows.append({
            "player": match,
            "market": r["market_code"],
            "line": round(line_val,2),
            "avg": round(float(avg_val),2),
            "P(Over)": p_over,
            "P(Under)": p_under,
            "team": row.get("team") if "team" in row else None,
        })
    prog.empty()

    results = pd.DataFrame(rows).sort_values("P(Over)", ascending=False)
    if results.empty:
        st.warning("Nothing matched (names/markets didn’t align after fuzzy match).")
        return

    st.markdown("### Simulated edges")
    st.caption("Probabilities use a conservative normal model to avoid 0% / 100% artifacts.")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        f"{sport_key.replace('_','-')}_{season}_sim.csv",
        "text/csv",
    )

# ---------------------------- Sidebar "pages" ----------------------------
page = st.sidebar.radio("Pages", ["NFL", "College Football (NCAAF)", "MLB", "Soccer (EPL)"])

if page == "NFL":
    market_map = {
        "player_pass_yds": "pass_yards",
        "player_rush_yds": "rush_yards",
        "player_reception_yds": "rec_yards",
        "player_receptions": "receptions",
        "player_pass_rush_yds": "pass_rush_yds",
        "player_rush_rec_yds": "rush_rec_yds",
    }
    render_sim_page(
        league_label="🏈 NFL — Player Prop Simulator",
        sport_key="americanfootball_nfl",
        season_default=2025,
        stats_loader=load_nfl_stats,
        market_map_to_statcols=market_map,
        score_cut=82,
        extra_key_label=None,
    )

elif page == "College Football (NCAAF)":
    market_map = {
        "player_rush_yds": "rush_yards",
        "player_reception_yds": "rec_yards",
        "player_receptions": "receptions",
        "player_rush_rec_yds": "rush_rec_yds",
    }
    render_sim_page(
        league_label="🏈 College Football (NCAAF) — Player Prop Simulator",
        sport_key="americanfootball_ncaaf",
        season_default=2025,
        stats_loader=load_ncaaf_stats,
        market_map_to_statcols=market_map,
        score_cut=82,
        extra_key_label="CFBD_API_KEY",  # asks for/pulls from secrets
    )

elif page == "MLB":
    market_map = {
        "player_hits": "hits",
        "player_home_runs": "hr",
        "player_rbis": "rbi",
        "player_total_bases": "tb",
        "player_strikeouts": "pitcher_strikeouts",
    }
    render_sim_page(
        league_label="⚾ MLB — Player Prop Simulator",
        sport_key="baseball_mlb",
        season_default=2025,
        stats_loader=load_mlb_stats,
        market_map_to_statcols=market_map,
        score_cut=80,
        extra_key_label=None,
    )

else:  # Soccer
    market_map = {
        "player_goals": "goals",
        "player_shots": "shots",
        "player_shots_on_target": "sog",
        "player_assists": "assists",
    }
    render_sim_page(
        league_label="⚽ Soccer — Player Prop Simulator (EPL)",
        sport_key="soccer_epl",
        season_default=2024,   # FBref uses the first year (e.g., 2024 for 2024-25)
        stats_loader=load_soccer_stats,
        market_map_to_statcols=market_map,
        score_cut=88,
        extra_key_label=None,
    )
