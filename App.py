# streamlit_app.py
import time, json, random
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

st.set_page_config(page_title="All-Sports PrizePicks Simulator", layout="wide")
st.title("📊 All-Sports PrizePicks Simulator (Real stats + Auto Simulation)")

# ---------------- UI ----------------
league = st.selectbox("League", ["NFL", "NBA", "MLB", "Soccer"])
season_default = {"NFL": 2024, "NBA": 2024, "MLB": 2024, "Soccer": 2024}[league]
season = st.number_input("Season", 2018, 2025, value=season_default, step=1)

st.caption(
    "We fetch the PrizePicks board → filter to the chosen league → fetch real per-game "
    "averages → fuzzy-match players/stat → simulate with a conservative normal model."
)

# ---------------- Helpers ----------------
def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").strip().lower()

def conservative_sd(avg, minimum=0.75, frac=0.30):
    if pd.isna(avg): return 1.25
    if avg <= 0:     return 1.0
    sd = max(frac * float(avg), minimum)
    return max(sd, 0.5)

def simulate_prob(avg, line):
    sd = conservative_sd(avg)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over * 100, 2), round(p_under * 100, 2), round(sd, 3)

# ---------------- PrizePicks fetch (robust) ----------------
PP_ENDPOINTS = [
    "https://api.prizepicks.com/projections",
    "https://site.api.prizepicks.com/api/v1/projections",
]

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.prizepicks.com",
    "Referer": "https://www.prizepicks.com/",
}

def league_param(league: str) -> str:
    return {"NFL": "nfl", "NBA": "nba", "MLB": "mlb", "Soccer": "soc"}[league]

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=8))
def _get(url, params):
    r = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=15)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
    return r

@st.cache_data(ttl=180, show_spinner=False)
def fetch_prizepicks_board(league: str) -> pd.DataFrame:
    params = {"per_page": 250, "single_stat": "true", "league": league_param(league)}
    last_err = None
    for base in PP_ENDPOINTS:
        try:
            r = _get(base, params=params)
            data = r.json()
            df = pd.json_normalize(data.get("data", []))
            # expected keys (schema can shift; we guard below)
            cols = {
                "attributes.league": "league",
                "attributes.display_stat": "stat",
                "attributes.stat_type": "stat_type",
                "attributes.line_score": "line",
                "attributes.player_name": "player",
                "attributes.team": "pp_team",
            }
            keep = [c for c in cols if c in df.columns]
            if not keep:
                raise ValueError("Unexpected PrizePicks schema.")
            out = df[keep].rename(columns=cols)
            out["line"] = pd.to_numeric(out["line"], errors="coerce")
            return out.dropna(subset=["line"]).reset_index(drop=True)
        except Exception as e:
            last_err = e
            time.sleep(0.5 + random.random())
            continue
    raise RuntimeError(f"PrizePicks fetch failed: {last_err}")

# -------------- Real stats providers --------------
def load_nfl_stats(season: int):
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
        "Pass Yards": "pass_yards", "Passing Yards": "pass_yards",
        "Rush Yards": "rush_yards", "Rushing Yards": "rush_yards",
        "Receiving Yards": "rec_yards",
        "Receptions": "receptions",
        "Pass TDs": "pass_tds", "Rush TDs": "rush_tds", "Receiving TDs": "rec_tds",
        "Interceptions": "ints",
        "Pass+Rush Yds": ["pass_yards", "rush_yards"],
        "Rush+Rec Yds": ["rush_yards", "rec_yards"],
        "Pass+Rush+Rec Yds": ["pass_yards", "rush_yards", "rec_yards"],
    }
    return out, mapping

def load_nba_stats(season: int):
    st.info("Loading NBA season averages (balldontlie)…")
    rows, page = [], 1
    while True:
        url = f"https://www.balldontlie.io/api/v1/season_averages?season={season}&per_page=100&page={page}"
        r = requests.get(url, timeout=15)
        js = r.json(); data = js.get("data", [])
        if not data: break
        rows.extend(data); page += 1
        if page > 40: break
    df = pd.DataFrame(rows)
    if df.empty: raise ValueError("No NBA data returned.")
    out = pd.DataFrame({
        "player_id": df["player_id"].astype(int),
        "points": df.get("pts"), "assists": df.get("ast"),
        "rebounds": df.get("reb"), "threes": df.get("fg3m"),
    })

    # map ids -> names
    ids, id_to_name = list(out["player_id"].unique()), {}
    for chunk in [ids[i:i+50] for i in range(0, len(ids), 50)]:
        url = "https://www.balldontlie.io/api/v1/players?per_page=100&" + "&".join([f"ids[]={x}" for x in chunk])
        rr = requests.get(url, timeout=15)
        for p in rr.json().get("data", []):
            id_to_name[int(p["id"])] = f"{p['first_name']} {p['last_name']}"
    out["player"] = out["player_id"].map(id_to_name)
    out = out.drop(columns=["player_id"]).dropna(subset=["player"])
    mapping = {
        "Points": "points", "Assists": "assists", "Rebounds": "rebounds", "3-PT Made": "threes",
        "Pts+Reb+Ast": ["points","rebounds","assists"], "Pts+Reb": ["points","rebounds"],
        "Pts+Ast": ["points","assists"], "Reb+Ast": ["rebounds","assists"],
    }
    return out, mapping

def load_mlb_stats(season: int):
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
        bat_out[c] = bat_out[c] / bat_out["pa"] * 4.2  # crude per-game proxy
    pit_out = pd.DataFrame({
        "player": pit.get("Name"),
        "pitch_strikeouts": pit.get("SO"),
        "innings": pit.get("IP")
    }).dropna(subset=["player"])
    out = pd.merge(bat_out.drop(columns=["pa"]), pit_out, on="player", how="outer")
    mapping = {
        "Hitter Fantasy Score": ["hits","hr","rbi","sb"],
        "Hits": "hits", "Home Runs": "hr", "RBIs": "rbi", "Stolen Bases": "sb",
        "Pitcher Strikeouts": "pitch_strikeouts",
    }
    return out, mapping

def load_soccer_stats(season: int):
    st.info("Loading Soccer player stats (FBref via soccerdata)…")
    import soccerdata as sd
    fb_season = season if season <= 2024 else 2024  # FBref key is first season year
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
        "Goals": "goals", "Shots": "shots", "Shots On Goal": "sog", "Assists": "assists",
        "Passes Created": "key_passes", "Shots+SOG": ["shots","sog"], "G+A": ["goals","assists"],
    }
    return out, mapping

def provider_for(league: str):
    return {"NFL": load_nfl_stats, "NBA": load_nba_stats, "MLB": load_mlb_stats, "Soccer": load_soccer_stats}[league]

def best_name_match(name, candidates, score_cut=82):
    name_c = clean_name(name)
    best, best_score = None, -1
    for c in candidates:
        s = fuzz.token_sort_ratio(name_c, clean_name(c))
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= score_cut else None

def value_from_mapping(row_stats: pd.Series, stat_map):
    if isinstance(stat_map, list):
        vals = [row_stats.get(c) for c in stat_map if c in row_stats.index]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan
    return row_stats.get(stat_map)

# -------------- Fetch board (with fallback) --------------
with st.expander("Fetch PrizePicks board (debug)"):
    st.write("Endpoints tried:", PP_ENDPOINTS)
    st.write("Params:", {"per_page": 250, "single_stat": True, "league": league_param(league)})

board = None
try:
    board = fetch_prizepicks_board(league)
except Exception as e:
    st.error(f"PrizePicks fetch/parse error: {e}")

if board is None or board.empty:
    st.warning("No board from PrizePicks. Upload a board CSV/JSON (columns: player, stat, line, league).")
    up = st.file_uploader("Upload PrizePicks board fallback", type=["csv","json"])
    if up is None:
        st.stop()
    if up.name.lower().endswith(".csv"):
        board = pd.read_csv(up)
    else:
        data = json.load(up)
        if isinstance(data, dict) and "data" in data:
            df = pd.json_normalize(data["data"])
            cols = {
                "attributes.league": "league",
                "attributes.display_stat": "stat",
                "attributes.line_score": "line",
                "attributes.player_name": "player",
                "attributes.team": "pp_team",
            }
            keep = [c for c in cols if c in df.columns]
            board = df[keep].rename(columns=cols)
        else:
            board = pd.DataFrame(data)
    board["line"] = pd.to_numeric(board["line"], errors="coerce")
    board = board.dropna(subset=["line"])
    if "league" in board.columns:
        board = board[board["league"].astype(str).str.contains(league, case=False, na=False)]

if board.empty:
    st.error("Board is empty after parsing.")
    st.stop()

st.success(f"Loaded {len(board)} board rows for {league}.")

# -------------- Real stats + mapping --------------
try:
    stats_df, market_map = provider_for(league)(season)
except Exception as e:
    st.error(f"Could not load {league} stats: {e}")
    st.stop()

stats_df = stats_df.dropna(subset=["player"]).copy()
players_set = list(stats_df["player"].unique())

# -------------- Simulate --------------
rows = []
prog = st.progress(0)
for i, r in board.iterrows():
    prog.progress((i+1)/len(board))
    pp_player, pp_stat, pp_line = r.get("player"), r.get("stat"), r.get("line")
    if pd.isna(pp_player) or pd.isna(pp_stat) or pd.isna(pp_line):
        continue

    stat_map = market_map.get(pp_stat)
    if stat_map is None:
        for k in market_map.keys():
            if fuzz.token_set_ratio(str(pp_stat).lower(), k.lower()) >= 90:
                stat_map = market_map[k]; break
    if stat_map is None:
        continue

    match = best_name_match(pp_player, players_set, score_cut=82)
    if match is None:
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
        "pp_player": pp_player,
        "team": row_stats.get("team"),
        "market": pp_stat,
        "line": round(line_val, 3),
        "avg": round(float(avg_val), 3),
        "model_sd": used_sd,
        "P(Over)": p_over,
        "P(Under)": p_under,
    })

prog.empty()
results = pd.DataFrame(rows).sort_values("P(Over)", ascending=False)

if results.empty:
    st.warning("Nothing matched (unsupported markets or unmatched players).")
    st.stop()

st.subheader("Simulated edges (conservative normal model)")
st.caption("Probabilities are based on **real per-game averages**. SD is conservative to avoid 0%/100% artifacts.")
st.dataframe(results, use_container_width=True)

st.download_button(
    "Download CSV",
    results.to_csv(index=False).encode("utf-8"),
    file_name=f"{league.lower()}_{season}_prizepicks_sim.csv",
    mime="text/csv",
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Top 10 Overs")
    st.dataframe(results.nlargest(10, "P(Over)")[["player","market","line","avg","P(Over)","P(Under)"]],
                 use_container_width=True)
with col2:
    st.markdown("#### Top 10 Unders")
    st.dataframe(results.nlargest(10, "P(Under)")[["player","market","line","avg","P(Over)","P(Under)"]],
                 use_container_width=True)
