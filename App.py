# pages/3_AllSports_PrizePicks_RealStats.py
import io
import gzip
import math
import time
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="All-Sports PrizePicks (Real Stats)", layout="wide")

PP_URL = "https://site.api.prizepicks.com/api/v1/projections?per_page=250&single_stat=true"

LEAGUE_OPTS = {
    "NFL": "nfl",
    "NBA": "nba",
    "MLB": "mlb",
    "NCAAF": "ncaaf",
}

# ------------ Small utilities
def http_get(url, headers=None, retries=3, backoff=1.0, timeout=12):
    err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers or {}, timeout=timeout)
            if r.status_code == 200:
                return r
            # 429/403 -> backoff
            if r.status_code in (403, 429, 522, 524):
                time.sleep(backoff * (2**i))
                continue
            # other non-200 -> stop
            r.raise_for_status()
            return r
        except Exception as e:
            err = e
            time.sleep(backoff * (2**i))
    raise RuntimeError(f"GET failed for {url}: {err}")

def pp_fetch_board():
    r = http_get(PP_URL, headers={"User-Agent": "Mozilla/5.0"})
    data = r.json()
    # Flatten PrizePicks payload to rows
    included = {x["id"]: x for x in data.get("included", [])}
    rows = []
    for proj in data.get("data", []):
        attr = proj.get("attributes", {})
        rel = proj.get("relationships", {})
        league = attr.get("league")
        line = attr.get("line_score")
        stat_type = attr.get("stat_type") or attr.get("stat_type_abbreviation")
        player_id = rel.get("new_player", {}).get("data", {}).get("id")
        player_name = None
        team = None
        if player_id and player_id in included:
            p = included[player_id]
            pa = p.get("attributes", {})
            player_name = pa.get("name")
            team = pa.get("team")
        if line is not None and stat_type and player_name:
            try:
                line = float(line)
            except:
                continue
            rows.append({
                "league": league,
                "player": player_name,
                "team": team,
                "market_raw": stat_type,
                "line": line,
            })
    return pd.DataFrame(rows)

def normalize_market(m: str, league_key: str) -> str:
    """Map prizepicks stat names to log fields per league."""
    if not m: return ""
    m = m.lower().strip()
    # Common aliases
    m = m.replace("passing", "pass").replace("rushing", "rush").replace("receiving", "rec")
    m = m.replace("yards", "yds").replace("points", "pts")
    m = m.replace("fantasy_score", "fantasy").replace("made_three_point_shots", "threes_made")

    # NFL
    if league_key == "nfl":
        if "pass_yd" in m or ("pass" in m and "yd" in m): return "pass_yds"
        if "rush_yd" in m: return "rush_yds"
        if "rec_yd" in m or ("rec" in m and "yd" in m): return "rec_yds"
        if "receptions" in m or "catches" in m: return "receptions"
        if "pass_td" in m: return "pass_tds"
        if "rush_td" in m: return "rush_tds"
        if "rec_td" in m: return "rec_tds"
        if "fantasy" in m: return "fantasy"

    # NBA
    if league_key == "nba":
        if "pts" in m: return "pts"
        if "reb" in m and "ast" in m: return "reb_ast"
        if "reb" in m: return "reb"
        if "ast" in m: return "ast"
        if "threes" in m or "3pt" in m or "3pm" in m: return "threes_made"
        if "pra" in m or ("pts" in m and "reb" in m and "ast" in m): return "pra"
        if "fantasy" in m: return "fantasy"

    # MLB
    if league_key == "mlb":
        if "strikeout" in m or "k" == m: return "pitcher_strikeouts"
        if "hits_runs_rbis" in m or "hrr" in m: return "hrr"
        if "total_bases" in m: return "total_bases"
        if "fantasy" in m: return "fantasy"

    # NCAAF similar to NFL
    if league_key == "ncaaf":
        if "pass" in m and "yd" in m: return "pass_yds"
        if "rush" in m and "yd" in m: return "rush_yds"
        if "rec" in m and "yd" in m: return "rec_yds"
        if "receptions" in m: return "receptions"
        if "fantasy" in m: return "fantasy"

    return m

# ------------ Real stats fetchers (no keys except CFBD)
def nba_game_logs(player_name: str, season: int, last_n=10):
    # balldontlie: search player -> id -> season stats per game
    try:
        # find player
        r = http_get(f"https://www.balldontlie.io/api/v1/players?search={requests.utils.quote(player_name)}")
        res = r.json()
        if not res.get("data"): return pd.DataFrame()
        pid = res["data"][0]["id"]
        logs = []
        page = 1
        while True:
            rr = http_get(
                f"https://www.balldontlie.io/api/v1/stats?player_ids[]={pid}&seasons[]={season}&per_page=100&page={page}"
            )
            js = rr.json()
            for it in js.get("data", []):
                s = it.get("stats") or it
                logs.append({
                    "pts": s.get("pts", 0),
                    "reb": s.get("reb", 0),
                    "ast": s.get("ast", 0),
                    "threes_made": s.get("fg3m", 0),
                })
            if page >= js.get("meta", {}).get("total_pages", 1): break
            page += 1
            if len(logs) >= 40: break
        df = pd.DataFrame(logs).tail(last_n)
        if df.empty: return df
        df["pra"] = df["pts"] + df["reb"] + df["ast"]
        return df
    except Exception:
        return pd.DataFrame()

def mlb_game_logs(player_name: str, season: int, last_n=10):
    # MLB Stats API needs a player id. Use a light search endpoint.
    try:
        sr = http_get(f"https://statsapi.mlb.com/api/v1/search?query={requests.utils.quote(player_name)}")
        j = sr.json()
        people = j.get("people", [])
        if not people: return pd.DataFrame()
        pid = people[0]["id"]
        # game logs
        gr = http_get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&season={season}")
        gjs = gr.json()
        splits = gjs.get("stats", [{}])[0].get("splits", [])
        rows = []
        for s in splits[-60:]:
            stat = s.get("stat", {})
            rows.append({
                "pitcher_strikeouts": float(stat.get("strikeOuts", 0)),
                "hrr": float(stat.get("hits", 0)) + float(stat.get("runs", 0)) + float(stat.get("rbi", 0)),
                "total_bases": float(stat.get("totalBases", 0)),
            })
        return pd.DataFrame(rows).tail(last_n)
    except Exception:
        return pd.DataFrame()

def nfl_game_logs(player_name: str, season: int, last_n=8):
    # nflfastR player-week logs (CSV.gz on GitHub)
    # We’ll use weekly rushing/receiving/passing summaries from "player_stats" file.
    try:
        url = f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/player_stats/player_stats_{season}.csv.gz"
        r = http_get(url)
        buf = io.BytesIO(r.content)
        with gzip.open(buf) as f:
            df = pd.read_csv(f, low_memory=False)
        # Simple name match; you could add a fuzzy step if needed.
        sub = df[df["player_name"].str.lower() == player_name.lower()].copy()
        if sub.empty:  # try contains (can pull wrong person on duplicates sometimes)
            sub = df[df["player_name"].str.lower().str.contains(player_name.lower(), na=False)]
        if sub.empty:
            return pd.DataFrame()
        # Keep last_n weeks
        sub = sub.sort_values(["season", "week"]).tail(last_n)
        # Map useful columns (fillna 0)
        out = pd.DataFrame({
            "pass_yds": sub.get("passing_yards", 0).fillna(0),
            "rush_yds": sub.get("rushing_yards", 0).fillna(0),
            "rec_yds": sub.get("receiving_yards", 0).fillna(0),
            "receptions": sub.get("receptions", 0).fillna(0),
            "pass_tds": sub.get("passing_tds", 0).fillna(0),
            "rush_tds": sub.get("rushing_tds", 0).fillna(0),
            "rec_tds": sub.get("receiving_tds", 0).fillna(0),
        })
        return out
    except Exception:
        return pd.DataFrame()

def ncaaf_game_logs(player_name: str, season: int, last_n=6):
    # CollegeFootballData — requires CFBD_API_KEY in st.secrets
    key = st.secrets.get("CFBD_API_KEY", None)
    if not key:
        return pd.DataFrame()
    try:
        # player search
        hdr = {"Authorization": f"Bearer {key}"}
        # This endpoint returns season stats by player; for per-game, we’ll use games/players
        pr = http_get(f"https://api.collegefootballdata.com/player/search?searchTerm={requests.utils.quote(player_name)}", headers=hdr)
        players = pr.json()
        if not players:
            return pd.DataFrame()
        pid = players[0].get("id")
        if not pid: return pd.DataFrame()
        # game logs
        gr = http_get(f"https://api.collegefootballdata.com/player/game?year={season}&playerId={pid}", headers=hdr)
        glogs = gr.json()
        rows = []
        for g in glogs[-20:]:
            rows.append({
                "pass_yds": float(g.get("passingYards") or 0),
                "rush_yds": float(g.get("rushingYards") or 0),
                "rec_yds": float(g.get("receivingYards") or 0),
                "receptions": float(g.get("receptions") or 0),
            })
        return pd.DataFrame(rows).tail(last_n)
    except Exception:
        return pd.DataFrame()

# ------------ Per-league adaptor
def fetch_logs_for_market(league_key: str, player: str, season: int, market: str) -> pd.Series | None:
    """Return a pandas Series of last-N values for the market, or None."""
    if league_key == "nba":
        df = nba_game_logs(player, season, last_n=12)
    elif league_key == "mlb":
        df = mlb_game_logs(player, season, last_n=10)
    elif league_key == "nfl":
        df = nfl_game_logs(player, season, last_n=8)
    elif league_key == "ncaaf":
        df = ncaaf_game_logs(player, season, last_n=6)
    else:
        return None
    if df is None or df.empty:
        return None
    if market not in df.columns:
        return None
    s = pd.to_numeric(df[market], errors="coerce").dropna()
    return s if not s.empty else None

# Conservative market SDs for fallback if a series has near-zero variance
DEFAULT_SD = {
    # NFL/NCAAF
    "pass_yds": 45.0, "rush_yds": 28.0, "rec_yds": 25.0, "receptions": 1.8,
    "pass_tds": 0.9, "rush_tds": 0.7, "rec_tds": 0.7,
    # NBA
    "pts": 6.5, "reb": 3.2, "ast": 3.0, "threes_made": 1.6, "pra": 9.0,
    # MLB
    "pitcher_strikeouts": 1.9, "hrr": 1.6, "total_bases": 1.4,
    # generic
    "fantasy": 7.5,
}

def prob_over_under_from_series(line: float, series: pd.Series, market: str):
    # shrink the mean slightly toward line to avoid overconfidence on tiny samples
    if series is None or len(series) == 0:
        mu = line
        sd = DEFAULT_SD.get(market, 3.0)
    else:
        mu_raw = float(series.mean())
        # shrinkage: 70% series mean + 30% line
        mu = 0.7 * mu_raw + 0.3 * line
        sd = float(series.std(ddof=1)) if len(series) >= 2 else 0.0
        if not np.isfinite(sd) or sd < 0.35:
            sd = DEFAULT_SD.get(market, 3.0)
    # Normal tail probabilities
    z = (line - mu) / sd if sd > 0 else 0.0
    p_over = 1.0 - norm.cdf((line + 1e-9 - mu) / sd) if sd > 0 else 0.5
    p_under = 1.0 - p_over
    return mu, sd, p_over, p_under

# ------------ UI
st.title("📊 All-Sports PrizePicks Simulator (Real player logs when possible)")

col1, col2 = st.columns([1,1])
with col1:
    league_ui = st.selectbox("League", list(LEAGUE_OPTS.keys()), index=0)
with col2:
    season = st.number_input("Season", min_value=2020, max_value=2030, value=2025, step=1)

league_key = LEAGUE_OPTS[league_ui]

with st.expander("How this works", expanded=False):
    st.markdown("""
- We pull the current PrizePicks board and filter **strictly** to the chosen league.
- For each projection row, we fetch **recent game logs** from a public stats source and compute a mean & stdev.
- We apply small **shrinkage toward the line** and a **floor on stdev** to avoid unrealistic 0% / 100% results from tiny samples.
- If logs aren't available or rate-limited, we fall back to a neutral model so the app still returns probabilities.
""")

# Fetch PP and filter
try:
    board = pp_fetch_board()
except Exception as e:
    st.error(f"PrizePicks fetch failed: {e}")
    st.stop()

board = board[board["league"].str.lower() == league_key].copy()
if board.empty:
    st.warning("No markets found for this league right now.")
    st.stop()

# normalize markets
board["market"] = board.apply(lambda r: normalize_market(str(r["market_raw"]), league_key), axis=1)

# Compute probabilities with real logs when available
rows = []
pbar = st.progress(0)
for i, rec in enumerate(board.itertuples(index=False)):
    player = rec.player
    mkt = rec.market
    line = rec.line

    series = fetch_logs_for_market(league_key, player, season, mkt)
    mu, sd, p_over, p_under = prob_over_under_from_series(line, series, mkt)

    rows.append({
        "player": player,
        "team": rec.team,
        "market": mkt,
        "line": line,
        "avg": round(mu, 2),
        "sd": round(sd, 2),
        "P(Over)": round(100*p_over, 1),
        "P(Under)": round(100*p_under, 1),
        "games_used": 0 if series is None else int(series.shape[0]),
        "used_real_logs": bool(series is not None and series.shape[0] > 0),
    })
    if (i+1) % max(1, len(board)//50) == 0:
        pbar.progress((i+1)/len(board))
pbar.empty()

df_out = pd.DataFrame(rows)

# Put better columns in front and sort by edge
df_out["edge_over"] = (df_out["P(Over)"] - 50.0).abs()
df_out = df_out.sort_values(["used_real_logs","edge_over"], ascending=[False, False]).drop(columns=["edge_over"])

st.subheader("Simulated probabilities (real logs when available)")
st.caption("If a row says used_real_logs=False, we fell back to a conservative neutral model.")
st.dataframe(df_out, use_container_width=True)

# Download
csv = df_out.to_csv(index=False)
st.download_button("Download CSV", data=csv, file_name=f"{league_key}_pp_sim_{season}.csv", mime="text/csv")
