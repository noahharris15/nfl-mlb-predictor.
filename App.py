# app.py — All-Sports Player Props (Odds API lines + Your CSV stats)
import os, re, time, math, json, random
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# ---------------- Page ----------------
st.set_page_config(page_title="All-Sports Player Props (Odds API + CSV)", layout="wide")
st.title("📊 All-Sports Player Props — Odds API lines + Your CSV stats")

# ---------------- League + season UI ----------------
league = st.selectbox("League", ["NFL", "NBA", "MLB", "Soccer"])
season_default = {"NFL": 2024, "NBA": 2024, "MLB": 2024, "Soccer": 2024}[league]
season = st.number_input("Season tag (for your CSV only)", 2018, 2026, value=season_default, step=1)

st.caption(
    "We fetch **player lines from Odds API**, you upload a **per-player stats CSV**, "
    "we fuzzy-match names + map the market to your columns, and simulate P(Over/Under) "
    "with a conservative normal model (no 0%/100% artifacts)."
)

# ---------------- Odds API key ----------------
def get_odds_api_key() -> str:
    if "ODDS_API_KEY" in st.secrets:
        return st.secrets["ODDS_API_KEY"]
    k = os.getenv("ODDS_API_KEY")
    if k:
        return k
    # fallback for quick start; for production, keep this in secrets/env only
    return "9ede18ca5b55fa2afc180a2b375367e2"

ODDS_API_KEY = get_odds_api_key()

# ---------------- Utilities ----------------
def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").strip().lower()

def conservative_sd(avg, minimum=0.75, frac=0.30):
    if pd.isna(avg): return 1.25
    if avg <= 0:     return max(1.0, minimum)
    return max(frac * float(avg), minimum)

def simulate_prob(avg, line, sd_min=0.75, sd_frac=0.30):
    sd = max(conservative_sd(avg, minimum=sd_min, frac=sd_frac), 0.5)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over * 100, 2), round(p_under * 100, 2), round(sd, 3)

def best_name_match(name, candidates, score_cut=86):
    name_c = clean_name(name)
    best, best_score = None, -1
    for c in candidates:
        s = fuzz.token_sort_ratio(name_c, clean_name(c))
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= score_cut else None

def value_from_columns(row_stats: pd.Series, cols):
    if isinstance(cols, list):
        vals = [row_stats.get(c) for c in cols if c in row_stats.index]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan
    return row_stats.get(cols)

# ---------------- Market mappings (edit right-hand side to match your CSV) ----------------
LEAGUE_MARKET_TO_COLS = {
    "NFL": {
        "player_pass_yards":       "pass_yards",
        "player_rush_yards":       "rush_yards",
        "player_receiving_yards":  "rec_yards",
        "player_receptions":       "receptions",
        "player_pass_tds":         "pass_tds",
        "player_rush_tds":         "rush_tds",
        "player_receiving_tds":    "rec_tds",
        "player_pass_plus_rush_yards": ["pass_yards","rush_yards"],
        "player_rush_plus_rec_yards":  ["rush_yards","rec_yards"],
    },
    "NBA": {
        "player_points":           "points",
        "player_assists":          "assists",
        "player_rebounds":         "rebounds",
        "player_three_points_made":"threes",
        "player_points_plus_rebounds_plus_assists": ["points","rebounds","assists"],
        "player_points_plus_assists":               ["points","assists"],
        "player_points_plus_rebounds":              ["points","rebounds"],
        "player_rebounds_plus_assists":             ["rebounds","assists"],
    },
    "MLB": {
        "player_hits":             "hits",
        "player_home_runs":        "hr",
        "player_rbis":             "rbi",
        "player_stolen_bases":     "sb",
        "player_pitcher_strikeouts":"pitch_strikeouts",
    },
    "Soccer": {
        "player_goals":            "goals",
        "player_shots":            "shots",
        "player_shots_on_goal":    "sog",
        "player_assists":          "assists",
        "player_goals_plus_assists":["goals","assists"],
    },
}

# Odds API sport key per league (primary feed)
SPORT_KEY = {
    "NFL":   "americanfootball_nfl",
    "NBA":   "basketball_nba",
    "MLB":   "baseball_mlb",
    "Soccer":"soccer_epl",  # change to other leagues if you want
}

# ---------------- Upload your CSV ----------------
st.markdown("### 1) Upload your **player stats CSV**")
stats_file = st.file_uploader(
    "CSV must have a 'player' column and columns referenced in the mapping below.",
    type=["csv"]
)
with st.expander("Current market → CSV columns mapping"):
    st.json(LEAGUE_MARKET_TO_COLS[league], expanded=False)

if stats_file is None:
    st.stop()

try:
    stats_df = pd.read_csv(stats_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if "player" not in stats_df.columns:
    st.error("Your CSV must include a 'player' column.")
    st.stop()

stats_df["player"] = stats_df["player"].astype(str)
with st.expander("Preview of your CSV"):
    st.dataframe(stats_df.head(20), use_container_width=True)

# ---------------- Odds API fetch ----------------
ODDS_BASE = "https://api.the-odds-api.com/v4/sports"

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=8))
def _get(url, params):
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:250]}")
    return r

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def parse_player_from_outcome(out):
    # Try to read a player name from various shapes
    for key in ("participant", "player", "description"):
        s = out.get(key)
        if isinstance(s, str) and s.strip():
            # remove trailing " Over/Under" if present
            parts = re.split(r"\s+(?:Over|Under)\b", s, maxsplit=1)
            return parts[0].strip()
    return None

def extract_rows_from_book(book, market_key):
    rows = []
    for m in book.get("markets", []) or []:
        if m.get("key") != market_key:
            continue
        outs = m.get("outcomes", []) or []
        if not outs:
            continue
        # player and common line (point)
        player_name, line = None, None
        for o in outs:
            player_name = parse_player_from_outcome(o) or player_name
            if "point" in o and o["point"] is not None:
                line = o["point"]
        if player_name and line is not None:
            rows.append({"player": player_name, "line": float(line), "book": book.get("title")})
    return rows

@st.cache_data(ttl=90, show_spinner=True)
def fetch_odds_board(league: str, market_keys: list[str], bookmaker_pref: str | None):
    sport = SPORT_KEY[league]
    all_rows = []
    for mk_chunk in chunked(market_keys, 6):
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": ",".join(mk_chunk),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        url = f"{ODDS_BASE}/{sport}/odds"
        r = _get(url, params=params)
        events = r.json() if r.content else []
        for ev in events:
            books = ev.get("bookmakers", []) or []
            if bookmaker_pref:
                # prefer a specific book if typed
                books = sorted(
                    books,
                    key=lambda b: (0 if bookmaker_pref.lower() in (b.get("key","")+b.get("title","")).lower() else 1)
                )
            # take the first (preferred) book per event to reduce duplicates
            for b in books[:1]:
                for mk in mk_chunk:
                    all_rows.extend(
                        [{"market": mk, **rr} for rr in extract_rows_from_book(
                            {"title": b.get("title"), "markets": b.get("markets", [])}, mk
                        )]
                    )
        time.sleep(0.25 + random.random()*0.25)
    df = pd.DataFrame(all_rows).dropna(subset=["player","line"]) if all_rows else pd.DataFrame()
    return df.reset_index(drop=True)

# Which markets we'll ask for (from your mapping)
market_to_cols = LEAGUE_MARKET_TO_COLS[league]
req_markets = list(market_to_cols.keys())

col1, col2 = st.columns([1,1])
with col1:
    bookmaker_pref = st.text_input("Preferred bookmaker (optional)", value="draftkings").strip()
with col2:
    st.write("Requested markets:", ", ".join(req_markets))

st.markdown("### 2) Fetch **current lines** from Odds API")
try:
    board = fetch_odds_board(league, req_markets, bookmaker_pref or None)
except Exception as e:
    st.error(f"Odds API fetch failed: {e}")
    st.stop()

if board.empty:
    st.warning("No player lines returned for the selected league/markets.")
    st.stop()

st.success(f"Fetched {len(board)} player lines from Odds API.")
with st.expander("Board (raw)"):
    st.dataframe(board, use_container_width=True)

# ---------------- Simulate with your CSV ----------------
st.markdown("### 3) Simulate using your CSV averages")
players_set = list(stats_df["player"].unique())

# model SD controls
sd_min = st.slider("Minimum SD (floor)", 0.3, 2.0, 0.75, 0.05)
sd_frac = st.slider("SD as fraction of average", 0.05, 0.80, 0.30, 0.05)

pretty_map = {
    # NFL
    "player_pass_yards":"Pass Yards", "player_rush_yards":"Rush Yards",
    "player_receiving_yards":"Receiving Yards", "player_receptions":"Receptions",
    "player_pass_tds":"Pass TDs","player_rush_tds":"Rush TDs","player_receiving_tds":"Receiving TDs",
    "player_pass_plus_rush_yards":"Pass+Rush Yds","player_rush_plus_rec_yards":"Rush+Rec Yds",
    # NBA
    "player_points":"Points","player_assists":"Assists","player_rebounds":"Rebounds",
    "player_three_points_made":"3PM","player_points_plus_rebounds_plus_assists":"PRA",
    "player_points_plus_assists":"P+A","player_points_plus_rebounds":"P+R","player_rebounds_plus_assists":"R+A",
    # MLB
    "player_hits":"Hits","player_home_runs":"Home Runs","player_rbis":"RBIs",
    "player_stolen_bases":"Stolen Bases","player_pitcher_strikeouts":"Pitcher Ks",
    # Soccer
    "player_goals":"Goals","player_shots":"Shots","player_shots_on_goal":"SOG",
    "player_assists":"Assists","player_goals_plus_assists":"G+A",
}

# do the matching + sim
rows = []
prog = st.progress(0)
board = board.sort_values(["market","player"]).reset_index(drop=True)

for i, r in board.iterrows():
    prog.progress((i+1)/len(board))
    mk = r["market"]
    cols = market_to_cols.get(mk)
    if not cols:
        continue

    book_player = str(r["player"])
    match = best_name_match(book_player, players_set, score_cut=86)
    if not match:
        continue

    row_stats = stats_df.loc[stats_df["player"] == match].iloc[0]
    avg_val = value_from_columns(row_stats, cols)
    if pd.isna(avg_val):
        continue

    line_val = float(r["line"])
    p_over, p_under, used_sd = simulate_prob(avg_val, line_val, sd_min, sd_frac)

    rows.append({
        "league": league,
        "player": match,              # matched to your CSV
        "board_player": book_player,  # as listed by book
        "market_key": mk,
        "market": pretty_map.get(mk, mk),
        "book": r.get("book"),
        "line": round(line_val, 3),
        "avg": round(float(avg_val), 3),
        "model_sd": used_sd,
        "P(Over)": p_over,
        "P(Under)": p_under,
        "edge": round(abs(p_over - 50.0), 2),
    })

prog.empty()
results = pd.DataFrame(rows)

if results.empty:
    st.warning("No matches between board and your CSV using current mapping.")
    st.stop()

results = results.sort_values(["edge","P(Over)"], ascending=[False, False]).reset_index(drop=True)

st.subheader("Simulated edges (conservative normal model)")
st.caption("Probabilities are based on your CSV averages, with a conservative SD to avoid 0%/100% artifacts.")
st.dataframe(results[["league","player","market","book","line","avg","model_sd","P(Over)","P(Under)"]],
             use_container_width=True)

st.download_button(
    "Download results (CSV)",
    results.to_csv(index=False).encode("utf-8"),
    file_name=f"{league.lower()}_{season}_oddsapi_sim.csv",
    mime="text/csv",
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Top Overs")
    st.dataframe(results.nlargest(12, "P(Over)")[["player","market","line","avg","P(Over)","P(Under)"]],
                 use_container_width=True)
with c2:
    st.markdown("#### Top Unders")
    st.dataframe(results.nlargest(12, "P(Under)")[["player","market","line","avg","P(Over)","P(Under)"]],
                 use_container_width=True)

st.info(
    "If something shows 50/50 everywhere or players don’t match: "
    "1) verify your CSV column names match the mapping above; "
    "2) lower the name-match threshold; 3) adjust SD sliders."
)
