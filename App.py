# streamlit_app.py
import os, re, time, json, math, random
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# --------------------------- Page ---------------------------
st.set_page_config(page_title="All-Sports OddsAPI Simulator", layout="wide")
st.title("📊 All-Sports Player Props (Odds API lines + Your CSV stats)")

# --------------------------- API Key ------------------------
def get_odds_api_key() -> str:
    # Preferred: secrets or env
    if "ODDS_API_KEY" in st.secrets:
        return st.secrets["ODDS_API_KEY"]
    k = os.getenv("ODDS_API_KEY")
    if k:
        return k
    # ⚠️ fallback for quick testing only — remove for production
    return "9ede18ca5b55fa2afc180a2b375367e2"

ODDS_API_KEY = get_odds_api_key()

# --------------------------- UI -----------------------------
colA, colB, colC = st.columns([1,1,1])
with colA:
    league = st.selectbox("League", ["NFL","NBA","MLB","Soccer"])
with colB:
    bookmaker_pref = st.text_input("Preferred bookmaker (optional)", value="draftkings").strip()
with colC:
    season = st.number_input("Season label (for your CSV)", 2018, 2026, value=2024, step=1)

st.caption(
    "• We pull the **current lines from Odds API** → "
    "you upload a **player stats CSV** → we **fuzzy-match** names and map markets → "
    "simulate P(Over/Under) with a conservative normal model (to avoid 0/100% artifacts)."
)

st.markdown("### 1) Upload your **player stats CSV**")
stats_file = st.file_uploader(
    "CSV must contain a `player` column and columns referenced in the market→columns map (below).",
    type=["csv"]
)

# ------------------- Market mappings per league -------------------
# Map Odds API market keys -> your CSV column(s)
# (Edit the right-hand side to match your CSV headers!)
LEAGUE_MARKET_TO_COLS = {
    "NFL": {
        "player_pass_yards":       "pass_yards",
        "player_rush_yards":       "rush_yards",
        "player_receiving_yards":  "rec_yards",
        "player_receptions":       "receptions",
        "player_pass_tds":         "pass_tds",
        "player_rush_tds":         "rush_tds",
        "player_receiving_tds":    "rec_tds",
        # combos (if your CSV has separate columns, we’ll sum them)
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
        # if you keep a custom fantasy score, add it here
        # "player_batter_fantasy_score": ["hits","hr","rbi","sb"],  # example proxy
    },
    "Soccer": {
        "player_goals":            "goals",
        "player_shots":            "shots",
        "player_shots_on_goal":    "sog",
        "player_assists":          "assists",
        "player_goals_plus_assists":["goals","assists"],
    },
}

# Odds API sport keys per league (pick the main one; add more if you want)
SPORT_KEY = {
    "NFL":   "americanfootball_nfl",
    "NBA":   "basketball_nba",
    "MLB":   "baseball_mlb",
    # Use EPL as default soccer board. You can add others: soccer_uefa_champs_league, soccer_spain_la_liga, etc.
    "Soccer":"soccer_epl",
}

# ------------------- Helpers -------------------
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

# ------------------- Odds API fetch -------------------
ODDS_BASE = "https://api.the-odds-api.com/v4/sports"

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=8))
def _get(url, params):
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
    return r

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def parse_player_from_outcome(out):
    # Try several shapes we see on player props
    for key in ("participant", "player", "description"):
        s = out.get(key)
        if isinstance(s, str) and s.strip():
            # strip trailing " Over"/" Under"
            m = re.split(r"\s+(?:Over|Under)\b", s, maxsplit=1)
            return m[0].strip()
    # Some books put it in 'name' when O/U are separate markets; ignore then
    return None

def extract_rows_from_book(book, market_key):
    rows = []
    for m in book.get("markets", []):
        if m.get("key") != market_key: 
            continue
        # We expect outcomes with 'Over' and 'Under'. The 'point' is the line.
        outcomes = m.get("outcomes", []) or []
        if not outcomes:
            continue
        # Try to detect the player name
        player_name = None
        for o in outcomes:
            player_name = parse_player_from_outcome(o) or player_name
        # If still None, sometimes books add 'player' at market level
        if not player_name:
            player_name = m.get("player") or m.get("description")

        # Get the common line (point) if present
        line = None
        for o in outcomes:
            if "point" in o and o["point"] is not None:
                line = o["point"]; break
        if player_name and line is not None:
            rows.append({"player": player_name, "line": float(line), "book": book.get("title")})
    return rows

@st.cache_data(ttl=60, show_spinner=True)
def fetch_odds_board(league: str, market_keys: list[str], bookmaker_pref: str|None):
    """Return a dataframe with columns: player, market, line, book."""
    sport = SPORT_KEY[league]
    all_rows = []
    # Odds API lets a comma list of markets, but keep chunks small to be safe
    for mk_chunk in chunked(market_keys, 6):
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",            # or 'us,eu,uk' etc.
            "markets": ",".join(mk_chunk),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        url = f"{ODDS_BASE}/{sport}/odds"
        r = _get(url, params=params)
        events = r.json() if r.content else []
        # events: list of {id, commence_time, home_team, away_team, bookmakers:[{key,title,markets:[...]}, ...]}
        for ev in events:
            books = ev.get("bookmakers", []) or []
            # Optional: sort to prefer a bookmaker
            if bookmaker_pref:
                books = sorted(books, key=lambda b: (0 if bookmaker_pref.lower() in (b.get("key","")+b.get("title","")).lower() else 1))
            for b in books:
                for mk in mk_chunk:
                    rows = extract_rows_from_book({"title": b.get("title"), "markets": b.get("markets", [])}, mk)
                    for rrow in rows:
                        all_rows.append({"market": mk, **rrow})
                # take only first (preferred) book per event to avoid dups
                break
        # be gentle
        time.sleep(0.25 + random.random()*0.25)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.dropna(subset=["player","line"]).reset_index(drop=True)
    return df

# ------------------- Load CSV -------------------
if stats_file is None:
    st.stop()

try:
    stats_df = pd.read_csv(stats_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if "player" not in stats_df.columns:
    st.error("Your CSV must have a 'player' column.")
    st.stop()

stats_df = stats_df.copy()
stats_df["player"] = stats_df["player"].astype(str)

# Show a peek so you can confirm columns
with st.expander("Preview of your CSV"):
    st.dataframe(stats_df.head(20), use_container_width=True)

# ------------------- Which markets to request -------------------
market_to_cols = LEAGUE_MARKET_TO_COLS[league]
req_markets = list(market_to_cols.keys())

st.markdown("### 2) Fetch **current lines** from Odds API")
with st.expander("Requested markets", expanded=False):
    st.write(req_markets)

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

# ------------------- Simulate -------------------
st.markdown("### 3) Simulate with your CSV averages")

players_set = list(stats_df["player"].unique())
rows = []
prog = st.progress(0)

# allow you to scale model SD a bit if you want
sd_min = st.slider("Minimum SD (conservative floor)", 0.3, 2.0, 0.75, 0.05)
sd_frac = st.slider("SD as fraction of average", 0.05, 0.80, 0.30, 0.05)

def conservative_sd_custom(avg, minimum=sd_min, frac=sd_frac):
    if pd.isna(avg): return max(1.25, minimum)
    if avg <= 0:     return max(1.0, minimum)
    return max(frac * float(avg), minimum)

def simulate_prob_custom(avg, line):
    sd = conservative_sd_custom(avg)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over * 100, 2), round(p_under * 100, 2), round(sd, 3)

board = board.sort_values(["market","player"]).reset_index(drop=True)
for i, r in board.iterrows():
    prog.progress((i+1)/len(board))
    market_key = r["market"]
    target_cols = market_to_cols.get(market_key)
    if not target_cols:
        continue

    player_board = str(r["player"])
    match = best_name_match(player_board, players_set, score_cut=86)
    if not match:
        continue

    row_stats = stats_df.loc[stats_df["player"] == match].iloc[0]
    avg_val = value_from_columns(row_stats, target_cols)
    if pd.isna(avg_val):
        continue

    line_val = float(r["line"])
    p_over, p_under, used_sd = simulate_prob_custom(avg_val, line_val)

    rows.append({
        "league": league,
        "player": match,
        "csv_player": player_board,         # as listed by the book
        "market_key": market_key,
        "book": r.get("book"),
        "line": round(line_val, 3),
        "avg": round(float(avg_val), 3),
        "model_sd": used_sd,
        "P(Over)": p_over,
        "P(Under)": p_under,
    })

prog.empty()
results = pd.DataFrame(rows)

if results.empty:
    st.warning("No matches between board and your CSV using current mapping.")
    st.stop()

# nicer market labels (optional)
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
results["market"] = results["market_key"].map(pretty_map).fillna(results["market_key"])

# sort by the bigger edge (distance from 50/50)
results["edge"] = (results["P(Over)"] - 50).abs()
results = results.sort_values(["edge","P(Over)"], ascending=[False, False]).reset_index(drop=True)

st.subheader("Simulated edges (using your CSV stats)")
st.caption("Probabilities use a conservative normal model to avoid 0%/100% artifacts.")
st.dataframe(results[["league","player","market","book","line","avg","model_sd","P(Over)","P(Under)"]],
             use_container_width=True)

st.download_button(
    "Download results (CSV)",
    results.to_csv(index=False).encode("utf-8"),
    file_name=f"{league.lower()}_{season}_oddsapi_sim.csv",
    mime="text/csv",
)

st.markdown("#### Top Overs")
st.dataframe(results.nlargest(12, "P(Over)")[["player","market","line","avg","P(Over)","P(Under)"]],
             use_container_width=True)
st.markdown("#### Top Unders")
st.dataframe(results.nlargest(12, "P(Under)")[["player","market","line","avg","P(Over)","P(Under)"]],
             use_container_width=True)

st.info(
    "If some players don’t match: tweak your CSV names or lower the fuzzy score, "
    "and adjust `LEAGUE_MARKET_TO_COLS` so the markets map to the correct CSV columns."
)
