# app.py  —  One-file Streamlit app for NFL/MLB player props using The Odds API
import os, math
import pandas as pd
import numpy as np
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz

# --------------------------- App + Sidebar ---------------------------
st.set_page_config(page_title="Player Props Simulator (Odds API + CSV)", layout="wide")
st.title("📊 Player Props Simulator (Odds API + Your CSV Stats)")

with st.sidebar:
    st.header("Settings")
    league = st.selectbox("League", ["NFL", "MLB"])
    api_key = st.text_input("Odds API key (or set env ODDS_API_KEY)", type="password") \
              or os.getenv("ODDS_API_KEY", "")

    # Markets allowed by The Odds API for each league (keep to valid keys)
    VALID_MARKETS = {
        "NFL": [
            "player_passing_yards",
            "player_rushing_yards",
            "player_receiving_yards",
            "player_receptions",
            "player_rush_and_receive_yards",
            "player_passing_touchdowns",
            "player_rushing_touchdowns",
            "player_receiving_touchdowns",
            "player_interceptions_thrown",
        ],
        "MLB": [
            "player_hits",
            "player_total_bases",
            "player_home_runs",
            "player_rbis",
            "player_runs_scored",
            "player_strikeouts",   # pitcher strikeouts
        ],
    }
    # Market -> column(s) expected in your CSV
    CSV_MAP = {
        "NFL": {
            "player_passing_yards": "passing_yards",
            "player_rushing_yards": "rushing_yards",
            "player_receiving_yards": "receiving_yards",
            "player_receptions": "receptions",
            "player_rush_and_receive_yards": ["rushing_yards", "receiving_yards"],
            "player_passing_touchdowns": "passing_tds",
            "player_rushing_touchdowns": "rushing_tds",
            "player_receiving_touchdowns": "receiving_tds",
            "player_interceptions_thrown": "interceptions_thrown",
        },
        "MLB": {
            "player_hits": "hits",
            "player_total_bases": "total_bases",
            "player_home_runs": "home_runs",
            "player_rbis": "rbis",
            "player_runs_scored": "runs",
            "player_strikeouts": "pitcher_strikeouts",
        },
    }

    # defaults per league
    defaults = {
        "NFL": ["player_passing_yards", "player_rushing_yards",
                "player_receiving_yards", "player_receptions"],
        "MLB": ["player_hits", "player_total_bases",
                "player_home_runs", "player_strikeouts"],
    }

    sel_markets = st.multiselect(
        "Markets to pull",
        options=VALID_MARKETS[league],
        default=[m for m in defaults[league] if m in VALID_MARKETS[league]],
    )

    bookmakers = st.text_input("Bookmakers (comma-separated)", "draftkings,betmgm,fanduel,caesars")

st.caption(
    "This app pulls **player lines from The Odds API**, then **simulates Over/Under** "
    "with a conservative normal model using **your uploaded per-game CSV stats**. "
    "We fuzzy-match player names."
)

# --------------------------- Helpers ---------------------------
SPORT_KEY = {
    "NFL": "americanfootball_nfl",
    "MLB": "baseball_mlb",
}

def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").replace("'", "").strip().lower()

def best_name_match(name, candidates, score_cut=84):
    name_c = clean_name(name)
    best, score_best = None, -1
    for c in candidates:
        sc = fuzz.token_sort_ratio(name_c, clean_name(c))
        if sc > score_best:
            best, score_best = c, sc
    return best if score_best >= score_cut else None

def conservative_sd(avg, minimum=0.75, frac=0.30):
    if avg is None or (isinstance(avg, float) and math.isnan(avg)): return 1.25
    if avg <= 0: return 1.0
    return max(frac * float(avg), minimum)

def simulate_prob(avg, line):
    sd = conservative_sd(avg)
    p_over = 1 - norm.cdf(line, loc=avg, scale=sd)
    p_under = norm.cdf(line, loc=avg, scale=sd)
    return round(100*p_over, 2), round(100*p_under, 2), round(sd, 3)

def value_from_mapping(row: pd.Series, mapping):
    if isinstance(mapping, list):
        vals = [row.get(c) for c in mapping if c in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan
    return row.get(mapping)

def fetch_odds_board(league: str, markets: list, bookmakers: str, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("Add your Odds API key in the sidebar (or set env ODDS_API_KEY).")
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY[league]}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "oddsFormat": "decimal",
        "markets": ",".join(markets),
        "bookmakers": bookmakers,
    }
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:400]}")
    data = r.json() or []
    rows = []
    for event in data:
        for bm in event.get("bookmakers", []):
            for m in bm.get("markets", []):
                mkey = m.get("key")
                if mkey not in markets:
                    continue
                for out in m.get("outcomes", []):
                    name, point = out.get("name"), out.get("point")
                    if name is None or point is None:
                        continue
                    rows.append({
                        "player": name,
                        "market": mkey,
                        "line": float(point),
                        "bookmaker": bm.get("key"),
                    })
    df = pd.DataFrame(rows)
    # collapse to a single line per player/market (median line across books)
    if not df.empty:
        df = (df.groupby(["player", "market"], as_index=False)
                .agg({"line": "median", "bookmaker": "first"}))
    return df

# --------------------------- CSV upload ---------------------------
st.markdown(f"### Upload {league} per-game stats CSV")
if league == "NFL":
    st.caption("Required column: `player`. Example columns: "
               "`passing_yards, rushing_yards, receiving_yards, receptions, passing_tds, "
               "rushing_tds, receiving_tds, interceptions_thrown`.")
else:
    st.caption("Required column: `player`. Example columns: "
               "`hits, total_bases, home_runs, rbis, runs, pitcher_strikeouts`.")

csv_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

if not sel_markets:
    st.warning("Select at least one market in the sidebar.")
    st.stop()

if csv_file is None:
    st.info("Upload your CSV to continue.")
    st.stop()

try:
    df_stats = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if "player" not in df_stats.columns:
    st.error("Your CSV must include a 'player' column.")
    st.stop()

df_stats["player"] = df_stats["player"].astype(str)
players = df_stats["player"].tolist()

# --------------------------- Fetch lines ---------------------------
try:
    board = fetch_odds_board(league, sel_markets, bookmakers, api_key)
except Exception as e:
    st.error(str(e))
    st.stop()

if board.empty:
    st.warning("No player lines returned for the selected markets. Try different markets/bookmakers.")
    st.stop()

st.success(f"Fetched {len(board)} player lines for {league}.")

# --------------------------- Simulate ---------------------------
CSV_MAP = {
    "NFL": {
        "player_passing_yards": "passing_yards",
        "player_rushing_yards": "rushing_yards",
        "player_receiving_yards": "receiving_yards",
        "player_receptions": "receptions",
        "player_rush_and_receive_yards": ["rushing_yards", "receiving_yards"],
        "player_passing_touchdowns": "passing_tds",
        "player_rushing_touchdowns": "rushing_tds",
        "player_receiving_touchdowns": "receiving_tds",
        "player_interceptions_thrown": "interceptions_thrown",
    },
    "MLB": {
        "player_hits": "hits",
        "player_total_bases": "total_bases",
        "player_home_runs": "home_runs",
        "player_rbis": "rbis",
        "player_runs_scored": "runs",
        "player_strikeouts": "pitcher_strikeouts",
    },
}

rows = []
missing_players = 0
missing_stats = 0

for _, r in board.iterrows():
    board_name = r["player"]
    market = r["market"]
    line = float(r["line"])

    match = best_name_match(board_name, players)
    if not match:
        missing_players += 1
        continue

    row = df_stats.loc[df_stats["player"] == match].iloc[0]
    mapping = CSV_MAP[league].get(market)
    if mapping is None:
        continue

    avg_val = value_from_mapping(row, mapping)
    if pd.isna(avg_val):
        missing_stats += 1
        continue

    p_over, p_under, sd = simulate_prob(float(avg_val), line)
    rows.append({
        "player_board": board_name,
        "player_csv": match,
        "market": market,
        "line": round(line, 3),
        "avg": round(float(avg_val), 3),
        "model_sd": sd,
        "P(Over)": p_over,
        "P(Under)": p_under,
    })

results = pd.DataFrame(rows).sort_values(["market", "P(Over)"], ascending=[True, False])

col1, col2 = st.columns(2)
with col1:
    st.metric("Matched props", len(results))
with col2:
    st.caption(f"Unmatched players: {missing_players} • Missing CSV stat values: {missing_stats}")

if results.empty:
    st.warning("No matches between Odds API players and your CSV for the selected markets.")
    st.stop()

st.subheader("Simulated probabilities")
st.caption("Model = conservative normal around your per-game average (prevents 0%/100% artifacts).")
st.dataframe(results, use_container_width=True)

st.download_button(
    "Download CSV",
    results.to_csv(index=False).encode("utf-8"),
    file_name=f"{league.lower()}_simulated_props.csv",
    mime="text/csv",
)

st.markdown("#### Top 10 Overs")
st.dataframe(results.nlargest(10, "P(Over)")[["player_board","market","line","avg","P(Over)","P(Under)"]],
             use_container_width=True)
st.markdown("#### Top 10 Unders")
st.dataframe(results.nlargest(10, "P(Under)")[["player_board","market","line","avg","P(Over)","P(Under)"]],
             use_container_width=True)
