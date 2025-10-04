# app.py — Player Props Simulator (Odds API lines + auto stats; NFL + MLB)
import os
import pandas as pd
import numpy as np
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz

st.set_page_config(page_title="Player Props Simulator (Odds API + Auto Stats)", layout="wide")
st.title("📊 Player Props Simulator (Odds API + Auto Stats)")

# ---------------- Sidebar ----------------
with st.sidebar:
    league = st.selectbox("League", ["NFL", "MLB"])
    season = st.number_input("Season", 2018, 2025, value=2024, step=1)
    api_key = st.text_input("Odds API key (or env ODDS_API_KEY)", type="password") \
              or os.getenv("ODDS_API_KEY", "")

    # ✅ Accepted market keys per The Odds API v4
    VALID_MARKETS = {
        "NFL": [
            "player_pass_yds",
            "player_pass_tds",
            "player_pass_interceptions",
            "player_rush_yds",
            "player_rush_tds",
            "player_reception_yds",
            "player_receptions",
            "player_rec_tds",
            "player_rush_rec_yds",   # combo
        ],
        "MLB": [
            "player_hits",
            "player_total_bases",
            "player_home_runs",
            "player_rbis",
            "player_runs",
            "player_strikeouts",    # pitcher strikeouts
        ],
    }
    defaults = {
        "NFL": ["player_pass_yds","player_rush_yds","player_reception_yds","player_receptions"],
        "MLB": ["player_hits","player_total_bases","player_home_runs","player_strikeouts"],
    }
    markets = st.multiselect("Markets to pull",
                             options=VALID_MARKETS[league],
                             default=[m for m in defaults[league] if m in VALID_MARKETS[league]])
    bookmakers = st.text_input("Bookmakers (comma-sep)", "draftkings,betmgm,fanduel,caesars")

st.caption(
    "We fetch **player lines from The Odds API**, auto-load **real per-game stats** "
    f"for **{league} {season}**, fuzzy-match names, and simulate with a conservative normal model."
)

# ---------------- Helpers ----------------
SPORT_KEY = {"NFL": "americanfootball_nfl", "MLB": "baseball_mlb"}

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
    if avg is None or (isinstance(avg, float) and np.isnan(avg)): return 1.25
    if avg <= 0: return 1.0
    return max(frac*float(avg), minimum)

def simulate_prob(avg, line):
    sd = conservative_sd(avg)
    p_over = 1 - norm.cdf(line, loc=avg, scale=sd)
    p_under = norm.cdf(line, loc=avg, scale=sd)
    return round(100*p_over, 2), round(100*p_under, 2), round(sd, 3)

def fetch_odds_board(league: str, markets: list, bookmakers: str, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("Add your Odds API key in the sidebar (or set env ODDS_API_KEY).")
    # Only send markets that are valid for this league
    markets = [m for m in markets if m in VALID_MARKETS[league]]
    if not markets:
        return pd.DataFrame()

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY[league]}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "oddsFormat": "decimal",
        "markets": ",".join(markets),
        "bookmakers": bookmakers,
        "includeLinks": "false",
    }
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:500]}\nURL: {r.url}")
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
    if not df.empty:
        df = (df.groupby(["player","market"], as_index=False)
                .agg({"line": "median", "bookmaker":"first"}))
    return df

# ---------------- Auto stats loaders ----------------
@st.cache_data(ttl=6*3600, show_spinner=False)
def load_nfl_stats(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl
    st.info("Loading NFL season data…", icon="🏈")
    df = nfl.import_seasonal_data([season])
    g = df["games"].replace({0: np.nan}) if "games" in df.columns else np.nan
    out = pd.DataFrame({
        "player": df.get("player_display_name"),
        "passing_yds": df.get("passing_yards")/g,
        "rushing_yds": df.get("rushing_yards")/g,
        "receiving_yds": df.get("receiving_yards")/g,
        "receptions": df.get("receptions")/g,
        "pass_tds": df.get("passing_tds")/g,
        "rush_tds": df.get("rushing_tds")/g,
        "rec_tds": df.get("receiving_tds")/g,
        "pass_ints": df.get("interceptions")/g,  # thrown
    })
    return out.dropna(subset=["player"]).reset_index(drop=True)

@st.cache_data(ttl=12*3600, show_spinner=False)
def load_mlb_stats(season: int) -> pd.DataFrame:
    st.info("Loading MLB season stats…", icon="⚾")
    from pybaseball import batting_stats, pitching_stats
    bat = batting_stats(season); pit = pitching_stats(season)
    bat_out = pd.DataFrame({
        "player": bat.get("Name"),
        "G": bat.get("G"),
        "hits": bat.get("H"),
        "total_bases": bat.get("TB"),
        "home_runs": bat.get("HR"),
        "rbis": bat.get("RBI"),
        "runs": bat.get("R"),
    }).dropna(subset=["player"])
    for c in ["hits","total_bases","home_runs","rbis","runs"]:
        bat_out[c] = bat_out[c] / bat_out["G"].replace({0:np.nan})
    pit_out = pd.DataFrame({
        "player": pit.get("Name"),
        "G": pit.get("G"),
        "pitcher_strikeouts": pit.get("SO"),
    }).dropna(subset=["player"])
    pit_out["pitcher_strikeouts"] = pit_out["pitcher_strikeouts"] / pit_out["G"].replace({0:np.nan})
    out = pd.merge(bat_out.drop(columns=["G"]), pit_out.drop(columns=["G"]), on="player", how="outer")
    return out.dropna(subset=["player"]).reset_index(drop=True)

def get_stats_df(league: str, season: int) -> pd.DataFrame:
    return load_nfl_stats(season) if league=="NFL" else load_mlb_stats(season)

# ✅ Odds market -> our stat columns
STAT_MAP = {
    "NFL": {
        "player_pass_yds": "passing_yds",
        "player_pass_tds": "pass_tds",
        "player_pass_interceptions": "pass_ints",
        "player_rush_yds": "rushing_yds",
        "player_rush_tds": "rush_tds",
        "player_reception_yds": "receiving_yds",
        "player_receptions": "receptions",
        "player_rec_tds": "rec_tds",
        "player_rush_rec_yds": ["rushing_yds","receiving_yds"],
    },
    "MLB": {
        "player_hits": "hits",
        "player_total_bases": "total_bases",
        "player_home_runs": "home_runs",
        "player_rbis": "rbis",
        "player_runs": "runs",
        "player_strikeouts": "pitcher_strikeouts",
    },
}

def value_from_mapping(row: pd.Series, mapping):
    if isinstance(mapping, list):
        vals = [row.get(c) for c in mapping if c in row.index]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.sum(vals)) if vals else np.nan
    return row.get(mapping)

# ---------------- Fetch, match, simulate ----------------
if not markets:
    st.warning("Select at least one market in the sidebar.")
    st.stop()

try:
    board = fetch_odds_board(league, markets, bookmakers, api_key)
except Exception as e:
    st.error(str(e))
    st.stop()

if board.empty:
    st.warning("No player lines returned. Try different markets/bookmakers.")
    st.stop()

stats_df = get_stats_df(league, season)
players = stats_df["player"].astype(str).tolist()

rows, missing_names, missing_stats = [], 0, 0
for _, r in board.iterrows():
    b_name, mkt, line = r["player"], r["market"], float(r["line"])
    mapping = STAT_MAP[league].get(mkt)
    if mapping is None:
        continue

    match = best_name_match(b_name, players)
    if not match:
        missing_names += 1
        continue

    row = stats_df.loc[stats_df["player"] == match].iloc[0]
    avg = value_from_mapping(row, mapping)
    if pd.isna(avg):
        missing_stats += 1
        continue

    p_over, p_under, sd = simulate_prob(float(avg), line)
    rows.append({
        "player": match,
        "board_name": b_name,
        "market": mkt,
        "line": round(line, 3),
        "avg": round(float(avg), 3),
        "model_sd": sd,
        "P(Over)": p_over,
        "P(Under)": p_under,
    })

results = pd.DataFrame(rows).sort_values(["market","P(Over)"], ascending=[True,False])

c1, c2, c3 = st.columns(3)
c1.metric("Matched props", len(results))
c2.caption(f"Unmatched names: {missing_names}")
c3.caption(f"Missing stat values: {missing_stats}")

if results.empty:
    st.warning("No matches between Odds API players and auto stats for the selected markets.")
    st.stop()

st.subheader("Simulated probabilities")
st.caption("Model = conservative normal around per-game average (avoids 0%/100% artifacts).")
st.dataframe(results, use_container_width=True)

st.download_button(
    "Download CSV",
    results.to_csv(index=False).encode("utf-8"),
    file_name=f"{league.lower()}_{season}_simulated_props.csv",
    mime="text/csv",
)

st.markdown("#### Top 10 Overs")
st.dataframe(results.nlargest(10,"P(Over)")[["board_name","market","line","avg","P(Over)","P(Under)"]],
             use_container_width=True)
st.markdown("#### Top 10 Unders")
st.dataframe(results.nlargest(10,"P(Under)")[["board_name","market","line","avg","P(Over)","P(Under)"]],
             use_container_width=True)
