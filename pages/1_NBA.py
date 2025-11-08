# NBA Player Props — Odds API + NBA Stats (nba_api) + Defense Scalars + 10k sims
# Place this file at: pages/1_NBA.py

import re, unicodedata, time, random
import numpy as np
import pandas as pd
import requests
import streamlit as st
from typing import Optional, Dict, List, Tuple

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Player Props — Defense-Adjusted Sims (10k)")

SIM_TRIALS = 10_000

# ---------------------------------------------------------
# ✅ TEAM NORMALIZATION TABLE (strict)
# ---------------------------------------------------------
TEAM_NORMALIZE = {
    # WEST
    "oklahoma city thunder": "Oklahoma City Thunder",
    "okc thunder": "Oklahoma City Thunder",
    "thunder": "Oklahoma City Thunder",

    "san antonio spurs": "San Antonio Spurs",
    "spurs": "San Antonio Spurs",

    "portland trail blazers": "Portland Trail Blazers",
    "portland blazers": "Portland Trail Blazers",
    "blazers": "Portland Trail Blazers",

    "denver nuggets": "Denver Nuggets",
    "nuggets": "Denver Nuggets",

    "dallas mavericks": "Dallas Mavericks",
    "mavericks": "Dallas Mavericks",
    "mavs": "Dallas Mavericks",

    "golden state warriors": "Golden State Warriors",
    "warriors": "Golden State Warriors",

    "houston rockets": "Houston Rockets",
    "rockets": "Houston Rockets",

    "los angeles clippers": "Los Angeles Clippers",
    "la clippers": "Los Angeles Clippers",
    "clippers": "Los Angeles Clippers",

    "los angeles lakers": "Los Angeles Lakers",
    "la lakers": "Los Angeles Lakers",
    "lakers": "Los Angeles Lakers",

    "memphis grizzlies": "Memphis Grizzlies",
    "grizzlies": "Memphis Grizzlies",

    "minnesota timberwolves": "Minnesota Timberwolves",
    "wolves": "Minnesota Timberwolves",
    "timberwolves": "Minnesota Timberwolves",

    "new orleans pelicans": "New Orleans Pelicans",
    "pelicans": "New Orleans Pelicans",

    "phoenix suns": "Phoenix Suns",
    "suns": "Phoenix Suns",

    "sacramento kings": "Sacramento Kings",
    "kings": "Sacramento Kings",

    "utah jazz": "Utah Jazz",
    "jazz": "Utah Jazz",

    # EAST
    "atlanta hawks": "Atlanta Hawks",
    "hawks": "Atlanta Hawks",

    "boston celtics": "Boston Celtics",
    "celtics": "Boston Celtics",

    "brooklyn nets": "Brooklyn Nets",
    "nets": "Brooklyn Nets",

    "charlotte hornets": "Charlotte Hornets",
    "hornets": "Charlotte Hornets",

    "chicago bulls": "Chicago Bulls",
    "bulls": "Chicago Bulls",

    "cleveland cavaliers": "Cleveland Cavaliers",
    "cavaliers": "Cleveland Cavaliers",
    "cavs": "Cleveland Cavaliers",

    "detroit pistons": "Detroit Pistons",
    "pistons": "Detroit Pistons",

    "indiana pacers": "Indiana Pacers",
    "pacers": "Indiana Pacers",

    "miami heat": "Miami Heat",
    "heat": "Miami Heat",

    "milwaukee bucks": "Milwaukee Bucks",
    "bucks": "Milwaukee Bucks",

    "new york knicks": "New York Knicks",
    "knicks": "New York Knicks",

    "orlando magic": "Orlando Magic",
    "magic": "Orlando Magic",

    "philadelphia 76ers": "Philadelphia 76ers",
    "76ers": "Philadelphia 76ers",
    "sixers": "Philadelphia 76ers",

    "toronto raptors": "Toronto Raptors",
    "raptors": "Toronto Raptors",

    "washington wizards": "Washington Wizards",
    "wizards": "Washington Wizards",
}

# ---------------------------------------------------------
# ✅ DEFENSE MULTIPLIERS (No fallback allowed)
# ---------------------------------------------------------
DEF_MULT = {
    "Oklahoma City Thunder": 1.031,
    "San Antonio Spurs": 1.053,
    "Portland Trail Blazers": 1.073,
    "Miami Heat": 1.073,
    "Denver Nuggets": 1.074,
    "Detroit Pistons": 1.076,
    "Cleveland Cavaliers": 1.084,
    "Dallas Mavericks": 1.093,
    "Boston Celtics": 1.097,
    "Orlando Magic": 1.100,
    "Houston Rockets": 1.106,
    "Golden State Warriors": 1.109,
    "Indiana Pacers": 1.112,
    "Philadelphia 76ers": 1.116,
    "Chicago Bulls": 1.122,
    "Atlanta Hawks": 1.123,
    "Los Angeles Lakers": 1.127,
    "Milwaukee Bucks": 1.133,
    "Minnesota Timberwolves": 1.135,
    "Phoenix Suns": 1.137,
    "New York Knicks": 1.138,
    "Los Angeles Clippers": 1.141,
    "Memphis Grizzlies": 1.147,
    "Charlotte Hornets": 1.149,
    "Utah Jazz": 1.150,
    "Toronto Raptors": 1.152,
    "Sacramento Kings": 1.153,
    "Washington Wizards": 1.167,
    "New Orleans Pelicans": 1.226,
    "Brooklyn Nets": 1.249,
}

# ---------------------------------------------------------
# ------------------ HELPER FUNCTIONS ---------------------
# ---------------------------------------------------------

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = n.lower().strip()
    n = re.sub(r"[.,'()-]", " ", n)
    n = re.sub(r"\s+", " ", n)
    return strip_accents(n)

def t_sim(mu: float, sd: float, line: float, sims=SIM_TRIALS):
    sd = max(sd, 1e-6)
    draws = mu + sd * np.random.standard_t(df=5, size=sims)
    prob = float((draws > line).mean())
    return prob, draws

def fetch_event_props(api_key: str, event_id: str, region: str, mkts: List[str]):
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    return requests.get(url, params={"apiKey": api_key, "regions": region, "markets": ",".join(mkts)}).json()

# ---------------------------------------------------------
# ✅ UI: Odds API
# ---------------------------------------------------------

st.markdown("### Enter Odds API Key")
api_key = st.text_input("Odds API Key", "", type="password")
region = st.selectbox("Region", ["us", "us2", "eu", "uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, 1)

if not api_key:
    st.stop()

# ---------------------------------------------------------
# ✅ EVENTS LIST
# ---------------------------------------------------------

def list_events(api_key, region, lookahead):
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    return requests.get(url, params={"apiKey": api_key, "regions": region, "daysFrom": 0, "daysTo": lookahead}).json()

events = list_events(api_key, region, lookahead)

if not events or isinstance(events, dict):
    st.error("No events returned. Check API key or usage limit.")
    st.stop()

# ---------------------------------------------------------
# ✅ BUILD LABELS (home/away parsed safely)
# ---------------------------------------------------------

labels = []
for e in events:
    away = e.get("away_team", "")
    home = e.get("home_team", "")
    time_str = e.get("commence_time", "")
    labels.append(f"{away} @ {home} — {time_str}")

pick = st.selectbox("Select Game", labels)
event = events[labels.index(pick)]
event_id = event["id"]

# ---------------------------------------------------------
# ✅ PLAYER PROJECTIONS
# ---------------------------------------------------------

st.markdown("### Build Projections")
build = st.button("Build NBA Projections")

@st.cache_data
def fetch_games(pid, season):
    time.sleep(0.25)
    df = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

if build:

    # pull props once so we know which players are involved
    props_preview = fetch_event_props(api_key, event_id, region, ["player_points"])
    players = set()

    for bk in props_preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            for o in m.get("outcomes", []):
                nm = normalize_name(o["description"])
                if nm:
                    players.add(nm)

    index = pd.DataFrame(nba_players.get_players())
    index["norm"] = index["full_name"].apply(normalize_name)

    rows = []

    for p in players:
        match = index.loc[index["norm"] == p]
        if match.empty:
            continue

        pid = int(match.iloc[0]["id"])

        df = fetch_games(pid, "2025-26")

        if df.empty:
            df = fetch_games(pid, "2024-25")

        if df.empty:
            continue

        def get(col):
            x = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return x.mean(), x.std(ddof=1)

        mu_pts, sd_pts   = get("PTS")
        mu_reb, sd_reb   = get("REB")
        mu_ast, sd_ast   = get("AST")
        mu_tpm, sd_tpm   = get("FG3M")

        rows.append({
            "Player": p,
            "norm": p,
            "mu_pts": mu_pts, "sd_pts": sd_pts,
            "mu_reb": mu_reb, "sd_reb": sd_reb,
            "mu_ast": mu_ast, "sd_ast": sd_ast,
            "mu_tpm": mu_tpm, "sd_tpm": sd_tpm,
        })

    df = pd.DataFrame(rows)
    st.session_state["proj"] = df
    st.success("Projections Built ✅")
    st.dataframe(df)

# ---------------------------------------------------------
# ✅ SIMULATION SECTION
# ---------------------------------------------------------

st.markdown("### 4) Run Simulation")
run = st.button("Run Simulation")

if run:

    if "proj" not in st.session_state:
        st.error("Build projections first.")
        st.stop()

    proj = st.session_state["proj"].set_index("norm")

    # fetch props
    props = fetch_event_props(api_key, event_id, region, ["player_points", "player_rebounds", "player_assists", "player_threes"])

    # get teams
    away_team_raw = normalize_name(event.get("away_team"))
    home_team_raw = normalize_name(event.get("home_team"))

    # normalize to official names
    if away_team_raw not in TEAM_NORMALIZE:
        st.error(f"Unrecognized team name: {event.get('away_team')}")
        st.stop()

    if home_team_raw not in TEAM_NORMALIZE:
        st.error(f"Unrecognized team name: {event.get('home_team')}")
        st.stop()

    away_team = TEAM_NORMALIZE[away_team_raw]
    home_team = TEAM_NORMALIZE[home_team_raw]

    results = []

    for bk in props.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey not in ["player_points", "player_rebounds", "player_assists", "player_threes"]:
                continue

            for o in m.get("outcomes", []):
                name = normalize_name(o["description"])
                side = o["name"]
                line = float(o["point"])

                if name not in proj.index:
                    continue

                row = proj.loc[name]

                # pick stat
                if mkey == "player_points":
                    mu, sd = row["mu_pts"], row["sd_pts"]
                elif mkey == "player_rebounds":
                    mu, sd = row["mu_reb"], row["sd_reb"]
                elif mkey == "player_assists":
                    mu, sd = row["mu_ast"], row["sd_ast"]
                elif mkey == "player_threes":
                    mu, sd = row["mu_tpm"], row["sd_tpm"]

                # ---------------------------------------------------------
                # ✅ DEFENSE ADJUSTMENT (STRICT — MUST MATCH A TEAM)
                # ---------------------------------------------------------
                opponent_raw = home_team if side == "Over" else away_team
                opponent = opponent_raw  # already normalized above

                if opponent not in DEF_MULT:
                    st.error(f"Defense multiplier missing for: {opponent}")
                    st.stop()

                mult = DEF_MULT[opponent]

                mu_adj = mu / mult
                sd_adj = sd / mult

                # simulate
                p_over, draws = t_sim(mu_adj, sd_adj, line)
                proj_val = float(np.median(draws))
                win_prob = p_over if side == "Over" else (1 - p_over)

                results.append({
                    "Player": row["Player"],
                    "Market": mkey,
                    "Side": side,
                    "Line": line,
                    "Projection": round(proj_val,2),
                    "Win %": round(win_prob*100,2),
                    "Opponent": opponent,
                    "Defense Mult": mult,
                })

    out = pd.DataFrame(results).sort_values("Win %", ascending=False)
    st.dataframe(out, use_container_width=True)

    st.download_button("Download CSV", out.to_csv(index=False), "nba_sim_results.csv")
