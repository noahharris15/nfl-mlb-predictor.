# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 10k sims
# Place this file at: pages/1_NBA.py

import re, unicodedata, time, random
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st

# NBA API
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo

st.title("🏀 NBA Player Props — 10k Sims + Defense Adjustments")

SIM_TRIALS = 10_000

# ---------------- DEFENSE RATINGS ----------------
DEF_RATINGS = {
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

# ------------ NORMALIZE NBA TEAM NAMES → ODDS API TEAM NAMES ------------
TEAM_NORMALIZER = {
    "Wizards": "Washington Wizards",
    "Mavericks": "Dallas Mavericks",
    "Lakers": "Los Angeles Lakers",
    "Clippers": "Los Angeles Clippers",
    "Warriors": "Golden State Warriors",
    "Kings": "Sacramento Kings",
    "Suns": "Phoenix Suns",
    "Spurs": "San Antonio Spurs",
    "Knicks": "New York Knicks",
    "Nets": "Brooklyn Nets",
    "Hawks": "Atlanta Hawks",
    "Hornets": "Charlotte Hornets",
    "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Celtics": "Boston Celtics",
    "Pistons": "Detroit Pistons",
    "Raptors": "Toronto Raptors",
    "Pelicans": "New Orleans Pelicans",
    "Jazz": "Utah Jazz",
    "Thunder": "Oklahoma City Thunder",
    "Heat": "Miami Heat",
    "Bucks": "Milwaukee Bucks",
    "Rockets": "Houston Rockets",
    "Timberwolves": "Minnesota Timberwolves",
    "Trail Blazers": "Portland Trail Blazers",
    "76ers": "Philadelphia 76ers",
    "Magic": "Orlando Magic",
    "Pacers": "Indiana Pacers",
    "Nuggets": "Denver Nuggets",
    "Grizzlies": "Memphis Grizzlies",
}

# -------------------- VALID MARKETS --------------------
VALID_MARKETS = [
    "player_points","player_rebounds","player_assists","player_threes",
    "player_blocks","player_steals","player_blocks_steals","player_turnovers",
    "player_points_rebounds_assists","player_points_rebounds",
    "player_points_assists","player_rebounds_assists",
    "player_field_goals","player_frees_made","player_frees_attempts",
]

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket","player_first_team_basket","player_double_double",
    "player_triple_double","player_points_q1","player_rebounds_q1",
    "player_assists_q1",
}

ODDS_SPORT = "basketball_nba"


# ------------------- UTILITIES -------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n).lower()


# ------------------- SIMULATION -------------------
def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS):
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean()), draws


# ------------------- SAFE GAMELOG FETCH -------------------
def fetch_gamelog(player_id, season, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(0.25)
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season",
                timeout=10
            )
            df = gl.get_data_frames()[0]
            return df
        except:
            if attempt == retries - 1:
                return pd.DataFrame()
            time.sleep(0.5)
    return pd.DataFrame()


# ------------------- AGGREGATE STATS -------------------
def sample_sd(sum_x, sum_x2, g, floor=0.0):
    if g <= 1:
        return float("nan")
    mean = sum_x / g
    var = (sum_x2 / g) - (mean**2)
    var *= (g / (g - 1))
    return float(max(np.sqrt(max(var, 1e-9)), floor))

def agg_full_season(df: pd.DataFrame) -> Dict[str, float]:
    g = df.shape[0]
    if g == 0:
        return {"g": 0}

    stats = {}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        stats[col] = arr.sum()
        stats["sq_"+col] = (arr**2).sum()

    out = {"g": g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        out["mu_"+col] = stats[col] / g
        out["sd_"+col] = sample_sd(stats[col], stats["sq_"+col], g)
    return out


# ------------------- PLAYER ID LOOKUP -------------------
@st.cache_data(show_spinner=False)
def _players_index():
    df = pd.DataFrame(nba_players.get_players())
    df["name_norm"] = df["full_name"].apply(normalize_name)
    df["first_norm"] = df["first_name"].apply(normalize_name)
    df["last_norm"] = df["last_name"].apply(normalize_name)
    return df

def find_player_id_by_name(name: str) -> Optional[int]:
    df = _players_index()
    n = normalize_name(name)

    hit = df.loc[df["name_norm"] == n]
    if not hit.empty:
        return int(hit.iloc[0]["id"])

    parts = n.split()
    if len(parts) == 2:
        first, last = parts
        cand = df[df["last_norm"].str.contains(last)]
        cand = cand[cand["first_norm"].str.contains(first)]
        if not cand.empty:
            return int(cand.iloc[0]["id"])

    cand = df[df["name_norm"].str.contains(n)]
    if not cand.empty:
        return int(cand.iloc[0]["id"])

    return None


# ------------------- ODDS API -------------------
def odds_get(url: str, params: dict):
    try:
        r = requests.get(url, params=params, timeout=25)
        return r.json()
    except:
        return {}

def list_events(api_key, lookahead, region):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "regions": region, "daysFrom": 0, "daysTo": lookahead}
    )

def fetch_event_props(api_key, event_id, region, markets):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
    )


# ------------------- UI SELECTION -------------------
st.markdown("### 1) Season locked to 2025-26")
SEASON = "2025-26"

st.markdown("### 2) Select game & markets")
api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"])
lookahead = st.slider("Lookahead days", 1, 7, 1)

markets = st.multiselect("Markets", VALID_MARKETS, default=VALID_MARKETS)


# ------------------- LOAD GAMES -------------------
events = list_events(api_key, lookahead, region) if api_key else []

if not events:
    st.stop()

event_labels = []
for e in events:
    away = e.get("away_team") or "Away"
    home = e.get("home_team") or "Home"
    date = e.get("commence_time","")
    event_labels.append(f"{away} @ {home} — {date}")

pick = st.selectbox("Select Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

home_team = event.get("home_team")
away_team = event.get("away_team")


# ****************** 3) BUILD PLAYER PROJECTIONS ******************
st.markdown("### 3) Build Player Projections")
if st.button("📥 Build Projections"):

    prop_preview = fetch_event_props(api_key, event_id, region, markets)
    player_names = set()

    for bk in prop_preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE:
                continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm:
                    player_names.add(nm)

    rows = []
    for pn in sorted(player_names):
        pid = find_player_id_by_name(pn)
        if not pid:
            continue

        # Fetch team from NBA API
        try:
            info = commonplayerinfo.CommonPlayerInfo(pid).get_data_frames()[0]
            team_short = info["TEAM_NAME"].iloc[0]
            team_name = TEAM_NORMALIZER.get(team_short, team_short)
        except:
            continue

        df = fetch_gamelog(pid, SEASON)
        stats = agg_full_season(df)
        if stats["g"] == 0:
            continue

        rows.append({"Player": pn, "Team": team_name, **stats})

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    st.session_state["proj"] = proj

    st.dataframe(proj)


# ****************** 4) SIMULATE GAME PROPS ******************
st.markdown("### 4) Run Simulation (10,000 Sims)")
if st.button("🎯 Simulate"):

    proj = st.session_state.get("proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build projections first.")
        st.stop()

    props = fetch_event_props(api_key, event_id, region, markets)
    proj = proj.set_index("player_norm")

    sim_rows = []

    for bk in props.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey not in VALID_MARKETS:
                continue

            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                if name not in proj.index:
                    continue

                row = proj.loc[name]
                line = float(o["point"])
                team = row["Team"]

                # Determine opponent
                if team == home_team:
                    opponent = away_team
                elif team == away_team:
                    opponent = home_team
                else:
                    opponent = "UNKNOWN"

                # Defense scaling
                def_mult = DEF_RATINGS.get(opponent, 1.0)

                # Determine mu/sd
                def grab(col):
                    return row[f"mu_{col}"], row[f"sd_{col}"]

                if mkey=="player_points": mu,sd = grab("PTS")
                elif mkey=="player_rebounds": mu,sd = grab("REB")
                elif mkey=="player_assists": mu,sd = grab("AST")
                elif mkey=="player_threes": mu,sd = grab("FG3M")
                elif mkey=="player_blocks": mu,sd = grab("BLK")
                elif mkey=="player_steals": mu,sd = grab("STL")
                elif mkey=="player_turnovers": mu,sd = grab("TOV")
                elif mkey=="player_field_goals": mu,sd = grab("FGM")
                elif mkey=="player_frees_made": mu,sd = grab("FTM")
                elif mkey=="player_frees_attempts": mu,sd = grab("FTA")
                elif mkey=="player_points_rebounds_assists":
                    mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)
                elif mkey=="player_points_rebounds":
                    mu = row["mu_PTS"]+row["mu_REB"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2)
                elif mkey=="player_points_assists":
                    mu = row["mu_PTS"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2)
                elif mkey=="player_rebounds_assists":
                    mu = row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2)
                elif mkey=="player_blocks_steals":
                    mu = row["mu_BLK"]+row["mu_STL"]
                    sd = np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2)
                else:
                    continue

                # APPLY DEFENSE MULTIPLIER
                mu_adj = mu * def_mult

                # Sim
                p_over, draws = t_over_prob(mu_adj, sd, line, SIM_TRIALS)
                proj_val = float(np.median(draws))
                win_prob = p_over if o["name"]=="Over" else (1-p_over)

                sim_rows.append({
                    "Player": row["Player"],
                    "Team": team,
                    "Opponent": opponent,
                    "Market": mkey,
                    "Side": o["name"],
                    "Line": line,
                    "Defense Mult": def_mult,
                    "Model Projection": round(proj_val,2),
                    "Win Prob %": round(win_prob*100,2),
                })

    results = pd.DataFrame(sim_rows)
    st.dataframe(results)
    

# ****************** 5) FULL SLATE MODE ******************
st.markdown("### 5) Full Slate: Best Value Across All Games")

if st.button("📊 Run Full Slate"):

    all_events = list_events(api_key, lookahead, region)
    master = []

    for ev in all_events:
        eid = ev["id"]
        h = ev.get("home_team")
        a = ev.get("away_team")

        props = fetch_event_props(api_key, eid, region, markets)
        if not props:
            continue

        # Get players
        names = set()
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") in UNSUPPORTED_MARKETS_HIDE:
                    continue
                for o in m.get("outcomes", []):
                    names.add(normalize_name(o.get("description")))

        rows = []
        for pn in names:
            pid = find_player_id_by_name(pn)
            if not pid:
                continue

            try:
                info = commonplayerinfo.CommonPlayerInfo(pid).get_data_frames()[0]
                team_short = info["TEAM_NAME"].iloc[0]
                team_name = TEAM_NORMALIZER.get(team_short, team_short)
            except:
                continue

            df = fetch_gamelog(pid, SEASON)
            stats = agg_full_season(df)
            if stats["g"] == 0:
                continue

            rows.append({"Player": pn, "Team": team_name, **stats})

        if not rows:
            continue

        proj = pd.DataFrame(rows).set_index("Player")

        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                mkey = m.get("key")
                if mkey not in VALID_MARKETS:
                    continue

                for o in m.get("outcomes", []):
                    nm = normalize_name(o["description"])
                    if nm not in proj.index:
                        continue

                    row = proj.loc[nm]
                    team = row["Team"]

                    # Opponent
                    if team == h:
                        opp = a
                    elif team == a:
                        opp = h
                    else:
                        opp = "UNKNOWN"

                    def_mult = DEF_RATINGS.get(opp, 1.0)
                    line = float(o["point"])

                    def grab(col):
                        return row[f"mu_{col}"], row[f"sd_{col}"]

                    if mkey=="player_points": mu,sd = grab("PTS")
                    elif mkey=="player_rebounds": mu,sd = grab("REB")
                    elif mkey=="player_assists": mu,sd = grab("AST")
                    elif mkey=="player_threes": mu,sd = grab("FG3M")
                    elif mkey=="player_blocks": mu,sd = grab("BLK")
                    elif mkey=="player_steals": mu,sd = grab("STL")
                    elif mkey=="player_turnovers": mu,sd = grab("TOV")
                    elif mkey=="player_field_goals": mu,sd = grab("FGM")
                    elif mkey=="player_frees_made": mu,sd = grab("FTM")
                    elif mkey=="player_frees_attempts": mu,sd = grab("FTA")
                    elif mkey=="player_points_rebounds_assists":
                        mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                        sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)
                    elif mkey=="player_points_rebounds":
                        mu = row["mu_PTS"]+row["mu_REB"]
                        sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2)
                    elif mkey=="player_points_assists":
                        mu = row["mu_PTS"]+row["mu_AST"]
                        sd = np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2)
                    elif mkey=="player_rebounds_assists":
                        mu = row["mu_REB"]+row["mu_AST"]
                        sd = np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2)
                    elif mkey=="player_blocks_steals":
                        mu = row["mu_BLK"]+row["mu_STL"]
                        sd = np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2)
                    else:
                        continue

                    mu_adj = mu * def_mult

                    p_over, draws = t_over_prob(mu_adj, sd, line)
                    proj_val = np.median(draws)
                    edge = proj_val - line

                    master.append({
                        "Game": f"{a} @ {h}",
                        "Player": row.name,
                        "Team": team,
                        "Opponent": opp,
                        "Market": mkey,
                        "Line": line,
                        "Model Projection": round(proj_val,2),
                        "Edge": round(edge,2),
                        "Defense Mult": def_mult,
                    })

    full = pd.DataFrame(master).sort_values("Edge", ascending=False)
    st.dataframe(full.head(50), use_container_width=True)
