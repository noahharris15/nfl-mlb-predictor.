# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 10k sims
# Place in: pages/1_NBA.py

import re, unicodedata, datetime as dt, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- NBA Stats ----------
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo


st.title("🏀 NBA Player Props — Odds API + NBA Stats (live)")

SIM_TRIALS = 10_000


# -------------------- VALID MARKETS --------------------
VALID_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_blocks_steals",
    "player_turnovers",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_rebounds_assists",
    "player_field_goals",
    "player_frees_made",
    "player_frees_attempts",
]

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket",
    "player_first_team_basket",
    "player_double_double",
    "player_triple_double",
    "player_points_q1",
    "player_rebounds_q1",
    "player_assists_q1",
}

ODDS_SPORT = "basketball_nba"


# ------------------ DEFENSE RATINGS ------------------
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


# ------------------ Team Normalizer ------------------
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


# ------------------ Utilities ------------------
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


def t_over_prob(mu: float, sd: float, line: float,
                trials: int = SIM_TRIALS) -> Tuple[float, np.ndarray]:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean()), draws


def sample_sd(sum_x: float, sum_x2: float, g: int,
              floor: float = 0.0) -> float:
    g = int(g)
    if g <= 1:
        return float("nan")
    mean = sum_x / g
    var = (sum_x2 / g) - (mean ** 2)
    var = var * (g / (g - 1))
    return float(max(np.sqrt(max(var, 1e-9)), floor))


@st.cache_data(show_spinner=False)
def _players_index() -> pd.DataFrame:
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    df["name_norm"] = df["full_name"].apply(normalize_name)
    df["first_norm"] = df["first_name"].apply(normalize_name)
    df["last_norm"] = df["last_name"].apply(normalize_name)
    return df[["id", "full_name", "name_norm", "first_norm", "last_norm"]]


def find_player_id_by_name(name: str) -> Optional[int]:
    df = _players_index()
    n = normalize_name(name)
    parts = n.split()

    hit = df.loc[df["name_norm"] == n]
    if not hit.empty:
        return int(hit.iloc[0]["id"])

    if len(parts) == 2:
        first, last = parts
        cand = df.loc[df["last_norm"].str.startswith(last)]
        if cand.empty:
            cand = df.loc[df["last_norm"].str.contains(last)]
        if not cand.empty:
            cand = cand.loc[
                cand["first_norm"].str.startswith(first[:1])
                | cand["first_norm"].str.contains(first)
            ]
            if not cand.empty:
                return int(cand.iloc[0]["id"])

    last = parts[-1] if parts else n
    cand = df.loc[df["name_norm"].str.contains(last)]
    if not cand.empty:
        return int(cand.iloc[0]["id"])

    return None


# ✅ Cached team lookup (FAST)
@st.cache_data
def get_player_team(pid):
    try:
        info = commonplayerinfo.CommonPlayerInfo(pid).get_data_frames()[0]
        name = info["TEAM_NAME"].iloc[0]
        return TEAM_NORMALIZER.get(name, name)
    except:
        return None


# ✅ Fast gamelog fetch (NO 10-minute freezes)
def fetch_gamelog(player_id, season, retries=1):
    for _ in range(retries):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season",
                timeout=8
            )
            return gl.get_data_frames()[0]
        except:
            time.sleep(0.3)
    return pd.DataFrame()


def agg_full_season(df: pd.DataFrame) -> Dict[str, float]:
    g = int(df.shape[0])
    if g == 0:
        return {"g": 0}

    out = {"g": g}
    for col in ["PTS", "REB", "AST", "STL", "BLK",
                "TOV", "FG3M", "FGM", "FTM", "FTA"]:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        s = float(x.sum())
        s2 = float((x ** 2).sum())
        out[f"mu_{col}"] = s / g
        out[f"sd_{col}"] = sample_sd(s, s2, g)
    return out


# ------------------ Season ------------------
st.markdown("### Season Locked to 2025–26")
season_locked = "2025-26"


# ------------------ Odds API ------------------
st.markdown("### 1) Game + Markets")
api_key = st.text_input("Odds API Key", "", type="password")
region = st.selectbox("Region", ["us", "us2", "eu", "uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets_pickable = VALID_MARKETS + list(UNSUPPORTED_MARKETS_HIDE)
markets = st.multiselect("Markets to fetch", markets_pickable,
                         default=VALID_MARKETS)


def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=20)
    return r.json()


def list_events(api_key, days, region):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "daysFrom": 0,
         "daysTo": days or 1, "regions": region}
    )


if not api_key:
    st.stop()

events = list_events(api_key, lookahead, region)
if not events:
    st.stop()

event_labels = []
for e in events:
    away = e.get("away_team") or "Away"
    home = e.get("home_team") or "Home"
    date = e.get("commence_time", "")
    event_labels.append(f"{away} @ {home} — {date}")

pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

home_team = event.get("home_team")
away_team = event.get("away_team")


# ------------------ Build Projections ------------------
st.markdown("### 2) Build Player Projections")
build = st.button("Build Projections")

if build:

    preview = odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region,
         "markets": ",".join(markets)}
    )

    player_names = set()
    for bk in preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE:
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

        team = get_player_team(pid)
        if not team:
            continue

        df = fetch_gamelog(pid, season_locked)
        stats = agg_full_season(df)

        if stats["g"] == 0:
            df2 = fetch_gamelog(pid, "2024-25")
            stats2 = agg_full_season(df2)
            if stats2["g"] != 0:
                stats = stats2
            else:
                continue

        rows.append({
            "Player": pn,
            "team": team,
            **stats
        })

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    st.session_state["nba_proj"] = proj

    st.dataframe(proj)


# ------------------ Simulation ------------------
st.markdown("### 3) Simulate Props (10k sims)")
go = st.button("Simulate")

if go:
    proj = st.session_state.get("nba_proj")
    if proj is None or proj.empty:
        st.stop()

    proj = proj.set_index("player_norm")

    odds_data = odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region,
         "markets": ",".join(markets)}
    )

    rows = []

    for bk in odds_data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m["key"]
            if mkey not in VALID_MARKETS:
                continue

            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                if name not in proj.index:
                    continue

                row = proj.loc[name]
                side = o["name"]
                line = float(o["point"])

                # determine opponent
                team = row["team"]
                if team == home_team:
                    opp = away_team
                elif team == away_team:
                    opp = home_team
                else:
                    opp = None

                mult = DEF_RATINGS.get(opp, 1.0)

                def grab(c):
                    return row[f"mu_{c}"] * mult, row[f"sd_{c}"] * mult

                if mkey == "player_points":
                    mu, sd = grab("PTS")
                elif mkey == "player_rebounds":
                    mu, sd = grab("REB")
                elif mkey == "player_assists":
                    mu, sd = grab("AST")
                elif mkey == "player_threes":
                    mu, sd = grab("FG3M")
                elif mkey == "player_blocks":
                    mu, sd = grab("BLK")
                elif mkey == "player_steals":
                    mu, sd = grab("STL")
                elif mkey == "player_turnovers":
                    mu, sd = grab("TOV")
                elif mkey == "player_field_goals":
                    mu, sd = grab("FGM")
                elif mkey == "player_frees_made":
                    mu, sd = grab("FTM")
                elif mkey == "player_frees_attempts":
                    mu, sd = grab("FTA")
                elif mkey == "player_points_rebounds_assists":
                    mu = (row["mu_PTS"] + row["mu_REB"] + row["mu_AST"]) * mult
                    sd = (row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)**0.5 * mult
                elif mkey == "player_points_rebounds":
                    mu = (row["mu_PTS"] + row["mu_REB"]) * mult
                    sd = (row["sd_PTS"]**2 + row["sd_REB"]**2)**0.5 * mult
                elif mkey == "player_points_assists":
                    mu = (row["mu_PTS"] + row["mu_AST"]) * mult
                    sd = (row["sd_PTS"]**2 + row["sd_AST"]**2)**0.5 * mult
                elif mkey == "player_rebounds_assists":
                    mu = (row["mu_REB"] + row["mu_AST"]) * mult
                    sd = (row["sd_REB"]**2 + row["sd_AST"]**2)**0.5 * mult
                elif mkey == "player_blocks_steals":
                    mu = (row["mu_BLK"] + row["mu_STL"]) * mult
                    sd = (row["sd_BLK"]**2 + row["sd_STL"]**2)**0.5 * mult
                else:
                    continue

                p_over, draws = t_over_prob(mu, sd, line)
                projection = float(np.median(draws))
                win_prob = p_over if side == "Over" else (1 - p_over)

                rows.append({
                    "Player": row["Player"],
                    "Market": mkey,
                    "Side": side,
                    "Line": round(line, 2),
                    "Model Projection": round(projection, 2),
                    "Win Prob %": round(win_prob * 100, 2),
                    "Opponent": opp,
                    "Defense Multiplier": mult
                })

    out = pd.DataFrame(rows).sort_values(
        ["Market", "Win Prob %"],
        ascending=[True, False]
    )

    st.dataframe(out)


# ------------------ FULL SLATE ------------------
st.markdown("### 4) Full-Slate Best Value Report")
run_full = st.button("Run Full Slate")

if run_full:

    all_events = list_events(api_key, lookahead, region)
    master = []

    for ev in all_events:

        eid = ev["id"]
        home = ev.get("home_team")
        away = ev.get("away_team")

        props = odds_get(
            f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
            {"apiKey": api_key, "regions": region,
             "markets": ",".join(markets)}
        )

        # collect players for this game
        pn_set = set()
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] in UNSUPPORTED_MARKETS_HIDE:
                    continue
                for o in m.get("outcomes", []):
                    nm = normalize_name(o.get("description"))
                    if nm:
                        pn_set.add(nm)

        rows = []
        for pn in sorted(pn_set):
            pid = find_player_id_by_name(pn)
            if not pid:
                continue

            team = get_player_team(pid)
            if not team:
                continue

            df = fetch_gamelog(pid, season_locked)
            stats = agg_full_season(df)

            if stats["g"] == 0:
                df2 = fetch_gamelog(pid, "2024-25")
                stats2 = agg_full_season(df2)
                if stats2["g"] != 0:
                    stats = stats2
                else:
                    continue

            rows.append({"Player": pn, "team": team, **stats})

        if not rows:
            continue

        proj = pd.DataFrame(rows)
        proj["player_norm"] = proj["Player"].apply(normalize_name)
        proj = proj.set_index("player_norm")

        # simulate props
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                mkey =
