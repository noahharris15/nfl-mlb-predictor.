# app.py — NFL & MLB sims + NFL Player Props (Odds API) — single file

import time, math, json, random
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz
from scipy.stats import norm

# Third-party sports data
import nfl_data_py as nfl
from pybaseball import schedule_and_record

# ------------------------------ App setup -------------------------------------
st.set_page_config(page_title="NFL & MLB Sims + Player Props (Odds API)", layout="wide")
st.title("🏈⚾ NFL & MLB Simulators + 🧪 NFL Player Props (Odds API)")

# ------------------------------ Constants -------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6             # small home edge (points)
EPS = 1e-9
ODDS_API_KEY = "7401399bd14e8778312da073b621094f"  # <— your key

# The Odds API basics
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
}

# Valid player prop market keys (avoid 422 INVALID_MARKET)
VALID_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_receptions",
]

BOOKMAKER_DEFAULTS = ["draftkings", "fanduel", "betmgm", "caesars"]

# ------------------------------ Small utils -----------------------------------
def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").strip().lower()

def conservative_sd(avg: float, minimum: float = 0.75, frac: float = 0.30) -> float:
    if pd.isna(avg): return 1.25
    if avg <= 0:     return 1.0
    sd = max(frac * float(avg), minimum)
    return max(sd, 0.5)

def simulate_prob(avg: float, line: float) -> Tuple[float, float, float]:
    sd = conservative_sd(avg)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over * 100, 2), round(p_under * 100, 2), round(sd, 3)

def best_name_match(name: str, candidates: List[str], score_cut: int = 82) -> Optional[str]:
    name_c = clean_name(name)
    best, best_score = None, -1
    for c in candidates:
        s = fuzz.token_sort_ratio(name_c, clean_name(c))
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= score_cut else None

# ---------------------- NFL 2025 team rates & schedule ------------------------
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    # standardize date col
    date_col: Optional[str] = None
    for c in ("gameday", "game_date"):
        if c in sched.columns:
            date_col = c
            break

    played = sched.dropna(subset=["home_score", "away_score"])

    home = played.rename(columns={
        "home_team": "team", "away_team": "opp",
        "home_score": "pf", "away_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    away = played.rename(columns={
        "away_team": "team", "home_team": "opp",
        "away_score": "pf", "home_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 22.5
        teams32 = [
            "Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills",
            "Carolina Panthers","Chicago Bears","Cincinnati Bengals","Cleveland Browns",
            "Dallas Cowboys","Denver Broncos","Detroit Lions","Green Bay Packers",
            "Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
            "Las Vegas Raiders","Los Angeles Chargers","Los Angeles Rams","Miami Dolphins",
            "Minnesota Vikings","New England Patriots","New Orleans Saints","New York Giants",
            "New York Jets","Philadelphia Eagles","Pittsburgh Steelers","San Francisco 49ers",
            "Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders",
        ]
        rates = pd.DataFrame({"team": teams32, "PF_pg": per, "PA_pg": per})
    else:
        team = long.groupby("team", as_index=False).agg(
            games=("pf", "size"), PF=("pf", "sum"), PA=("pa", "sum")
        )
        rates = pd.DataFrame({
            "team": team["team"],
            "PF_pg": team["PF"] / team["games"],
            "PA_pg": team["PA"] / team["games"],
        })
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"] / 4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink) * rates["PF_pg"] + shrink * prior
        rates["PA_pg"] = (1 - shrink) * rates["PA_pg"] + shrink * prior

    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][
        ["home_team", "away_team"] + ([date_col] if date_col else [])
    ].copy()
    if date_col:
        upcoming = upcoming.rename(columns={date_col: "date"})
    else:
        upcoming["date"] = ""

    for col in ["home_team", "away_team"]:
        upcoming[col] = upcoming[col].astype(str).str.replace(r"\s+", " ", regex=True)

    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"]) / 2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"]) / 2.0)
    return mu_home, mu_away

# -------------------------- MLB 2025 team RS/RA -------------------------------
MLB_TEAMS_2025 = {
    "ARI": "Arizona Diamondbacks","ATL": "Atlanta Braves","BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox","CHC": "Chicago Cubs","CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds","CLE": "Cleveland Guardians","COL": "Colorado Rockies",
    "DET": "Detroit Tigers","HOU": "Houston Astros","KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels","LAD": "Los Angeles Dodgers","MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers","MIN": "Minnesota Twins","NYM": "New York Mets",
    "NYY": "New York Yankees","OAK": "Oakland Athletics","PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates","SDP": "San Diego Padres","SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants","STL": "St. Louis Cardinals","TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers","TOR": "Toronto Blue Jays","WSN": "Washington Nationals",
}

@st.cache_data(show_spinner=False)
def mlb_team_rates_2025():
    rows = []
    for br, name in MLB_TEAMS_2025.items():
        try:
            sar = schedule_and_record(2025, br)
            sar = sar[pd.to_numeric(sar.get("R"), errors="coerce").notna()]
            sar = sar[pd.to_numeric(sar.get("RA"), errors="coerce").notna()]
            if sar.empty:
                RS_pg = RA_pg = 4.5
                games = 0
            else:
                sar["R"] = sar["R"].astype(float)
                sar["RA"] = sar["RA"].astype(float)
                games = int(len(sar))
                RS_pg = float(sar["R"].sum() / games)
                RA_pg = float(sar["RA"].sum() / games)
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg, "games": games})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5, "games": 0})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean())
        league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9 * df["RS_pg"] + 0.1 * league_rs
        df["RA_pg"] = 0.9 * df["RA_pg"] + 0.1 * league_ra
    return df

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)  # neutral HFA
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

# ------------------------------ Odds API helpers ------------------------------
def _odds_get(path: str, params: Dict) -> requests.Response:
    params = dict(params or {})
    params["apiKey"] = ODDS_API_KEY
    r = requests.get(f"{ODDS_BASE}{path}", params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text}")
    return r

def list_events(sport_key: str, days_ahead: int = 3) -> List[Dict]:
    """List upcoming events for a sport."""
    r = _odds_get(
        f"/sports/{sport_key}/events",
        {"daysFrom": 0, "daysTo": days_ahead}
    )
    return r.json()

def event_props(event_id: str, regions: str, markets: List[str], bookmakers: List[str]) -> Dict:
    """Fetch props for one game (event) with selected markets."""
    params = {
        "regions": regions,
        "markets": ",".join(markets),
        "bookmakers": ",".join(bookmakers) if bookmakers else None,
        "oddsFormat": "american",
    }
    r = _odds_get(f"/sports/{SPORT_KEYS['NFL']}/events/{event_id}/odds", params)
    return r.json()

def parse_player_lines(evt_json: Dict) -> pd.DataFrame:
    """Flatten Odds API event response -> rows: player, market, line, book."""
    rows = []
    for bk in evt_json.get("bookmakers", []):
        bkkey = bk.get("key")
        for mkt in bk.get("markets", []):
            mkey = mkt.get("key")
            for oc in mkt.get("outcomes", []):
                name = oc.get("description")  # player name for props
                if not name:
                    continue
                line = oc.get("point")
                side = oc.get("name")  # Over or Under (we’ll store one row per side)
                if line is None or side not in ("Over", "Under"):
                    continue
                rows.append({
                    "book": bkkey,
                    "market": mkey,
                    "player": name,
                    "side": side,
                    "line": float(line),
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # reduce to one row per (player, market, book) — keep the Over row, line is same
    df = df.sort_values(["player", "market", "book"]).drop_duplicates(
        subset=["player","market","book","line","side"], keep="first"
    )
    # pivot to have Over/Under in columns if needed
    # for our simulation we just need the line once; Over/Under prob comes from line
    df = df[df["side"]=="Over"].drop(columns=["side"]).reset_index(drop=True)
    return df

# ------------------------------ NFL per-game stats ----------------------------
@st.cache_data(show_spinner=False)
def nfl_per_game_averages(season: int) -> pd.DataFrame:
    df = nfl.import_seasonal_data([season])
    g = df["games"].replace(0, np.nan) if "games" in df.columns else np.nan
    out = pd.DataFrame({
        "player": df.get("player_display_name"),
        "team": df.get("recent_team"),
        "pass_yards": df.get("passing_yards") / g if "games" in df else df.get("passing_yards"),
        "pass_tds":   df.get("passing_tds") / g if "games" in df else df.get("passing_tds"),
        "rush_yards": df.get("rushing_yards") / g if "games" in df else df.get("rushing_yards"),
        "receptions": df.get("receptions") / g if "games" in df else df.get("receptions"),
    }).dropna(subset=["player"])
    return out

PROP_TO_STAT = {
    "player_pass_yds": "pass_yards",
    "player_pass_tds": "pass_tds",
    "player_rush_yds": "rush_yards",
    "player_receptions": "receptions",
}

# ------------------------------ UI (pages) ------------------------------------
page = st.sidebar.radio("Page", ["NFL", "MLB", "Player Props (NFL)"], horizontal=False)

# ------------------------------------ NFL -------------------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 Team Matchup Simulator")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build NFL rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming games found yet.")
        st.stop()

    if "date" in upcoming.columns and upcoming["date"].astype(str).str.len().gt(0).any():
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"] +
                   " — " + upcoming["date"].astype(str)).tolist()
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"]).tolist()

    pick = st.selectbox("Matchup", choices, index=0)
    pair = pick.split(" — ")[0]
    home, away = pair.split(" vs ")

    try:
        mu_h, mu_a = nfl_matchup_mu(nfl_rates, home, away)
        # Poisson game sim
        mu_home = max(0.1, float(mu_h))
        mu_away = max(0.1, float(mu_a))
        h = np.random.poisson(mu_home, size=SIM_TRIALS)
        a = np.random.poisson(mu_away, size=SIM_TRIALS)
        wins_home = (h > a).astype(np.float64)
        ties = (h == a)
        if ties.any():
            wins_home[ties] = 0.53
        p_home = float(wins_home.mean())
        p_away = 1.0 - p_home
        exp_h, exp_a, exp_t = float(h.mean()), float(a.mean()), float((h+a).mean())

        c1, c2, c3 = st.columns(3)
        with c1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with c2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        with c3: st.metric(label="Exp total", value=f"{exp_t:.1f}")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
    except Exception as e:
        st.error(str(e))

# ------------------------------------ MLB -------------------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 Team Matchup Simulator")
    try:
        mlb_rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't load MLB data: {e}")
        st.stop()

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("**Team scoring rates (RS/RA per game)**")
        st.dataframe(
            mlb_rates.sort_values("team").reset_index(drop=True),
            use_container_width=True,
            height=520
        )

    with right:
        st.markdown("**Pick any MLB matchup**")
        teams = mlb_rates["team"].sort_values().tolist()
        if not teams:
            st.info("No MLB team data yet.")
            st.stop()

        home = st.selectbox("Home team", teams, index=0, key="mlb_home")
        away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")

        try:
            mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away)
            mu_home = max(0.1, float(mu_h))
            mu_away = max(0.1, float(mu_a))
            h = np.random.poisson(mu_home, size=SIM_TRIALS)
            a = np.random.poisson(mu_away, size=SIM_TRIALS)
            wins_home = (h > a).astype(np.float64)
            ties = (h == a)
            if ties.any(): wins_home[ties] = 0.5
            p_home = float(wins_home.mean())
            p_away = 1.0 - p_home
            exp_h, exp_a, exp_t = float(h.mean()), float(a.mean()), float((h+a).mean())

            c1, c2, c3 = st.columns(3)
            with c1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
            with c2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
            with c3: st.metric(label="Exp total", value=f"{exp_t:.1f}")
            st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
        except Exception as e:
            st.error(str(e))

# ------------------------------- Player Props (NFL) ---------------------------
else:
    st.subheader("🧪 NFL Player Props — Lines from The Odds API → Auto Over/Under Sims")

    # Controls
    season = st.number_input("Season (for player per-game stats)", 2018, 2025, value=2025, step=1)
    bookmakers = st.multiselect("Bookmakers (average across selected; empty = all available)",
                                BOOKMAKER_DEFAULTS, default=BOOKMAKER_DEFAULTS)
    markets = st.multiselect("Prop markets",
                             VALID_PROP_MARKETS,
                             default=["player_pass_yds","player_pass_tds","player_rush_yds","player_receptions"])

    c1, c2 = st.columns([1,1])
    with c1:
        days_ahead = st.slider("Lookahead days for events", 1, 7, 3)
    with c2:
        region = st.selectbox("Region", ["us","us2","eu","uk","au"], index=0)

    # List events
    try:
        events = list_events(SPORT_KEYS["NFL"], days_ahead)
    except Exception as e:
        st.error(f"Could not list NFL events: {e}")
        st.stop()

    if not events:
        st.info("No upcoming NFL events in the selected window.")
        st.stop()

    # Dropdown of events
    evt_opts = [f"{e.get('away_team')} @ {e.get('home_team')} — {e.get('commence_time')}|{e.get('id')}"
                for e in events]
    pick = st.selectbox("Choose game", evt_opts, index=0)
    evt_id = pick.split("|")[-1]

    # Fetch per-game averages (season)
    try:
        stats_df = nfl_per_game_averages(season)
    except Exception as e:
        st.error(f"Could not load NFL season averages: {e}")
        st.stop()

    players_set = list(stats_df["player"].unique())

    # Fetch props for selected event
    if st.button("Fetch lines & simulate"):
        try:
            evt = event_props(evt_id, region, markets, bookmakers)
        except Exception as e:
            st.error(f"Props fetch failed: {e}")
            st.stop()

        lines_df = parse_player_lines(evt)
        if lines_df.empty:
            st.warning("No player lines returned for chosen markets/bookmakers.")
            st.stop()

        # Average line across selected bookmakers for each (player,market)
        agg = (lines_df.groupby(["player","market"])["line"]
               .mean().reset_index().rename(columns={"line":"line_avg"}))

        rows = []
        for _, r in agg.iterrows():
            player = r["player"]; market = r["market"]; line = float(r["line_avg"])
            stat_col = PROP_TO_STAT.get(market)
            if not stat_col: 
                continue

            match = best_name_match(player, players_set, score_cut=82)
            if match is None:
                continue
            row_stats = stats_df.loc[stats_df["player"] == match].iloc[0]
            avg_val = row_stats.get(stat_col)
            if pd.isna(avg_val):
                continue

            p_over, p_under, used_sd = simulate_prob(float(avg_val), float(line))
            rows.append({
                "player": match,
                "prop_market": market,
                "line": round(line, 2),
                "avg": round(float(avg_val), 2),
                "model_sd": used_sd,
                "P(Over)%": p_over,
                "P(Under)%": p_under,
            })

        results = pd.DataFrame(rows).sort_values("P(Over)%", ascending=False)
        if results.empty:
            st.warning("Nothing matched (unrecognized names or missing stats).")
            st.stop()

        st.markdown("#### Simulated Over/Under (conservative normal model)")
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="nfl_props_sim.csv",
            mime="text/csv",
        )

        st.markdown("##### Top 8 Overs")
        st.dataframe(results.nlargest(8, "P(Over)%")[["player","prop_market","line","avg","P(Over)%","P(Under)%"]],
                     use_container_width=True)
        st.markdown("##### Top 8 Unders")
        st.dataframe(results.nlargest(8, "P(Under)%")[["player","prop_market","line","avg","P(Over)%","P(Under)%"]],
                     use_container_width=True)
