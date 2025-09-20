# app.py — NFL + MLB predictor (2025 only, stats-only) + NFL Player Props (beta via Sleeper)
from __future__ import annotations

import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
from typing import Optional, Tuple

# ---- NFL team model ----------------------------------------------------------
import nfl_data_py as nfl

# ---- MLB team model ----------------------------------------------------------
from pybaseball import schedule_and_record
from mlbstatsapi import MLBStatsAPI

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6   # ~0.6 pts to the home offense mean (small HFA)
EPS = 1e-9

MLB_TEAMS_2025 = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles","BOS":"Boston Red Sox",
    "CHC":"Chicago Cubs","CHW":"Chicago White Sox","CIN":"Cincinnati Reds","CLE":"Cleveland Guardians",
    "COL":"Colorado Rockies","DET":"Detroit Tigers","HOU":"Houston Astros","KCR":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins","MIL":"Milwaukee Brewers",
    "MIN":"Minnesota Twins","NYM":"New York Mets","NYY":"New York Yankees","OAK":"Oakland Athletics",
    "PHI":"Philadelphia Phillies","PIT":"Pittsburgh Pirates","SDP":"San Diego Padres","SEA":"Seattle Mariners",
    "SFG":"San Francisco Giants","STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays","TEX":"Texas Rangers",
    "TOR":"Toronto Blue Jays","WSN":"Washington Nationals",
}

# -------------------------- utility -------------------------------------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53
    p_home = float(wins_home.mean())
    p_away = 1.0 - p_home
    return p_home, p_away, float(h.mean()), float(a.mean()), float((h + a).mean())

# -------------------------- NFL (teams) ---------------------------------------
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])
    date_col = "gameday" if "gameday" in sched.columns else ("game_date" if "game_date" in sched.columns else None)

    played = sched.dropna(subset=["home_score","away_score"])
    home = played.rename(columns={"home_team":"team","away_team":"opp","home_score":"pf","away_score":"pa"})[["team","opp","pf","pa"]]
    away = played.rename(columns={"away_team":"team","home_team":"opp","away_score":"pf","home_score":"pa"})[["team","opp","pf","pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 45.0/2.0
        teams32 = [
            "Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers","Chicago Bears",
            "Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions","Green Bay Packers",
            "Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs","Las Vegas Raiders",
            "Los Angeles Chargers","Los Angeles Rams","Miami Dolphins","Minnesota Vikings","New England Patriots",
            "New Orleans Saints","New York Giants","New York Jets","Philadelphia Eagles","Pittsburgh Steelers",
            "San Francisco 49ers","Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders"
        ]
        rates = pd.DataFrame({"team":teams32,"PF_pg":per,"PA_pg":per})
    else:
        team = long.groupby("team", as_index=False).agg(games=("pf","size"), PF=("pf","sum"), PA=("pa","sum"))
        rates = team.assign(PF_pg=team.PF/team.games, PA_pg=team.PA/team.games)[["team","PF_pg","PA_pg"]]
        league_total = float((long["pf"]+long["pa"]).mean())
        prior = league_total/2.0
        shrink = np.clip(1 - team["games"]/4.0, 0.0, 1.0)
        rates["PF_pg"] = (1-shrink)*rates["PF_pg"] + shrink*prior
        rates["PA_pg"] = (1-shrink)*rates["PA_pg"] + shrink*prior

    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][["home_team","away_team"]].copy()
    if date_col and date_col in sched.columns:
        upcoming["date"] = sched.loc[upcoming.index, date_col].astype(str)
    else:
        upcoming["date"] = ""
    for col in ["home_team","away_team"]:
        upcoming[col] = upcoming[col].astype(str).str.replace(r"\s+"," ",regex=True)
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

# -------------------------- MLB (teams + probables) ---------------------------
@st.cache_data(show_spinner=False)
def mlb_team_rates_2025():
    rows = []
    for br, name in MLB_TEAMS_2025.items():
        try:
            sar = schedule_and_record(2025, br)
            sar = sar[pd.to_numeric(sar.get("R"), errors="coerce").notna()]
            sar = sar[pd.to_numeric(sar.get("RA"), errors="coerce").notna()]
            if sar.empty:
                RS_pg = RA_pg = 4.5; games = 0
            else:
                sar["R"] = sar["R"].astype(float); sar["RA"] = sar["RA"].astype(float)
                games = int(len(sar)); RS_pg = float(sar["R"].sum()/games); RA_pg = float(sar["RA"].sum()/games)
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg, "games": games})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5, "games": 0})
    df = pd.DataFrame(rows).drop_duplicates("team").reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean()); league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9*df["RS_pg"] + 0.1*league_rs
        df["RA_pg"] = 0.9*df["RA_pg"] + 0.1*league_ra
    return df

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

# -------- MLB probable pitchers (name + ERA) using MLB-StatsAPI ---------------
@st.cache_data(show_spinner=False, ttl=60*60)
def mlb_probables_today():
    """Return DataFrame with columns: side, team, pitcher, ERA for today's games."""
    api = MLBStatsAPI()
    games = api.get_schedule()  # today's schedule
    rows = []
    for g in games.get("dates", []):
        for game in g.get("games", []):
            for side in ["home","away"]:
                club = game.get("teams", {}).get(side, {})
                team_name = club.get("team", {}).get("name")
                probable = club.get("probablePitcher") or {}
                pid = probable.get("id")
                name = probable.get("fullName")
                era = None
                if pid:
                    try:
                        p = api.get_person(pid)
                        stats = p.get("people", [{}])[0].get("stats", [])
                        for s in stats:
                            if s.get("group", {}).get("displayName") == "pitching":
                                splits = s.get("splits", [])
                                if splits:
                                    era = splits[0].get("stat", {}).get("era")
                                    break
                    except Exception:
                        pass
                rows.append({"side":"Home" if side=="home" else "Away", "team":team_name, "pitcher":name, "ERA":era})
    df = pd.DataFrame(rows).dropna(subset=["team"], how="all")
    return df

# -------------------------- NFL Player Props (Sleeper) ------------------------
SLEEPER_BASES = [
    "https://api.sleeper.app/v1",   # common
    "https://api.sleeper.com/v1",   # fallback domain
]

@st.cache_data(show_spinner=False, ttl=10*60)
def sleeper_players_map() -> dict:
    """player_id -> player dict (name, positions, team)."""
    for base in SLEEPER_BASES:
        try:
            r = requests.get(f"{base}/players/nfl", timeout=20)
            if r.status_code == 200:
                data = r.json()
                # data is a dict of id -> player
                return data
        except Exception:
            continue
    return {}

def _first_key(d: dict, keys: list[str]) -> Optional[str]:
    for k in keys:
        if k in d: return k
    return None

@st.cache_data(show_spinner=False, ttl=10*60)
def sleeper_projections_2025(week: int) -> pd.DataFrame:
    """
    Fetch 2025 regular-season projections for a given week from Sleeper.
    Returns a tidy DataFrame with common stat columns when available.
    """
    data = None
    for base in SLEEPER_BASES:
        try:
            url = f"{base}/projections/nfl/2025?season_type=regular&week={week}"
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            continue
    if not data:
        raise RuntimeError("Could not fetch Sleeper projections.")

    players = sleeper_players_map()
    rows = []
    for item in data:
        pid = str(item.get("player_id",""))
        stats = item.get("stats", {}) or {}
        team = item.get("team")
        pos = item.get("position") or (players.get(pid, {}).get("position"))
        name = players.get(pid, {}).get("full_name") or players.get(pid, {}).get("first_name","") + " " + players.get(pid, {}).get("last_name","")

        # Try multiple possible key aliases
        pass_yd = stats.get(_first_key(stats, ["pass_yd","pass_yds","pass_yards","passing_yards"]), np.nan)
        rush_yd = stats.get(_first_key(stats, ["rush_yd","rush_yds","rushing_yards"]), np.nan)
        rec_yd  = stats.get(_first_key(stats, ["rec_yd","rec_yds","receiving_yards"]), np.nan)
        rec_rec = stats.get(_first_key(stats, ["rec","receptions"]), np.nan)
        pass_td = stats.get(_first_key(stats, ["pass_td","passing_tds","passing_touchdowns"]), np.nan)
        rush_td = stats.get(_first_key(stats, ["rush_td","rushing_tds"]), np.nan)
        rec_td  = stats.get(_first_key(stats, ["rec_td","receiving_tds"]), np.nan)

        rows.append({
            "player_id": pid, "player": name.strip(), "team": team, "pos": pos,
            "pass_yd": pass_yd, "rush_yd": rush_yd, "rec_yd": rec_yd, "rec": rec_rec,
            "pass_td": pass_td, "rush_td": rush_td, "rec_td": rec_td
        })

    df = pd.DataFrame(rows)
    # keep only players with any projection
    keep_cols = ["pass_yd","rush_yd","rec_yd","rec","pass_td","rush_td","rec_td"]
    df = df.dropna(subset=keep_cols, how="all")
    return df

def prob_over_normal(mean: float, line: float, sd: float) -> float:
    """Simple normal model for O/U probability."""
    if sd <= 0: sd = max(5.0, 0.5*abs(mean))
    z = (line - mean) / sd
    # P(X > line) under N(mean, sd^2)
    return float(1 - 0.5*(1 + np.math.erf(z/np.sqrt(2))))

# --------------------------------- UI -----------------------------------------
st.set_page_config(page_title="NFL + MLB Predictor — 2025", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Win probabilities use **team scoring rates only** (NFL: PF/PA; MLB: RS/RA). "
    "NFL player props (beta) use **Sleeper 2025 weekly projections** with a simple variance model. "
    "No injuries, travel, or betting market inputs."
)

sport = st.radio("Pick a sport", ["NFL", "MLB", "NFL Props (beta)"], horizontal=True)

# -------------------------- NFL (teams) ---------------------------------------
if sport == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build team rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming games found in the 2025 schedule yet.")
        st.stop()

    # build choices like "CAR vs ATL — 2025-09-21"
    if "date" in upcoming.columns and upcoming["date"].astype(str).str.len().gt(0).any():
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"] + " — " + upcoming["date"].astype(str)).tolist()
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"]).tolist()

    pick = st.selectbox("Matchup", choices, index=0)
    home, away = pick.split(" — ")[0].split(" vs ")

    try:
        mu_h, mu_a = nfl_matchup_mu(nfl_rates, home, away)
        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
        c1, c2 = st.columns(2)
        with c1: st.metric(f"{home} win %", f"{p_home*100:.1f}%")
        with c2: st.metric(f"{away} win %", f"{p_away*100:.1f}%")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}** (total {exp_t:.1f})")
    except Exception as e:
        st.error(str(e))

# -------------------------- MLB (teams + probables) ---------------------------
elif sport == "MLB":
    st.subheader("⚾ MLB — 2025 season")

    try:
        mlb_rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't load MLB team data: {e}")
        st.stop()

    left, right = st.columns([1,2], gap="large")
    with left:
        st.markdown("**Team scoring rates (RS/RA per game)**")
        st.dataframe(mlb_rates.sort_values("team").reset_index(drop=True), use_container_width=True, height=520)

        with st.expander("Probable pitchers today (if available)"):
            probs = mlb_probables_today()
            if probs.empty:
                st.info("No probables returned yet.")
            else:
                st.dataframe(probs, use_container_width=True, height=320)

    with right:
        teams = mlb_rates["team"].sort_values().tolist()
        home = st.selectbox("Home team", teams, index=0, key="mlb_home")
        away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")
        use_probables = st.checkbox("Use today's probable pitchers (if available)", value=True)

        # show which probables we found
        prob_df = mlb_probables_today()
        show = None
        if use_probables and not prob_df.empty:
            mask = (prob_df["team"].isin([home, away]))
            show = prob_df[mask].copy()
            if not show.empty:
                st.markdown("**Probables used**")
                st.dataframe(show, use_container_width=True, height=120)

        # base means from team RS/RA
        mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away)

        # (optional) tiny pitcher ERA nudge
        if use_probables and show is not None and {"team","ERA"}.issubset(show.columns):
            try:
                # If ERA available, adjust ± up to ~0.4 runs vs league-average starter ERA (~4.20)
                LGE_ERA = 4.20
                adj = 0.08  # scale
                def era_bump(era):
                    if era is None or pd.isna(era): return 0.0
                    try:
                        return float(adj*(LGE_ERA - float(era)))
                    except Exception:
                        return 0.0
                h_era = show.loc[show["team"]==home, "ERA"]
                a_era = show.loc[show["team"]==away, "ERA"]
                if not h_era.empty: mu_h += era_bump(h_era.iloc[0])
                if not a_era.empty: mu_a += era_bump(a_era.iloc[0])
            except Exception:
                pass

        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
        c1, c2 = st.columns(2)
        with c1: st.metric(f"{home} win %", f"{p_home*100:.1f}%")
        with c2: st.metric(f"{away} win %", f"{p_away*100:.1f}%")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}** (total {exp_t:.1f})")

# -------------------------- NFL Player Props (beta) ---------------------------
else:
    st.subheader("🏈 NFL Player Props — 2025 REG (beta)")

    with st.expander("What this uses"):
        st.write(
            "- **Sleeper 2025 weekly projections** (no API key) for yards/TDs/receptions.\n"
            "- Simple normal model to turn a projection into O/U win probabilities.\n"
            "- You can also **upload a CSV** (columns like `player, team, pos, rec, rec_yd, rush_yd, pass_yd, ...`) "
            "or paste a public CSV URL if you prefer your own numbers."
        )

    week = st.number_input("Week", min_value=1, max_value=18, value=1, step=1)

    # try Sleeper first
    proj_df = None
    err = None
    try:
        proj_df = sleeper_projections_2025(week)
    except Exception as e:
        err = str(e)

    # fallbacks: CSV URL or file upload
    colu, colf = st.columns([1,1])
    with colu:
        csv_url = st.text_input("Or paste CSV URL (optional)")
        if csv_url:
            try:
                proj_df = pd.read_csv(csv_url)
                err = None
            except Exception as e:
                err = f"CSV URL error: {e}"
    with colf:
        csv_up = st.file_uploader("Or upload CSV (optional)", type=["csv"])

    if csv_up is not None:
        try:
            proj_df = pd.read_csv(csv_up)
            err = None
        except Exception as e:
            err = f"CSV upload error: {e}"

    if proj_df is None or proj_df.empty:
        if err:
            st.error(f"Couldn't load weekly player stats: {err}")
        else:
            st.warning("No player projection data available yet.")
        st.stop()

    # keep useful columns if present
    keep_cols = ["player","team","pos","rec","rec_yd","rush_yd","pass_yd","pass_td","rush_td","rec_td"]
    have = [c for c in keep_cols if c in proj_df.columns]
    view = proj_df[["player","team","pos"] + have[3:]] if len(have) else proj_df
    with st.expander("Preview projections", expanded=False):
        st.dataframe(view.head(25), use_container_width=True)

    # pick player + stat
    players = proj_df["player"].fillna("Unknown").tolist()
    psel = st.selectbox("Player", players, index=0)
    stat_choices = [c for c in ["pass_yd","rush_yd","rec_yd","rec","pass_td","rush_td","rec_td"] if c in proj_df.columns]
    ssel = st.selectbox("Stat", stat_choices, index=0)
    line = st.number_input("Prop line", value=float(proj_df.loc[proj_df["player"]==psel, ssel].iloc[0] or 0.0), step=1.0)

    # simple variance guess per-stat
    sd_defaults = {
        "pass_yd": 65.0, "rush_yd": 28.0, "rec_yd": 32.0, "rec": 2.6,
        "pass_td": 0.9, "rush_td": 0.6, "rec_td": 0.6
    }
    sd = sd_defaults.get(ssel, max(5.0, 0.5*abs(line)))

    mean = float(proj_df.loc[proj_df["player"]==psel, ssel].iloc[0])
    p_over = prob_over_normal(mean, line, sd)
    p_under = 1 - p_over

    c1, c2 = st.columns(2)
    with c1: st.metric(f"Over {line:g}", f"{p_over*100:.1f}%")
    with c2: st.metric(f"Under {line:g}", f"{p_under*100:.1f}%")
    st.caption(f"Model: Normal(mean={mean:.1f}, sd≈{sd:.1f}) from weekly projection source.")
