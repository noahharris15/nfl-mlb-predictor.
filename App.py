# app.py — NFL + MLB predictor (teams) + NFL Player Props (beta) — 2025 only

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple

# ---- NFL ---------------------------------------------------------------------
import nfl_data_py as nfl

# ---- MLB ---------------------------------------------------------------------
from pybaseball import schedule_and_record
try:
    # optional: improves MLB pitcher names + ERA
    from mlbstatsapi import MLBStatsAPI
    _mlb_api = MLBStatsAPI()
except Exception:
    _mlb_api = None

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6   # small home bump to home offense mean
EPS = 1e-9

MLB_TEAMS_2025 = {
    "ARI": "Arizona Diamondbacks","ATL": "Atlanta Braves","BAL": "Baltimore Orioles","BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs","CHW": "Chicago White Sox","CIN": "Cincinnati Reds","CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies","DET": "Detroit Tigers","HOU": "Houston Astros","KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels","LAD": "Los Angeles Dodgers","MIA": "Miami Marlins","MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins","NYM": "New York Mets","NYY": "New York Yankees","OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies","PIT": "Pittsburgh Pirates","SDP": "San Diego Padres","SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants","STL": "St. Louis Cardinals","TBR": "Tampa Bay Rays","TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays","WSN": "Washington Nationals",
}

# -------------------------- shared sim helpers --------------------------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # tiny NFL tiebreak to home
    return float(wins_home.mean()), float(1 - wins_home.mean()), float(h.mean()), float(a.mean()), float((h+a).mean())

def simulate_normal_over(mean: float, std: float, line: float, trials: int = SIM_TRIALS) -> Tuple[float, float]:
    mean = float(mean)
    std = max(1e-6, float(std))
    draws = np.random.normal(mean, std, size=trials)
    p_over = float((draws > line).mean())
    return p_over, 1.0 - p_over

# -------------------------- NFL team page -------------------------------------
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    # normalize date column
    date_col = "gameday" if "gameday" in sched.columns else ("game_date" if "game_date" in sched.columns else None)

    played = sched.dropna(subset=["home_score", "away_score"])
    home = played.rename(columns={"home_team":"team", "away_team":"opp", "home_score":"pf", "away_score":"pa"})[["team","opp","pf","pa"]]
    away = played.rename(columns={"away_team":"team", "home_team":"opp", "away_score":"pf", "home_score":"pa"})[["team","opp","pf","pa"]]
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
        rates = pd.DataFrame({"team": teams32, "PF_pg": per, "PA_pg": per})
    else:
        team = long.groupby("team", as_index=False).agg(games=("pf","size"), PF=("pf","sum"), PA=("pa","sum"))
        rates = pd.DataFrame({"team":team["team"], "PF_pg":team["PF"]/team["games"], "PA_pg":team["PA"]/team["games"]})
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total/2.0
        shrink = np.clip(1.0 - team["games"]/4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink)*rates["PF_pg"] + shrink*prior
        rates["PA_pg"] = (1 - shrink)*rates["PA_pg"] + shrink*prior

    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][["home_team","away_team"] + ([date_col] if date_col else [])].copy()
    if date_col:
        upcoming = upcoming.rename(columns={date_col:"date"})
    else:
        upcoming["date"] = ""
    for c in ["home_team","away_team"]:
        upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+"," ", regex=True)
    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower()==home.lower()]
    rA = rates.loc[rates["team"].str.lower()==away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"])/2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"])/2.0)
    return mu_home, mu_away

# -------------------------- MLB team page -------------------------------------
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

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean()); league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9*df["RS_pg"] + 0.1*league_rs
        df["RA_pg"] = 0.9*df["RA_pg"] + 0.1*league_ra
    return df

def mlb_probables_today() -> pd.DataFrame:
    """Return today's probables with ERA if MLB-StatsAPI is installed."""
    if _mlb_api is None:
        return pd.DataFrame(columns=["team","pitcher","era"])
    try:
        games = _mlb_api.schedule(date=None, sportId=1)  # defaults to today in UTC
        rows = []
        for g in games:
            for side in ("home","away"):
                t = g.get(side, {})
                name = t.get("team", {}).get("name")
                p = g.get("probablePitchers", {}).get(side)
                if name and p:
                    era = None
                    if "era" in p and p["era"] not in (None, "None"):
                        era = float(p["era"])
                    rows.append({"team": name, "pitcher": p.get("fullName") or p.get("name"), "era": era})
        return pd.DataFrame(rows).dropna(subset=["team"]).drop_duplicates(subset=["team"])
    except Exception:
        return pd.DataFrame(columns=["team","pitcher","era"])

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str, prob: Optional[pd.DataFrame]=None):
    rH = rates.loc[rates["team"].str.lower()==home.lower()]
    rA = rates.loc[rates["team"].str.lower()==away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = (H["RS_pg"] + A["RA_pg"])/2.0
    mu_away = (A["RS_pg"] + H["RA_pg"])/2.0

    # small ERA-based bump if available (±0.2 runs per ERA diff from 4.00)
    if prob is not None and not prob.empty:
        for side, team, flip in [("home", home, +1), ("away", away, -1)]:
            row = prob.loc[prob["team"].str.lower()==team.lower()]
            if not row.empty and pd.notna(row.iloc[0].get("era")):
                adj = 0.2*(4.00 - float(row.iloc[0]["era"]))  # lower ERA -> subtract from opponent scoring
                if side == "home":
                    mu_away = max(EPS, mu_away - adj)
                else:
                    mu_home = max(EPS, mu_home - adj)
    return float(mu_home), float(mu_away)

# -------------------------- NFL Props (beta) ----------------------------------
PROPS = {
    "Passing Yards": {"stat_col": "passing_yards", "per_g": True},
    "Rushing Yards": {"stat_col": "rushing_yards", "per_g": True},
    "Receiving Yards": {"stat_col": "receiving_yards", "per_g": True},
    "Receptions": {"stat_col": "receptions", "per_g": True},
    "Pass Attempts": {"stat_col": "attempts", "per_g": True},
}

@st.cache_data(show_spinner=False)
def load_weekly_players_2025() -> pd.DataFrame:
    """Try nfl_data_py weekly player data for 2025; return empty on failure."""
    try:
        df = nfl.import_weekly_data(years=[2025])
        # standardize column names we need
        df = df.rename(columns={
            "player_name": "player", "recent_team": "team",
            "passing_yards": "passing_yards",
            "rushing_yards": "rushing_yards",
            "receiving_yards": "receiving_yards",
            "receptions": "receptions",
            "attempts": "attempts",
            "opponent_team": "opp",
        })
        # keep sensible columns
        keep = ["season","week","player","team","opp","position",
                "passing_yards","rushing_yards","receiving_yards","receptions","attempts"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df = df.query("season == 2025")
        return df
    except Exception:
        return pd.DataFrame()

def recent_form(df_player: pd.DataFrame, col: str, k: int = 5) -> Tuple[float, float]:
    """Return mean & std of last k games for player for the selected stat."""
    x = pd.to_numeric(df_player[col], errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return 0.0, 1.0
    x = x.tail(k)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) >= 2 else max(1.0, mean*0.35)
    return mean, max(1.0, std)

@st.cache_data(show_spinner=False)
def defense_allowed_2025(stat_col: str) -> pd.DataFrame:
    """
    Build per-opponent allowed-per-game table for 2025 from weekly data.
    Returns DataFrame columns: team, allowed_pg
    """
    df = load_weekly_players_2025()
    if df.empty or stat_col not in df.columns:
        return pd.DataFrame(columns=["team","allowed_pg"])
    # For a given stat, sum it by opponent in each game, then average
    df_use = df[["opp", stat_col]].copy()
    df_use[stat_col] = pd.to_numeric(df_use[stat_col], errors="coerce").fillna(0.0)
    agg = df_use.groupby("opp", as_index=False)[stat_col].mean().rename(columns={"opp":"team", stat_col:"allowed_pg"})
    return agg

def props_line_sim(player_games: pd.DataFrame, stat_col: str, opp_team: Optional[str], line: float) -> Tuple[float, float, float, float]:
    # player recent
    mu, sd = recent_form(player_games, stat_col, k=5)
    # defense adjust (±10% scale around league average)
    if opp_team:
        df_def = defense_allowed_2025(stat_col)
        if not df_def.empty and opp_team in df_def["team"].values:
            league = float(df_def["allowed_pg"].mean())
            opp = float(df_def.loc[df_def["team"]==opp_team, "allowed_pg"].iloc[0])
            if league > 0:
                factor = np.clip(opp/league, 0.7, 1.3)  # tame extremes
                mu *= factor
    p_over, p_under = simulate_normal_over(mu, sd, line, SIM_TRIALS)
    return p_over, p_under, mu, sd

# --------------------------------- UI -----------------------------------------
st.set_page_config(page_title="NFL + MLB Predictor — 2025", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025")
st.caption(
    "Win probabilities use team scoring rates only (NFL: PF/PA; MLB: RS/RA). "
    "MLB probables/ERA can nudge totals. **NFL Props (beta)** uses recent player form "
    "+ opponent defense allowed per game. No injuries, travel, or betting lines."
)

tab = st.radio("Pick a sport", ["NFL", "MLB", "NFL Props (beta)"], horizontal=True)

# -------------------------- NFL page ------------------------------------------
if tab == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build team rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming games found in the 2025 schedule yet.")
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"]).tolist()
        pick = st.selectbox("Matchup", choices, index=0)
        home, away = pick.split(" vs ")
        try:
            mu_h, mu_a = nfl_matchup_mu(nfl_rates, home, away)
            p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
            c1, c2 = st.columns(2)
            with c1: st.metric(f"{home} win %", f"{p_home*100:.1f}%")
            with c2: st.metric(f"{away} win %", f"{p_away*100:.1f}%")
            st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}** (total {exp_t:.1f})")
        except Exception as e:
            st.error(str(e))

# -------------------------- MLB page ------------------------------------------
elif tab == "MLB":
    st.subheader("⚾ MLB — 2025 season")

    try:
        mlb_rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't load MLB data: {e}")
        st.stop()

    prob = mlb_probables_today()
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**Team scoring rates (RS/RA per game)**")
        st.dataframe(mlb_rates.sort_values("team").reset_index(drop=True), use_container_width=True, height=520)
        if _mlb_api is None:
            st.info("Pitcher ERA boost is available if `MLB-StatsAPI` is installed.")

    with right:
        st.markdown("**Pick any MLB matchup**")
        teams = mlb_rates["team"].sort_values().tolist()
        home = st.selectbox("Home team", teams, index=0, key="mlb_home")
        away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")

        show_prob = st.checkbox("Use today's probable pitchers (if available)", value=True)
        prob_used = prob if show_prob else pd.DataFrame(columns=["team","pitcher","era"])

        if show_prob and not prob_used.empty:
            used = prob_used.loc[prob_used["team"].str.lower().isin([home.lower(), away.lower()])]
            if not used.empty:
                st.markdown("**Probables used**")
                st.dataframe(used.assign(side=lambda d: d["team"].map({home:"Home", away:"Away"}))[["side","team","pitcher","era"]],
                             use_container_width=True, height=140)

        try:
            mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away, prob_used)
            p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
            c1, c2 = st.columns(2)
            with c1: st.metric(f"{home} win %", f"{p_home*100:.1f}%")
            with c2: st.metric(f"{away} win %", f"{p_away*100:.1f}%")
            tag = " (pitcher-adjusted)" if show_prob and not prob_used.empty else ""
            st.caption(f"Expected score{tag}: **{home} {exp_h:.1f} — {away} {exp_a:.1f}** (total {exp_t:.1f})")
        except Exception as e:
            st.error(str(e))

# -------------------------- NFL Props (beta) ----------------------------------
else:
    st.subheader("🏈 NFL Player Props — 2025 REG (beta)")

    with st.expander("What this uses"):
        st.write(
            "- **Recent form**: last 5 games for the player in 2025 weekly stats.\n"
            "- **Defense adjust**: scales the mean by opponent allowed-per-game for that stat.\n"
            "- **Distribution**: Normal(mean, std) simulation (10k draws)."
        )
        st.caption("If 2025 weekly player data isn’t available yet from nflverse, upload a CSV: "
                   "`season,week,player,team,opp,passing_yards,rushing_yards,receiving_yards,receptions,attempts`")

    # try load 2025 weekly; let user upload a CSV fallback
    base = load_weekly_players_2025()
    up = st.file_uploader("Optional: upload weekly players CSV to use instead of auto data", type=["csv"])
    if up is not None:
        try:
            base = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if base.empty:
        st.error("Couldn't load weekly player stats for 2025. Upload a CSV to proceed.")
        st.stop()

    # clean
    for c in ["player","team","opp"]:
        if c in base.columns:
            base[c] = base[c].astype(str)

    prop_name = st.selectbox("Prop type", list(PROPS.keys()), index=0)
    stat_col = PROPS[prop_name]["stat_col"]
    if stat_col not in base.columns:
        st.error(f"This dataset does not have column `{stat_col}`.")
        st.stop()

    # pick team -> player list
    teams = sorted(base["team"].dropna().unique().tolist())
    tsel = st.selectbox("Team", teams, index=0)
    players = sorted(base.loc[base["team"]==tsel, "player"].dropna().unique().tolist())
    psel = st.selectbox("Player", players, index=0)

    # choose opponent (optional)
    opps = sorted(base["opp"].dropna().unique().tolist())
    opp = st.selectbox("Opponent (defense adjust)", ["(none)"] + opps, index=0)
    opp_team = None if opp == "(none)" else opp

    # (optional) Show recent game log
    show_log = st.checkbox("Show player's last 10 games (for this stat)", value=False)
    pgames = base.loc[base["player"]==psel].sort_values(["season","week"])
    if show_log:
        show_cols = ["season","week","team","opp",stat_col]
        st.dataframe(pgames[show_cols].tail(10).reset_index(drop=True), use_container_width=True, height=240)

    # enter line
    default_line = float(pd.to_numeric(pgames[stat_col], errors="coerce").dropna().tail(5).mean() if not pgames.empty else 0.0)
    line = st.number_input(f"{prop_name} line", value=round(default_line,1), step=1.0)

    # simulate
    if st.button("Simulate prop"):
        try:
            p_over, p_under, mu, sd = props_line_sim(pgames, stat_col, opp_team, line)
            c1, c2 = st.columns(2)
            with c1: st.metric("Over %", f"{p_over*100:.1f}%")
            with c2: st.metric("Under %", f"{p_under*100:.1f}%")
            st.caption(f"Model mean ± sd: **{mu:.1f} ± {sd:.1f}** | Opp adj: **{opp_team or 'none'}**")
        except Exception as e:
            st.error(f"Prop simulation failed: {e}")
