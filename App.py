# app.py — NFL + MLB predictor (2025 only, stats-only) with MLB probable-pitcher ERA (robust ID lookup)

import math
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from typing import Optional, Tuple, Dict

# ---- NFL (nfl_data_py) -------------------------------------------------------
import nfl_data_py as nfl

# ---- MLB (pybaseball + statsapi) ---------------------------------------------
from pybaseball import schedule_and_record
try:
    import statsapi  # for probables + pitcher stats (ERA)
    HAS_STATSAPI = True
except Exception:
    HAS_STATSAPI = False

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6   # ~0.6 pts to the home offense mean (small HFA)
EPS = 1e-9

# Stable Baseball-Reference team IDs for 2025 (BR codes)
MLB_TEAMS_2025: Dict[str, str] = {
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

# -------------------------- generic: Poisson sim ------------------------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # tiny home tiebreak (NFL-ish)
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean()), float((h + a).mean())

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups (matchups-only UI)
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    date_col: Optional[str] = None
    for c in ("gameday", "game_date"):
        if c in sched.columns:
            date_col = c
            break

    played = sched.dropna(subset=["home_score", "away_score"])
    home = played.rename(columns={
        "home_team": "team", "away_team": "opp", "home_score": "pf", "away_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    away = played.rename(columns={
        "away_team": "team", "home_team": "opp", "away_score": "pf", "home_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 45.0 / 2.0
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
            games=("pf","size"), PF=("pf","sum"), PA=("pa","sum")
        )
        rates = pd.DataFrame({
            "team": team["team"],
            "PF_pg": team["PF"] / team["games"],
            "PA_pg": team["PA"] / team["games"],
        })
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"]/4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink) * rates["PF_pg"] + shrink * prior
        rates["PA_pg"] = (1 - shrink) * rates["PA_pg"] + shrink * prior

    upcoming_cols = ["home_team","away_team"]
    if date_col:
        upcoming_cols.append(date_col)
    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][upcoming_cols].copy()
    if date_col:
        upcoming = upcoming.rename(columns={date_col: "date"})
    else:
        upcoming["date"] = ""

    for c in ["home_team","away_team"]:
        upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+", " ", regex=True)

    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"])/2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"])/2.0)
    return mu_home, mu_away

# ==============================================================================
# MLB (2025) — team RS/RA from BR + probable pitchers with ERA (statsapi)
# ==============================================================================

@st.cache_data(show_spinner=False)
def mlb_team_rates_2025() -> pd.DataFrame:
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

def _statsapi_player_era(pid: int, season: int = 2025) -> Optional[float]:
    """Return pitcher's season ERA for the given season (fallback to prior season)."""
    if not HAS_STATSAPI:
        return None
    for yr in (season, season-1):
        try:
            d = statsapi.player_stat_data(pid, group="pitching", type="season", season=str(yr))
            for s in d.get("stats", []):
                era_str = s.get("stats", {}).get("era")
                if era_str and era_str not in ("-", "--", "—"):
                    try:
                        return float(era_str)
                    except Exception:
                        continue
        except Exception:
            continue
    return None

def _lookup_pitcher_id_by_name(name: str) -> Optional[int]:
    """Fallback: use name to find a pitcher ID if schedule didn't include it."""
    if not HAS_STATSAPI or not name:
        return None
    try:
        candidates = statsapi.lookup_player(name)
        # Prefer exact fullName match, else first with a 'P' position if present, else first
        exact = [c for c in candidates if str(c.get("fullName","")).lower() == name.lower()]
        if exact:
            return int(exact[0]["id"])
        pitchers = [c for c in candidates if str(c.get("primaryPosition",{}).get("abbreviation","")).upper() == "P"]
        if pitchers:
            return int(pitchers[0]["id"])
        if candidates:
            return int(candidates[0]["id"])
    except Exception:
        return None
    return None

def _today_probables() -> pd.DataFrame:
    """Return probables for today's MLB schedule with pitcher IDs + ERA."""
    cols = ["side","team","pitcher","ERA"]
    if not HAS_STATSAPI:
        return pd.DataFrame(columns=cols)

    today = date.today().strftime("%Y-%m-%d")
    try:
        sched = statsapi.schedule(date=today)
    except Exception:
        return pd.DataFrame(columns=cols)

    rows = []
    for g in sched:
        home_team = g.get("home_name") or g.get("home_name_full")
        away_team = g.get("away_name") or g.get("away_name_full")

        # Official keys used by statsapi for probables
        home_name = g.get("home_probable_pitcher") or g.get("probable_pitcher_home") or g.get("probable_pitcher")
        away_name = g.get("away_probable_pitcher") or g.get("probable_pitcher_away")

        home_id = g.get("home_probable_pitcher_id") or g.get("probable_pitcher_home_id")
        away_id = g.get("away_probable_pitcher_id") or g.get("probable_pitcher_away_id")

        # Fallback: lookup id by name if missing
        if not home_id and home_name:
            home_id = _lookup_pitcher_id_by_name(home_name)
        if not away_id and away_name:
            away_id = _lookup_pitcher_id_by_name(away_name)

        if home_team and home_name:
            era = _statsapi_player_era(int(home_id)) if home_id else None
            rows.append({"side":"Home","team":home_team,"pitcher":home_name,"ERA":era})
        if away_team and away_name:
            era = _statsapi_player_era(int(away_id)) if away_id else None
            rows.append({"side":"Away","team":away_team,"pitcher":away_name,"ERA":era})

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=cols)
    return df.drop_duplicates(subset=["side","team"]).reset_index(drop=True)

def _apply_pitcher_adjustment(mu_home: float, mu_away: float,
                              home_era: Optional[float], away_era: Optional[float],
                              league_era: float = 4.30) -> Tuple[float,float]:
    """
    Adjust team Poisson means by ERA ratio:
      adjusted_runs = base_runs * (league_era / pitcher_era)
    Guard against extreme ERAs and None.
    """
    def adj(mu: float, era: Optional[float]) -> float:
        if era is None or era <= 0:
            return mu
        factor = league_era / float(era)
        factor = float(np.clip(factor, 0.6, 1.4))  # clamp effect
        return max(EPS, mu * factor)

    return adj(mu_home, home_era), adj(mu_away, away_era)

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Win probabilities are simulated from **team scoring rates only** "
    "(NFL: points for/against in 2025; MLB: runs scored/allowed in 2025). "
    "No injuries, travel, depth charts, or betting data."
)

sport = st.radio("Pick a sport", ["NFL", "MLB"], horizontal=True)

# -------------------------- NFL page (matchups only) --------------------------
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
        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with c2:
            st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}** (total {exp_t:.1f})")
    except Exception as e:
        st.error(str(e))

# -------------------------- MLB page (probables + ERA) ------------------------
else:
    st.subheader("⚾ MLB — 2025 season")

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
            height=500
        )

    with right:
        st.markdown("**Pick any MLB matchup**")
        teams = mlb_rates["team"].sort_values().tolist()
        if not teams:
            st.info("No MLB team data yet.")
            st.stop()

        home = st.selectbox("Home team", teams, index=0, key="mlb_home")
        away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")

        use_prob = st.checkbox("Use today's probable pitchers (if available)", value=True)

        prob_tbl = None
        h_era = a_era = None
        if use_prob:
            probables = _today_probables()
            if not probables.empty:
                h_row = probables[(probables["side"] == "Home") & (probables["team"].str.lower() == home.lower())]
                a_row = probables[(probables["side"] == "Away") & (probables["team"].str.lower() == away.lower())]
                prob_tbl = pd.DataFrame(columns=["side","team","pitcher","ERA"])
                if not h_row.empty:
                    h_era = h_row.iloc[0]["ERA"]
                    prob_tbl = pd.concat([prob_tbl, h_row[["side","team","pitcher","ERA"]]], ignore_index=True)
                if not a_row.empty:
                    a_era = a_row.iloc[0]["ERA"]
                    prob_tbl = pd.concat([prob_tbl, a_row[["side","team","pitcher","ERA"]]], ignore_index=True)

        try:
            mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away)
            if use_prob and (h_era is not None or a_era is not None):
                mu_h, mu_a = _apply_pitcher_adjustment(mu_h, mu_a, h_era, a_era)

            if prob_tbl is not None and not prob_tbl.empty:
                st.markdown("**Probables used**")
                # pretty-print ERA with 2 decimals, leave blank if None
                pt = prob_tbl.copy()
                pt["ERA"] = pt["ERA"].apply(lambda x: "" if (x is None or (isinstance(x,float) and math.isnan(x))) else f"{float(x):.2f}")
                st.dataframe(pt.reset_index(drop=True), use_container_width=True, height=140)

            p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
            with c2:
                st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")

            foot = "Expected score"
            if (h_era is not None) or (a_era is not None):
                foot += " (pitcher-adjusted)"
            st.caption(f"{foot}: **{home} {exp_h:.1f} — {away} {exp_a:.1f}** (total {exp_t:.1f})")

        except Exception as e:
            st.error(str(e))
