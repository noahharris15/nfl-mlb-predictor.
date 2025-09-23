# app.py — NFL + MLB Predictor (2025, stats-only) + Player Props with optional defense CSV
# Clean restart build

from __future__ import annotations
import math
from datetime import date
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ---- NFL data (for team pages) ----
import nfl_data_py as nfl

# ---- MLB team records + probables (optional) ----
from pybaseball import schedule_and_record
try:
    import statsapi
    HAS_STATSAPI = True
except Exception:
    HAS_STATSAPI = False

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6    # small NFL home bump in Poisson mean (points)
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

# -------------------------- helpers ------------------------------------------
def _poisson_sim(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # tiny home tiebreak
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

def _team_code_clean(s: str) -> str:
    return (s or "").strip().upper()

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups (matchups-only UI)
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    # identify any date-like column in schedule
    date_col: Optional[str] = None
    for c in ("gameday", "game_date", "start_time"):
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
        # prior if season hasn't started in the data source
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

    # upcoming games (may be empty)
    upcoming_cols = [c for c in ["home_team","away_team"] if c in sched.columns]
    if not upcoming_cols:
        upcoming = pd.DataFrame(columns=["home_team","away_team","date"])
    else:
        filt = sched["home_score"].isna() & sched["away_score"].isna()
        upcoming = sched.loc[filt, upcoming_cols].copy()
        if date_col and date_col in sched.columns:
            upcoming["date"] = sched[date_col].astype(str)
        else:
            upcoming["date"] = ""

    # tidy strings
    for c in ["home_team","away_team"]:
        if c in upcoming.columns:
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
# MLB (2025) — team RS/RA from BR + (optional) probable pitchers ERA
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
            else:
                sar["R"] = sar["R"].astype(float)
                sar["RA"] = sar["RA"].astype(float)
                games = int(len(sar))
                RS_pg = float(sar["R"].sum() / games)
                RA_pg = float(sar["RA"].sum() / games)
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean())
        league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9 * df["RS_pg"] + 0.1 * league_rs
        df["RA_pg"] = 0.9 * df["RA_pg"] + 0.1 * league_ra
    return df

# ==============================================================================
# Player Props — CSVs + optional defense table
# ==============================================================================
def _read_any_table(upload) -> pd.DataFrame:
    """Accept CSV or Excel."""
    if upload is None:
        return pd.DataFrame()
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(upload)
    # try CSV by default
    return pd.read_csv(upload)

def _defense_factor_table(def_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with columns: TEAM (abbr, upper) and DEF_FACTOR (float).
    If the upload already has DEF_FACTOR, use it.
    Else, derive from EPA/PLAY (lower EPA = tougher = factor < 1).
    """
    if def_df is None or def_df.empty:
        return pd.DataFrame(columns=["TEAM","DEF_FACTOR"])

    # guess team col
    team_col = None
    for c in def_df.columns:
        cl = str(c).strip().lower()
        if cl in ("team","team_code","abbr","tm","team_abbr","defense"):
            team_col = c
            break
    if team_col is None:
        # often first column is team name
        team_col = def_df.columns[0]

    out = pd.DataFrame({"TEAM": def_df[team_col].astype(str).str.upper().str[:3]})

    # if DEF_FACTOR exists, use it
    dfc = None
    for c in def_df.columns:
        if str(c).strip().upper() == "DEF_FACTOR":
            dfc = c
            break
    if dfc is not None:
        out["DEF_FACTOR"] = pd.to_numeric(def_df[dfc], errors="coerce")
        out = out.dropna(subset=["DEF_FACTOR"]).drop_duplicates(subset=["TEAM"])
        return out

    # else, look for EPA/PLAY
    epa_col = None
    for c in def_df.columns:
        if "epa" in str(c).lower() and "play" in str(c).lower():
            epa_col = c
            break
    if epa_col is None:
        # fallback: neutral
        out["DEF_FACTOR"] = 1.00
        return out.drop_duplicates(subset=["TEAM"])

    # map EPA/play to factor ~ 0.85..1.15 (lower EPA => stronger => smaller factor)
    epa = pd.to_numeric(def_df[epa_col], errors="coerce")
    mu, sd = float(epa.mean()), float(epa.std(ddof=0) or 1.0)
    z = (epa - mu) / (sd if sd > 1e-9 else 1.0)
    # convert z to factor with soft clamp
    factor = 1.0 + np.clip(z, -2.0, 2.0) * 0.075  # ~±15% across ±2σ
    out["DEF_FACTOR"] = factor.astype(float)
    return out.dropna(subset=["DEF_FACTOR"]).drop_duplicates(subset=["TEAM"])

def _yardage_column_guess(df: pd.DataFrame, pos: str) -> str:
    # very forgiving column names
    cand = [c for c in df.columns if str(c).lower() in ("y/g","yd/g","yards/g","avg_yards","py/g","ry/g","rec y/g","yds/g")]
    if cand:
        return cand[0]
    cand2 = [c for c in df.columns if str(c).lower() in ("pass yds","pass_yds","passyds","passing yards","py","pyards") and pos=="QB"]
    if cand2:
        return cand2[0]
    cand3 = [c for c in df.columns if str(c).lower() in ("yds","yards")]
    if cand3:
        return cand3[0]
    # last resort: try first numeric column after the name
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1]

def _player_column_guess(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower() in ("player","name"):
            return c
    return df.columns[0]

def _estimate_sd(mean_val: float, pos: str) -> float:
    mean_val = float(mean_val)
    if pos == "QB":
        # passing yards fluctuate a lot
        return max(35.0, 0.60 * mean_val)
    if pos == "RB":
        return max(20.0, 0.75 * mean_val)
    # WR receiving yards are spiky
    return max(22.0, 0.85 * mean_val)

def run_prop_sim(mean_yards: float, line: float, sd: float) -> Tuple[float,float]:
    # Normal approx; clamp sd
    sd = max(5.0, float(sd))
    z = (line - mean_yards) / sd
    # P(X > line) under Normal
    p_over = float(1.0 - 0.5*(1.0 + math.erf(z / math.sqrt(2))))
    p_over = np.clip(p_over, 0.0, 1.0)
    return p_over, 1.0 - p_over

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Team pages use **2025 scoring rates only** (NFL: PF/PA; MLB: RS/RA). "
    "Player Props: upload your QB/RB/WR CSVs, pick a player, set a line. "
    "Optionally upload a **Defense CSV** (with `EPA/PLAY` or `DEF_FACTOR`) to auto-adjust."
)

page = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()

    # Build choices safely (no iloc mismatch)
    if not upcoming.empty and all(c in upcoming.columns for c in ["home_team","away_team"]):
        labels = []
        for _, r in upcoming.iterrows():
            lab = f"{r['home_team']} vs {r['away_team']} — {r.get('date','')}"
            labels.append(lab)
        sel = st.selectbox("Select upcoming game", labels) if labels else None
        if sel:
            # parse label back to teams
            try:
                teams_part = sel.split(" — ")[0]
                home, away = [t.strip() for t in teams_part.split(" vs ")]
            except Exception:
                home = away = None
        else:
            home = away = None
    else:
        st.info("No upcoming games found. Pick any two teams to compare:")
        t1 = st.selectbox("Home team", rates["team"].tolist())
        t2 = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != t1])
        home, away = t1, t2

    if home and away:
        try:
            mu_h, mu_a = nfl_matchup_mu(rates, home, away)
            pH, pA, mH, mA = _poisson_sim(mu_h, mu_a)
            st.markdown(
                f"**{home}** vs **{away}** — "
                f"Expected points: {mH:.1f}–{mA:.1f} · "
                f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
            )
        except Exception as e:
            st.error(str(e))

# -------------------------- MLB page --------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 REG season (team scoring rates only)")
    try:
        rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't load MLB team rates: {e}")
        st.stop()

    t1 = st.selectbox("Home team", rates["team"].tolist())
    t2 = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != t1])
    H = rates.loc[rates["team"] == t1].iloc[0]
    A = rates.loc[rates["team"] == t2].iloc[0]
    mu_home = (H["RS_pg"] + A["RA_pg"]) / 2.0
    mu_away = (A["RS_pg"] + H["RA_pg"]) / 2.0

    # optional probables adjustment if available
    home_era = away_era = None
    if HAS_STATSAPI:
        try:
            today = date.today().strftime("%Y-%m-%d")
            sched = statsapi.schedule(date=today)
            for g in sched:
                # try fuzzy match by substring team names; keep simple
                h = (g.get("home_name") or "").lower()
                a = (g.get("away_name") or "").lower()
                if t1.split()[-1].lower() in h and t2.split()[-1].lower() in a:
                    home_era = None
                    away_era = None
                    break
        except Exception:
            pass

    pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)
    st.markdown(
        f"**{t1}** vs **{t2}** — "
        f"Expected runs: {mH:.1f}–{mA:.1f} · "
        f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**"
    )

# -------------------------- Player Props page --------------------------
else:
    st.subheader("🎯 Player Props — upload CSVs, pick player, set a line")

    c1, c2, c3 = st.columns(3)
    with c1:
        qb_up = st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2:
        rb_up = st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3:
        wr_up = st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

    def_up = st.file_uploader("Optional: Defense CSV (has EPA/PLAY or DEF_FACTOR)", type=["csv","xlsx"], key="def")
    df_def = _defense_factor_table(_read_any_table(def_up))

    # Build a pool of players
    dfs = {}
    if qb_up: dfs["QB"] = _read_any_table(qb_up).copy()
    if rb_up: dfs["RB"] = _read_any_table(rb_up).copy()
    if wr_up: dfs["WR"] = _read_any_table(wr_up).copy()

    if not dfs:
        st.info("Upload at least one of QB/RB/WR CSVs to begin.")
        st.stop()

    # choose position first
    pos = st.selectbox("Market", ["QB","RB","WR"])
    df = dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded yet.")
        st.stop()

    name_col = _player_column_guess(df)
    yard_col = _yardage_column_guess(df, pos)

    players = df[name_col].astype(str).tolist()
    player = st.selectbox("Player", players)

    # opponent code text (optional)
    opp = st.text_input("Opponent team code (optional — e.g., DAL, PHI). Leave blank for league-average defense.").strip().upper()

    # yardage line
    default_mean = float(pd.to_numeric(df.loc[df[name_col]==player, yard_col], errors="coerce").fillna(0).mean())
    line = st.number_input("Yardage line", value=round(default_mean or 0.0, 2), step=0.5)

    # --- compute mean + sd from CSV row ---
    row = df.loc[df[name_col]==player].head(1)
    csv_mean = float(pd.to_numeric(row[yard_col], errors="coerce").fillna(0).mean()) if not row.empty else 0.0
    est_sd = _estimate_sd(csv_mean, pos)

    # defense factor (1.00 if not provided or not found)
    def_factor = 1.0
    if opp and not df_def.empty:
        code = _team_code_clean(opp)
        r = df_def.loc[df_def["TEAM"] == code]
        if not r.empty:
            try:
                val = float(r.iloc[0]["DEF_FACTOR"])
                if math.isfinite(val) and 0.3 <= val <= 2.0:
                    def_factor = float(val)
            except Exception:
                pass

    adj_mean = csv_mean * def_factor

    # run simulation
    p_over, p_under = run_prop_sim(adj_mean, line, est_sd)

    # show result
    st.success(
        f"**{player} — {('Passing' if pos=='QB' else 'Rush' if pos=='RB' else 'Receiving')} Yards**  \n"
        f"CSV mean: **{csv_mean:.1f}** · Defense factor: **×{def_factor:.3f}** → "
        f"Adjusted mean: **{adj_mean:.1f}**  \n"
        f"Line: **{line:.1f}** → **P(over) = {100*p_over:.1f}%**, **P(under) = {100*p_under:.1f}%**"
    )

    with st.expander("Show player row used"):
        st.dataframe(row if not row.empty else df.head(5))

    if not df_def.empty:
        with st.expander("Defense table (normalized)"):
            st.dataframe(df_def.sort_values("DEF_FACTOR"))
