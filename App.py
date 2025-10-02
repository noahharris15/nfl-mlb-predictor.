# app.py — NFL + MLB Predictor (2025, stats-only) + Player Props
# Defense EPA table is embedded; optional Injury CSV uploads are supported
# (NFL: team offense/defense multipliers; Props: player + team offense multipliers).

from __future__ import annotations
import io, math, re
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ---- NFL team schedules/scores ----
import nfl_data_py as nfl

# ---- MLB team records (BR) ----
from pybaseball import schedule_and_record

# ======================= EMBEDDED DEFENSE TABLE ===============================
# You can paste a clean two-column CSV (Team, EPA/PLAY) here.
# If left messy or blank, the script will try to extract pairs; if it fails,
# it falls back to a safe, hand-entered EPA map so DEF factors still work.
DEFENSE_CSV_TEXT = """\
Team,EPA/PLAY
MIN,-0.27
JAX,-0.15
GB,-0.13
SF,-0.11
ATL,-0.10
IND,-0.08
LAC,-0.08
DEN,-0.08
LAR,-0.07
SEA,-0.07
PHI,-0.06
TB,-0.05
CAR,-0.05
ARI,-0.03
CLE,-0.02
WAS,-0.02
HOU,0.00
KC,0.01
DET,0.01
LV,0.03
PIT,0.05
CIN,0.05
NO,0.05
BUF,0.05
CHI,0.06
NE,0.09
NYJ,0.10
TEN,0.11
BAL,0.11
NYG,0.13
DAL,0.21
MIA,0.28
"""

# Fallback (used only if DEFENSE_CSV_TEXT cannot be parsed)
DEF_EPA_2025_FALLBACK = {
    "MIN": -0.27, "JAX": -0.15, "GB": -0.13, "SF": -0.11, "ATL": -0.10,
    "IND": -0.08, "LAC": -0.08, "DEN": -0.08, "LAR": -0.07, "SEA": -0.07,
    "PHI": -0.06, "TB": -0.05, "CAR": -0.05, "ARI": -0.03, "CLE": -0.02,
    "WAS": -0.02, "HOU":  0.00, "KC": 0.01, "DET": 0.01, "LV": 0.03,
    "PIT": 0.05, "CIN": 0.05, "NO": 0.05, "BUF": 0.05, "CHI": 0.06,
    "NE": 0.09, "NYJ": 0.10, "TEN": 0.11, "BAL": 0.11, "NYG": 0.13,
    "DAL": 0.21, "MIA": 0.28,
}

# Accept lots of alias codes and map to our keys above.
ALIAS_TO_STD = {
    # PFR-style
    "GNB": "GB", "SFO": "SF", "KAN": "KC", "NWE": "NE", "NOR": "NO", "TAM": "TB",
    "LVR": "LV", "SDG": "LAC", "STL": "LAR",
    # common alternates
    "JAC": "JAX", "WSH": "WAS", "LA": "LAR", "OAK": "LV",
}

def norm_team_code(code: str) -> str:
    c = (code or "").strip().upper()
    return ALIAS_TO_STD.get(c, c)

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6    # small NFL home bump
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

# -------------------------- generic helpers -----------------------------------
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

# ---------------------- Defense loading + factor map --------------------------
def _extract_epa_pairs_loose(text: str) -> Dict[str, float]:
    """
    Very forgiving extractor:
    - First try normal CSV read.
    - If that fails, pull pairs (TEAM, number) via regex scanning.
    """
    text = (text or "").strip()
    if not text:
        return {}

    # 1) Try direct CSV
    try:
        df = pd.read_csv(io.StringIO(text), comment="#")
        cols_low = {str(c).strip().lower(): c for c in df.columns}
        team_candidates = ["team","team_code","abbr","tm"]
        epa_candidates  = ["epa/play","epa per play","epa_play","epa"]
        team_col = next((cols_low[k] for k in team_candidates if k in cols_low), None)
        epa_col  = next((cols_low[k] for k in epa_candidates if k in cols_low), None)
        if team_col is None:
            # If the first col looks like codes, use it
            team_col = df.columns[0]
        if epa_col is None:
            # Choose the last numeric column
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            epa_col = num_cols[-1] if num_cols else df.columns[-1]
        out = {}
        for _, r in df[[team_col, epa_col]].dropna().iterrows():
            code = norm_team_code(str(r[team_col]))
            try:
                out[code] = float(r[epa_col])
            except Exception:
                pass
        if out:
            return out
    except Exception:
        pass

    # 2) Regex scan for TEAM and EPA number on same/adjacent lines
    out = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    last_team = None
    team_pat = re.compile(r'^[A-Za-z .&()-]{2,}$')
    epa_pat = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')

    for ln in lines:
        # short team code
        if re.fullmatch(r'[A-Za-z]{2,4}', ln):
            last_team = norm_team_code(ln)
            continue
        # long team name
        if team_pat.match(ln) and ' ' in ln and len(ln) > 3:
            # Try to convert to common code by first three letters heuristics
            # (if you paste full names, change them to codes in the CSV for best results)
            last_team = norm_team_code(ln.split()[-1][:3].upper())
            continue
        # numeric?
        nums = epa_pat.findall(ln)
        if last_team and nums:
            try:
                out[last_team] = float(nums[0])
            except Exception:
                pass
            last_team = None

    return {k: v for k, v in out.items() if isinstance(v, (int,float))}

def build_def_factor_map(epa_map: Dict[str, float]) -> Dict[str, float]:
    """Convert EPA/play (lower = tougher) to a multiplicative factor (~0.85..1.15)."""
    if not epa_map:
        return {}
    series = pd.Series(epa_map, dtype=float)
    sd = float(series.std(ddof=0))
    if sd < 1e-12:
        # All equal → neutral factors (avoid x1.000 confusion upstream)
        return {k: 1.00 for k in epa_map.keys()}
    mu = float(series.mean())
    z = (series - mu) / sd
    # Lower EPA (better D) => lower factor; clamp ±2σ → 0.85..1.15
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k, v in factors.items()}

def load_embedded_defense() -> Tuple[Dict[str,float], Dict[str,float], str]:
    """Parse DEFENSE_CSV_TEXT; else fallback."""
    epa = _extract_epa_pairs_loose(DEFENSE_CSV_TEXT)
    if epa:
        return build_def_factor_map(epa), epa, "Embedded CSV"
    return build_def_factor_map(DEF_EPA_2025_FALLBACK), DEF_EPA_2025_FALLBACK, "Fallback map"

# ---------------------- Injury CSV parsing (optional) -------------------------
# Supported formats:
# Team-level row (affects NFL page game totals and Props team-off adjustment):
#   Team,OffenseMultiplier,DefenseMultiplier
# Player-level row (affects Props mean directly):
#   Player,Multiplier   OR   Player,Status  (Status in {out,doubtful,questionable,limited,active})
STATUS_TO_MULT = {
    "out": 0.0,
    "doubtful": 0.5,
    "questionable": 0.85,
    "limited": 0.9,
    "probable": 0.95,
    "active": 1.0,
}

def load_injuries_table(upload) -> Tuple[Dict[str,float], Dict[str,float], Dict[str,float]]:
    """Returns (team_off_mult, team_def_mult, player_mult). Missing → 1.0."""
    if upload is None:
        return {}, {}, {}
    # Read any CSV/XLSX
    name = (upload.name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(upload)
    else:
        df = pd.read_csv(upload)

    cols = {str(c).strip().lower(): c for c in df.columns}

    # Team-level
    team_col = next((cols[k] for k in ["team","abbr","team_code","tm"] if k in cols), None)
    off_col  = next((cols[k] for k in ["offensemultiplier","off_mult","offense_mult","off"] if k in cols), None)
    def_col  = next((cols[k] for k in ["defensemultiplier","def_mult","defense_mult","def"] if k in cols), None)

    team_off, team_def = {}, {}
    if team_col and (off_col or def_col):
        for _, r in df.iterrows():
            code = norm_team_code(str(r.get(team_col, "")))
            if not code:
                continue
            if off_col and pd.notna(r.get(off_col)):
                try: team_off[code] = float(r.get(off_col))
                except Exception: pass
            if def_col and pd.notna(r.get(def_col)):
                try: team_def[code] = float(r.get(def_col))
                except Exception: pass

    # Player-level
    player_col = next((cols[k] for k in ["player","name","player_name"] if k in cols), None)
    mult_col   = next((cols[k] for k in ["multiplier","adjmult","adj_multiplier"] if k in cols), None)
    status_col = next((cols[k] for k in ["status","injury_status"] if k in cols), None)

    player_mult = {}
    if player_col and (mult_col or status_col):
        for _, r in df.iterrows():
            p = str(r.get(player_col, "")).strip().lower()
            if not p:
                continue
            if mult_col and pd.notna(r.get(mult_col)):
                try:
                    player_mult[p] = float(r.get(mult_col))
                    continue
                except Exception:
                    pass
            if status_col and pd.notna(r.get(status_col)):
                s = str(r.get(status_col)).strip().lower()
                if s in STATUS_TO_MULT:
                    player_mult[p] = STATUS_TO_MULT[s]

    return team_off, team_def, player_mult

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups (simple UI)
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    # date-like column if present
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

    # upcoming games list (guard for missing cols)
    if {"home_team","away_team"}.issubset(set(sched.columns)):
        filt = sched["home_score"].isna() & sched["away_score"].isna()
        upcoming = sched.loc[filt, ["home_team","away_team"]].copy()
        upcoming["date"] = sched.loc[filt, date_col].astype(str) if date_col else ""
    else:
        upcoming = pd.DataFrame(columns=["home_team","away_team","date"])

    for c in ["home_team","away_team"]:
        if c in upcoming.columns:
            upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+", " ", regex=True)

    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str,
                   team_off_mult: Dict[str,float] | None = None,
                   team_def_mult: Dict[str,float] | None = None) -> Tuple[float,float]:
    team_off_mult = team_off_mult or {}
    team_def_mult = team_def_mult or {}

    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]

    # Try to turn full names into short codes for multipliers (if user uses codes)
    def guess_code(team_name: str) -> str:
        words = team_name.split()
        return norm_team_code(words[-1][:3]) if words else norm_team_code(team_name)

    home_code = guess_code(str(H["team"]))
    away_code = guess_code(str(A["team"]))

    # Offense of side + Defense of opponent both affect expected scoring
    home_pf = float(H["PF_pg"]) * float(team_off_mult.get(home_code, 1.0))
    away_pa = float(A["PA_pg"]) * float(team_def_mult.get(away_code, 1.0))
    mu_home = max(EPS, (home_pf + away_pa)/2.0 + HOME_EDGE_NFL)

    away_pf = float(A["PF_pg"]) * float(team_off_mult.get(away_code, 1.0))
    home_pa = float(H["PA_pg"]) * float(team_def_mult.get(home_code, 1.0))
    mu_away = max(EPS, (away_pf + home_pa)/2.0)

    return mu_home, mu_away

# ==============================================================================
# MLB (2025) — team RS/RA from BR
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
# Player Props — CSVs + embedded defense + injury CSV (optional)
# ==============================================================================
def _yardage_column_guess(df: pd.DataFrame, pos: str) -> str:
    prefer = ["Y/G","Yds/G","YDS/G","Yards/G","PY/G","RY/G","Rec Y/G",
              "Yds","Yards","yds","yards"]
    low = [c.lower() for c in df.columns]
    for wanted in [p.lower() for p in prefer]:
        if wanted in low:
            return df.columns[low.index(wanted)]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1]

def _player_column_guess(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in ("player","name"):
            return c
    return df.columns[0]

def _team_column_guess(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in ("team","tm","abbr","team_code"):
            return c
    return None

def _estimate_sd(mean_val: float, pos: str) -> float:
    mean_val = float(mean_val)
    if pos == "QB": return max(35.0, 0.60 * mean_val)
    if pos == "RB": return max(20.0, 0.75 * mean_val)
    return max(22.0, 0.85 * mean_val)  # WR

def run_prop_sim(mean_yards: float, line: float, sd: float) -> Tuple[float,float]:
    sd = max(5.0, float(sd))
    z = (line - mean_yards) / sd
    p_over = float(1.0 - 0.5*(1.0 + math.erf(z / math.sqrt(2))))
    return np.clip(p_over, 0.0, 1.0), np.clip(1.0 - p_over, 0.0, 1.0)

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Team pages use **2025 scoring rates only** (NFL: PF/PA; MLB: RS/RA). "
    "Player Props: upload your QB/RB/WR CSVs, pick a player, set a line. "
    "Defense is **embedded**; Injuries CSV is optional but used if provided."
)

page = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()

    inj_file = st.file_uploader("Optional: Injury CSV (Team Offense/Defense multipliers)", type=["csv","xlsx"], key="inj_nfl")
    team_off_mult, team_def_mult, _ = load_injuries_table(inj_file)

    if not upcoming.empty and {"home_team","away_team"}.issubset(set(upcoming.columns)):
        labels = [f"{r['home_team']} vs {r['away_team']} — {r.get('date','')}" for _, r in upcoming.iterrows()]
        sel = st.selectbox("Select upcoming game", labels) if labels else None
        if sel:
            try:
                teams_part = sel.split(" — ")[0]
                home, away = [t.strip() for t in teams_part.split(" vs ")]
            except Exception:
                home = away = None
        else:
            home = away = None
    else:
        st.info("No upcoming games list available — pick any two teams:")
        home = st.selectbox("Home team", rates["team"].tolist())
        away = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != home])

    if home and away:
        try:
            mu_h, mu_a = nfl_matchup_mu(rates, home, away, team_off_mult, team_def_mult)
            pH, pA, mH, mA = _poisson_sim(mu_h, mu_a)
            st.markdown(
                f"**{home}** vs **{away}** — "
                f"Expected points: {mH:.1f}–{mA:.1f} · "
                f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
            )
            if team_off_mult or team_def_mult:
                st.caption("Injuries applied: "
                           f"Off mult(H/A)={team_off_mult.get(home[:3].upper(),'—')}/{team_off_mult.get(away[:3].upper(),'—')}, "
                           f"Def mult(H/A)={team_def_mult.get(home[:3].upper(),'—')}/{team_def_mult.get(away[:3].upper(),'—')}")
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
    pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)
    st.markdown(
        f"**{t1}** vs **{t2}** — "
        f"Expected runs: {mH:.1f}–{mA:.1f} · "
        f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**"
    )

# -------------------------- Player Props page --------------------------
else:
    st.subheader("🎯 Player Props — drop in your CSVs")

    # Build defense factors from embedded CSV (or fallback)
    DEF_FACTOR_2025, DEF_EPA_USED, DEF_SOURCE = load_embedded_defense()
    st.caption(f"Defense source in use: **{DEF_SOURCE}**")

    # Optional Injuries CSV (player + team offense multipliers)
    inj_file = st.file_uploader("Optional: Injury CSV (Player & Team offense multipliers)", type=["csv","xlsx"], key="inj_props")
    team_off_mult, _, player_mult = load_injuries_table(inj_file)

    # Player CSV uploads
    c1, c2, c3 = st.columns(3)
    with c1:
        qb_up = st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2:
        rb_up = st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3:
        wr_up = st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

    def _read_any_table(up):
        if up is None: return pd.DataFrame()
        nm = (up.name or "").lower()
        if nm.endswith(".xlsx") or nm.endswith(".xls"):
            return pd.read_excel(up)
        return pd.read_csv(up)

    dfs = {}
    if qb_up: dfs["QB"] = _read_any_table(qb_up).copy()
    if rb_up: dfs["RB"] = _read_any_table(rb_up).copy()
    if wr_up: dfs["WR"] = _read_any_table(wr_up).copy()

    if not dfs:
        st.info("Upload at least one of QB/RB/WR CSVs to begin.")
        st.stop()

    pos = st.selectbox("Market", ["QB","RB","WR"])
    df = dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded yet.")
        st.stop()

    name_col = _player_column_guess(df)
    yard_col = _yardage_column_guess(df, pos)
    team_col = _team_column_guess(df)

    # Show only names in the dropdown
    name_list = df[name_col].astype(str).tolist()
    player_name = st.selectbox("Player", name_list)

    # Opponent code input (aliases OK)
    opp_in = st.text_input("Opponent team code (e.g., DAL, PHI). Aliases like KAN/NOR/GNB/SFO are OK.", value="")
    opp = norm_team_code(opp_in)

    # Selected row + base mean from CSV
    row = df.loc[df[name_col] == player_name].head(1)
    csv_mean = float(pd.to_numeric(row[yard_col], errors="coerce").fillna(0).mean()) if not row.empty else 0.0

    # Try to infer the player's own team code from CSV if present
    player_team_code = None
    if team_col and not row.empty:
        player_team_code = norm_team_code(str(row.iloc[0][team_col]))

    # Player multiplier from injuries CSV (by name)
    mult_player = player_mult.get(player_name.strip().lower(), 1.0)

    # Team offense multiplier (if player's team present in injuries CSV)
    mult_team_off = team_off_mult.get(player_team_code, 1.0) if player_team_code else 1.0

    # Defense factor from embedded EPA map
    def_factor = DEF_FACTOR_2025.get(opp, 1.00) if opp else 1.00

    # Combine adjustments (player * team_offense * defense)
    adj_mean = csv_mean * mult_player * mult_team_off * def_factor

    # Default line = CSV mean (editable)
    line = st.number_input("Yardage line", value=round(csv_mean or 0.0, 2), step=0.5)

    # Simple SD by position
    est_sd = _estimate_sd(csv_mean, pos)

    # Simulation
    p_over, p_under = run_prop_sim(adj_mean, line, est_sd)

    st.success(
        f"**{player_name} — {('Passing' if pos=='QB' else 'Rush' if pos=='RB' else 'Receiving')} Yards**  \n"
        f"CSV mean: **{csv_mean:.1f}** · Adj = Player×TeamOff×Defense = "
        f"**{mult_player:.2f} × {mult_team_off:.2f} × {def_factor:.3f}** → "
        f"Adjusted mean: **{adj_mean:.1f}**  \n"
        f"Line: **{line:.1f}** → **P(over) = {100*p_over:.1f}%**, **P(under) = {100*p_under:.1f}%**"
    )

    with st.expander("Show selected CSV row"):
        st.dataframe(row if not row.empty else df.head(5))

    with st.expander("Defense factors in use"):
        show = pd.DataFrame({
            "TEAM": list(DEF_EPA_USED.keys()),
            "EPA/play": list(DEF_EPA_USED.values()),
            "DEF_FACTOR": [build_def_factor_map(DEF_EPA_USED)[k] for k in DEF_EPA_USED.keys()],
        }).sort_values("DEF_FACTOR")
        st.dataframe(show.reset_index(drop=True))
