# app.py — NFL + MLB Predictor (2025, stats-only) + Player Props
# - NFL & MLB pages unchanged in spirit (team scoring only)
# - Defense CSV is EMBEDDED (paste into DEFENSE_CSV_TEXT)
# - Player Props uses embedded defense factors
# - OPTIONAL: Injury report upload on NFL & Player Props pages

from __future__ import annotations
import io
import math
import re
from datetime import date
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ---- NFL (team pages) ----
import nfl_data_py as nfl

# ---- MLB team records (BR) ----
from pybaseball import schedule_and_record

# ======================= PASTE YOUR DEFENSE CSV HERE ==========================
# Acceptable headers include: Team,EPA/PLAY  or  TEAM,epa_play  or  team_code,epa
# Replace ONLY the block below with your full CSV text (including header row)
DEFENSE_CSV_TEXT = """\
Team,EPA/PLAY
# Team   Season  EPA/Play ...
Minnesota Vikings,-0.27
Jacksonville Jaguars,-0.15
Green Bay Packers,-0.13
San Francisco 49ers,-0.11
Atlanta Falcons,-0.10
Indianapolis Colts,-0.08
Los Angeles Chargers,-0.08
Denver Broncos,-0.08
Los Angeles Rams,-0.07
Seattle Seahawks,-0.07
Philadelphia Eagles,-0.06
Tampa Bay Buccaneers,-0.05
Carolina Panthers,-0.05
Arizona Cardinals,-0.03
Cleveland Browns,-0.02
Washington Commanders,-0.02
Houston Texans,0.00
Kansas City Chiefs,0.01
Detroit Lions,0.01
Las Vegas Raiders,0.03
Pittsburgh Steelers,0.05
Cincinnati Bengals,0.05
New Orleans Saints,0.05
Buffalo Bills,0.05
Chicago Bears,0.06
New England Patriots,0.09
New York Jets,0.10
Tennessee Titans,0.11
Baltimore Ravens,0.11
New York Giants,0.13
Dallas Cowboys,0.21
Miami Dolphins,0.28
"""

# Fallback (used only if DEFENSE_CSV_TEXT is empty or unparsable)
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

TEAM_NAME_TO_CODE = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF",
    "SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
}

def norm_team_code(code: str) -> str:
    if not code: return ""
    c = (code or "").strip().upper()
    return ALIAS_TO_STD.get(c, c)

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6    # small NFL home bump
EPS = 1e-9

# MLB BR team names (for schedule_and_record)
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
def _def_find_team(text: str) -> str | None:
    t = str(text or "").upper()
    for name, code in TEAM_NAME_TO_CODE.items():
        if t.startswith(name):
            return code
    # maybe it's already a code or clean name
    t_clean = re.sub(r"[^A-Z ]","",t).strip()
    if t_clean in TEAM_NAME_TO_CODE: return TEAM_NAME_TO_CODE[t_clean]
    if t_clean in TEAM_NAME_TO_CODE.values(): return t_clean
    return None

def _def_epa_from_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    cols_low = {str(c).strip().lower(): c for c in df.columns}
    team_candidates = ["team", "team_code", "def_team", "abbr", "tm", "defense", "opponent", "opp", "code"]
    epa_candidates  = ["epa/play", "epa per play", "epa_play", "def_epa", "def epa", "epa", "epa_per_play"]

    team_col = next((cols_low[k] for k in team_candidates if k in cols_low), None)
    if team_col is None:
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                team_col = c; break

    epa_col = next((cols_low[k] for k in epa_candidates if k in cols_low), None)
    if epa_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        epa_col = num_cols[-1] if num_cols else None

    if team_col is None or epa_col is None:
        return {}

    out = {}
    for _, r in df[[team_col, epa_col]].dropna().iterrows():
        code = _def_find_team(str(r[team_col])) or norm_team_code(str(r[team_col]))
        try:
            out[code] = float(r[epa_col])
        except Exception:
            continue
    return out

def build_def_factor_map(epa_map: Dict[str, float]) -> Dict[str, float]:
    # Convert EPA/play (lower = tougher) → multiplicative factor (~0.85..1.15)
    if not epa_map:
        return {}
    series = pd.Series(epa_map, dtype=float)
    mu, sd = float(series.mean()), float(series.std(ddof=0) or 1.0)
    z = (series - mu) / (sd if sd > 1e-9 else 1.0)
    # Lower EPA (better D) => lower factor; clamp ±2σ → 0.85..1.15
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k, v in factors.items()}

def load_embedded_defense() -> Tuple[Dict[str,float], Dict[str,float], str]:
    text = (DEFENSE_CSV_TEXT or "").strip()
    if text:
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            except Exception:
                df = pd.read_fwf(io.StringIO(text))
        epa_map = _def_epa_from_df(df)
        if epa_map:
            return build_def_factor_map(epa_map), epa_map, "Embedded CSV (in script)"
    # fallback
    return build_def_factor_map(DEF_EPA_2025_FALLBACK), DEF_EPA_2025_FALLBACK, "Embedded fallback"

# Precompute defense factors once
DEF_FACTOR_2025, DEF_EPA_USED, DEF_SOURCE = load_embedded_defense()

# -------------------------- Injury parsing ------------------------------------
POS_OFF = {"QB","RB","WR","TE","T","G","C","OL"}
POS_DEF = {"DE","DT","EDGE","LB","CB","S","DB","NT","DL"}
SEV_WEIGHT = {
    "OUT": 1.00, "DOUBTFUL": 0.75, "QUESTIONABLE": 0.40,
    "DNP": 0.35, "-", 0.0, "": 0.0, None: 0.0
}

def _norm_status(row) -> float:
    gs = str(row.get("Game Status","")).strip().upper()
    if gs in SEV_WEIGHT: return SEV_WEIGHT[gs]
    s = " ".join(str(row.get(c,"")) for c in row.keys()).upper()
    if "OUT" in s: return SEV_WEIGHT["OUT"]
    if "DOUBTFUL" in s: return SEV_WEIGHT["DOUBTFUL"]
    if "DNP" in s and "FULL" not in s: return SEV_WEIGHT["DNP"]
    if "QUESTIONABLE" in s: return SEV_WEIGHT["QUESTIONABLE"]
    return 0.0

def _safe_read_injury(upload) -> pd.DataFrame:
    if upload is None: return pd.DataFrame()
    try:
        df = pd.read_csv(upload); 
        if df.shape[1] >= 3: return df
    except Exception: pass
    try:
        upload.seek(0)
        df = pd.read_csv(upload, sep=None, engine="python")
        if df.shape[1] >= 3: return df
    except Exception: pass
    try:
        upload.seek(0)
        df = pd.read_fwf(upload); 
        return df
    except Exception:
        return pd.DataFrame()

def _find_team_in_text(text: str) -> str | None:
    t = str(text or "").upper()
    for name, code in TEAM_NAME_TO_CODE.items():
        if t.startswith(name):
            return code
    t_clean = re.sub(r"[^A-Z ]","",t).strip()
    if t_clean in TEAM_NAME_TO_CODE: return TEAM_NAME_TO_CODE[t_clean]
    if t_clean in TEAM_NAME_TO_CODE.values(): return t_clean
    return None

def parse_injury_upload(upload) -> dict:
    """
    Returns: injuries = { 'SF': {'off_mult': 0.92, 'def_opp_scoring_mult': 1.08}, ... }
    """
    df = _safe_read_injury(upload)
    if df.empty: return {}

    df_cols = {str(c).strip(): c for c in df.columns}
    rows = [{k: r[v] for k, v in df_cols.items()} for _, r in df.iterrows()]

    impacts: dict[str, dict] = {}
    for r in rows:
        # try explicit Team column
        team_code = None
        for k in r.keys():
            if str(k).strip().lower().startswith("team"):
                team_code = _find_team_in_text(r[k])
                if team_code: break
        if not team_code:
            # fallback from the first field (smashed text)
            first_field = next(iter(r.values())) if r else ""
            team_code = _find_team_in_text(first_field)

        if not team_code:
            continue

        # position
        pos = str(r.get("Pos", "") or r.get("Position","")).upper()
        if not pos:
            m = re.search(r"\b(QB|RB|WR|TE|CB|S|LB|DE|DT|EDGE|OL|T|G|C|DB|NT|DL)\b", " ".join(map(str,r.values())).upper())
            pos = m.group(1) if m else ""

        sev = _norm_status(r)
        if sev <= 0: 
            continue

        off_delta = 0.0
        def_delta = 0.0
        if pos == "QB":
            off_delta += -0.30 * sev
        elif pos == "RB":
            off_delta += -0.07 * sev
        elif pos == "WR":
            off_delta += -0.08 * sev
        elif pos == "TE":
            off_delta += -0.05 * sev
        elif pos in {"T","G","C","OL"}:
            off_delta += -0.03 * sev

        if pos in {"CB","S","DB"}:
            def_delta += +0.05 * sev
        elif pos in {"DE","EDGE","DT","NT","DL","LB"}:
            def_delta += +0.04 * sev

        entry = impacts.setdefault(team_code, {"off": 0.0, "def": 0.0})
        entry["off"] += off_delta
        entry["def"] += def_delta

    out = {}
    for tm, d in impacts.items():
        off = float(np.clip(d["off"], -0.40, 0.00))
        ddf = float(np.clip(d["def"], 0.00, 0.25))
        out[tm] = {"off_mult": 1.0 + off, "def_opp_scoring_mult": 1.0 + ddf}
    return out

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    date_col: Optional[str] = None
    for c in ("gameday", "game_date", "start_time"):
        if c in sched.columns:
            date_col = c; break

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
# Player Props helpers
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
    f"Defense source in use: **{DEF_SOURCE}** (EPA/play → factors)."
)

page = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()

    inj_file = st.file_uploader("Optional: Injury report CSV (any format)", type=["csv","txt","tsv"], key="inj_nfl")
    inj_map = parse_injury_upload(inj_file) if inj_file else {}

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
            mu_h, mu_a = nfl_matchup_mu(rates, home, away)

            # Injury impacts (offense and defense)
            home_code = _def_find_team(home) or norm_team_code(home)
            away_code = _def_find_team(away) or norm_team_code(away)

            if home_code in inj_map:
                mu_h *= inj_map[home_code]["off_mult"]
                mu_a *= inj_map[home_code]["def_opp_scoring_mult"]
            if away_code in inj_map:
                mu_a *= inj_map[away_code]["off_mult"]
                mu_h *= inj_map[away_code]["def_opp_scoring_mult"]

            pH, pA, mH, mA = _poisson_sim(mu_h, mu_a)
            st.markdown(
                f"**{home}** vs **{away}** — "
                f"Expected points: {mH:.1f}–{mA:.1f} · "
                f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
            )

            if inj_map:
                st.caption("Injuries applied.")
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

    c1, c2, c3 = st.columns(3)
    with c1:
        qb_up = st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2:
        rb_up = st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3:
        wr_up = st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

    def _read_any_table(up):
        if up is None: return pd.DataFrame()
        name = (up.name or "").lower()
        if name.endswith(".csv"):
            return pd.read_csv(up)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(up)
        return pd.read_csv(up)

    dfs = {}
    if qb_up: dfs["QB"] = _read_any_table(qb_up).copy()
    if rb_up: dfs["RB"] = _read_any_table(rb_up).copy()
    if wr_up: dfs["WR"] = _read_any_table(wr_up).copy()

    if not dfs:
        st.info("Upload at least one of QB/RB/WR CSVs to begin.")
        st.stop()

    # Optional: injury report to shift defense and/or your player's own team offense
    inj_file_props = st.file_uploader("Optional: Injury report CSV for this week", type=["csv","txt","tsv"], key="inj_props")
    inj_map_props = parse_injury_upload(inj_file_props) if inj_file_props else {}

    pos = st.selectbox("Market", ["QB","RB","WR"])
    df = dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded yet.")
        st.stop()

    name_col = _player_column_guess(df)
    yard_col = _yardage_column_guess(df, pos)

    # Player dropdown shows ONLY names
    players = df[name_col].astype(str).tolist()
    player = st.selectbox("Player", sorted(players))

    # Opponent defense code (aliases ok)
    opp_in = st.text_input("Opponent team code (e.g., DAL, PHI). Aliases like KAN/NOR/GNB/SFO are OK.", value="")
    opp = norm_team_code(opp_in)

    # (Optional) player's own team code for injury-offense penalty
    my_team_in = st.text_input("(Optional) Player's team code for injury adjustment", value="")
    my_team = norm_team_code(my_team_in)

    # Pull the selected player's mean from the CSV row
    row = df.loc[df[name_col].astype(str) == player].head(1)
    csv_mean = float(pd.to_numeric(row[yard_col], errors="coerce").fillna(0).mean()) if not row.empty else 0.0
    line = st.number_input("Yardage line", value=round(csv_mean or 0.0, 2), step=0.5)

    est_sd = _estimate_sd(csv_mean, pos)

    # Defense factor from embedded table + injuries on opponent defense
    def_factor = DEF_FACTOR_2025.get(opp, 1.00) if opp else 1.00
    inj_def_boost = inj_map_props.get(opp, {}).get("def_opp_scoring_mult", 1.0) if opp else 1.0
    def_factor *= float(inj_def_boost)

    # Own team offensive injury penalty (optional)
    own_off_mult = inj_map_props.get(my_team, {}).get("off_mult", 1.0) if my_team else 1.0

    adj_mean = csv_mean * def_factor * own_off_mult
    p_over, p_under = run_prop_sim(adj_mean, line, est_sd)

    st.success(
        f"**{player} — {('Passing' if pos=='QB' else 'Rush' if pos=='RB' else 'Receiving')} Yards**  \n"
        f"CSV mean: **{csv_mean:.1f}** · Defense factor ({opp or 'AVG'}): **×{def_factor:.3f}** · "
        f"Own-off mult ({my_team or '—'}): **×{own_off_mult:.3f}** → Adjusted mean: **{adj_mean:.1f}**  \n"
        f"Line: **{line:.1f}** → **P(over) = {100*p_over:.1f}%**, **P(under) = {100*p_under:.1f}%**"
    )

    with st.expander("Show player row used"):
        st.dataframe(row if not row.empty else df.head(5))

    with st.expander("Defense factors in use"):
        if DEF_EPA_USED:
            show = pd.DataFrame({
                "TEAM": list(DEF_EPA_USED.keys()),
                "EPA/play": list(DEF_EPA_USED.values()),
                "DEF_FACTOR": [build_def_factor_map(DEF_EPA_USED)[k] for k in DEF_EPA_USED.keys()],
            }).sort_values("DEF_FACTOR")
            st.dataframe(show.reset_index(drop=True))
        else:
            st.write("No defense table parsed; using fallback.")
