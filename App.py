# app.py — NFL + MLB Predictor (2025, stats-only) + Player Props
# - Embedded Defense CSV (robust parser, works with alternating "2025 line / Team name" format)
# - Injury Report upload wired into BOTH NFL model page and Player Props page
# - Player drop-down shows clean names only

from __future__ import annotations
import io
import math
import re
from typing import Optional, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
import streamlit as st

# ---- NFL team schedules/results ----
import nfl_data_py as nfl

# ---- MLB team records (BR) ----
from pybaseball import schedule_and_record


# ============================================================================
# 1) EMBED YOUR DEFENSE CSV HERE  (leave header; paste full CSV rows below)
#    It can be tidy "Team,EPA/PLAY" OR your alternating "2025 row / Team row".
#    The parser below handles both.
# ============================================================================
DEFENSE_CSV_TEXT = """\
Team,EPA/PLAY
# Example: works with alternating "season-row, then team-row"
2025,-0.17,,,,,,,,,,,,,
Minnesota Vikings
2025,-0.13,,,,,,,,,,,,,
Jacksonville Jaguars
2025,-0.11,,,,,,,,,,,,,
Denver Broncos
2025,-0.11,,,,,,,,,,,,,
Los Angeles Chargers
2025,-0.09,,,,,,,,,,,,,
Detroit Lions
2025,-0.08,,,,,,,,,,,,,
Philadelphia Eagles
2025,-0.08,,,,,,,,,,,,,
Houston Texans
2025,-0.08,,,,,,,,,,,,,
Los Angeles Rams
2025,-0.07,,,,,,,,,,,,,
Seattle Seahawks
2025,-0.06,,,,,,,,,,,,,
San Francisco 49ers
2025,-0.06,,,,,,,,,,,,,
Tampa Bay Buccaneers
2025,-0.05,,,,,,,,,,,,,
Atlanta Falcons
2025,-0.05,,,,,,,,,,,,,
Cleveland Browns
2025,-0.05,,,,,,,,,,,,,
Indianapolis Colts
2025,-0.02,,,,,,,,,,,,,
Kansas City Chiefs
2025,-0.01,,,,,,,,,,,,,
Arizona Cardinals
2025,-0.01,,,,,,,,,,,,,
Las Vegas Raiders
2025,0.00,,,,,,,,,,,,,
Green Bay Packers
2025,0.00,,,,,,,,,,,,,
Chicago Bears
2025,0.02,,,,,,,,,,,,,
Buffalo Bills
2025,0.04,,,,,,,,,,,,,
Carolina Panthers
2025,0.04,,,,,,,,,,,,,
Pittsburgh Steelers
2025,0.04,,,,,,,,,,,,,
Washington Commanders
2025,0.05,,,,,,,,,,,,,
New England Patriots
2025,0.07,,,,,,,,,,,,,
New York Giants
2025,0.07,,,,,,,,,,,,,
New Orleans Saints
2025,0.10,,,,,,,,,,,,,
Cincinnati Bengals
2025,0.11,,,,,,,,,,,,,
New York Jets
2025,0.12,,,,,,,,,,,,,
Tennessee Titans
2025,0.14,,,,,,,,,,,,,
Baltimore Ravens
2025,0.25,,,,,,,,,,,,,
Dallas Cowboys
2025,0.25,,,,,,,,,,,,,
Miami Dolphins
"""


# ============================================================================
# 2) CONSTANTS / MAPS
# ============================================================================
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6   # small NFL home bump
EPS = 1e-9

# Common alias normalization (codes typed on props page)
ALIAS_TO_STD = {
    # PFR-ish
    "GNB": "GB", "SFO": "SF", "KAN": "KC", "NWE": "NE", "NOR": "NO", "TAM": "TB",
    "LVR": "LV", "SDG": "LAC", "STL": "LAR",
    # common alternates
    "JAC": "JAX", "WSH": "WAS", "LA": "LAR", "OAK": "LV",
}
def norm_code(code: str) -> str:
    c = (code or "").strip().upper()
    return ALIAS_TO_STD.get(c, c)

# Full-name → code (used for defense+injury lookups)
TEAM_NAME_TO_CODE = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

# MLB (BR codes to names)
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

# Injury → multiplier
INJURY_MULTIPLIERS = {
    "Out": 0.0,
    "IR": 0.0,
    "Doubtful": 0.25,
    "DNP": 0.35,  # practice DNP
    "Questionable": 0.60,
    "Limited": 0.75,
    "Full": 1.0,
    "-": 1.0,
    "": 1.0,
    None: 1.0,
}

# Positional “impact” weights (team offense scaling)
POS_WEIGHTS = {
    "QB": 0.35,
    "WR": 0.12,
    "RB": 0.12,
    "TE": 0.08,
    # lump OL positions
    "T": 0.10, "G": 0.10, "C": 0.10, "OL": 0.10,
    # default small
    "_DEFAULT": 0.05,
}


# ============================================================================
# 3) DEFENSE CSV PARSING → EPA map → multiplicative factor (~0.85..1.15)
# ============================================================================
def _name_to_code(s: str) -> Optional[str]:
    s = str(s).strip()
    if not s: return None
    if s in TEAM_NAME_TO_CODE: return TEAM_NAME_TO_CODE[s]
    if s.upper() in TEAM_NAME_TO_CODE.values(): return s.upper()
    # loose match
    for name, code in TEAM_NAME_TO_CODE.items():
        if s.lower() == name.lower(): return code
    return None

def _robust_read_csv(text: str) -> pd.DataFrame:
    # tolerate odd commas/widths; treat '#' as comments
    return pd.read_csv(io.StringIO(text), sep=",", engine="python", comment="#", header=0, dtype=str)

def defense_text_to_epa_map(text: str) -> Dict[str, float]:
    """
    Handles tidy 'Team,EPA/PLAY' or alternating '2025,epa,...' then 'Team Name' lines.
    Returns {TEAM_CODE: epa_per_play}.
    """
    text = (text or "").strip()
    if not text:
        return {}

    df = _robust_read_csv(text)
    if df.empty:
        return {}

    # Case A: tidy columns present
    cols = {c.lower(): c for c in df.columns}
    team_col = None
    epa_col = None
    for k in ["team", "team_code", "def_team", "abbr"]:
        if k in cols: team_col = cols[k]; break
    for k in ["epa/play", "epa per play", "epa", "epa_play"]:
        if k in cols: epa_col = cols[k]; break
    epa_map: Dict[str, float] = {}

    if team_col and epa_col:
        tmp = df[[team_col, epa_col]].copy()
        for _, r in tmp.dropna().iterrows():
            code = _name_to_code(r[team_col]) or norm_code(str(r[team_col]))
            try:
                val = float(str(r[epa_col]).strip())
            except Exception:
                continue
            if code:
                epa_map[code] = val
        if epa_map:
            return epa_map

    # Case B: alternating season/stat rows followed by a single-field team name
    # Expect first numeric token line is EPA, then the next single string line is team.
    epa_last: Optional[float] = None
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"): 
            continue
        parts = [p.strip() for p in raw.split(",")]
        # line with numeric in second field? ex: "2025,-0.17,...."
        if len(parts) >= 2 and re.fullmatch(r"-?\d+(\.\d+)?", parts[1] or ""):
            try:
                epa_last = float(parts[1])
            except Exception:
                epa_last = None
            continue
        # otherwise, treat as team name line
        code = _name_to_code(parts[0])
        if code and epa_last is not None:
            epa_map[code] = epa_last
            epa_last = None

    return epa_map

def build_def_factor_map(epa_map: Dict[str, float]) -> Dict[str, float]:
    """Lower EPA (better D) → lower factor. Clamp ±2σ to ~0.85..1.15."""
    if not epa_map:
        return {}
    s = pd.Series(epa_map, dtype=float)
    mu, sd = float(s.mean()), float(s.std(ddof=0) or 1.0)
    z = (s - mu) / (sd if sd > 1e-9 else 1.0)
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k, v in factors.items()}


# ============================================================================
# 4) INJURY REPORT HANDLING
# ============================================================================
def _read_any_table(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    name = (upload.name or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(upload, sep=None, engine="python")
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(upload)
        return pd.read_csv(upload, sep=None, engine="python")
    except Exception:
        return pd.DataFrame()

def _pos_bucket(pos: str) -> str:
    p = (pos or "").upper().strip()
    if p in POS_WEIGHTS: return p
    if p in {"LT","RT","LG","RG","C","OL"}: return "OL"
    if p.startswith("WR"): return "WR"
    if p.startswith("RB"): return "RB"
    if p.startswith("TE"): return "TE"
    if p.startswith("QB"): return "QB"
    return "_DEFAULT"

def parse_injury_file(upload) -> pd.DataFrame:
    """
    Expect columns like:
      Team, Player, Pos, Game Status
    but will also infer from 'Participation *' columns when 'Game Status' absent.
    Returns a tidy df with columns: team_code, player, pos_bucket, status, mult, team_name
    """
    df = _read_any_table(upload)
    if df.empty:
        return df

    cols = {c.lower(): c for c in df.columns}

    def _col(*names: Iterable[str]) -> Optional[str]:
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        return None

    c_team = _col("Team","team")
    c_player = _col("Player","player","name")
    c_pos = _col("Pos","Position","pos")
    c_status = _col("Game Status","Status","game status")

    # fallback: derive status from participation columns
    if c_status is None:
        for day in ["Wednesday","Thursday","Friday","Tuesday","Monday"]:
            c = _col(f"Participation {day}", f"Participation_{day}")
            if c and c in df.columns:
                df["__status_tmp__"] = df[c]
                c_status = "__status_tmp__"
                break

    out_rows = []
    for _, r in df.iterrows():
        team_name = str(r.get(c_team, "")).strip() if c_team else ""
        team_code = _name_to_code(team_name) or norm_code(team_name)
        player = str(r.get(c_player, "")).strip() if c_player else ""
        pos_bucket = _pos_bucket(str(r.get(c_pos, "")))
        raw_status = str(r.get(c_status, "")).strip() if c_status else ""

        # normalize status tokens
        tok = raw_status.title()
        if tok.upper() in {"DNP","DID NOT PRACTICE"}: tok = "DNP"
        mult = INJURY_MULTIPLIERS.get(tok, 1.0)

        out_rows.append({
            "team_name": team_name,
            "team_code": team_code,
            "player": player,
            "pos_bucket": pos_bucket,
            "status": tok,
            "mult": mult,
        })

    tidy = pd.DataFrame(out_rows)
    return tidy


def team_offense_multiplier(inj_df: pd.DataFrame, team_name_or_code: str) -> float:
    """Combine player-level multipliers into a single team offense factor."""
    if inj_df is None or inj_df.empty:
        return 1.0
    # match by either full name or code
    code = _name_to_code(team_name_or_code) or norm_code(team_name_or_code)
    name = None
    for n, c in TEAM_NAME_TO_CODE.items():
        if c == code: name = n; break

    sub = inj_df[(inj_df["team_code"] == code) | (inj_df["team_name"] == name)]
    if sub.empty:
        return 1.0

    factor = 1.0
    for _, r in sub.iterrows():
        w = POS_WEIGHTS.get(r["pos_bucket"], POS_WEIGHTS["_DEFAULT"])
        m = float(r["mult"])
        # convex blend: (1-w) + w*m
        factor *= (1.0 - w) + w * m
    return float(np.clip(factor, 0.60, 1.05))  # clamp a bit


# ============================================================================
# 5) GENERIC HELPERS
# ============================================================================
def _poisson_sim(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # small OT tilt
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

def _yardage_col_guess(df: pd.DataFrame, pos: str) -> str:
    prefer = ["Y/G","Yds/G","YDS/G","Yards/G","PY/G","RY/G","Rec Y/G",
              "Yds","Yards","yds","yards"]
    low = [c.lower() for c in df.columns]
    for wanted in [p.lower() for p in prefer]:
        if wanted in low: return df.columns[low.index(wanted)]
    # fallback: first numeric
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return df.columns[-1]

def _player_col_guess(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower() in ("player","name"): return c
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


# ============================================================================
# 6) NFL TEAM RATES (2025)
# ============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    # date-ish column
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
        per = 45.0/2.0
        teams32 = list(TEAM_NAME_TO_CODE.keys())
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


# ============================================================================
# 7) MLB TEAM RATES (2025)
# ============================================================================
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


# ============================================================================
# 8) STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "• NFL & MLB pages: win probs from 2025 scoring rates only (no betting data).  \n"
    "• Player Props: upload your QB/RB/WR CSVs, pick a player, set a line.  \n"
    "• Defense effects come from the embedded EPA/Play CSV.  \n"
    "• Optional: upload an **Injury Report** to adjust team offense and player props."
)

page = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)


# -------------------------- NFL page --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()

    # Defense factors (from embedded CSV)
    EPA_MAP = defense_text_to_epa_map(DEFENSE_CSV_TEXT)
    DEF_FACTORS = build_def_factor_map(EPA_MAP)
    with st.expander("Defense table in use", expanded=False):
        st.dataframe(pd.DataFrame({
            "TEAM": list(EPA_MAP.keys()),
            "EPA/play": list(EPA_MAP.values()),
            "DEF_FACTOR": [DEF_FACTORS[k] for k in EPA_MAP.keys()]
        }).sort_values("DEF_FACTOR").reset_index(drop=True))

    # Optional injury file (team-level impact)
    inj_up = st.file_uploader("Injury Report (optional) — CSV/XLSX", type=["csv","xlsx"], key="inj_nfl")
    inj_df = parse_injury_file(inj_up) if inj_up else pd.DataFrame()

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

            # Apply OFFENSIVE injury multipliers (reduce PF side), then DEFENSE factor:
            # For simplicity, map defense to opponent factor acting on offense mean.
            home_code = TEAM_NAME_TO_CODE.get(home, "")
            away_code = TEAM_NAME_TO_CODE.get(away, "")

            # injury
            inj_home = team_offense_multiplier(inj_df, home) if not inj_df.empty else 1.0
            inj_away = team_offense_multiplier(inj_df, away) if not inj_df.empty else 1.0

            # defense factors (opponent defense affects this side's mean)
            def_vs_home = DEF_FACTORS.get(away_code, 1.0)
            def_vs_away = DEF_FACTORS.get(home_code, 1.0)

            mu_h_adj = max(EPS, mu_h * inj_home * def_vs_home)
            mu_a_adj = max(EPS, mu_a * inj_away * def_vs_away)

            pH, pA, mH, mA = _poisson_sim(mu_h_adj, mu_a_adj)

            st.markdown(
                f"**{home}** vs **{away}** — "
                f"Expected points: {mH:.1f}–{mA:.1f} · "
                f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
            )
            with st.expander("Adjustments used"):
                st.write({
                    "home_off_injury_mult": round(inj_home, 3),
                    "away_off_injury_mult": round(inj_away, 3),
                    "def_factor_vs_home_off": round(def_vs_home, 3),
                    "def_factor_vs_away_off": round(def_vs_away, 3),
                })
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
    st.subheader("🎯 Player Props — upload your CSVs")

    # Player CSVs
    c1, c2, c3 = st.columns(3)
    with c1:
        qb_up = st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2:
        rb_up = st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3:
        wr_up = st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

    dfs: Dict[str, pd.DataFrame] = {}
    if qb_up is not None: dfs["QB"] = _read_any_table(qb_up).copy()
    if rb_up is not None: dfs["RB"] = _read_any_table(rb_up).copy()
    if wr_up is not None: dfs["WR"] = _read_any_table(wr_up).copy()

    if not dfs:
        st.info("Upload at least one of QB/RB/WR CSVs to begin.")
        st.stop()

    # Optional injury file (player-level impact)
    inj_up_props = st.file_uploader("Injury Report (optional) — CSV/XLSX", type=["csv","xlsx"], key="inj_props")
    inj_df_props = parse_injury_file(inj_up_props) if inj_up_props else pd.DataFrame()

    # Defense factors for props (opponent code input)
    EPA_MAP = defense_text_to_epa_map(DEFENSE_CSV_TEXT)
    DEF_FACTORS = build_def_factor_map(EPA_MAP)

    pos = st.selectbox("Market", ["QB","RB","WR"])
    df = dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded yet.")
        st.stop()

    # Find name + yardage columns
    name_col = _player_col_guess(df)
    yard_col = _yardage_col_guess(df, pos)

    # Players list = clean names only
    players = df[name_col].astype(str).str.strip().tolist()
    player = st.selectbox("Player", players)

    # Opponent defense (code)
    opp_code_in = st.text_input("Opponent team code (e.g., DAL, PHI). Aliases like KAN/NOR/GNB/SFO are OK.", value="")
    opp_code = norm_code(opp_code_in)

    # CSV mean from that row
    row = df.loc[df[name_col].astype(str).str.strip() == player].head(1)
    csv_mean = float(pd.to_numeric(row[yard_col], errors="coerce").fillna(0).mean()) if not row.empty else 0.0

    # Default line = CSV mean
    line = st.number_input("Yardage line", value=round(csv_mean, 1), step=0.5)

    # SD by position
    est_sd = _estimate_sd(csv_mean, pos)

    # Defense factor
    def_factor = DEF_FACTORS.get(opp_code, 1.00) if opp_code else 1.00

    # Player-level injury multiplier (match by name, any team)
    inj_mult = 1.0
    if not inj_df_props.empty:
        match = inj_df_props.loc[inj_df_props["player"].str.lower() == player.lower()]
        if not match.empty:
            inj_mult = float(match.iloc[0]["mult"])

    adj_mean = csv_mean * def_factor * inj_mult
    p_over, p_under = run_prop_sim(adj_mean, line, est_sd)

    st.success(
        f"**{player} — {('Passing' if pos=='QB' else 'Rushing' if pos=='RB' else 'Receiving')} Yards**\n"
        f"CSV mean: **{csv_mean:.1f}** · "
        f"Defense factor ({opp_code or 'AVG'}): **×{def_factor:.3f}** · "
        f"Injury: **×{inj_mult:.3f}** → Adjusted mean: **{adj_mean:.1f}**\n"
        f"Line: **{line:.1f}** → **P(over) = {100*p_over:.1f}%**, **P(under) = {100*p_under:.1f}%**"
    )

    with st.expander("Show player row used"):
        st.dataframe(row if not row.empty else df.head(5))

    with st.expander("Defense factors in use"):
        st.dataframe(pd.DataFrame({
            "TEAM": list(EPA_MAP.keys()),
            "EPA/play": list(EPA_MAP.values()),
            "DEF_FACTOR": [DEF_FACTORS[k] for k in EPA_MAP.keys()],
        }).sort_values("DEF_FACTOR").reset_index(drop=True))

    if not inj_df_props.empty:
        with st.expander("Parsed injury report (props)"):
            st.dataframe(inj_df_props)
