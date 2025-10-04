# app.py — NFL + MLB Predictor (2025, stats-only) + Player Props
# - NFL & MLB pages: Poisson from team scoring rates
# - Player Props: upload QB/RB/WR CSVs, choose player, input line
# - Defense EPA/play is EMBEDDED below (no upload needed)
# - Fix: full team names → standard codes so defense factor works

from __future__ import annotations
import io, math
from datetime import date
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ---- NFL pages ---------------------------------------------------------------
import nfl_data_py as nfl

# ---- MLB (BR team records) ---------------------------------------------------
from pybaseball import schedule_and_record

# ====================== EMBEDDED DEFENSE CSV (EPA/play) =======================
# Paste a new export over this block any time. Parser tolerates noisy rows.
DEFENSE_CSV_TEXT = """Team,EPA/PLAY
2025,-0.17,-38.71,0.4174,-0.37,0.06,685,0.6762,3,521,4,5.8,0.0813,0.065,0.0163
Minnesota Vikings
2025,-0.13,-33.41,0.379,-0.17,-0.05,984,0.5962,7,331,1,8.15,0.0409,0.0468,0.0526
Jacksonville Jaguars
2025,-0.11,-26.37,0.3796,-0.1,-0.12,853,0.5746,2,397,2,9.94,0.0974,0.0325,0.0065
Denver Broncos
2025,-0.11,-25.57,0.3667,-0.17,0.01,710,0.5938,3,445,3,7.69,0.0823,0.1076,0.019
Los Angeles Chargers
2025,-0.09,-21.51,0.3783,0,-0.22,894,0.6271,7,376,4,10.44,0.1,0.0571,0.0214
Detroit Lions
2025,-0.08,-20.49,0.4291,-0.11,-0.04,860,0.5693,5,504,3,8.43,0.0329,0.0658,0.0197
Philadelphia Eagles
2025,-0.08,-19.59,0.4149,-0.16,0.04,790,0.5714,3,409,4,8.3,0.0733,0.04,0.0133
Houston Texans
2025,-0.08,-18.57,0.4042,-0.12,0,851,0.664,5,394,2,8.12,0.0952,0.0544,0.0204
Los Angeles Rams
2025,-0.07,-18.65,0.4075,0,-0.19,910,0.6645,6,359,0,7.03,0.069,0.0575,0.0402
Seattle Seahawks
2025,-0.06,-14.94,0.4344,-0.09,-0.03,690,0.6829,5,462,2,7.02,0.0368,0.0588,0
San Francisco 49ers
2025,-0.06,-13.91,0.4202,-0.02,-0.11,832,0.6429,6,340,3,6.54,0.0645,0.1226,0.0065
Tampa Bay Buccaneers
2025,-0.05,-11.24,0.4279,-0.13,0.05,602,0.5769,5,436,2,10.08,0.0806,0.0806,0.0242
Atlanta Falcons
2025,-0.05,-10.73,0.379,0.06,-0.17,689,0.6442,8,281,2,7.95,0.0924,0.0336,0.0168
Cleveland Browns
2025,-0.05,-11.02,0.4638,-0.04,-0.05,946,0.6643,8,384,2,6.69,0.0633,0.0506,0.0253
Indianapolis Colts
2025,-0.02,-3.64,0.4487,-0.09,0.09,778,0.6694,4,508,4,8.27,0.0694,0.0903,0.0208
Kansas City Chiefs
2025,-0.01,-2.38,0.4559,0.06,-0.14,1068,0.6369,5,384,2,8.06,0.0444,0.0222,0.0111
Arizona Cardinals
2025,-0.01,-1.85,0.4315,0.14,-0.22,948,0.6565,5,411,4,7.98,0.0544,0.0544,0.0136
Las Vegas Raiders
2025,0,0.2,0.4094,0.03,-0.07,886,0.6815,6,310,3,6.87,0.0632,0.0345,0.0115
Green Bay Packers
2025,0,0.89,0.4912,0.01,0,886,0.7368,10,658,4,6.75,0.0407,0.0325,0.0569
Chicago Bears
2025,0.02,3.33,0.4208,-0.06,0.1,564,0.6214,6,657,5,6.87,0.0732,0.0894,0.0163
Buffalo Bills
2025,0.04,9.13,0.4133,0.03,0.05,802,0.6239,4,517,5,7.5,0.0155,0.0775,0.031
Carolina Panthers
2025,0.04,11.09,0.461,0.11,-0.05,1131,0.6957,7,488,4,7.6,0.087,0.0559,0.0311
Pittsburgh Steelers
2025,0.04,10.44,0.4183,0.18,-0.12,1062,0.6098,7,430,3,10.83,0.0714,0.05,0.0071
Washington Commanders
2025,0.05,12.43,0.4693,0.19,-0.15,1024,0.712,7,310,2,7.68,0.0725,0.0217,0.0217
New England Patriots
2025,0.07,18.22,0.4613,-0.01,0.19,1021,0.6375,5,612,6,7.88,0.0562,0.0449,0.0169
New York Giants
2025,0.07,17.94,0.4417,0.2,-0.06,884,0.7117,9,475,4,7.4,0.0853,0.0543,0.0078
New Orleans Saints
2025,0.1,27.12,0.4731,0.13,0.04,1089,0.6536,8,543,5,6.99,0.0366,0.0305,0.0305
Cincinnati Bengals
2025,0.11,25.77,0.3959,0.23,-0.03,834,0.6577,7,522,4,6.11,0.0476,0.0714,0
New York Jets
2025,0.12,30.47,0.4435,0.16,0.07,935,0.6984,6,566,7,6.82,0.0294,0.0441,0.0221
Tennessee Titans
2025,0.14,39.54,0.4685,0.14,0.12,1084,0.6667,9,565,7,8.04,0.0233,0.0523,0.0058
Baltimore Ravens
2025,0.25,65.26,0.4943,0.4,0.06,1237,0.7333,10,493,6,9.19,0.034,0.0476,0.0068
Dallas Cowboys
2025,0.25,59.66,0.5397,0.34,0.12,941,0.7757,7,632,5,6.15,0.0615,0.1154,0
Miami Dolphins
"""

# Fallback if parsing fails
DEF_EPA_2025_FALLBACK = {
    "MIN": -0.27,"JAX": -0.15,"GB": -0.13,"SF": -0.11,"ATL": -0.10,"IND": -0.08,
    "LAC": -0.08,"DEN": -0.08,"LAR": -0.07,"SEA": -0.07,"PHI": -0.06,"TB": -0.05,
    "CAR": -0.05,"ARI": -0.03,"CLE": -0.02,"WAS": -0.02,"HOU": 0.00,"KC": 0.01,
    "DET": 0.01,"LV": 0.03,"PIT": 0.05,"CIN": 0.05,"NO": 0.05,"BUF": 0.05,
    "CHI": 0.06,"NE": 0.09,"NYJ": 0.10,"TEN": 0.11,"BAL": 0.11,"NYG": 0.13,
    "DAL": 0.21,"MIA": 0.28,
}

# Code aliases
ALIAS_TO_STD = {
    "GNB":"GB","SFO":"SF","KAN":"KC","NWE":"NE","NOR":"NO","TAM":"TB",
    "LVR":"LV","SDG":"LAC","STL":"LAR","JAC":"JAX","WSH":"WAS","LA":"LAR","OAK":"LV"
}
def norm_team_code(code: str) -> str:
    c = (code or "").strip().upper()
    return ALIAS_TO_STD.get(c, c)

# FULL NAME → CODE (fix for ×1.000)
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

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6
EPS = 1e-9

# MLB map
MLB_TEAMS_2025: Dict[str,str] = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles","BOS":"Boston Red Sox",
    "CHC":"Chicago Cubs","CHW":"Chicago White Sox","CIN":"Cincinnati Reds","CLE":"Cleveland Guardians",
    "COL":"Colorado Rockies","DET":"Detroit Tigers","HOU":"Houston Astros","KCR":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins","MIL":"Milwaukee Brewers",
    "MIN":"Minnesota Twins","NYM":"New York Mets","NYY":"New York Yankees","OAK":"Oakland Athletics",
    "PHI":"Philadelphia Phillies","PIT":"Pittsburgh Pirates","SDP":"San Diego Padres","SEA":"Seattle Mariners",
    "SFG":"San Francisco Giants","STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays","TEX":"Texas Rangers",
    "TOR":"Toronto Blue Jays","WSN":"Washington Nationals",
}

# -------------------------- generic helpers -----------------------------------
def _poisson_sim(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any(): wins_home[ties] = 0.53
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

# ---------------------- Defense loading + factor map --------------------------
def build_def_factor_map(epa_map: Dict[str, float]) -> Dict[str, float]:
    """Convert EPA/play (lower = tougher) → multiplicative factor (~0.85..1.15)."""
    if not epa_map:
        return {}
    s = pd.Series(epa_map, dtype=float)
    mu, sd = float(s.mean()), float(s.std(ddof=0) or 1.0)
    z = (s - mu) / (sd if sd > 1e-9 else 1.0)
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k, v in factors.items()}

def _def_epa_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Get {STD_CODE: epa} from very noisy CSV. Accepts full team names or codes."""
    if df.empty:
        return {}
    cols_low = {str(c).strip().lower(): c for c in df.columns}
    team_candidates = ["team","team_code","def_team","abbr","tm","defense","opponent","opp","code","club"]
    epa_candidates  = ["epa/play","epa per play","epa_play","def_epa","def epa","epa"]

    team_col = next((cols_low.get(k) for k in team_candidates if k in cols_low), None)
    if team_col is None:  # fallback: first non-numeric column
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]): team_col = c; break

    epa_col = next((cols_low.get(k) for k in epa_candidates if k in cols_low), None)
    if epa_col is None:  # fallback: last numeric column
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        epa_col = nums[-1] if nums else None

    if team_col is None or epa_col is None:
        return {}

    out = {}
    for _, r in df[[team_col, epa_col]].dropna().iterrows():
        raw = str(r[team_col]).strip()
        full_upper = raw.upper()
        code = TEAM_NAME_TO_CODE.get(full_upper, norm_team_code(raw))
        try:
            out[code] = float(r[epa_col])
        except Exception:
            continue
    # Keep only real NFL codes
    out = {k: v for k, v in out.items() if k in TEAM_NAME_TO_CODE.values()}
    return out

def load_embedded_defense() -> Tuple[Dict[str,float], Dict[str,float], str]:
    """Parse DEFENSE_CSV_TEXT; if it fails, use fallback."""
    txt = (DEFENSE_CSV_TEXT or "").strip()
    if txt:
        try:
            df0 = pd.read_csv(io.StringIO(txt), sep=None, engine="python", header=0)
        except Exception:
            df0 = pd.read_csv(io.StringIO(txt), header=None)
        # handle “one-column CSV”
        if df0.shape[1] == 1:
            s = df0.iloc[:,0].astype(str)
            # heuristic: pair (metrics line → EPA) + (team name line)
            epa_vals, teams, buf = [], [], []
            for line in s:
                t = line.strip()
                if not t: continue
                buf.append(t)
                if len(buf) == 2:
                    nums = [w for w in buf[0].replace(",", " ").split()
                            if w.replace(".", "", 1).replace("-", "", 1).isdigit()]
                    if nums:
                        try:
                            epa = float(nums[1] if len(nums) > 1 else nums[0])
                            epa_vals.append(epa); teams.append(buf[1])
                        except Exception:
                            pass
                    buf = []
            df = pd.DataFrame({"team": teams, "epa_play": epa_vals})
        else:
            df = df0

        epa_map = _def_epa_from_df(df)
        if epa_map:
            return build_def_factor_map(epa_map), epa_map, "Embedded CSV (in script)"
    # fallback
    return build_def_factor_map(DEF_EPA_2025_FALLBACK), DEF_EPA_2025_FALLBACK, "Embedded fallback"

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])
    date_col: Optional[str] = None
    for c in ("gameday","game_date","start_time"):
        if c in sched.columns:
            date_col = c; break

    played = sched.dropna(subset=["home_score","away_score"])
    home = played.rename(columns={"home_team":"team","away_team":"opp","home_score":"pf","away_score":"pa"})[["team","opp","pf","pa"]]
    away = played.rename(columns={"away_team":"team","home_team":"opp","away_score":"pf","home_score":"pa"})[["team","opp","pf","pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 22.5
        teams32 = ["Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers","Chicago Bears",
                   "Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions","Green Bay Packers",
                   "Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs","Las Vegas Raiders",
                   "Los Angeles Chargers","Los Angeles Rams","Miami Dolphins","Minnesota Vikings","New England Patriots",
                   "New Orleans Saints","New York Giants","New York Jets","Philadelphia Eagles","Pittsburgh Steelers",
                   "San Francisco 49ers","Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders"]
        rates = pd.DataFrame({"team": teams32, "PF_pg": per, "PA_pg": per})
    else:
        team = long.groupby("team", as_index=False).agg(games=("pf","size"), PF=("pf","sum"), PA=("pa","sum"))
        rates = pd.DataFrame({"team": team["team"], "PF_pg": team["PF"]/team["games"], "PA_pg": team["PA"]/team["games"]})
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"]/4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink) * rates["PF_pg"] + shrink * prior
        rates["PA_pg"] = (1 - shrink) * rates["PA_pg"] + shrink * prior

    if {"home_team","away_team"}.issubset(sched.columns):
        filt = sched["home_score"].isna() & sched["away_score"].isna()
        upcoming = sched.loc[filt, ["home_team","away_team"]].copy()
        if date_col:
            dser = pd.to_datetime(sched.loc[filt, date_col], errors="coerce")
            today = pd.Timestamp(date.today())
            upcoming["date"] = dser.dt.date.astype(str)
            mask_future = (dser.notna()) & (dser.dt.date >= today.date())
            if mask_future.any():
                upcoming = upcoming.loc[mask_future].copy()
        else:
            upcoming["date"] = ""
        for c in ["home_team","away_team"]:
            upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+"," ", regex=True)
        upcoming = upcoming.reset_index(drop=True)
    else:
        upcoming = pd.DataFrame(columns=["home_team","away_team","date"])
    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = rates.loc[rates["team"].str.lower()==home.lower()]
    rA = rates.loc[rates["team"].str.lower()==away.lower()]
    if rH.empty or rA.empty: raise ValueError(f"Unknown team(s): {home}, {away}")
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
                sar["R"] = sar["R"].astype(float); sar["RA"] = sar["RA"].astype(float)
                g = int(len(sar)); RS_pg = float(sar["R"].sum()/g); RA_pg = float(sar["RA"].sum()/g)
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5})
    df = pd.DataFrame(rows).drop_duplicates("team").reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean()); league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9*df["RS_pg"] + 0.1*league_rs
        df["RA_pg"] = 0.9*df["RA_pg"] + 0.1*league_ra
    return df

# ==============================================================================
# Player Props — CSVs + embedded defense
# ==============================================================================
def _read_any_table(up) -> pd.DataFrame:
    if up is None: return pd.DataFrame()
    name = (up.name or "").lower()
    if name.endswith((".xlsx",".xls")): return pd.read_excel(up)
    try:
        df = pd.read_csv(up, sep=None, engine="python")
    except Exception:
        up.seek(0); df = pd.read_csv(up)
    if df.shape[1] == 1:
        s = df.iloc[:,0].astype(str)
        if s.str.contains(",").mean() > 0.5 or s.iloc[0].lower().startswith("rk,player"):
            text = "\n".join(s.tolist())
            try: df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            except Exception: pass
    return df

def _yardage_column_guess(df: pd.DataFrame, pos: str) -> str:
    prefer = ["Y/G","Yds/G","YDS/G","Yards/G","PY/G","RY/G","Rec Y/G","Yds","Yards","yds","yards"]
    low = [c.lower() for c in df.columns]
    for w in [p.lower() for p in prefer]:
        if w in low: return df.columns[low.index(w)]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return df.columns[-1]

def _player_column_guess(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in ("player","name","player name","player_name"):
            return c
    return df.columns[0]

def _player_name_series(df: pd.DataFrame) -> pd.Series:
    for c in df.columns:
        if str(c).strip().lower() in ("player","name","player name","player_name"):
            s = df[c].astype(str)
            return s[~s.str.lower().str.startswith("rk")].reset_index(drop=True)
    if df.shape[1] == 1:
        s = df.iloc[:,0].astype(str)
        if s.str.contains(",").mean() > 0.5:
            return s.apply(lambda x: x.split(",")[1].strip() if "," in x else x)
    return df.iloc[:,0].astype(str)

def _estimate_sd(mean_val: float, pos: str) -> float:
    mean_val = float(mean_val)
    if pos == "QB": return max(35.0, 0.60 * mean_val)
    if pos == "RB": return max(20.0, 0.75 * mean_val)
    return max(22.0, 0.85 * mean_val)

def run_prop_sim(mean_yards: float, line: float, sd: float) -> Tuple[float,float]:
    sd = max(5.0, float(sd))
    z = (line - mean_yards) / sd
    p_over = float(1.0 - 0.5*(1.0 + math.erf(z / math.sqrt(2))))
    return np.clip(p_over,0.0,1.0), np.clip(1.0 - p_over,0.0,1.0)

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Team pages use **2025 scoring rates** (NFL: PF/PA; MLB: RS/RA). "
    "Player Props: upload your QB/RB/WR CSVs, pick a player, set a line. "
    "Defense adjustment uses your **embedded EPA/play**."
)

page = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()

    if not upcoming.empty:
        labels = (upcoming["home_team"] + " vs " + upcoming["away_team"] +
                  ((" — " + upcoming["date"].astype(str)) if "date" in upcoming.columns else ""))
        sel = st.selectbox("Select upcoming game", labels.tolist())
        idx = labels[labels == sel].index[0]
        home = str(upcoming.loc[idx, "home_team"]); away = str(upcoming.loc[idx, "away_team"])
    else:
        st.info("No upcoming games list available — pick any two teams:")
        home = st.selectbox("Home team", rates["team"].tolist())
        away = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != home])

    try:
        mu_h, mu_a = nfl_matchup_mu(rates, home, away)
        pH, pA, mH, mA = _poisson_sim(mu_h, mu_a)
        st.markdown(f"**{home}** vs **{away}** — Expected points: {mH:.1f}–{mA:.1f} · "
                    f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**")
    except Exception as e:
        st.error(str(e))

# -------------------------- MLB page --------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 REG season (team scoring rates only)")
    rates = mlb_team_rates_2025()
    t1 = st.selectbox("Home team", rates["team"].tolist())
    t2 = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != t1])
    H = rates.loc[rates["team"] == t1].iloc[0]; A = rates.loc[rates["team"] == t2].iloc[0]
    mu_home = (H["RS_pg"] + A["RA_pg"]) / 2.0; mu_away = (A["RS_pg"] + H["RA_pg"]) / 2.0
    pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)
    st.markdown(f"**{t1}** vs **{t2}** — Expected runs: {mH:.1f}–{mA:.1f} · "
                f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**")

# -------------------------- Player Props page --------------------------
else:
    st.subheader("🎯 Player Props — drop in your CSVs")

    # Build defense map
    DEF_FACTOR_2025, DEF_EPA_USED, DEF_SOURCE = load_embedded_defense()
    st.caption(f"Defense source: **{DEF_SOURCE}** · teams parsed: **{len(DEF_EPA_USED)}**")

    c1, c2, c3 = st.columns(3)
    with c1: qb_up = st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2: rb_up = st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3: wr_up = st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

    dfs = {}
    if qb_up: dfs["QB"] = _read_any_table(qb_up).copy()
    if rb_up: dfs["RB"] = _read_any_table(rb_up).copy()
    if wr_up: dfs["WR"] = _read_any_table(wr_up).copy()

    if not dfs:
        st.info("Upload at least one of QB/RB/WR CSVs to begin."); st.stop()

    pos = st.selectbox("Market", ["QB","RB","WR"])
    df = dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded yet."); st.stop()

    name_col = _player_column_guess(df)
    yard_col = _yardage_column_guess(df, pos)
    names_series = _player_name_series(df)
    players = names_series.tolist()
    player = st.selectbox("Player", players)
    sel_idx = names_series.index[players.index(player)]
    row = df.iloc[[sel_idx]]

    opp_in = st.text_input("Opponent team code (e.g., DAL, PHI). Aliases like KAN/NOR/GNB/SFO are OK.", value="")
    opp = norm_team_code(opp_in.strip())
    def_factor = DEF_FACTOR_2025.get(opp, 1.00) if opp else 1.00

    csv_mean = float(pd.to_numeric(row[yard_col], errors="coerce").fillna(0).mean())
    line = st.number_input("Yardage line", value=round(csv_mean or 0.0, 2), step=0.5)
    est_sd = _estimate_sd(csv_mean, pos)

    adj_mean = csv_mean * def_factor
    p_over, p_under = run_prop_sim(adj_mean, line, est_sd)

    st.success(f"**{player} — {('Passing' if pos=='QB' else 'Rush' if pos=='RB' else 'Receiving')} Yards**  \n"
               f"CSV mean: **{csv_mean:.1f}** · Defense factor ({opp or 'AVG'}): **×{def_factor:.3f}** → "
               f"Adjusted mean: **{adj_mean:.1f}**  \n"
               f"Line: **{line:.1f}** → **P(over) = {100*p_over:.1f}%**, **P(under) = {100*p_under:.1f}%**")

    with st.expander("Show player row used"): st.dataframe(row)
    with st.expander("Defense factors in use"):
        if DEF_EPA_USED:
            df_show = pd.DataFrame({"TEAM": list(DEF_EPA_USED.keys()), "EPA/play": list(DEF_EPA_USED.values())})
            df_show["DEF_FACTOR"] = df_show["TEAM"].map(build_def_factor_map(DEF_EPA_USED))
            st.dataframe(df_show.sort_values("DEF_FACTOR").reset_index(drop=True))
        else:
            st.write("No defense table parsed; using fallback.")
