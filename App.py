# app.py — NFL + MLB + Player Props + College Football (stats-only)
# - NFL/MLB simulate off team scoring rates
# - Player Props uses uploaded CSVs + embedded NFL defense factors
# - CFB auto-loads this season from CollegeFootballData (requires CFBD_API_KEY in secrets)

from __future__ import annotations
import io
import math
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------- Config -----------------------------------------
st.set_page_config(page_title="NFL + MLB + CFB + Props (stats only)", layout="wide")
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6
EPS = 1e-9

# =============================== NFL ==========================================
import nfl_data_py as nfl

# --- Embedded NFL Defense (edit/replace the CSV text below if you want) ------
DEFENSE_CSV_TEXT = """Team,EPA/PLAY
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

# Fallback if the CSV above is removed/unparsable
DEF_EPA_2025_FALLBACK = {
    "MIN": -0.27, "JAX": -0.15, "GB": -0.13, "SF": -0.11, "ATL": -0.10,
    "IND": -0.08, "LAC": -0.08, "DEN": -0.08, "LAR": -0.07, "SEA": -0.07,
    "PHI": -0.06, "TB": -0.05, "CAR": -0.05, "ARI": -0.03, "CLE": -0.02,
    "WAS": -0.02, "HOU": 0.00, "KC": 0.01, "DET": 0.01, "LV": 0.03,
    "PIT": 0.05, "CIN": 0.05, "NO": 0.05, "BUF": 0.05, "CHI": 0.06,
    "NE": 0.09, "NYJ": 0.10, "TEN": 0.11, "BAL": 0.11, "NYG": 0.13,
    "DAL": 0.21, "MIA": 0.28,
}

ALIAS_TO_STD = {
    "GNB": "GB", "SFO": "SF", "KAN": "KC", "NWE": "NE", "NOR": "NO", "TAM": "TB",
    "LVR": "LV", "SDG": "LAC", "STL": "LAR",
    "JAC": "JAX", "WSH": "WAS", "LA": "LAR", "OAK": "LV",
}
def norm_team_code(code: str) -> str:
    c = (code or "").strip().upper()
    return ALIAS_TO_STD.get(c, c)

def _poisson(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53
    return float(wins_home.mean()), float(1-wins_home.mean()), float(h.mean()), float(a.mean())

def build_def_factor_map(epa_map: Dict[str, float]) -> Dict[str, float]:
    if not epa_map:
        return {}
    s = pd.Series(epa_map, dtype=float)
    mu, sd = float(s.mean()), float(s.std(ddof=0) or 1.0)
    z = (s - mu) / (sd if sd > 1e-9 else 1.0)
    # lower EPA/better D -> lower factor; clamp
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k, v in factors.items()}

def _def_epa_from_any_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    # flexible header handling
    cols = {str(c).strip().lower(): c for c in df.columns}
    team_col = None
    for k in ["team","team_code","abbr","defense","opp","code"]:
        if k in cols: team_col = cols[k]; break
    if team_col is None:
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                team_col = c; break
    epa_col = None
    for k in ["epa/play","epa per play","epa_play","def_epa","epa"]:
        if k in cols: epa_col = cols[k]; break
    if epa_col is None:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        epa_col = nums[-1] if nums else None
    if team_col is None or epa_col is None:
        return {}
    out = {}
    for _, r in df[[team_col, epa_col]].dropna().iterrows():
        out[norm_team_code(str(r[team_col]))] = float(r[epa_col])
    return out

def load_embedded_defense():
    try:
        df = pd.read_csv(io.StringIO(DEFENSE_CSV_TEXT))
        m = _def_epa_from_any_df(df)
        if m:
            return build_def_factor_map(m), m, "Embedded CSV"
    except Exception:
        pass
    return build_def_factor_map(DEF_EPA_2025_FALLBACK), DEF_EPA_2025_FALLBACK, "Fallback"

@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])
    date_col: Optional[str] = None
    for c in ("gameday","game_date","start_time"):
        if c in sched.columns: date_col = c; break

    played = sched.dropna(subset=["home_score","away_score"])
    home = played.rename(columns={"home_team":"team","away_team":"opp","home_score":"pf","away_score":"pa"})[["team","opp","pf","pa"]]
    away = played.rename(columns={"away_team":"team","home_team":"opp","away_score":"pf","home_score":"pa"})[["team","opp","pf","pa"]]
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
        team = long.groupby("team", as_index=False).agg(games=("pf","size"), PF=("pf","sum"), PA=("pa","sum"))
        rates = pd.DataFrame({
            "team": team["team"],
            "PF_pg": team["PF"]/team["games"],
            "PA_pg": team["PA"]/team["games"]
        })
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total/2.0
        shrink = np.clip(1.0 - team["games"]/4.0, 0.0, 1.0)
        rates["PF_pg"] = (1-shrink)*rates["PF_pg"] + shrink*prior
        rates["PA_pg"] = (1-shrink)*rates["PA_pg"] + shrink*prior

    if {"home_team","away_team"}.issubset(sched.columns):
        filt = sched["home_score"].isna() & sched["away_score"].isna()
        upcoming = sched.loc[filt, ["home_team","away_team"]].copy()
        upcoming["date"] = sched.loc[filt, date_col].astype(str) if date_col else ""
    else:
        upcoming = pd.DataFrame(columns=["home_team","away_team","date"])

    for c in ["home_team","away_team"]:
        if c in upcoming.columns:
            upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+"," ",regex=True)

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

# =============================== MLB ==========================================
from pybaseball import schedule_and_record
MLB_TEAMS_2025: Dict[str, str] = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles",
    "BOS":"Boston Red Sox","CHC":"Chicago Cubs","CHW":"Chicago White Sox",
    "CIN":"Cincinnati Reds","CLE":"Cleveland Guardians","COL":"Colorado Rockies",
    "DET":"Detroit Tigers","HOU":"Houston Astros","KCR":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins",
    "MIL":"Milwaukee Brewers","MIN":"Minnesota Twins","NYM":"New York Mets",
    "NYY":"New York Yankees","OAK":"Oakland Athletics","PHI":"Philadelphia Phillies",
    "PIT":"Pittsburgh Pirates","SDP":"San Diego Padres","SEA":"Seattle Mariners",
    "SFG":"San Francisco Giants","STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays",
    "TEX":"Texas Rangers","TOR":"Toronto Blue Jays","WSN":"Washington Nationals",
}

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
                g = int(len(sar))
                RS_pg = float(sar["R"].sum() / g)
                RA_pg = float(sar["RA"].sum() / g)
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean())
        league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9*df["RS_pg"] + 0.1*league_rs
        df["RA_pg"] = 0.9*df["RA_pg"] + 0.1*league_ra
    return df

# ============================ Player Props ====================================
def _read_any_table(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(upload)
    return pd.read_csv(upload)

def _yardage_column_guess(df: pd.DataFrame) -> str:
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
    # show only player names (strip rank if present)
    for c in df.columns:
        if str(c).lower() in ("player","name"):
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

# ============================ College Football ================================
import requests

def _get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return default

CFBD_API_KEY = _get_secret("CFBD_API_KEY", "")

@st.cache_data(show_spinner=False)
def cfb_team_stats_2025() -> pd.DataFrame:
    """Pulls Points Per Game (offense/defense) from CollegeFootballData."""
    if not CFBD_API_KEY:
        return pd.DataFrame()
    url = "https://api.collegefootballdata.com/stats/season?year=2025"
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"CFBD request failed: {r.status_code} — {r.text[:200]}")
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError("CFBD request returned no data.")
    raw = pd.DataFrame(data)
    # We expect rows with statName like "Points Per Game" and "Opponent Points Per Game"
    pv = raw.pivot_table(index="team", columns="statName", values="statValue", aggfunc="mean").reset_index()
    # normalize column names
    rename_map = {}
    for col in list(pv.columns):
        lc = str(col).lower()
        if "points per game" == lc:
            rename_map[col] = "off_ppg"
        if "opponent points per game" == lc:
            rename_map[col] = "def_ppg"
    pv = pv.rename(columns=rename_map)
    # If the exact lower-casing didn't match, fall back with contains:
    if "off_ppg" not in pv.columns:
        for col in pv.columns:
            if isinstance(col,str) and "points per game" in col.lower() and "opponent" not in col.lower():
                pv = pv.rename(columns={col:"off_ppg"})
    if "def_ppg" not in pv.columns:
        for col in pv.columns:
            if isinstance(col,str) and "opponent points per game" in col.lower():
                pv = pv.rename(columns={col:"def_ppg"})
    keep = ["team"] + [c for c in ["off_ppg","def_ppg"] if c in pv.columns]
    pv = pv[keep].fillna(0.0)
    if "off_ppg" not in pv.columns: pv["off_ppg"] = 28.0
    if "def_ppg" not in pv.columns: pv["def_ppg"] = 28.0
    return pv.sort_values("team").reset_index(drop=True)

def cfb_matchup_mu(df: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = df.loc[df["team"] == home]
    rA = df.loc[df["team"] == away]
    if rH.empty or rA.empty:
        raise ValueError("Unknown CFB team(s)")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (float(H["off_ppg"]) + float(A["def_ppg"])) / 2.0)
    mu_away = max(EPS, (float(A["off_ppg"]) + float(H["def_ppg"])) / 2.0)
    return mu_home, mu_away

# ================================== UI ========================================
st.title("(stats only)")
st.caption(
    "NFL & MLB pages use team scoring rates. Player Props uses your CSV + embedded NFL defense. "
    "College Football auto-loads this season from CollegeFootballData (set CFBD_API_KEY in Secrets)."
)

page = st.radio("Pick a page", ["NFL", "MLB", "College Football", "Player Props"], horizontal=True)

# ---------------- NFL ----------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()
    if not upcoming.empty:
        labels = [f"{r['home_team']} vs {r['away_team']} — {r.get('date','')}" for _, r in upcoming.iterrows()]
        sel = st.selectbox("Select upcoming game", labels) if labels else None
        if sel:
            teams_part = sel.split(" — ")[0]
            home, away = [t.strip() for t in teams_part.split(" vs ")]
        else:
            home = away = None
    else:
        st.info("No upcoming games list available — pick any two teams:")
        home = st.selectbox("Home team", rates["team"].tolist())
        away = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != home])
    if home and away:
        mu_h, mu_a = nfl_matchup_mu(rates, home, away)
        pH, pA, mH, mA = _poisson(mu_h, mu_a)
        st.markdown(
            f"**{home}** vs **{away}** — Expected points: {mH:.1f}–{mA:.1f} · "
            f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
        )

# ---------------- MLB ----------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 REG season (team scoring rates only)")
    rates = mlb_team_rates_2025()
    t1 = st.selectbox("Home team", rates["team"].tolist())
    t2 = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != t1])
    H = rates.loc[rates["team"] == t1].iloc[0]
    A = rates.loc[rates["team"] == t2].iloc[0]
    mu_home = (H["RS_pg"] + A["RA_pg"]) / 2.0
    mu_away = (A["RS_pg"] + H["RA_pg"]) / 2.0
    pH, pA, mH, mA = _poisson(mu_home, mu_away)
    st.markdown(
        f"**{t1}** vs **{t2}** — Expected runs: {mH:.1f}–{mA:.1f} · "
        f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**"
    )

# -------------- College Football --------------
elif page == "College Football":
    st.subheader("🏈🎓 College Football — 2025 (auto from CFBD)")
    if not CFBD_API_KEY:
        st.info('Add your CFBD key in Streamlit "Secrets" as `CFBD_API_KEY = "..."`.')
        st.stop()
    try:
        df_cfb = cfb_team_stats_2025()
    except Exception as e:
        st.error(f"CFBD request failed or returned no data. Double-check your API key and try again.\n\n{e}")
        st.stop()
    if df_cfb.empty:
        st.warning("No CFB data available.")
        st.stop()
    home = st.selectbox("Home team", df_cfb["team"].tolist())
    away = st.selectbox("Away team", [t for t in df_cfb["team"].tolist() if t != home])
    mu_h, mu_a = cfb_matchup_mu(df_cfb, home, away)
    pH, pA, mH, mA = _poisson(mu_h, mu_a)
    st.markdown(
        f"**{home}** vs **{away}** — Expected points: {mH:.1f}–{mA:.1f} · "
        f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
    )
    with st.expander("Show team table"):
        st.dataframe(df_cfb)

# -------------- Player Props --------------
else:
    st.subheader("🎯 Player Props — drop in your CSVs (defense is embedded)")

    DEF_FACTOR_2025, DEF_EPA_USED, DEF_SRC = load_embedded_defense()
    st.caption(f"Defense source in use: **{DEF_SRC}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        qb_up = st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2:
        rb_up = st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3:
        wr_up = st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

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
    yard_col = _yardage_column_guess(df)

    # Clean names in dropdown (strip ranks, keep only the name)
    def _clean_name(x: str) -> str:
        s = str(x).strip()
        if "," in s:  # handles "1, Player" type rows
            parts = s.split(",")
            if len(parts) >= 2:
                s = parts[1].strip()
        return s

    display_names = df[name_col].astype(str).map(_clean_name).tolist()
    # map back: display -> original row selector
    idx_by_display = {display_names[i]: i for i in range(len(display_names))}
    player_display = st.selectbox("Player", display_names)
    row = df.iloc[[idx_by_display[player_display]]]

    opp_in = st.text_input("Opponent team code (e.g., DAL). Aliases like KAN/NOR/GNB/SFO are OK.", value="")
    opp = norm_team_code(opp_in)

    csv_mean = float(pd.to_numeric(row[yard_col], errors="coerce").fillna(0).mean())
    line = st.number_input("Yardage line", value=round(csv_mean or 0.0, 2), step=0.5)
    est_sd = _estimate_sd(csv_mean, pos)

    def_factor = DEF_FACTOR_2025.get(opp, 1.00) if opp else 1.00
    adj_mean = csv_mean * def_factor
    p_over, p_under = run_prop_sim(adj_mean, line, est_sd)

    st.success(
        f"**{player_display} — {('Passing' if pos=='QB' else 'Rush' if pos=='RB' else 'Receiving')} Yards**  \n"
        f"CSV mean: **{csv_mean:.1f}** · Defense factor ({opp or 'AVG'}): **×{def_factor:.3f}** → "
        f"Adjusted mean: **{adj_mean:.1f}**  \n"
        f"Line: **{line:.1f}** → **P(over) = {100*p_over:.1f}%**, **P(under) = {100*p_under:.1f}%**"
    )

    with st.expander("Show player row used"):
        st.dataframe(row)

    with st.expander("Defense factors in use"):
        show = pd.DataFrame({
            "TEAM": list(DEF_EPA_USED.keys()),
            "EPA/play": list(DEF_EPA_USED.values()),
            "DEF_FACTOR": [build_def_factor_map(DEF_EPA_USED)[k] for k in DEF_EPA_USED.keys()],
        }).sort_values("DEF_FACTOR")
        st.dataframe(show.reset_index(drop=True))
