# app.py — NFL + MLB + College Football (2025, stats-only) + Player Props
# - NFL/MLB team pages use 2025 scoring rates.
# - Player Props uses your uploaded QB/RB/WR CSVs and an embedded NFL defense table.
# - College Football pulls this season automatically from CollegeFootballData (CFBD)
#   using a Streamlit Secret: CFBD_API_KEY = "<your-key>"

from __future__ import annotations
import io
import math
from datetime import date
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import requests

# ---- NFL (team pages) ----
import nfl_data_py as nfl

# ---- MLB team records (BR) ----
from pybaseball import schedule_and_record

# ======================= (OPTIONAL) EMBED DEFENSE CSV TEXT =====================
# If you want to paste a raw CSV of defense EPA/play here, do it between the quotes.
# If this is empty or cannot be parsed, we fall back to the static map below.
DEFENSE_CSV_TEXT = """\
"""  # <-- leave empty to use fallback map below

# ------------------------- Fallback 2025 defense EPA/play ----------------------
DEF_EPA_2025_FALLBACK = {
    "MIN": -0.27, "JAX": -0.15, "GB": -0.13, "SF": -0.11, "ATL": -0.10,
    "IND": -0.08, "LAC": -0.08, "DEN": -0.08, "LAR": -0.07, "SEA": -0.07,
    "PHI": -0.06, "TB": -0.05, "CAR": -0.05, "ARI": -0.03, "CLE": -0.02,
    "WAS": -0.02, "HOU":  0.00, "KC": 0.01, "DET": 0.01, "LV": 0.03,
    "PIT": 0.05, "CIN": 0.05, "NO": 0.05, "BUF": 0.05, "CHI": 0.06,
    "NE": 0.09, "NYJ": 0.10, "TEN": 0.11, "BAL": 0.11, "NYG": 0.13,
    "DAL": 0.21, "MIA": 0.28,
}

# Accept alias codes and map to our keys above.
ALIAS_TO_STD = {
    "GNB": "GB", "SFO": "SF", "KAN": "KC", "NWE": "NE", "NOR": "NO", "TAM": "TB",
    "LVR": "LV", "SDG": "LAC", "STL": "LAR",
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
def build_def_factor_map(epa_map: Dict[str, float]) -> Dict[str, float]:
    """Convert EPA/play (lower = tougher) into a multiplicative factor (~0.85..1.15)."""
    if not epa_map:
        return {}
    series = pd.Series(epa_map, dtype=float)
    mu, sd = float(series.mean()), float(series.std(ddof=0) or 1.0)
    z = (series - mu) / (sd if sd > 1e-9 else 1.0)
    # Lower EPA (better D) => lower factor; clamp ±2σ → 0.85..1.15
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k, v in factors.items()}

def _def_epa_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Extract {TEAM_CODE: epa_per_play} from arbitrary CSV/XLSX-like table."""
    if df.empty:
        return {}
    cols_low = {str(c).strip().lower(): c for c in df.columns}

    team_candidates = ["team", "team_code", "def_team", "abbr", "tm", "defense", "opponent", "opp", "code"]
    epa_candidates  = ["epa/play", "epa per play", "epa_play", "def_epa", "def epa", "epa"]

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
        code = norm_team_code(str(r[team_col]))
        try:
            out[code] = float(r[epa_col])
        except Exception:
            continue
    return out

def load_embedded_defense() -> Tuple[Dict[str,float], Dict[str,float], str]:
    """Parse DEFENSE_CSV_TEXT if present; else use fallback.
    Returns: (def_factor_map, def_epa_map_used, source_label)
    """
    text = (DEFENSE_CSV_TEXT or "").strip()
    if text:
        try:
            df = pd.read_csv(io.StringIO(text))
            epa_map = _def_epa_from_df(df)
            if epa_map:
                return build_def_factor_map(epa_map), epa_map, "Embedded CSV (in script)"
        except Exception:
            pass
    return build_def_factor_map(DEF_EPA_2025_FALLBACK), DEF_EPA_2025_FALLBACK, "Embedded fallback"

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups (simple UI)
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    # a date-like column if present
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

    # upcoming games list (defensive guard for missing cols)
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
# College Football (CFBD) — auto from API using Secrets
# ==============================================================================
CFBD_BASE = "https://api.collegefootballdata.com"

def _cfbd_headers():
    key = st.secrets.get("CFBD_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}

@st.cache_data(show_spinner=False)
def cfbd_team_ppg_2025():
    """Return a DataFrame with team, off_ppg, def_ppg for the current season."""
    headers = _cfbd_headers()
    if not headers:
        raise RuntimeError("No CFBD_API_KEY in secrets.")

    # Teams
    r = requests.get(f"{CFBD_BASE}/teams/fbs?year=2025", headers=headers, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("CFBD 401: Unauthorized (check CFBD_API_KEY in Secrets).")
    r.raise_for_status()
    teams = pd.DataFrame(r.json())
    if teams.empty:
        raise RuntimeError("CFBD returned no teams.")

    # Stats
    roff = requests.get(f"{CFBD_BASE}/stats/season?year=2025&category=offense", headers=headers, timeout=30)
    roff.raise_for_status()
    off = pd.DataFrame(roff.json())

    rdef = requests.get(f"{CFBD_BASE}/stats/season?year=2025&category=defense", headers=headers, timeout=30)
    rdef.raise_for_status()
    deff = pd.DataFrame(rdef.json())

    def pick_ppg(df: pd.DataFrame, side: str):
        # Look for "points per game" variants
        m = df[df["statName"].str.lower().isin(["points per game", "pointspergame", "points_per_game"])]
        if m.empty:
            m = df[df["statName"].str.contains("points", case=False, na=False)]
        col = "off_ppg" if side == "offense" else "def_ppg"
        return m[["team", "statValue"]].rename(columns={"statValue": col})

    off_ppg = pick_ppg(off, "offense")
    def_ppg = pick_ppg(deff, "defense")

    merged = teams[["school"]].rename(columns={"school": "team"}) \
        .merge(off_ppg, on="team", how="left") \
        .merge(def_ppg, on="team", how="left")

    for c in ["off_ppg", "def_ppg"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged["off_ppg"] = merged["off_ppg"].fillna(28.0)
    merged["def_ppg"] = merged["def_ppg"].fillna(28.0)

    return merged.sort_values("team").reset_index(drop=True)

def cfb_matchup_mu(df: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = df.loc[df["team"].str.lower() == home.lower()]
    rA = df.loc[df["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["off_ppg"] + A["def_ppg"]) / 2.0)  # no extra home edge by default
    mu_away = max(EPS, (A["off_ppg"] + H["def_ppg"]) / 2.0)
    return mu_home, mu_away

# ==============================================================================
# Player Props — CSVs + embedded defense
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
    # Try common name columns; else first string column
    for c in df.columns:
        if str(c).strip().lower() in ("player","name","player name","full name"):
            return c
    for c in df.columns:
        if df[c].dtype == object:
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
st.set_page_config(page_title="NFL + MLB + CFB — 2025 (stats only)", layout="wide")
st.title("🏈⚾🏈‍🎓 NFL + MLB + College Football — 2025 (stats only)")
st.caption(
    "NFL & MLB pages use team scoring rates. Player Props uses your CSV + embedded NFL defense. "
    "College Football auto-loads this season from CollegeFootballData when a key is set in secrets."
)

page = st.radio("Pick a page", ["NFL", "MLB", "College Football", "Player Props"], horizontal=True)

# -------------------------- NFL page --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()

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
    pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)
    st.markdown(
        f"**{t1}** vs **{t2}** — "
        f"Expected runs: {mH:.1f}–{mA:.1f} · "
        f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**"
    )

# -------------------------- College Football page --------------------------
elif page == "College Football":
    st.subheader("🏈‍🎓 College Football — 2025 (auto from CFBD)")
    if not st.secrets.get("CFBD_API_KEY"):
        st.info('Add your CFBD key in **Settings → Secrets** as `CFBD_API_KEY = "..."`.')
        st.stop()

    try:
        cfb_rates = cfbd_team_ppg_2025()
    except Exception as e:
        st.error(f"CFBD error: {e}")
        st.stop()

    home = st.selectbox("Home team", cfb_rates["team"].tolist())
    away = st.selectbox("Away team", [t for t in cfb_rates["team"].tolist() if t != home])

    mu_home, mu_away = cfb_matchup_mu(cfb_rates, home, away)
    pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)
    st.markdown(
        f"**{home}** vs **{away}** — "
        f"Expected points: {mH:.1f}–{mA:.1f} · "
        f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
    )

    with st.expander("Show team table"):
        st.dataframe(cfb_rates)

# -------------------------- Player Props page --------------------------
else:
    st.subheader("🎯 Player Props — drop in your CSVs")

    # Build defense factors from embedded CSV (or fallback)
    DEF_FACTOR_2025, DEF_EPA_USED, DEF_SOURCE = load_embedded_defense()
    st.caption(f"Defense source in use: **{DEF_SOURCE}**")

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

    pos = st.selectbox("Market", ["QB","RB","WR"])
    df = dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded yet.")
        st.stop()

    name_col = _player_column_guess(df)
    yard_col = _yardage_column_guess(df, pos)

    # Only names in dropdown (no extra CSV fields)
    players = df[name_col].astype(str).tolist()
    player = st.selectbox("Player", players)

    opp_in = st.text_input("Opponent team code (e.g., DAL, PHI). Aliases like KAN/NOR/GNB/SFO are OK.", value="")
    opp = norm_team_code(opp_in)

    row = df.loc[df[name_col]==player].head(1)
    csv_mean = float(pd.to_numeric(row[yard_col], errors="coerce").fillna(0).mean()) if not row.empty else 0.0
    line = st.number_input("Yardage line", value=round(csv_mean or 0.0, 2), step=0.5)

    est_sd = _estimate_sd(csv_mean, pos)
    def_factor = DEF_FACTOR_2025.get(opp, 1.00) if opp else 1.00
    adj_mean = csv_mean * def_factor
    p_over, p_under = run_prop_sim(adj_mean, line, est_sd)

    st.success(
        f"**{player} — {('Passing' if pos=='QB' else 'Rush' if pos=='RB' else 'Receiving')} Yards**  \n"
        f"CSV mean: **{csv_mean:.1f}** · Defense factor ({opp or 'AVG'}): **×{def_factor:.3f}** → "
        f"Adjusted mean: **{adj_mean:.1f}**  \n"
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
