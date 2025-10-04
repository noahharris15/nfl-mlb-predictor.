# App.py — NFL + MLB + Player Props + College Football (stats-only)
import io
import math
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import requests
import streamlit as st
import pandas as pd
import requests

# Load secrets
CFBD_API_KEY = st.secrets["CFBD_API_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

# =========================
# Fetch College Football Stats
# =========================
@st.cache_data
def get_cfb_team_stats(year=2025):
    url = f"https://api.collegefootballdata.com/stats/season?year={year}"
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"CFBD API error: {response.status_code} {response.text}")
        return pd.DataFrame()

    data = response.json()

    # Convert into dataframe
    df = pd.DataFrame(data)

    # Ensure we have team, offense PPG, defense PPG
    # CFBD splits stats by category (Offense/Defense), so pivot it
    df = df.pivot_table(index="team", columns="statName", values="statValue", aggfunc="mean").reset_index()

    # Rename to match expected names
    if "Points Per Game" in df.columns:
        df = df.rename(columns={"Points Per Game": "off_ppg"})
    if "Opponent Points Per Game" in df.columns:
        df = df.rename(columns={"Opponent Points Per Game": "def_ppg"})

    # Fill missing values with 0
    df = df.fillna(0)
    return df[["team", "off_ppg", "def_ppg"]]

# =========================
# UI
# =========================
st.title("🏈 College Football — 2025 (auto from CFBD)")

df = get_cfb_team_stats(2025)

if df.empty:
    st.warning("No data pulled. Double-check your CFBD key in Streamlit Secrets.")
else:
    home_team = st.selectbox("Home team", df["team"].unique())
    away_team = st.selectbox("Away team", df["team"].unique())

    if st.button("Simulate Matchup"):
        home_stats = df[df["team"] == home_team].iloc[0]
        away_stats = df[df["team"] == away_team].iloc[0]

        # Simple PPG-based estimate
        home_exp = (home_stats["off_ppg"] + away_stats["def_ppg"]) / 2
        away_exp = (away_stats["off_ppg"] + home_stats["def_ppg"]) / 2

        total = home_exp + away_exp
        home_prob = home_exp / total if total > 0 else 0.5
        away_prob = away_exp / total if total > 0 else 0.5

        st.write(
            f"**{home_team} vs {away_team}** — Expected points: {home_exp:.1f}-{away_exp:.1f}. "
            f"P({home_team} win) = {home_prob:.1%}, P({away_team} win) = {away_prob:.1%}"
        )

    with st.expander("Show team table"):
        st.dataframe(df)
# ----------------------------- App config -------------------------------------
st.set_page_config(page_title="NFL + MLB + CFB Predictor — 2025 (stats only)",
                   layout="wide")

# ==============================================================================
# Embedded NFL DEFENSE (EPA/play)  — paste new CSV over DEFENSE_CSV_TEXT anytime
# ==============================================================================

DEFENSE_CSV_TEXT = """\
Team,EPA/PLAY
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

# Fallback (only used if the CSV above is emptied or cannot be parsed)
DEF_EPA_2025_FALLBACK = {
    "MIN": -0.27, "JAX": -0.15, "GB": -0.13, "SF": -0.11, "ATL": -0.10,
    "IND": -0.08, "LAC": -0.08, "DEN": -0.08, "LAR": -0.07, "SEA": -0.07,
    "PHI": -0.06, "TB": -0.05, "CAR": -0.05, "ARI": -0.03, "CLE": -0.02,
    "WAS": -0.02, "HOU":  0.00, "KC": 0.01, "DET": 0.01, "LV": 0.03,
    "PIT": 0.05, "CIN": 0.05, "NO": 0.05, "BUF": 0.05, "CHI": 0.06,
    "NE": 0.09, "NYJ": 0.10, "TEN": 0.11, "BAL": 0.11, "NYG": 0.13,
    "DAL": 0.21, "MIA": 0.28,
}

# Normalize team-code aliases
ALIAS_TO_STD = {
    "GNB": "GB", "SFO": "SF", "KAN": "KC", "NWE": "NE", "NOR": "NO", "TAM": "TB",
    "LVR": "LV", "SDG": "LAC", "STL": "LAR", "JAC": "JAX", "WSH": "WAS",
    "LA": "LAR", "OAK": "LV"
}

def _norm_team_code(code: str) -> str:
    c = (code or "").strip().upper()
    return ALIAS_TO_STD.get(c, c)

def _epa_csv_to_map(text: str) -> Dict[str, float]:
    try:
        df = pd.read_csv(io.StringIO(text))
        cols = {c.lower(): c for c in df.columns}
        # flexible name matching
        team_col = next((cols[k] for k in cols if k in ("team", "def_team", "name")), None)
        epa_col  = next((cols[k] for k in cols if k in ("epa/play", "epa per play", "epa", "def_epa", "epa_play")), None)
        if team_col is None or epa_col is None:
            # assume two columns: Team, value
            if len(df.columns) >= 2:
                team_col, epa_col = df.columns[0], df.columns[1]
        out = {}
        for _, r in df[[team_col, epa_col]].dropna().iterrows():
            k = _norm_team_code(str(r[team_col]))
            out[k] = float(r[epa_col])
        return out
    except Exception:
        return {}

def _build_def_factor_map(epa_map: Dict[str, float]) -> Dict[str, float]:
    """Lower EPA (better D) -> < 1.0 factor; clamp effect so it stays reasonable."""
    if not epa_map:
        return {}
    s = pd.Series(epa_map, dtype=float)
    mu, sd = float(s.mean()), float(s.std(ddof=0) or 1.0)
    z = (s - mu) / (sd if sd > 1e-9 else 1.0)
    # +/-2σ → 0.85..1.15 roughly
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k, v in factors.items()}

def _load_embedded_defense() -> tuple[Dict[str, float], Dict[str, float], str]:
    epa_map = _epa_csv_to_map(DEFENSE_CSV_TEXT.strip())
    if epa_map:
        return _build_def_factor_map(epa_map), epa_map, "Embedded CSV"
    return _build_def_factor_map(DEF_EPA_2025_FALLBACK), DEF_EPA_2025_FALLBACK, "Fallback"

# ----------------------------- Common sim utils -------------------------------
SIM_TRIALS = 10_000
EPS = 1e-9
HOME_EDGE_NFL = 0.6  # small bump to home mean

def _poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wh = (h > a).astype(float)
    ties = (h == a)
    if ties.any():
        wh[ties] = 0.53
    p_home = float(wh.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

# ==============================================================================
# NFL page (2025 schedule stats)
# ==============================================================================
import nfl_data_py as nfl

@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])
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

    # Upcoming list (robust date)
    date_col = next((c for c in ("gameday","game_date","start_time") if c in sched.columns), None)
    if {"home_team","away_team"}.issubset(sched.columns):
        filt = sched["home_score"].isna() & sched["away_score"].isna()
        upcoming = sched.loc[filt, ["home_team","away_team"]].copy()
        upcoming["date"] = sched.loc[filt, date_col].astype(str) if date_col else ""
    else:
        upcoming = pd.DataFrame(columns=["home_team","away_team","date"])
    return rates, upcoming

def page_nfl():
    st.subheader("🏈 NFL — 2025 regular season")
    rates, upcoming = nfl_team_rates_2025()

    if not upcoming.empty:
        labels = [f"{r['home_team']} vs {r['away_team']} — {r.get('date','')}" for _, r in upcoming.iterrows()]
        sel = st.selectbox("Select upcoming game", labels)
        try:
            teams_part = sel.split(" — ")[0]
            home, away = [t.strip() for t in teams_part.split(" vs ")]
        except Exception:
            home = away = None
    else:
        st.info("No upcoming games available; pick any two teams:")
        home = st.selectbox("Home team", rates["team"].tolist())
        away = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != home])

    if home and away:
        H = rates.loc[rates["team"] == home].iloc[0]
        A = rates.loc[rates["team"] == away].iloc[0]
        mu_home = (H["PF_pg"] + A["PA_pg"]) / 2.0 + HOME_EDGE_NFL
        mu_away = (A["PF_pg"] + H["PA_pg"]) / 2.0
        pH, pA, mH, mA = _poisson_game(mu_home, mu_away)
        st.markdown(
            f"**{home}** vs **{away}** — Expected points: {mH:.1f}–{mA:.1f} · "
            f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
        )

# ==============================================================================
# MLB page (2025 team RS/RA)
# ==============================================================================
from pybaseball import schedule_and_record

@st.cache_data(show_spinner=False)
def mlb_team_rates_2025() -> pd.DataFrame:
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
        df["RS_pg"] = 0.9 * df["RS_pg"] + 0.1 * league_rs
        df["RA_pg"] = 0.9 * df["RA_pg"] + 0.1 * league_ra
    return df

def page_mlb():
    st.subheader("⚾ MLB — 2025 regular season")
    rates = mlb_team_rates_2025()
    t1 = st.selectbox("Home team", rates["team"].tolist())
    t2 = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != t1])
    H = rates.loc[rates["team"] == t1].iloc[0]
    A = rates.loc[rates["team"] == t2].iloc[0]
    mu_home = (H["RS_pg"] + A["RA_pg"]) / 2.0
    mu_away = (A["RS_pg"] + H["RA_pg"]) / 2.0
    pH, pA, mH, mA = _poisson_game(mu_home, mu_away)
    st.markdown(
        f"**{t1}** vs **{t2}** — Expected runs: {mH:.1f}–{mA:.1f} · "
        f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**"
    )

# ==============================================================================
# Player Props (upload QB/RB/WR CSVs) + embedded defense
# ==============================================================================

def _player_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in ("player","name"):
            return c
    return df.columns[0]

def _yards_col(df: pd.DataFrame) -> str:
    preferred = ["y/g","yds/g","yards/g","py/g","ry/g","rec y/g","yds","yards"]
    low = [c.lower() for c in df.columns]
    for k in preferred:
        if k in low:
            return df.columns[low.index(k)]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return df.columns[-1]

def _est_sd(mean_val: float, pos: str) -> float:
    m = float(mean_val)
    if pos == "QB": return max(35.0, 0.60*m)
    if pos == "RB": return max(20.0, 0.75*m)
    return max(22.0, 0.85*m)

def _read_any(up):
    if up is None: return pd.DataFrame()
    name = (up.name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(up)
    return pd.read_csv(up)

def page_props():
    st.subheader("🎯 Player Props — upload your CSVs (QB / RB / WR)")

    DEF_FACTOR_2025, DEF_EPA_USED, SRC = _load_embedded_defense()
    st.caption(f"Defense source in use: **{SRC}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        up_qb = st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2:
        up_rb = st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3:
        up_wr = st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

    dfs = {}
    if up_qb is not None: dfs["QB"] = _read_any(up_qb)
    if up_rb is not None: dfs["RB"] = _read_any(up_rb)
    if up_wr is not None: dfs["WR"] = _read_any(up_wr)
    if not dfs:
        st.info("Upload at least one CSV to begin.")
        return

    pos = st.selectbox("Market", ["QB","RB","WR"])
    df = dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded.")
        return

    name_col = _player_col(df)
    yard_col = _yards_col(df)

    # present clean names only
    names = df[name_col].astype(str).tolist()
    player = st.selectbox("Player", names)

    # opponent code (aliases OK)
    opp_in = st.text_input("Opponent team code (e.g., DAL / PHI — aliases KAN/NOR/GNB/SFO OK):", value="")
    opp = _norm_team_code(opp_in)

    row = df.loc[df[name_col] == player].head(1)
    csv_mean = float(pd.to_numeric(row[yard_col], errors="coerce").fillna(0).mean()) if not row.empty else 0.0
    line = st.number_input("Yardage line", value=round(csv_mean or 0.0, 1), step=0.5)
    sd = _est_sd(csv_mean, pos)

    def_factor = DEF_FACTOR_2025.get(opp, 1.00) if opp else 1.00
    adj_mean = csv_mean * def_factor

    # normal approx
    z = (line - adj_mean) / max(5.0, sd)
    p_over = float(1.0 - 0.5*(1.0 + math.erf(z / math.sqrt(2))))
    p_under = 1.0 - p_over

    st.success(
        f"**{player} — {('Passing' if pos=='QB' else 'Rush' if pos=='RB' else 'Receiving')} Yards**\n\n"
        f"CSV mean: **{csv_mean:.1f}** · Defense factor ({opp or 'AVG'}): **×{def_factor:.3f}** "
        f"→ Adjusted mean: **{adj_mean:.1f}**\n"
        f"Line: **{line:.1f}** → **P(over) {100*p_over:.1f}%** · **P(under) {100*p_under:.1f}%**"
    )

    with st.expander("Defense table (in use)"):
        if DEF_EPA_USED:
            show = pd.DataFrame({
                "TEAM": list(DEF_EPA_USED.keys()),
                "EPA/play": list(DEF_EPA_USED.values()),
                "DEF_FACTOR": [_build_def_factor_map(DEF_EPA_USED)[k] for k in DEF_EPA_USED.keys()],
            }).sort_values("DEF_FACTOR")
            st.dataframe(show.reset_index(drop=True))

# ==============================================================================
# COLLEGE FOOTBALL (auto from CFBD)
# ==============================================================================

CFBD_BASE = "https://api.collegefootballdata.com"

def _cfbd_headers() -> Dict[str, str] | None:
    api_key = (st.secrets.get("CFBD_API_KEY") if "CFBD_API_KEY" in st.secrets else None)
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}

@st.cache_data(show_spinner=False)
def cfb_team_stats_2025() -> pd.DataFrame:
    headers = _cfbd_headers()
    if not headers:
        # no key: return empty; UI will instruct user
        return pd.DataFrame()

    # Season team stats (points per game & opponent points per game)
    url = f"{CFBD_BASE}/stats/season"
    params = {"year": 2025, "seasonType": "regular"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    # CFBD returns JSON list; guard for non-JSON (401/HTML/etc.)
    try:
        data = r.json()
    except Exception:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty or "team" not in df.columns or "statName" not in df.columns:
        return pd.DataFrame()

    off = df[df["statName"] == "pointsPerGame"][["team","statValue"]].rename(columns={"statValue":"off_ppg"})
    dff = df[df["statName"] == "opponentPointsPerGame"][["team","statValue"]].rename(columns={"statValue":"def_ppg"})
    out = pd.merge(off, dff, on="team", how="inner")
    # numeric & small smoothing in case of tiny samples
    out["off_ppg"] = pd.to_numeric(out["off_ppg"], errors="coerce")
    out["def_ppg"] = pd.to_numeric(out["def_ppg"], errors="coerce")
    out = out.dropna()
    if out.empty:
        return out
    league_off = float(out["off_ppg"].mean())
    league_def = float(out["def_ppg"].mean())
    out["off_ppg"] = 0.9*out["off_ppg"] + 0.1*league_off
    out["def_ppg"] = 0.9*out["def_ppg"] + 0.1*league_def
    return out.sort_values("team").reset_index(drop=True)

def page_cfb():
    st.subheader("🏈🎓 College Football — 2025 (auto from CFBD)")
    headers = _cfbd_headers()
    if not headers:
        st.info("Add your CFBD key in **Manage app → Secrets** as:\n\n`CFBD_API_KEY = \"your_key_here\"`")
        return

    df = cfb_team_stats_2025()
    if df.empty:
        st.error("CFBD request failed or returned no data. Double-check your API key and try again.")
        return

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home team", df["team"].tolist(), index=0)
    with col2:
        away = st.selectbox("Away team", [t for t in df["team"].tolist() if t != home], index=1)

    H = df.loc[df["team"] == home].iloc[0]
    A = df.loc[df["team"] == away].iloc[0]
    mu_home = (float(H["off_ppg"]) + float(A["def_ppg"])) / 2.0
    mu_away = (float(A["off_ppg"]) + float(H["def_ppg"])) / 2.0
    pH, pA, mH, mA = _poisson_game(mu_home, mu_away)

    st.markdown(
        f"**{home}** vs **{away}** — Expected points: {mH:.1f}–{mA:.1f} · "
        f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
    )

    with st.expander("Show team table"):
        st.dataframe(df, use_container_width=True)

# ==============================================================================
# MAIN NAV
# ==============================================================================

st.title("(stats only)")
st.caption(
    "NFL & MLB pages use team scoring rates. Player Props uses your CSV + embedded NFL defense. "
    "College Football auto-loads this season from CollegeFootballData when a key is set in secrets."
)

page = st.radio("Pick a page", ["NFL", "MLB", "College Football", "Player Props"], horizontal=True)

if page == "NFL":
    page_nfl()
elif page == "MLB":
    page_mlb()
elif page == "College Football":
    page_cfb()
else:
    page_props()
