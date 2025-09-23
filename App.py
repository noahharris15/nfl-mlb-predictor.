# app.py — NFL + MLB predictor (2025 only, stats-only) + Player Props with weekly Defense CSV
# - NFL page: team PF/PA Poisson (unchanged behavior)
# - MLB page: team RS/RA Poisson + probables ERA adjustment (unchanged behavior)
# - Player Props: upload QB/RB/WR CSVs + upload weekly Defense CSV (or file on disk),
#                 choose player + yardage line + type opponent defense, sim win prob.

import math
import time
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO
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
SIM_TRIALS = 20000
HOME_EDGE_NFL = 0.6  # ~0.6 pts to the home offense mean (small HFA)
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
def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

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

# normalize NFL team text to a canonical key (robust to abbreviations/spacing)
_TEAM_ALIASES = {
    # NFC
    "ari": "arizona cardinals", "cardinals": "arizona cardinals", "arizona": "arizona cardinals",
    "atl": "atlanta falcons", "falcons": "atlanta falcons",
    "car": "carolina panthers", "panthers": "carolina panthers",
    "chi": "chicago bears", "bears": "chicago bears",
    "dal": "dallas cowboys", "cowboys": "dallas cowboys",
    "det": "detroit lions", "lions": "detroit lions",
    "gb": "green bay packers", "gnb": "green bay packers", "packers": "green bay packers",
    "lar": "los angeles rams", "la rams": "los angeles rams", "rams": "los angeles rams",
    "min": "minnesota vikings", "vikings": "minnesota vikings",
    "no": "new orleans saints", "nor": "new orleans saints", "saints": "new orleans saints",
    "nyg": "new york giants", "giants": "new york giants",
    "phi": "philadelphia eagles", "eagles": "philadelphia eagles",
    "sf": "san francisco 49ers", "sfo": "san francisco 49ers", "49ers": "san francisco 49ers",
    "sea": "seattle seahawks", "seahawks": "seattle seahawks",
    "tb": "tampa bay buccaneers", "tam": "tampa bay buccaneers", "buccaneers": "tampa bay buccaneers",
    "wsh": "washington commanders", "was": "washington commanders", "commanders": "washington commanders",
    # AFC
    "bal": "baltimore ravens", "ravens": "baltimore ravens",
    "buf": "buffalo bills", "bills": "buffalo bills",
    "cin": "cincinnati bengals", "bengals": "cincinnati bengals",
    "cle": "cleveland browns", "browns": "cleveland browns",
    "den": "denver broncos", "broncos": "denver broncos",
    "hou": "houston texans", "texans": "houston texans",
    "ind": "indianapolis colts", "colts": "indianapolis colts",
    "jac": "jacksonville jaguars", "jax": "jacksonville jaguars", "jaguars": "jacksonville jaguars",
    "kc": "kansas city chiefs", "kan": "kansas city chiefs", "chiefs": "kansas city chiefs",
    "lv": "las vegas raiders", "lvr": "las vegas raiders", "raiders": "las vegas raiders",
    "lac": "los angeles chargers", "la chargers": "los angeles chargers", "chargers": "los angeles chargers",
    "mia": "miami dolphins", "dolphins": "miami dolphins",
    "ne": "new england patriots", "nwe": "new england patriots", "patriots": "new england patriots",
    "nyj": "new york jets", "jets": "new york jets",
    "pit": "pittsburgh steelers", "steelers": "pittsburgh steelers",
    "ten": "tennessee titans", "titans": "tennessee titans",
}

def norm_team(s: str) -> str:
    if s is None:
        return ""
    base = " ".join(str(s).strip().lower().replace(".", "").split())
    return _TEAM_ALIASES.get(base, base)

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups (matchups-only UI)
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    # This pulls fresh schedule/scores each run — will include last week's games when available
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
    if not HAS_STATSAPI or not name:
        return None
    try:
        candidates = statsapi.lookup_player(name)
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

        home_name = g.get("home_probable_pitcher") or g.get("probable_pitcher_home") or g.get("probable_pitcher")
        away_name = g.get("away_probable_pitcher") or g.get("probable_pitcher_away")

        home_id = g.get("home_probable_pitcher_id") or g.get("probable_pitcher_home_id")
        away_id = g.get("away_probable_pitcher_id") or g.get("probable_pitcher_away_id")

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
# Player Props helpers
# ==============================================================================
def _defense_from_upload(upload: Optional[bytes]) -> Optional[pd.DataFrame]:
    """Load defense CSV (uploaded) else try disk file 'NFL_defense.csv'."""
    df = None
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception:
            # Some browsers send bytes-like, try decoding stringio
            try:
                df = pd.read_csv(StringIO(upload.getvalue().decode("utf-8")))
            except Exception:
                df = None
    if df is None:
        try:
            df = pd.read_csv("NFL_defense.csv")
        except Exception:
            return None
    return df

def _extract_def_epa(df: pd.DataFrame) -> pd.DataFrame:
    """Expect columns like TEAM (or Team), SEASON (optional), EPA/PLAY (or similar)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["team","epa_play"])

    cols = {c.lower(): c for c in df.columns}
    # Team column
    team_col = None
    for k in ["team","defense","def_team","club","tm","name"]:
        if k in cols: team_col = cols[k]; break
    if team_col is None:
        # Try first column
        team_col = df.columns[0]

    # EPA/play column
    epa_col = None
    for k in ["epa/play","epa_play","epa per play","epa_per_play","def_epa/play","def_epa"]:
        if k in cols: epa_col = cols[k]; break
    if epa_col is None:
        # Look for any column containing 'epa' and 'play'
        cand = [c for c in df.columns if "epa" in c.lower() and "play" in c.lower()]
        if cand: epa_col = cand[0]

    out = pd.DataFrame({
        "team": df[team_col].astype(str).map(norm_team),
        "epa_play": pd.to_numeric(df[epa_col], errors="coerce") if epa_col else np.nan
    })
    out = out.dropna(subset=["team"]).drop_duplicates("team")
    return out

def _def_adjust_factor(epa_play: float) -> float:
    """
    Simple defense factor from EPA/play:
      def_strength = -epa_play  (positive means tougher D)
      factor = 1 - 0.5 * def_strength  -> clamp [0.7, 1.3]
      Examples:
        EPA/play = -0.16 (tough) -> def_strength=+0.16 -> factor=1 - 0.08 = 0.92  (mean -8%)
        EPA/play = +0.14 (weak)  -> def_strength=-0.14 -> factor=1 + 0.07 = 1.07 (mean +7%)
    """
    if pd.isna(epa_play):
        return 1.0
    def_strength = -float(epa_play)
    factor = 1.0 - 0.5 * def_strength
    return float(np.clip(factor, 0.7, 1.3))

def _basic_mean_sd_from_row(row: pd.Series, stat: str) -> Tuple[float, float]:
    """
    Pull a simple per-game mean (Y/G if present, else Yds/games) and a basic SD.
    No knobs. If multiple games present, we try to infer SD from per-game data if supplied,
    else fallback to a fraction of the mean.
    """
    # Try Y/G first
    mean = None
    for k in ["Y/G","Yds/G","YPG","Y_per_G","Yards/G","Y/G "]:
        if k in row and _safe_float(row[k]) == _safe_float(row[k]):  # not NaN
            mean = float(row[k])
            break
    # Otherwise compute from Yds and G
    if mean is None:
        # columns typically: Yds, G
        yds_key = None
        for k in ["Yds","YDS","Yards","Yds "]:
            if k in row: yds_key = k; break
        g_key = None
        for k in ["G","GP","Games"]:
            if k in row: g_key = k; break
        if yds_key and g_key and _safe_float(row[yds_key]) == _safe_float(row[yds_key]) and _safe_float(row[g_key]) > 0:
            mean = float(row[yds_key]) / float(row[g_key])
        else:
            mean = 0.0

    # Very simple SD: use in-data SD if we can spot "Succ%" (as volatility proxy) otherwise fallback
    # We'll just use a fraction of mean by position/stat:
    m = float(mean)
    if stat == "Pass Yards":
        sd = max(10.0, 0.55 * m)
    elif stat == "Rush Yards":
        sd = max(8.0, 0.50 * m)
    else:  # "Rec Yards"
        sd = max(7.0, 0.48 * m)

    return m, sd

def _stat_column_names(stat: str):
    # which yardage column name to prefer when calculating season sums if needed
    if stat == "Pass Yards":
        return ["Yds","PassYds","PYds"]
    if stat == "Rush Yards":
        return ["Yds","RushYds","RYds"]
    return ["Yds","RecYds"]

def _position_for_stat(stat: str) -> str:
    if stat == "Pass Yards":
        return "QB"
    if stat == "Rush Yards":
        return "RB"
    return "WR"

def _def_side_for_stat(stat: str) -> str:
    # Which defensive facet we proxy with overall EPA/play here (kept basic as requested)
    # If you later add pass/rush split columns we can branch here.
    if stat == "Rush Yards":
        return "rush"
    elif stat == "Pass Yards":
        return "pass"
    return "pass"  # WR receiving ~ pass defense

# Monte Carlo on (truncated) Normal
def _prob_over(mean: float, sd: float, line: float, trials: int = SIM_TRIALS) -> Tuple[float, float]:
    sd = max(1e-6, float(sd))
    draws = np.random.normal(loc=float(mean), scale=sd, size=trials)
    draws = np.maximum(0.0, draws)  # yards can't be negative
    p_over = float((draws > float(line)).mean())
    return p_over, float(draws.mean())

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Win probabilities are simulated from **team scoring rates only** "
    "(NFL: points for/against in 2025; MLB: runs scored/allowed in 2025). "
    "Player Props uses your uploaded player CSVs and a weekly Defense CSV (EPA/play)."
)

page = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page (matchups only) --------------------------
if page == "NFL":
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
                   " — " + upcoming["date"].astype(str))
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"])

    pick = st.selectbox("Choose a matchup", choices, index=0)
    sel = upcoming.iloc[list(choices).index(pick)]
    home, away = sel["home_team"], sel["away_team"]

    st.write(f"**Selected:** {home} (home) vs {away} (away)")

    try:
        muH, muA = nfl_matchup_mu(nfl_rates, home, away)
        pH, pA, eH, eA, eTot = simulate_poisson_game(muH, muA, trials=SIM_TRIALS)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Home win %", f"{pH*100:.1f}%")
        with col2: st.metric("Away win %", f"{pA*100:.1f}%")
        with col3: st.metric("Exp total pts", f"{eTot:.1f}")
        st.caption(f"Exp pts — {home}: {eH:.1f}, {away}: {eA:.1f}")
    except Exception as e:
        st.error(f"Simulation error: {e}")

# -------------------------- MLB page (unchanged) ------------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 REG season")

    try:
        mlb_rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build MLB team rates: {e}")
        st.stop()

    home = st.selectbox("Home", sorted(mlb_rates["team"].tolist()))
    away = st.selectbox("Away", sorted([t for t in mlb_rates["team"].tolist() if t != home]))

    muH, muA = mlb_matchup_mu(mlb_rates, home, away)

    probables = _today_probables()
    homeEra = awayEra = None
    if not probables.empty:
        h_row = probables.loc[(probables["side"]=="Home") & (probables["team"].str.lower()==home.lower())]
        a_row = probables.loc[(probables["side"]=="Away") & (probables["team"].str.lower()==away.lower())]
        if not h_row.empty:
            homeEra = _safe_float(h_row.iloc[0]["ERA"], None)
        if not a_row.empty:
            awayEra = _safe_float(a_row.iloc[0]["ERA"], None)

    muH_adj, muA_adj = _apply_pitcher_adjustment(muH, muA, homeEra, awayEra)
    pH, pA, eH, eA, eTot = simulate_poisson_game(muH_adj, muA_adj, trials=SIM_TRIALS)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Home win %", f"{pH*100:.1f}%")
    with col2: st.metric("Away win %", f"{pA*100:.1f}%")
    with col3: st.metric("Exp total runs", f"{eTot:.2f}")
    st.caption(f"Exp runs — {home}: {eH:.2f}, {away}: {eA:.2f}")
    if homeEra is not None or awayEra is not None:
        st.caption(f"Probables ERA — {home}: {homeEra if homeEra is not None else 'N/A'}, "
                   f"{away}: {awayEra if awayEra is not None else 'N/A'}")

# -------------------------- Player Props page ---------------------------------
else:
    st.subheader("📈 Player Props — Upload CSVs + Type Opponent Defense")

    st.markdown("**Upload your CSVs** (just like you’ve been pasting/exporting):")
    c1, c2 = st.columns(2)
    with c1:
        qb_file = st.file_uploader("QB CSV", type=["csv"], key="qb_csv")
        rb_file = st.file_uploader("RB CSV", type=["csv"], key="rb_csv")
    with c2:
        wr_file = st.file_uploader("WR CSV", type=["csv"], key="wr_csv")
        def_file = st.file_uploader("Defense CSV (EPA/play)", type=["csv"], key="def_csv")

    # Load CSVs if provided
    def_df = _extract_def_epa(_defense_from_upload(def_file))
    if def_df is None or def_df.empty:
        st.warning("No defense CSV found (upload one or place `NFL_defense.csv` in the app folder). Using **no adjustment**.")
    else:
        st.caption("Defense table loaded (EPA/play). Stronger D => lower expected yards.")

    def_lookup = {}
    if def_df is not None and not def_df.empty:
        def_lookup = {t: e for t, e in zip(def_df["team"], def_df["epa_play"])}

    # Read player CSVs
    def _read_player_csv(fileobj) -> Optional[pd.DataFrame]:
        if fileobj is None:
            return None
        try:
            df = pd.read_csv(fileobj)
            return df
        except Exception:
            try:
                df = pd.read_csv(StringIO(fileobj.getvalue().decode("utf-8")))
                return df
            except Exception:
                return None

    qbs = _read_player_csv(qb_file)
    rbs = _read_player_csv(rb_file)
    wrs = _read_player_csv(wr_file)

    # Build a list of players (no team dropdown – you just pick the player)
    def _player_options(df: Optional[pd.DataFrame], pos_tag: str):
        if df is None or df.empty: return []
        name_col = None
        for k in ["Player","player","Name","NAME"]:
            if k in df.columns: name_col = k; break
        if name_col is None:
            # assume first col is player
            name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        # attach pos tag so you can pick which stat model to apply
        return [f"{str(x)} [{pos_tag}]" for x in df[name_col].astype(str).tolist()]

    opts = []
    opts += _player_options(qbs, "QB")
    opts += _player_options(rbs, "RB")
    opts += _player_options(wrs, "WR")

    if not opts:
        st.info("Upload at least one of QB / RB / WR CSVs to pick a player.")
        st.stop()

    player_pick = st.selectbox("Choose a player", sorted(opts))
    stat_pick = st.radio("Stat", ["Pass Yards","Rush Yards","Rec Yards"], horizontal=True)
    line_val = st.number_input("Yardage line (e.g., 250.5 for QB, 65.5 for RB, etc.)", value=50.5, step=0.5)

    # free-text defense team (no dropdown)
    opp_def_text = st.text_input("Opponent defense (type team, e.g., 'Dallas Cowboys' or 'DAL')", value="")

    # locate row for selected player
    sel_pos = player_pick.rsplit("[",1)[-1].strip(" ]")
    player_name = player_pick.rsplit("[",1)[0].strip()

    sel_df = qbs if sel_pos == "QB" else (rbs if sel_pos == "RB" else wrs)

    # figure name column
    name_col = None
    for k in ["Player","player","Name","NAME"]:
        if sel_df is not None and k in sel_df.columns:
            name_col = k; break
    if name_col is None and sel_df is not None:
        name_col = sel_df.columns[1] if len(sel_df.columns) > 1 else sel_df.columns[0]

    row = None
    if sel_df is not None:
        cand = sel_df.loc[sel_df[name_col].astype(str) == player_name]
        if not cand.empty:
            row = cand.iloc[0]

    if row is None:
        st.error("Couldn't find selected player in the uploaded CSV.")
        st.stop()

    # compute basic mean/sd
    mean_base, sd_base = _basic_mean_sd_from_row(row, stat_pick)

    # defense factor
    opp_key = norm_team(opp_def_text) if opp_def_text else ""
    epa_val = def_lookup.get(opp_key, np.nan) if opp_key else np.nan
    d_factor = _def_adjust_factor(epa_val) if opp_key else 1.0

    mean_adj = float(mean_base) * float(d_factor)
    sd_adj = float(sd_base)  # keep SD basic as requested (no weighting knobs)

    # simulate probability of going over the line
    if st.button("Run simulation"):
        p_over, sim_mean = _prob_over(mean_adj, sd_adj, line_val, trials=SIM_TRIALS)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("P(Over)", f"{p_over*100:.1f}%")
        with col2: st.metric("Mean (adj.)", f"{sim_mean:.1f}")
        with col3: st.metric("Def factor", f"{d_factor:.3f}")
        if not np.isnan(epa_val):
            st.caption(f"Defense EPA/play: {epa_val:+.3f}  → factor {d_factor:.3f} "
                       f"({opp_key if opp_key else 'N/A'})")
        else:
            st.caption("No defense match found — using base mean (factor 1.00).")

        # small trace table
        st.write(pd.DataFrame({
            "Player":[player_name],
            "Pos":[sel_pos],
            "Stat":[stat_pick],
            "Base mean":[round(mean_base,1)],
            "Base SD":[round(sd_base,1)],
            "Adj mean":[round(mean_adj,1)],
            "Line":[line_val],
            "P(Over %)":[round(p_over*100,1)],
        }))
