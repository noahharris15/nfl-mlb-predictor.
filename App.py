# app.py — NFL + MLB + Player Props + College Football (auto)
# - NFL/MLB pages: matchups from 2025 team scoring rates (your original flow)
# - Player Props: upload QB/RB/WR CSVs; opponent defense baked in (EPA->factor)
# - College Football: auto pulls this season from CollegeFootballData (cfbd)

from __future__ import annotations
import io, math
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st

# ---- NFL teams/schedules ----
import nfl_data_py as nfl

# ---- MLB team results (Baseball-Reference via pybaseball) ----
from pybaseball import schedule_and_record

# ---- College Football (auto via CFBD) ----
HAS_CFBD = False
try:
    import cfbd
    from cfbd.rest import ApiException
    HAS_CFBD = True
except Exception:
    HAS_CFBD = False

# ==================== Embedded NFL defense EPA/play (fallback) =================
# Lower EPA = tougher defense. These are normalized to a multiplicative factor.
DEF_EPA_2025_FALLBACK = {
    "MIN": -0.27, "JAX": -0.15, "GB": -0.13, "SF": -0.11, "ATL": -0.10,
    "IND": -0.08, "LAC": -0.08, "DEN": -0.08, "LAR": -0.07, "SEA": -0.07,
    "PHI": -0.06, "TB": -0.05, "CAR": -0.05, "ARI": -0.03, "CLE": -0.02,
    "WAS": -0.02, "HOU":  0.00, "KC": 0.01, "DET": 0.01, "LV": 0.03,
    "PIT": 0.05, "CIN": 0.05, "NO": 0.05, "BUF": 0.05, "CHI": 0.06,
    "NE": 0.09, "NYJ": 0.10, "TEN": 0.11, "BAL": 0.11, "NYG": 0.13,
    "DAL": 0.21, "MIA": 0.28,
}
ALIAS_TO_STD = {
    "GNB":"GB","SFO":"SF","KAN":"KC","NWE":"NE","NOR":"NO","TAM":"TB",
    "LVR":"LV","SDG":"LAC","STL":"LAR","JAC":"JAX","WSH":"WAS","LA":"LAR","OAK":"LV",
}
def norm_team_code(code: str) -> str:
    c = (code or "").strip().upper()
    return ALIAS_TO_STD.get(c, c)

def build_def_factor_map(epa_map: Dict[str,float]) -> Dict[str,float]:
    if not epa_map: return {}
    s = pd.Series(epa_map, dtype=float)
    mu, sd = float(s.mean()), float(s.std(ddof=0) or 1.0)
    z = (s - mu) / (sd if sd > 1e-9 else 1.0)
    # map z-scores to ~0.85..1.15 (lower EPA => lower factor)
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k, v in factors.items()}

DEF_FACTOR_2025 = build_def_factor_map(DEF_EPA_2025_FALLBACK)

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6
EPS = 1e-9

# MLB BR team names
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

# -------------------------- generic helpers -----------------------------------
def _poisson_sim(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

# ==============================================================================
# NFL (2025): team PF/PA + upcoming matchups
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    date_col: Optional[str] = None
    for c in ("gameday","game_date","start_time"):
        if c in sched.columns:
            date_col = c; break

    played = sched.dropna(subset=["home_score","away_score"])
    home = played.rename(columns={
        "home_team":"team","away_team":"opp","home_score":"pf","away_score":"pa"
    })[["team","opp","pf","pa"]]
    away = played.rename(columns={
        "away_team":"team","home_team":"opp","away_score":"pf","home_score":"pa"
    })[["team","opp","pf","pa"]]
    long = pd.concat([home,away], ignore_index=True)

    if long.empty:
        per = 45.0/2.0
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
        rates = pd.DataFrame({"team":teams32,"PF_pg":per,"PA_pg":per})
    else:
        team = long.groupby("team", as_index=False).agg(
            games=("pf","size"), PF=("pf","sum"), PA=("pa","sum")
        )
        rates = pd.DataFrame({
            "team":team["team"],
            "PF_pg":team["PF"]/team["games"],
            "PA_pg":team["PA"]/team["games"],
        })
        league_total = float((long["pf"]+long["pa"]).mean())
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
            upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+"," ", regex=True)

    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = rates.loc[rates["team"].str.lower()==home.lower()]
    rA = rates.loc[rates["team"].str.lower()==away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"])/2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"])/2.0)
    return mu_home, mu_away

# ==============================================================================
# MLB (2025): team RS/RA from BR
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
                RS_pg = float(sar["R"].sum()/games)
                RA_pg = float(sar["RA"].sum()/games)
            rows.append({"team":name,"RS_pg":RS_pg,"RA_pg":RA_pg})
        except Exception:
            rows.append({"team":name,"RS_pg":4.5,"RA_pg":4.5})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean()); league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9*df["RS_pg"] + 0.1*league_rs
        df["RA_pg"] = 0.9*df["RA_pg"] + 0.1*league_ra
    return df

# ==============================================================================
# College Football — auto via CFBD
# ==============================================================================
def _init_cfbd_client():
    if not HAS_CFBD: return None
    api_key = st.secrets.get("CFBD_API_KEY", "")
    if not api_key: return None
    cfg = cfbd.Configuration()
    cfg.api_key["Authorization"] = api_key
    cfg.api_key_prefix["Authorization"] = "Bearer"
    return cfbd.ApiClient(cfg)

@st.cache_data(show_spinner=False)
def cfb_team_rates_by_games(season: int = 2025):
    client = _init_cfbd_client()
    if client is None:
        return pd.DataFrame(), pd.DataFrame(), "no_key"
    try:
        games_api = cfbd.GamesApi(client)
        games = games_api.get_games(year=season, division="fbs", season_type="both")
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"api_error: {e}"
    if not games:
        return pd.DataFrame(), pd.DataFrame(), "no_games"

    rows_done, rows_upcoming = [], []
    for g in games:
        home = getattr(g, "home_team", None)
        away = getattr(g, "away_team", None)
        hs   = getattr(g, "home_points", None)
        as_  = getattr(g, "away_points", None)
        date = str(getattr(g, "start_date", "")) or ""
        if home and away and hs is not None and as_ is not None:
            rows_done.append({"team":home, "opp":away, "pf":hs, "pa":as_})
            rows_done.append({"team":away, "opp":home, "pf":as_, "pa":hs})
        elif home and away:
            rows_upcoming.append({"home_team":home, "away_team":away, "date":date})
    df_done = pd.DataFrame(rows_done)
    df_upc  = pd.DataFrame(rows_upcoming)

    if df_done.empty:
        return pd.DataFrame(), df_upc, "no_results"

    team = df_done.groupby("team", as_index=False).agg(
        games=("pf","size"), PF=("pf","sum"), PA=("pa","sum")
    )
    rates = pd.DataFrame({
        "team": team["team"],
        "PF_pg": team["PF"]/team["games"],
        "PA_pg": team["PA"]/team["games"],
    })

    league_total = float((df_done["pf"]+df_done["pa"]).mean()) if not df_done.empty else 56.0
    prior = league_total/2.0
    shrink = np.clip(1.0 - team["games"]/4.0, 0.0, 1.0)
    rates["PF_pg"] = (1-shrink)*rates["PF_pg"] + shrink*prior
    rates["PA_pg"] = (1-shrink)*rates["PA_pg"] + shrink*prior

    return rates.sort_values("team").reset_index(drop=True), df_upc, "ok"

# ==============================================================================
# Player Props helpers
# ==============================================================================
def _yardage_column_guess(df: pd.DataFrame, pos: str) -> str:
    prefer = ["Y/G","Yds/G","YDS/G","Yards/G","PY/G","RY/G","Rec Y/G",
              "Yds","Yards","yds","yards"]
    low = [c.lower() for c in df.columns]
    for wanted in [p.lower() for p in prefer]:
        if wanted in low: return df.columns[low.index(wanted)]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return df.columns[-1]

def _player_column_guess(df: pd.DataFrame) -> str:
    # prefer "Player" / "Name" / last column without commas
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("player","name"): return c
    return df.columns[0]

def _clean_player_name(val: str) -> str:
    # handles inputs like "12,John Doe,..." → "John Doe"
    s = str(val)
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        # find the first part that has a space (likely a name)
        for p in parts:
            if " " in p and not p.replace(" ","").isdigit():
                return p
        return parts[0]
    return s

def _estimate_sd(mean_val: float, pos: str) -> float:
    m = float(mean_val)
    if pos == "QB": return max(35.0, 0.60*m)
    if pos == "RB": return max(20.0, 0.75*m)
    return max(22.0, 0.85*m)  # WR

def run_prop_sim(mean_yards: float, line: float, sd: float) -> Tuple[float,float]:
    sd = max(5.0, float(sd))
    z = (line - mean_yards) / sd
    p_over = float(1.0 - 0.5*(1.0 + math.erf(z / math.sqrt(2))))
    return np.clip(p_over, 0.0, 1.0), 1.0 - np.clip(p_over, 0.0, 1.0)

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB + CFB Predictor", layout="wide")
st.title("🏈⚾🏈 NFL + MLB + College Football — 2025 (stats only)")
st.caption(
    "NFL & MLB pages use team scoring rates. Player Props uses your CSV + embedded NFL defense. "
    "College Football auto-loads this season from CollegeFootballData when a key is set in secrets."
)

page = st.radio("Pick a page", ["NFL","MLB","College Football","Player Props"], horizontal=True)

# -------------------------- NFL --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()
    if not upcoming.empty and {"home_team","away_team"}.issubset(upcoming.columns):
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
        st.info("No upcoming list — pick any two teams:")
        home = st.selectbox("Home team", rates["team"].tolist())
        away = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != home])
    if home and away:
        mu_h, mu_a = nfl_matchup_mu(rates, home, away)
        pH, pA, mH, mA = _poisson_sim(mu_h, mu_a)
        st.markdown(
            f"**{home}** vs **{away}** — Expected points: {mH:.1f}–{mA:.1f} · "
            f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
        )

# -------------------------- MLB --------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 REG season (team scoring rates)")
    rates = mlb_team_rates_2025()
    t1 = st.selectbox("Home team", rates["team"].tolist())
    t2 = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != t1])
    H = rates.loc[rates["team"] == t1].iloc[0]
    A = rates.loc[rates["team"] == t2].iloc[0]
    mu_home = (H["RS_pg"] + A["RA_pg"]) / 2.0
    mu_away = (A["RS_pg"] + H["RA_pg"]) / 2.0
    pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)
    st.markdown(
        f"**{t1}** vs **{t2}** — Expected runs: {mH:.1f}–{mA:.1f} · "
        f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**"
    )

# -------------------------- College Football page --------------------------
elif page == "College Football":
    st.subheader("🏈 College Football — 2025 (auto from CFBD)")

    # Quick indicator so we can tell if the key is visible
    key_present = bool(st.secrets.get("CFBD_API_KEY"))
    if not key_present:
        st.info('Add your CFBD key in **Manage app → Settings → Secrets** as:\n\n`CFBD_API_KEY="YOUR_KEY"`')
        st.stop()

    # Lazy import so the rest of the app works even if cfbd isn't installed locally
    try:
        import cfbd  # pip package: cfbd
    except Exception as e:
        st.error("The `cfbd` package is not installed. Add `cfbd` to requirements.txt and redeploy.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def _cfbd_rates_for_year(year: int):
        """Fetch all regular-season games for `year` and return PF/PA per team."""
        cfg = cfbd.Configuration()
        # CFBD uses an Authorization: Bearer <key> header
        cfg.api_key["Authorization"] = st.secrets["CFBD_API_KEY"]
        cfg.api_key_prefix["Authorization"] = "Bearer"
        client = cfbd.ApiClient(cfg)
        games_api = cfbd.GamesApi(client)

        # Pull regular-season games
        games = games_api.get_games(year=year, season_type="regular")

        rows = []
        for g in games:
            # Some early games can be missing scores; guard it
            if (getattr(g, "home_points", None) is None) or (getattr(g, "away_points", None) is None):
                continue
            rows.append({"team": g.home_team, "pf": g.home_points, "pa": g.away_points})
            rows.append({"team": g.away_team, "pf": g.away_points, "pa": g.home_points})

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(), 0

        agg = df.groupby("team", as_index=False).agg(
            games=("pf", "size"),
            PF=("pf", "sum"),
            PA=("pa", "sum"),
        )
        agg["PF_pg"] = agg["PF"] / agg["games"]
        agg["PA_pg"] = agg["PA"] / agg["games"]

        # Light shrink toward the national average for small sample sizes
        league_total = float((df["pf"] + df["pa"]).mean()) if not df.empty else 52.0
        prior = league_total / 2.0
        shrink = np.clip(1.0 - agg["games"] / 4.0, 0.0, 1.0)
        agg["PF_pg"] = (1 - shrink) * agg["PF_pg"] + shrink * prior
        agg["PA_pg"] = (1 - shrink) * agg["PA_pg"] + shrink * prior

        return agg[["team", "PF_pg", "PA_pg"]], len(games)

    # Try 2025; if there are zero scored games, fall back to 2024 so page isn’t blank
    try:
        rates_2025, n_games_2025 = _cfbd_rates_for_year(2025)
        if rates_2025.empty:
            st.warning("No 2025 scored games returned yet; showing 2024 as a fallback.")
            rates, _ = _cfbd_rates_for_year(2024)
            season_label = "2024"
        else:
            rates = rates_2025
            season_label = "2025"
    except Exception as e:
        st.error(f"CFBD error: {e}")
        st.stop()

    if rates.empty:
        st.info("No team data available yet.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("Home team", rates["team"].tolist())
    with c2:
        away = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != home])

    rH = rates.loc[rates["team"] == home].iloc[0]
    rA = rates.loc[rates["team"] == away].iloc[0]
    mu_home = max(0.1, (float(rH["PF_pg"]) + float(rA["PA_pg"])) / 2.0)  # no HFA by default in CFB
    mu_away = max(0.1, (float(rA["PF_pg"]) + float(rH["PA_pg"])) / 2.0)

    pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)
    st.markdown(
        f"**{home}** vs **{away}** — {season_label}  \n"
        f"Expected points: **{mH:.1f}–{mA:.1f}**  ·  "
        f"P({home} win) **{100*pH:.1f}%**,  P({away} win) **{100*pA:.1f}%**"
    )

    with st.expander("Show team rates"):
        st.dataframe(rates.sort_values("team").reset_index(drop=True))
# -------------------------- Player Props --------------------------
else:
    st.subheader("🎯 Player Props — upload QB/RB/WR CSVs")

    c1, c2, c3 = st.columns(3)
    with c1: qb_up = st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2: rb_up = st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3: wr_up = st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

    def _read_any(up):
        if up is None: return pd.DataFrame()
        nm = (up.name or "").lower()
        if nm.endswith((".xlsx",".xls")): return pd.read_excel(up)
        return pd.read_csv(up)

    dfs = {}
    if qb_up: dfs["QB"] = _read_any(qb_up).copy()
    if rb_up: dfs["RB"] = _read_any(rb_up).copy()
    if wr_up: dfs["WR"] = _read_any(wr_up).copy()
    if not dfs:
        st.info("Upload at least one CSV to begin."); st.stop()

    pos = st.selectbox("Market", ["QB","RB","WR"])
    df = dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded yet."); st.stop()

    name_col = _player_column_guess(df)
    yard_col = _yardage_column_guess(df, pos)

    # Show name-only choices
    names = [_clean_player_name(v) for v in df[name_col].astype(str).tolist()]
    player = st.selectbox("Player", names)

    # Find the original row for the chosen clean name
    row_idx = None
    for i, raw in enumerate(df[name_col].astype(str).tolist()):
        if _clean_player_name(raw) == player:
            row_idx = i; break
    row = df.iloc[[row_idx]] if row_idx is not None else pd.DataFrame()

    opp_in = st.text_input("Opponent team code (e.g., DAL, PHI). Aliases like KAN/NOR/GNB/SFO are OK.", value="")
    opp = norm_team_code(opp_in)

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

    with st.expander("Embedded NFL defense factors"):
        show = pd.DataFrame({
            "TEAM": list(DEF_FACTOR_2025.keys()),
            "EPA/play": [DEF_EPA_2025_FALLBACK[k] for k in DEF_FACTOR_2025.keys()],
            "DEF_FACTOR": list(DEF_FACTOR_2025.values())
        }).sort_values("DEF_FACTOR")
        st.dataframe(show.reset_index(drop=True))
