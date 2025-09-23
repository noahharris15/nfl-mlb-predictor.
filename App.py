# app.py — NFL + MLB predictor (2025) + Player Props
# Uses a defense table loaded automatically from 'nfl Defense.xlsx' (or .csv) in the repo.
# If not found, falls back to a baked-in defense table.

import math
from datetime import date
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------- Streamlit page ---------------------------------
st.set_page_config(page_title="NFL + MLB Predictor — 2025", layout="wide")

# ==============================================================================
# CONSTANTS
# ==============================================================================
SIM_TRIALS = 50000
HOME_EDGE_NFL = 0.6
EPS = 1e-9

# Fixed SDs for props simulation (no sliders)
SD_QB_PASS_YDS = 55.0
SD_RB_RUSH_YDS = 22.0
SD_WR_REC_YDS  = 30.0

# ------------------------------------------------------------------------------
# FALLBACK: baked-in defense EPA/pass & EPA/rush (used only if file missing)
# ------------------------------------------------------------------------------
_FALLBACK_DEF = [
    ("SFO", -0.27, -0.01), ("ATL", -0.18, -0.05), ("LAR", -0.20,  0.00),
    ("JAX", -0.20,  0.03), ("GNB", -0.08, -0.19), ("DEN", -0.03, -0.19),
    ("LAC", -0.20,  0.18), ("LVR",  0.12, -0.45), ("MIN", -0.25,  0.13),
    ("WAS", -0.06, -0.04), ("PHI", -0.08,  0.00), ("SEA",  0.06, -0.17),
    ("IND", -0.13,  0.17), ("DET",  0.16, -0.25), ("ARI", -0.01, -0.05),
    ("BAL", -0.06,  0.12), ("CLE",  0.15, -0.18), ("NOR",  0.05, -0.04),
    ("TAM",  0.08, -0.09), ("CIN",  0.03, -0.01), ("TEN",  0.03,  0.04),
    ("HOU",  0.03,  0.08), ("BUF",  0.05,  0.05), ("KAN",  0.19,  0.04),
    ("CAR",  0.12,  0.12), ("CHI",  0.27,  0.01), ("NYG",  0.12,  0.15),
    ("NYJ",  0.16,  0.11), ("NWE",  0.33, -0.22), ("PIT",  0.23,  0.09),
    ("DAL",  0.26,  0.06), ("MIA",  0.41,  0.13),
]
_FALLBACK_DF = pd.DataFrame(_FALLBACK_DEF, columns=["team","epa_pass","epa_rush"])

# ==============================================================================
# DEFENSE LOADER (reads your repo file)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_defense_table() -> pd.DataFrame:
    """
    Load defense EPA table from repo root.
    Accepts:
      - 'nfl Defense.xlsx'  (your file)
      - 'nfl_defense.xlsx'
      - 'nfl Defense.csv' or 'nfl_defense.csv'
    Required columns (case-insensitive, flexible names):
      - team code: any of ['team','tm','code','opponent','opp']
      - pass EPA:  name containing 'epa' and 'pass'
      - rush EPA:  name containing 'epa' and 'rush'
    Returns DataFrame with: team, epa_pass, epa_rush, factor_vs_pass, factor_vs_rush
    """
    candidates = [
        "nfl Defense.xlsx", "nfl_defense.xlsx",
        "nfl Defense.csv",  "nfl_defense.csv"
    ]

    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower().strip(): c for c in df.columns}
        # team col
        team_col = None
        for k in ["team","tm","code","opponent","opp"]:
            if k in cols:
                team_col = cols[k]; break
        # pass/rush epa columns
        pass_col = None; rush_col = None
        for c in df.columns:
            cl = c.lower()
            if "epa" in cl and "pass" in cl and pass_col is None:
                pass_col = c
            if "epa" in cl and ("rush" in cl or "run" in cl) and rush_col is None:
                rush_col = c
        if team_col is None or pass_col is None or rush_col is None:
            raise ValueError("Could not detect columns for team / EPA pass / EPA rush.")

        out = df[[team_col, pass_col, rush_col]].copy()
        out.columns = ["team","epa_pass","epa_rush"]
        out["team"] = out["team"].astype(str).str.upper().str.strip()
        # keep only likely NFL codes (3 letters)
        out = out[out["team"].str.len().between(2,4)]
        # coerce numerics
        out["epa_pass"] = pd.to_numeric(out["epa_pass"], errors="coerce")
        out["epa_rush"] = pd.to_numeric(out["epa_rush"], errors="coerce")
        out = out.dropna(subset=["epa_pass","epa_rush"])
        out = out.drop_duplicates(subset=["team"])
        # normalize to 32 teams if extra rows exist
        return out

    for fn in candidates:
        try:
            if fn.lower().endswith(".xlsx"):
                raw = pd.read_excel(fn)
            else:
                raw = pd.read_csv(fn)
            base = _standardize(raw)
            if base.empty:
                continue
            # build factors 0.85..1.15 (lower EPA = tougher = smaller)
            pass_pct = base["epa_pass"].rank(pct=True, method="average")
            rush_pct = base["epa_rush"].rank(pct=True, method="average")
            base["factor_vs_pass"] = 0.85 + 0.30 * pass_pct
            base["factor_vs_rush"] = 0.85 + 0.30 * rush_pct
            return base.set_index("team")
        except Exception:
            continue

    # fallback
    fb = _FALLBACK_DF.copy()
    pass_pct = fb["epa_pass"].rank(pct=True, method="average")
    rush_pct = fb["epa_rush"].rank(pct=True, method="average")
    fb["factor_vs_pass"] = 0.85 + 0.30 * pass_pct
    fb["factor_vs_rush"] = 0.85 + 0.30 * rush_pct
    return fb.set_index("team")

DEF_FACTORS = load_defense_table()

def defense_factor(opponent_code: Optional[str], market: str) -> float:
    if not opponent_code:
        return 1.00
    code = str(opponent_code).strip().upper()
    if code in DEF_FACTORS.index:
        if market in ("QB_PASS_YDS", "WR_REC_YDS"):
            return float(DEF_FACTORS.loc[code, "factor_vs_pass"])
        elif market == "RB_RUSH_YDS":
            return float(DEF_FACTORS.loc[code, "factor_vs_rush"])
    return 1.00

# ==============================================================================
# NFL team rates + schedule (unchanged)
# ==============================================================================
try:
    import nfl_data_py as nfl
except Exception:
    nfl = None

@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    if nfl is None:
        raise RuntimeError("nfl_data_py is required for the NFL page.")
    sched = nfl.import_schedules([2025])

    date_col = next((c for c in ("gameday","game_date") if c in sched.columns), None)

    played = sched.dropna(subset=["home_score","away_score"])
    home = played.rename(columns={
        "home_team":"team","away_team":"opp","home_score":"pf","away_score":"pa"
    })[["team","opp","pf","pa"]]
    away = played.rename(columns={
        "away_team":"team","home_team":"opp","away_score":"pf","home_score":"pa"
    })[["team","opp","pf","pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 45.0 / 2.0
        teams32 = ["Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills",
            "Carolina Panthers","Chicago Bears","Cincinnati Bengals","Cleveland Browns",
            "Dallas Cowboys","Denver Broncos","Detroit Lions","Green Bay Packers",
            "Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
            "Las Vegas Raiders","Los Angeles Chargers","Los Angeles Rams","Miami Dolphins",
            "Minnesota Vikings","New England Patriots","New Orleans Saints","New York Giants",
            "New York Jets","Philadelphia Eagles","Pittsburgh Steelers","San Francisco 49ers",
            "Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders"]
        rates = pd.DataFrame({"team":teams32,"PF_pg":per,"PA_pg":per})
    else:
        team = long.groupby("team", as_index=False).agg(games=("pf","size"), PF=("pf","sum"), PA=("pa","sum"))
        rates = pd.DataFrame({
            "team": team["team"],
            "PF_pg": team["PF"]/team["games"],
            "PA_pg": team["PA"]/team["games"],
        })
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"]/4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink)*rates["PF_pg"] + shrink*prior
        rates["PA_pg"] = (1 - shrink)*rates["PA_pg"] + shrink*prior

    upcoming_cols = ["home_team","away_team"] + ([date_col] if date_col else [])
    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][upcoming_cols].copy()
    if date_col:
        upcoming = upcoming.rename(columns={date_col:"date"})
    else:
        upcoming["date"] = ""
    for c in ["home_team","away_team"]:
        upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+"," ", regex=True)
    return rates, upcoming

def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home)); mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials); a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64); ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean()), float((h+a).mean())

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"]) / 2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"]) / 2.0)
    return mu_home, mu_away

# ==============================================================================
# MLB page (unchanged core)
# ==============================================================================
try:
    from pybaseball import schedule_and_record
    import statsapi
    HAS_STATSAPI = True
except Exception:
    schedule_and_record = None
    statsapi = None
    HAS_STATSAPI = False

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

@st.cache_data(show_spinner=False)
def mlb_team_rates_2025() -> pd.DataFrame:
    if schedule_and_record is None:
        raise RuntimeError("pybaseball is required for the MLB page.")
    rows = []
    for br, name in MLB_TEAMS_2025.items():
        try:
            sar = schedule_and_record(2025, br)
            sar = sar[pd.to_numeric(sar.get("R"), errors="coerce").notna()]
            sar = sar[pd.to_numeric(sar.get("RA"), errors="coerce").notna()]
            if sar.empty:
                RS_pg = RA_pg = 4.5; games = 0
            else:
                sar["R"] = sar["R"].astype(float); sar["RA"] = sar["RA"].astype(float)
                games = int(len(sar)); RS_pg = float(sar["R"].sum() / games); RA_pg = float(sar["RA"].sum() / games)
            rows.append({"team":name,"RS_pg":RS_pg,"RA_pg":RA_pg,"games":games})
        except Exception:
            rows.append({"team":name,"RS_pg":4.5,"RA_pg":4.5,"games":0})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean()); league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9*df["RS_pg"] + 0.1*league_rs
        df["RA_pg"] = 0.9*df["RA_pg"] + 0.1*league_ra
    return df

# ==============================================================================
# PLAYER PROPS (CSV uploads + built-in defense + optional injury multiplier)
# ==============================================================================
def _read_player_csv(upload, pos: str) -> pd.DataFrame:
    df = pd.read_csv(upload)
    df.columns = [c.strip() for c in df.columns]
    for col in ["Player","Yds","Team"]:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in {pos} CSV.")
    return df

def _player_mean(df: pd.DataFrame, player: str) -> float:
    row = df.loc[df["Player"].str.strip().str.lower() == player.strip().lower()]
    if row.empty:
        raise ValueError(f"Player '{player}' not found in uploaded CSV.")
    return float(row.iloc[0]["Yds"])

def _run_over_prob(mean_val: float, sd: float, line: float) -> float:
    draws = np.random.normal(loc=mean_val, scale=sd, size=SIM_TRIALS)
    return float((draws > line).mean())

# ==============================================================================
# UI LAYOUT
# ==============================================================================
st.title("🏈⚾ NFL + MLB Predictor — 2025")
mode = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page --------------------------
if mode == "NFL":
    st.subheader("🏈 NFL — team win probabilities (2025, schedule-based)")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build team rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming games found.")
        st.stop()

    if "date" in upcoming.columns and upcoming["date"].astype(str).str.len().gt(0).any():
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"] + " — " + upcoming["date"].astype(str))
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"])
    pick = st.selectbox("Upcoming matchup", choices)

    if pick:
        home = pick.split(" vs ")[0].split(" — ")[0].strip()
        away = pick.split(" vs ")[1].split(" — ")[0].strip()
        mu_h, mu_a = nfl_matchup_mu(nfl_rates, home, away)
        pH, pA, mh, ma, tot = simulate_poisson_game(mu_h, mu_a)
        st.write(f"**{home}** vs **{away}** — mean points: {mh:.1f}–{ma:.1f} (total {tot:.1f})")
        st.success(f"P({home} wins) = {pH:.1%} • P({away} wins) = {pA:.1%}")

# -------------------------- MLB page --------------------------
elif mode == "MLB":
    st.subheader("⚾ MLB — team win probabilities (2025)")
    try:
        rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build MLB rates: {e}")
        st.stop()
    home = st.selectbox("Home team", sorted(rates["team"].unique()))
    away = st.selectbox("Away team", sorted([t for t in rates["team"].unique() if t != home]))
    rH = rates.loc[rates["team"] == home].iloc[0]
    rA = rates.loc[rates["team"] == away].iloc[0]
    mu_home = max(EPS, (rH["RS_pg"] + rA["RA_pg"]) / 2.0)
    mu_away = max(EPS, (rA["RS_pg"] + rH["RA_pg"]) / 2.0)
    pH, pA, mh, ma, tot = simulate_poisson_game(mu_home, mu_away)
    st.write(f"**{home}** vs **{away}** — mean runs: {mh:.2f}–{ma:.2f} (total {tot:.2f})")
    st.success(f"P({home} wins) = {pH:.1%} • P({away} wins) = {pA:.1%}")

# -------------------------- Player Props page --------------------------
else:
    st.subheader("🎯 NFL Player Props — CSV upload + repo defense + injuries")
    st.caption("Defense multipliers are auto-loaded from the repo file each deploy.")

    colA, colB = st.columns(2)
    with colA:
        qb_csv = st.file_uploader("Upload **QB** CSV", type=["csv"], key="qbcsv")
        rb_csv = st.file_uploader("Upload **RB** CSV", type=["csv"], key="rbcsv")
        wr_csv = st.file_uploader("Upload **WR** CSV", type=["csv"], key="wrcsv")
    with colB:
        market = st.selectbox(
            "Market",
            ["Passing Yards (QB)", "Rushing Yards (RB)", "Receiving Yards (WR)"]
        )
        line_val = st.number_input("Yardage line", value=230.0, step=0.5, format="%.2f")
        opp = st.text_input("Opponent (team code, e.g., DAL). Leave blank for league-average.")
        injury_mult = st.number_input("Injury multiplier (0.0–1.2)", value=1.00, min_value=0.0, max_value=1.2, step=0.01)
        player = st.text_input("Player name (must match 'Player' exactly in your CSV)")

    if st.button("Run Player Prop Simulation"):
        try:
            if market.startswith("Passing"):
                if qb_csv is None: raise ValueError("Upload QB CSV first.")
                df = _read_player_csv(qb_csv, "QB")
                base_mean = _player_mean(df, player)
                factor = defense_factor(opp, "QB_PASS_YDS")
                sd = SD_QB_PASS_YDS
                used = df.loc[df["Player"].str.lower()==player.lower()]

            elif market.startswith("Rushing"):
                if rb_csv is None: raise ValueError("Upload RB CSV first.")
                df = _read_player_csv(rb_csv, "RB")
                base_mean = _player_mean(df, player)
                factor = defense_factor(opp, "RB_RUSH_YDS")
                sd = SD_RB_RUSH_YDS
                used = df.loc[df["Player"].str.lower()==player.lower()]

            else:  # Receiving
                if wr_csv is None: raise ValueError("Upload WR CSV first.")
                df = _read_player_csv(wr_csv, "WR")
                base_mean = _player_mean(df, player)
                factor = defense_factor(opp, "WR_REC_YDS")
                sd = SD_WR_REC_YDS
                used = df.loc[df["Player"].str.lower()==player.lower()]

            adj_mean = base_mean * factor * float(injury_mult)
            pover = _run_over_prob(adj_mean, sd, float(line_val))
            st.success(
                f"**{player} — {market}**\n\n"
                f"CSV mean: **{base_mean:.1f}** yds · "
                f"Defense factor: ×**{factor:.3f}** · "
                f"Injury: ×**{injury_mult:.2f}** → "
                f"Adjusted mean: **{adj_mean:.1f}** yds\n\n"
                f"Line: **{line_val:.1f}** → P(over) = **{pover:.1%}**, "
                f"P(under) = **{1 - pover:.1%}**"
            )

            with st.expander("Show player row used"):
                st.dataframe(used)

            with st.expander("Defense table (loaded)"):
                st.dataframe(DEF_FACTORS.reset_index())

        except Exception as e:
            st.error(str(e))
