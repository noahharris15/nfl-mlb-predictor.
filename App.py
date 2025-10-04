# app.py — NFL + MLB predictors + Player Props (defense-adjusted, 2025 season)
# ✅ Defense CSV is EMBEDDED in DEFENSE_CSV_TEXT (vertical format you sent).
# ✅ No defense upload needed. NFL & MLB team sims use 2025 scoring rates.
# ✅ Player Props page uses your uploaded QB/RB/WR CSVs and adjusts by opponent defense.

import re
from io import StringIO
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------- Third-party data libs ----------------------------
import nfl_data_py as nfl                          # NFL schedule / scores
from pybaseball import schedule_and_record         # MLB team runs for/against

# ------------------------------ UI / Constants --------------------------------
st.set_page_config(page_title="NFL & MLB Predictors + Player Props (2025)", layout="wide")

SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6   # small home-field points bump in NFL sim
EPS = 1e-9

# ---------------- MLB: hard-coded team names (stable across libs) -------------
MLB_TEAMS_2025 = {
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

# ---------------- DEFENSE (your embedded vertical table) -----------------------
# Format: repeated blocks of 15 numeric lines followed by the team full name line.
# Column order per block:
#   Season, EPA/Play, Total EPA, Success %, EPA/Pass, EPA/Rush, Pass Yards,
#   Comp %, Pass TD, Rush Yards, Rush TD, ADoT, Sack %, Scramble %, Int %, Team Name
DEFENSE_CSV_TEXT = """\
Team
Season
EPA/Play
Total EPA
Success %
EPA/Pass
EPA/Rush
Pass Yards
Comp %
Pass TD
Rush Yards
Rush TD
ADoT
Sack %
Scramble %
Int %

2025.0
-0.17
-38.71
0.4174
-0.37
0.06
685.0
0.6762
3.0
521.0
4.0
5.8
0.0813
0.065
0.0163
Minnesota Vikings

2025.0
-0.13
-33.41
0.379
-0.17
-0.05
984.0
0.5962
7.0
331.0
1.0
8.15
0.0409
0.0468
0.0526
Jacksonville Jaguars

2025.0
-0.11
-26.37
0.3796
-0.1
-0.12
853.0
0.5746
2.0
397.0
2.0
9.94
0.0974
0.0325
0.0065
Denver Broncos

2025.0
-0.11
-25.57
0.3667
-0.17
0.01
710.0
0.5938
3.0
445.0
3.0
7.69
0.0823
0.1076
0.019
Los Angeles Chargers

2025.0
-0.09
-21.51
0.3783
0.0
-0.22
894.0
0.6271
7.0
376.0
4.0
10.44
0.1
0.0571
0.0214
Detroit Lions

2025.0
-0.08
-20.49
0.4291
-0.11
-0.04
860.0
0.5693
5.0
504.0
3.0
8.43
0.0329
0.0658
0.0197
Philadelphia Eagles

2025.0
-0.08
-19.59
0.4149
-0.16
0.04
790.0
0.5714
3.0
409.0
4.0
8.3
0.0733
0.04
0.0133
Houston Texans

2025.0
-0.08
-18.57
0.4042
-0.12
0.0
851.0
0.664
5.0
394.0
2.0
8.12
0.0952
0.0544
0.0204
Los Angeles Rams

2025.0
-0.07
-18.65
0.4075
0.0
-0.19
910.0
0.6645
6.0
359.0
0.0
7.03
0.069
0.0575
0.0402
Seattle Seahawks

2025.0
-0.06
-14.94
0.4344
-0.09
-0.03
690.0
0.6829
5.0
462.0
2.0
7.02
0.0368
0.0588
0.0
San Francisco 49ers

2025.0
-0.06
-13.91
0.4202
-0.02
-0.11
832.0
0.6429
6.0
340.0
3.0
6.54
0.0645
0.1226
0.0065
Tampa Bay Buccaneers

2025.0
-0.05
-11.24
0.4279
-0.13
0.05
602.0
0.5769
5.0
436.0
2.0
10.08
0.0806
0.0806
0.0242
Atlanta Falcons

2025.0
-0.05
-10.73
0.379
0.06
-0.17
689.0
0.6442
8.0
281.0
2.0
7.95
0.0924
0.0336
0.0168
Cleveland Browns

2025.0
-0.05
-11.02
0.4638
-0.04
-0.05
946.0
0.6643
8.0
384.0
2.0
6.69
0.0633
0.0506
0.0253
Indianapolis Colts

2025.0
-0.02
-3.64
0.4487
-0.09
0.09
778.0
0.6694
4.0
508.0
4.0
8.27
0.0694
0.0903
0.0208
Kansas City Chiefs

2025.0
-0.01
-2.38
0.4559
0.06
-0.14
1068.0
0.6369
5.0
384.0
2.0
8.06
0.0444
0.0222
0.0111
Arizona Cardinals

2025.0
-0.01
-1.85
0.4315
0.14
-0.22
948.0
0.6565
5.0
411.0
4.0
7.98
0.0544
0.0544
0.0136
Las Vegas Raiders

2025.0
0.0
0.2
0.4094
0.03
-0.07
886.0
0.6815
6.0
310.0
3.0
6.87
0.0632
0.0345
0.0115
Green Bay Packers

2025.0
0.0
0.89
0.4912
0.01
0.0
886.0
0.7368
10.0
658.0
4.0
6.75
0.0407
0.0325
0.0569
Chicago Bears

2025.0
0.02
3.33
0.4208
-0.06
0.1
564.0
0.6214
6.0
657.0
5.0
6.87
0.0732
0.0894
0.0163
Buffalo Bills

2025.0
0.04
9.13
0.4133
0.03
0.05
802.0
0.6239
4.0
517.0
5.0
7.5
0.0155
0.0775
0.031
Carolina Panthers

2025.0
0.04
11.09
0.461
0.11
-0.05
1131.0
0.6957
7.0
488.0
4.0
7.6
0.087
0.0559
0.0311
Pittsburgh Steelers

2025.0
0.04
10.44
0.4183
0.18
-0.12
1062.0
0.6098
7.0
430.0
3.0
10.83
0.0714
0.05
0.0071
Washington Commanders

2025.0
0.05
12.43
0.4693
0.19
-0.15
1024.0
0.712
7.0
310.0
2.0
7.68
0.0725
0.0217
0.0217
New England Patriots

2025.0
0.07
18.22
0.4613
-0.01
0.19
1021.0
0.6375
5.0
612.0
6.0
7.88
0.0562
0.0449
0.0169
New York Giants

2025.0
0.07
17.94
0.4417
0.2
-0.06
884.0
0.7117
9.0
475.0
4.0
7.4
0.0853
0.0543
0.0078
New Orleans Saints

2025.0
0.1
27.12
0.4731
0.13
0.04
1089.0
0.6536
8.0
543.0
5.0
6.99
0.0366
0.0305
0.0305
Cincinnati Bengals

2025.0
0.11
25.77
0.3959
0.23
-0.03
834.0
0.6577
7.0
522.0
4.0
6.11
0.0476
0.0714
0.0
New York Jets

2025.0
0.12
30.47
0.4435
0.16
0.07
935.0
0.6984
6.0
566.0
7.0
6.82
0.0294
0.0441
0.0221
Tennessee Titans

2025.0
0.14
39.54
0.4685
0.14
0.12
1084.0
0.6667
9.0
565.0
7.0
8.04
0.0233
0.0523
0.0058
Baltimore Ravens

2025.0
0.25
65.26
0.4943
0.4
0.06
1237.0
0.7333
10.0
493.0
6.0
9.19
0.034
0.0476
0.0068
Dallas Cowboys

2025.0
0.25
59.66
0.5397
0.34
0.12
941.0
0.7757
7.0
632.0
5.0
6.15
0.0615
0.1154
0.0
Miami Dolphins
"""

# ---------------------- Team name → abbreviation mapping ----------------------
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

# --------------------- Parse your vertical defense table ----------------------
def parse_vertical_defense(text: str) -> pd.DataFrame:
    """
    The provided table is vertical: 15 numeric lines then a team name, repeated.
    We parse blocks into rows with columns in the known order.
    """
    if not text or not text.strip():
        return pd.DataFrame()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Remove header words-only lines that appear first
    header_words = set([
        "team","season","epa/play","total epa","success %","epa/pass","epa/rush",
        "pass yards","comp %","pass td","rush yards","rush td","adot","sack %","scramble %","int %"
    ])
    cleaned = []
    for ln in lines:
        if ln.lower() in header_words:
            continue
        cleaned.append(ln)
    lines = cleaned

    rows = []
    buf = []
    for ln in lines:
        # number or decimal like -0.17
        if re.fullmatch(r"-?\d+(\.\d+)?", ln):
            buf.append(float(ln))
        else:
            # non-numeric: assume team name ends block
            team = ln
            if len(buf) >= 15:
                # Map the first 15 numbers to the columns
                (season, epa_play, total_epa, success, epa_pass, epa_rush,
                 pass_yards, comp_pct, pass_td, rush_yards, rush_td,
                 adot, sack_pct, scramble_pct, int_pct) = buf[:15]
                rows.append({
                    "team_name": team,
                    "season": season,
                    "epa_play": epa_play,
                    "epa_pass": epa_pass,
                    "epa_rush": epa_rush,
                })
            buf = []

    df = pd.DataFrame(rows)
    # attach abbreviations
    df["abbr"] = df["team_name"].map(TEAM_NAME_TO_ABBR)
    df = df.dropna(subset=["abbr"]).reset_index(drop=True)
    return df

def build_defense_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert EPA metrics (lower = better defense) into multiplicative factors.
    We compute z-scores and clamp to ~0.85..1.15 range.
    """
    if df.empty:
        return pd.DataFrame()

    def _scale(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        mu, sd = float(s.mean()), float(s.std(ddof=0) or 1.0)
        z = (s - mu) / (sd if sd > 1e-9 else 1.0)
        # Lower EPA (better D) -> smaller factor
        return 1.0 + np.clip(z, -2.0, 2.0) * 0.075

    out = df.copy()
    out["factor_all"]  = _scale(out["epa_play"])
    # If pass/rush present, use them; else fall back to all
    out["factor_pass"] = _scale(out["epa_pass"]) if "epa_pass" in out.columns and out["epa_pass"].notna().any() else out["factor_all"]
    out["factor_rush"] = _scale(out["epa_rush"]) if "epa_rush" in out.columns and out["epa_rush"].notna().any() else out["factor_all"]

    # Clamp just in case
    for c in ["factor_all","factor_pass","factor_rush"]:
        out[c] = out[c].clip(0.80, 1.20)
    return out[["abbr","team_name","factor_all","factor_pass","factor_rush"]]

@st.cache_data(show_spinner=False)
def load_embedded_defense() -> pd.DataFrame:
    raw = parse_vertical_defense(DEFENSE_CSV_TEXT)
    if raw.empty:
        return pd.DataFrame(columns=["abbr","team_name","factor_all","factor_pass","factor_rush"])
    return build_defense_factors(raw)

def defense_scalers(opp_abbrev: str, def_factors: pd.DataFrame) -> Dict[str, float]:
    row = def_factors.loc[def_factors["abbr"].str.upper() == opp_abbrev.upper()]
    if row.empty:
        return {"pass": 1.0, "rush": 1.0, "recv": 1.0}
    r = row.iloc[0]
    # Use pass vs. rush; receiving uses pass factor as a proxy
    return {"pass": float(r["factor_pass"]), "rush": float(r["factor_rush"]), "recv": float(r["factor_pass"])}

# ----------------------- Small sim helpers (Poisson/Normal) -------------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # slight home tiebreak
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean()), float((h + a).mean())

def simulate_normal_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS):
    sd = max(1e-6, float(sd))
    samples = np.random.normal(mu, sd, size=trials)
    return float((samples > line).mean())

# -------------------------- NFL: 2025 team rates + schedule -------------------
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    # normalize date col naming
    date_col: Optional[str] = None
    for c in ("gameday", "game_date"):
        if c in sched.columns:
            date_col = c
            break

    played = sched.dropna(subset=["home_score", "away_score"])

    home = played.rename(columns={
        "home_team": "team", "away_team": "opp",
        "home_score": "pf", "away_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    away = played.rename(columns={
        "away_team": "team", "home_team": "opp",
        "away_score": "pf", "home_score": "pa"
    })[["team", "opp", "pf", "pa"]]
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
        team = long.groupby("team", as_index=False).agg(
            games=("pf", "size"), PF=("pf", "sum"), PA=("pa", "sum")
        )
        rates = pd.DataFrame({
            "team": team["team"],
            "PF_pg": team["PF"] / team["games"],
            "PA_pg": team["PA"] / team["games"],
        })
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"] / 4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink) * rates["PF_pg"] + shrink * prior
        rates["PA_pg"] = (1 - shrink) * rates["PA_pg"] + shrink * prior

    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][
        ["home_team", "away_team"] + ([date_col] if date_col else [])
    ].copy()
    if date_col:
        upcoming = upcoming.rename(columns={date_col: "date"})
    else:
        upcoming["date"] = ""

    for col in ["home_team", "away_team"]:
        upcoming[col] = upcoming[col].astype(str).str.replace(r"\s+", " ", regex=True)

    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"]) / 2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"]) / 2.0)
    return mu_home, mu_away

# -------------------------- MLB: 2025 team RS/RA per game ---------------------
@st.cache_data(show_spinner=False)
def mlb_team_rates_2025():
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

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)  # neutral HFA in MLB
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

# -------------------------- CSV cleaning helpers (props) ----------------------
def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_any_csv(uploaded_file) -> pd.DataFrame:
    """Robust loader for pasted CSVs that include extra header lines."""
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    # Find the header row that starts with 'Rk,Player,...' (common FBref export)
    lines = raw.strip().splitlines()
    header_idx = 0
    for i, ln in enumerate(lines[:10]):
        if ln.startswith("Rk,Player"):
            header_idx = i
            break
    cleaned = "\n".join(lines[header_idx:]) if lines else raw
    try:
        return pd.read_csv(StringIO(cleaned))
    except Exception:
        return pd.read_csv(StringIO(raw))  # last resort

def build_qb_projection_row(qb_row: pd.Series, opp_scaler: float) -> dict:
    ypg = qb_row.get("Y/G"); td = qb_row.get("TD"); g = qb_row.get("G")
    try:
        base_yards = float(ypg)
    except Exception:
        try:
            base_yards = float(qb_row.get("Yds")) / max(1.0, float(g))
        except Exception:
            base_yards = 225.0
    try:
        base_tds = float(td) / max(1.0, float(g))
    except Exception:
        base_tds = 1.5
    yards_mu = base_yards * float(opp_scaler)
    tds_mu   = base_tds   * (opp_scaler ** 0.7)
    yards_sd = max(8.0, 0.18 * yards_mu)
    tds_sd   = max(0.25, 0.55 * tds_mu)
    return {
        "Player": qb_row.get("Player"), "Team": qb_row.get("Team"),
        "Adj_PassYds_mu": yards_mu, "Adj_PassYds_sd": yards_sd,
        "Adj_PassTD_mu": tds_mu, "Adj_PassTD_sd": tds_sd
    }

def build_rb_projection_row(rb_row: pd.Series, opp_scaler: float) -> dict:
    ypg = rb_row.get("Y/G"); td = rb_row.get("TD"); g = rb_row.get("G")
    try:
        base_yards = float(ypg)
    except Exception:
        try:
            base_yards = float(rb_row.get("Yds")) / max(1.0, float(g))
        except Exception:
            base_yards = 60.0
    try:
        base_tds = float(td) / max(1.0, float(g))
    except Exception:
        base_tds = 0.5
    yards_mu = base_yards * float(opp_scaler)
    tds_mu   = base_tds   * (opp_scaler ** 0.7)
    yards_sd = max(6.0, 0.22 * yards_mu)
    tds_sd   = max(0.20, 0.65 * tds_mu)
    return {
        "Player": rb_row.get("Player"), "Team": rb_row.get("Team"),
        "Adj_RushYds_mu": yards_mu, "Adj_RushYds_sd": yards_sd,
        "Adj_RushTD_mu": tds_mu, "Adj_RushTD_sd": tds_sd
    }

def build_wr_projection_row(wr_row: pd.Series, opp_scaler: float) -> dict:
    ypg = wr_row.get("Y/G"); td = wr_row.get("TD"); g = wr_row.get("G")
    try:
        base_yards = float(ypg)
    except Exception:
        try:
            base_yards = float(wr_row.get("Yds")) / max(1.0, float(g))
        except Exception:
            base_yards = 55.0
    try:
        base_tds = float(td) / max(1.0, float(g))
    except Exception:
        base_tds = 0.4
    yards_mu = base_yards * float(opp_scaler)
    tds_mu   = base_tds   * (opp_scaler ** 0.7)
    yards_sd = max(6.0, 0.20 * yards_mu)
    tds_sd   = max(0.20, 0.70 * tds_mu)
    return {
        "Player": wr_row.get("Player"), "Team": wr_row.get("Team"),
        "Adj_RecYds_mu": yards_mu, "Adj_RecYds_sd": yards_sd,
        "Adj_RecTD_mu": tds_mu, "Adj_RecTD_sd": tds_sd
    }

# ----------------------------------- UI ---------------------------------------
st.title("🏈⚾ NFL & MLB Predictors + Player Props — 2025")
st.caption("NFL & MLB team matchups (team scoring rates only) + a Player Props page that ingests your QB/RB/WR CSVs and adjusts by **embedded** opponent defense.")

page = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# ------------------------------------ NFL -------------------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 Regular Season (matchups)")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build NFL rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming games found yet.")
        st.stop()

    if "date" in upcoming.columns and upcoming["date"].astype(str).str.len().gt(0).any():
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"] +
                   " — " + upcoming["date"].astype(str)).tolist()
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"]).tolist()

    pick = st.selectbox("Matchup", choices, index=0)
    pair = pick.split(" — ")[0]
    home, away = pair.split(" vs ")

    try:
        mu_h, mu_a = nfl_matchup_mu(nfl_rates, home, away)
        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with col2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        with col3: st.metric(label="Exp total", value=f"{exp_t:.1f}")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
    except Exception as e:
        st.error(str(e))

# ------------------------------------ MLB -------------------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 season (team RS/RA)")
    try:
        mlb_rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't load MLB data: {e}")
        st.stop()

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("**Team scoring rates (RS/RA per game)**")
        st.dataframe(
            mlb_rates.sort_values("team").reset_index(drop=True),
            use_container_width=True,
            height=520
        )

    with right:
        st.markdown("**Pick any MLB matchup**")
        teams = mlb_rates["team"].sort_values().tolist()
        if not teams:
            st.info("No MLB team data yet.")
            st.stop()

        home = st.selectbox("Home team", teams, index=0, key="mlb_home")
        away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")

        try:
            mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away)
            p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
            with col2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
            with col3: st.metric(label="Exp total", value=f"{exp_t:.1f}")
            st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
        except Exception as e:
            st.error(str(e))

# -------------------------------- Player Props --------------------------------
else:
    st.subheader("📈 Player Props (upload your QB / RB / WR CSVs)")
    st.caption("We parse your CSVs, adjust by **embedded defense** (EPA-based factors), and give quick O/U sims + Excel export.")

    # Load embedded defense factors (from your vertical table)
    def_factors = load_embedded_defense()
    if def_factors.empty:
        st.error("Embedded defense table failed to parse. (Ping me and I’ll tweak the parser.)")
        st.stop()

    opp = st.selectbox("Opponent (defense to adjust for)", def_factors["abbr"].tolist(), index=0)
    scalers = defense_scalers(opp, def_factors)

    with st.expander("Defense factors in use"):
        st.dataframe(def_factors.sort_values("abbr").reset_index(drop=True), use_container_width=True)

    colA, colB, colC = st.columns(3)
    with colA:
        qb_file = st.file_uploader("Upload **QB CSV**", type=["csv"], key="qb_csv")
    with colB:
        rb_file = st.file_uploader("Upload **RB CSV**", type=["csv"], key="rb_csv")
    with colC:
        wr_file = st.file_uploader("Upload **WR CSV**", type=["csv"], key="wr_csv")

    qb_table = rb_table = wr_table = None
    proj_qb = proj_rb = proj_wr = None

    # -------------------- QBs --------------------
    if qb_file is not None:
        try:
            qb_table = _load_any_csv(qb_file)
            qb_table = _coerce_numeric(qb_table, ["Y/G","Yds","TD","G","Att","Cmp","Rate"])
            st.markdown("**Parsed QB table**")
            st.dataframe(qb_table, use_container_width=True, height=260)

            rows = []
            for _, r in qb_table.iterrows():
                rows.append(build_qb_projection_row(r, scalers["pass"]))
            proj_qb = pd.DataFrame(rows)
            st.markdown("**QB projections (defense-adjusted)**")
            st.dataframe(proj_qb, use_container_width=True)
        except Exception as e:
            st.error(f"QB CSV error: {e}")

    # -------------------- RBs --------------------
    if rb_file is not None:
        try:
            rb_table = _load_any_csv(rb_file)
            rb_table = _coerce_numeric(rb_table, ["Y/G","Yds","TD","G","Att"])
            st.markdown("**Parsed RB table**")
            st.dataframe(rb_table, use_container_width=True, height=260)

            rows = []
            for _, r in rb_table.iterrows():
                rows.append(build_rb_projection_row(r, scalers["rush"]))
            proj_rb = pd.DataFrame(rows)
            st.markdown("**RB projections (defense-adjusted)**")
            st.dataframe(proj_rb, use_container_width=True)
        except Exception as e:
            st.error(f"RB CSV error: {e}")

    # -------------------- WRs --------------------
    if wr_file is not None:
        try:
            wr_table = _load_any_csv(wr_file)
            wr_table = _coerce_numeric(wr_table, ["Y/G","Yds","TD","G","Tgt","Rec"])
            st.markdown("**Parsed WR table**")
            st.dataframe(wr_table, use_container_width=True, height=260)

            rows = []
            for _, r in wr_table.iterrows():
                rows.append(build_wr_projection_row(r, scalers["recv"]))
            proj_wr = pd.DataFrame(rows)
            st.markdown("**WR projections (defense-adjusted)**")
            st.dataframe(proj_wr, use_container_width=True)
        except Exception as e:
            st.error(f"WR CSV error: {e}")

    # ---------------- Optional quick O/U calculator ----------------
    st.markdown("---")
    st.markdown("### Quick Over/Under probability (Normal model)")
    colx, coly, colz = st.columns(3)
    with colx:
        which = st.selectbox("Market", ["QB Passing Yards","QB Passing TDs","RB Rushing Yards","WR Receiving Yards"])
    with coly:
        line = st.number_input("Your line (e.g., 249.5)", value=249.5, step=0.5)
    with colz:
        if which in ["QB Passing Yards","RB Rushing Yards","WR Receiving Yards"]:
            sd_default = 35.0 if "QB" in which else 22.0
        else:
            sd_default = 0.9
        sd_user = st.number_input("Std Dev override (optional)", value=sd_default, step=0.1)

    player_name = st.text_input("Player (must exist in the parsed table above)", value="")
    calc_btn = st.button("Compute Over%")

    if calc_btn and player_name.strip():
        source_df = None
        mu_col = None
        if which == "QB Passing Yards" and proj_qb is not None:
            source_df = proj_qb; mu_col = "Adj_PassYds_mu"
        elif which == "QB Passing TDs" and proj_qb is not None:
            source_df = proj_qb; mu_col = "Adj_PassTD_mu"
        elif which == "RB Rushing Yards" and proj_rb is not None:
            source_df = proj_rb; mu_col = "Adj_RushYds_mu"
        elif which == "WR Receiving Yards" and proj_wr is not None:
            source_df = proj_wr; mu_col = "Adj_RecYds_mu"

        if source_df is None:
            st.warning("Upload the matching CSV first (and ensure projections show above).")
        else:
            row = source_df.loc[source_df["Player"].astype(str).str.lower() == player_name.lower()]
            if row.empty:
                st.warning("Player not found in the projection table.")
            else:
                mu_val = float(row.iloc[0][mu_col])
                over_p = simulate_normal_over_prob(mu_val, sd_user, line, SIM_TRIALS)
                st.metric("Over probability", f"{over_p*100:.1f}%")
                st.caption(f"μ={mu_val:.1f}, σ={sd_user:.1f}, line={line}")

    # -------------------- Excel Download (all combined) ------------------------
    if any(x is not None for x in [qb_table, rb_table, wr_table]):
        with pd.ExcelWriter("player_props_adjusted.xlsx", engine="xlsxwriter") as writer:
            if qb_table is not None:
                qb_table.to_excel(writer, index=False, sheet_name="QB_raw")
            if proj_qb is not None:
                proj_qb.to_excel(writer, index=False, sheet_name="QB_proj")
            if rb_table is not None:
                rb_table.to_excel(writer, index=False, sheet_name="RB_raw")
            if proj_rb is not None:
                proj_rb.to_excel(writer, index=False, sheet_name="RB_proj")
            if wr_table is not None:
                wr_table.to_excel(writer, index=False, sheet_name="WR_raw")
            if proj_wr is not None:
                proj_wr.to_excel(writer, index=False, sheet_name="WR_proj")
        with open("player_props_adjusted.xlsx", "rb") as f:
            st.download_button(
                label="⬇️ Download Excel (QB/RB/WR raw + projections)",
                data=f,
                file_name="player_props_adjusted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
