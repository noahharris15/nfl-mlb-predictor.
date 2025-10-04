# app.py — NFL & MLB predictors + Player Props (defense-adjusted, 2025 season)

import re
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List
from io import StringIO

# ---------------------------- Third-party data libs ----------------------------
import nfl_data_py as nfl                          # NFL schedules
from pybaseball import schedule_and_record         # MLB RS/RA by team

# ---------------------------------- UI ----------------------------------------
st.set_page_config(page_title="NFL & MLB Predictors + Player Props (2025)", layout="wide")
st.title("🏈⚾ NFL & MLB Predictors + Player Props — 2025")

SIM_TRIALS = 10_000
HOME_EDGE_NFL = 0.6
EPS = 1e-9

page = st.sidebar.radio("Pages", ["NFL", "MLB", "Player Props"], index=2)

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

# --------- Team name ↔ abbr (used for parsing the pasted defense blob) --------
TEAM_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

# ---------------- DEFENSE TEXT (your pasted table – embedded) -----------------
RAW_DEF_TEXT = r"""
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

# ---- Parse that blob into pass/rush factors (lower = tougher defense) --------
def _extract_floats_near(lines: List[str], idx: int, window: int = 20) -> List[float]:
    nums = []
    for ln in lines[max(0, idx-window):idx]:
        for t in re.findall(r"[-+]?\d+\.\d+|\d+", ln):
            try:
                nums.append(float(t))
            except Exception:
                pass
    return nums

def _parse_def_blob(raw: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    out = []
    for i, ln in enumerate(lines):
        if ln in TEAM_ABBR:
            abbr = TEAM_ABBR[ln]
            floats = _extract_floats_near(lines, i, window=20)
            cand = floats[-15:] if len(floats) >= 5 else floats
            if len(cand) >= 5:
                epa_pass = cand[3]
                epa_rush  = cand[4]
            else:
                epa_pass = 0.0
                epa_rush  = 0.0
            k = 0.7
            pass_factor = float(np.clip(1.0 + k * epa_pass, 0.60, 1.40))
            rush_factor = float(np.clip(1.0 + k * epa_rush, 0.60, 1.40))
            out.append({"abbr": abbr, "pass_factor": pass_factor, "rush_factor": rush_factor, "recv_factor": pass_factor})
    return pd.DataFrame(out).drop_duplicates(subset=["abbr"])

def load_embedded_defense() -> pd.DataFrame:
    parsed = _parse_def_blob(RAW_DEF_TEXT)
    if parsed.empty:
        return pd.DataFrame({
            "abbr": list(TEAM_ABBR.values()),
            "pass_factor": 1.0, "rush_factor": 1.0, "recv_factor": 1.0
        })
    return parsed

def defense_scalers(opp_abbrev: str, def_df: pd.DataFrame) -> dict:
    row = def_df.loc[def_df["abbr"] == (opp_abbrev or "").upper()]
    if row.empty:
        return {"pass": 1.0, "rush": 1.0, "recv": 1.0}
    r = row.iloc[0]
    return {"pass": float(r["pass_factor"]), "rush": float(r["rush_factor"]), "recv": float(r["recv_factor"])}

# ----------------------- Sim helpers -------------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS) -> Tuple[float,float,float,float,float]:
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean()), float((h + a).mean())

def simulate_normal_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    samples = np.random.normal(mu, sd, size=trials)
    return float((samples > line).mean())

# -------------------------- NFL: 2025 team rates + schedule -------------------
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    date_col: Optional[str] = None
    for c in ("gameday", "game_date"):
        if c in sched.columns:
            date_col = c; break

    played = sched.dropna(subset=["home_score", "away_score"])
    home = played.rename(columns={"home_team":"team","away_team":"opp","home_score":"pf","away_score":"pa"})[["team","opp","pf","pa"]]
    away = played.rename(columns={"away_team":"team","home_team":"opp","away_score":"pf","home_score":"pa"})[["team","opp","pf","pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 22.5
        rates = pd.DataFrame({"team": list(TEAM_ABBR.values()), "PF_pg": per, "PA_pg": per})
    else:
        team = long.groupby("team", as_index=False).agg(games=("pf","size"), PF=("pf","sum"), PA=("pa","sum"))
        rates = pd.DataFrame({"team": team["team"], "PF_pg": team["PF"]/team["games"], "PA_pg": team["PA"]/team["games"]})
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"] / 4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink) * rates["PF_pg"] + shrink * prior
        rates["PA_pg"] = (1 - shrink) * rates["PA_pg"] + shrink * prior

    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][["home_team","away_team"] + ([date_col] if date_col else [])].copy()
    if date_col: upcoming = upcoming.rename(columns={date_col:"date"})
    else: upcoming["date"] = ""
    for col in ["home_team","away_team"]:
        upcoming[col] = upcoming[col].astype(str).str.replace(r"\s+"," ", regex=True)
    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
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
                RS_pg = RA_pg = 4.5; games = 0
            else:
                sar["R"] = sar["R"].astype(float); sar["RA"] = sar["RA"].astype(float)
                games = int(len(sar))
                RS_pg = float(sar["R"].sum() / games)
                RA_pg = float(sar["RA"].sum() / games)
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg, "games": games})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5, "games": 0})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean()); league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9 * df["RS_pg"] + 0.1 * league_rs
        df["RA_pg"] = 0.9 * df["RA_pg"] + 0.1 * league_ra
    return df

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

# -------------------------- CSV helpers (props) ----------------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Series(df.columns)
        .astype(str).str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    low = {c.lower(): c for c in df.columns}
    # common variants → canonical
    mapping = {
        next((low[k] for k in low if k in ("player","player name","name")), None): "Player",
        next((low[k] for k in low if k in ("team","tm","club")), None): "Team",
        next((low[k] for k in low if k in ("g","games")), None): "G",
        next((low[k] for k in low if k in ("y/g","yds/g","yards/g","yards per game")), None): "Y/G",
        next((low[k] for k in low if k in ("yds","yards","pass yds","rush yds","rec yds")), None): "Yds",
        next((low[k] for k in low if k in ("att","att.","attempts")), None): "Att",
        next((low[k] for k in low if k in ("cmp","comp","completions")), None): "Cmp",
        next((low[k] for k in low if k in ("rate","rating","passer rating","qb rating")), None): "Rate",
        next((low[k] for k in low if k in ("tgt","targets")), None): "Tgt",
        next((low[k] for k in low if k in ("rec","receptions","catches")), None): "Rec",
        next((low[k] for k in low if k in ("td","tds","touchdowns")), None): "TD",
    }
    for src, dst in mapping.items():
        if src and dst:
            df.rename(columns={src: dst}, inplace=True)
    # final fallback for Player
    if "Player" not in df.columns:
        for c in df.columns:
            nonnum_ratio = df[c].apply(lambda x: not pd.to_numeric(pd.Series([x]), errors="coerce").notna()[0]).mean()
            if nonnum_ratio > 0.8:
                df.rename(columns={c: "Player"}, inplace=True)
                break
    return df

def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("Y/G","Yds","TD","G","Att","Cmp","Rate","Tgt","Rec"):
        if c in df.columns:
            df[c] = (df[c].astype(str)
                     .str.replace(",", "", regex=False)
                     .str.replace("%", "", regex=False)
                     .str.strip())
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_any(uploaded_file) -> pd.DataFrame:
    """
    Robust reader for your CSVs that sometimes include a junk first row like:
    'Passing,Passing,Passing,...'
    We scan the first ~200 lines and start at the first line that contains 'player'.
    """
    if uploaded_file.name.lower().endswith(".csv"):
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        lines = raw.splitlines()
        header_idx = 0
        for i, ln in enumerate(lines[:200]):
            if "player" in ln.lower() and "," in ln:
                header_idx = i
                break
        cleaned = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(cleaned))
    else:
        df = pd.read_excel(uploaded_file)

    df = _norm_cols(df)
    df = _coerce_numeric_cols(df)
    return df

# ---------- Position projection builders (defense-scaled per-game) ------------
def build_qb_projection_row(qb_row: pd.Series, opp_scaler: float) -> dict:
    ypg = qb_row.get("Y/G"); td = qb_row.get("TD"); g  = qb_row.get("G")
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
    tds_mu = base_tds * (opp_scaler ** 0.7)
    yards_sd = max(8.0, 0.18 * yards_mu)
    tds_sd = max(0.25, 0.55 * tds_mu)
    return {"Player": qb_row.get("Player"), "Team": qb_row.get("Team"),
            "Adj_PassYds_mu": yards_mu, "Adj_PassYds_sd": yards_sd,
            "Adj_PassTD_mu": tds_mu, "Adj_PassTD_sd": tds_sd}

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
    tds_mu = base_tds * (opp_scaler ** 0.7)
    yards_sd = max(6.0, 0.22 * yards_mu)
    tds_sd = max(0.20, 0.65 * tds_mu)
    return {"Player": rb_row.get("Player"), "Team": rb_row.get("Team"),
            "Adj_RushYds_mu": yards_mu, "Adj_RushYds_sd": yards_sd,
            "Adj_RushTD_mu": tds_mu, "Adj_RushTD_sd": tds_sd}

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
    tds_mu = base_tds * (opp_scaler ** 0.7)
    yards_sd = max(6.0, 0.20 * yards_mu)
    tds_sd = max(0.20, 0.70 * tds_mu)
    return {"Player": wr_row.get("Player"), "Team": wr_row.get("Team"),
            "Adj_RecYds_mu": yards_mu, "Adj_RecYds_sd": yards_sd,
            "Adj_RecTD_mu": tds_mu, "Adj_RecTD_sd": tds_sd}

# ------------------------------------ NFL PAGE --------------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 Regular Season (matchups)")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build NFL rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming games found yet."); st.stop()

    if "date" in upcoming.columns and upcoming["date"].astype(str).str.len().gt(0).any():
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"] + " — " + upcoming["date"].astype(str)).tolist()
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"]).tolist()

    pick = st.selectbox("Matchup", choices, index=0)
    pair = pick.split(" — ")[0]
    home, away = pair.split(" vs ")

    try:
        mu_h, mu_a = nfl_matchup_mu(nfl_rates, home, away)
        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with c2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        with c3: st.metric(label="Exp total", value=f"{exp_t:.1f}")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
    except Exception as e:
        st.error(str(e))

# ------------------------------------ MLB PAGE --------------------------------
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
        st.dataframe(mlb_rates.sort_values("team").reset_index(drop=True), use_container_width=True, height=520)

    with right:
        st.markdown("**Pick any MLB matchup**")
        teams = mlb_rates["team"].sort_values().tolist()
        if not teams:
            st.info("No MLB team data yet."); st.stop()
        home = st.selectbox("Home team", teams, index=0, key="mlb_home")
        away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")
        try:
            mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away)
            p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
            with c2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
            with c3: st.metric(label="Exp total", value=f"{exp_t:.1f}")
            st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
        except Exception as e:
            st.error(str(e))

# ---------------------------- PLAYER PROPS PAGE ---------------------------
else:
    st.subheader("📈 Player Props (defense-adjusted)")

    # Defense factors (lower = tougher)
    def_factors = load_embedded_defense()
    opp_pick = st.selectbox("Opponent defense (abbr)", def_factors["abbr"].tolist(), index=0)
    with st.expander("Defense factors parsed from your table (lower = tougher)", expanded=False):
        st.dataframe(def_factors.sort_values("abbr").reset_index(drop=True),
                     use_container_width=True, height=260)

    # One uploader, accept multiple; auto-detect QB/RB/WR
    uploads = st.file_uploader("Drop your CSV/XLSX files (QB / RB / WR). I’ll auto-detect.",
                               type=["csv","xlsx"], accept_multiple_files=True)

    raw_qb = raw_rb = raw_wr = None
    if uploads:
        previews = []
        for f in uploads:
            try:
                df = _load_any(f)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                continue
            cols = {c.lower() for c in df.columns}
            is_qb = {"cmp","att","y/g"}.intersection(cols) or "rate" in cols
            is_wr = {"tgt","rec"}.intersection(cols)
            is_rb = ("att" in cols and "y/g" in cols) or ("rush" in " ".join(cols))
            if is_qb:
                raw_qb = df if raw_qb is None else pd.concat([raw_qb, df], ignore_index=True)
                previews.append(("QB", f.name, df.head(3)))
            elif is_wr:
                raw_wr = df if raw_wr is None else pd.concat([raw_wr, df], ignore_index=True)
                previews.append(("WR/TE", f.name, df.head(3)))
            else:
                raw_rb = df if raw_rb is None else pd.concat([raw_rb, df], ignore_index=True)
                previews.append(("RB", f.name, df.head(3)))

        with st.expander("👀 Parsed previews (top 3 rows from each file)", expanded=False):
            for pos, name, head3 in previews:
                st.caption(f"**{pos}** · {name} · columns: {list(head3.columns)}")
                st.dataframe(head3, use_container_width=True, height=120)

    # Controls strip
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col1:
        market = st.selectbox("Market", ["QB", "RB", "WR"], index=0)

    player_source = {"QB": raw_qb, "RB": raw_rb, "WR": raw_wr}[market]

    # Build player list
    players = []
    if isinstance(player_source, pd.DataFrame) and not player_source.empty:
        if "Player" not in player_source.columns:
            player_source = _norm_cols(player_source)
        if "Player" in player_source.columns:
            players = sorted(pd.Series(player_source["Player"].astype(str).dropna().unique()).tolist())

    with col2:
        player_name = st.selectbox("Player", players or [],
                                   index=0 if players else None,
                                   placeholder="Upload a CSV above first")

    with col3:
        OPP_ALIASES = {"KAN":"KC","NOR":"NO","GNB":"GB","SFO":"SF"}
        opp_input = st.text_input(
            "Opponent team code (e.g., DAL, PHI). Aliases like KAN/NOR/GNB/SFO are OK.",
            value=opp_pick
        )

    line = st.number_input("Yardage line", value=188.3, step=0.1)

    # Compute
    if st.button("Compute"):
        if not isinstance(player_source, pd.DataFrame) or player_source.empty or not players:
            st.warning("Upload a file for this market first. (Scroll up to the uploader.)")
            st.stop()

        opp_abbr = OPP_ALIASES.get((opp_input or "").strip().upper(), (opp_input or "").strip().upper())
        dscale = defense_scalers(opp_abbr, def_factors)

        row = player_source.loc[player_source["Player"].astype(str).str.lower() == str(player_name).lower()]
        if row.empty:
            st.warning("Player not found in the uploaded table. Check the preview expander to verify names.")
            st.stop()

        if market == "QB":
            proj = build_qb_projection_row(row.iloc[0], dscale["pass"])
            csv_mean = float(row.iloc[0].get("Y/G")) if pd.notna(row.iloc[0].get("Y/G")) else \
                       float(row.iloc[0].get("Yds", 0)) / max(1.0, float(row.iloc[0].get("G", 1) or 1))
            mu = proj["Adj_PassYds_mu"]; sd = proj["Adj_PassYds_sd"]
            market_label = "Passing Yards"; factor_used = dscale["pass"]
        elif market == "RB":
            proj = build_rb_projection_row(row.iloc[0], dscale["rush"])
            csv_mean = float(row.iloc[0].get("Y/G")) if pd.notna(row.iloc[0].get("Y/G")) else \
                       float(row.iloc[0].get("Yds", 0)) / max(1.0, float(row.iloc[0].get("G", 1) or 1))
            mu = proj["Adj_RushYds_mu"]; sd = proj["Adj_RushYds_sd"]
            market_label = "Rushing Yards"; factor_used = dscale["rush"]
        else:
            proj = build_wr_projection_row(row.iloc[0], dscale["recv"])
            csv_mean = float(row.iloc[0].get("Y/G")) if pd.notna(row.iloc[0].get("Y/G")) else \
                       float(row.iloc[0].get("Yds", 0)) / max(1.0, float(row.iloc[0].get("G", 1) or 1))
            mu = proj["Adj_RecYds_mu"]; sd = proj["Adj_RecYds_sd"]
            market_label = "Receiving Yards"; factor_used = dscale["recv"]

        p_over = simulate_normal_over_prob(mu, sd, line, SIM_TRIALS)
        p_under = 1.0 - p_over

        st.success(
            f"**{player_name} — {market_label}**\n\n"
            f"CSV mean: **{csv_mean:.1f}** · Defense factor (AVG): **×{factor_used:.3f}** → Adjusted mean: **{mu:.1f}**\n\n"
            f"Line: **{line:.1f}** → **P(over) = {p_over*100:.1f}%**, **P(under) = {p_under*100:.1f}%**"
        )

        with st.expander("Show player row used", expanded=False):
            st.dataframe(row.reset_index(drop=True), use_container_width=True)
