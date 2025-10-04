# app.py — NFL + MLB predictors + Player Props (defense-adjusted, 2025 season)

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional
from io import StringIO

# ---------------------------- Third-party data libs ----------------------------
# NFL schedules (unchanged from your working version idea)
import nfl_data_py as nfl

# MLB team schedule-and-record (team RS/RA) — keep if your current build works
from pybaseball import schedule_and_record

# ------------------------------ UI / Constants --------------------------------
st.set_page_config(page_title="NFL & MLB Predictors + Player Props (2025)", layout="wide")

SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6   # small HFA point bump in NFL sim
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

# ---------------- DEFENSE STRENGTH (embedded; no external files) --------------
# Baselines to scale by (league-ish)
LEAGUE_PASS_YPA = 6.9
LEAGUE_RUSH_YPA = 4.3
LEAGUE_REC_YPT  = 7.6

_DEFENSE_CSV = """Team,pass_ypa_allowed,rush_ypa_allowed,rec_ypt_allowed
ARI,6.9,4.3,7.6
ATL,6.9,4.3,7.6
BAL,6.9,4.3,7.6
BUF,6.9,4.3,7.6
CAR,6.9,4.3,7.6
CHI,6.9,4.3,7.6
CIN,6.9,4.3,7.6
CLE,6.9,4.3,7.6
DAL,6.9,4.3,7.6
DEN,6.9,4.3,7.6
DET,6.9,4.3,7.6
GB,6.9,4.3,7.6
HOU,6.9,4.3,7.6
IND,6.9,4.3,7.6
JAX,6.9,4.3,7.6
KC,6.9,4.3,7.6
LAC,6.9,4.3,7.6
LAR,6.9,4.3,7.6
LV,6.9,4.3,7.6
MIA,6.9,4.3,7.6
MIN,6.9,4.3,7.6
NE,6.9,4.3,7.6
NO,6.9,4.3,7.6
NYG,6.9,4.3,7.6
NYJ,6.9,4.3,7.6
PHI,6.9,4.3,7.6
PIT,6.9,4.3,7.6
SEA,6.9,4.3,7.6
SF,6.9,4.3,7.6
TB,6.9,4.3,7.6
TEN,6.9,4.3,7.6
WAS,6.9,4.3,7.6
"""

def load_embedded_defense() -> pd.DataFrame:
    df = pd.read_csv(StringIO(_DEFENSE_CSV))
    for c, base in {"pass_ypa_allowed": LEAGUE_PASS_YPA,
                    "rush_ypa_allowed": LEAGUE_RUSH_YPA,
                    "rec_ypt_allowed":  LEAGUE_REC_YPT}.items():
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(base).clip(lower=0.1)
    df["Team"] = df["Team"].str.upper()
    return df

def defense_scalers(opp_abbrev: str, def_df: pd.DataFrame) -> dict:
    row = def_df.loc[def_df["Team"] == opp_abbrev.upper()]
    if row.empty:
        return {"pass": 1.0, "rush": 1.0, "recv": 1.0}
    r = row.iloc[0]
    return {
        "pass": float(r["pass_ypa_allowed"] / LEAGUE_PASS_YPA),
        "rush": float(r["rush_ypa_allowed"] / LEAGUE_RUSH_YPA),
        "recv": float(r["rec_ypt_allowed"]  / LEAGUE_REC_YPT),
    }

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
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)  # neutral HFA in baseball
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

# -------------------------- CSV cleaning helpers (props) ----------------------
def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.replace("%", "")
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_any_csv(uploaded_file) -> pd.DataFrame:
    """Robust loader for your pasted CSVs that include extra header lines."""
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    # Find the header row that starts with 'Rk,Player,...'
    lines = raw.strip().splitlines()
    header_idx = 0
    for i, ln in enumerate(lines[:10]):  # search early lines
        if ln.startswith("Rk,Player"):
            header_idx = i
            break
    cleaned = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(cleaned))
    return df

def build_qb_projection_row(qb_row: pd.Series, opp_scaler: float) -> dict:
    # Expect columns: Player, Team, G, Yds, Y/G, TD, Att, Cmp, Rate ... (your export has them)
    ypg = qb_row.get("Y/G")
    td = qb_row.get("TD")
    g  = qb_row.get("G")
    att = qb_row.get("Att")
    # Base per-game estimates
    try:
        base_yards = float(ypg)
    except Exception:
        # fallback from totals if needed
        try:
            base_yards = float(qb_row.get("Yds")) / max(1.0, float(g))
        except Exception:
            base_yards = 225.0
    try:
        base_tds = float(td) / max(1.0, float(g))
    except Exception:
        base_tds = 1.5
    # Defense adjust (pass)
    yards_mu = base_yards * float(opp_scaler)
    tds_mu = base_tds * (opp_scaler ** 0.7)  # soften TD scaling a bit
    # crude variance: 18% of mean
    yards_sd = max(8.0, 0.18 * yards_mu)
    tds_sd = max(0.25, 0.55 * tds_mu)
    return {
        "Player": qb_row.get("Player"), "Team": qb_row.get("Team"),
        "Adj_PassYds_mu": yards_mu, "Adj_PassYds_sd": yards_sd,
        "Adj_PassTD_mu": tds_mu, "Adj_PassTD_sd": tds_sd
    }

def build_rb_projection_row(rb_row: pd.Series, opp_scaler: float) -> dict:
    # Expect: Player, Team, G, Yds, Y/G, TD, Att
    ypg = rb_row.get("Y/G")
    td = rb_row.get("TD"); g = rb_row.get("G")
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
    return {
        "Player": rb_row.get("Player"), "Team": rb_row.get("Team"),
        "Adj_RushYds_mu": yards_mu, "Adj_RushYds_sd": yards_sd,
        "Adj_RushTD_mu": tds_mu, "Adj_RushTD_sd": tds_sd
    }

def build_wr_projection_row(wr_row: pd.Series, opp_scaler: float) -> dict:
    # Expect: Player, Team, G, Yds, Y/G, TD, Tgt, Rec
    ypg = wr_row.get("Y/G")
    td = wr_row.get("TD"); g = wr_row.get("G")
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
    return {
        "Player": wr_row.get("Player"), "Team": wr_row.get("Team"),
        "Adj_RecYds_mu": yards_mu, "Adj_RecYds_sd": yards_sd,
        "Adj_RecTD_mu": tds_mu, "Adj_RecTD_sd": tds_sd
    }

# ----------------------------------- UI ---------------------------------------
st.title("🏈⚾ NFL & MLB Predictors + Player Props — 2025")
st.caption("NFL & MLB team matchups (team scoring rates only) + a Player Props page that ingests your QB/RB/WR CSVs and adjusts by opponent defense strength (embedded).")

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
    st.caption("We’ll parse your CSVs, adjust by **opponent defense** (baked in), and let you download a combined Excel.")

    def_df = load_embedded_defense()
    opp = st.selectbox("Opponent (defense to adjust for)", def_df["Team"].tolist(), index=0)
    scalers = defense_scalers(opp, def_df)

    with st.expander("Opponent defense used (lower = tougher)"):
        st.dataframe(def_df.loc[def_df["Team"] == opp].reset_index(drop=True), use_container_width=True)

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
