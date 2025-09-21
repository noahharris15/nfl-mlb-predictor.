# app.py — NFL + MLB predictor (2025 only, stats-only)
# Adds Player Props page that ingests your QB/RB/WR CSVs plus a Defense CSV
# and auto-adjusts projections by opponent defense (EPA/pass or EPA/rush).
# No team dropdown on props page; optional Opponent text box only.

import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from typing import Optional, Tuple, Dict

# ---- NFL (nfl_data_py) -------------------------------------------------------
import nfl_data_py as nfl

# ---- MLB (pybaseball + statsapi) ---------------------------------------------
from pybaseball import schedule_and_record
try:
    import statsapi  # probables + pitcher stats (ERA)
    HAS_STATSAPI = True
except Exception:
    HAS_STATSAPI = False

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6   # small HFA in points
EPS = 1e-9

# Default per-prop simulation SDs (kept basic per your request)
SD_QB_PASS_YDS = 65.0
SD_RB_RUSH_YDS = 25.0
SD_WR_REC_YDS  = 35.0

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

# -------------------------- generic -------------------------------------------
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

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups (unchanged)
# ==============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
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
# MLB (2025) — team RS/RA + probables ERA (unchanged behavior)
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
                RS_pg = float(sar["R"].sum() / games) if games else 4.5
                RA_pg = float(sar["RA"].sum() / games) if games else 4.5
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

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float,float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

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

# ==============================================================================
# Player Props — CSVs + Defense EPA adjustment (NEW)
# ==============================================================================
def _read_csv(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        try:
            # Some sheets export with semicolons or tabs
            return pd.read_csv(uploaded, sep=None, engine="python")
        except Exception:
            return None

def _normalize_def_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts your EPA defense table; normalizes column names."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["TEAM","EPA_PASS","EPA_RUSH","EPA_PLAY"])
    cols = {c: c.strip().upper() for c in df.columns}
    df = df.rename(columns=cols)
    # common aliases
    rename_map = {
        "EPA/PASS": "EPA_PASS",
        "EPA/RUSH": "EPA_RUSH",
        "EPA/PLAY": "EPA_PLAY",
        "TEAM": "TEAM",
        "DEFENSE": "TEAM",
    }
    for k,v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k:v})
    # keep only needed
    keep = [c for c in ["TEAM","EPA_PASS","EPA_RUSH","EPA_PLAY"] if c in df.columns]
    df = df[keep].copy()
    # team codes: uppercase and trim
    df["TEAM"] = df["TEAM"].astype(str).str.strip()
    return df

def _def_factor(def_df: pd.DataFrame, opp_code: Optional[str], pos_group: str) -> float:
    """
    Convert defense EPA to a simple multiplier around 1.0.
    - pos_group: "QB" or "WR" => EPA_PASS; "RB" => EPA_RUSH
    - If opp_code missing or not found => 1.0 (league average)
    """
    if def_df is None or def_df.empty or not opp_code:
        return 1.0
    opp = str(opp_code).strip().upper()
    row = def_df.loc[def_df["TEAM"].str.upper() == opp]
    if row.empty:
        return 1.0

    col = "EPA_PASS" if pos_group in ("QB","WR") else "EPA_RUSH"
    if col not in def_df.columns:
        return 1.0

    val = float(row.iloc[0][col])
    # league mean & std
    mean = float(def_df[col].mean())
    std = float(def_df[col].std(ddof=0)) if def_df[col].std(ddof=0) > 0 else 0.05

    # Negative EPA is good defense. We invert linearly:
    # z < 0 (better defense) -> factor < 1 ; z > 0 (worse defense) -> factor > 1
    z = (val - mean) / std
    factor = 1.0 - 0.25 * z   # gentle scaling
    factor = float(np.clip(factor, 0.7, 1.3))
    return factor

def _pick_player_row(df: pd.DataFrame, player_name: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    mask = df["Player"].astype(str).str.lower() == str(player_name).lower()
    if not mask.any():
        # try contains
        mask = df["Player"].astype(str).str.lower().str.contains(str(player_name).lower())
    if not mask.any():
        return None
    return df.loc[mask].iloc[0]

def _yards_per_game(row: pd.Series, pos_group: str) -> Optional[float]:
    # Your CSVs include Y/G for QBs (passing), RBs (rushing), WRs (receiving)
    for col in ["Y/G","Yds/G","YPG","Yds per G"]:
        if col in row.index:
            try:
                return float(row[col])
            except Exception:
                continue
    # fallback: total / games if present
    total_col = {"QB":"Yds", "RB":"Yds", "WR":"Yds"}.get(pos_group, "Yds")
    if total_col in row.index and "G" in row.index:
        try:
            return float(row[total_col]) / max(1.0, float(row["G"]))
        except Exception:
            return None
    return None

def _simulate_yards(mean_yards: float, line: float, sd: float) -> float:
    x = np.random.normal(loc=mean_yards, scale=sd, size=SIM_TRIALS)
    return float((x >= float(line)).mean())

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "NFL/MLB game pages use team scoring rates only. "
    "Player Props page uses your CSVs and adjusts by defense EPA (no team dropdown needed)."
)

page = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page (unchanged) ------------------------------
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
                   " — " + upcoming["date"].astype(str).fillna(""))
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"])

    game = st.selectbox("Choose matchup", choices)
    home = upcoming.iloc[choices.get_indexer([game])[0]]["home_team"]
    away = upcoming.iloc[choices.get_indexer([game])[0]]["away_team"]

    muH, muA = nfl_matchup_mu(nfl_rates, home, away)
    pH, pA, mH, mA, tot = simulate_poisson_game(muH, muA)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win %", f"{pH*100:.1f}%")
    c2.metric(f"{away} win %", f"{pA*100:.1f}%")
    c3.metric("Expected total (pts)", f"{tot:.1f}")

# -------------------------- MLB page (unchanged) ------------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 Season (team RS/RA + probables ERA)")
    rates = mlb_team_rates_2025()
    prob = _today_probables()

    st.write("**Today’s Probables (with ERA if available):**")
    if prob is None or prob.empty:
        st.info("No probables found (or statsapi not available).")
    else:
        st.dataframe(prob, use_container_width=True)

    st.write("**Team scoring rates:**")
    st.dataframe(rates.sort_values("team"), use_container_width=True)

    # Quick sim widget (unchanged behavior)
    home_team = st.selectbox("Home team", rates["team"].tolist())
    away_team = st.selectbox("Away team", [t for t in rates["team"].tolist() if t != home_team])

    muH, muA = mlb_matchup_mu(rates, home_team, away_team)

    # Attach pitcher ERA adjustment if we have probables
    home_era = None
    away_era = None
    if prob is not None and not prob.empty:
        hrow = prob.loc[(prob["side"]=="Home") & (prob["team"].str.lower()==home_team.lower())]
        arow = prob.loc[(prob["side"]=="Away") & (prob["team"].str.lower()==away_team.lower())]
        if not hrow.empty:
            home_era = hrow.iloc[0]["ERA"]
        if not arow.empty:
            away_era = arow.iloc[0]["ERA"]
    muH_adj, muA_adj = _apply_pitcher_adjustment(muH, muA, home_era, away_era)

    pH, pA, mH, mA, tot = simulate_poisson_game(muH_adj, muA_adj)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team} win %", f"{pH*100:.1f}%")
    c2.metric(f"{away_team} win %", f"{pA*100:.1f}%")
    c3.metric("Expected total (runs)", f"{tot:.2f}")

# -------------------------- Player Props page (NEW) ---------------------------
else:
    st.subheader("📈 Player Props — CSV Uploads + Defense Adjustment")

    st.markdown("Upload your **QB**, **RB**, and **WR** CSVs (the same format you pasted). "
                "Then pick a player and enter a yardage line. "
                "Optionally upload a **Defense EPA** CSV and type the opponent code (e.g., `PHI`).")

    c_up1, c_up2, c_up3 = st.columns(3)
    up_qb = c_up1.file_uploader("QB CSV", type=["csv"], key="qb_csv")
    up_rb = c_up2.file_uploader("RB CSV", type=["csv"], key="rb_csv")
    up_wr = c_up3.file_uploader("WR CSV", type=["csv"], key="wr_csv")

    up_def = st.file_uploader("Defense CSV (EPA table you pasted)", type=["csv"], key="def_csv")

    df_qb = _read_csv(up_qb)
    df_rb = _read_csv(up_rb)
    df_wr = _read_csv(up_wr)
    df_def = _normalize_def_cols(_read_csv(up_def)) if up_def else pd.DataFrame(columns=["TEAM","EPA_PASS","EPA_RUSH","EPA_PLAY"])

    # Build a combined player list for selection
    options = []
    sources = []
    for df, tag in [(df_qb,"QB"), (df_rb,"RB"), (df_wr,"WR")]:
        if df is None or df.empty or "Player" not in (df.columns):
            continue
        temp = df.copy()
        temp["__POSGROUP__"] = tag
        options.append(temp[["Player","Team","__POSGROUP__"]])
        sources.append((tag, df))
    if options:
        show = pd.concat(options, ignore_index=True)
        st.dataframe(show.rename(columns={"__POSGROUP__":"Pos"}), use_container_width=True)
    else:
        st.info("Upload at least one of the QB / RB / WR CSVs to begin.")

    # UI controls (no team dropdown)
    sel_player = st.text_input("Player name (exact or partial)")
    prop_type = st.selectbox("Prop type", ["Passing Yards (QB)", "Rushing Yards (RB)", "Receiving Yards (WR)"])
    line_val = st.number_input("Yardage line", min_value=0.0, step=0.5)
    opp_code = st.text_input("Opponent (team code, optional — e.g., PHI). Leave blank to use league average.")

    if st.button("Run Simulation", use_container_width=True):
        if not sel_player:
            st.warning("Enter a player name.")
            st.stop()

        # determine which table to search
        pos_group = "QB" if "Passing" in prop_type else ("RB" if "Rushing" in prop_type else "WR")
        target_df = {"QB": df_qb, "RB": df_rb, "WR": df_wr}.get(pos_group)
        if target_df is None or target_df.empty:
            st.error(f"Please upload the {pos_group} CSV.")
            st.stop()

        row = _pick_player_row(target_df, sel_player)
        if row is None:
            st.error("Player not found in the uploaded CSV.")
            st.stop()

        base_mean = _yards_per_game(row, pos_group)
        if base_mean is None:
            st.error("Couldn't locate Y/G (yards per game) in your CSV row for this player.")
            st.stop()

        # defense multiplier from EPA
        factor = _def_factor(df_def, opp_code if opp_code else None, pos_group)

        # apply factor
        adj_mean = float(base_mean) * float(factor)

        # fixed SDs per position (kept simple)
        sd = SD_QB_PASS_YDS if pos_group == "QB" else (SD_RB_RUSH_YDS if pos_group == "RB" else SD_WR_REC_YDS)

        prob_over = _simulate_yards(adj_mean, line_val, sd)
        st.success(
            f"**{sel_player} — {prop_type}**\n\n"
            f"Mean (from CSV): **{base_mean:.1f}** yds\n"
            f"Defense factor (EPA): **×{factor:.3f}** → Adjusted mean: **{adj_mean:.1f}** yds\n"
            f"Line: **{line_val:.1f}** → **P(over) = {prob_over*100:.1f}%**, **P(under) = {(1-prob_over)*100:.1f}%**"
        )

        with st.expander("Show player row used"):
            st.dataframe(pd.DataFrame(row).T)

        if not df_def.empty:
            with st.expander("Defense table (normalized)"):
                st.dataframe(df_def, use_container_width=True)
