# app.py — NFL + MLB predictor (2025 only) + matchup-aware Player Props (CSV-only)
# Run: streamlit run app.py

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple

# ---- NFL (nfl_data_py) -------------------------------------------------------
import nfl_data_py as nfl

# ---- MLB (pybaseball + optional MLB-StatsAPI) --------------------------------
from pybaseball import schedule_and_record
try:
    import statsapi  # pip install MLB-StatsAPI
    HAS_STATSAPI = True
except Exception:
    HAS_STATSAPI = False

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10_000
HOME_EDGE_NFL = 0.6   # ~0.6 pts to home mean
EPS = 1e-9

# Hard-coded BR team IDs (stable)
MLB_TEAMS_2025 = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
}

# -------------------------- SHARED: simple Poisson/Normal sims ----------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # tiny home tiebreak (NFL vibe)
    p_home = float(wins_home.mean())
    p_away = 1.0 - p_home
    return p_home, p_away, float(h.mean()), float(a.mean()), float((h + a).mean())

def prob_over_normal(mean: float, sd: float, line: float, trials: int = SIM_TRIALS):
    sd = max(1e-6, float(sd))
    sims = np.random.normal(loc=mean, scale=sd, size=trials)
    sims = np.clip(sims, 0, None)
    p_over = float((sims > line).mean())
    return p_over

def prob_over_poisson(lmbda: float, line: float, trials: int = SIM_TRIALS):
    lmbda = max(1e-6, float(lmbda))
    sims = np.random.poisson(lmbda, size=trials)
    p_over = float((sims > line).mean())
    return p_over

# =============================== NFL ==========================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    date_col = None
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
        per = 45.0 / 2.0
        teams32 = [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
            "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
            "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
            "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
            "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
            "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
            "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
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

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float, float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"]) / 2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"]) / 2.0)
    return mu_home, mu_away

# =============================== MLB ==========================================
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
            else:
                sar["R"] = sar["R"].astype(float)
                sar["RA"] = sar["RA"].astype(float)
                g = max(1, int(len(sar)))
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

def get_probable_pitchers_and_era(home_team: str, away_team: str):
    if not HAS_STATSAPI:
        return None, None, None, None
    try:
        sched = statsapi.schedule()
    except Exception:
        return None, None, None, None
    h_name = h_era = a_name = a_era = None
    for g in sched:
        if g.get("home_name") == home_team and g.get("away_name") == away_team:
            hp = g.get("home_probable_pitcher"); ap = g.get("away_probable_pitcher")
            if hp:
                try:
                    pid = statsapi.lookup_player(hp)[0]["id"]
                    ps = statsapi.player_stats(pid, group="pitching", type="season")
                    h_era = float(ps[0]["era"]) if ps and ps[0].get("era") not in (None, "", "--") else None
                except Exception:
                    pass
                h_name = hp
            if ap:
                try:
                    pid = statsapi.lookup_player(ap)[0]["id"]
                    ps = statsapi.player_stats(pid, group="pitching", type="season")
                    a_era = float(ps[0]["era"]) if ps and ps[0].get("era") not in (None, "", "--") else None
                except Exception:
                    pass
                a_name = ap
            break
    return h_name, h_era, a_name, a_era

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str,
                   h_pit_era: Optional[float], a_pit_era: Optional[float]) -> Tuple[float, float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)  # team means
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    # Nudge by ERA (very gentle)
    LgERA = 4.30
    scale = 0.05
    if h_pit_era is not None:
        mu_away = max(EPS, mu_away - scale * (LgERA - h_pit_era))
    if a_pit_era is not None:
        mu_home = max(EPS, mu_home - scale * (LgERA - a_pit_era))
    return mu_home, mu_away

# ============================ CSV helpers =====================================
def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in df.columns.values
        ]
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.fullmatch(r"")]
    return df

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "O":
            s = df[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            df[c] = pd.to_numeric(s, errors="ignore")
    return df

def read_user_csv(upload) -> pd.DataFrame:
    if not upload:
        return pd.DataFrame()
    try:
        df = pd.read_csv(upload)
        df = clean_headers(df)
        df = coerce_numeric(df)
        return df
    except Exception as e:
        st.warning(f"Could not read CSV: {e}")
        return pd.DataFrame()

# ================================ UI ==========================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Matchup win % from **team scoring rates** (NFL: PF/PA; MLB: RS/RA), "
    "MLB optionally nudged by **probable starters + ERA**. "
    "Player props use **your CSVs** with matchup adjustments."
)

page = st.sidebar.radio(
    "Choose a page",
    ["NFL (2025)", "MLB (2025)", "NFL Player Props (CSV-only, matchup-aware)"],
    index=0
)

# -------------------------- NFL page ------------------------------------------
if page == "NFL (2025)":
    st.subheader("🏈 NFL — pick an upcoming matchup")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build NFL team rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming 2025 games yet.")
        st.stop()

    show_dates = "date" in upcoming.columns and upcoming["date"].astype(str).str.len().gt(0).any()
    if show_dates:
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
        with col3: st.metric(label="Expected total", value=f"{exp_t:.1f}")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
    except Exception as e:
        st.error(str(e))

# -------------------------- MLB page ------------------------------------------
elif page == "MLB (2025)":
    st.subheader("⚾ MLB — pick any matchup")
    try:
        mlb_rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't load MLB team rates: {e}")
        st.stop()

    teams = mlb_rates["team"].sort_values().tolist()
    if not teams:
        st.info("No MLB team data yet.")
        st.stop()

    home = st.selectbox("Home team", teams, index=0, key="mlb_home")
    away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")

    h_name = h_era = a_name = a_era = None
    if HAS_STATSAPI:
        h_name, h_era, a_name, a_era = get_probable_pitchers_and_era(home, away)

    try:
        mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away, h_era, a_era)
        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with col2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        with col3: st.metric(label="Expected total", value=f"{exp_t:.1f}")
        if h_name or a_name:
            st.caption(
                f"Probable starters — {home}: **{h_name or 'TBD'}**"
                f"{(' (ERA ' + str(h_era) + ')') if h_era is not None else ''} • "
                f"{away}: **{a_name or 'TBD'}**"
                f"{(' (ERA ' + str(a_era) + ')') if a_era is not None else ''}"
            )
        else:
            st.caption("No probable starters found — using team rates only.")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
    except Exception as e:
        st.error(str(e))

# --------------------- NFL Player Props (CSV-only, matchup-aware) -------------
else:
    st.subheader("🎯 NFL Player Props — CSV-only (QB/RB/WR) with matchup adjustments")

    st.markdown("**Upload your player CSVs** (any headers; we autodetect common ones)")
    up_qb = st.file_uploader("QB CSV", type=["csv"], key="qbcsv")
    up_rb = st.file_uploader("RB CSV (optional)", type=["csv"], key="rbcsv")
    up_wr = st.file_uploader("WR/TE CSV (optional)", type=["csv"], key="wrcsv")

    def std_pg(df: pd.DataFrame, pos_hint: str) -> pd.DataFrame:
        """Compute per-game fields if totals present; keep original if already per-game."""
        if df.empty: return df
        df = df.copy()
        for base in ["G","Cmp","Att","Yds","TD","Int","Tgt","Rec","Rush Att","Rush Yds","Rec Yds"]:
            if base not in df.columns:
                # try common aliases
                alias = {
                    "Rush Att": ["Att"], "Rush Yds": ["Yds"], "Rec Yds": ["Yds"]
                }.get(base, [])
                for a in alias:
                    if a in df.columns and base not in df.columns:
                        df[base] = df[a]
        G = df["G"] if "G" in df.columns else np.nan

        def mk_pg(total, pg_name):
            if total in df.columns and "G" in df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    df[pg_name] = df[total] / df["G"].replace(0, np.nan)
            return

        # QB
        if pos_hint == "QB":
            if "Yds_pg" not in df.columns and "Yds" in df.columns: mk_pg("Yds","Yds_pg")
            if "Cmp_pg" not in df.columns and "Cmp" in df.columns: mk_pg("Cmp","Cmp_pg")
            if "Att_pg" not in df.columns and "Att" in df.columns: mk_pg("Att","Att_pg")
            if "TD_pg"  not in df.columns and "TD"  in df.columns: mk_pg("TD","TD_pg")
            if "Int_pg" not in df.columns and "Int" in df.columns: mk_pg("Int","Int_pg")
        # RB
        if pos_hint == "RB":
            if "RushYds_pg" not in df.columns and "Rush Yds" in df.columns: mk_pg("Rush Yds","RushYds_pg")
            if "RushAtt_pg" not in df.columns and "Rush Att" in df.columns: mk_pg("Rush Att","RushAtt_pg")
            # sometimes RB receiving included
            if "Rec_pg" not in df.columns and "Rec" in df.columns: mk_pg("Rec","Rec_pg")
            if "RecYds_pg" not in df.columns and "Rec Yds" in df.columns: mk_pg("Rec Yds","RecYds_pg")
        # WR/TE
        if pos_hint == "WR":
            if "Rec_pg" not in df.columns and "Rec" in df.columns: mk_pg("Rec","Rec_pg")
            if "RecYds_pg" not in df.columns and "Rec Yds" in df.columns: mk_pg("Rec Yds","RecYds_pg")
            if "Tgt_pg" not in df.columns and "Tgt" in df.columns: mk_pg("Tgt","Tgt_pg")

        # fill per-game NaNs with 0 for display
        for c in [c for c in df.columns if c.endswith("_pg")]:
            df[c] = df[c].fillna(0.0)
        return df

    qbs = std_pg(read_user_csv(up_qb), "QB")
    rbs = std_pg(read_user_csv(up_rb), "RB")
    wrs = std_pg(read_user_csv(up_wr), "WR")

    # Must have at least one table
    if qbs.empty and rbs.empty and wrs.empty:
        st.info("Upload at least one CSV (QB/RB/WR).")
        st.stop()

    # Optional Defense CSV for matchup scaling
    st.markdown("**(Optional) Defense allowed per-game CSV**")
    st.caption("Include columns like: team, pass_yds_allowed_pg, pass_td_allowed_pg, int_made_pg, rush_yds_allowed_pg, rec_yds_allowed_pg, receptions_allowed_pg …")
    defcsv = st.file_uploader("Defense allowed CSV", type=["csv"], key="defcsv")
    defense = read_user_csv(defcsv)
    if not defense.empty and "team" in defense.columns:
        defense["team_key"] = defense["team"].str.lower().str.strip()
        def_league_means = {c: defense[c].mean() for c in defense.columns if c.endswith("_allowed_pg") or c.endswith("_made_pg")}
    else:
        def_league_means = {}
        if defcsv: st.warning("Defense CSV loaded but missing 'team' column; no defense adjustment will be applied.")

    # Upcoming schedule for opponent selection
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception:
        nfl_rates, upcoming = pd.DataFrame(), pd.DataFrame()
    all_teams = sorted(set(upcoming.get("home_team", pd.Series(dtype=str))).union(set(upcoming.get("away_team", pd.Series(dtype=str))))) if not upcoming.empty else []

    # Combine for selection
    pools = []
    if not qbs.empty: pools.append(qbs.assign(Pos="QB"))
    if not rbs.empty: pools.append(rbs.assign(Pos="RB"))
    if not wrs.empty: pools.append(wrs.assign(Pos="WR/TE"))
    players_all = pd.concat(pools, ignore_index=True)

    st.markdown("### Pick player + matchup")
    c1, c2 = st.columns([1,1])
    with c1:
        player = st.selectbox("Player", players_all["Player"].dropna().astype(str).sort_values().tolist())
        market = st.selectbox(
            "Market",
            [
                "Pass Yds", "Completions", "Pass TD", "Interceptions",  # QB
                "Rush Yds", "Rush Att",                                 # RB/QB
                "Rec Yds", "Receptions",                                # WR/RB/TE
            ],
            index=0
        )
    with c2:
        defaults = {
            "Pass Yds": 240.5, "Completions": 22.5, "Pass TD": 1.5, "Interceptions": 0.5,
            "Rush Yds": 58.5, "Rush Att": 13.5,
            "Rec Yds": 55.5, "Receptions": 4.5,
        }
        line = st.number_input("Prop line", min_value=0.0, value=float(defaults.get(market, 1.5)), step=0.5)

    prow = players_all.loc[players_all["Player"] == player].iloc[0]
    player_team = str(prow.get("Team", "") or "")
    opp = st.selectbox("Opponent team", all_teams if all_teams else [""], index=0 if all_teams else None, placeholder="Type opponent…")

    # Strength knobs — “regular” SD defaults
    st.markdown("### Matchup strength (how much to weigh the opponent and team context)")
    w_def = st.slider("Defense weight (0=no effect, 1=full scale)", 0.0, 1.0, 0.6, 0.05)
    use_team_off = st.checkbox("Nudge by offense scoring strength (team PF vs league avg)", value=True)

    sd_defaults = {
        "Pass Yds": 60.0, "Completions": 6.0,
        "Pass TD": None, "Interceptions": None,
        "Rush Yds": 20.0, "Rush Att": 4.0,
        "Rec Yds": 18.0, "Receptions": 2.5,
    }
    sd_base = sd_defaults.get(market, 10.0)
    sd = None
    if market in ("Pass Yds","Completions","Rush Yds","Rush Att","Rec Yds","Receptions"):
        sd = st.slider("Simulation SD (volatility)", 2.0, 120.0, float(sd_base), 1.0)

    # ---- Build neutral per-game from row (handles different CSV schemas) ------
    G = float(prow.get("G
