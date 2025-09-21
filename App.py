# app.py — NFL + MLB predictor (2025 only, stats-only) + Player Props (QB/RB/WR) with embedded 2025 defense EPA

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
    import statsapi  # for probables + pitcher stats (ERA)
    HAS_STATSAPI = True
except Exception:
    HAS_STATSAPI = False

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 20000
HOME_EDGE_NFL = 0.6   # ~0.6 pts to the home offense mean (small HFA)
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

# -------------------------- generic: Poisson sim ------------------------------
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
# NFL (2025) — team PF/PA + upcoming matchups (matchups-only UI)
# (UNCHANGED from your script)
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
# MLB (2025) — team RS/RA from BR + probable pitchers with ERA (statsapi)
# (UNCHANGED logic for teams; probables helper included)
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
# Embedded 2025 NFL Defense EPA table (lightweight; you don't need to upload)
# Columns: team, epa_play, epa_pass, epa_rush  (negative = better defense)
# ==============================================================================
_DEF_ROWS = [
    ("San Francisco 49ers", -0.16, -0.27, -0.01),
    ("Atlanta Falcons", -0.14, -0.18, -0.05),
    ("Los Angeles Rams", -0.13, -0.20,  0.00),
    ("Jacksonville Jaguars", -0.13, -0.20,  0.03),
    ("Green Bay Packers", -0.12, -0.08, -0.19),
    ("Denver Broncos", -0.10, -0.03, -0.19),
    ("Las Vegas Raiders", -0.08,  0.12, -0.45),
    ("Minnesota Vikings", -0.06, -0.25,  0.13),
    ("Washington Commanders", -0.06, -0.06, -0.04),
    ("Philadelphia Eagles", -0.05, -0.08,  0.00),
    ("Seattle Seahawks", -0.04,  0.06, -0.17),
    ("Indianapolis Colts", -0.03, -0.13,  0.17),
    ("Detroit Lions", -0.03,  0.16, -0.25),
    ("Arizona Cardinals", -0.02, -0.01, -0.05),
    ("Baltimore Ravens",  0.00, -0.06,  0.12),
    ("Cleveland Browns",  0.01,  0.15, -0.18),
    ("New Orleans Saints", 0.01,  0.05, -0.04),
    ("Tampa Bay Buccaneers", 0.02,  0.08, -0.09),
    ("Cincinnati Bengals", 0.02,  0.03, -0.01),
    ("Tennessee Titans",   0.03,  0.03,  0.04),
    ("Houston Texans",     0.05,  0.03,  0.08),
    ("Buffalo Bills",      0.05,  0.05,  0.05),
    ("Kansas City Chiefs", 0.12,  0.19,  0.04),
    ("Carolina Panthers",  0.12,  0.12,  0.12),
    ("Chicago Bears",      0.14,  0.27,  0.01),
    ("New York Giants",    0.14,  0.12,  0.15),
    ("New York Jets",      0.14,  0.16,  0.11),
    ("New England Patriots", 0.14, 0.33, -0.22),
    ("Pittsburgh Steelers", 0.16, 0.23,  0.09),
    ("Dallas Cowboys",     0.17,  0.26,  0.06),
    ("Miami Dolphins",     0.28,  0.41,  0.13),
    ("Los Angeles Chargers", -0.10, -0.20, 0.18),  # from your list (placed here)
]
DEFENSE_2025 = pd.DataFrame(_DEF_ROWS, columns=["team","epa_play","epa_pass","epa_rush"])

# Simple percentile-based factor (0.85 .. 1.15). Lower EPA (better D) -> factor < 1.
def _defense_factor(stat_type: str, opp_team: Optional[str]) -> float:
    if not opp_team:
        return 1.00
    d = DEFENSE_2025.copy()
    key = "epa_pass" if stat_type == "QB Passing Yards" else ("epa_rush" if stat_type == "RB Rushing Yards" else "epa_pass")
    d = d.sort_values(key)
    d["pct"] = np.linspace(0, 1, len(d))
    row = d.loc[d["team"].str.lower() == opp_team.lower()]
    if row.empty:
        return 1.00
    pct = float(row["pct"].iloc[0])  # 0 best (most negative) ... 1 worst
    # Map percentile to factor: best D ~0.85, worst D ~1.15
    return float(0.85 + 0.30 * pct)

# Truncated normal sampler (non-negative)
def _trunc_norm(mean: float, sd: float, n: int) -> np.ndarray:
    mean = max(0.0, float(mean))
    sd = max(1.0, float(sd))
    x = np.random.normal(mean, sd, size=n)
    return np.clip(x, 0, None)

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Win probabilities: **team scoring rates only** (NFL: PF/PA; MLB: RS/RA). "
    "Player Props: upload your **QB/RB/WR CSVs** and set a line; we'll simulate with an embedded 2025 defense table."
)

# NOTE: Keeping your NFL & MLB pages as-is; adding a third radio option for Player Props
sport = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page (matchups only) --------------------------
if sport == "NFL":
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
    pick = st.selectbox("Choose a matchup", options=choices)
    row = upcoming.iloc[list(choices).index(pick)]
    home, away = str(row["home_team"]), str(row["away_team"])

    muH, muA = nfl_matchup_mu(nfl_rates, home, away)
    pH, pA, eH, eA, eTot = simulate_poisson_game(muH, muA)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric(f"{home} win %", f"{pH*100:.1f}%")
    with c2: st.metric(f"{away} win %", f"{pA*100:.1f}%")
    with c3: st.metric("Expected total pts", f"{eTot:.1f}")

# -------------------------- MLB page (unchanged team logic) -------------------
elif sport == "MLB":
    st.subheader("⚾ MLB — 2025 REG season")
    rates = mlb_team_rates_2025()

    home = st.selectbox("Home team", sorted(rates["team"].unique().tolist()))
    away = st.selectbox("Away team", sorted([t for t in rates["team"].unique().tolist() if t != home]))

    muH, muA = mlb_matchup_mu(rates, home, away)

    # Optional: probable pitcher ERA adjustment (if statsapi is present)
    if HAS_STATSAPI:
        with st.expander("Probable Pitchers (auto ERA adjustment)", expanded=False):
            probs = _today_probables()
            st.dataframe(probs, use_container_width=True)
            # find ERAs if teams match
            era_home = float(probs.loc[(probs["side"]=="Home") & (probs["team"]==home)]["ERA"].fillna(np.nan).head(1).values[0]) if not probs.empty and (probs["team"]==home).any() else None
            era_away = float(probs.loc[(probs["side"]=="Away") & (probs["team"]==away)]["ERA"].fillna(np.nan).head(1).values[0]) if not probs.empty and (probs["team"]==away).any() else None
            if (era_home is not None) or (era_away is not None):
                muH, muA = _apply_pitcher_adjustment(muH, muA, era_home, era_away)

    pH, pA, eH, eA, eTot = simulate_poisson_game(muH, muA)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric(f"{home} win %", f"{pH*100:.1f}%")
    with c2: st.metric(f"{away} win %", f"{pA*100:.1f}%")
    with c3: st.metric("Expected total runs", f"{eTot:.1f}")

# -------------------------- Player Props page ---------------------------------
else:
    st.subheader("📈 Player Props — upload your CSVs, set a line, simulate")

    st.markdown("**Inputs:** Upload any (or all) of your CSVs. Then choose a player and enter the yardage line. "
                "Defense adjustment uses a built-in 2025 table (EPA/pass & EPA/rush).")

    cqb, crb, cwr = st.columns(3)
    with cqb:
        qb_file = st.file_uploader("QB CSV", type=["csv"], key="qb_up")
    with crb:
        rb_file = st.file_uploader("RB CSV", type=["csv"], key="rb_up")
    with cwr:
        wr_file = st.file_uploader("WR CSV", type=["csv"], key="wr_up")

    def _read_csv(f):
        try:
            return pd.read_csv(f)
        except Exception:
            try:
                return pd.read_csv(f, engine="python")
            except Exception:
                return None

    qb_df = _read_csv(qb_file) if qb_file else None
    rb_df = _read_csv(rb_file) if rb_file else None
    wr_df = _read_csv(wr_file) if wr_file else None

    # What do we need from each:
    # - QB: columns like Player, Y/G (passing yards per game) or Yds & G to derive Y/G
    # - RB: Player, Y/G (rushing)
    # - WR: Player, Y/G (receiving)
    # We'll auto-detect per-game mean.
    def _mean_from_df(df: pd.DataFrame, stat: str) -> pd.Series:
        # Try Y/G directly; else Yds/Games; else fallback to Yds
        s = None
        cols = {c.lower(): c for c in df.columns}
        yg = None
        yds = None
        g = None
        for cand in ["y/g","yg","y_per_g","yards_per_game"]:
            if cand in cols:
                yg = cols[cand]; break
        if yg is None:
            # passing: often "Y/G" or "Yds" + "G"
            for c in df.columns:
                if c.strip().lower() in ("y/g","y per g","yds/g"):
                    yg = c
                    break
        for cand in ["yds","yards","pass yds","rush yds","rec yds"]:
            if cand in cols:
                yds = cols[cand]; break
        for cand in ["g","games"]:
            if cand in cols:
                g = cols[cand]; break
        if yg and yg in df.columns:
            s = pd.to_numeric(df[yg], errors="coerce")
        elif (yds and g and yds in df.columns and g in df.columns):
            num = pd.to_numeric(df[yds], errors="coerce")
            den = pd.to_numeric(df[g], errors="coerce").replace(0, np.nan)
            s = num / den
        elif yds and yds in df.columns:
            # fallback: just use total; assume 1 game (rough)
            s = pd.to_numeric(df[yds], errors="coerce")
        else:
            s = pd.Series([np.nan]*len(df))
        return s

    # Build a combined list of players with their default mean and stat type
    pools = []
    if qb_df is not None and len(qb_df):
        m = _mean_from_df(qb_df, "pass")
        pools.append(pd.DataFrame({"Player": qb_df["Player"], "mean": m, "Type": "QB Passing Yards"}))
    if rb_df is not None and len(rb_df):
        m = _mean_from_df(rb_df, "rush")
        pools.append(pd.DataFrame({"Player": rb_df["Player"], "mean": m, "Type": "RB Rushing Yards"}))
    if wr_df is not None and len(wr_df):
        m = _mean_from_df(wr_df, "rec")
        pools.append(pd.DataFrame({"Player": wr_df["Player"], "mean": m, "Type": "WR Receiving Yards"}))

    if pools:
        players_all = pd.concat(pools, ignore_index=True)
        players_all = players_all.dropna(subset=["Player"]).drop_duplicates(subset=["Player","Type"])
    else:
        players_all = pd.DataFrame(columns=["Player","mean","Type"])

    # Selection widgets
    if players_all.empty:
        st.info("Upload at least one CSV (QB, RB, or WR) to select a player.")
        st.stop()

    stat_types = sorted(players_all["Type"].unique().tolist())
    pick_type = st.selectbox("Stat type", stat_types)

    subset = players_all.loc[players_all["Type"]==pick_type].copy()
    # Default mean from CSV (per-game if present)
    subset["mean"] = pd.to_numeric(subset["mean"], errors="coerce")
    subset = subset.dropna(subset=["mean"])

    player = st.selectbox("Player", sorted(subset["Player"].astype(str).tolist()))
    base_mean = float(subset.loc[subset["Player"]==player, "mean"].iloc[0])

    # Optional defense name (no dropdown required). If blank -> league average factor = 1.00
    opp = st.text_input("Opponent defense (type exact full team name, optional)", value="", placeholder="e.g., Dallas Cowboys")

    # Defense factor by EPA (embedded table)
    factor = _defense_factor(pick_type, opp.strip())
    adj_mean = base_mean * factor

    st.caption(f"Defense adj. factor = **{factor:.3f}** (1.00 = league avg).  Base mean: {base_mean:.1f} → Adjusted mean: {adj_mean:.1f}")

    line = st.number_input("Yardage line", min_value=0.0, value=float(round(max(10.0, adj_mean))))
    n_trials = st.slider("Simulation trials", 5000, 50000, SIM_TRIALS, step=5000)

    # Simple variance model (no manual SD): sd = max(10, 0.45 * mean)
    sd = max(10.0, 0.45 * adj_mean)

    if st.button("Run simulation"):
        sims = _trunc_norm(adj_mean, sd, int(n_trials))
        over_p = float((sims > line).mean())
        under_p = 1.0 - over_p
        st.metric("Over probability", f"{over_p*100:.1f}%")
        st.metric("Under probability", f"{under_p*100:.1f}%")

        # Quick summary table
        out = pd.DataFrame({
            "Player": [player],
            "Type": [pick_type],
            "Opponent (opt.)": [opp if opp.strip() else "(league avg)"],
            "Base Mean": [round(base_mean,1)],
            "Adj Mean": [round(adj_mean,1)],
            "SD (auto)": [round(sd,1)],
            "Line": [line],
            "Over %": [round(over_p*100,1)],
            "Under %": [round(under_p*100,1)],
            "Trials": [int(n_trials)]
        })
        st.dataframe(out, use_container_width=True)
