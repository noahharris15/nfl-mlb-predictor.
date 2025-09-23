# app.py — NFL + MLB predictor (2025 only) + Player Props with Defense CSV
# - NFL & MLB pages unchanged from your original behavior
# - Player Props: pick player via dropdown, enter line, simulate
# - Defense strength comes from uploaded file or 'nfl Defense.xlsx/.csv' in repo root

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
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6   # small home points edge in NFL Poisson
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
        wins_home[ties] = 0.53  # tiny home tiebreaker
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean()), float((h + a).mean())

# ==============================================================================
# NFL (2025) — team PF/PA + upcoming matchups (matchups-only UI)
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
    return pd.DataFrame(rows, columns=cols).drop_duplicates(subset=["side","team"]).reset_index(drop=True)

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
# Player Props helpers (CSV upload + defense merge + simple sim)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_player_csv(upload, kind: str) -> pd.DataFrame:
    """
    Accepts either an uploaded file or a blank (returns empty).
    Columns expected (any order) — we’ll auto-detect common ones:
      QB: 'Player','Yds','Y/G','Team','Pos'
      RB: 'Player','Yds','Y/G','Team','Pos'
      WR: 'Player','Yds','Y/G','Team','Pos'
    """
    if upload is None:
        return pd.DataFrame()
    df = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    # normalize
    rename_map = {c.lower(): c for c in df.columns}
    def get(colnames):
        for c in df.columns:
            if c.strip().lower() in [n.lower() for n in colnames]:
                return c
        return None
    want_player = get(["player"])
    want_team   = get(["team"])
    want_pos    = get(["pos","position"])
    # yards columns vary by table
    if kind == "QB":
        want_yg = get(["y/g","pass y/g","py/g","pass_yg"])
        want_y  = get(["yds","pass yds","py","pass_yds","yards"])
    elif kind == "RB":
        want_yg = get(["y/g","rush y/g","ry/g","rush_yg"])
        want_y  = get(["yds","rush yds","ry","rush_yds","yards"])
    else:  # WR
        want_yg = get(["y/g","rec y/g","ry/g","rec_yg"])
        want_y  = get(["yds","rec yds","r yds","rec_yds","yards"])

    # build clean frame
    out = pd.DataFrame()
    if want_player is not None: out["Player"] = df[want_player].astype(str).str.strip()
    if want_team   is not None: out["Team"]   = df[want_team].astype(str).str.strip()
    if want_pos    is not None: out["Pos"]    = df[want_pos].astype(str).str.upper()
    if want_yg     is not None: out["Y_perG"] = pd.to_numeric(df[want_yg], errors="coerce")
    if want_y      is not None and "Y_perG" not in out:
        # derive per-game mean if only Yds and G present
        gcol = get(["g","games"])
        if gcol is not None:
            out["Y_perG"] = pd.to_numeric(df[want_y], errors="coerce") / pd.to_numeric(df[gcol], errors="coerce")
    return out.dropna(subset=["Player","Y_perG"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_defense_strength(upload) -> pd.DataFrame:
    """
    Reads defense strength table:
     - If a file is uploaded, use it
     - else tries repo root: 'nfl Defense.xlsx' then 'nfl Defense.csv'
     - else returns empty (neutral)
    Expected columns (flexible):
      'Team' and at least one of: 'EPA/RUSH', 'EPA/PASS', or a single 'EPA/PLAY'
    We’ll compute a simple blend per position:
      QB uses EPA/PASS (fallback EPA/PLAY)
      RB uses EPA/RUSH (fallback EPA/PLAY)
      WR uses EPA/PASS (fallback EPA/PLAY)
    """
    df = None
    # 1) uploader
    if upload is not None:
        try:
            df = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
        except Exception:
            df = None
    # 2) repo root files
    if df is None:
        for fname in ["nfl Defense.xlsx", "nfl Defense.csv", "NFL defense.xlsx", "NFL defense.csv"]:
            try:
                if fname.lower().endswith(".csv"):
                    df = pd.read_csv(fname)
                else:
                    df = pd.read_excel(fname)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    break
            except Exception:
                continue
    if df is None:
        return pd.DataFrame(columns=["Team","EPA_RUSH","EPA_PASS","EPA_PLAY"])

    # Normalize columns
    def collike(df, options):
        for c in df.columns:
            if c.strip().lower() in [o.lower() for o in options]:
                return c
        return None

    tcol = collike(df, ["team","defense","def team","club"])
    eplay = collike(df, ["epa/play","epa_per_play","epa"])
    erush = collike(df, ["epa/rush","epa_rush","rush epa"])
    epass = collike(df, ["epa/pass","epa_pass","pass epa"])

    out = pd.DataFrame()
    if tcol is None:
        return pd.DataFrame(columns=["Team","EPA_RUSH","EPA_PASS","EPA_PLAY"])
    out["Team"] = df[tcol].astype(str).str.strip()

    if erush is not None:
        out["EPA_RUSH"] = pd.to_numeric(df[erush], errors="coerce")
    if epass is not None:
        out["EPA_PASS"] = pd.to_numeric(df[epass], errors="coerce")
    if eplay is not None:
        out["EPA_PLAY"] = pd.to_numeric(df[eplay], errors="coerce")

    return out

def _def_adj_factor(row: pd.Series, pos: str) -> float:
    """
    Convert defense EPA into a simple multiplicative factor on the player's mean yards.
    Negative EPA (good defense) -> factor < 1, Positive EPA (bad defense) -> factor > 1.
    We clamp to a mild range so it never explodes.
    """
    base = 0.0
    if pos == "QB" or pos == "WR":
        # passing-related
        e = row.get("EPA_PASS")
        if pd.isna(e): e = row.get("EPA_PLAY", 0.0)
        base = float(e if not pd.isna(e) else 0.0)
    elif pos == "RB":
        e = row.get("EPA_RUSH")
        if pd.isna(e): e = row.get("EPA_PLAY", 0.0)
        base = float(e if not pd.isna(e) else 0.0)
    else:
        base = float(row.get("EPA_PLAY", 0.0))

    # Linear map: factor = 1 + k * EPA ; k ~ 0.6 chosen to give ~±12–20% at typical EPA ±0.2–0.3
    k = 0.6
    factor = 1.0 + k * base
    return float(np.clip(factor, 0.75, 1.25))

def _simple_sd(mean_val: float) -> float:
    # Basic, no knobs: protects against zero variance props
    return max(8.0, 0.75 * float(mean_val))

def simulate_prop(prob_mean: float, line: float, trials: int = 20000) -> float:
    mu = float(prob_mean)
    sd = _simple_sd(mu)
    sims = np.random.normal(loc=mu, scale=sd, size=trials)
    return float((sims > float(line)).mean())

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025")

sport = st.radio("Pick a page", ["NFL", "MLB", "Player Props"], horizontal=True)

# -------------------------- NFL page (matchups only) --------------------------
if sport == "NFL":
    st.subheader("🏈 NFL — 2025 Regular Season (team scoring only)")

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
    pick = st.selectbox("Select upcoming game", choices)
    idx = choices[choices == pick].index[0]
    home = str(upcoming.iloc[idx]["home_team"])
    away = str(upcoming.iloc[idx]["away_team"])

    try:
        mu_home, mu_away = nfl_matchup_mu(nfl_rates, home, away)
        pH, pA, mH, mA, tot = simulate_poisson_game(mu_home, mu_away)
        st.write(f"**{home}** vs **{away}**")
        st.write(f"Sim means: {home} {mH:.2f} — {away} {mA:.2f} | Total ~ {tot:.2f}")
        st.write(f"Win %: {home} {100*pH:.1f}% — {away} {100*pA:.1f}%")
    except Exception as e:
        st.error(str(e))

# -------------------------- MLB page (matchups only) --------------------------
elif sport == "MLB":
    st.subheader("⚾ MLB — 2025 Season (team runs + probables ERA)")

    try:
        rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build MLB team run rates: {e}")
        st.stop()

    teams = sorted(rates["team"].unique().tolist())
    c1, c2 = st.columns(2)
    with c1: home = st.selectbox("Home team", teams, index=teams.index("Los Angeles Dodgers") if "Los Angeles Dodgers" in teams else 0)
    with c2: away = st.selectbox("Away team", [t for t in teams if t != home])

    mu_home, mu_away = mlb_matchup_mu(rates, home, away)

    # Try to adjust by probables
    prob = _today_probables()
    eh = ea = None
    if not prob.empty:
        ph = prob.loc[(prob["side"]=="Home") & (prob["team"].str.lower()==home.lower())]
        pa = prob.loc[(prob["side"]=="Away") & (prob["team"].str.lower()==away.lower())]
        if not ph.empty: eh = ph.iloc[0]["ERA"]
        if not pa.empty: ea = pa.iloc[0]["ERA"]
    adj_home, adj_away = _apply_pitcher_adjustment(mu_home, mu_away, eh, ea)

    pH, pA, mH, mA, tot = simulate_poisson_game(adj_home, adj_away)
    st.write(f"**{home}** vs **{away}**")
    st.write(f"Sim means: {home} {mH:.2f} — {away} {mA:.2f} | Total ~ {tot:.2f}")
    st.write(f"Win %: {home} {100*pH:.1f}% — {away} {100*pA:.1f}%")
    if eh is not None or ea is not None:
        st.caption(f"Pitcher ERA adj used — Home: {eh if eh is not None else '—'}, Away: {ea if ea is not None else '—'}")

# -------------------------- Player Props page ---------------------------------
else:
    st.subheader("📈 Player Props — dropdown player, defense-adjusted, no knobs")

    c_up = st.container()
    with c_up:
        st.write("Upload your **QB**, **RB**, and **WR** CSV/Excel (same format you’ve been using).")
        up_qb = st.file_uploader("QB CSV/XLSX", type=["csv","xlsx"], key="qb_up")
        up_rb = st.file_uploader("RB CSV/XLSX", type=["csv","xlsx"], key="rb_up")
        up_wr = st.file_uploader("WR CSV/XLSX", type=["csv","xlsx"], key="wr_up")
        st.write("Upload weekly **Defense** CSV/Excel (or place `nfl Defense.xlsx` in repo root).")
        up_def = st.file_uploader("Defense CSV/XLSX (EPA columns)", type=["csv","xlsx"], key="def_up")

    df_qb = load_player_csv(up_qb, "QB")
    df_rb = load_player_csv(up_rb, "RB")
    df_wr = load_player_csv(up_wr, "WR")
    df_def = load_defense_strength(up_def)

    # Quick guidance
    if df_def.empty:
        st.warning("Defense table not found — using **neutral** adjustment (factor = 1.00). "
                   "Upload a defense file or put `nfl Defense.xlsx` (or .csv) in repo root.")
    else:
        st.caption("Defense strength loaded. We’ll use EPA/PASS for QBs/WRs and EPA/RUSH for RBs (fallback to EPA/PLAY).")

    # Build a single list of players by position for dropdown selection
    tabs = st.tabs(["QB Passing Yards", "RB Rushing Yards", "WR Receiving Yards"])

    # ---- QB tab ----
    with tabs[0]:
        if df_qb.empty:
            st.info("Upload a QB file to enable this tab.")
        else:
            qbs = df_qb["Player"].tolist()
            qb_pick = st.selectbox("Choose QB", qbs)
            qb_line = st.number_input("Line (passing yards)", min_value=0, value=250, step=1)
            qb_row = df_qb.loc[df_qb["Player"] == qb_pick].iloc[0]
            qb_team = str(qb_row.get("Team","")).strip()
            qb_mu = float(qb_row["Y_perG"])

            # Defense join (by team name, case-insensitive, trimmed)
            factor = 1.0
            if not df_def.empty and qb_team:
                drow = df_def.loc[df_def["Team"].str.strip().str.lower() == qb_team.lower()]
                if not drow.empty:
                    factor = _def_adj_factor(drow.iloc[0], "QB")
            adj_mu = qb_mu * factor

            p_over = simulate_prop(adj_mu, qb_line, trials=30000)
            st.write(f"Mean (season-to-date): {qb_mu:.1f}   | Defense factor: {factor:.3f}   | Adjusted mean: {adj_mu:.1f}")
            st.success(f"**Over {qb_line}**: {100*p_over:.1f}%   | **Under**: {100*(1-p_over):.1f}%")

    # ---- RB tab ----
    with tabs[1]:
        if df_rb.empty:
            st.info("Upload an RB file to enable this tab.")
        else:
            rbs = df_rb["Player"].tolist()
            rb_pick = st.selectbox("Choose RB", rbs)
            rb_line = st.number_input("Line (rushing yards)", min_value=0, value=60, step=1)
            rb_row = df_rb.loc[df_rb["Player"] == rb_pick].iloc[0]
            rb_team = str(rb_row.get("Team","")).strip()
            rb_mu = float(rb_row["Y_perG"])

            factor = 1.0
            if not df_def.empty and rb_team:
                drow = df_def.loc[df_def["Team"].str.strip().str.lower() == rb_team.lower()]
                if not drow.empty:
                    factor = _def_adj_factor(drow.iloc[0], "RB")
            adj_mu = rb_mu * factor

            p_over = simulate_prop(adj_mu, rb_line, trials=30000)
            st.write(f"Mean (season-to-date): {rb_mu:.1f}   | Defense factor: {factor:.3f}   | Adjusted mean: {adj_mu:.1f}")
            st.success(f"**Over {rb_line}**: {100*p_over:.1f}%   | **Under**: {100*(1-p_over):.1f}%")

    # ---- WR tab ----
    with tabs[2]:
        if df_wr.empty:
            st.info("Upload a WR file to enable this tab.")
        else:
            wrs = df_wr["Player"].tolist()
            wr_pick = st.selectbox("Choose WR", wrs)
            wr_line = st.number_input("Line (receiving yards)", min_value=0, value=65, step=1)
            wr_row = df_wr.loc[df_wr["Player"] == wr_pick].iloc[0]
            wr_team = str(wr_row.get("Team","")).strip()
            wr_mu = float(wr_row["Y_perG"])

            factor = 1.0
            if not df_def.empty and wr_team:
                drow = df_def.loc[df_def["Team"].str.strip().str.lower() == wr_team.lower()]
                if not drow.empty:
                    factor = _def_adj_factor(drow.iloc[0], "WR")
            adj_mu = wr_mu * factor

            p_over = simulate_prop(adj_mu, wr_line, trials=30000)
            st.write(f"Mean (season-to-date): {wr_mu:.1f}   | Defense factor: {factor:.3f}   | Adjusted mean: {adj_mu:.1f}")
            st.success(f"**Over {wr_line}**: {100*p_over:.1f}%   | **Under**: {100*(1-p_over):.1f}%")
