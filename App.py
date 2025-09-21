# app.py — NFL + MLB predictor (2025 only, stats-only) with MLB probable-pitcher ERA
# + SIMPLE NFL PLAYER PROPS (auto defense strength, no sliders)

import io
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
# NFL (2025) — team PF/PA + upcoming matchups
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
# Helpers for the Props page
# ==============================================================================
def _clean_pasted_csv(text: str) -> str:
    if not text:
        return text
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # drop banner line like "Receiving,Receiving,...,-additional"
    if lines:
        toks = [t.strip() for t in lines[0].split(",") if t.strip()]
        if len(set(toks)) == 1 or (len(toks) > 2 and toks[-1].lower() == "-additional"):
            lines = lines[1:]

    # drop last sentinel column (e.g., "-9999")
    if lines:
        hdr = [t.strip() for t in lines[0].split(",")]
        if hdr and (hdr[-1].startswith("-") or hdr[-1].isdigit()):
            keep = len(hdr) - 1
            lines = [",".join(row.split(",")[:keep]) for row in lines]

    return "\n".join(lines)

def _read_csv_widget(label: str, key_prefix: str) -> Optional[pd.DataFrame]:
    c1, c2 = st.columns(2)
    with c1:
        f = st.file_uploader(f"{label} — Upload CSV", type=["csv"], key=f"{key_prefix}_file")
    with c2:
        txt = st.text_area(f"{label} — Or paste CSV text", height=160, key=f"{key_prefix}_paste")

    if f is not None:
        try:
            return pd.read_csv(io.StringIO(f.getvalue().decode("utf-8")), engine="python")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            return None
    if txt and txt.strip():
        try:
            cleaned = _clean_pasted_csv(txt)
            return pd.read_csv(io.StringIO(cleaned), engine="python")
        except Exception as e:
            st.error(f"Could not parse pasted CSV: {e}")
            return None
    return None

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    drop = [c for c in df.columns if c == "" or c.lower().startswith("unnamed")]
    if drop:
        df = df.drop(columns=drop)

    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    ren = {}
    mapping = {
        "Player": pick("player","name"),
        "Team": pick("team","tm"),
        "Pos": pick("pos","position"),
        "G": pick("g","games"),
        "Season": pick("season","yr","year"),
        "Yds": pick("yds","yards","pass yds","rush yds","rec yds"),
        "Y_per_G": pick("y/g","yds/g","ypg","yards/g"),
    }
    for canon, src in mapping.items():
        if src and src != canon:
            ren[src] = canon
    df = df.rename(columns=ren)

    if "Y_per_G" not in df.columns:
        if {"Yds","G"}.issubset(df.columns):
            y = pd.to_numeric(df["Yds"], errors="coerce")
            g = pd.to_numeric(df["G"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                df["Y_per_G"] = y / g
        else:
            df["Y_per_G"] = np.nan
    return df

def _ensure_pos(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    df = df.copy()
    if "Pos" not in df.columns or df["Pos"].isna().all():
        df["Pos"] = pos
    return df

def _fair_ml_from_prob(p: float) -> str:
    if p <= 0: return "∞"
    if p >= 0.5:  # negative odds
        return str(int(round(-100 * p / (1 - p))))
    return f"+{int(round(100 * (1 - p) / p))}"

# ==============================================================================
# UI
# ==============================================================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Win probabilities are simulated from **team scoring rates only**. "
    "Props page uses your player CSVs and **auto opponent defense** from team points allowed."
)

page = st.radio("Pick a page", ["NFL", "MLB", "NFL Props"], horizontal=True)

# -------------------------- NFL page --------------------------
if page == "NFL":
    st.subheader("🏈 NFL — 2025 REG season (team-only)")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build team rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming games found in the 2025 schedule yet.")
        st.stop()

    if "date" in upcoming.columns and upcoming["date"].astype(str).str.len().gt(0).any():
        choices_df = upcoming.copy()
        choices_df["label"] = choices_df["home_team"] + " vs " + choices_df["away_team"] + " — " + choices_df["date"].astype(str)
    else:
        choices_df = upcoming.copy()
        choices_df["label"] = choices_df["home_team"] + " vs " + choices_df["away_team"]

    pick = st.selectbox("Upcoming matchup", choices_df["label"].tolist())
    sel = choices_df.loc[choices_df["label"] == pick].iloc[0]
    home, away = sel["home_team"], sel["away_team"]

    mu_home, mu_away = nfl_matchup_mu(nfl_rates, home, away)
    p_home, p_away, mean_h, mean_a, total = simulate_poisson_game(mu_home, mu_away)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win %", f"{p_home*100:.1f}%")
    c2.metric(f"{away} win %", f"{p_away*100:.1f}%")
    c3.metric("Avg total (pts)", f"{total:.2f}")
    st.caption(f"Poisson means — {home}: {mu_home:.2f}, {away}: {mu_away:.2f}")

# -------------------------- MLB page --------------------------
elif page == "MLB":
    st.subheader("⚾ MLB — 2025 REG season (team-only + probables ERA)")
    rates = mlb_team_rates_2025()
    st.dataframe(rates, use_container_width=True, height=350)

    st.markdown("**Today's Probable Pitchers (ERA)**")
    prob = _today_probables()
    if prob.empty:
        st.info("No probables found (or statsapi unavailable).")
    else:
        st.dataframe(prob, use_container_width=True)

    st.markdown("**Quick Matchup Sim**")
    teams = sorted(rates["team"].tolist())
    home = st.selectbox("Home", teams, index=min(teams.index("Los Angeles Dodgers") if "Los Angeles Dodgers" in teams else 0, len(teams)-1))
    away = st.selectbox("Away", teams, index=min(teams.index("San Francisco Giants") if "San Francisco Giants" in teams else 1, len(teams)-1))

    mu_home, mu_away = mlb_matchup_mu(rates, home, away)

    # try attach ERA if available
    h_era = None
    a_era = None
    if not prob.empty:
        he = prob.loc[(prob["side"] == "Home") & (prob["team"] == home)]
        ae = prob.loc[(prob["side"] == "Away") & (prob["team"] == away)]
        if not he.empty: h_era = he.iloc[0]["ERA"]
        if not ae.empty: a_era = ae.iloc[0]["ERA"]
    mu_home, mu_away = _apply_pitcher_adjustment(mu_home, mu_away, h_era, a_era)

    p_home, p_away, mean_h, mean_a, total = simulate_poisson_game(mu_home, mu_away)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win %", f"{p_home*100:.1f}%")
    c2.metric(f"{away} win %", f"{p_away*100:.1f}%")
    c3.metric("Avg total (runs)", f"{total:.2f}")
    st.caption(f"Poisson means — {home}: {mu_home:.2f}, {away}: {mu_away:.2f} (ERA-adjusted)")

# -------------------------- NFL Props page --------------------------
else:
    st.subheader("🏈 NFL Player Props — Simple (auto defense strength)")

    st.markdown("**1) Load your player CSVs**")
    qbs = _read_csv_widget("QBs CSV", "qbs")
    rbs = _read_csv_widget("RBs CSV", "rbs")
    wrs = _read_csv_widget("WRs CSV", "wrs")

    frames = []
    if qbs is not None: frames.append(_ensure_pos(_standardize_cols(qbs), "QB"))
    if rbs is not None: frames.append(_ensure_pos(_standardize_cols(rbs), "RB"))
    if wrs is not None: frames.append(_ensure_pos(_standardize_cols(wrs), "WR"))

    if not frames:
        st.info("Upload/paste at least one of QBs/RBs/WRs to continue.")
        st.stop()

    players = pd.concat(frames, ignore_index=True)
    keep = [c for c in ["Player","Team","Pos","Season","G","Yds","Y_per_G"] if c in players.columns]
    players = players[keep].copy()

    st.markdown("**2) Choose market & player**")
    market = st.selectbox("Market", ["Pass Yds (QB)", "Rush Yds (RB)", "Rec Yds (WR)"])

    if market == "Pass Yds (QB)":
        pool = players[players["Pos"] == "QB"].copy()
        auto_sd = 40.0
    elif market == "Rush Yds (RB)":
        pool = players[players["Pos"] == "RB"].copy()
        auto_sd = 25.0
    else:
        pool = players[players["Pos"] == "WR"].copy()
        auto_sd = 30.0

    if pool.empty:
        st.warning("No players for that market in your uploads.")
        st.stop()

    player_name = st.selectbox("Player", sorted(pool["Player"].dropna().astype(str).unique().tolist()))
    row = pool.loc[pool["Player"] == player_name].iloc[0]
    base_mean = float(pd.to_numeric(row.get("Y_per_G", 0.0), errors="coerce") or 0.0)
    team = str(row.get("Team", ""))

    # Auto defense factor from NFL team PA_pg (no user file required)
    st.markdown("**3) Opponent**")
    nfl_rates, _upcoming = nfl_team_rates_2025()
    opp = st.selectbox("Opponent team (from NFL team list)", [""] + sorted(nfl_rates["team"].tolist()))
    def_factor = 1.0
    if opp:
        # Use PA_pg as a generic defensive strength proxy across markets.
        # Scale vs league average PA_pg (lower PA => tougher defense => factor < 1).
        league_pa = float(nfl_rates["PA_pg"].mean())
        opp_pa = float(nfl_rates.loc[nfl_rates["team"] == opp, "PA_pg"].iloc[0])
        if league_pa > 0:
            def_factor = opp_pa / league_pa
        # Clamp a bit to avoid extremes early season
        def_factor = float(np.clip(def_factor, 0.75, 1.25))

    st.markdown("**4) Prop input**")
    prop_line = st.number_input("Prop line", value=float(np.round(base_mean, 1)), step=0.5, format="%.2f")

    if st.button("Compute Over%"):
        adjusted_mean = max(0.0, base_mean * def_factor)
        sd = auto_sd  # fixed by market — no user tuning

        sims = np.random.normal(loc=adjusted_mean, scale=sd, size=10000)
        sims = np.clip(sims, 0, None)

        over_p = float((sims > prop_line).mean())
        under_p = 1.0 - over_p

        st.write("---")
        st.subheader(f"{player_name} — {market}")
        opp_note = opp if opp else "Neutral"
        st.caption(
            f"Team: {team} | Baseline Y/G: {base_mean:.2f} | "
            f"Opponent: {opp_note} | Defense factor (from PA_pg): {def_factor:.2f} | "
            f"Adjusted mean: {adjusted_mean:.2f} | SD used: {sd:.1f}"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Over %", f"{over_p*100:.1f}%")
        c2.metric("Under %", f"{under_p*100:.1f}%")
        c3.metric("Fair ML (Over)", _fair_ml_from_prob(over_p))
