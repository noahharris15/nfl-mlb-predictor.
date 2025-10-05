# app.py — NFL & MLB sims + Player Props (Odds API + your CSV stats)
import json
import math
import time
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =============== App config ===============
st.set_page_config(page_title="NFL/MLB Sims + Player Props (Odds API + CSVs)", layout="wide")
st.title("🏈⚾ NFL & MLB Sims + Player Props (Odds API + Your CSV Stats)")

SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6          # small home boost in NFL Poisson sim
EPS = 1e-9

# =============== Odds API (event-odds endpoint) ===============
DEFAULT_ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "")
VALID_MARKETS = [
    "player_pass_tds",
    "player_rush_yds",
    "player_receptions",
    "player_anytime_td",
]

def odds_headers() -> Dict[str, str]:
    return {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def fetch_events(sport_key: str, api_key: str, days_ahead: int, region: str = "us") -> pd.DataFrame:
    """
    List upcoming events to let user pick a single event for props.
    The 'events' endpoint is cheap; then we use the 'event-odds' endpoint for props.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    params = {
        "apiKey": api_key,
        "regions": region,
        "dateFormat": "iso",
        "daysFrom": 0,
        "daysTo": max(0, int(days_ahead)),
        "oddsFormat": "decimal",
    }
    r = requests.get(url, params=params, headers=odds_headers(), timeout=25)
    r.raise_for_status()
    data = r.json()
    rows = []
    for ev in data:
        rows.append({
            "id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
        })
    return pd.DataFrame(rows)

def fetch_event_props(
    sport_key: str,
    event_id: str,
    api_key: str,
    markets: List[str],
    region: str = "us",
) -> dict:
    """
    Pull props for ONE event using the event-odds endpoint.
    Only uses markets you select (must be valid for your plan). Returns raw JSON.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "regions": region,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
        "apiKey": api_key,
    }
    r = requests.get(url, params=params, headers=odds_headers(), timeout=30)
    # If a market is invalid for your plan, Odds API returns 422 INVALID_MARKET
    r.raise_for_status()
    return r.json()

# =============== Simple Poisson helpers ===============
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

# =============== NFL team rates ======================
# We keep the team Poisson model as before, built from results if available or a neutral prior.
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    import nfl_data_py as nfl
    sched = nfl.import_schedules([2025])
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

    # upcoming slate
    date_col = None
    for c in ("gameday", "game_date"):
        if c in sched.columns:
            date_col = c; break
    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][
        ["home_team", "away_team"] + ([date_col] if date_col else [])
    ].copy()
    if date_col:
        upcoming = upcoming.rename(columns={date_col: "date"})
    else:
        upcoming["date"] = ""
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

# =============== MLB team RS/RA ======================
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

@st.cache_data(show_spinner=False)
def mlb_team_rates_2025():
    from pybaseball import schedule_and_record
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
                RS_pg = float(sar["R"].sum() / len(sar))
                RA_pg = float(sar["RA"].sum() / len(sar))
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean()); league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9 * df["RS_pg"] + 0.1 * league_rs
        df["RA_pg"] = 0.9 * df["RA_pg"] + 0.1 * league_ra
    return df

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

# =============== Defense EPA table (embedded + optional override) ===============
# Columns: Team,EPA_Pass,EPA_Rush
# NOTE: Values are seeded neutrally (0). You can paste your new table into the textarea to override at runtime.
_DEF_EPA_CSV = """Team,EPA_Pass,EPA_Rush
Arizona Cardinals,0.00,0.00
Atlanta Falcons,0.00,0.00
Baltimore Ravens,0.00,0.00
Buffalo Bills,0.00,0.00
Carolina Panthers,0.00,0.00
Chicago Bears,0.00,0.00
Cincinnati Bengals,0.00,0.00
Cleveland Browns,0.00,0.00
Dallas Cowboys,0.00,0.00
Denver Broncos,0.00,0.00
Detroit Lions,0.00,0.00
Green Bay Packers,0.00,0.00
Houston Texans,0.00,0.00
Indianapolis Colts,0.00,0.00
Jacksonville Jaguars,0.00,0.00
Kansas City Chiefs,0.00,0.00
Las Vegas Raiders,0.00,0.00
Los Angeles Chargers,0.00,0.00
Los Angeles Rams,0.00,0.00
Miami Dolphins,0.00,0.00
Minnesota Vikings,0.00,0.00
New England Patriots,0.00,0.00
New Orleans Saints,0.00,0.00
New York Giants,0.00,0.00
New York Jets,0.00,0.00
Philadelphia Eagles,0.00,0.00
Pittsburgh Steelers,0.00,0.00
San Francisco 49ers,0.00,0.00
Seattle Seahawks,0.00,0.00
Tampa Bay Buccaneers,0.00,0.00
Tennessee Titans,0.00,0.00
Washington Commanders,0.00,0.00
"""

def load_def_epa(csv_text: str = _DEF_EPA_CSV) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    df["Team"] = df["Team"].astype(str)
    for c in ("EPA_Pass", "EPA_Rush"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    return df

def defense_scalers(team_name: str, def_df: pd.DataFrame) -> Dict[str, float]:
    """
    Convert EPA (+better offense) to a defensive *multiplier* for yardage/TDs.
    Negative EPA (better defense) -> <1 scaler. We keep it very conservative.
    """
    r = def_df.loc[def_df["Team"].str.lower() == str(team_name).lower()]
    if r.empty:
        return {"pass": 1.0, "rush": 1.0, "rec": 1.0}
    ep = float(r.iloc[0]["EPA_Pass"])
    er = float(r.iloc[0]["EPA_Rush"])
    # Very conservative mapping: +/-0.20 EPA ~ +/-10% scale
    sp = float(np.clip(1.0 + ep * 0.5, 0.80, 1.20))
    sr = float(np.clip(1.0 + er * 0.5, 0.80, 1.20))
    return {"pass": sp, "rush": sr, "rec": sp}

# =============== CSV parsing for player tables ===============
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    # find first line that looks like a header
    lines = raw.strip().splitlines()
    header_idx = 0
    for i, ln in enumerate(lines[:20]):
        if "Player" in ln and "Y/G" in ln:
            header_idx = i
            break
    cleaned = "\n".join(lines[header_idx:])
    return pd.read_csv(StringIO(cleaned))

# Build simple per-game projections (μ, σ) with optional defense scaler
def qb_proj_row(row: pd.Series, def_scale_pass: float) -> dict:
    g = row.get("G")
    ypg = row.get("Y/G")
    td = row.get("TD")
    try:
        base_yards = float(ypg)
    except Exception:
        try:
            base_yards = float(row.get("Yds")) / max(1.0, float(g))
        except Exception:
            base_yards = 225.0
    try:
        base_tds = float(td) / max(1.0, float(g))
    except Exception:
        base_tds = 1.5
    mu_y = base_yards * def_scale_pass
    mu_td = base_tds * def_scale_pass**0.7
    sd_y = max(8.0, 0.18 * mu_y)
    sd_td = max(0.25, 0.55 * mu_td)
    return {
        "Player": row.get("Player"), "Team": row.get("Team"),
        "mu_pass_yds": mu_y, "sd_pass_yds": sd_y,
        "mu_pass_tds": mu_td, "sd_pass_tds": sd_td,
    }

def rb_proj_row(row: pd.Series, def_scale_rush: float) -> dict:
    g = row.get("G")
    ypg = row.get("Y/G")
    td = row.get("TD")
    try:
        base_yards = float(ypg)
    except Exception:
        try:
            base_yards = float(row.get("Yds")) / max(1.0, float(g))
        except Exception:
            base_yards = 60.0
    try:
        base_tds = float(td) / max(1.0, float(g))
    except Exception:
        base_tds = 0.5
    mu_y = base_yards * def_scale_rush
    mu_td = base_tds * def_scale_rush**0.7
    sd_y = max(6.0, 0.22 * mu_y)
    sd_td = max(0.2, 0.65 * mu_td)
    return {
        "Player": row.get("Player"), "Team": row.get("Team"),
        "mu_rush_yds": mu_y, "sd_rush_yds": sd_y,
        "mu_rush_tds": mu_td, "sd_rush_tds": sd_td,
    }

def wr_proj_row(row: pd.Series, def_scale_rec: float) -> dict:
    g = row.get("G")
    ypg = row.get("Y/G")
    try:
        base_yards = float(ypg)
    except Exception:
        try:
            base_yards = float(row.get("Yds")) / max(1.0, float(g))
        except Exception:
            base_yards = 55.0
    # receptions per game if present
    rec_pg = row.get("Rec")
    try:
        rec_pg = float(rec_pg) / max(1.0, float(g))
    except Exception:
        try:
            rec_pg = float(row.get("Rec/G"))
        except Exception:
            rec_pg = 4.2
    mu_y = base_yards * def_scale_rec
    mu_rec = rec_pg * np.clip(def_scale_rec, 0.85, 1.15)
    sd_y = max(6.0, 0.20 * mu_y)
    sd_rec = max(0.8, 0.35 * mu_rec)
    return {
        "Player": row.get("Player"), "Team": row.get("Team"),
        "mu_rec_yds": mu_y, "sd_rec_yds": sd_y,
        "mu_receptions": mu_rec, "sd_receptions": sd_rec,
    }

def norm_over_prob(mu: float, sd: float, line: float) -> float:
    sd = max(1e-6, float(sd))
    z = (line - mu) / sd
    # P(X > line) = 1 - Phi(z)
    return float(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))

# =============== Sidebar navigation ===============
page = st.sidebar.radio("Pages", ["NFL Sim", "MLB Sim", "Player Props"], index=0)

# ======================================================================================
# NFL SIM PAGE
# ======================================================================================
if page == "NFL Sim":
    st.subheader("🏈 NFL Poisson Simulation (2025)")

    rates, upcoming = nfl_team_rates_2025()
    if upcoming.empty:
        st.info("No upcoming NFL games found yet.")
        st.stop()

    # Pretty picker
    choices = []
    for _, r in upcoming.iterrows():
        label = f"{r['home_team']} vs {r['away_team']}"
        if str(r.get('date', '')).strip():
            label += f" — {r['date']}"
        choices.append(label)
    pick = st.selectbox("Matchup", choices)
    pair = pick.split(" — ")[0] if " — " in pick else pick
    home, away = pair.split(" vs ")

    mu_h, mu_a = nfl_matchup_mu(rates, home, away)
    p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win %", f"{p_home*100:.1f}%")
    c2.metric(f"{away} win %", f"{p_away*100:.1f}%")
    c3.metric("Expected total", f"{exp_t:.1f}")
    st.caption(f"Expected: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")

# ======================================================================================
# MLB SIM PAGE
# ======================================================================================
elif page == "MLB Sim":
    st.subheader("⚾ MLB Poisson Simulation (2025)")

    rates = mlb_team_rates_2025()
    teams = rates["team"].sort_values().tolist()
    if not teams:
        st.info("No MLB team data yet.")
        st.stop()

    home = st.selectbox("Home team", teams, index=0, key="mlb_home")
    away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")

    mu_h, mu_a = mlb_matchup_mu(rates, home, away)
    p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win %", f"{p_home*100:.1f}%")
    c2.metric(f"{away} win %", f"{p_away*100:.1f}%")
    c3.metric("Expected total", f"{exp_t:.1f}")
    st.caption(f"Expected: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")

# ======================================================================================
# PLAYER PROPS PAGE
# ======================================================================================
else:
    st.subheader("📈 Player Props — Odds API lines + Your CSV stats")

    # --- Odds API controls ---
    st.markdown("#### 1) Odds settings")
    colA, colB = st.columns([2, 1])
    with colA:
        api_key = st.text_input("Odds API key", value=DEFAULT_ODDS_API_KEY, type="password")
    with colB:
        region = st.selectbox("Region", ["us", "us2", "eu", "uk"], index=0)

    sport = st.selectbox("Sport (for props)", ["americanfootball_nfl"], index=0)
    days = st.slider("Lookahead days for events", 0, 7, value=1)
    markets = st.multiselect("Markets (valid for your plan)", VALID_MARKETS, default=VALID_MARKETS)

    # --- Defense EPA controls ---
    st.markdown("#### 2) Defense adjustment (optional)")
    use_def = st.toggle("Apply defense adjustment by EPA (embedded)", value=False)
    def_df = load_def_epa(_DEF_EPA_CSV)
    with st.expander("Paste/override defense EPA CSV (Team,EPA_Pass,EPA_Rush)", expanded=False):
        txt = st.text_area("CSV text", value=_DEF_EPA_CSV, height=180)
        if st.button("Use pasted EPA table"):
            try:
                def_df = load_def_epa(txt)
                st.success("Defense EPA table updated.")
            except Exception as e:
                st.error(f"EPA CSV parse error: {e}")

    # --- Player CSVs ---
    st.markdown("#### 3) Upload player CSVs (QB / RB / WR) — we’ll use your per-game stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        qb_file = st.file_uploader("QB CSV", type=["csv"])
    with col2:
        rb_file = st.file_uploader("RB CSV", type=["csv"])
    with col3:
        wr_file = st.file_uploader("WR CSV", type=["csv"])

    qb_proj = rb_proj = wr_proj = None

    # we need opponent team name to get defense scaler per position — grab from selected event
    st.markdown("#### 4) Pick an event")
    events_df = pd.DataFrame()
    if api_key:
        try:
            events_df = fetch_events(sport, api_key, days, region)
        except requests.HTTPError as e:
            st.error(f"Events fetch error: {e}")
    else:
        st.info("Enter your Odds API key to list events.")

    event_label = None
    event_id = None
    home_team = away_team = None
    if not events_df.empty:
        events_df["label"] = events_df["home_team"] + " @ " + events_df["away_team"] + " — " + events_df["commence_time"].astype(str)
        event_label = st.selectbox("Event", events_df["label"].tolist())
        ev_row = events_df.loc[events_df["label"] == event_label].iloc[0]
        event_id = ev_row["id"]
        home_team, away_team = ev_row["home_team"], ev_row["away_team"]

    # Build projections from CSVs (apply defense scalers if toggled and opponent known)
    def_scale_pass = def_scale_rush = def_scale_rec = 1.0
    if use_def and (home_team and away_team):
        # For the sake of adjustment we let the user pick which defense to scale against:
        # If the player plays for 'away_team', we scale by HOME defense, etc.
        st.caption("Select which **defense** to adjust against (who the player faces).")
        opp_choice = st.selectbox("Opponent defense", [home_team, away_team])
        scales = defense_scalers(opp_choice, def_df)
        def_scale_pass, def_scale_rush, def_scale_rec = scales["pass"], scales["rush"], scales["rec"]
        with st.expander("Current defense scalers"):
            st.write(scales)

    # Parse CSVs and compute μ, σ
    if qb_file is not None:
        try:
            qb = _load_any_csv(qb_file)
            qb = _coerce_numeric(qb, ["G", "Y/G", "Yds", "TD", "Att", "Cmp", "Rate"])
            rows = [qb_proj_row(r, def_scale_pass) for _, r in qb.iterrows()]
            qb_proj = pd.DataFrame(rows)
            st.markdown("**QB projections**")
            st.dataframe(qb_proj, use_container_width=True, height=260)
        except Exception as e:
            st.error(f"QB CSV error: {e}")

    if rb_file is not None:
        try:
            rb = _load_any_csv(rb_file)
            rb = _coerce_numeric(rb, ["G", "Y/G", "Yds", "TD", "Att"])
            rows = [rb_proj_row(r, def_scale_rush) for _, r in rb.iterrows()]
            rb_proj = pd.DataFrame(rows)
            st.markdown("**RB projections**")
            st.dataframe(rb_proj, use_container_width=True, height=260)
        except Exception as e:
            st.error(f"RB CSV error: {e}")

    if wr_file is not None:
        try:
            wr = _load_any_csv(wr_file)
            wr = _coerce_numeric(wr, ["G", "Y/G", "Yds", "Rec", "Rec/G"])
            rows = [wr_proj_row(r, def_scale_rec) for _, r in wr.iterrows()]
            wr_proj = pd.DataFrame(rows)
            st.markdown("**WR projections**")
            st.dataframe(wr_proj, use_container_width=True, height=260)
        except Exception as e:
            st.error(f"WR CSV error: {e}")

    st.markdown("#### 5) Fetch Odds API **player props** for this event and simulate")
    go = st.button("Fetch lines & simulate", disabled=(not api_key or not event_id or not markets))
    if go:
        try:
            raw = fetch_event_props(sport, event_id, api_key, markets, region)
        except requests.HTTPError as e:
            st.error(f"Props fetch failed: {e}")
            st.stop()

        # Flatten bookmaker markets → a dataframe of (book, market, player, line)
        rows = []
        for bk in raw.get("bookmakers", []):
            bk_key = bk.get("key")
            for m in bk.get("markets", []):
                mkey = m.get("key")
                if mkey not in markets:
                    continue
                outs = m.get("outcomes", [])
                # group O/U pairs by player (description)
                # We'll average the 'point' over all books & keep the player name
                for o in outs:
                    name = o.get("name")   # Over/Under or player for anytime_td
                    player = o.get("description") if o.get("description") else o.get("name")
                    point = o.get("point")
                    rows.append({
                        "book": bk_key, "market": mkey,
                        "player": player, "name": name, "point": point
                    })
        props_df = pd.DataFrame(rows)
        if props_df.empty:
            st.warning("No props found for this event with the selected markets.")
            st.stop()

        # Build consensus line per (player, market) → average of available 'point'
        # For anytime TDs there's no 'point' — we treat line as 0.5 TDs for a yes/no style (Over = scores).
        def consensus_line(g: pd.DataFrame, market: str) -> float:
            pts = pd.to_numeric(g["point"], errors="coerce")
            if market == "player_anytime_td" or pts.isna().all():
                return 0.5
            return float(pts.mean())

        grouped = (
            props_df.groupby(["player", "market"], as_index=False)
            .apply(lambda g: pd.Series({"line": consensus_line(g, g["market"].iloc[0])}))
            .reset_index(drop=True)
        )

        # Match to our projections and compute Over/Under probs
        sim_rows = []
        for _, r in grouped.iterrows():
            player = str(r["player"]).strip()
            market = r["market"]
            line = float(r["line"])

            mu = sd = None
            if market == "player_pass_tds" and qb_proj is not None:
                m = qb_proj.loc[qb_proj["Player"].astype(str).str.lower() == player.lower()]
                if not m.empty:
                    mu = float(m.iloc[0]["mu_pass_tds"]); sd = float(m.iloc[0]["sd_pass_tds"])
            elif market == "player_rush_yds" and rb_proj is not None:
                m = rb_proj.loc[rb_proj["Player"].astype(str).str.lower() == player.lower()]
                if not m.empty:
                    mu = float(m.iloc[0]["mu_rush_yds"]); sd = float(m.iloc[0]["sd_rush_yds"])
            elif market == "player_receptions" and wr_proj is not None:
                m = wr_proj.loc[wr_proj["Player"].astype(str).str.lower() == player.lower()]
                if not m.empty:
                    mu = float(m.iloc[0]["mu_receptions"]); sd = float(m.iloc[0]["sd_receptions"])
            elif market == "player_anytime_td":
                # crude mapping: use best available TD μ (QB pass TD doesn't score for QB; we skip QBs)
                m_rb = rb_proj.loc[rb_proj["Player"].astype(str).str.lower() == player.lower()] if rb_proj is not None else pd.DataFrame()
                mu = float(m_rb.iloc[0]["mu_rush_tds"]) if not m_rb.empty else None
                sd = 0.8 if mu is not None else None

            if mu is None or sd is None or math.isnan(mu) or math.isnan(sd):
                continue

            over_p = norm_over_prob(mu, sd, line)
            sim_rows.append({
                "player": player,
                "market": market,
                "line": round(line, 3),
                "mu": round(mu, 3),
                "sd": round(sd, 3),
                "P(Over)%": round(over_p * 100, 2),
                "P(Under)%": round((1 - over_p) * 100, 2),
            })

        results = pd.DataFrame(sim_rows).sort_values("P(Over)%", ascending=False)
        if results.empty:
            st.warning("No matches between Odds API players and your CSVs. Check name spellings (they must match).")
        else:
            st.success(f"Simulated {len(results)} player markets.")
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download results CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="player_props_sim.csv",
                mime="text/csv",
            )
