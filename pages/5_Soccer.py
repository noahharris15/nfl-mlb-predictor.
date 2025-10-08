# pages/5_Soccer.py
# Soccer Player Props — Odds API + soccerdata (FBref) (per-game means; 10k sims)
# Drop this file into your Streamlit "pages/" folder.
# Requires: soccerdata (FBref backend)

import math
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---- NEW: soccerdata (FBref) ----
try:
    from soccerdata import Fbref
except Exception as e:
    Fbref = None

# ---------------- UI / constants ----------------
st.set_page_config(page_title="Soccer Player Props — Odds API + soccerdata", layout="wide")
st.title("⚽ Soccer Player Props — Odds API + soccerdata (FBref)")

SIM_TRIALS = 10_000

VALID_MARKETS = [
    # Yes/No
    "player_goal_scorer_anytime",
    "player_first_goal_scorer",
    "player_last_goal_scorer",
    "player_to_receive_card",
    "player_to_receive_red_card",
    # O/U
    "player_shots_on_target",
    "player_shots",
    "player_assists",
]

# League mapping: (UI name, FBref code for soccerdata, Odds API sport key)
LEAGUES = [
    ("English Premier League",   "ENG-Premier League",     "soccer_epl"),
    ("La Liga (Spain)",          "ESP-La Liga",            "soccer_spain_la_liga"),
    ("Serie A (Italy)",          "ITA-Serie A",            "soccer_italy_serie_a"),
    ("Bundesliga (Germany)",     "GER-Bundesliga",         "soccer_germany_bundesliga"),
    ("Ligue 1 (France)",         "FRA-Ligue 1",            "soccer_france_ligue_one"),
    ("UEFA Champions League",    "UEFA-Champions League",  "soccer_uefa_champs_league"),
    ("MLS (USA)",                "USA-MLS",                "soccer_usa_mls"),
]

# ---------------- Helpers ----------------
def _norm_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining(c))

def _f(x) -> float:
    try: return float(x)
    except Exception: return float("nan")

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def poisson_yes(lam: float) -> float:
    lam = max(1e-6, float(lam))
    return 1.0 - math.exp(-lam)

def _sd_from_sums(sum_x, sum_x2, n, floor=0.1) -> float:
    if n <= 1: return float("nan")
    mean = sum_x / n
    var = max((sum_x2 / n) - mean**2, 0.0)
    var *= n / (n - 1)
    return float(max(var**0.5, floor))

def _season_list_for_dates(start_ymd: str, end_ymd: str) -> List[int]:
    """
    FBref seasons are labeled by the SPRING year (e.g., 2025 season spans ~Aug 2024–May 2025).
    We'll include both start.year and end.year to be safe.
    """
    s = datetime.strptime(start_ymd, "%Y/%m/%d").year
    e = datetime.strptime(end_ymd, "%Y/%m/%d").year
    years = list(range(min(s, e), max(s, e)+1))
    # Also include +1 to capture spring if spanning autumn → spring
    if e == s:
        years.append(e+1)
    return sorted(list(dict.fromkeys(years)))

# ---------------- soccerdata / FBref loader ----------------
@st.cache_data(show_spinner=True)
def build_soccer_means_fbref(fbref_league: str, start_slash: str, end_slash: str) -> pd.DataFrame:
    """
    Uses soccerdata.Fbref to pull player match logs for the league across seasons that
    cover the desired date range, filters by date, and aggregates per-player per-game means/SDs.
    """
    if Fbref is None:
        st.error("soccerdata not installed. Add 'soccerdata' to your requirements and restart.")
        return pd.DataFrame()

    # Seasons to request (see helper above)
    seasons = _season_list_for_dates(start_slash, end_slash)

    # Read player match stats (standard table) → includes: date, player, team, gls, ast, sh, sog, crdy, crdr, etc.
    try:
        fb = Fbref(leagues=[fbref_league], seasons=seasons, data_dir=None)
        df = fb.read_player_match_stats(stat_type="standard")
    except Exception as e:
        st.error(f"FBref read failed for {fbref_league}, seasons {seasons}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize column names we need
    # Typical FBref columns: 'date', 'player', 'team', 'gls', 'ast', 'sh', 'sot', 'crdy', 'crdr'
    # Some installs may use full names; standardize to expected set.
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    col_date  = pick("date")
    col_player= pick("player")
    col_team  = pick("team","squad")
    col_goals = pick("gls","goals")
    col_ast   = pick("ast","assists")
    col_sh    = pick("sh","shots")
    col_sot   = pick("sot","shots_on_target","sot.1")
    col_yc    = pick("crdy","cards_yellow","yellow_cards")
    col_rc    = pick("crdr","cards_red","red_cards")
    col_min   = pick("min","minutes")

    need = [col_date, col_player, col_goals, col_ast, col_sh, col_sot, col_yc, col_rc]
    if any(x is None for x in need):
        st.error("FBref schema missing expected columns (date/player/goals/assists/shots/SoT/YC/RC).")
        return pd.DataFrame()

    df1 = df[[col_date, col_player, col_goals, col_ast, col_sh, col_sot, col_yc, col_rc] + ([col_team] if col_team else [])].copy()
    df1.columns = ["date","player","goals","assists","shots","sog","yc","rc"] + (["team"] if col_team else [])
    # Filter by date range
    df1["date"] = pd.to_datetime(df1["date"], errors="coerce")
    sdt = datetime.strptime(start_slash, "%Y/%m/%d")
    edt = datetime.strptime(end_slash, "%Y/%m/%d") + timedelta(days=1)
    df1 = df1[(df1["date"] >= sdt) & (df1["date"] < edt)].copy()
    if df1.empty:
        return pd.DataFrame()

    # Coerce numerics
    for c in ["goals","assists","shots","sog","yc","rc"]:
        df1[c] = pd.to_numeric(df1[c], errors="coerce").fillna(0.0)

    # Normalize names
    df1["Player"] = df1["player"].astype(str).map(_norm_name)

    # Build sums, sumsqs, and game count (appearance if any stat > 0)
    totals: Dict[str, Dict[str, float]] = {}
    sumsqs: Dict[str, Dict[str, float]] = {}
    games: Dict[str, int] = {}

    def pinit(p):
        if p not in totals:
            totals[p] = {"goals":0.0,"assists":0.0,"shots":0.0,"sog":0.0,"yc":0.0,"rc":0.0}
            sumsqs[p] = {k:0.0 for k in totals[p]}
            games[p]  = 0

    for _, r in df1.iterrows():
        p = r["Player"]; pinit(p)
        if any(float(r[k]) > 0 for k in totals[p].keys()):
            games[p] += 1
        for k in totals[p]:
            v = float(r.get(k, 0.0))
            totals[p][k] += v
            sumsqs[p][k] += v*v

    rows = []
    for p, sums in totals.items():
        g = max(1, games.get(p, 0))
        row = {"Player": p, "g": g}
        for k, s in sums.items():
            row[f"mu_{k}"] = s / g
            row[f"sd_{k}"] = _sd_from_sums(s, sumsqs[p][k], g, floor=0.25 if k in ["goals","assists"] else 0.6)
        rows.append(row)

    return pd.DataFrame(rows)

# ---------------- Odds API helpers ----------------
def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_soccer_events_odds(api_key: str, sport_key: str, lookahead_days: int, region: str):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/events",
        {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region},
    )

def fetch_event_props(api_key: str, sport_key: str, event_id: str, region: str, markets: List[str]):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"},
    )

# ---------------- UI: league + date range ----------------
st.header("1) League & Date Range")
col0, col1, col2 = st.columns([1.6,1,1])
with col0:
    league = st.selectbox("League", LEAGUES, format_func=lambda x: x[0])
    LEAGUE_NAME, FBREF_LEAGUE, ODDS_KEY = league
with col1:
    start_date = st.text_input("Start (YYYY/MM/DD)", value=datetime.now().strftime("%Y/%m/01"))
with col2:
    end_date   = st.text_input("End (YYYY/MM/DD)",   value=datetime.now().strftime("%Y/%m/%d"))

# ---------------- Build projections ----------------
st.header("2) Build per-player averages from soccerdata / FBref")
if st.button("📥 Build Soccer projections"):
    soc = build_soccer_means_fbref(FBREF_LEAGUE, start_date, end_date)
    if soc.empty:
        st.error("No data returned from soccerdata/FBref for this league/date window.")
        st.stop()
    # Store
    st.session_state["soc_proj"] = soc.copy()

    # Preview: raw μ table
    with st.expander("Preview — Per-game averages (μ) & σ", expanded=False):
        cols = ["Player","g",
                "mu_goals","sd_goals",
                "mu_assists","sd_assists",
                "mu_shots","sd_shots",
                "mu_sog","sd_sog",
                "mu_yc","sd_yc","mu_rc","sd_rc"]
        view = [c for c in cols if c in soc.columns]
        st.dataframe(soc[view].sort_values("mu_goals", ascending=False).head(50), use_container_width=True)
    st.success(f"Built projections for {len(soc)} players.")

# ---------------- Odds API: pick match + markets ----------------
st.header("3) Pick a match & markets from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=2)
markets = st.multiselect("Markets to fetch", VALID_MARKETS, default=VALID_MARKETS)

events = []
if api_key:
    try:
        events = list_soccer_events_odds(api_key, ODDS_KEY, lookahead, region)
    except Exception as e:
        st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key to list matches in this league.")
    st.stop()

labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Match", labels)
event = events[labels.index(pick)]
event_id = event["id"]

# ---------------- Simulate ----------------
st.header("4) Fetch lines & simulate")
if st.button("🎲 Fetch lines & simulate (Soccer)"):
    proj = st.session_state.get("soc_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build Soccer projections first (Step 2).")
        st.stop()

    # index by normalized name
    proj = proj.copy()
    proj["PN"] = proj["Player"].apply(_norm_name)
    proj.set_index("PN", inplace=True)

    try:
        odds = fetch_event_props(api_key, ODDS_KEY, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    rows = []
    for bk in odds.get("bookmakers", []):
        for m in bk.get("markets", []):
            key = m.get("key")
            for o in m.get("outcomes", []):
                name = _norm_name(o.get("description"))
                side = o.get("name")
                point = o.get("point")
                if key not in VALID_MARKETS or not name or not side:
                    continue
                rows.append({
                    "market": key,
                    "player": name,
                    "side": side,
                    "point": None if point is None else float(point),
                })
    if not rows:
        st.warning("No player outcomes returned for these markets.")
        st.stop()

    props = (pd.DataFrame(rows)
             .groupby(["market","player","side"], as_index=False)
             .agg(line=("point","median"), books=("point","size")))

    out = []

    for _, r in props.iterrows():
        pl, mkt, side, line = r["player"], r["market"], r["side"], r["line"]
        if pl not in proj.index:
            continue
        pr = proj.loc[pl]

        # Yes/No markets
        if mkt == "player_goal_scorer_anytime":
            lam = float(pr.get("mu_goals", np.nan))
            if np.isnan(lam): continue
            p_yes = poisson_yes(lam)
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            out.append({"market": mkt, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,3), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue

        if mkt in ("player_first_goal_scorer", "player_last_goal_scorer"):
            lam = float(pr.get("mu_goals", np.nan))
            if np.isnan(lam): continue
            p_any = poisson_yes(lam)
            frac = 0.25  # conservative share
            p = p_any * frac
            p = p if side in ("Yes","Over") else (1.0 - p)
            out.append({"market": mkt, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,3), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue

        if mkt == "player_to_receive_card":
            lam = float(pr.get("mu_yc", np.nan))
            if np.isnan(lam): continue
            p_yes = float(np.clip(lam, 0.0, 0.95))  # simple Bernoulli with cap
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            out.append({"market": mkt, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,3), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue

        if mkt == "player_to_receive_red_card":
            lam = float(pr.get("mu_rc", np.nan))
            if np.isnan(lam): continue
            p_yes = float(np.clip(lam, 0.0, 0.50))  # reds are rare; clamp to 50%
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            out.append({"market": mkt, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,3), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue

        # Over/Under markets
        if mkt == "player_shots_on_target":
            mu, sd = float(pr.get("mu_sog", np.nan)), float(pr.get("sd_sog", np.nan))
        elif mkt == "player_shots":
            mu, sd = float(pr.get("mu_shots", np.nan)), float(pr.get("sd_shots", np.nan))
        elif mkt == "player_assists":
            mu, sd = float(pr.get("mu_assists", np.nan)), float(pr.get("sd_assists", np.nan))
        else:
            continue

        if line is None or np.isnan(mu) or np.isnan(sd):
            continue
        p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
        p = p_over if side == "Over" else (1.0 - p_over)

        out.append({"market": mkt, "player": pl, "side": side, "line": float(line),
                    "μ (per-game)": round(mu,2), "σ (per-game)": round(sd,2),
                    "Win Prob %": round(100*p,2), "books": int(r["books"])})

    if not out:
        st.warning("No matched props to simulate.")
        st.stop()

    results = (pd.DataFrame(out)
               .drop_duplicates(subset=["market","player","side"])
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    cfg = {
        "player": st.column_config.TextColumn("Player", width="large"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.2f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "books": st.column_config.NumberColumn("#Books", width="small"),
    }
    st.subheader(f"Results — {LEAGUE_NAME}")
    st.dataframe(results, use_container_width=True, hide_index=True, column_config=cfg)
    st.download_button(
        "⬇️ Download Soccer results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="soccer_props_results.csv",
        mime="text/csv",
    )
