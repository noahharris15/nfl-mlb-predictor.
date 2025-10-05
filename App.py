# app.py — NFL Player Props (All games, Odds API + embedded Defense EPA), single page
import math
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------- UI & page config ----------------------------
st.set_page_config(page_title="NFL Player Props — Odds API + Defense EPA", layout="wide")
st.title("🏈 NFL Player Props — Odds API + Defense EPA (All Games)")

# ---------------------------- Inputs ----------------------------
with st.sidebar:
    st.subheader("Settings")
    api_key = st.text_input("The Odds API key", value="", type="password", help="Paste your key here.")
    lookahead_days = st.slider("Lookahead days for events", 0, 7, 1)
    region = st.selectbox("Region", ["us", "us2", "us_il"], index=0)
    bookmakers = st.multiselect(
        "Bookmakers (average line across selected; leave empty = all)",
        ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus", "barstool", "betrivers"],
        default=["draftkings","fanduel","betmgm","caesars"]
    )
    markets = st.multiselect(
        "Markets to pull",
        ["player_pass_yds","player_rush_yds","player_rec_yds","player_receptions","player_anytime_td"],
        default=["player_pass_yds","player_rush_yds","player_rec_yds","player_receptions","player_anytime_td"]
    )
    trials = st.number_input("Simulation trials", 2000, 20000, 8000, step=1000)
    st.caption("We simulate with a conservative normal model for yards/rec; TD is Bernoulli.")
    run_btn = st.button("Fetch ALL games & simulate", use_container_width=True)

# ---------------------------- Embedded Defense EPA (your table) ----------------------------
# Columns in your sheet (for offense-vs-defense scaling we use EPA/Pass and EPA/Rush):
# Team | Season | EPA/Play | Total EPA | Success % | EPA/Pass | EPA/Rush | Pass Yards | Comp % | Pass TD | Rush Yards | Rush TD | ADoT | Sack % | Scramble % | Int %
_DEF_EPA_ROWS = [
    # team,           epa_pass, epa_rush  (2025 values from your screenshot)
    ("Minnesota Vikings",        -0.37,  0.06),
    ("Jacksonville Jaguars",     -0.17, -0.05),
    ("Denver Broncos",           -0.10, -0.12),
    ("Los Angeles Chargers",     -0.17,  0.01),
    ("Detroit Lions",             0.00, -0.22),
    ("Philadelphia Eagles",      -0.11, -0.04),
    ("Houston Texans",           -0.16,  0.04),
    ("Los Angeles Rams",         -0.12,  0.00),
    ("Seattle Seahawks",          0.00, -0.19),
    ("San Francisco 49ers",      -0.02, -0.11),
    ("Tampa Bay Buccaneers",     -0.13,  0.05),
    ("Atlanta Falcons",           0.06, -0.17),
    ("Cleveland Browns",         -0.04, -0.05),
    ("Indianapolis Colts",       -0.09,  0.09),
    ("Kansas City Chiefs",       -0.09,  0.11),
    ("Arizona Cardinals",         0.06, -0.14),
    ("Las Vegas Raiders",         0.14, -0.22),
    ("Green Bay Packers",         0.03, -0.07),
    ("Chicago Bears",             0.01,  0.01),
    ("Buffalo Bills",             0.06,  0.06),
    ("Carolina Panthers",         0.05,  0.05),
    ("Pittsburgh Steelers",       0.10,  0.00),
    ("Washington Commanders",     0.18, -0.12),
    ("New England Patriots",     -0.01,  0.19),
    ("New York Giants",           0.20, -0.06),
    ("New Orleans Saints",        0.20, -0.06),
    ("Cincinnati Bengals",        0.13,  0.04),
    ("New York Jets",             0.23, -0.03),
    ("Tennessee Titans",          0.16,  0.12),
    ("Baltimore Ravens",          0.40,  0.06),
    ("Dallas Cowboys",            0.34,  0.12),
    ("Miami Dolphins",            0.34,  0.12),  # from your last rows (rounded)
]

DEF_EPA = pd.DataFrame(_DEF_EPA_ROWS, columns=["team","epa_pass","epa_rush"])

# scaling -> convert EPA to multiplicative adjustment around 1.0
# EPA range is about [-0.4, +0.4]; alpha tunes strength of effect.
ALPHA_PASS = 0.80   # bigger effect for passing (receptions/rec_yds also use pass EPA)
ALPHA_RUSH = 0.75

def pass_adj(team: str) -> float:
    r = DEF_EPA.loc[DEF_EPA["team"].str.lower()==team.lower()]
    if r.empty: return 1.0
    e = float(r.iloc[0]["epa_pass"])
    return float(np.clip(1.0 + ALPHA_PASS*e, 0.6, 1.6))

def rush_adj(team: str) -> float:
    r = DEF_EPA.loc[DEF_EPA["team"].str.lower()==team.lower()]
    if r.empty: return 1.0
    e = float(r.iloc[0]["epa_rush"])
    return float(np.clip(1.0 + ALPHA_RUSH*e, 0.6, 1.6))

# ---------------------------- Helpers: Odds API ----------------------------
ODDS_BASE = "https://api.the-odds-api.com/v4"

def _get(url: str, params: dict, tries=3) -> requests.Response:
    last = None
    for _ in range(tries):
        r = requests.get(url, params=params, timeout=25)
        if r.status_code == 200: return r
        last = r
        time.sleep(0.8)
    if last is None:
        raise RuntimeError("No response from Odds API.")
    last.raise_for_status()
    return last

def list_events(api_key: str, region: str, days: int) -> List[dict]:
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/events"
    params = {"apiKey": api_key, "regions": region, "daysFrom": days}
    r = _get(url, params)
    return r.json()

def game_odds(api_key: str, event_id: str, region: str, markets: List[str], bookmakers: List[str]) -> dict:
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    r = _get(url, params)
    return r.json()

# ---------------------------- Player → Team map (nfl_data_py) ----------------------------
@st.cache_data(show_spinner=False)
def build_player_team_map() -> Dict[str, str]:
    """Return {clean_player_name -> full team name} for the current season roster."""
    import nfl_data_py as nfl
    for yr in [2025, 2024]:
        try:
            df = nfl.import_seasonal_data([yr])
            break
        except Exception:
            df = None
    if df is None or df.empty:
        return {}
    tmp = df[["player_display_name","recent_team"]].dropna().drop_duplicates()
    # map NFL abbreviations to full names used by Odds API
    abbr_to_full = {
        "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
        "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
        "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GNB":"Green Bay Packers","GB":"Green Bay Packers",
        "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","JAC":"Jacksonville Jaguars",
        "KAN":"Kansas City Chiefs","KC":"Kansas City Chiefs","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams",
        "LVR":"Las Vegas Raiders","LV":"Las Vegas Raiders","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
        "NWE":"New England Patriots","NE":"New England Patriots","NOR":"New Orleans Saints","NO":"New Orleans Saints",
        "NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers",
        "SEA":"Seattle Seahawks","SFO":"San Francisco 49ers","SF":"San Francisco 49ers",
        "TAM":"Tampa Bay Buccaneers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders",
    }
    tmp["team_full"] = tmp["recent_team"].map(abbr_to_full)
    tmp = tmp.dropna(subset=["team_full"])
    def _clean(s): return str(s).replace(".","").replace("-"," ").lower().strip()
    mapping = {_clean(n): t for n,t in zip(tmp["player_display_name"], tmp["team_full"])}
    return mapping

PLAYER_TO_TEAM = build_player_team_map()

def clean_name(s: str) -> str:
    return str(s).replace(".","").replace("-"," ").lower().strip()

# ---------------------------- Simulation primitives ----------------------------
SD_DEFAULTS = {
    "player_pass_yds": 38.0,
    "player_rush_yds": 22.0,
    "player_rec_yds": 24.0,
    "player_receptions": 1.35,
}

def normal_over_prob(mu: float, sd: float, line: float, n: int) -> float:
    sd = max(1e-6, sd)
    return float((np.random.normal(mu, sd, size=n) > line).mean())

def american_to_prob(odds: float) -> float:
    # +120 -> 0.4545, -120 -> 0.5455 (no vigorish adjustment)
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

# ---------------------------- Core processing ----------------------------
def aggregate_lines(outcomes: List[dict]) -> Dict[Tuple[str,str], dict]:
    """
    Collapse bookmaker outcomes into one average line per (player, side) pair.
    Returns {(player, side): {"point": line, "price": avg_price}}
    """
    buckets: Dict[Tuple[str,str], List[Tuple[float,Optional[float]]]] = {}
    for o in outcomes:
        player = o.get("description") or o.get("participant") or ""
        side = o.get("name")  # "Over" / "Under" (or "Yes"/"No" for anytime TD sometimes)
        pt = o.get("point")
        price = o.get("price")
        if player and side and (pt is not None):
            buckets.setdefault((player, side), []).append((float(pt), price if price is not None else np.nan))
    out: Dict[Tuple[str,str], dict] = {}
    for key, vals in buckets.items():
        pts = [v for v,_ in vals]
        prices = [p for _,p in vals if pd.notna(p)]
        out[key] = {"point": float(np.mean(pts)), "price": float(np.mean(prices)) if prices else np.nan}
    return out

def defense_for_player(player: str, home_team: str, away_team: str) -> Tuple[str, float, float]:
    """Return (opp_team, pass_adj, rush_adj) by mapping player to home/away."""
    team = PLAYER_TO_TEAM.get(clean_name(player))
    if team is None:
        # heuristic: if name contains home/away QB/WR stars is unknown; fall back to away defense (neutral-ish)
        # but try a soft guess: if player's last name appears in home team city/nickname, pick home. Rare, so ignore.
        opp = away_team  # default assume player on home team
        p_adj = pass_adj(opp); r_adj = rush_adj(opp)
        return opp, p_adj, r_adj
    # If player team equals home -> opponent defense is away
    if team.lower() == home_team.lower():
        opp = away_team
    elif team.lower() == away_team.lower():
        opp = home_team
    else:
        # team mismatch (traded, naming variance) → choose the defense closer by string match
        hsim = 1 if home_team.lower() in team.lower() else 0
        opp = away_team if hsim else home_team
    return opp, pass_adj(opp), rush_adj(opp)

def mu_from_line_with_defense(market: str, line: float, p_adj: float, r_adj: float) -> float:
    """
    Start from the consensus line as a neutral mean, then tilt by defense EPA.
    """
    if market in ("player_pass_yds",):
        return float(line * p_adj)
    if market in ("player_rec_yds","player_receptions"):
        return float(line * p_adj)
    if market in ("player_rush_yds",):
        return float(line * r_adj)
    if market == "player_anytime_td":
        # we'll turn this into a Bernoulli p; 'mu' unused for TD
        return float(line)
    return float(line)

def simulate_market_prob(market: str, side: str, line: float, mu: float, price_hint: Optional[float], n: int) -> float:
    if market == "player_anytime_td":
        # convert price (if present) to base prob, then nudge by blended adj from mu (already line)
        if not pd.isna(price_hint):
            p0 = american_to_prob(price_hint)
        else:
            p0 = 0.33  # neutral baseline if price missing
        # keep as-is; side can be Over/Yes or Under/No (Odds API often uses Over/Under with point=0.5)
        p = p0
        return float(p*100.0) if side.lower() in ("over","yes") else float((1.0-p)*100.0)
    else:
        sd = SD_DEFAULTS.get(market, 25.0)
        pov = normal_over_prob(mu, sd, line, n)
        return float(pov*100.0) if side.lower()=="over" else float((1.0-pov)*100.0)

# ---------------------------- Run end-to-end ----------------------------
def run_all_games(api_key: str, region: str, days: int, markets: List[str], bookmakers: List[str], trials: int) -> pd.DataFrame:
    events = list_events(api_key, region, days)
    if not events:
        raise RuntimeError("No upcoming NFL events returned.")
    rows = []
    for ev in events:
        ev_id = ev["id"]
        home = ev["home_team"]; away = ev["away_team"]
        start = ev.get("commence_time","")
        try:
            data = game_odds(api_key, ev_id, region, markets, bookmakers)
        except requests.HTTPError as e:
            # skip noisy 422/404 for unsupported markets
            continue
        # Merge all bookmaker markets together
        all_outcomes = []
        for bk in data.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                for oc in mkt.get("outcomes", []):
                    oc2 = oc.copy()
                    oc2["_market"] = mkt.get("key","")
                    all_outcomes.append(oc2)
        if not all_outcomes:
            continue
        # group by market
        by_market: Dict[str, List[dict]] = {}
        for oc in all_outcomes:
            by_market.setdefault(oc["_market"], []).append(oc)
        for mname, oclist in by_market.items():
            agg = aggregate_lines(oclist)  # {(player, side) -> {"point","price"}}
            for (player, side), d in agg.items():
                opp_team, p_adj, r_adj = defense_for_player(player, home, away)
                mu = mu_from_line_with_defense(mname, d["point"], p_adj, r_adj)
                prob = simulate_market_prob(mname, side, d["point"], mu, d.get("price", np.nan), trials)
                rows.append({
                    "event_time": start,
                    "home_team": home, "away_team": away,
                    "market": mname, "player": player, "side": side,
                    "line": round(d["point"], 3),
                    "price_hint": d.get("price", np.nan),
                    "mu": round(mu, 3),
                    "prob": round(prob, 2),
                    "opp_def": opp_team,
                    "pass_adj": round(p_adj, 3),
                    "rush_adj": round(r_adj, 3),
                })
    return pd.DataFrame(rows)

# ---------------------------- Main UI flow ----------------------------
if run_btn:
    if not api_key.strip():
        st.error("Please paste your Odds API key in the sidebar.")
        st.stop()
    try:
        with st.spinner("Fetching all events, pulling props, mapping players → teams, simulating…"):
            df = run_all_games(api_key, region, lookahead_days, markets, bookmakers, int(trials))
        if df.empty:
            st.warning("No props returned for the chosen window/markets/bookmakers.")
        else:
            st.success(f"Simulated {len(df):,} player props across all games.")
            # Friendly ordering
            order = ["event_time","home_team","away_team","market","player","side",
                     "line","price_hint","mu","prob","opp_def","pass_adj","rush_adj"]
            show = [c for c in order if c in df.columns]
            st.dataframe(df[show].sort_values(["event_time","market","player"]), use_container_width=True, height=620)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download all props (CSV)", csv, file_name="props_sim_results.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

# Footer tip
st.caption("Note: ‘prob’ is the simulated **P(Over)** for Over, and **P(Under)** for Under. Defense multipliers come from your embedded 2025 EPA table (pass→pass/rec; rush→rush).")
