# app_props_only.py  — NFL Player Props (All Games) with embedded Defense EPA
# Requirements (same env you used before):
#   pip install streamlit numpy pandas requests tenacity rapidfuzz scipy nfl_data_py

import os
import math
import time
import json
import random
import numpy as np
import pandas as pd
import requests
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from rapidfuzz import fuzz
from scipy.stats import norm, poisson

# ──────────────────────────── Streamlit UI ────────────────────────────
st.set_page_config(page_title="NFL Player Props Simulator (All Games)", layout="wide")
st.title("🏈 NFL Player Props — All Games (Odds API + Real Stats + Defense EPA)")

with st.sidebar:
    st.markdown("### API & Options")
    ODDS_API_KEY = st.text_input("The Odds API key", value=os.getenv("ODDS_API_KEY", ""), type="password")
    lookahead_days = st.slider("Lookahead (days)", 1, 7, 2)
    chosen_books = st.multiselect(
        "Bookmakers to average (empty = all returned)",
        ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet_us", "bet365_us"],
        default=["draftkings", "fanduel"]
    )
    markets_wanted = st.multiselect(
        "Markets to fetch",
        ["player_pass_yds", "player_rush_yds", "player_rec_yds", "player_receptions", "player_anytime_td"],
        default=["player_pass_yds", "player_rush_yds", "player_rec_yds", "player_receptions", "player_anytime_td"]
    )
    st.caption("We fetch all upcoming NFL events within the window above, then call the **event-odds** endpoint for each game to get player props.")

if not ODDS_API_KEY:
    st.warning("Paste your Odds API key in the sidebar to run.")
    st.stop()

# ─────────────────────── Embedded Defense EPA (2025) ───────────────────────
# From your table. We use pass_EPA for pass/rec/receptions; rush_EPA for rushing.
# (If a team is missing here, we default to league average 0.0)
DEF_EPA = {
    "Minnesota Vikings":      {"epa_play":-0.17,"pass_epa":-0.37,"rush_epa":0.06},
    "Jacksonville Jaguars":   {"epa_play":-0.13,"pass_epa":-0.17,"rush_epa":-0.05},
    "Denver Broncos":         {"epa_play":-0.11,"pass_epa":-0.10,"rush_epa":-0.12},
    "Los Angeles Chargers":   {"epa_play":-0.11,"pass_epa":-0.17,"rush_epa":0.01},
    "Detroit Lions":          {"epa_play":-0.09,"pass_epa":0.00,"rush_epa":-0.22},
    "Philadelphia Eagles":    {"epa_play":-0.08,"pass_epa":-0.11,"rush_epa":-0.04},
    "Houston Texans":         {"epa_play":-0.08,"pass_epa":-0.16,"rush_epa":0.00},
    "Los Angeles Rams":       {"epa_play":-0.08,"pass_epa":-0.12,"rush_epa":0.00},
    "Seattle Seahawks":       {"epa_play":-0.07,"pass_epa":0.00,"rush_epa":-0.19},
    "San Francisco 49ers":    {"epa_play":-0.06,"pass_epa":-0.02,"rush_epa":-0.11},
    "Tampa Bay Buccaneers":   {"epa_play":-0.06,"pass_epa":-0.11,"rush_epa":-0.13},
    "Atlanta Falcons":        {"epa_play":-0.05,"pass_epa":0.06,"rush_epa":-0.17},
    "Cleveland Browns":       {"epa_play":-0.05,"pass_epa":-0.04,"rush_epa":-0.05},
    "Indianapolis Colts":     {"epa_play":-0.03,"pass_epa":-0.09,"rush_epa":0.09},
    "Kansas City Chiefs":     {"epa_play":-0.02,"pass_epa":-0.01,"rush_epa":0.19},
    "Arizona Cardinals":      {"epa_play":-0.01,"pass_epa":0.06,"rush_epa":-0.14},
    "Las Vegas Raiders":      {"epa_play":-0.01,"pass_epa":0.14,"rush_epa":-0.22},
    "Green Bay Packers":      {"epa_play":0.00,"pass_epa":0.03,"rush_epa":-0.07},
    "Chicago Bears":          {"epa_play":0.00,"pass_epa":0.01,"rush_epa":0.00},
    "Buffalo Bills":          {"epa_play":0.02,"pass_epa":-0.06,"rush_epa":0.10},
    "Carolina Panthers":      {"epa_play":0.04,"pass_epa":0.03,"rush_epa":0.05},
    "Pittsburgh Steelers":    {"epa_play":0.04,"pass_epa":0.11,"rush_epa":-0.05},
    "Washington Commanders":  {"epa_play":0.05,"pass_epa":0.18,"rush_epa":-0.12},
    "New England Patriots":   {"epa_play":0.05,"pass_epa":0.01,"rush_epa":0.00},
    "New York Giants":        {"epa_play":0.07,"pass_epa":-0.01,"rush_epa":0.12},
    "New Orleans Saints":     {"epa_play":0.07,"pass_epa":0.20,"rush_epa":-0.06},
    "Cincinnati Bengals":     {"epa_play":0.10,"pass_epa":0.13,"rush_epa":0.04},
    "New York Jets":          {"epa_play":0.11,"pass_epa":0.23,"rush_epa":-0.03},
    "Tennessee Titans":       {"epa_play":0.12,"pass_epa":0.16,"rush_epa":0.07},
    "Baltimore Ravens":       {"epa_play":0.25,"pass_epa":0.40,"rush_epa":0.06},
    "Dallas Cowboys":         {"epa_play":0.25,"pass_epa":0.34,"rush_epa":0.12},
    "Miami Dolphins":         {"epa_play":0.25,"pass_epa":0.34,"rush_epa":0.12},  # from your last row
}

# ───────────── EPA -> multiplier (bounded; tougher D => <1, softer => >1) ─────────────
def defense_multiplier(team: str, kind: str) -> float:
    d = DEF_EPA.get(team, {"pass_epa":0.0, "rush_epa":0.0})
    e = d["pass_epa"] if kind in ("pass","rec") else d["rush_epa"]
    # Standardize-ish: clamp EPA range to [-0.35, +0.35] and convert to ~8% swing per 0.35
    e = max(-0.35, min(0.35, float(e)))
    # Negative EPA (better defense) should DECREASE player output (multiplier < 1)
    # We flip sign: better (negative) => mult = 1 - k*|e| ; worse (positive) => 1 + k*e
    k = 0.25 / 0.35  # ~±25% at extremes
    mult = 1.0 + (e * k)
    return max(0.75, min(1.25, mult))

# ────────────────────────── Real per-game stats ──────────────────────────
@st.cache_data(show_spinner=False)
def load_player_averages():
    import nfl_data_py as nfl
    # Try 2025; if empty, fallback to 2024
    try_years = [2025, 2024]
    df = None
    for yr in try_years:
        try:
            tmp = nfl.import_seasonal_data([yr])
            if tmp is not None and not tmp.empty:
                df = tmp
                break
        except Exception:
            continue
    if df is None or df.empty:
        raise RuntimeError("NFL seasonal stats not available (2025/2024).")

    g = df["games"].replace(0, np.nan) if "games" in df.columns else np.nan
    out = pd.DataFrame({
        "player": df.get("player_display_name"),
        "team": df.get("recent_team"),
        "pass_yds": (df.get("passing_yards") / g),
        "rush_yds": (df.get("rushing_yards") / g),
        "rec_yds":  (df.get("receiving_yards") / g),
        "receptions": (df.get("receptions") / g),
        "tds": ((df.get("passing_tds") + df.get("rushing_tds") + df.get("receiving_tds")) / g),
    })
    out = out.dropna(subset=["player"]).reset_index(drop=True)
    # reasonable fill for missing cols
    for c in ["pass_yds","rush_yds","rec_yds","receptions","tds"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

@st.cache_data(show_spinner=False)
def build_player_team_map():
    import nfl_data_py as nfl
    df = None
    for yr in [2025, 2024]:
        try:
            df = nfl.import_rosters([yr])
            if df is not None and not df.empty:
                break
        except Exception:
            continue
    if df is None or df.empty:
        return {}

    # name formats can vary; we’ll map with fuzzy match on demand
    # Build candidates by team: {team: [names]}
    team_to_players = {}
    for _, r in df.iterrows():
        nm = str(r.get("full_name") or r.get("player_name") or "").strip()
        tm = str(r.get("team") or r.get("recent_team") or "").strip()
        if not nm or not tm:
            continue
        team_to_players.setdefault(tm, []).append(nm)
    return team_to_players

PLAYER_AVG = load_player_averages()
TEAM_ROSTER = build_player_team_map()

def clean(s): return (s or "").replace(".", "").replace("-", " ").strip().lower()

def guess_player_team(player_name: str) -> str|None:
    # Try direct match in our averages first
    row = PLAYER_AVG.loc[PLAYER_AVG["player"].str.lower()==player_name.lower()]
    if not row.empty:
        t = row.iloc[0].get("team")
        if pd.notna(t) and str(t).strip():
            return str(t).strip()
    # Fuzzy across roster buckets
    best_team, best_score = None, -1
    for tm, names in TEAM_ROSTER.items():
        for nm in names:
            sc = fuzz.token_sort_ratio(clean(nm), clean(player_name))
            if sc > best_score:
                best_team, best_score = tm, sc
    return best_team if best_score >= 90 else None

# ──────────────────────────── Odds API helpers ────────────────────────────
BASE = "https://api.the-odds-api.com/v4"

def _params(extra: dict):
    p = {"apiKey": ODDS_API_KEY}
    p.update(extra)
    return p

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 6))
def _get(url: str, params: dict):
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
    return r.json()

@st.cache_data(ttl=60, show_spinner=False)
def list_events(days_ahead: int):
    url = f"{BASE}/sports/americanfootball_nfl/events"
    js = _get(url, _params({"daysFrom": 0, "daysTo": days_ahead, "regions": "us"}))
    # Keep id, start, home, away
    ev = [{
        "id": e["id"],
        "start": e.get("commence_time"),
        "home": e.get("home_team"),
        "away": e.get("away_team")
    } for e in js]
    return ev

def fetch_event_props(event_id: str, markets: list[str], books: list[str]|None):
    url = f"{BASE}/sports/americanfootball_nfl/events/{event_id}/odds"
    extra = {"regions": "us", "markets": ",".join(markets), "oddsFormat": "american"}
    if books:
        extra["bookmakers"] = ",".join(books)
    js = _get(url, _params(extra))
    # structure: bookmakers -> markets -> outcomes
    rows = []
    for bk in js.get("bookmakers", []):
        book_key = bk.get("key")
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                rows.append({
                    "book": book_key,
                    "market": mkey,
                    "name": o.get("name"),               # Over / Under / Yes / No
                    "player": o.get("description"),      # player name (if provided)
                    "point": o.get("point"),             # line for O/U markets
                    "price": o.get("price")              # moneyline price if relevant
                })
    return pd.DataFrame(rows)

# ───────────────────────── Simulation pieces ─────────────────────────
def conservative_sd(mu, floor=1.0, frac=0.30):
    try:
        mu = float(mu)
    except Exception:
        return 10.0
    sd = max(floor, abs(mu) * frac)
    return sd

def ou_prob(mu, line, sd):
    return float(1 - norm.cdf(line, loc=float(mu), scale=max(1e-6, float(sd))))

def adj_mu_for_def(mu, team_def_name: str, market: str):
    # Which kind?
    kind = "pass"
    if "rush" in market: kind = "rush"
    elif "rec_" in market or "receptions" in market: kind = "rec"
    mult = defense_multiplier(team_def_name, kind)
    return float(mu) * mult, mult

def pick_opponent_for_player(player_team: str, home: str, away: str) -> str:
    if not player_team:
        # unknown → neutral (home defense by default)
        return away  # arbitrary but deterministic
    # Try basic mapping between full team names and TLA rosters from nfl_data_py
    # If player's team matches home, opponent is away; else if matches away, opponent is home.
    # We allow loose matching.
    def same(a,b): return fuzz.token_sort_ratio(clean(a), clean(b)) >= 92
    if same(player_team, home): return away
    if same(player_team, away): return home
    # loose: if player's team contains nickname part
    if any(w in clean(home) for w in clean(player_team).split()): return away
    if any(w in clean(away) for w in clean(player_team).split()): return home
    # fallback: assume away as opponent
    return away

def derive_player_mu(player: str):
    row = PLAYER_AVG.loc[PLAYER_AVG["player"].str.lower()==player.lower()]
    if row.empty:
        # try fuzzy
        best_idx, best_score = None, -1
        for idx, nm in enumerate(PLAYER_AVG["player"]):
            sc = fuzz.token_sort_ratio(clean(nm), clean(player))
            if sc > best_score:
                best_idx, best_score = idx, sc
        if best_score < 90:
            return None
        row = PLAYER_AVG.iloc[[best_idx]]

    r = row.iloc[0]
    return {
        "team": str(r.get("team") or ""),
        "pass_yds": float(r.get("pass_yds") or 0.0),
        "rush_yds": float(r.get("rush_yds") or 0.0),
        "rec_yds":  float(r.get("rec_yds") or 0.0),
        "receptions": float(r.get("receptions") or 0.0),
        "tds": float(r.get("tds") or 0.0),
    }

# ───────────────────────── Run for ALL games ─────────────────────────
st.markdown("#### 1) Fetching upcoming games…")
events = list_events(lookahead_days)
st.write(f"Found **{len(events)}** events in the next {lookahead_days} day(s).")

if not markets_wanted:
    st.warning("Pick at least one market in the sidebar.")
    st.stop()

do_run = st.button("Fetch props for all games & simulate", type="primary")
if not do_run:
    st.stop()

all_rows = []
progress = st.progress(0.0)

for i, ev in enumerate(events):
    progress.progress((i+1)/max(1,len(events)))
    home, away, eid = ev["home"], ev["away"], ev["id"]
    try:
        props = fetch_event_props(eid, markets_wanted, chosen_books)
    except Exception as e:
        st.error(f"Event {home} vs {away} fetch failed: {e}")
        continue

    if props.empty:
        continue

    # Average the line across selected books per (market,player,side)
    # For anytime TD, 'point' is None; we simulate Poisson on mu (no line)
    props["side"] = props["name"].str.title()  # Over / Under / Yes / No
    grp_cols = ["market","player","side"]
    agg = props.groupby(grp_cols, as_index=False).agg(
        line=("point","mean"),
        price=("price","mean"),
        n_books=("book","size")
    )

    for _, r in agg.iterrows():
        market = r["market"]
        player = str(r["player"] or "").strip()
        if not player:
            continue

        base = derive_player_mu(player)
        if base is None:
            # no averages → skip
            continue

        # Identify player's team and the opponent defense
        p_team = guess_player_team(player) or base["team"]
        opp = pick_opponent_for_player(p_team, home, away)

        # Choose which base mu to use
        if market == "player_pass_yds":
            mu0 = base["pass_yds"]; kind="pass"
        elif market == "player_rush_yds":
            mu0 = base["rush_yds"]; kind="rush"
        elif market == "player_rec_yds":
            mu0 = base["rec_yds"]; kind="rec"
        elif market == "player_receptions":
            mu0 = base["receptions"]; kind="rec"
        elif market == "player_anytime_td":
            mu0 = base["tds"]; kind="rush"  # neutral; TDs correlate more with rush in redzone
        else:
            continue

        mu_adj, mult = adj_mu_for_def(mu0, opp, market)

        if market == "player_anytime_td":
            # Poisson on mean TDs to get Pr(score ≥ 1)
            lam = max(0.01, mu_adj)
            prob_yes = float(1 - math.exp(-lam))  # P(N>=1) for Poisson(lam)
            row_yes = {
                "game": f"{away} @ {home}",
                "start": ev["start"],
                "player": player,
                "player_team": p_team,
                "opponent_def": opp,
                "market": market,
                "side": "Yes" if r["side"].lower()=="yes" else r["side"],
                "line": None,
                "mu": round(mu_adj,3),
                "sd": None,
                "prob": round(prob_yes*100,2),
                "books": int(r["n_books"]),
                "def_mult": round(mult,3)
            }
            all_rows.append(row_yes)
            continue

        # O/U markets
        line = float(r["line"]) if pd.notna(r["line"]) else None
        if line is None:
            continue

        sd = conservative_sd(mu_adj, floor=8.0 if "yds" in market else 0.8, frac=0.30)
        p_over = ou_prob(mu_adj, line, sd)
        prob = p_over if r["side"].lower()=="over" else (1-p_over)

        all_rows.append({
            "game": f"{away} @ {home}",
            "start": ev["start"],
            "player": player,
            "player_team": p_team,
            "opponent_def": opp,
            "market": market,
            "side": r["side"],
            "line": round(line,3),
            "mu": round(mu_adj,3),
            "sd": round(sd,3),
            "prob": round(prob*100,2),
            "books": int(r["n_books"]),
            "def_mult": round(mult,3),
        })

progress.empty()

results = pd.DataFrame(all_rows)
if results.empty:
    st.error("No simulated rows. Try different markets, bookmakers, or a larger lookahead.")
    st.stop()

# Order & nice columns
col_order = ["start","game","player","player_team","opponent_def","market","side","line","mu","sd","prob","books","def_mult"]
for c in col_order:
    if c not in results.columns:
        results[c] = None
results = results[col_order].sort_values(["start","game","market","player"]).reset_index(drop=True)

st.success(f"Simulated {len(results):,} props across {results['game'].nunique()} games.")
st.dataframe(results, use_container_width=True, height=560)

csv = results.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV (all games)", data=csv, file_name="props_sim_results.csv", mime="text/csv")

st.caption("""
**Notes**
- Defense multipliers come from your embedded EPA table (pass/rush EPA). Pass EPA is applied to pass/rec/receptions, Rush EPA to rushing.  
- Over/Under SD is conservative (30% of μ, with a floor) to avoid 0%/100% artifacts.  
- Anytime TD probability uses a Poisson(μ_TDs) model: `P(TD ≥ 1) = 1 - e^{-μ}`.  
- Player→team mapping comes from `nfl_data_py` rosters (2025 → 2024 fallback) with fuzzy matching.
""")
