# NHL Player Props — Odds API + ESPN (per-game means; 10k sims)
# Place this file at: pages/4_NHL.py

import math, re, unicodedata
from typing import List, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NHL Player Props — Odds API + ESPN", layout="wide")
st.title("🏒 NHL Player Props — Odds API + ESPN")

SIM_TRIALS = 10_000
VALID_MARKETS = [
    "player_points",
    "player_power_play_points",
    "player_assists",
    "player_blocked_shots",
    "player_shots_on_goal",
    "player_goals",
    "player_total_saves",
    "player_goal_scorer_first",
    "player_goal_scorer_last",
    "player_goal_scorer_anytime",
]

SB_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
SUM_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary"

def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
def norm_name(n: str) -> str:
    n = str(n or "")
    n = re.sub(r"[.,'–-]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n)

@st.cache_data(show_spinner=False)
def list_dates_events(start: str, end: str) -> List[str]:
    """start/end YYYYMMDD inclusive; ESPN NHL scoreboard uses 'dates' param"""
    # ESPN supports a single date per call → loop
    dates = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="D")
    out = []
    for d in dates:
        js = http_get(SB_URL, {"dates": d.strftime("%Y%m%d")})
        if not js: continue
        out += [str(e.get("id")) for e in js.get("events", []) if e.get("id")]
    return out

@st.cache_data(show_spinner=False)
def fetch_box(event_id: str) -> Optional[dict]:
    return http_get(SUM_URL, {"event": event_id})

def _to_float(x):
    try: return float(x)
    except Exception: return float("nan")

def parse_box_players(box: dict) -> pd.DataFrame:
    """
    Extract skater + goalie lines
    """
    rows = []
    try:
        teams = box.get("boxscore", {}).get("players", [])
        for t in teams:
            team_abbr = t.get("team", {}).get("abbreviation")
            for block in t.get("statistics", []):
                label = (block.get("name") or "").lower()
                for a in block.get("athletes", []):
                    nm = norm_name(a.get("athlete", {}).get("displayName"))
                    stats = a.get("stats") or []
                    # Skater block (common order): G, A, PTS, +/- , SOG, BLK, PIM, PPG, SHG, ...
                    if "skaters" in label or "skater" in label or "players" in label:
                        # Try common indices (ESPN sometimes varies but usually stable early season)
                        g  = _to_float(stats[0]) if len(stats) > 0 else 0.0
                        a_ = _to_float(stats[1]) if len(stats) > 1 else 0.0
                        pts= _to_float(stats[2]) if len(stats) > 2 else g + a_
                        sog= _to_float(stats[4]) if len(stats) > 4 else np.nan
                        blk= _to_float(stats[5]) if len(stats) > 5 else np.nan
                        ppg= _to_float(stats[7]) if len(stats) > 7 else np.nan
                        rows.append({"Player": nm, "team": team_abbr,
                                     "goals": g, "assists": a_, "points": pts,
                                     "sog": sog, "blocks": blk, "pp_points": ppg})
                    # Goalie block: SV, SA, SV%, GA, ...
                    if "goalies" in label or "goalie" in label:
                        sv = _to_float(stats[0]) if len(stats) > 0 else np.nan
                        rows.append({"Player": nm, "team": team_abbr,
                                     "saves": sv})
    except Exception:
        pass
    if not rows:
        return pd.DataFrame(columns=["Player","team","goals","assists","points","sog","blocks","pp_points","saves"])
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ["Player","team"]]
    return df.groupby(["Player","team"], as_index=False)[num_cols].sum(numeric_only=True)

@st.cache_data(show_spinner=True)
def build_span_agg(start: str, end: str) -> pd.DataFrame:
    events = list_dates_events(start, end)
    if not events: return pd.DataFrame()

    totals, sumsqs, games = {}, {}, {}
    def init_p(p):
        if p not in totals:
            totals[p] = {"goals":0.0,"assists":0.0,"points":0.0,"sog":0.0,"blocks":0.0,"pp_points":0.0,"saves":0.0}
            sumsqs[p] = {"goals":0.0,"assists":0.0,"points":0.0,"sog":0.0,"blocks":0.0,"pp_points":0.0,"saves":0.0}
            games[p]  = 0
    prog = st.progress(0.0, text=f"Crawling {len(events)} games…")
    for j, ev in enumerate(events, 1):
        box = fetch_box(ev)
        if box:
            df = parse_box_players(box)
            for _, r in df.iterrows():
                p = norm_name(r["Player"]); init_p(p)
                if any(float(r.get(k,0)) > 0 for k in ["goals","assists","points","sog","blocks","saves"]):
                    games[p] += 1
                for k in totals[p]:
                    v = float(r.get(k, 0.0)) if not pd.isna(r.get(k, np.nan)) else 0.0
                    totals[p][k] += v
                    sumsqs[p][k] += v*v
        prog.progress(j/len(events))
    rows = []
    for p, stat in totals.items():
        g = max(1, games.get(p, 0))
        rows.append({"Player": p, "g": g, **stat, **{f"sq_{k}": sumsqs[p][k] for k in stat}})
    return pd.DataFrame(rows)

def sample_sd(sum_x, sum_x2, g_val):
    g_val = int(g_val)
    if g_val <= 1: return np.nan
    mean = sum_x / g_val
    var  = (sum_x2 / g_val) - (mean**2)
    var  = var * (g_val / (g_val - 1))
    return float(np.sqrt(max(var, 1e-6)))

def t_over_prob(mu: float, sd: float, line: float, trials=SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

# ---------------- UI: date range ----------------
st.markdown("### 1) Date range (regular-season days)")
c1, c2 = st.columns(2)
with c1:
    start_date = st.text_input("Start (YYYY/MM/DD)", value="2025/10/01")
with c2:
    end_date   = st.text_input("End (YYYY/MM/DD)", value="2025/10/07")

# ---------------- Build projections ----------------
st.markdown("### 2) Build per-player projections from ESPN")
if st.button("📥 Build NHL projections"):
    span = build_span_agg(start_date.replace("/",""), end_date.replace("/",""))
    if span.empty:
        st.error("No data returned from ESPN for this date range."); st.stop()

    g = span["g"].clip(lower=1)
    for k in ["goals","assists","points","sog","blocks","pp_points","saves"]:
        span[f"mu_{k}"] = span[k] / g
        sd_raw = span.apply(lambda r: sample_sd(r[k], r[f"sq_{k}"], r["g"]), axis=1)
        floor = 0.6 if k in ["goals","assists","points","pp_points"] else 1.0
        span[f"sd_{k}"] = np.maximum(floor, np.nan_to_num(sd_raw)) * 1.10

    st.session_state["nhl_proj"] = span.copy()
    st.success(f"Built projections for {len(span)} players.")
    st.dataframe(span[["Player","g","mu_points","mu_goals","mu_sog","mu_saves"]].head(20), use_container_width=True)

# ---------------- Odds API ----------------
st.markdown("### 3) Pick a game & markets from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets", VALID_MARKETS, default=VALID_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200: raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()
def list_events(api_key: str, lookahead_days: int, region: str):
    return odds_get("https://api.the-odds-api.com/v4/sports/icehockey_nhl/events",
                    {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region})
def fetch_props(api_key: str, event_id: str, region: str, markets: List[str]):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/events/{event_id}/odds",
                    {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"})

events = []
if api_key:
    try: events = list_events(api_key, lookahead, region)
    except Exception as e: st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key to list NHL games."); st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]; event_id = event["id"]

# ---------------- Simulate ----------------
st.markdown("### 4) Fetch lines & simulate (per-game means only)")
go = st.button("🎲 Fetch lines & simulate (NHL)")

if go:
    proj = st.session_state.get("nhl_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build NHL projections first (Step 2)."); st.stop()

    # prepare lookup
    proj = proj.copy(); proj["PN"] = proj["Player"].apply(norm_name); proj.set_index("PN", inplace=True)

    try:
        data = fetch_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}"); st.stop()

    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = norm_name(o.get("description"))
                side = o.get("name")
                line = o.get("point")
                if mkey not in VALID_MARKETS or not name or not side:
                    continue
                rows.append({"market": mkey, "player": name, "side": side, "point": (None if line is None else float(line))})
    if not rows:
        st.warning("No player outcomes returned for these markets."); st.stop()

    props = (pd.DataFrame(rows).groupby(["market","player","side"], as_index=False)
             .agg(line=("point","median"), books=("point","size")))

    def sim_yes_from_rate(lam: float) -> float:
        # Poisson yes-probability
        lam = max(1e-6, float(lam))
        return 1.0 - math.exp(-lam)

    out = []
    for _, r in props.iterrows():
        pl, m, side, line = r["player"], r["market"], r["side"], r["line"]
        if pl not in proj.index: 
            continue
        pr = proj.loc[pl]

        # Over/Under markets driven by t-dist around per-game mean
        if m == "player_points":
            mu, sd = float(pr["mu_points"]), float(pr["sd_points"])
        elif m == "player_power_play_points":
            mu, sd = float(pr["mu_pp_points"]), float(pr["sd_pp_points"])
        elif m == "player_assists":
            mu, sd = float(pr["mu_assists"]), float(pr["sd_assists"])
        elif m == "player_blocked_shots":
            mu, sd = float(pr["mu_blocks"]), float(pr["sd_blocks"])
        elif m == "player_shots_on_goal":
            mu, sd = float(pr["mu_sog"]), float(pr["sd_sog"])
        elif m == "player_goals":
            mu, sd = float(pr["mu_goals"]), float(pr["sd_goals"])
        elif m == "player_total_saves":
            mu, sd = float(pr["mu_saves"]), float(pr["sd_saves"])
        elif m in ("player_goal_scorer_anytime","player_goal_scorer_first","player_goal_scorer_last"):
            # Approximate from goals rate (λ = goals per game).
            lam = float(pr["mu_goals"])
            p_yes = sim_yes_from_rate(lam)
            # very rough proxies for first/last (subset of anytime)
            if m == "player_goal_scorer_anytime":
                p = p_yes
            else:
                p = p_yes * 0.25  # conservative share for first/last
            p = p if side in ("Yes","Over") else (1.0 - p)
            out.append({"market": m, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,2), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue
        else:
            continue

        if line is None or np.isnan(mu) or np.isnan(sd):
            continue
        p_over = float((mu + sd * np.random.standard_t(df=5, size=SIM_TRIALS) > float(line)).mean())
        p = p_over if side == "Over" else (1.0 - p_over)

        out.append({"market": m, "player": pl, "side": side, "line": float(line),
                    "μ (per-game)": round(mu,2), "σ (per-game)": round(sd,2),
                    "Win Prob %": round(100*p,2), "books": int(r["books"])})

    if not out:
        st.warning("No matched props to simulate."); st.stop()

    results = (pd.DataFrame(out)
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    colcfg = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.2f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "books": st.column_config.NumberColumn("#Books", width="small"),
    }
    st.dataframe(results, hide_index=True, use_container_width=True, column_config=colcfg)
    st.download_button("⬇️ Download CSV", results.to_csv(index=False).encode("utf-8"), "nhl_props_results.csv", "text/csv")
