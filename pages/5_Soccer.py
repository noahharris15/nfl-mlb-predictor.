# pages/5_Soccer.py
# Soccer Player Props — Odds API + ESPN (per-game means; 10k sims)
# Drop this file into your Streamlit "pages/" folder.

import math
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- UI / constants ----------------
st.set_page_config(page_title="Soccer Player Props — Odds API + ESPN", layout="wide")
st.title("⚽ Soccer Player Props — Odds API + ESPN")

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

# League mapping: (UI name, ESPN league key, Odds API sport key)
LEAGUES = [
    ("English Premier League",   "eng.1",   "soccer_epl"),
    ("La Liga (Spain)",          "esp.1",   "soccer_spain_la_liga"),
    ("Serie A (Italy)",          "ita.1",   "soccer_italy_serie_a"),
    ("Bundesliga (Germany)",     "ger.1",   "soccer_germany_bundesliga"),
    ("Ligue 1 (France)",         "fra.1",   "soccer_france_ligue_one"),
    ("UEFA Champions League",    "uefa.champions", "soccer_uefa_champs_league"),
    ("MLS (USA)",                "usa.1",   "soccer_usa_mls"),
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

# ---------------- ESPN API ----------------
def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def _dates_list(start_slash: str, end_slash: str) -> List[str]:
    # return ESPN-friendly YYYYMMDD strings
    start = datetime.strptime(start_slash, "%Y/%m/%d").date()
    end   = datetime.strptime(end_slash, "%Y/%m/%d").date()
    out, cur = [], start
    while cur <= end:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out

def list_soccer_events(espn_key: str, dates: List[str]) -> List[str]:
    # ESPN soccer scoreboard accepts a single "dates" per call, so loop.
    SB_URL = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_key}/scoreboard"
    evs = []
    for d in dates:
        js = http_get(SB_URL, {"dates": d})
        if not js: 
            continue
        evs += [str(e.get("id")) for e in js.get("events", []) if e.get("id")]
    # unique preserve order
    return list(dict.fromkeys(evs))

def fetch_box_soccer(espn_key: str, event_id: str) -> Optional[dict]:
    SUM_URL = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_key}/summary"
    return http_get(SUM_URL, {"event": event_id})

# ---------------- ESPN soccer boxscore parser ----------------
def _parse_soccer_box(box: dict) -> pd.DataFrame:
    """
    Return one row per player for a single match with:
      goals, assists, shots, sog, yc, rc
    Tries multiple formats used by ESPN soccer boxscores.
    """
    rows: List[Dict] = []
    try:
        players_sections = (box.get("boxscore", {}) or {}).get("players", []) or []
        for team_blk in players_sections:
            team_abbr = (team_blk.get("team", {}) or {}).get("abbreviation") or (team_blk.get("team", {}) or {}).get("shortDisplayName")
            for stat_table in team_blk.get("statistics", []) or []:
                label = (stat_table.get("name") or "").lower()
                # Preferred: if headers provided, build a name->index map
                headers = stat_table.get("labels") or stat_table.get("keys") or stat_table.get("descriptions")
                idx = {}
                if headers and isinstance(headers, list):
                    for i, h in enumerate(headers):
                        hlow = str(h).lower()
                        if "goal" in hlow and "allowed" not in hlow:
                            idx["goals"] = i
                        if hlow in ("a", "assists") or "assist" in hlow:
                            idx["assists"] = i
                        if ("sog" == hlow) or ("shots on goal" in hlow):
                            idx["sog"] = i
                        if ("sh" == hlow) or ("shots" == hlow) or ("total shots" in hlow):
                            idx["shots"] = i
                        if ("yc" == hlow) or ("yellow" in hlow and "card" in hlow):
                            idx["yc"] = i
                        if ("rc" == hlow) or ("red" in hlow and "card" in hlow):
                            idx["rc"] = i
                for a in stat_table.get("athletes", []) or []:
                    nm = _norm_name(a.get("athlete", {}).get("displayName"))
                    stats = a.get("stats") or []
                    # Fallback guesses if headers are absent: many ESPN tables use fixed orders:
                    # Common skater row order: MIN, G, A, SH, SOG, YC, RC, ...
                    g   = _f(stats[idx["goals"]])   if "goals"   in idx and idx["goals"]   < len(stats) else (_f(stats[1]) if len(stats) > 1 else 0.0)
                    ast = _f(stats[idx["assists"]]) if "assists" in idx and idx["assists"] < len(stats) else (_f(stats[2]) if len(stats) > 2 else 0.0)
                    sh  = _f(stats[idx["shots"]])   if "shots"   in idx and idx["shots"]   < len(stats) else (_f(stats[3]) if len(stats) > 3 else np.nan)
                    sog = _f(stats[idx["sog"]])     if "sog"     in idx and idx["sog"]     < len(stats) else (_f(stats[4]) if len(stats) > 4 else np.nan)
                    yc  = _f(stats[idx["yc"]])      if "yc"      in idx and idx["yc"]      < len(stats) else (_f(stats[5]) if len(stats) > 5 else 0.0)
                    rc  = _f(stats[idx["rc"]])      if "rc"      in idx and idx["rc"]      < len(stats) else (_f(stats[6]) if len(stats) > 6 else 0.0)
                    if any(x > 0 for x in [g, ast, _f(sh if not np.isnan(sh) else 0), _f(sog if not np.isnan(sog) else 0), yc, rc]):
                        rows.append({
                            "Player": nm, "team": team_abbr,
                            "goals": g, "assists": ast,
                            "shots": sh, "sog": sog,
                            "yc": yc, "rc": rc
                        })
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=["Player","team","goals","assists","shots","sog","yc","rc"])
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ["Player","team"]]
    return df.groupby(["Player","team"], as_index=False)[num_cols].sum(numeric_only=True)

# ---------------- Aggregation over date range ----------------
@st.cache_data(show_spinner=True)
def build_soccer_means(espn_key: str, start_slash: str, end_slash: str) -> pd.DataFrame:
    dates = _dates_list(start_slash, end_slash)
    events = list_soccer_events(espn_key, dates)
    if not events:
        return pd.DataFrame()

    totals, sumsqs, games = {}, {}, {}
    def init_p(p):
        if p not in totals:
            totals[p] = {"goals":0.0,"assists":0.0,"shots":0.0,"sog":0.0,"yc":0.0,"rc":0.0}
            sumsqs[p] = {k:0.0 for k in totals[p]}
            games[p]  = 0

    prog = st.progress(0.0, text=f"Crawling {len(events)} matches…")
    for j, ev in enumerate(events, 1):
        box = fetch_box_soccer(espn_key, ev)
        if box:
            df = _parse_soccer_box(box)
            for _, r in df.iterrows():
                p = _norm_name(r["Player"]); init_p(p)
                if any(_f(r.get(k, 0)) > 0 for k in totals[p].keys()):
                    games[p] += 1
                for k in totals[p]:
                    v = _f(r.get(k, 0.0))
                    totals[p][k] += 0.0 if np.isnan(v) else v
                    sumsqs[p][k] += 0.0 if np.isnan(v) else (v*v)
        prog.progress(j/len(events))

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
    LEAGUE_NAME, ESPN_KEY, ODDS_KEY = league
with col1:
    start_date = st.text_input("Start (YYYY/MM/DD)", value=datetime.now().strftime("%Y/%m/01"))
with col2:
    end_date   = st.text_input("End (YYYY/MM/DD)",   value=datetime.now().strftime("%Y/%m/%d"))

# ---------------- Build projections ----------------
st.header("2) Build per-player averages from ESPN")
if st.button("📥 Build Soccer projections"):
    soc = build_soccer_means(ESPN_KEY, start_date, end_date)
    if soc.empty:
        st.error("No data returned from ESPN for this league/date window.")
        st.stop()
    # Store
    st.session_state["soc_proj"] = soc.copy()

    # Preview: raw μ table
    with st.expander("Preview — Per-game averages (μ) & σ", expanded=False):
        cols = ["Player","g","mu_goals","sd_goals","mu_assists","sd_assists","mu_shots","sd_shots","mu_sog","sd_sog","mu_yc","sd_yc","mu_rc","sd_rc"]
        st.dataframe(soc[cols].sort_values("mu_goals", ascending=False).head(40), use_container_width=True)
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
            lam = float(pr["mu_goals"])
            p_yes = poisson_yes(lam)
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            out.append({"market": mkt, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,3), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue

        if mkt in ("player_first_goal_scorer", "player_last_goal_scorer"):
            # Conservative proxy: subset of anytime. You can refine with team goal models later.
            lam = float(pr["mu_goals"])
            p_any = poisson_yes(lam)
            frac = 0.25  # ~ rough share
            p = (p_any * frac)
            p = p if side in ("Yes","Over") else (1.0 - p)
            out.append({"market": mkt, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,3), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue

        if mkt == "player_to_receive_card":
            lam = float(pr["mu_yc"])
            p_yes = min(0.95, max(0.0, lam))  # per-game yellow rate (bounded)
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            out.append({"market": mkt, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,3), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue

        if mkt == "player_to_receive_red_card":
            lam = float(pr["mu_rc"])
            p_yes = min(0.50, max(0.0, lam))  # reds are rare; clamp to 50% ceiling
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            out.append({"market": mkt, "player": pl, "side": side, "line": None,
                        "μ (per-game)": round(lam,3), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue

        # Over/Under markets
        if mkt == "player_shots_on_target":
            mu, sd = float(pr["mu_sog"]), float(pr["sd_sog"])
        elif mkt == "player_shots":
            mu, sd = float(pr["mu_shots"]), float(pr["sd_shots"])
        elif mkt == "player_assists":
            mu, sd = float(pr["mu_assists"]), float(pr["sd_assists"])
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
