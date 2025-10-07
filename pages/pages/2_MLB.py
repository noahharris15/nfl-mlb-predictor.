# MLB Player Props — Odds API + ESPN (per-game averages, 10k sims)
# Place this file at: pages/2_MLB.py

import re, unicodedata, datetime as dt
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.title("⚾ MLB Player Props — Odds API + ESPN")

SIM_TRIALS = 10_000
VALID_MARKETS = [
    "player_hits",
    "player_total_bases",
    "player_home_runs",
    "player_rbis",
    "player_runs_scored",
]
ODDS_SPORT = "baseball_mlb"

def strip_accents(s: str) -> str:
    import unicodedata as u
    return "".join(c for c in u.normalize("NFKD", s) if not u.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]; n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n); n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n)

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def to_float(x) -> float:
    try: return float(x)
    except Exception: return float("nan")

def sample_sd(sum_x, sum_x2, g_val):
    g = int(g_val)
    if g <= 1: return np.nan
    mean = sum_x / g
    var = (sum_x2 / g) - (mean**2)
    var = var * (g / (g - 1))
    return float(np.sqrt(max(var, 1e-6)))

# ESPN endpoints (date range)
BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
SCOREBOARD = f"{BASE}/scoreboard"
SUMMARY    = f"{BASE}/summary"

def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def list_event_ids_by_dates(start: dt.date, end: dt.date) -> List[str]:
    rng = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    js = http_get(SCOREBOARD, params={"dates": rng})
    if not js: return []
    return [str(e.get("id")) for e in js.get("events", []) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id: str) -> Optional[dict]:
    return http_get(SUMMARY, params={"event": event_id})

def _extract_team_players(box: dict):
    out = []
    try:
        sec = box.get("boxscore", {}).get("players", [])
        for team in sec:
            tname = team.get("team", {}).get("shortDisplayName")
            for p in team.get("statistics", []):
                labels = p.get("labels", []) or p.get("descriptions", [])
                for a in p.get("athletes", []):
                    out.append({
                        "team": tname,
                        "name": a.get("athlete", {}).get("displayName"),
                        "labels": [str(x).upper() for x in (labels or [])],
                        "vals": a.get("stats") or [],
                    })
    except Exception:
        pass
    return out

def parse_boxscore_players_mlb(box: dict) -> pd.DataFrame:
    rows = _extract_team_players(box)
    keep = {}
    for r in rows:
        nm = normalize_name(r.get("name"))
        labels = r.get("labels") or []
        vals = r.get("vals", [])
        idx = {lab: i for i, lab in enumerate(labels)}

        def grab(label):
            i = idx.get(label)
            if i is not None and i < len(vals):
                v = vals[i]
                try: return float(v)
                except Exception:
                    if isinstance(v, str) and "-" in v:
                        try: return float(v.split("-")[0])
                        except Exception: return np.nan
            return np.nan

        rec = keep.setdefault(nm, {"Player": nm, "H":0.0, "TB":0.0, "HR":0.0, "RBI":0.0, "R":0.0})
        for lab, key in [("H","H"), ("TB","TB"), ("HR","HR"), ("RBI","RBI"), ("R","R")]:
            v = grab(lab)
            if not np.isnan(v): rec[key] += v

    if not keep: return pd.DataFrame(columns=["Player","H","TB","HR","RBI","R"])
    return pd.DataFrame(list(keep.values()))

@st.cache_data(show_spinner=True)
def build_espx_season_avg(start: dt.date, end: dt.date) -> pd.DataFrame:
    events = list_event_ids_by_dates(start, end)
    if not events: return pd.DataFrame()
    totals, sumsqs, games = {}, {}, {}
    def init_p(p):
        if p not in totals:
            totals[p] = {"H":0.0,"TB":0.0,"HR":0.0,"RBI":0.0,"R":0.0}
            sumsqs[p] = {"H":0.0,"TB":0.0,"HR":0.0,"RBI":0.0,"R":0.0}
            games[p]  = 0
    prog = st.progress(0.0, text=f"Crawling {len(events)} games…")
    for j, ev in enumerate(events, 1):
        box = fetch_boxscore_event(ev)
        if box:
            df = parse_boxscore_players_mlb(box)
            for _, r in df.iterrows():
                p = r["Player"]; init_p(p)
                played = any(to_float(r[k]) > 0 for k in ["H","TB","HR","RBI","R"])
                if played: games[p] += 1
                for k in ["H","TB","HR","RBI","R"]:
                    v = to_float(r.get(k, np.nan))
                    if not np.isnan(v):
                        totals[p][k] += v
                        sumsqs[p][k] += v*v
        prog.progress(j/len(events))
    rows = []
    for p, stat in totals.items():
        g = max(1, int(games.get(p, 0)))
        rows.append({"Player": p, "g": g, **stat,
                     "sq_H":stat["H"]*0+0, "sq_TB":stat["TB"]*0+0, "sq_HR":stat["HR"]*0+0,
                     "sq_RBI":stat["RBI"]*0+0, "sq_R":stat["R"]*0+0})
    # We didn’t track sq_* above; redo with correct squares:
    for row in rows:
        p = row["Player"]
        # recompute using totals dict and a pass above? Simpler: rebuild sums once more (we already computed in sumsqs)
        # Replace with stored sumsqs:
        row["sq_H"]  = sumsqs[p]["H"]
        row["sq_TB"] = sumsqs[p]["TB"]
        row["sq_HR"] = sumsqs[p]["HR"]
        row["sq_RBI"]= sumsqs[p]["RBI"]
        row["sq_R"]  = sumsqs[p]["R"]
    return pd.DataFrame(rows)

# ------------------ UI: dates ------------------
st.markdown("### 1) Pick a date range")
season = st.number_input("Season year", min_value=2015, max_value=2100, value=2025, step=1)
default_start = dt.date(season, 4, 1)   # Apr 1
default_end   = dt.date(season, 10, 1)  # Oct 1
start, end = st.date_input("Dates (inclusive)", (default_start, default_end), format="YYYY-MM-DD")

# ------------------ Build projections ------------------
st.markdown("### 2) Build per-player projections from ESPN")
if st.button("📥 Build MLB projections"):
    if isinstance(start, tuple): start, end = start
    season_df = build_espx_season_avg(start, end)
    if season_df.empty:
        st.error("No data returned from ESPN for the selected dates."); st.stop()

    g = season_df["g"].clip(lower=1)

    # Per-game means
    season_df["mu_hits_raw"] = season_df["H"]   / g
    season_df["mu_tb_raw"]   = season_df["TB"]  / g
    season_df["mu_hr_raw"]   = season_df["HR"]  / g
    season_df["mu_rbi_raw"]  = season_df["RBI"] / g
    season_df["mu_runs_raw"] = season_df["R"]   / g

    # SDs
    sd_H  = season_df.apply(lambda r: sample_sd(r["H"],  r["sq_H"],  r["g"]), axis=1)
    sd_TB = season_df.apply(lambda r: sample_sd(r["TB"], r["sq_TB"], r["g"]), axis=1)
    sd_HR = season_df.apply(lambda r: sample_sd(r["HR"], r["sq_HR"], r["g"]), axis=1)
    sd_RBI= season_df.apply(lambda r: sample_sd(r["RBI"],r["sq_RBI"],r["g"]), axis=1)
    sd_R  = season_df.apply(lambda r: sample_sd(r["R"],  r["sq_R"],  r["g"]), axis=1)

    SDI = 1.10
    season_df["sd_hits"] = np.where(np.isnan(sd_H),  np.nan, np.maximum(0.8, sd_H)  * SDI)
    season_df["sd_tb"]   = np.where(np.isnan(sd_TB), np.nan, np.maximum(1.2, sd_TB) * SDI)
    season_df["sd_hr"]   = np.where(np.isnan(sd_HR), np.nan, np.maximum(0.3, sd_HR) * SDI)
    season_df["sd_rbi"]  = np.where(np.isnan(sd_RBI),np.nan, np.maximum(0.8, sd_RBI)* SDI)
    season_df["sd_runs"] = np.where(np.isnan(sd_R),  np.nan, np.maximum(0.8, sd_R)  * SDI)

    # Neutral scaling
    season_df["mu_hits"] = season_df["mu_hits_raw"]
    season_df["mu_tb"]   = season_df["mu_tb_raw"]
    season_df["mu_hr"]   = season_df["mu_hr_raw"]
    season_df["mu_rbi"]  = season_df["mu_rbi_raw"]
    season_df["mu_runs"] = season_df["mu_runs_raw"]

    st.session_state["mlb_proj"] = season_df[[
        "Player","g",
        "mu_hits_raw","mu_hits","sd_hits",
        "mu_tb_raw","mu_tb","sd_tb",
        "mu_hr_raw","mu_hr","sd_hr",
        "mu_rbi_raw","mu_rbi","sd_rbi",
        "mu_runs_raw","mu_runs","sd_runs",
    ]]
    st.dataframe(st.session_state["mlb_proj"].head(20), use_container_width=True)

# ------------------ Odds API ------------------
st.markdown("### 3) Pick a game & markets (Odds API)")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets to fetch", VALID_MARKETS, default=VALID_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200: raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_events(api_key: str, lookahead_days: int, region: str):
    base = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events"
    return odds_get(base, {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region})

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    base = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds"
    return odds_get(base, {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"})

events = []
if api_key:
    try: events = list_events(api_key, lookahead, region)
    except Exception as e: st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list upcoming games."); st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]; event_id = event["id"]

# ------------------ Simulate ------------------
st.markdown("### 4) Fetch lines & simulate")
go = st.button("Fetch lines & simulate (MLB)")

if go:
    proj = st.session_state.get("mlb_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build MLB projections first (Step 2)."); st.stop()

    try:
        data = fetch_event_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}"); st.stop()

    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                side = o.get("name"); point = o.get("point")
                if mkey not in VALID_MARKETS or not name or side not in ("Over","Under"): continue
                rows.append({"market": mkey,"player_norm": name,"side": side,"point": (None if point is None else float(point))})
    if not rows:
        st.warning("No player outcomes returned for selected markets."); st.stop()

    props_df = (pd.DataFrame(rows).groupby(["market","player_norm","side"], as_index=False)
                              .agg(line=("point","median"), n_books=("point","size")))

    out = []
    names = set(proj["Player"])
    for _, r in props_df.iterrows():
        market, player, point, side = r["market"], r["player_norm"], r["line"], r["side"]
        if player not in names or pd.isna(point): continue
        row = proj.loc[proj["Player"] == player].iloc[0]

        if   market == "player_hits":         mu, sd, mu_raw = float(row["mu_hits"]), float(row["sd_hits"]), float(row["mu_hits_raw"])
        elif market == "player_total_bases":  mu, sd, mu_raw = float(row["mu_tb"]),   float(row["sd_tb"]),   float(row["mu_tb_raw"])
        elif market == "player_home_runs":    mu, sd, mu_raw = float(row["mu_hr"]),   float(row["sd_hr"]),   float(row["mu_hr_raw"])
        elif market == "player_rbis":         mu, sd, mu_raw = float(row["mu_rbi"]),  float(row["sd_rbi"]),  float(row["mu_rbi_raw"])
        elif market == "player_runs_scored":  mu, sd, mu_raw = float(row["mu_runs"]), float(row["sd_runs"]), float(row["mu_runs_raw"])
        else: continue

        if np.isnan(mu) or np.isnan(sd): continue
        p_over = t_over_prob(mu, sd, float(point), SIM_TRIALS)
        p = p_over if side == "Over" else 1.0 - p_over

        out.append({
            "market": market, "player": player, "side": side,
            "line": round(float(point), 2),
            "Avg (raw)": round(mu_raw, 2),
            "μ (scaled)": round(mu, 2),
            "σ (per-game)": round(sd, 2),
            "Games": int(row["g"]),
            "Win Prob %": round(100*p, 2),
            "books": int(r["n_books"]),
        })

    if not out: st.warning("No props matched your projections."); st.stop()
    results = (pd.DataFrame(out)
        .drop_duplicates(subset=["market","player","side"])
        .sort_values(["market","Win Prob %"], ascending=[True, False])
        .reset_index(drop=True))

    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="mlb_props_sim_results.csv", mime="text/csv")
