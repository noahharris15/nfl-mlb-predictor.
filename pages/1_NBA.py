# NBA Player Props — Odds API + ESPN (per-game averages, 10k sims)
# Place this file at: pages/1_NBA.py
# IMPORTANT: Do NOT call st.set_page_config here (it's in app.py / NFL page).

import re, unicodedata, datetime as dt
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.title("🏀 NBA Player Props — Odds API + ESPN")

SIM_TRIALS = 10_000

# -------------------- VALID MARKETS --------------------
VALID_MARKETS = [
    "player_points",
    "player_points_q1",
    "player_rebounds",
    "player_rebounds_q1",
    "player_assists",
    "player_assists_q1",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_blocks_steals",
    "player_turnovers",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_rebounds_assists",
    "player_field_goals",
    "player_frees_made",
    "player_frees_attempts",
]

ODDS_SPORT = "basketball_nba"

# ------------------ Utils ------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
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

# ------------------ ESPN endpoints ------------------
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SCOREBOARD = f"{BASE}/scoreboard"
SUMMARY = f"{BASE}/summary"

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
                labels = p.get("labels") or p.get("descriptions") or []
                for a in p.get("athletes", []):
                    out.append({
                        "team": tname,
                        "name": a.get("athlete", {}).get("displayName"),
                        "labels": [str(x).upper() for x in labels],
                        "vals": a.get("stats") or [],
                    })
    except Exception:
        pass
    return out

def parse_boxscore_players_nba(box: dict) -> pd.DataFrame:
    rows = _extract_team_players(box)
    keep = {}
    for r in rows:
        nm = normalize_name(r.get("name"))
        labels = r.get("labels") or []
        vals = r.get("vals", [])
        idx = {lab: i for i, lab in enumerate(labels)}

        def grab(label, alt_keys=()):
            for lab in (label, *alt_keys):
                i = idx.get(lab)
                if i is None or i >= len(vals): continue
                v = vals[i]
                try: return float(v)
                except Exception:
                    if isinstance(v, str) and "-" in v:
                        try: return float(v.split("-")[0])
                        except: return np.nan
                    return np.nan
            return np.nan

        rec = keep.setdefault(nm, {
            "Player": nm, "pts": 0.0, "reb": 0.0, "ast": 0.0, "tpm": 0.0,
            "blk": 0.0, "stl": 0.0, "turn": 0.0, "fgm": 0.0, "fta": 0.0, "ftm": 0.0
        })

        pts = grab("PTS"); reb = grab("REB"); ast = grab("AST")
        tpm = grab("3PM", ("3PTM", "3P MADE", "3PT"))
        blk = grab("BLK"); stl = grab("STL"); turn = grab("TO", ("TOV",))
        fgm = grab("FGM"); fta = grab("FTA"); ftm = grab("FTM")

        for k, v in {"pts": pts, "reb": reb, "ast": ast, "tpm": tpm, "blk": blk, "stl": stl,
                     "turn": turn, "fgm": fgm, "fta": fta, "ftm": ftm}.items():
            if not np.isnan(v): rec[k] += v

    if not keep: return pd.DataFrame(columns=list(rec.keys()))
    return pd.DataFrame(list(keep.values()))

@st.cache_data(show_spinner=True)
def build_espx_season_avg(start: dt.date, end: dt.date) -> pd.DataFrame:
    events = list_event_ids_by_dates(start, end)
    if not events: return pd.DataFrame()
    totals, sumsqs, games = {}, {}, {}

    def init_p(p):
        if p not in totals:
            totals[p] = {k: 0.0 for k in
                         ["pts","reb","ast","tpm","blk","stl","turn","fgm","fta","ftm"]}
            sumsqs[p] = {k: 0.0 for k in totals[p]}
            games[p] = 0

    prog = st.progress(0.0, text=f"Crawling {len(events)} games…")
    for j, ev in enumerate(events, 1):
        box = fetch_boxscore_event(ev)
        if box:
            df = parse_boxscore_players_nba(box)
            for _, r in df.iterrows():
                p = r["Player"]; init_p(p)
                played = any(to_float(r[k]) > 0 for k in totals[p].keys())
                if played: games[p] += 1
                for k in totals[p]:
                    v = to_float(r.get(k, np.nan))
                    if not np.isnan(v):
                        totals[p][k] += v
                        sumsqs[p][k] += v*v
        prog.progress(j/len(events))

    rows = []
    for p, stat in totals.items():
        g = max(1, int(games.get(p, 0)))
        row = {"Player": p, "g": g}
        for k, v in stat.items():
            row[k] = v
            row["sq_"+k] = sumsqs[p][k]
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------ UI ------------------
st.markdown("### 1) Pick a date range")
season = st.number_input("Season year", 2015, 2100, 2025)
default_start = dt.date(season, 10, 1)
default_end = dt.date(season+1, 6, 30)
start, end = st.date_input("Dates (inclusive)", (default_start, default_end))

st.markdown("### 2) Build per-player projections")
if st.button("📥 Build NBA projections"):
    if isinstance(start, tuple): start, end = start
    df = build_espx_season_avg(start, end)
    if df.empty:
        st.error("No data from ESPN."); st.stop()

    g = df["g"].clip(lower=1)
    for stat in ["pts","reb","ast","tpm","blk","stl","turn","fgm","fta","ftm"]:
        df[f"mu_{stat}"] = df[stat]/g
        df[f"sd_{stat}"] = df.apply(lambda r, s=stat: sample_sd(r[s], r["sq_"+s], r["g"]), axis=1)

    st.session_state["nba_proj"] = df.copy()
    st.dataframe(df.head(20), use_container_width=True)

# ------------------ Odds API ------------------
st.markdown("### 3) Pick a game & markets (Odds API)")
api_key = (st.secrets.get("odds_api_key") if hasattr(st,"secrets") else None) or st.text_input(
    "Odds API Key", value="", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, 1)
markets = st.multiselect("Markets to fetch", VALID_MARKETS, default=VALID_MARKETS)

def odds_get(url, params): 
    r=requests.get(url,params=params,timeout=25)
    if r.status_code!=200: raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
    return r.json()

def list_events(api_key, lookahead, region):
    url=f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events"
    return odds_get(url, {"apiKey":api_key,"daysFrom":0,"daysTo":lookahead,"regions":region})

def fetch_event_props(api_key,event_id,region,markets):
    url=f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds"
    return odds_get(url,{"apiKey":api_key,"regions":region,"markets":",".join(markets),"oddsFormat":"american"})

events=[]
if api_key:
    try: events=list_events(api_key,lookahead,region)
    except Exception as e: st.error(f"Events fetch error: {e}")
if not events:
    st.info("Enter Odds API key and lookahead"); st.stop()

event_labels=[f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick=st.selectbox("Game",event_labels)
event=events[event_labels.index(pick)]
event_id=event["id"]

# ------------------ Simulate ------------------
st.markdown("### 4) Fetch lines & simulate")
go=st.button("Fetch lines & simulate (NBA)")

if go:
    proj=st.session_state.get("nba_proj",pd.DataFrame())
    if proj.empty: st.warning("Build projections first."); st.stop()
    try:
        data=fetch_event_props(api_key,event_id,region,markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}"); st.stop()

    rows=[]
    for bk in data.get("bookmakers",[]):
        for m in bk.get("markets",[]):
            mkey=m.get("key")
            for o in m.get("outcomes",[]):
                name=normalize_name(o.get("description"))
                side=o.get("name")
                point=o.get("point")
                if mkey not in VALID_MARKETS or not name: continue
                rows.append({"market":mkey,"player_norm":name,"side":side,"point":float(point) if point else None})
    if not rows:
        st.warning("No props found."); st.stop()

    props_df=(pd.DataFrame(rows).groupby(["market","player_norm","side"],as_index=False)
                .agg(line=("point","median"),n_books=("point","size")))

    out=[]
    for _,r in props_df.iterrows():
        mkt=r["market"]; player=r["player_norm"]; line=r["line"]; side=r["side"]
        sub=proj.loc[proj["Player"].apply(lambda x: normalize_name(x)==player)]
        if sub.empty or pd.isna(line): continue
        row=sub.iloc[0]

        mu=sd=raw=None

        if mkt=="player_points":
            mu,sd,raw=row["mu_pts"],row["sd_pts"],row["mu_pts"]
        elif mkt=="player_rebounds":
            mu,sd,raw=row["mu_reb"],row["sd_reb"],row["mu_reb"]
        elif mkt=="player_assists":
            mu,sd,raw=row["mu_ast"],row["sd_ast"],row["mu_ast"]
        elif mkt=="player_threes":
            mu,sd,raw=row["mu_tpm"],row["sd_tpm"],row["mu_tpm"]
        elif mkt=="player_blocks":
            mu,sd,raw=row["mu_blk"],row["sd_blk"],row["mu_blk"]
        elif mkt=="player_steals":
            mu,sd,raw=row["mu_stl"],row["sd_stl"],row["mu_stl"]
        elif mkt=="player_turnovers":
            mu,sd,raw=row["mu_turn"],row["sd_turn"],row["mu_turn"]
        elif mkt=="player_field_goals":
            mu,sd,raw=row["mu_fgm"],row["sd_fgm"],row["mu_fgm"]
        elif mkt=="player_frees_made":
            mu,sd,raw=row["mu_ftm"],row["sd_ftm"],row["mu_ftm"]
        elif mkt=="player_frees_attempts":
            mu,sd,raw=row["mu_fta"],row["sd_fta"],row["mu_fta"]
        elif mkt=="player_points_rebounds_assists":
            mu=row["mu_pts"]+row["mu_reb"]+row["mu_ast"]
            sd=np.sqrt(row["sd_pts"]**2+row["sd_reb"]**2+row["sd_ast"]**2)
            raw=row["mu_pts"]+row["mu_reb"]+row["mu_ast"]
        elif mkt=="player_points_rebounds":
            mu=row["mu_pts"]+row["mu_reb"]
            sd=np.sqrt(row["sd_pts"]**2+row["sd_reb"]**2)
            raw=row["mu_pts"]+row["mu_reb"]
        elif mkt=="player_points_assists":
            mu=row["mu_pts"]+row["mu_ast"]
            sd=np.sqrt(row["sd_pts"]**2+row["sd_ast"]**2)
            raw=row["mu_pts"]+row["mu_ast"]
        elif mkt=="player_rebounds_assists":
            mu=row["mu_reb"]+row["mu_ast"]
            sd=np.sqrt(row["sd_reb"]**2+row["sd_ast"]**2)
            raw=row["mu_reb"]+row["mu_ast"]
        elif mkt=="player_blocks_steals":
            mu=row["mu_blk"]+row["mu_stl"]
            sd=np.sqrt(row["sd_blk"]**2+row["sd_stl"]**2)
            raw=row["mu_blk"]+row["mu_stl"]
        else:
            continue  # skip hidden markets

        if np.isnan(mu) or np.isnan(sd): continue
        p_over=t_over_prob(mu,sd,line,SIM_TRIALS)
        p=p_over if side=="Over" else 1-p_over

        out.append({
            "market":mkt,
            "player":row["Player"],
            "side":side,
            "line":round(float(line),2),
            "Avg (raw)":round(float(raw),2),
            "μ (scaled)":round(float(mu),2),
            "σ (per-game)":round(float(sd),2),
            "Win Prob %":round(100*p,2),
            "books":int(r["n_books"])
        })

    if not out:
        st.warning("No props matched projections."); st.stop()

    results=(pd.DataFrame(out)
             .drop_duplicates(subset=["market","player","side"])
             .sort_values(["market","Win Prob %"],ascending=[True,False])
             .reset_index(drop=True))

    st.dataframe(results,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="nba_props_sim_results.csv",mime="text/csv")
