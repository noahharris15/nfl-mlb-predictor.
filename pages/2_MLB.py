# MLB Player Props — Odds API + ESPN (per-day crawl; per-game averages)
# Place in: pages/2_MLB.py  |  Run app normally; this becomes the MLB page.

import re, math, unicodedata, time, datetime as dt
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="MLB Player Props — Odds API + ESPN", layout="wide")
st.title("⚾ MLB Player Props — Odds API + ESPN (per-game averages)")

SIM_TRIALS = 10_000

# Odds API markets (MLB)
VALID_MLB_MARKETS = [
    # batters
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
    "batter_rbis",
    "batter_runs_scored",
    "batter_hits_runs_rbis",
    # pitchers
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_walks",
    "pitcher_earned_runs",
    "pitcher_outs",
    "pitcher_record_a_win",
]

# ---------------- ESPN endpoints (MLB) ----------------
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"

def http_get(url, params=None, timeout=25, retries=2, sleep=0.6) -> Optional[dict]:
    for i in range(retries+1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(sleep)
    return None

def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)

def normalize_name(n: str) -> str:
    import unicodedata, re
    n = str(n or "")
    n = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.split("(")[0].replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

@st.cache_data(show_spinner=False)
def list_event_ids_by_date(day: dt.date) -> List[str]:
    # ESPN requires dates as YYYYMMDD
    js = http_get(SCOREBOARD, params={"dates": day.strftime("%Y%m%d"), "limit": 300})
    if not js: return []
    return [str(e.get("id")) for e in js.get("events", []) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id: str) -> Optional[dict]:
    return http_get(SUMMARY, params={"event": event_id})

# ---------- Boxscore parsing ----------
def parse_espn_mlb_box(box: dict) -> pd.DataFrame:
    """
    Return per-player stats for ONE game.
      Batters: hits, total bases, HR, RBI, R, (H+R+RBI)
      Pitchers: SO (K), H allowed, BB, ER, Outs, Win (1/0)
    """
    rows = []
    players = box.get("boxscore", {}).get("players", []) or []
    for team in players:
        for group in team.get("statistics", []):
            label = (group.get("name") or "").lower()
            for a in group.get("athletes", []):
                nm = normalize_name(a.get("athlete", {}).get("displayName"))
                vals = a.get("stats") or []

                # batting group labels vary across ESPN feeds (e.g., "Batting")
                if "bat" in label:
                    # Typical order (not 100% guaranteed, but widely used):
                    # AB,R,H,2B,3B,HR,RBI,BB,SO,SF,AVG,OBP,SLG,OPS,TB,...
                    # We’ll defensively map by index with try/except
                    def f(i):
                        try:
                            s = str(vals[i])
                            return float(s) if s.replace(".","",1).isdigit() else 0.0
                        except Exception:
                            return 0.0
                    H   = f(2)
                    _2B = f(3)
                    _3B = f(4)
                    HR  = f(5)
                    RBI = f(6)
                    R   = f(1)
                    TB  = H + _2B + 2*_2B + 3*_3B + 4*HR - (_2B + _3B + HR)  # crude fallback
                    # if TB provided later in list, prefer it
                    for v in vals:
                        if isinstance(v, str) and v.endswith(" TB"):
                            try:
                                TB = float(v.split()[0])
                            except Exception:
                                pass
                    HRRBIH = H + R + RBI
                    rows.append({
                        "Player": nm,
                        "is_pitcher": 0,
                        "hits": H, "tb": TB, "hr": HR, "rbi": RBI, "runs": R, "hrrbih": HRRBIH,
                        "so": 0.0, "h_allowed": 0.0, "bb": 0.0, "er": 0.0, "outs": 0.0, "win": 0.0,
                    })

                elif "pitch" in label:
                    # Usual: IP,H,R,ER,BB,SO,HR,PC-ST,ERA
                    # We need: SO, H allowed, BB, ER, Outs, Win (from 'notes' or decisions)
                    def f(i):
                        try:
                            s = str(vals[i])
                            return float(s) if s.replace(".","",1).isdigit() else 0.0
                        except Exception:
                            return 0.0
                    H_allowed = f(1)
                    ER        = f(3)
                    BB        = f(4)
                    SO        = f(5)
                    # IP -> outs (e.g., 6.1 = 6 and 1/3)
                    def ip_to_outs(ip_str):
                        try:
                            s = str(ip_str)
                            whole, dot, frac = s.partition(".")
                            outs = int(whole)*3
                            outs += 1 if frac == "1" else (2 if frac == "2" else 0)
                            return float(outs)
                        except Exception:
                            return 0.0
                    outs = ip_to_outs(vals[0] if vals else "0")
                    # Win detection: sometimes stored under 'athlete' -> 'stats' extras, else decisions list
                    win_flag = 0.0
                    note = (a.get("note") or "").lower()
                    if "win" in note and "(w" in note:
                        win_flag = 1.0
                    rows.append({
                        "Player": nm,
                        "is_pitcher": 1,
                        "hits": 0.0, "tb": 0.0, "hr": 0.0, "rbi": 0.0, "runs": 0.0, "hrrbih": 0.0,
                        "so": SO, "h_allowed": H_allowed, "bb": BB, "er": ER, "outs": outs, "win": win_flag,
                    })

    if not rows:
        return pd.DataFrame(columns=["Player","is_pitcher","hits","tb","hr","rbi","runs","hrrbih","so","h_allowed","bb","er","outs","win"])

    # Some players appear in multiple lines; sum within game
    return pd.DataFrame(rows).groupby(["Player","is_pitcher"], as_index=False).sum(numeric_only=True)

# ---------- Crawl date range & aggregate per-player ----------
@st.cache_data(show_spinner=True)
def build_mlb_espn_agg(start_date: str, end_date: str) -> pd.DataFrame:
    s = dt.datetime.strptime(start_date, "%Y/%m/%d").date()
    e = dt.datetime.strptime(end_date,   "%Y/%m/%d").date()
    if e < s: s, e = e, s

    all_events = []
    day_counts = []
    for d in daterange(s, e):
        evs = list_event_ids_by_date(d)
        day_counts.append({"date": int(d.strftime("%Y%m%d")), "events": len(evs)})
        all_events.extend(evs)

    st.expander("ESPN day-by-day event counts", expanded=False).dataframe(pd.DataFrame(day_counts), use_container_width=True)

    if not all_events:
        return pd.DataFrame()

    totals, sumsqs, games = {}, {}, {}
    def init(p, is_pitcher):
        if p not in totals:
            totals[p] = {"is_pitcher": int(is_pitcher),
                         "hits":0.0,"tb":0.0,"hr":0.0,"rbi":0.0,"runs":0.0,"hrrbih":0.0,
                         "so":0.0,"h_allowed":0.0,"bb":0.0,"er":0.0,"outs":0.0,"win":0.0}
            sumsqs[p] = {"hits":0.0,"tb":0.0,"hr":0.0,"rbi":0.0,"runs":0.0,"hrrbih":0.0,
                         "so":0.0,"h_allowed":0.0,"bb":0.0,"er":0.0,"outs":0.0}
            games[p]  = 0

    prog = st.progress(0.0, text=f"Crawling {len(all_events)} MLB games from ESPN…")
    for j, ev in enumerate(all_events, 1):
        box = fetch_boxscore_event(ev)
        if box:
            gdf = parse_espn_mlb_box(box)
            for _, r in gdf.iterrows():
                p = r["Player"]
                init(p, r["is_pitcher"])
                played = any(float(r[k]) > 0 for k in ["hits","tb","hr","rbi","runs","so","h_allowed","bb","er","outs"])
                if played: games[p] += 1
                for k in totals[p]:
                    if k == "is_pitcher": continue
                    v = float(r.get(k, 0.0))
                    totals[p][k] += v
                for k in sumsqs[p]:
                    v = float(r.get(k, 0.0))
                    sumsqs[p][k] += v*v
        prog.progress(j/len(all_events))

    rows = []
    for p, stat in totals.items():
        g = max(1, int(games.get(p, 0)))
        rows.append({"Player": p, "g": g, **stat, **{f"sq_{k}": sumsqs[p][k] for k in sumsqs[p]}})
    return pd.DataFrame(rows)

def sample_sd(sum_x: float, sum_x2: float, g: int, floor: float) -> float:
    g = int(g)
    if g <= 1: return float("nan")
    mean = sum_x / g
    var  = (sum_x2 / g) - (mean**2)
    var  = var * (g / (g - 1))
    return float(max(floor, math.sqrt(max(var, 1e-6))))

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

# ---------------- UI: date range ----------------
st.markdown("### 1) Date range")
c1, c2 = st.columns(2)
with c1:
    start_date = st.text_input("Start date (YYYY/MM/DD)", value="2025/09/01")
with c2:
    end_date   = st.text_input("End date (YYYY/MM/DD)",   value="2025/10/07")

# ---------------- Build projections ----------------
st.markdown("### 2) Build per-player projections from ESPN 🔗")
if st.button("📥 Build MLB projections"):
    season_df = build_mlb_espn_agg(start_date, end_date)
    if season_df.empty:
        st.error("No data returned from ESPN for this date range."); st.stop()

    g = season_df["g"].clip(lower=1)

    # Per-game means (no shrinkage)
    season_df["mu_hits"]  = season_df["hits"] / g
    season_df["mu_tb"]    = season_df["tb"]   / g
    season_df["mu_hr"]    = season_df["hr"]   / g
    season_df["mu_rbi"]   = season_df["rbi"]  / g
    season_df["mu_runs"]  = season_df["runs"] / g
    season_df["mu_hrrbiH"]= season_df["hrrbih"] / g

    season_df["mu_k"]     = season_df["so"]  / g
    season_df["mu_ha"]    = season_df["h_allowed"] / g
    season_df["mu_bb"]    = season_df["bb"]  / g
    season_df["mu_er"]    = season_df["er"]  / g
    season_df["mu_outs"]  = season_df["outs"]/ g
    season_df["p_win"]    = (season_df["win"] / g).clip(0, 1)

    # Sample SDs (floors keep sims sane)
    season_df["sd_hits"]  = season_df.apply(lambda r: sample_sd(r["hits"], r["sq_hits"], r["g"], 0.4), axis=1)
    season_df["sd_tb"]    = season_df.apply(lambda r: sample_sd(r["tb"],   r["sq_tb"],   r["g"], 0.6), axis=1)
    season_df["sd_hr"]    = season_df.apply(lambda r: sample_sd(r["hr"],   r["sq_hr"],   r["g"], 0.20), axis=1)
    season_df["sd_rbi"]   = season_df.apply(lambda r: sample_sd(r["rbi"],  r["sq_rbi"],  r["g"], 0.60), axis=1)
    season_df["sd_runs"]  = season_df.apply(lambda r: sample_sd(r["runs"], r["sq_runs"], r["g"], 0.60), axis=1)
    season_df["sd_hrrbih"]= season_df.apply(lambda r: sample_sd(r["hrrbih"], r["sq_hrrbih"], r["g"], 0.9), axis=1)

    season_df["sd_k"]     = season_df.apply(lambda r: sample_sd(r["so"],  r["sq_so"],  r["g"], 0.6), axis=1)
    season_df["sd_ha"]    = season_df.apply(lambda r: sample_sd(r["h_allowed"], r["sq_h_allowed"], r["g"], 0.7), axis=1)
    season_df["sd_bb"]    = season_df.apply(lambda r: sample_sd(r["bb"],  r["sq_bb"],  r["g"], 0.6), axis=1)
    season_df["sd_er"]    = season_df.apply(lambda r: sample_sd(r["er"],  r["sq_er"],  r["g"], 0.6), axis=1)
    season_df["sd_outs"]  = season_df.apply(lambda r: sample_sd(r["outs"],r["sq_outs"],r["g"], 1.2), axis=1)

    # Store slim projections
    keep = [
        "Player","g","is_pitcher",
        "mu_hits","sd_hits","mu_tb","sd_tb","mu_hr","sd_hr","mu_rbi","sd_rbi","mu_runs","sd_runs","mu_hrrbiH","sd_hrrbih",
        "mu_k","sd_k","mu_ha","sd_ha","mu_bb","sd_bb","mu_er","sd_er","mu_outs","sd_outs","p_win"
    ]
    st.session_state["mlb_proj"] = season_df[keep].copy()
    st.success("Built MLB projections from ESPN.")
    st.dataframe(st.session_state["mlb_proj"].head(20), use_container_width=True)

# ---------------- Odds API ----------------
st.markdown("### 3) Pick a game & markets (Odds API)")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets to fetch", VALID_MLB_MARKETS, default=VALID_MLB_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_mlb_events(api_key: str, lookahead_days: int, region: str):
    base = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
    params = {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region}
    return odds_get(base, params)

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    base = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"}
    return odds_get(base, params)

events = []
if api_key:
    try:
        events = list_mlb_events(api_key, lookahead, region)
    except Exception as e:
        st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list upcoming games.")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ---------------- Simulate ----------------
st.markdown("### 4) Fetch lines & simulate")
go = st.button("🎲 Fetch lines & simulate (MLB)")

if go:
    proj = st.session_state.get("mlb_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build MLB projections first (Step 2)."); st.stop()

    proj = proj.copy()
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    idx = proj.set_index("player_norm")

    try:
        data = fetch_event_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                side = o.get("name")
                point = o.get("point")
                if mkey not in VALID_MLB_MARKETS or not name or not side:
                    continue
                rows.append({
                    "market": mkey,
                    "player_norm": name,
                    "side": side,
                    "point": (None if point is None else float(point)),
                })

    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    props_df = (pd.DataFrame(rows)
                .groupby(["market","player_norm","side"], as_index=False)
                .agg(line=("point","median"), n_books=("point","size")))

    out_rows = []
    for _, r in props_df.iterrows():
        mkt, name, side, line = r["market"], r["player_norm"], r["side"], r["line"]
        if name not in idx.index or pd.isna(line):
            continue
        row = idx.loc[name]

        # Map to (mu, sd) or special handling
        if mkt == "batter_hits":
            mu, sd = float(row["mu_hits"]), float(row["sd_hits"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "batter_total_bases":
            mu, sd = float(row["mu_tb"]), float(row["sd_tb"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "batter_home_runs":
            lam = float(row["mu_hr"])
            p_over = float((np.random.poisson(lam=max(1e-6,lam), size=SIM_TRIALS) > float(line)).mean())
            mu, sd = lam, None
        elif mkt == "batter_rbis":
            mu, sd = float(row["mu_rbi"]), float(row["sd_rbi"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "batter_runs_scored":
            mu, sd = float(row["mu_runs"]), float(row["sd_runs"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "batter_hits_runs_rbis":
            mu, sd = float(row["mu_hrrbiH"]), float(row["sd_hrrbih"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "pitcher_strikeouts":
            mu, sd = float(row["mu_k"]), float(row["sd_k"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "pitcher_hits_allowed":
            mu, sd = float(row["mu_ha"]), float(row["sd_ha"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "pitcher_walks":
            mu, sd = float(row["mu_bb"]), float(row["sd_bb"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "pitcher_earned_runs":
            mu, sd = float(row["mu_er"]), float(row["sd_er"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "pitcher_outs":
            mu, sd = float(row["mu_outs"]), float(row["sd_outs"])
            p_over = t_over_prob(mu, sd, float(line))
        elif mkt == "pitcher_record_a_win":
            p_yes = float(row["p_win"])
            p_over = p_yes  # treat Over/Yes as yes-prob
            mu, sd = p_yes, None
        else:
            continue

        p = p_over if side in ("Over","Yes") else 1.0 - p_over
        out_rows.append({
            "market": mkt,
            "player": row["Player"],
            "side": side,
            "line": round(float(line),2) if line is not None else None,
            "μ (per-game)": None if mu is None else round(float(mu),2),
            "σ (per-game)": None if (sd is None or (isinstance(sd,float) and np.isnan(sd))) else round(float(sd),2),
            "Win Prob %": round(100*p,2),
            "#Books": int(r["n_books"]),
        })

    if not out_rows:
        st.warning("No props matched projections.")
        st.stop()

    results = (pd.DataFrame(out_rows)
               .drop_duplicates(subset=["market","player","side"])
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    colcfg = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.2f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "#Books": st.column_config.NumberColumn("#Books", width="small"),
    }
    st.subheader("Results")
    st.dataframe(results, use_container_width=True, hide_index=True, column_config=colcfg)

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="mlb_props_results.csv",
        mime="text/csv",
    )
