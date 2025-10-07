# 2_MLB.py
# MLB Player Props — Odds API + ESPN (robust boxscore fetch, batters + pitchers)
# Run: streamlit run 2_MLB.py    (or as a Streamlit multipage in pages/)

import math
import re
import unicodedata
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- UI / constants ----------------
st.set_page_config(page_title="MLB Player Props — Odds API + ESPN", layout="wide")
st.title("⚾ MLB Player Props — Odds API + ESPN (batters & pitchers)")

SIM_TRIALS = 10_000

# Odds API MLB player markets (exact keys)
VALID_MLB_MARKETS = [
    # Batters
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
    "batter_rbis",
    "batter_runs_scored",
    "batter_hits_runs_rbis",
    "batter_singles",
    "batter_doubles",
    "batter_triples",
    "batter_walks",
    "batter_strikeouts",
    "batter_stolen_bases",
    # Pitchers
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_walks",
    "pitcher_earned_runs",
    "pitcher_outs",
    "pitcher_record_a_win",
]

# ------------- helpers -------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]            # "John Smith (L)" -> "John Smith"
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n)

def http_get(url: str, params: Optional[dict] = None, timeout: int = 25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def bernoulli_yes_prob(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return p

def sample_sd(sum_x: float, sum_x2: float, n: int, floor: float = 0.1) -> float:
    """Bessel-corrected SD from sums; returns NaN if n <= 1."""
    if n <= 1:
        return float("nan")
    mean = sum_x / n
    var = max((sum_x2 / n) - mean**2, 0.0)
    var *= n / (n - 1)
    return float(max(math.sqrt(var), floor))

# ---------------- ESPN endpoints ----------------
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SITE_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"
CORE_BOX_FMT = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{eid}/competitions/{eid}/boxscore"

@st.cache_data(show_spinner=False)
def list_events_for_day(yyyymmdd: str) -> List[str]:
    """Return ESPN event ids for a specific calendar day (YYYYMMDD)."""
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    js = http_get(SCOREBOARD, params={"dates": dt.strftime("%Y%m%d")})
    if not js:
        return []
    return [e.get("id") for e in js.get("events", []) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore(event_id: str) -> Optional[dict]:
    """
    Fetch MLB boxscore. Try site 'summary?event=' first; if that has no players,
    fall back to Core API boxscore and re-shape to a site-like schema.
    """
    site = http_get(SITE_SUMMARY, params={"event": event_id})
    if site and site.get("boxscore", {}).get("players"):
        return site

    core = http_get(CORE_BOX_FMT.format(eid=event_id))
    if not core:
        return None

    # Convert Core schema -> Site-like
    try:
        players = []
        for t in core.get("teams", []):
            team_name = t.get("team", {}).get("displayName") or t.get("team", {}).get("abbreviation")
            team_block = {"team": {"shortDisplayName": team_name}, "statistics": []}

            for cat in t.get("statistics", []):
                # cat example: {"name":"batting","labels":[...],"athletes":[...]}
                team_block["statistics"].append({
                    "name": cat.get("name", ""),
                    "labels": cat.get("labels", []),
                    "athletes": cat.get("athletes", []),
                })
            players.append(team_block)
        return {"boxscore": {"players": players}}
    except Exception:
        return None

def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _pull_stat(labels: List[str], stats: List[str], key: str, default=np.nan) -> float:
    """Find a stat by label name, tolerant to slight label differences (e.g., 'TB', 'Total Bases')."""
    if not labels or not stats:
        return default
    # map label -> value
    m = {str(labels[i]).strip().lower(): stats[i] for i in range(min(len(labels), len(stats)))}
    # possible aliases
    aliases = {
        "ab": ["ab", "at bats", "at-bats"],
        "h": ["h", "hits"],
        "1b": ["1b", "singles"],
        "2b": ["2b", "doubles"],
        "3b": ["3b", "triples"],
        "hr": ["hr", "home runs", "home run"],
        "r": ["r", "runs"],
        "rbi": ["rbi", "runs batted in"],
        "bb": ["bb", "walks"],
        "so": ["so", "strikeouts", "k", "k's"],
        "sb": ["sb", "stolen bases"],
        "tb": ["tb", "total bases"],
        "outs": ["outs", "outs recorded"],
        "er": ["er", "earned runs"],
        "h_allowed": ["h", "hits"],
        "bb_allowed": ["bb", "walks"],
        "so_pitch": ["so", "strikeouts", "k"],
        "ip": ["ip", "innings pitched"],
        "win": ["decision", "dec"],
    }
    for alias in aliases.get(key, [key]):
        for lab, val in m.items():
            if alias == lab:
                return _to_float(val)
    return default

def parse_mlb_boxscore(box: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (batters_df, pitchers_df) with per-game stats for one event.
    Site-like schema is required (we handle core->site conversion in fetch_boxscore).
    """
    bat_rows, pit_rows = [], []
    try:
        teams = box.get("boxscore", {}).get("players", [])
        for team in teams:
            team_name = team.get("team", {}).get("shortDisplayName", "")
            for sec in team.get("statistics", []):
                label = (sec.get("name") or "").lower()
                labels = sec.get("labels") or []
                for guy in sec.get("athletes", []):
                    nm = normalize_name(guy.get("athlete", {}).get("displayName"))
                    stats = guy.get("stats") or []

                    if "bat" in label:  # batting
                        row = {
                            "Player": nm, "Team": team_name,
                            "H": _pull_stat(labels, stats, "h"),
                            "TB": _pull_stat(labels, stats, "tb"),
                            "HR": _pull_stat(labels, stats, "hr"),
                            "R": _pull_stat(labels, stats, "r"),
                            "RBI": _pull_stat(labels, stats, "rbi"),
                            "BB": _pull_stat(labels, stats, "bb"),
                            "SO": _pull_stat(labels, stats, "so"),
                            "SB": _pull_stat(labels, stats, "sb"),
                            "1B": _pull_stat(labels, stats, "1b", default=np.nan),
                            "2B": _pull_stat(labels, stats, "2b", default=np.nan),
                            "3B": _pull_stat(labels, stats, "3b", default=np.nan),
                        }
                        bat_rows.append(row)
                    elif "pitch" in label:  # pitching
                        # Pitching labels are similar but represent allowed stats
                        row = {
                            "Player": nm, "Team": team_name,
                            "IP": _pull_stat(labels, stats, "ip"),
                            "H_allowed": _pull_stat(labels, stats, "h_allowed"),
                            "ER": _pull_stat(labels, stats, "er"),
                            "BB_allowed": _pull_stat(labels, stats, "bb_allowed"),
                            "SO_pitch": _pull_stat(labels, stats, "so_pitch"),
                            "Outs": _pull_stat(labels, stats, "outs"),
                            # Decision parsing (win / loss). If label present and includes 'W', mark win = 1 else 0.
                            "Win": 1.0 if isinstance(_pull_stat(labels, stats, "win", default=np.nan), float) and not np.isnan(_pull_stat(labels, stats, "win")) and str(guy.get("shortText","")).upper().startswith("W") else 0.0,
                        }
                        pit_rows.append(row)
    except Exception:
        pass

    bat_df = pd.DataFrame(bat_rows) if bat_rows else pd.DataFrame(
        columns=["Player","Team","H","TB","HR","R","RBI","BB","SO","SB","1B","2B","3B"]
    )
    pit_df = pd.DataFrame(pit_rows) if pit_rows else pd.DataFrame(
        columns=["Player","Team","IP","H_allowed","ER","BB_allowed","SO_pitch","Outs","Win"]
    )
    # Some core boxscores omit 1B but give H and (2B,3B,HR). Infer 1B if missing.
    if "1B" in bat_df.columns:
        mask = bat_df["1B"].isna()
        bat_df.loc[mask, "1B"] = (
            bat_df.loc[mask, "H"].fillna(0)
            - bat_df.loc[mask, ["2B","3B","HR"]].fillna(0).sum(axis=1)
        ).clip(lower=0)
    return bat_df, pit_df

# ------------- Build season window from ESPN -------------
st.header("1) Date range")
col1, col2 = st.columns(2)
with col1:
    d0 = st.text_input("Start date (YYYY/MM/DD)", value="2025/09/01")
with col2:
    d1 = st.text_input("End date (YYYY/MM/DD)", value="2025/09/07")

def _dates_yyyymmdd(a: str, b: str) -> List[str]:
    start = datetime.strptime(a, "%Y/%m/%d").date()
    end   = datetime.strptime(b, "%Y/%m/%d").date()
    days = []
    cur = start
    while cur <= end:
        days.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return days

@st.cache_data(show_spinner=False)
def crawl_event_ids(a: str, b: str) -> List[str]:
    out = []
    for d in _dates_yyyymmdd(a, b):
        ids = list_events_for_day(d)
        st.write(f"ESPN site scoreboard OK for {d} → {len(ids)} events")
        out.extend(ids)
    # de-dup; keep order
    seen, uniq = set(), []
    for e in out:
        if e and e not in seen:
            uniq.append(e); seen.add(e)
    return uniq

@st.cache_data(show_spinner=True)
def build_mlb_averages(a: str, b: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crawl day-by-day between [a,b], fetch boxscores, aggregate to per-player
    sums, games, and sample SDs. Returns (batters_proj, pitchers_proj).
    """
    eids = crawl_event_ids(a, b)
    if not eids:
        return pd.DataFrame(), pd.DataFrame()

    # rolling sums
    bat_sum: Dict[str, Dict[str, float]] = {}
    bat_sum2: Dict[str, Dict[str, float]] = {}
    bat_g: Dict[str, int] = {}

    pit_sum: Dict[str, Dict[str, float]] = {}
    pit_sum2: Dict[str, Dict[str, float]] = {}
    pit_g: Dict[str, int] = {}

    def binit(p):
        if p not in bat_sum:
            bat_sum[p]  = {"H":0,"TB":0,"HR":0,"R":0,"RBI":0,"BB":0,"SO":0,"SB":0,"1B":0,"2B":0,"3B":0}
            bat_sum2[p] = {k:0.0 for k in bat_sum[p]}
            bat_g[p]    = 0
    def pinit(p):
        if p not in pit_sum:
            pit_sum[p]  = {"SO_pitch":0,"H_allowed":0,"BB_allowed":0,"ER":0,"Outs":0,"Win":0}
            pit_sum2[p] = {k:0.0 for k in pit_sum[p]}
            pit_g[p]    = 0

    prog = st.progress(0.0, text="Fetching boxscores…")
    for j, eid in enumerate(eids, 1):
        box = fetch_boxscore(eid)
        if not box:
            prog.progress(j/len(eids)); continue
        bat_df, pit_df = parse_mlb_boxscore(box)

        # batters
        for _, r in bat_df.iterrows():
            p = normalize_name(r["Player"]); binit(p)
            played = any(_to_float(r.get(k, 0)) > 0 for k in bat_sum[p].keys())
            if played: bat_g[p] += 1
            for k in bat_sum[p]:
                v = _to_float(r.get(k, 0))
                if not np.isnan(v):
                    bat_sum[p][k]  += v
                    bat_sum2[p][k] += v*v

        # pitchers
        for _, r in pit_df.iterrows():
            p = normalize_name(r["Player"]); pinit(p)
            # treat any pitching stat > 0 as an appearance
            played = any(_to_float(r.get(k, 0)) > 0 for k in pit_sum[p].keys())
            if played: pit_g[p] += 1
            for k in pit_sum[p]:
                v = _to_float(r.get(k, 0))
                if not np.isnan(v):
                    pit_sum[p][k]  += v
                    pit_sum2[p][k] += v*v
        prog.progress(j/len(eids))

    # Build per-game avgs + SDs
    bat_rows = []
    for p, sums in bat_sum.items():
        g = max(1, bat_g[p])
        row = {"Player": p, "g": g}
        for k, s in sums.items():
            row[f"mu_{k.lower()}"] = s / g
            row[f"sd_{k.lower()}"] = sample_sd(s, bat_sum2[p][k], g, floor=0.15)
        # composite H+R+RBI
        row["mu_hrr"] = row["mu_h"] + row["mu_r"] + row["mu_rbi"]
        row["sd_hrr"] = math.sqrt(row["sd_h"]**2 + row["sd_r"]**2 + row["sd_rbi"]**2)
        bat_rows.append(row)
    pit_rows = []
    for p, sums in pit_sum.items():
        g = max(1, pit_g[p])
        row = {"Player": p, "g": g}
        for k, s in sums.items():
            row[f"mu_{k.lower()}"] = s / g
            row[f"sd_{k.lower()}"] = sample_sd(s, pit_sum2[p][k], g, floor=0.15)
        # convert outs->IP if needed (keep outs per game too)
        pit_rows.append(row)

    bat_proj = pd.DataFrame(bat_rows)
    pit_proj = pd.DataFrame(pit_rows)
    return bat_proj, pit_proj

# ---------------- Step 2 — build projections ----------------
st.header("2) Build per-player projections from ESPN 🔗")
if st.button("📥 Build MLB projections"):
    bat_proj, pit_proj = build_mlb_averages(d0, d1)
    if bat_proj.empty and pit_proj.empty:
        st.error("No data returned from ESPN for this date range.")
    else:
        st.success(f"Built projections — Batters: {len(bat_proj)} | Pitchers: {len(pit_proj)}")
        st.session_state["bat_proj"] = bat_proj
        st.session_state["pit_proj"] = pit_proj
        with st.expander("Preview — Batters (per-game μ / σ)"):
            cols = ["Player","g","mu_h","sd_h","mu_tb","sd_tb","mu_hr","sd_hr","mu_r","sd_r","mu_rbi","sd_rbi","mu_hrr","sd_hrr"]
            st.dataframe(bat_proj[cols].sort_values("mu_tb", ascending=False).head(25), use_container_width=True)
        with st.expander("Preview — Pitchers (per-game μ / σ)"):
            cols = ["Player","g","mu_so_pitch","sd_so_pitch","mu_h_allowed","sd_h_allowed","mu_bb_allowed","sd_bb_allowed","mu_er","sd_er","mu_outs","sd_outs","mu_win","sd_win"]
            # ensure win rate exists (if not, set zeros)
            for c in cols:
                if c not in pit_proj.columns:
                    pit_proj[c] = 0.0
            st.dataframe(pit_proj[cols].sort_values("mu_so_pitch", ascending=False).head(25), use_container_width=True)

# ---------------- Odds API ----------------
st.header("3) Pick a game & markets from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets to fetch", VALID_MLB_MARKETS, default=["batter_hits","batter_total_bases","pitcher_strikeouts"])

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_mlb_events(api_key: str, lookahead_days: int, region: str):
    return odds_get("https://api.the-odds-api.com/v4/sports/baseball_mlb/events",
                    {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region})

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds",
                    {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"})

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
st.header("4) Fetch lines & simulate")
go = st.button("🎲 Fetch lines & simulate (MLB)")
if go:
    bat_proj = st.session_state.get("bat_proj", pd.DataFrame())
    pit_proj = st.session_state.get("pit_proj", pd.DataFrame())
    if bat_proj.empty and pit_proj.empty:
        st.warning("Build MLB projections first (Step 2)."); st.stop()

    try:
        data = fetch_event_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}"); st.stop()

    # Collate bookmaker outcomes -> median line per (market, player, side)
    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                side = o.get("name")
                point = o.get("point")
                # For Yes/No markets (pitcher_record_a_win), point can be None
                if mkey not in VALID_MLB_MARKETS or not name or not side:
                    continue
                rows.append({
                    "market": mkey,
                    "player": name,
                    "side": side,
                    "point": (None if point is None else float(point)),
                })
    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    props = (pd.DataFrame(rows)
               .groupby(["market","player","side"], as_index=False)
               .agg(line=("point","median"), n_books=("point","size")))

    out = []
    bat_idx = {normalize_name(p): i for i, p in enumerate(bat_proj["Player"])} if not bat_proj.empty else {}
    pit_idx = {normalize_name(p): i for i, p in enumerate(pit_proj["Player"])} if not pit_proj.empty else {}

    for _, r in props.iterrows():
        market, player, side, line = r["market"], r["player"], r["side"], r["line"]

        # --- batters ---
        if player in bat_idx:
            row = bat_proj.iloc[bat_idx[player]]
            if market == "batter_hits":
                mu, sd = float(row["mu_h"]), float(row["sd_h"])
            elif market == "batter_total_bases":
                mu, sd = float(row["mu_tb"]), float(row["sd_tb"])
            elif market == "batter_home_runs":
                mu, sd = float(row["mu_hr"]), float(row["sd_hr"])
            elif market == "batter_rbis":
                mu, sd = float(row["mu_rbi"]), float(row["sd_rbi"])
            elif market == "batter_runs_scored":
                mu, sd = float(row["mu_r"]), float(row["sd_r"])
            elif market == "batter_hits_runs_rbis":
                mu, sd = float(row["mu_hrr"]), float(row["sd_hrr"])
            elif market == "batter_singles":
                mu, sd = float(row["mu_1b"]), float(row["sd_1b"])
            elif market == "batter_doubles":
                mu, sd = float(row["mu_2b"]), float(row["sd_2b"])
            elif market == "batter_triples":
                mu, sd = float(row["mu_3b"]), float(row["sd_3b"])
            elif market == "batter_walks":
                mu, sd = float(row["mu_bb"]), float(row["sd_bb"])
            elif market == "batter_strikeouts":
                mu, sd = float(row["mu_so"]), float(row["sd_so"])
            elif market == "batter_stolen_bases":
                mu, sd = float(row["mu_sb"]), float(row["sd_sb"])
            else:
                continue

            if pd.isna(line):
                continue
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            out.append({
                "market": market, "player": player, "side": side,
                "line": None if pd.isna(line) else round(float(line), 2),
                "μ (per-game)": round(mu, 3), "σ (per-game)": round(sd, 3),
                "Win Prob %": round(100*p, 2), "books": int(r["n_books"])
            })
            continue

        # --- pitchers ---
        if player in pit_idx:
            row = pit_proj.iloc[pit_idx[player]]
            if market == "pitcher_strikeouts":
                mu, sd = float(row["mu_so_pitch"]), float(row["sd_so_pitch"])
                if pd.isna(line): continue
                p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
                p = p_over if side == "Over" else 1.0 - p_over
            elif market == "pitcher_hits_allowed":
                mu, sd = float(row["mu_h_allowed"]), float(row["sd_h_allowed"])
                if pd.isna(line): continue
                p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
                p = p_over if side == "Over" else 1.0 - p_over
            elif market == "pitcher_walks":
                mu, sd = float(row["mu_bb_allowed"]), float(row["sd_bb_allowed"])
                if pd.isna(line): continue
                p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
                p = p_over if side == "Over" else 1.0 - p_over
            elif market == "pitcher_earned_runs":
                mu, sd = float(row["mu_er"]), float(row["sd_er"])
                if pd.isna(line): continue
                p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
                p = p_over if side == "Over" else 1.0 - p_over
            elif market == "pitcher_outs":
                mu, sd = float(row["mu_outs"]), float(row["sd_outs"])
                if pd.isna(line): continue
                p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
                p = p_over if side == "Over" else 1.0 - p_over
            elif market == "pitcher_record_a_win":
                # Yes/No market — probability from per-appearance win rate
                mu = float(row.get("mu_win", 0.0))
                p_yes = bernoulli_yes_prob(mu)
                p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
                line = None; sd = float("nan")
            else:
                continue

            out.append({
                "market": market, "player": player, "side": side,
                "line": None if line is None or pd.isna(line) else round(float(line), 2),
                "μ (per-game)": None if np.isnan(mu) else round(mu, 3),
                "σ (per-game)": None if (isinstance(sd, float) and np.isnan(sd)) else round(sd, 3),
                "Win Prob %": round(100*p, 2),
                "books": int(r["n_books"])
            })

    if not out:
        st.warning("Could not match any props to built projections.")
        st.stop()

    results = (pd.DataFrame(out)
                 .drop_duplicates(subset=["market","player","side"])
                 .sort_values(["market","Win Prob %"], ascending=[True, False])
                 .reset_index(drop=True))

    st.subheader("Results")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "player": st.column_config.TextColumn("Player", width="medium"),
            "side": st.column_config.TextColumn("Side", width="small"),
            "line": st.column_config.NumberColumn("Line", format="%.2f"),
            "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.2f"),
            "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
            "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
            "books": st.column_config.NumberColumn("#Books", width="small"),
        },
    )

    st.download_button(
        "⬇️ Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="mlb_props_sim_results.csv",
        mime="text/csv",
    )
