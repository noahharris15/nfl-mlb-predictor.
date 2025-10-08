# pages/3_Soccer.py
# Streamlit multi-page: Soccer props via The Odds API + soccerdata/FBref
# Run whole app: streamlit run App.py

import math
import sys, subprocess, importlib
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
import requests
import re
import unicodedata

# ----------------------- Robust auto-installs -----------------------
def _ensure_pkg(pkg_name, pip_name=None):
    try:
        importlib.import_module(pkg_name)
        return True
    except ImportError:
        if st.session_state.get(f"__installed_{pkg_name}", False):
            return False
        with st.spinner(f"Installing {pip_name or pkg_name}…"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg_name])
                st.session_state[f"__installed_{pkg_name}"] = True
                importlib.invalidate_caches()
                importlib.import_module(pkg_name)
                return True
            except Exception as e:
                st.error(f"Auto-install failed for {pip_name or pkg_name}: {e}")
                return False

ok = True
ok &= _ensure_pkg("soccerdata", "soccerdata>=1.8.7")
ok &= _ensure_pkg("bs4", "beautifulsoup4>=4.12")
ok &= _ensure_pkg("lxml", "lxml>=4.9")
ok &= _ensure_pkg("html5lib", "html5lib>=1.1")
ok &= _ensure_pkg("requests_cache", "requests-cache>=1.2.1")
if not ok:
    st.stop()

from soccerdata import FBref  # now safe to import

# ----------------------- Page setup -----------------------
st.set_page_config(page_title="Soccer Player Props — Odds API + soccerdata", layout="wide")
st.title("⚽ Soccer Player Props — Odds API + soccerdata/FBref")

SIM_TRIALS = 10_000
VALID_MARKETS = [
    "player_goal_scorer_anytime",
    "player_first_goal_scorer",
    "player_last_goal_scorer",
    "player_to_receive_card",
    "player_to_receive_red_card",
    "player_shots_on_target",
    "player_shots",
    "player_assists",
]

# ----------------------- Helpers -----------------------
def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.replace("–","-")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining(c))
    return n

def fuzzy_pick(name: str, candidates: List[str], cutoff=86) -> Optional[str]:
    if not candidates: return None
    res = process.extractOne(normalize_name(name), [normalize_name(x) for x in candidates], scorer=fuzz.token_sort_ratio)
    return candidates[res[2]] if res and res[1] >= cutoff else None

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:250]}")
    return r.json()

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def poisson_yes(lam: float) -> float:
    lam = max(1e-8, float(lam))
    return 1.0 - math.exp(-lam)

# ----------------------- 1) Scope -----------------------
st.header("1) Choose leagues & date window")
cols = st.columns([1,1,1,1])
with cols[0]:
    # Common domestic leagues + MLS + UCL
    leagues = st.multiselect(
        "Leagues",
        [
            "ENG-Premier League",
            "ESP-La Liga",
            "ITA-Serie A",
            "GER-Bundesliga",
            "FRA-Ligue 1",
            "USA-MLS",
            "UEFA-Champions League",
        ],
        default=["ENG-Premier League"]
    )
with cols[1]:
    d1 = st.date_input("Start date", value=(datetime.utcnow() - timedelta(days=14)).date(), format="YYYY/MM/DD")
with cols[2]:
    d2 = st.date_input("End date", value=datetime.utcnow().date(), format="YYYY/MM/DD")
with cols[3]:
    minutes_floor = st.number_input("Min minutes to count a game", 1, 90, value=15)

st.caption("We compute **per-game averages** between these dates (inclusive).")

# ----------------------- 2) Build averages -----------------------
st.header("2) Build per-player averages from soccerdata / FBref")
build = st.button("📥 Build Soccer projections")

@st.cache_data(show_spinner=True)
def fetch_fbref_player_game_logs(leagues: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for lg in leagues:
        try:
            # season string: FBref expects e.g. '2024-2025' but game logs carry dates; we’ll pull both current + prior season
            this_year = datetime.strptime(start_date, "%Y-%m-%d").year
            # Cover two adjacent seasons to be safe around summer breaks
            seasons = [f"{this_year-1}-{this_year}", f"{this_year}-{this_year+1}"]
            fb = FBref(leagues=[lg], seasons=seasons, no_cache=True)
            # Player match logs (standard + shooting + passing + cards)
            std = fb.read_player_match_stats(stat_type="standard")     # minutes, goals, cards (y/r)
            shooting = fb.read_player_match_stats(stat_type="shooting") # shots, sot
            passing = fb.read_player_match_stats(stat_type="passing")   # assists in standard too, but keep
            # Merge on identifiers
            df = std.merge(shooting, on=["player_id","player","date","team","opponent","competition","match_report"], how="left", suffixes=("","_shot"))
            df = df.merge(passing[["player_id","date","team","assists"]], on=["player_id","date","team"], how="left")
            df["league"] = lg
            frames.append(df)
        except Exception as e:
            st.warning(f"FBref fetch warning for {lg}: {e}")
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    # Date filter
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    mask = (all_df["date"] >= pd.to_datetime(start_date)) & (all_df["date"] <= pd.to_datetime(end_date))
    return all_df.loc[mask].copy()

@st.cache_data(show_spinner=False)
def build_player_avgs(leagues: List[str], d1: datetime.date, d2: datetime.date, minutes_floor: int) -> pd.DataFrame:
    logs = fetch_fbref_player_game_logs(leagues, str(d1), str(d2))
    if logs.empty:
        return pd.DataFrame()

    # Defensive columns names vary by soccerdata version; normalize best-effort
    # Standard table usually has:
    # 'player','minutes','goals','assists','shots','shots_on_target','cards_yellow','cards_red'
    rename_map = {
        "minutes": "min",
        "goals": "goals",
        "assists": "assists",
        "shots": "shots",
        "shots_on_target": "sot",
        "cards_yellow": "yc",
        "cards_red": "rc",
    }
    # Coerce to numeric
    for col in rename_map:
        if col in logs.columns:
            logs[col] = pd.to_numeric(logs[col], errors="coerce")
    # Some versions label shots columns differently
    for alt in ["sh", "shots total", "shots_total"]:
        if "shots" not in logs.columns and alt in logs.columns:
            logs["shots"] = pd.to_numeric(logs[alt], errors="coerce")
    for alt in ["sot", "shots on target", "shots_on_target_total"]:
        if "shots_on_target" not in logs.columns and alt in logs.columns:
            logs["shots_on_target"] = pd.to_numeric(logs[alt], errors="coerce")
    # Cards fallbacks
    if "cards_yellow" not in logs.columns and "yellow_cards" in logs.columns:
        logs["cards_yellow"] = pd.to_numeric(logs["yellow_cards"], errors="coerce")
    if "cards_red" not in logs.columns and "red_cards" in logs.columns:
        logs["cards_red"] = pd.to_numeric(logs["red_cards"], errors="coerce")

    logs["min"] = pd.to_numeric(logs.get("minutes", logs.get("min", np.nan)), errors="coerce")
    logs["player"] = logs["player"].astype(str)

    use = logs[logs["min"].fillna(0) >= float(minutes_floor)].copy()
    if use.empty:
        return pd.DataFrame()

    grp = use.groupby("player", dropna=False).agg(
        g=("player","size"),
        goals=("goals","sum"),
        assists=("assists","sum"),
        shots=("shots","sum"),
        sot=("shots_on_target","sum"),
        yc=("cards_yellow","sum"),
        rc=("cards_red","sum"),
        minutes=("min","sum"),
    ).reset_index()

    # Per-game averages
    grp["mu_goals"] = grp["goals"] / grp["g"]
    grp["mu_assists"] = grp["assists"] / grp["g"]
    grp["mu_shots"] = grp["shots"] / grp["g"]
    grp["mu_sot"] = grp["sot"] / grp["g"]
    grp["mu_yc"] = grp["yc"] / grp["g"]
    grp["mu_rc"] = grp["rc"] / grp["g"]

    # SDs for O/U (empirical with Bessel)
    def sd_from_logs(name):
        s = use.groupby("player")[name].std(ddof=1)
        return s

    for col, out in [("shots","sd_shots"), ("shots_on_target","sd_sot"), ("assists","sd_ast")]:
        if col in use.columns:
            s = sd_from_logs(col)
            grp[out] = grp["player"].map(s).fillna(1.0) * 1.05
        else:
            grp[out] = 1.0

    grp.rename(columns={"player":"Player"}, inplace=True)
    grp["Player_norm"] = grp["Player"].map(normalize_name)
    return grp

if build:
    avgs = build_player_avgs(leagues, d1, d2, minutes_floor)
    if avgs.empty:
        st.error("No data returned from soccerdata/FBref for this selection.")
        st.stop()
    st.session_state["soccer_avgs"] = avgs
    st.success(f"Built {len(avgs)} per-player averages.")
    st.dataframe(avgs[["Player","g","mu_goals","mu_assists","mu_shots","mu_sot","mu_yc","mu_rc"]]
                 .sort_values("g", ascending=False).head(50),
                 use_container_width=True)

# ----------------------- 3) Odds API game picker -----------------------
st.header("3) Pick a game & markets from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets to fetch", VALID_MARKETS, default=VALID_MARKETS)

def list_soccer_events(api_key: str, lookahead_days: int, region: str):
    # Use generic soccer endpoint (covers top leagues; Odds API groups many under soccer)
    return odds_get("https://api.the-odds-api.com/v4/sports/soccer/events",
                    {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region})

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/soccer/events/{event_id}/odds",
                    {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"})

events = []
if api_key:
    try:
        events = list_soccer_events(api_key, lookahead, region)
    except Exception as e:
        st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list soccer matches.")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Match", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ----------------------- 4) Simulate -----------------------
st.header("4) Fetch lines & simulate")
go = st.button("🎲 Fetch lines & simulate (Soccer)")

if go:
    avgs = st.session_state.get("soccer_avgs", pd.DataFrame())
    if avgs.empty:
        st.warning("Build Soccer projections first (Step 2).")
        st.stop()

    try:
        data = fetch_event_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    # Flatten bookmaker outcomes, median line per (market, player, side)
    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = str(m.get("key"))
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description") or o.get("participant") or "")
                side = o.get("name")  # Over/Under or Yes/No
                point = o.get("point")
                if mkey not in VALID_MARKETS or not name or not side:
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

    # Prepare name matching
    name_map = dict(zip(avgs["Player_norm"], avgs["Player"]))
    out = []

    for _, r in props_df.iterrows():
        market, pnorm, side, line = r["market"], r["player_norm"], r["side"], r["line"]
        # Direct match or fuzzy
        match_key = pnorm if pnorm in name_map else fuzzy_pick(pnorm, list(name_map.keys()), cutoff=86)
        if not match_key:
            continue
        player = name_map[match_key]
        row = avgs.loc[avgs["Player"] == player].iloc[0]

        # Build model inputs from per-game averages
        if market == "player_shots" and pd.notna(line):
            mu = float(row["mu_shots"]); sd = float(row["sd_shots"])
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            out.append({"market":market,"player":player,"side":side,"line":line,
                        "μ (per-game)":round(mu,2),"σ (per-game)":round(sd,2),"Win Prob %":round(100*p,2),"books":int(r["n_books"])})

        elif market == "player_shots_on_target" and pd.notna(line):
            mu = float(row["mu_sot"]); sd = float(row["sd_sot"])
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            out.append({"market":market,"player":player,"side":side,"line":line,
                        "μ (per-game)":round(mu,2),"σ (per-game)":round(sd,2),"Win Prob %":round(100*p,2),"books":int(r["n_books"])})

        elif market == "player_assists" and pd.notna(line):
            mu = float(row["mu_assists"]); sd = float(row["sd_ast"])
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            out.append({"market":market,"player":player,"side":side,"line":line,
                        "μ (per-game)":round(mu,3),"σ (per-game)":round(sd,3),"Win Prob %":round(100*p,2),"books":int(r["n_books"])})

        elif market in ("player_goal_scorer_anytime","player_first_goal_scorer","player_last_goal_scorer"):
            # Poisson for goals per game; “first/last” approximated by scaling anytime (heuristic)
            lam = float(row["mu_goals"])
            p_any = poisson_yes(lam)
            if market == "player_goal_scorer_anytime":
                p = p_any if side in ("Yes","Over") else (1.0 - p_any)
            else:
                # crude: assume first/last ≈ p_any * 0.35 (depends on team scoring share & match pace)
                scale = 0.35
                p_fl = min(max(p_any * scale, 0.0), 1.0)
                p = p_fl if side in ("Yes","Over") else (1.0 - p_fl)
            out.append({"market":market,"player":player,"side":side,"line":None,
                        "μ (per-game)":round(lam,3),"σ (per-game)":None,"Win Prob %":round(100*p,2),"books":int(r["n_books"])})

        elif market == "player_to_receive_card":
            lam = float(row["mu_yc"])
            p = poisson_yes(lam) if side in ("Yes","Over") else (1.0 - poisson_yes(lam))
            out.append({"market":market,"player":player,"side":side,"line":None,
                        "μ (per-game)":round(lam,3),"σ (per-game)":None,"Win Prob %":round(100*p,2),"books":int(r["n_books"])})

        elif market == "player_to_receive_red_card":
            lam = float(row["mu_rc"])
            p = poisson_yes(lam) if side in ("Yes","Over") else (1.0 - poisson_yes(lam))
            out.append({"market":market,"player":player,"side":side,"line":None,
                        "μ (per-game)":round(lam,4),"σ (per-game)":None,"Win Prob %":round(100*p,2),"books":int(r["n_books"])})

        else:
            continue

    if not out:
        st.warning("Could not match props to any players with averages.")
        st.stop()

    results = (pd.DataFrame(out)
               .drop_duplicates(subset=["market","player","side"])
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    st.subheader("Results")
    colcfg = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.3f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.3f"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "books": st.column_config.NumberColumn("#Books", width="small"),
    }
    st.dataframe(results, use_container_width=True, hide_index=True, column_config=colcfg)

    # Quick “Top Picks” chart (excluding Yes/No with no line is fine)
    top = results.sort_values("Win Prob %", ascending=False).head(15)
    st.bar_chart(top.set_index("player")["Win Prob %"], use_container_width=True)

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="soccer_props_results.csv",
        mime="text/csv",
    )
