# pages/2_MLB.py
# MLB Player Props — Odds API + ESPN (batters + pitchers)
# Appears as a separate page in Streamlit multi-page apps.

import re, math, unicodedata
from io import StringIO
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ------------------ Page header (no set_page_config on subpage) ------------------
st.title("⚾ MLB Player Props — Odds API + ESPN (batters + pitchers)")

SIM_TRIALS = 10_000
ODDS_SPORT = "baseball_mlb"

# Preferred MLB markets we support (must also be supported by your region/book)
PREFERRED_MLB_MARKETS = [
    # batters
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
    "batter_rbis",
    "batter_runs_scored",
    # pitchers
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_walks",
    "pitcher_earned_runs",
    "pitcher_outs",
    "pitcher_record_a_win",
]

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

def to_float(x):
    try:
        if isinstance(x, str) and x.strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def beta_smooth_success_rate(success: float, total: float, a: float = 1.5, b: float = 1.5) -> float:
    # Beta prior to avoid 0/1 extremes
    return (success + a) / (total + a + b)

# ------------------ ESPN crawl ------------------
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"

def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def list_week_event_ids(year: int, week: int) -> List[str]:
    js = http_get(SCOREBOARD, params={"year": year, "week": week})
    if not js:
        return []
    return [str(e.get("id")) for e in js.get("events", []) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id: str) -> Optional[dict]:
    return http_get(SUMMARY, params={"event": event_id})

def _ip_to_outs(ip_str: str) -> float:
    """Convert ESPN IP string like '6.2' to outs (6*3 + 2)."""
    try:
        s = str(ip_str)
        if ":" in s:  # handle '6:2' just in case
            whole, frac = s.split(":")
            return int(whole) * 3 + int(frac)
        if "." in s:
            whole, frac = s.split(".")
            return int(whole) * 3 + int(frac)
        return int(float(s)) * 3
    except Exception:
        return float("nan")

def _extract_competitor_winners(box: dict) -> Dict[str, bool]:
    """Map team shortDisplayName to winner True/False."""
    winners = {}
    try:
        comp = box.get("header", {}).get("competitions", [])[0]
        for c in comp.get("competitors", []):
            team = c.get("team", {})
            winners[team.get("shortDisplayName")] = bool(c.get("winner"))
    except Exception:
        pass
    return winners

def _extract_team_players_with_labels(box: dict) -> List[dict]:
    """
    Pull out per-team stat sections with labels so we can map safely.
    Each record: {"team": "...", "section": "batting"/"pitching", "labels": [...], "ath": {...}}
    """
    out = []
    try:
        sec = box.get("boxscore", {}).get("players", [])
        for team in sec:
            team_name = team.get("team", {}).get("shortDisplayName")
            for stat_block in team.get("statistics", []):
                section = (stat_block.get("name") or "").lower()  # "batting" / "pitching"
                labels = [str(x).lower() for x in stat_block.get("labels", [])]
                for a in stat_block.get("athletes", []):
                    out.append({
                        "team": team_name,
                        "section": section,
                        "labels": labels,
                        "ath": a
                    })
    except Exception:
        pass
    return out

def parse_boxscore_players(box: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (batters_df, pitchers_df) aggregated per game.
    Batters: hits, TB, HR, RBI, R
    Pitchers: K, H, BB, ER, Outs, Win(0/1, heuristic: starter + team win + >=15 outs)
    """
    rows = _extract_team_players_with_labels(box)
    winners = _extract_competitor_winners(box)

    bat_rows, pit_rows = [], []
    for r in rows:
        team = r.get("team")
        section = r.get("section")
        labels = r.get("labels", [])
        a = r.get("ath", {})
        name = normalize_name(a.get("athlete", {}).get("displayName"))
        stats = a.get("stats") or []
        starter = bool(a.get("starter"))

        # Make a dict label->value for safer indexing
        lbl_map = {labels[i]: stats[i] for i in range(min(len(labels), len(stats)))}

        if section == "batting":
            hits = to_float(lbl_map.get("h"))
            tb   = to_float(lbl_map.get("tb"))
            hr   = to_float(lbl_map.get("hr"))
            rbi  = to_float(lbl_map.get("rbi"))
            runs = to_float(lbl_map.get("r"))
            if not all(np.isnan([hits, tb, hr, rbi, runs])):
                bat_rows.append({
                    "Player": name, "H": hits, "TB": tb, "HR": hr, "RBI": rbi, "R": runs
                })

        elif section == "pitching":
            ip    = lbl_map.get("ip")
            outs  = _ip_to_outs(ip) if ip is not None else float("nan")
            so    = to_float(lbl_map.get("k"))
            bb    = to_float(lbl_map.get("bb"))
            h     = to_float(lbl_map.get("h"))
            er    = to_float(lbl_map.get("er"))

            # Heuristic "win" credit: starter on winner & >= 15 outs (5.0 IP)
            win = float(
                starter and winners.get(team, False) and (not np.isnan(outs) and outs >= 15)
            )

            if not all(np.isnan([outs, so, bb, h, er])):
                pit_rows.append({
                    "Player": name, "OUTS": outs, "K": so, "BB": bb, "H": h, "ER": er, "WIN": win
                })

    bat_df = pd.DataFrame(bat_rows)
    pit_df = pd.DataFrame(pit_rows)

    # Group per player for this game (rare dup rows)
    if not bat_df.empty:
        bat_df = bat_df.groupby("Player", as_index=False).sum(numeric_only=True)
    if not pit_df.empty:
        pit_df = pit_df.groupby("Player", as_index=False).sum(numeric_only=True)

    return bat_df, pit_df

@st.cache_data(show_spinner=True)
def build_espn_season_agg(year: int, weeks: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crawl ESPN box scores and return (batters_totals, pitchers_totals) for chosen weeks.
    Each includes totals, sums of squares, and games-played counters for SD computation.
    """
    bat_totals, bat_sumsq, bat_games = {}, {}, {}
    pit_totals, pit_sumsq, pit_games = {}, {}, {}

    def init_b(p):
        if p not in bat_totals:
            bat_totals[p] = {"H": 0.0, "TB": 0.0, "HR": 0.0, "RBI": 0.0, "R": 0.0}
            bat_sumsq[p]  = {"H": 0.0, "TB": 0.0, "HR": 0.0, "RBI": 0.0, "R": 0.0}
            bat_games[p]  = 0

    def init_p(p):
        if p not in pit_totals:
            pit_totals[p] = {"OUTS": 0.0, "K": 0.0, "BB": 0.0, "H": 0.0, "ER": 0.0, "WIN": 0.0}
            pit_sumsq[p]  = {"OUTS": 0.0, "K": 0.0, "BB": 0.0, "H": 0.0, "ER": 0.0}
            pit_games[p]  = 0

    events = []
    for wk in weeks:
        events.extend(list_week_event_ids(year, wk))

    if not events:
        return pd.DataFrame(), pd.DataFrame()

    prog = st.progress(0.0, text=f"Crawling {len(events)} games...")
    for j, ev in enumerate(events, 1):
        box = fetch_boxscore_event(ev)
        if not box:
            prog.progress(j/len(events)); continue

        bat_df, pit_df = parse_boxscore_players(box)

        if not bat_df.empty:
            for _, r in bat_df.iterrows():
                p = r["Player"]; init_b(p)
                played = any(to_float(r[k]) > 0 for k in bat_totals[p].keys())
                if played: bat_games[p] += 1
                for k in bat_totals[p]:
                    v = to_float(r.get(k, 0))
                    if not np.isnan(v):
                        bat_totals[p][k] += v
                        bat_sumsq[p][k]  += v*v

        if not pit_df.empty:
            for _, r in pit_df.iterrows():
                p = r["Player"]; init_p(p)
                played = any(to_float(r[k]) > 0 for k in ["OUTS","K","BB","H","ER"])
                if played: pit_games[p] += 1
                for k in ["OUTS","K","BB","H","ER"]:
                    v = to_float(r.get(k, 0))
                    if not np.isnan(v):
                        pit_totals[p][k] += v
                        pit_sumsq[p][k]  += v*v
                pit_totals[p]["WIN"] += to_float(r.get("WIN", 0))

        prog.progress(j/len(events))

    # Build output frames
    bat_rows, pit_rows = [], []
    for p, stat in bat_totals.items():
        g = max(1, int(bat_games.get(p, 0)))
        bat_rows.append({"Player": p, "g": g, **stat, **{f"sq_{k}": bat_sumsq[p][k] for k in stat.keys()}})

    for p, stat in pit_totals.items():
        g = max(1, int(pit_games.get(p, 0)))
        pit_rows.append({"Player": p, "g": g, **stat, **{f"sq_{k}": pit_sumsq[p].get(k, 0.0) for k in ["OUTS","K","BB","H","ER"]}})

    return pd.DataFrame(bat_rows), pd.DataFrame(pit_rows)

# ------------------ Odds API helpers ------------------
def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_supported_markets(api_key: str) -> List[str]:
    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/odds-markets"
    try:
        r = requests.get(url, params={"apiKey": api_key}, timeout=20)
        if r.status_code == 200:
            return [str(x) for x in r.json()]
    except Exception:
        pass
    return []

def list_events(api_key: str, lookahead_days: int, region: str):
    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events"
    return odds_get(url, {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region})

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    base = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds"
    return odds_get(base, {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"})

# ------------------ UI: scope ------------------
st.markdown("### 1) Season scope")
c1, c2 = st.columns([1,1])
with c1:
    season = st.number_input("Season (year)", min_value=2017, max_value=2100, value=2025, step=1)
with c2:
    week_range = st.slider("Weeks", 1, 27, (1, 27))
weeks = list(range(week_range[0], week_range[1] + 1))

# ------------------ Build projections ------------------
st.markdown("### 2) Build per-player projections from ESPN")
if st.button("📥 Build MLB projections"):
    bat_df, pit_df = build_espn_season_agg(season, weeks)
    if bat_df.empty and pit_df.empty:
        st.error("No data returned from ESPN."); st.stop()

    # ====== Batters: per-game means and SDs ======
    if not bat_df.empty:
        g = bat_df["g"].astype(float).clip(lower=1)
        # per-game means
        bat_df["mu_hits_raw"]  = bat_df["H"]  / g
        bat_df["mu_tb_raw"]    = bat_df["TB"] / g
        bat_df["mu_hr_raw"]    = bat_df["HR"] / g
        bat_df["mu_rbi_raw"]   = bat_df["RBI"]/ g
        bat_df["mu_runs_raw"]  = bat_df["R"]  / g

        # sample SDs (Bessel), minimum safeguards
        def sample_sd(sum_x, sum_x2, n):
            n = int(n)
            if n <= 1: return np.nan
            m = sum_x / n
            var = (sum_x2 / n) - (m*m)
            var = var * (n / (n - 1))
            return float(np.sqrt(max(var, 1e-6)))

        bat_df["sd_hits"] =  bat_df.apply(lambda r: sample_sd(r["H"],  r["sq_H"],  r["g"]), axis=1)
        bat_df["sd_tb"]   =  bat_df.apply(lambda r: sample_sd(r["TB"], r["sq_TB"], r["g"]), axis=1)
        bat_df["sd_hr"]   =  bat_df.apply(lambda r: sample_sd(r["HR"], r["sq_HR"], r["g"]), axis=1)
        bat_df["sd_rbi"]  =  bat_df.apply(lambda r: sample_sd(r["RBI"],r["sq_RBI"],r["g"]), axis=1)
        bat_df["sd_runs"] =  bat_df.apply(lambda r: sample_sd(r["R"],  r["sq_R"],  r["g"]), axis=1)

        # modest inflate SD to avoid overconfidence
        INF = 1.15
        for c in ["sd_hits","sd_tb","sd_hr","sd_rbi","sd_runs"]:
            bat_df[c] = np.where(np.isnan(bat_df[c]), np.nan, bat_df[c] * INF)

        st.session_state["bat_proj"] = bat_df[[
            "Player","mu_hits_raw","sd_hits","mu_tb_raw","sd_tb","mu_hr_raw","sd_hr","mu_rbi_raw","sd_rbi",
            "mu_runs_raw","sd_runs"
        ]].copy()

    # ====== Pitchers: per-start means and SDs ======
    if not pit_df.empty:
        g = pit_df["g"].astype(float).clip(lower=1)
        pit_df["mu_k_raw"]   = pit_df["K"]    / g
        pit_df["mu_bb_raw"]  = pit_df["BB"]   / g
        pit_df["mu_h_raw"]   = pit_df["H"]    / g
        pit_df["mu_er_raw"]  = pit_df["ER"]   / g
        pit_df["mu_outs_raw"]= pit_df["OUTS"] / g
        pit_df["p_win_raw"]  = pit_df.apply(lambda r: beta_smooth_success_rate(r["WIN"], r["g"], 1.5, 1.5), axis=1)

        def sample_sd(sum_x, sum_x2, n):
            n = int(n)
            if n <= 1: return np.nan
            m = sum_x / n
            var = (sum_x2 / n) - (m*m)
            var = var * (n / (n - 1))
            return float(np.sqrt(max(var, 1e-6)))

        pit_df["sd_k"]    = pit_df.apply(lambda r: sample_sd(r["K"],    r["sq_K"],    r["g"]), axis=1)
        pit_df["sd_bb"]   = pit_df.apply(lambda r: sample_sd(r["BB"],   r["sq_BB"],   r["g"]), axis=1)
        pit_df["sd_h"]    = pit_df.apply(lambda r: sample_sd(r["H"],    r["sq_H"],    r["g"]), axis=1)
        pit_df["sd_er"]   = pit_df.apply(lambda r: sample_sd(r["ER"],   r["sq_ER"],   r["g"]), axis=1)
        pit_df["sd_outs"] = pit_df.apply(lambda r: sample_sd(r["OUTS"], r["sq_OUTS"], r["g"]), axis=1)

        INF = 1.15
        for c in ["sd_k","sd_bb","sd_h","sd_er","sd_outs"]:
            pit_df[c] = np.where(np.isnan(pit_df[c]), np.nan, pit_df[c] * INF)

        st.session_state["pit_proj"] = pit_df[[
            "Player","mu_k_raw","sd_k","mu_bb_raw","sd_bb","mu_h_raw","sd_h","mu_er_raw","sd_er",
            "mu_outs_raw","sd_outs","p_win_raw"
        ]].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Batters (derived)")
        st.dataframe(st.session_state.get("bat_proj", pd.DataFrame()).head(12), use_container_width=True)
    with c2:
        st.subheader("Pitchers (derived)")
        st.dataframe(st.session_state.get("pit_proj", pd.DataFrame()).head(12), use_container_width=True)

# ------------------ Odds API: pick game/markets ------------------
st.markdown("### 3) Pick a game & markets (Odds API)")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)

supported = list_supported_markets(api_key) if api_key else []
valid_markets = [m for m in PREFERRED_MLB_MARKETS if m in supported]
if not valid_markets and supported:
    valid_markets = supported

if supported:
    with st.expander("Supported markets for MLB (from Odds API)"):
        st.write(sorted(supported))

markets = st.multiselect("Markets to fetch", valid_markets, default=valid_markets[:4])

events = []
if api_key:
    try:
        events = list_events(api_key, lookahead, region)
    except Exception as e:
        st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list upcoming games.")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ------------------ Simulate ------------------
st.markdown("### 4) Fetch lines & simulate")
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

    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                side = o.get("name")
                point = o.get("point")
                if not name or not side or mkey not in markets:
                    continue
                rows.append({
                    "market": mkey, "player_norm": name, "side": side,
                    "point": (None if point is None else float(point))
                })

    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    df = (pd.DataFrame(rows)
          .groupby(["market","player_norm","side"], as_index=False)
          .agg(line=("point","median"), n_books=("point","size")))

    out_rows = []
    bnames = set(bat_proj["Player"]) if not bat_proj.empty else set()
    pnames = set(pit_proj["Player"]) if not pit_proj.empty else set()

    for _, r in df.iterrows():
        market, player, line, side = r["market"], r["player_norm"], r["line"], r["side"]

        # batters
        if market in ["batter_hits","batter_total_bases","batter_home_runs","batter_rbis","batter_runs_scored"] and player in bnames:
            row = bat_proj.loc[bat_proj["Player"] == player].iloc[0]
            if market == "batter_hits":
                mu, sd = float(row["mu_hits_raw"]), float(row["sd_hits"])
            elif market == "batter_total_bases":
                mu, sd = float(row["mu_tb_raw"]), float(row["sd_tb"])
            elif market == "batter_home_runs":
                mu, sd = float(row["mu_hr_raw"]), float(row["sd_hr"])
            elif market == "batter_rbis":
                mu, sd = float(row["mu_rbi_raw"]), float(row["sd_rbi"])
            else:  # runs scored
                mu, sd = float(row["mu_runs_raw"]), float(row["sd_runs"])
            if np.isnan(mu) or np.isnan(sd) or pd.isna(line): 
                continue
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else (1.0 - p_over)
            out_rows.append({
                "market": market, "player": player, "side": side,
                "line": round(float(line),2),
                "μ (per-game)": round(mu,2), "σ (per-game)": round(sd,2),
                "Win Prob %": round(100*p,2), "books": int(r["n_books"]),
            })
            continue

        # pitchers (O/U)
        if market in ["pitcher_strikeouts","pitcher_hits_allowed","pitcher_walks","pitcher_earned_runs","pitcher_outs"] and player in pnames:
            row = pit_proj.loc[pit_proj["Player"] == player].iloc[0]
            if market == "pitcher_strikeouts":
                mu, sd = float(row["mu_k_raw"]), float(row["sd_k"])
            elif market == "pitcher_hits_allowed":
                mu, sd = float(row["mu_h_raw"]), float(row["sd_h"])
            elif market == "pitcher_walks":
                mu, sd = float(row["mu_bb_raw"]), float(row["sd_bb"])
            elif market == "pitcher_earned_runs":
                mu, sd = float(row["mu_er_raw"]), float(row["sd_er"])
            else:  # outs
                mu, sd = float(row["mu_outs_raw"]), float(row["sd_outs"])
            if np.isnan(mu) or np.isnan(sd) or pd.isna(line):
                continue
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else (1.0 - p_over)
            out_rows.append({
                "market": market, "player": player, "side": side,
                "line": round(float(line),2),
                "μ (per-start)": round(mu,2), "σ (per-start)": round(sd,2),
                "Win Prob %": round(100*p,2), "books": int(r["n_books"]),
            })
            continue

        # pitchers (Yes/No: record a win)
        if market == "pitcher_record_a_win" and player in pnames:
            row = pit_proj.loc[pit_proj["Player"] == player].iloc[0]
            p_yes = float(row["p_win_raw"])  # Beta-smoothed rate
            if side in ("Yes","Over"):
                p = p_yes
            else:
                p = 1.0 - p_yes
            out_rows.append({
                "market": market, "player": player, "side": side,
                "line": None,
                "μ (per-start)": None, "σ (per-start)": None,
                "Win Prob %": round(100*p,2), "books": int(r["n_books"]),
            })
            continue

    if not out_rows:
        st.warning("No props matched projections.")
        st.stop()

    results = (pd.DataFrame(out_rows)
               .drop_duplicates(subset=["market","player","side"])
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    st.subheader("Results")
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="mlb_props_sim_results.csv", mime="text/csv")
