# pages/2_MLB.py
# MLB Props — Odds API + ESPN (robust ESPN fetch: regular+postseason; averages-based)
# Put this file under: pages/2_MLB.py  (Streamlit multipage)
# Requires: requests, pandas, numpy, streamlit, rapidfuzz

import math, time, random, re, unicodedata
from io import StringIO
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import process, fuzz

SIM_TRIALS = 10_000

# ---------------- UI ----------------
st.set_page_config(page_title="MLB Player Props — Odds API + ESPN", layout="wide")
st.title("⚾ MLB Player Props — Odds API + ESPN (RS + Postseason)")

# ---- Odds API markets (MLB correct keys) ----
VALID_MARKETS = [
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

# ---------------- helpers ----------------
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

def poisson_over_prob(lam: float, line: float, trials: int = SIM_TRIALS) -> float:
    lam = max(1e-6, float(lam))
    return float((np.random.poisson(lam=lam, size=trials) > line).mean())

def bernoulli_prob(p: float) -> float:
    return float(np.clip(p, 0.0, 1.0))

def to_float(x) -> float:
    try: return float(x)
    except Exception: return float("nan")

# ---------------- HTTP with headers + retry ----------------
def http_get_json(url, params=None, timeout=25, retries=3, backoff=0.5) -> Optional[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.espn.com/"
    }
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(backoff * (2 ** i) + random.random() * 0.25)
    return None

# ---------------- ESPN scoreboard → (event_id, competition_id) ----------------
@st.cache_data(show_spinner=False)
def list_mlb_events_by_date(yyyymmdd: str) -> List[dict]:
    """
    Returns list of dicts: {"event_id": "...", "competition_id": "...", "home": "...", "away": "..."}
    Works for both regular-season and postseason days.
    """
    js = http_get_json(
        "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
        params={"dates": yyyymmdd}
    )
    out = []
    if not js:
        return out
    for e in js.get("events", []):
        ev_id = str(e.get("id"))
        comps = e.get("competitions") or []
        comp_id = str(comps[0].get("id")) if comps else ev_id
        home = away = ""
        if comps and comps[0].get("competitors"):
            for c in comps[0]["competitors"]:
                if c.get("homeAway") == "home":
                    home = c.get("team", {}).get("shortDisplayName", "")
                if c.get("homeAway") == "away":
                    away = c.get("team", {}).get("shortDisplayName", "")
        out.append({"event_id": ev_id, "competition_id": comp_id, "home": home, "away": away})
    return out

# ---------------- Boxscore fetch (site summary → core fallback) ----------------
@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id: str, competition_id: Optional[str] = None) -> Optional[dict]:
    # 1) Try site summary
    site = http_get_json(
        "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary",
        params={"event": event_id},
    )
    if site and site.get("boxscore"):
        return site

    # 2) Fallback to core (needs competition_id for reliability)
    comp_id = competition_id or event_id
    core_url = (
        "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
        f"/events/{event_id}/competitions/{comp_id}/boxscore"
    )
    core = http_get_json(core_url)
    if core:
        return {"boxscore": core}
    return None

# ---------------- Parse MLB boxscore ----------------
def parse_mlb_boxscore_players(box: dict) -> pd.DataFrame:
    """
    Produce one row per player per game with batter & pitcher counting stats we need.
    Tries to handle both site + core shapes (we only navigate what we need).
    """
    rows = []

    try:
        bx = box.get("boxscore", {})
        # site summary shape may include "players" sections; core uses "teams" with references
        # We'll extract any "players" first:
        players_sections = bx.get("players") or []
        if players_sections:
            for team_section in players_sections:
                team_obj = team_section.get("team") or {}
                team = team_obj.get("shortDisplayName") or team_obj.get("abbreviation") or ""
                for stat_group in team_section.get("statistics", []):
                    label = (stat_group.get("name") or "").lower()
                    for a in stat_group.get("athletes", []):
                        nm = normalize_name(a.get("athlete", {}).get("displayName"))
                        vals = a.get("stats") or []
                        # batting
                        if "batting" in label:
                            # ESPN batting fields vary; typical order: AB, R, H, 2B, 3B, HR, RBI, ...
                            H  = to_float(vals[2]) if len(vals) > 2 else np.nan
                            R  = to_float(vals[1]) if len(vals) > 1 else np.nan
                            _2B = to_float(vals[3]) if len(vals) > 3 else 0.0
                            _3B = to_float(vals[4]) if len(vals) > 4 else 0.0
                            HR = to_float(vals[5]) if len(vals) > 5 else 0.0
                            RBI = to_float(vals[6]) if len(vals) > 6 else 0.0
                            TB = float(H + _2B + 2*_2B + 3*_3B + 4*HR) if not np.isnan(H) else np.nan
                            H_R_RBI = float((H if not np.isnan(H) else 0) + (R if not np.isnan(R) else 0) + (RBI if not np.isnan(RBI) else 0))
                            rows.append({
                                "Player": nm, "team": team,
                                "is_pitcher": 0,
                                "hits": H, "tb": TB, "hr": HR, "rbi": RBI, "runs": R, "hrr": H_R_RBI
                            })
                        # pitching
                        if "pitching" in label:
                            # typical: IP, H, R, ER, BB, K, HR, ...
                            IP  = a.get("minutes")  # not present here; use stats:
                            H_a = to_float(vals[1]) if len(vals) > 1 else np.nan
                            ER  = to_float(vals[3]) if len(vals) > 3 else np.nan
                            BB  = to_float(vals[4]) if len(vals) > 4 else np.nan
                            K   = to_float(vals[5]) if len(vals) > 5 else np.nan
                            # Outs: IP like "5.1" isn't given reliably via "vals"; many scoreboards provide IP string field
                            ip_str = a.get("athlete", {}).get("pitching", {}).get("inningsPitched")
                            outs = np.nan
                            if ip_str and isinstance(ip_str, str) and ip_str.replace(".", "", 1).isdigit():
                                ip = float(ip_str)
                                outs = math.floor(ip) * 3 + round((ip - math.floor(ip)) * 10)  # .1 = 1 out; .2 = 2 outs
                            rows.append({
                                "Player": nm, "team": team, "is_pitcher": 1,
                                "p_k": K, "p_hits_allowed": H_a, "p_bb": BB, "p_er": ER, "p_outs": outs,
                                "p_win": None  # set by post parse
                            })
        else:
            # minimal core shape fallback — try "teams"->"players" resources (omitted to keep simple & stable)
            pass
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=[
            "Player","team","is_pitcher",
            "hits","tb","hr","rbi","runs","hrr",
            "p_k","p_hits_allowed","p_bb","p_er","p_outs","p_win"
        ])

    df = pd.DataFrame(rows)
    # set p_win by team-level status (only reliable in site summary JSON)
    try:
        comps = box.get("boxscore", {}).get("competitions") or []
        if comps:
            comp = comps[0]
            winner_id = None
            for c in comp.get("competitors", []):
                if c.get("winner"): winner_id = c.get("id")
            id_to_team = {c.get("id"): (c.get("team", {}).get("shortDisplayName") or "") for c in comp.get("competitors", [])}
            winning_team = id_to_team.get(winner_id, "")
            df["p_win"] = np.where((df["is_pitcher"]==1) & (df["team"]==winning_team), 1.0, 0.0)
    except Exception:
        pass

    return df

# ---------------- UI: date range ----------------
st.markdown("### 1) Choose date range to aggregate per-game stats")
c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start date", value=pd.to_datetime("2025-09-23"))
with c2:
    end_date = st.date_input("End date", value=pd.to_datetime("2025-10-07"))
if start_date > end_date:
    st.stop()

# small tool to preview events/day
with st.expander("ESPN day-by-day event counts", expanded=False):
    dates = pd.date_range(start_date, end_date, freq="D")
    counts = []
    for d in dates:
        evs = list_mlb_events_by_date(d.strftime("%Y%m%d"))
        counts.append({"date": d.strftime("%Y%m%d"), "events": len(evs)})
    st.dataframe(pd.DataFrame(counts), use_container_width=True)

# ---------------- Build projections from ESPN ----------------
st.markdown("### 2) Build per-player projections from ESPN 🔗")
if st.button("📥 Build MLB projections"):
    dates = pd.date_range(start_date, end_date, freq="D")
    events = []
    for d in dates:
        events.extend(list_mlb_events_by_date(d.strftime("%Y%m%d")))

    if not events:
        st.error("No data returned from ESPN for this date range.")
        st.stop()

    prog = st.progress(0.0, text=f"Fetching {len(events)} games from ESPN ...")
    all_games = []
    for j, ev in enumerate(events, 1):
        js = fetch_boxscore_event(ev["event_id"], ev["competition_id"])
        if js:
            df_g = parse_mlb_boxscore_players(js)
            if not df_g.empty:
                all_games.append(df_g)
        prog.progress(j/len(events))

    if not all_games:
        st.error("No data returned from ESPN for this date range.")
        st.stop()

    gdf = pd.concat(all_games, ignore_index=True)

    # Split batters & pitchers
    bat = gdf[gdf["is_pitcher"]==0].copy()
    pit = gdf[gdf["is_pitcher"]==1].copy()

    # Per-player game counts
    bat_g = bat.groupby("Player")["team"].size().rename("g").to_frame()
    pit_g = pit.groupby("Player")["team"].size().rename("g").to_frame()

    # Sum + sample variance helpers
    def agg_and_sd(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sums = df.groupby("Player", dropna=False)[cols].sum(min_count=1)
        sqs  = df.groupby("Player", dropna=False)[cols].apply(lambda x: (x**2).sum(min_count=1))
        return sums, sqs

    bat_cols = ["hits","tb","hr","rbi","runs","hrr"]
    pit_cols = ["p_k","p_hits_allowed","p_bb","p_er","p_outs","p_win"]

    bat_sum, bat_sq = agg_and_sd(bat, bat_cols)
    pit_sum, pit_sq = agg_and_sd(pit, pit_cols)

    # Per-game means
    def per_game(mu_sum: pd.Series, g: pd.Series) -> pd.Series:
        return mu_sum.div(g.reindex(mu_sum.index).clip(lower=1).astype(float), fill_value=np.nan)

    # Sample SD (Bessel)
    def sample_sd(sum_x, sum_x2, g):
        g = g.astype(float)
        mean = sum_x.div(g.clip(lower=1))
        var = sum_x2.div(g.clip(lower=1)) - (mean**2)
        var = var * g.div(g.clip(lower=2)-1)  # Bessel; if g<2 -> inf; guard below
        out = (var.clip(lower=0).fillna(0.0))**0.5
        out[g < 2] = np.nan
        return out

    bat_mu = pd.DataFrame({
        "mu_hits": per_game(bat_sum["hits"], bat_g["g"]),
        "mu_tb":   per_game(bat_sum["tb"],   bat_g["g"]),
        "mu_hr":   per_game(bat_sum["hr"],   bat_g["g"]),
        "mu_rbi":  per_game(bat_sum["rbi"],  bat_g["g"]),
        "mu_runs": per_game(bat_sum["runs"], bat_g["g"]),
        "mu_hrr":  per_game(bat_sum["hrr"],  bat_g["g"]),
    })
    bat_sd = pd.DataFrame({
        "sd_hits": sample_sd(bat_sum["hits"], bat_sq["hits"], bat_g["g"]),
        "sd_tb":   sample_sd(bat_sum["tb"],   bat_sq["tb"],   bat_g["g"]),
        "sd_hr":   sample_sd(bat_sum["hr"],   bat_sq["hr"],   bat_g["g"]),
        "sd_rbi":  sample_sd(bat_sum["rbi"],  bat_sq["rbi"],  bat_g["g"]),
        "sd_runs": sample_sd(bat_sum["runs"], bat_sq["runs"], bat_g["g"]),
        "sd_hrr":  sample_sd(bat_sum["hrr"],  bat_sq["hrr"],  bat_g["g"]),
    })

    pit_mu = pd.DataFrame({
        "mu_p_k":   per_game(pit_sum["p_k"],   pit_g["g"]),
        "mu_p_h":   per_game(pit_sum["p_hits_allowed"], pit_g["g"]),
        "mu_p_bb":  per_game(pit_sum["p_bb"],  pit_g["g"]),
        "mu_p_er":  per_game(pit_sum["p_er"],  pit_g["g"]),
        "mu_p_out": per_game(pit_sum["p_outs"], pit_g["g"]),
        "mu_p_win": per_game(pit_sum["p_win"], pit_g["g"]),  # win probability per appearance
    })
    pit_sd = pd.DataFrame({
        "sd_p_k":   sample_sd(pit_sum["p_k"],   pit_sq["p_k"],   pit_g["g"]),
        "sd_p_h":   sample_sd(pit_sum["p_hits_allowed"], pit_sq["p_hits_allowed"], pit_g["g"]),
        "sd_p_bb":  sample_sd(pit_sum["p_bb"],  pit_sq["p_bb"],  pit_g["g"]),
        "sd_p_er":  sample_sd(pit_sum["p_er"],  pit_sq["p_er"],  pit_g["g"]),
        "sd_p_out": sample_sd(pit_sum["p_outs"], pit_sq["p_outs"], pit_g["g"]),
        # no SD for win (Bernoulli handled later)
    })

    bat_proj = (bat_mu.join(bat_sd, how="outer").join(bat_g, how="left")).reset_index().rename(columns={"index":"Player"})
    pit_proj = (pit_mu.join(pit_sd, how="outer").join(pit_g, how="left")).reset_index().rename(columns={"index":"Player"})

    # save to session
    st.session_state["bat_proj"] = bat_proj
    st.session_state["pit_proj"] = pit_proj

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Batter per-game averages (from ESPN)")
        st.dataframe(bat_proj.head(20), use_container_width=True)
    with c2:
        st.subheader("Pitcher per-game averages (from ESPN)")
        st.dataframe(pit_proj.head(20), use_container_width=True)

# ---------------- Odds API ----------------
st.markdown("### 3) Pick game & markets (Odds API)")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets to fetch (MLB)", VALID_MARKETS, default=VALID_MARKETS)

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
    st.info("Enter your Odds API key and pick a lookahead to list MLB games.")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ---------------- Simulate ----------------
st.markdown("### 4) Fetch lines & simulate")
go = st.button("Fetch lines & simulate (MLB)")

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
                if mkey not in VALID_MARKETS or not name or not side:
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

    props_df = (pd.DataFrame(rows)
                .groupby(["market","player","side"], as_index=False)
                .agg(line=("point","median"), n_books=("point","size")))

    # Quick name sets
    bat_names = set(bat_proj["Player"]) if not bat_proj.empty else set()
    pit_names = set(pit_proj["Player"]) if not pit_proj.empty else set()

    out_rows = []
    for _, r in props_df.iterrows():
        mkt, pl, side, line = r["market"], r["player"], r["side"], r["line"]

        # -------- batters --------
        if pl in bat_names and mkt in {
            "batter_hits","batter_total_bases","batter_home_runs",
            "batter_rbis","batter_runs_scored","batter_hits_runs_rbis"
        }:
            row = bat_proj.loc[bat_proj["Player"] == pl].iloc[0]
            if mkt == "batter_hits":
                mu, sd = float(row["mu_hits"]), float(row["sd_hits"])
                p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
            elif mkt == "batter_total_bases":
                mu, sd = float(row["mu_tb"]), float(row["sd_tb"])
                p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
            elif mkt == "batter_home_runs":
                lam = max(1e-6, float(row["mu_hr"]))
                mu, sd = lam, float("nan")
                p_over = poisson_over_prob(lam, line, SIM_TRIALS)
            elif mkt == "batter_rbis":
                lam = max(1e-6, float(row["mu_rbi"]))
                mu, sd = lam, float("nan")
                p_over = poisson_over_prob(lam, line, SIM_TRIALS)
            elif mkt == "batter_runs_scored":
                lam = max(1e-6, float(row["mu_runs"]))
                mu, sd = lam, float("nan")
                p_over = poisson_over_prob(lam, line, SIM_TRIALS)
            elif mkt == "batter_hits_runs_rbis":
                mu, sd = float(row["mu_hrr"]), float(row["sd_hrr"])
                p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
            else:
                continue
            p = p_over if side in ("Over","Yes") else (1.0 - p_over)
            out_rows.append({
                "market": mkt, "player": pl, "side": side, "line": round(line,2),
                "μ (per-game)": None if np.isnan(mu) else round(mu, 3),
                "σ (per-game)": None if (isinstance(sd, float) and np.isnan(sd)) else (None if pd.isna(sd) else round(float(sd), 3)),
                "Win Prob %": round(100*p, 2), "books": int(r["n_books"])
            })
            continue

        # -------- pitchers --------
        if pl in pit_names and mkt in {
            "pitcher_strikeouts","pitcher_hits_allowed","pitcher_walks",
            "pitcher_earned_runs","pitcher_outs","pitcher_record_a_win"
        }:
            row = pit_proj.loc[pit_proj["Player"] == pl].iloc[0]
            if mkt == "pitcher_strikeouts":
                mu, sd = float(row["mu_p_k"]), float(row["sd_p_k"])
                p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
            elif mkt == "pitcher_hits_allowed":
                mu, sd = float(row["mu_p_h"]), float(row["sd_p_h"])
                p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
            elif mkt == "pitcher_walks":
                mu, sd = float(row["mu_p_bb"]), float(row["sd_p_bb"])
                p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
            elif mkt == "pitcher_earned_runs":
                mu, sd = float(row["mu_p_er"]), float(row["sd_p_er"])
                p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
            elif mkt == "pitcher_outs":
                mu, sd = float(row["mu_p_out"]), float(row["sd_p_out"])
                p_over = t_over_prob(mu, sd, line, SIM_TRIALS)
            elif mkt == "pitcher_record_a_win":
                p_yes = bernoulli_prob(float(row["mu_p_win"]))
                mu, sd = float(row["mu_p_win"]), float("nan")
                p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
                out_rows.append({
                    "market": mkt, "player": pl, "side": side, "line": None,
                    "μ (per-game)": round(mu, 3), "σ (per-game)": None,
                    "Win Prob %": round(100*p, 2), "books": int(r["n_books"])
                })
                continue
            else:
                continue
            p = p_over if side in ("Over","Yes") else (1.0 - p_over)
            out_rows.append({
                "market": mkt, "player": pl, "side": side, "line": round(line,2),
                "μ (per-game)": None if np.isnan(mu) else round(mu, 3),
                "σ (per-game)": None if (isinstance(sd, float) and np.isnan(sd)) else (None if pd.isna(sd) else round(float(sd), 3)),
                "Win Prob %": round(100*p, 2), "books": int(r["n_books"])
            })
            continue

    if not out_rows:
        st.warning("Could not match any props to ESPN-built projections.")
        st.stop()

    results = (pd.DataFrame(out_rows)
                 .drop_duplicates(subset=["market","player","side"])
                 .sort_values(["market","Win Prob %"], ascending=[True, False])
                 .reset_index(drop=True))

    st.subheader("Sim results (per-game averages; ESPN boxscores)")
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

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="mlb_props_sim_results.csv",
        mime="text/csv",
    )
