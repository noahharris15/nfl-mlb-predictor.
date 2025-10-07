# pages/2_MLB.py
# MLB Player Props — Odds API + ESPN (batters & pitchers)
# Run: streamlit run App.py

import re
import math
import datetime as dt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- UI / constants ----------------
st.set_page_config(page_title="MLB Player Props — Odds API + ESPN", layout="wide")
st.title("⚾ MLB Player Props — Odds API + ESPN (batters & pitchers)")

SIM_TRIALS = 10_000

VALID_MARKETS = [
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

# ---------------- Odds API helpers ----------------
def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_mlb_events(api_key: str, lookahead_days: int, region: str):
    return odds_get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/events",
        {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region},
    )

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds",
        {
            "apiKey": api_key,
            "regions": region,
            "markets": ",".join(markets),
            "oddsFormat": "american",
        },
    )

# ---------------- ESPN HTTP helpers ----------------
def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

SCOREBOARD_SITE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
EVENTS_CORE     = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events"
SUMMARY_SITE    = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"

# Robust “date → event-ids” (tries site → site(groups=9) → core)
@st.cache_data(show_spinner=False)
def list_event_ids_by_date(day: dt.date) -> List[str]:
    ymd = day.strftime("%Y%m%d")
    ids: List[str] = []

    # 1) Site API (plain)
    js = http_get(SCOREBOARD_SITE, params={"dates": ymd})
    if js and js.get("events"):
        try:
            ids = [str(e["id"]) for e in js["events"] if e.get("id")]
        except Exception:
            ids = []
    if ids:
        st.write(f"ESPN site scoreboard OK for {ymd} → {len(ids)} events")
        return sorted(set(ids))

    # 2) Site API with groups/limit
    js = http_get(SCOREBOARD_SITE, params={"dates": ymd, "groups": 9, "limit": 500})
    if js and js.get("events"):
        try:
            ids = [str(e["id"]) for e in js["events"] if e.get("id")]
        except Exception:
            ids = []
    if ids:
        st.write(f"ESPN site scoreboard (groups=9) OK for {ymd} → {len(ids)} events")
        return sorted(set(ids))

    # 3) Core API fallback
    core = http_get(EVENTS_CORE, params={"dates": ymd, "limit": 500})
    ids_core = []
    if core:
        items = core.get("items")
        if isinstance(items, list):
            for it in items:
                ref = it.get("$ref") or it.get("href") or ""
                m = re.search(r"/events/(\d+)", ref)
                if m:
                    ids_core.append(m.group(1))
                elif it.get("id"):
                    ids_core.append(str(it["id"]))
    if ids_core:
        st.write(f"ESPN core events OK for {ymd} → {len(ids_core)} events")
        return sorted(set(ids_core))

    st.write(f"ESPN returned 0 events for {ymd} (all strategies).")
    return []

@st.cache_data(show_spinner=False)
def fetch_boxscore(event_id: str) -> Optional[dict]:
    return http_get(SUMMARY_SITE, params={"event": event_id})

# ------------- Parsing (reads labels so order doesn’t matter) -------------
def _valmap_from_labels(stat_block: dict) -> Dict[str, float]:
    labels = (stat_block or {}).get("labels") or []
    athletes = (stat_block or {}).get("athletes") or []
    out: Dict[str, float] = {}
    for a in athletes:
        vals = a.get("stats") or []
        amap = {lbl: vals[i] if i < len(vals) else None for i, lbl in enumerate(labels)}
        # coerce to float where possible
        for k, v in amap.items():
            try:
                out[k] = float(str(v).replace("+","").replace(",",""))
            except Exception:
                out[k] = np.nan
    return out

def _ip_to_outs(ip_val: str) -> float:
    # ESPN IP like "5.1" == 5 and 1/3; "5.2" == 5 and 2/3
    try:
        s = str(ip_val)
        if "." in s:
            whole, frac = s.split(".")
            whole = int(whole or 0)
            frac = int(frac or 0)
            return whole * 3 + frac  # 0,1,2 represent outs in the partial inning
        return float(ip_val) * 3.0
    except Exception:
        return np.nan

def parse_game_players(box: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bat_rows, pit_rows = [], []
    try:
        teams = (box or {}).get("boxscore", {}).get("players", []) or []
        for team in teams:
            tname = team.get("team", {}).get("shortDisplayName")
            for stat_block in team.get("statistics", []):
                name = (stat_block.get("name") or "").lower()
                labels = stat_block.get("labels") or []
                for a in stat_block.get("athletes", []) or []:
                    player = a.get("athlete", {}).get("displayName") or ""
                    vals = a.get("stats") or []
                    amap = {lbl: (vals[i] if i < len(vals) else None) for i, lbl in enumerate(labels)}

                    def f(key):
                        try:
                            return float(str(amap.get(key)).replace("+","").replace(",",""))
                        except Exception:
                            return np.nan

                    if "batting" in name:
                        bat_rows.append({
                            "Player": player, "Team": tname,
                            "H": f("H") if "H" in labels else f("Hits"),
                            "TB": f("TB") if "TB" in labels else np.nan,
                            "HR": f("HR") if "HR" in labels else np.nan,
                            "RBI": f("RBI") if "RBI" in labels else np.nan,
                            "R": f("R") if "R" in labels else np.nan,
                            "1B": f("1B") if "1B" in labels else np.nan,
                            "2B": f("2B") if "2B" in labels else np.nan,
                            "3B": f("3B") if "3B" in labels else np.nan,
                            "BB": f("BB") if "BB" in labels else np.nan,
                            "SO": f("K") if "K" in labels else (f("SO") if "SO" in labels else np.nan),
                            "SB": f("SB") if "SB" in labels else np.nan,
                        })

                    if "pitching" in name:
                        ip_raw = amap.get("IP")
                        outs = _ip_to_outs(ip_raw) if ip_raw is not None else np.nan
                        pit_rows.append({
                            "Player": player, "Team": tname,
                            "SO": f("SO") if "SO" in labels else (f("K") if "K" in labels else np.nan),
                            "H": f("H") if "H" in labels else np.nan,
                            "BB": f("BB") if "BB" in labels else np.nan,
                            "ER": f("ER") if "ER" in labels else np.nan,
                            "OUTS": outs,
                            "W": 1.0 if str(amap.get("W","0")).strip() in ("1","True","true","Yes") else 0.0,
                        })
    except Exception:
        pass

    bat_df = pd.DataFrame(bat_rows) if bat_rows else pd.DataFrame(
        columns=["Player","Team","H","TB","HR","RBI","R","1B","2B","3B","BB","SO","SB"]
    )
    pit_df = pd.DataFrame(pit_rows) if pit_rows else pd.DataFrame(
        columns=["Player","Team","SO","H","BB","ER","OUTS","W"]
    )
    # Sum in case the same player appears multiple lines
    bat_df = bat_df.groupby(["Player","Team"], as_index=False).sum(numeric_only=True)
    pit_df = pit_df.groupby(["Player","Team"], as_index=False).sum(numeric_only=True)
    return bat_df, pit_df

# ------------- Aggregate to per-game means + sample SD -------------
@st.cache_data(show_spinner=True)
def build_mlb_agg(start_date: dt.date, end_date: dt.date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # walk each day; gather all event ids
    days = [start_date + dt.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    all_ids: List[str] = []
    for d in days:
        all_ids.extend(list_event_ids_by_date(d))
    all_ids = sorted(set(all_ids))
    if not all_ids:
        return pd.DataFrame(), pd.DataFrame()

    # accumulators
    sums_b = defaultdict(lambda: defaultdict(float))
    sums2_b = defaultdict(lambda: defaultdict(float))
    games_b = defaultdict(int)

    sums_p = defaultdict(lambda: defaultdict(float))
    sums2_p = defaultdict(lambda: defaultdict(float))
    games_p = defaultdict(int)

    prog = st.progress(0.0, text=f"Crawling {len(all_ids)} games...")
    for j, ev in enumerate(all_ids, 1):
        bx = fetch_boxscore(ev)
        if bx:
            bat_df, pit_df = parse_game_players(bx)

            for _, r in bat_df.iterrows():
                key = (r["Player"], r.get("Team",""))
                # metrics we support for batters:
                for k in ["H","TB","HR","RBI","R","1B","2B","3B","BB","SO","SB"]:
                    v = float(r.get(k, 0.0) or 0.0)
                    sums_b[key][k] += v
                    sums2_b[key][k] += v*v
                games_b[key] += 1

            for _, r in pit_df.iterrows():
                key = (r["Player"], r.get("Team",""))
                for k in ["SO","H","BB","ER","OUTS","W"]:
                    v = float(r.get(k, 0.0) or 0.0)
                    sums_p[key][k] += v
                    sums2_p[key][k] += v*v
                games_p[key] += 1

        prog.progress(j / len(all_ids))

    # to dataframe (batters)
    brow = []
    for (player, team), g in games_b.items():
        row = {"Player": player, "Team": team, "g": g}
        for k in ["H","TB","HR","RBI","R","1B","2B","3B","BB","SO","SB"]:
            sx, sx2 = sums_b[(player,team)][k], sums2_b[(player,team)][k]
            mu = sx / g
            var = max(0.0, (sx2 / g) - mu*mu)
            var = var * (g / (g - 1)) if g > 1 else var
            row[f"mu_{k}"] = mu
            row[f"sd_{k}"] = math.sqrt(var) if g > 1 else np.nan
        brow.append(row)
    bat_agg = pd.DataFrame(brow)

    # pitchers
    prow = []
    for (player, team), g in games_p.items():
        row = {"Player": player, "Team": team, "g": g}
        for k in ["SO","H","BB","ER","OUTS","W"]:
            sx, sx2 = sums_p[(player,team)][k], sums2_p[(player,team)][k]
            mu = sx / g
            var = max(0.0, (sx2 / g) - mu*mu)
            var = var * (g / (g - 1)) if g > 1 else var
            row[f"mu_{k}"] = mu
            row[f"sd_{k}"] = math.sqrt(var) if g > 1 else np.nan
        prow.append(row)
    pit_agg = pd.DataFrame(prow)

    return bat_agg, pit_agg

# t-draw helper for O/U
def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

# ---------------- UI 1: Date range ----------------
st.header("1) Date range")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date (YYYY/MM/DD)", value=dt.date(2025, 9, 1))
with col2:
    end_date = st.date_input("End date (YYYY/MM/DD)", value=dt.date(2025, 9, 7))

# ---------------- Build projections ----------------
st.header("2) Build per-player projections from ESPN 🔗")
if st.button("📥 Build MLB projections"):
    bat_agg, pit_agg = build_mlb_agg(start_date, end_date)
    if bat_agg.empty and pit_agg.empty:
        st.error("No data returned from ESPN for this date range.")
    else:
        st.success(f"Built projections. Batters: {len(bat_agg)} | Pitchers: {len(pit_agg)}")
        st.session_state["bat_proj"] = bat_agg
        st.session_state["pit_proj"] = pit_agg

    # Small diagnostic table: how many events per day ESPN returned
    with st.expander("ESPN day-by-day event counts", expanded=False):
        rows = []
        d = start_date
        while d <= end_date:
            cnt = len(list_event_ids_by_date(d))
            rows.append({"date": d.strftime("%Y%m%d"), "events": cnt})
            d += dt.timedelta(days=1)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------------- Odds API selection ----------------
st.header("3) Pick a game & markets from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets to fetch", VALID_MARKETS, default=VALID_MARKETS)

events = []
if api_key:
    try:
        events = list_mlb_events(api_key, lookahead, region)
    except Exception as e:
        st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list upcoming games.")
    st.stop()

labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", labels)
event = events[labels.index(pick)]
event_id = event["id"]

# ---------------- Simulation ----------------
st.header("4) Fetch lines & simulate")
if st.button("🎲 Fetch lines & simulate (MLB)"):
    bat_proj = st.session_state.get("bat_proj", pd.DataFrame())
    pit_proj = st.session_state.get("pit_proj", pd.DataFrame())
    if bat_proj.empty and pit_proj.empty:
        st.warning("Build MLB projections first (Step 2).")
        st.stop()

    # quick lookup frames
    bat_idx = bat_proj.set_index("Player") if not bat_proj.empty else pd.DataFrame()
    pit_idx = pit_proj.set_index("Player") if not pit_proj.empty else pd.DataFrame()

    try:
        odds = fetch_event_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    # flatten odds → one row per (market, player, side) with median 'point' across books
    rows = []
    for bk in odds.get("bookmakers", []):
        for m in bk.get("markets", []):
            key = m.get("key")
            for o in m.get("outcomes", []):
                name = (o.get("description") or "").strip()
                side = o.get("name")
                point = o.get("point")
                if key not in VALID_MARKETS or not name or not side:
                    continue
                rows.append({
                    "market": key, "player": name, "side": side,
                    "point": None if point is None else float(point),
                })
    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    props = (pd.DataFrame(rows)
             .groupby(["market","player","side"], as_index=False)
             .agg(line=("point","median"), n_books=("point","size")))

    # map lines to player mean/sd
    out = []
    for _, r in props.iterrows():
        mkt, player, side, line = r["market"], r["player"], r["side"], r["line"]

        def sim_cont(mu, sd, line):
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            return p_over if side == "Over" else 1.0 - p_over

        if mkt.startswith("batter_") and not bat_idx.empty and player in bat_idx.index:
            b = bat_idx.loc[player]
            mu = sd = mu_raw = None

            if mkt == "batter_hits":
                mu, sd = b["mu_H"], b["sd_H"]
            elif mkt == "batter_total_bases":
                mu, sd = b["mu_TB"], b["sd_TB"]
            elif mkt == "batter_home_runs":
                mu, sd = b["mu_HR"], b["sd_HR"]
            elif mkt == "batter_rbis":
                mu, sd = b["mu_RBI"], b["sd_RBI"]
            elif mkt == "batter_runs_scored":
                mu, sd = b["mu_R"], b["sd_R"]
            elif mkt == "batter_hits_runs_rbis":
                mu  = float(b["mu_H"]  + b["mu_R"] + b["mu_RBI"])
                sd2 = float((b["sd_H"] or 0)**2 + (b["sd_R"] or 0)**2 + (b["sd_RBI"] or 0)**2)
                sd  = math.sqrt(sd2)
            elif mkt == "batter_singles":
                mu, sd = b["mu_1B"], b["sd_1B"]
            elif mkt == "batter_doubles":
                mu, sd = b["mu_2B"], b["sd_2B"]
            elif mkt == "batter_triples":
                mu, sd = b["mu_3B"], b["sd_3B"]
            elif mkt == "batter_walks":
                mu, sd = b["mu_BB"], b["sd_BB"]
            elif mkt == "batter_strikeouts":
                mu, sd = b["mu_SO"], b["sd_SO"]
            elif mkt == "batter_stolen_bases":
                mu, sd = b["mu_SB"], b["sd_SB"]
            else:
                continue

            if pd.isna(line) or pd.isna(mu) or pd.isna(sd):
                continue
            p = sim_cont(float(mu), float(sd), float(line))
            out.append({
                "market": mkt, "player": player, "side": side,
                "line": round(float(line),2),
                "μ (per-game)": round(float(mu),2),
                "σ (per-game)": None if pd.isna(sd) else round(float(sd),2),
                "Win Prob %": round(100*p,2),
                "books": int(r["n_books"]),
            })

        elif mkt.startswith("pitcher_") and not pit_idx.empty and player in pit_idx.index:
            prow = pit_idx.loc[player]
            mu = sd = None
            if mkt == "pitcher_strikeouts":
                mu, sd = prow["mu_SO"], prow["sd_SO"]
            elif mkt == "pitcher_hits_allowed":
                mu, sd = prow["mu_H"], prow["sd_H"]
            elif mkt == "pitcher_walks":
                mu, sd = prow["mu_BB"], prow["sd_BB"]
            elif mkt == "pitcher_earned_runs":
                mu, sd = prow["mu_ER"], prow["sd_ER"]
            elif mkt == "pitcher_outs":
                mu, sd = prow["mu_OUTS"], prow["sd_OUTS"]
            elif mkt == "pitcher_record_a_win":
                # treat as Yes/No market: use empirical win rate
                win_rate = float(prow["mu_W"]) if not pd.isna(prow["mu_W"]) else 0.0
                if side in ("Yes","Over"):
                    p = np.clip(win_rate, 0.0, 1.0)
                else:
                    p = 1.0 - np.clip(win_rate, 0.0, 1.0)
                out.append({
                    "market": mkt, "player": player, "side": side,
                    "line": None, "μ (per-game)": round(win_rate,3),
                    "σ (per-game)": None, "Win Prob %": round(100*p,2),
                    "books": int(r["n_books"]),
                })
                continue
            else:
                continue

            if pd.isna(line) or pd.isna(mu) or pd.isna(sd):
                continue
            p = t_over_prob(float(mu), float(sd), float(line), SIM_TRIALS)
            p = p if side == "Over" else 1.0 - p
            out.append({
                "market": mkt, "player": player, "side": side,
                "line": round(float(line),2),
                "μ (per-game)": round(float(mu),2),
                "σ (per-game)": None if pd.isna(sd) else round(float(sd),2),
                "Win Prob %": round(100*p,2),
                "books": int(r["n_books"]),
            })

    if not out:
        st.warning("No props matched the players we built projections for.")
        st.stop()

    results = (pd.DataFrame(out)
               .drop_duplicates(subset=["market","player","side"])
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    st.subheader("Results")
    cfg = {
        "player": st.column_config.TextColumn("Player", width="large"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.2f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "books": st.column_config.NumberColumn("#Books", width="small"),
    }
    st.dataframe(results, use_container_width=True, hide_index=True, column_config=cfg)

    st.download_button(
        "⬇️ Download MLB results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="mlb_props_results.csv",
        mime="text/csv",
    )
