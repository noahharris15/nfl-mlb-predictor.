# pages/2_MLB.py
# MLB Props — Odds API + ESPN (batters & pitchers)
# Run app multi-page: streamlit run App.py

import re
import math
import datetime as dt
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------- UI / constants -----------------------------
st.set_page_config(page_title="MLB Player Props — Odds API + ESPN", layout="wide")
st.title("⚾ MLB Player Props — Odds API + ESPN")

SIM_TRIALS = 10_000

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

# ----------------------------- Odds API helpers ---------------------------
def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_mlb_events(api_key: str, lookahead_days: int, region: str):
    base = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
    return odds_get(base, {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region})

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    base = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
    return odds_get(base, {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"})

# ----------------------------- ESPN helpers -------------------------------
SITE_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
CORE_EVENTS     = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events"
SUMMARY         = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}

def http_get_json(url: str, params=None, timeout=14):
    try:
        r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def list_day_event_ids(date_str: str) -> List[str]:
    """Get ESPN event IDs for a single day YYYYMMDD.
    Try site scoreboard (regular), site scoreboard (postseason), then core events.
    """
    ids: List[str] = []

    js = http_get_json(SITE_SCOREBOARD, params={"dates": date_str})
    if js and js.get("events"):
        ids.extend(str(e.get("id")) for e in js["events"] if e.get("id"))

    if not ids:
        js = http_get_json(SITE_SCOREBOARD, params={"dates": date_str, "seasontype": 3})
        if js and js.get("events"):
            ids.extend(str(e.get("id")) for e in js["events"] if e.get("id"))

    if not ids:
        js = http_get_json(CORE_EVENTS, params={"dates": date_str, "limit": 1000})
        if js and js.get("items"):
            for url in js["items"]:
                try:
                    eid = url.rstrip("/").split("/")[-1]
                    if eid.isdigit():
                        ids.append(eid)
                except Exception:
                    continue

    # dedupe preserving order
    seen = set(); out = []
    for i in ids:
        if i not in seen:
            out.append(i); seen.add(i)
    return out

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id: str) -> Optional[dict]:
    return http_get_json(SUMMARY, params={"event": event_id})

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = re.sub(r"\s+", " ", n.replace("-", " ")).strip()
    return n

def _extract_players(box: dict) -> List[dict]:
    players = []
    try:
        sec = box.get("boxscore", {}).get("players", [])
        for team in sec:
            tname = team.get("team", {}).get("shortDisplayName")
            for block in team.get("statistics", []):
                section = (block.get("name") or "").lower()  # "batting", "pitching", "fielding"
                labels = [str(x).lower() for x in (block.get("labels") or [])]
                for a in (block.get("athletes") or []):
                    players.append({
                        "team": tname, "section": section, "labels": labels, "ath": a
                    })
    except Exception:
        pass
    return players

def _parse_comp(value: str):
    """Parse '2-4' style string -> (2,4)."""
    try:
        s = str(value)
        if "-" in s:
            a,b = s.split("-",1)
            return float(a), float(b)
        return float(s), float("nan")
    except Exception:
        return float("nan"), float("nan")

def parse_boxscore_players(box: dict) -> Dict[str, pd.DataFrame]:
    """Return dict with two dataframes: bat (batters), pit (pitchers) for THIS game."""
    rows_bat, rows_pit = [], []
    for r in _extract_players(box):
        section = r["section"]; labels = r["labels"]; a = r["ath"]
        name = normalize_name(a.get("athlete", {}).get("displayName"))
        stats = a.get("stats") or []

        # Map label -> value (strings)
        lbl = {labels[i]: stats[i] for i in range(min(len(labels), len(stats)))}

        try:
            if section == "batting":
                # Common ESPN labels: "ab","r","h","rbi","bb","so","hr","tb","2b","3b","sb"
                H  = float(lbl.get("h")  or (stats[2] if len(stats)>2 else np.nan))
                R  = float(lbl.get("r")  or (stats[1] if len(stats)>1 else np.nan))
                RBI= float(lbl.get("rbi") or (stats[3] if len(stats)>3 else np.nan))
                HR = float(lbl.get("hr") or 0)
                TB = lbl.get("tb")
                if TB is not None:
                    TB = float(TB)
                else:
                    dbl = float(lbl.get("2b") or 0)
                    trp = float(lbl.get("3b") or 0)
                    # TB = H + 2B + 2*3B + 3*HR  (since TB = 1B + 2*2B + 3*3B + 4*HR and 1B = H-2B-3B-HR)
                    TB = float(H + dbl + 2*trp + 3*HR)
                H1 = None
                if "1b" in lbl:
                    try: H1 = float(lbl.get("1b"))
                    except: H1 = None

                rows_bat.append({
                    "Player": name,
                    "hits": H, "tb": TB, "hr": HR, "rbi": RBI, "runs": R,
                    "h1": H1
                })

            elif section == "pitching":
                # Labels vary but usually: "ip","h","r","er","bb","k","hr","pc-st"
                ip = lbl.get("ip") or (stats[0] if len(stats)>0 else None)
                outs = None
                if ip is not None:
                    # ip like "5.2" meaning 5 + 2/3 innings -> outs = innings*3
                    try:
                        s = str(ip)
                        if "." in s:
                            innings, frac = s.split(".",1)
                            outs = int(3*int(innings) + int(frac))
                        else:
                            outs = int(3*int(float(s)))
                    except Exception:
                        outs = None

                K  = float(lbl.get("k")  or (stats[5] if len(stats)>5 else np.nan))
                H  = float(lbl.get("h")  or (stats[1] if len(stats)>1 else np.nan))
                BB = float(lbl.get("bb") or (stats[4] if len(stats)>4 else np.nan))
                ER = float(lbl.get("er") or (stats[3] if len(stats)>3 else np.nan))
                # Decision (win)
                dec = lbl.get("dec")
                win = 1.0 if dec and str(dec).strip().upper() == "W" else 0.0

                rows_pit.append({
                    "Player": name,
                    "so": K, "hits_allowed": H, "walks": BB, "er": ER,
                    "outs": float(outs) if outs is not None else np.nan,
                    "win": win
                })

        except Exception:
            continue

    bat = pd.DataFrame(rows_bat) if rows_bat else pd.DataFrame(columns=["Player","hits","tb","hr","rbi","runs","h1"])
    pit = pd.DataFrame(rows_pit) if rows_pit else pd.DataFrame(columns=["Player","so","hits_allowed","walks","er","outs","win"])

    # Some players appear multiple times (PH + starter) → sum per game
    bat = bat.groupby("Player", as_index=False).sum(numeric_only=True)
    pit = pit.groupby("Player", as_index=False).sum(numeric_only=True)
    return {"bat": bat, "pit": pit}

# ----------------------------- Aggregation --------------------------------
def sample_sd(sum_x, sum_x2, n):
    n = int(n)
    if n <= 1: return np.nan
    m = sum_x / n
    var = (sum_x2 / n) - (m*m)
    var = var * (n / (n - 1))
    return float(np.sqrt(max(var, 1e-8)))

@st.cache_data(show_spinner=True)
def build_mlb_agg(start_date: dt.date, end_date: dt.date) -> Dict[str, pd.DataFrame]:
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    all_events = []
    per_day = []
    for d in days:
        ds = d.strftime("%Y%m%d")
        ids = list_day_event_ids(ds)
        per_day.append((ds, len(ids)))
        all_events.extend(ids)

    with st.expander("ESPN day-by-day event counts", expanded=False):
        st.dataframe(pd.DataFrame(per_day, columns=["date","events"]), use_container_width=True)

    all_events = list(dict.fromkeys(all_events))
    if not all_events:
        return {"bat": pd.DataFrame(), "pit": pd.DataFrame()}

    totals_bat, sumsqs_bat, games_bat = {}, {}, {}
    totals_pit, sumsqs_pit, games_pit = {}, {}, {}

    def init_b(p):
        if p not in totals_bat:
            totals_bat[p] = {"hits":0.0,"tb":0.0,"hr":0.0,"rbi":0.0,"runs":0.0,"h1":0.0}
            sumsqs_bat[p] = {"hits":0.0,"tb":0.0,"hr":0.0,"rbi":0.0,"runs":0.0}
            games_bat[p]  = 0
    def init_p(p):
        if p not in totals_pit:
            totals_pit[p] = {"so":0.0,"hits_allowed":0.0,"walks":0.0,"er":0.0,"outs":0.0,"win":0.0}
            sumsqs_pit[p] = {"so":0.0,"hits_allowed":0.0,"walks":0.0,"er":0.0,"outs":0.0}
            games_pit[p]  = 0

    prog = st.progress(0.0, text=f"Crawling {len(all_events)} games...")
    for j, eid in enumerate(all_events, 1):
        box = fetch_boxscore_event(eid)
        if not box:
            prog.progress(j/len(all_events)); continue
        parsed = parse_boxscore_players(box)

        # batters
        bdf = parsed["bat"]
        for _, r in bdf.iterrows():
            p = r["Player"]; init_b(p)
            played = any(pd.notna(r.get(k)) and float(r.get(k)) > 0 for k in ["hits","tb","hr","rbi","runs"])
            if played: games_bat[p] += 1
            for k in totals_bat[p]:
                v = float(r.get(k, 0) or 0)
                totals_bat[p][k] += v
                if k in sumsqs_bat[p]:
                    sumsqs_bat[p][k] += v*v

        # pitchers
        pdf = parsed["pit"]
        for _, r in pdf.iterrows():
            p = r["Player"]; init_p(p)
            played = any(pd.notna(r.get(k)) and float(r.get(k)) > 0 for k in ["so","hits_allowed","walks","er","outs"])
            if played or pd.notna(r.get("win")): games_pit[p] += 1
            for k in totals_pit[p]:
                v = float(r.get(k, 0) or 0)
                totals_pit[p][k] += v
                if k in sumsqs_pit[p]:
                    sumsqs_pit[p][k] += v*v

        prog.progress(j/len(all_events))

    # Build season tables
    bat_rows = []
    for p, stat in totals_bat.items():
        g = max(1, games_bat.get(p, 0))
        bat_rows.append({
            "Player": p, "g": g, **stat,
            "sq_hits":sumsqs_bat[p]["hits"], "sq_tb":sumsqs_bat[p]["tb"], "sq_hr":sumsqs_bat[p]["hr"],
            "sq_rbi":sumsqs_bat[p]["rbi"], "sq_runs":sumsqs_bat[p]["runs"]
        })
    pit_rows = []
    for p, stat in totals_pit.items():
        g = max(1, games_pit.get(p, 0))
        pit_rows.append({
            "Player": p, "g": g, **stat,
            "sq_so":sumsqs_pit[p]["so"], "sq_hits_allowed":sumsqs_pit[p]["hits_allowed"],
            "sq_walks":sumsqs_pit[p]["walks"], "sq_er":sumsqs_pit[p]["er"], "sq_outs":sumsqs_pit[p]["outs"]
        })

    bat = pd.DataFrame(bat_rows) if bat_rows else pd.DataFrame()
    pit = pd.DataFrame(pit_rows) if pit_rows else pd.DataFrame()

    if bat.empty and pit.empty:
        return {"bat": pd.DataFrame(), "pit": pd.DataFrame()}

    # per-game means
    for df in (bat, pit):
        if df.empty: continue
        df["g"] = df["g"].astype(float).clip(lower=1)

    if not bat.empty:
        g = bat["g"]
        bat["mu_hits"] = bat["hits"]/g
        bat["mu_tb"]   = bat["tb"]/g
        bat["mu_hr"]   = bat["hr"]/g
        bat["mu_rbi"]  = bat["rbi"]/g
        bat["mu_runs"] = bat["runs"]/g
        bat["mu_hrr"]  = (bat["hits"]+bat["runs"]+bat["rbi"])/g  # hits+runs+rbi combo

        bat["sd_hits"] = bat.apply(lambda r: sample_sd(r["hits"], r["sq_hits"], r["g"]), axis=1)*1.10
        bat["sd_tb"]   = bat.apply(lambda r: sample_sd(r["tb"],   r["sq_tb"],   r["g"]), axis=1)*1.10
        # Poisson for HR; still keep SD for display
        bat["sd_hr"]   = np.sqrt(np.maximum(bat["mu_hr"], 1e-6))
        bat["sd_rbi"]  = bat.apply(lambda r: sample_sd(r["rbi"],  r["sq_rbi"],  r["g"]), axis=1)*1.10
        bat["sd_runs"] = bat.apply(lambda r: sample_sd(r["runs"], r["sq_runs"], r["g"]), axis=1)*1.10
        bat["sd_hrr"]  = np.sqrt(bat["sd_hits"]**2 + bat["sd_runs"]**2 + bat["sd_rbi"]**2)

    if not pit.empty:
        g = pit["g"]
        pit["mu_so"]   = pit["so"]/g
        pit["mu_hits_allowed"] = pit["hits_allowed"]/g
        pit["mu_walks"]= pit["walks"]/g
        pit["mu_er"]   = pit["er"]/g
        pit["mu_outs"] = pit["outs"]/g
        pit["p_win"]   = (pit["win"]/g).clip(0.0, 1.0)

        # SDs (Poisson-like for K, BB, ER, H; t-based for outs)
        pit["sd_so"]   = np.sqrt(np.maximum(pit["mu_so"], 1e-6))
        pit["sd_hits_allowed"] = np.sqrt(np.maximum(pit["mu_hits_allowed"], 1e-6))
        pit["sd_walks"]= np.sqrt(np.maximum(pit["mu_walks"], 1e-6))
        pit["sd_er"]   = np.sqrt(np.maximum(pit["mu_er"], 1e-6))
        pit["sd_outs"] = pit.apply(lambda r: sample_sd(r["outs"], r["sq_outs"], r["g"]), axis=1)*1.10

    return {"bat": bat, "pit": pit}

# ----------------------------- Prob models --------------------------------
def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-8, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def poisson_over_prob(lam: float, line: float, trials: int = SIM_TRIALS) -> float:
    lam = max(1e-8, float(lam))
    return float((np.random.poisson(lam=lam, size=trials) > line).mean())

# ----------------------------- UI Step 1 ----------------------------------
st.header("1) Choose date range")
c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=14))
with c2:
    end_date   = st.date_input("End date",   value=dt.date.today())

st.header("2) Build per-player projections from ESPN")
if st.button("📥 Build MLB projections"):
    agg = build_mlb_agg(start_date, end_date)
    bat = agg["bat"]; pit = agg["pit"]

    if bat.empty and pit.empty:
        st.error("No data returned from ESPN for this date range.")
        st.stop()

    st.session_state["mlb_bat"] = bat
    st.session_state["mlb_pit"] = pit

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Batters (per-game averages)")
        if not bat.empty:
            show = bat[["Player","g","mu_hits","mu_tb","mu_hr","mu_rbi","mu_runs","mu_hrr"]].sort_values("mu_tb", ascending=False).head(20)
            st.dataframe(show, use_container_width=True)
        else:
            st.info("No batters in range.")
    with c2:
        st.subheader("Pitchers (per-game averages)")
        if not pit.empty:
            show = pit[["Player","g","mu_so","mu_hits_allowed","mu_walks","mu_er","mu_outs","p_win"]].sort_values("mu_so", ascending=False).head(20)
            st.dataframe(show, use_container_width=True)
        else:
            st.info("No pitchers in range.")

# ----------------------------- Odds API UI --------------------------------
st.header("3) Pick a game & markets (Odds API)")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to session)", value="", type="password"
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

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ----------------------------- Simulate -----------------------------------
st.header("4) Fetch lines & simulate")
go = st.button("🎲 Fetch lines & simulate (MLB)")

if go:
    bat = st.session_state.get("mlb_bat", pd.DataFrame())
    pit = st.session_state.get("mlb_pit", pd.DataFrame())
    if bat.empty and pit.empty:
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
                    "market": mkey, "player": name, "side": side,
                    "point": (None if point is None else float(point))
                })

    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    props_df = (pd.DataFrame(rows)
                .groupby(["market","player","side"], as_index=False)
                .agg(line=("point","median"), n_books=("point","size")))

    bat_index = set(bat["Player"]) if not bat.empty else set()
    pit_index = set(pit["Player"]) if not pit.empty else set()

    out_rows = []
    for _, r in props_df.iterrows():
        market, player, line, side = r["market"], r["player"], r["line"], r["side"]

        # ---- Batters ----
        if market in ("batter_hits","batter_total_bases","batter_rbis","batter_runs_scored","batter_hits_runs_rbis") and player in bat_index:
            row = bat.loc[bat["Player"] == player].iloc[0]
            if market == "batter_hits":
                mu, sd = float(row["mu_hits"]), float(row["sd_hits"])
            elif market == "batter_total_bases":
                mu, sd = float(row["mu_tb"]), float(row["sd_tb"])
            elif market == "batter_rbis":
                mu, sd = float(row["mu_rbi"]), float(row["sd_rbi"])
            elif market == "batter_runs_scored":
                mu, sd = float(row["mu_runs"]), float(row["sd_runs"])
            else:  # hits+runs+rbi
                mu, sd = float(row["mu_hrr"]), float(row["sd_hrr"])
            if pd.isna(line) or np.isnan(mu) or np.isnan(sd): 
                continue
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            out_rows.append({
                "market": market, "player": player, "side": side, "line": round(float(line),2),
                "μ (per-game)": round(mu,2), "σ (per-game)": round(sd,2), "Win Prob %": round(100*p,2),
                "books": int(r["n_books"])
            })
            continue

        if market == "batter_home_runs" and player in bat_index:
            row = bat.loc[bat["Player"] == player].iloc[0]
            lam = float(row["mu_hr"])
            if pd.isna(line) or np.isnan(lam): 
                continue
            p_over = poisson_over_prob(lam, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            out_rows.append({
                "market": market, "player": player, "side": side, "line": round(float(line),2),
                "μ (per-game)": round(lam,3), "σ (per-game)": round(float(np.sqrt(max(lam,1e-6))),3),
                "Win Prob %": round(100*p,2), "books": int(r["n_books"])
            })
            continue

        # ---- Pitchers ----
        if market in ("pitcher_strikeouts","pitcher_walks","pitcher_earned_runs","pitcher_hits_allowed") and player in pit_index:
            row = pit.loc[pit["Player"] == player].iloc[0]
            if market == "pitcher_strikeouts":
                lam = float(row["mu_so"])
            elif market == "pitcher_walks":
                lam = float(row["mu_walks"])
            elif market == "pitcher_earned_runs":
                lam = float(row["mu_er"])
            else:
                lam = float(row["mu_hits_allowed"])
            if pd.isna(line) or np.isnan(lam):
                continue
            p_over = poisson_over_prob(lam, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            out_rows.append({
                "market": market, "player": player, "side": side, "line": round(float(line),2),
                "μ (per-game)": round(lam,3), "σ (per-game)": round(float(np.sqrt(max(lam,1e-6))),3),
                "Win Prob %": round(100*p,2), "books": int(r["n_books"])
            })
            continue

        if market == "pitcher_outs" and player in pit_index:
            row = pit.loc[pit["Player"] == player].iloc[0]
            mu, sd = float(row["mu_outs"]), float(row["sd_outs"])
            if pd.isna(line) or np.isnan(mu) or np.isnan(sd):
                continue
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            out_rows.append({
                "market": market, "player": player, "side": side, "line": round(float(line),2),
                "μ (per-game)": round(mu,2), "σ (per-game)": round(sd,2), "Win Prob %": round(100*p,2),
                "books": int(r["n_books"])
            })
            continue

        if market == "pitcher_record_a_win" and player in pit_index:
            row = pit.loc[pit["Player"] == player].iloc[0]
            p_yes = float(row["p_win"])
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            out_rows.append({
                "market": market, "player": player, "side": side, "line": None,
                "μ (per-game)": round(p_yes,3), "σ (per-game)": None, "Win Prob %": round(100*p,2),
                "books": int(r["n_books"])
            })
            continue

    if not out_rows:
        st.warning("No props matched your ESPN projections.")
        st.stop()

    results = (pd.DataFrame(out_rows)
                 .drop_duplicates(subset=["market","player","side"])
                 .sort_values(["market","Win Prob %"], ascending=[True, False])
                 .reset_index(drop=True))

    st.subheader("Results")
    colcfg = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.2f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
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
