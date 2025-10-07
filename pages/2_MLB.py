# pages/2_MLB.py
# MLB Player Props — Odds API + MLB StatsAPI (batters & pitchers)
# Run: streamlit run 2_MLB.py   (or as a Streamlit multipage in pages/)

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
st.set_page_config(page_title="MLB Player Props — Odds API + MLB StatsAPI", layout="wide")
st.title("⚾ MLB Player Props — Odds API + MLB StatsAPI")

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

# ---------------- StatsAPI (official MLB) helpers ----------------
def _statsapi_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def _dates_ymd_list(a_slash: str, b_slash: str) -> List[str]:
    start = datetime.strptime(a_slash, "%Y/%m/%d").date()
    end   = datetime.strptime(b_slash, "%Y/%m/%d").date()
    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))  # StatsAPI expects dashed Y-M-D
        cur += timedelta(days=1)
    return out

def _list_game_ids(ymd: str) -> List[int]:
    js = _statsapi_get("https://statsapi.mlb.com/api/v1/schedule", params={"sportId": 1, "date": ymd})
    if not js:
        return []
    ids = []
    for d in js.get("dates", []):
        for g in d.get("games", []):
            if g.get("gamePk"):
                ids.append(int(g["gamePk"]))
    return ids

def _fetch_live(game_pk: int) -> Optional[dict]:
    return _statsapi_get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")

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

def _sd_from_sums(sum_x, sum_x2, n, floor=0.1) -> float:
    if n <= 1: return float("nan")
    mean = sum_x / n
    var = max((sum_x2 / n) - mean**2, 0.0)
    var *= n / (n - 1)
    return float(max(var**0.5, floor))

def _parse_one_game(js: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse one StatsAPI live feed into (batters_df, pitchers_df), one row per player for that game.
    Batters: H, TB, HR, R, RBI, BB, SO, SB, 1B, 2B, 3B
    Pitchers: SO_pitch, H_allowed, BB_allowed, ER, Outs, Win(0/1)
    """
    if not js:
        return pd.DataFrame(), pd.DataFrame()
    live = js.get("liveData", {}) or {}
    box  = live.get("boxscore", {}) or {}
    teams = [box.get("teams", {}).get("away", {}), box.get("teams", {}).get("home", {})]

    # decisions (winner pitcher id) — useful to tag wins robustly
    winner_id = None
    decisions = live.get("decisions") or js.get("gameData", {}).get("decisions") or {}
    if decisions and isinstance(decisions, dict) and decisions.get("winner", {}):
        try:
            winner_id = int(decisions.get("winner", {}).get("id"))
        except Exception:
            winner_id = None

    bat_rows, pit_rows = [], []
    for t in teams:
        team_abbr = (t.get("team", {}) or {}).get("abbreviation") or (t.get("team", {}) or {}).get("name")
        players = t.get("players", {}) or {}
        for pkey, pdata in players.items():
            person = pdata.get("person", {})
            pid = person.get("id")
            name = _norm_name(person.get("fullName"))
            stats = pdata.get("stats", {}) or {}

            # Batting totals
            b = stats.get("batting", {}) or {}
            if b:
                H   = _f(b.get("hits"))
                HR  = _f(b.get("homeRuns"))
                R   = _f(b.get("runs"))
                RBI = _f(b.get("rbi"))
                BB  = _f(b.get("baseOnBalls"))
                SO  = _f(b.get("strikeOuts"))
                SB  = _f(b.get("stolenBases"))
                _2B = _f(b.get("doubles"))
                _3B = _f(b.get("triples"))
                _1B = (H - _2B - _3B - HR) if all(not math.isnan(x) for x in [H,_2B,_3B,HR]) else float("nan")
                TB  = _f(b.get("totalBases")) if b.get("totalBases") is not None else (
                      1*_1B + 2*_2B + 3*_3B + 4*HR if all(not math.isnan(x) for x in [_1B,_2B,_3B,HR]) else float("nan"))
                bat_rows.append({
                    "Player": name, "Team": team_abbr,
                    "H": H, "TB": TB, "HR": HR, "R": R, "RBI": RBI, "BB": BB, "SO": SO, "SB": SB,
                    "1B": _1B, "2B": _2B, "3B": _3B
                })

            # Pitching totals (allowed)
            p = stats.get("pitching", {}) or {}
            if p:
                SOp   = _f(p.get("strikeOuts"))
                H_all = _f(p.get("hits"))
                BB_a  = _f(p.get("baseOnBalls"))
                ER    = _f(p.get("earnedRuns"))
                outs  = _f(p.get("outs"))
                win   = 1.0 if (winner_id is not None and pid == winner_id) else 0.0
                pit_rows.append({
                    "Player": name, "Team": team_abbr,
                    "SO_pitch": SOp, "H_allowed": H_all, "BB_allowed": BB_a, "ER": ER, "Outs": outs, "Win": win
                })

    bat_df = pd.DataFrame(bat_rows)
    pit_df = pd.DataFrame(pit_rows)
    # Sum per player (in case of duplicate keys)
    if not bat_df.empty:
        bat_df = bat_df.groupby(["Player","Team"], as_index=False).sum(numeric_only=True)
    if not pit_df.empty:
        pit_df = pit_df.groupby(["Player","Team"], as_index=False).sum(numeric_only=True)
    return bat_df, pit_df

# ------------- Aggregate to per-game means + sample SD -------------
@st.cache_data(show_spinner=True)
def build_mlb_averages_statsapi(start_slash: str, end_slash: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    days = _dates_ymd_list(start_slash, end_slash)
    if not days:
        return pd.DataFrame(), pd.DataFrame()

    # Gather unique game IDs
    prog = st.progress(0.0, text="Listing games…")
    games: List[int] = []
    for i, d in enumerate(days, 1):
        games.extend(_list_game_ids(d))
        prog.progress(i / len(days))
    games = list(dict.fromkeys(games))
    if not games:
        return pd.DataFrame(), pd.DataFrame()

    # Accumulators
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

    prog = st.progress(0.0, text=f"Fetching {len(games)} boxscores…")
    for j, gid in enumerate(games, 1):
        js = _fetch_live(gid)
        bat, pit = _parse_one_game(js)

        # batters
        for _, r in bat.iterrows():
            p = _norm_name(r["Player"]); binit(p)
            # consider played if any batting stat > 0
            if any(_f(r.get(k, 0)) > 0 for k in bat_sum[p]):
                bat_g[p] += 1
            for k in bat_sum[p]:
                v = _f(r.get(k, 0))
                if not np.isnan(v):
                    bat_sum[p][k]  += v
                    bat_sum2[p][k] += v*v

        # pitchers
        for _, r in pit.iterrows():
            p = _norm_name(r["Player"]); pinit(p)
            if any(_f(r.get(k, 0)) > 0 for k in pit_sum[p]):  # pitched appearance
                pit_g[p] += 1
            for k in pit_sum[p]:
                v = _f(r.get(k, 0))
                if not np.isnan(v):
                    pit_sum[p][k]  += v
                    pit_sum2[p][k] += v*v

        prog.progress(j / len(games))

    # Build per-game projections
    bat_rows = []
    for p, sums in bat_sum.items():
        g = max(1, bat_g[p])
        row = {"Player": p, "g": g}
        for k, s in sums.items():
            row[f"mu_{k.lower()}"] = s / g
            row[f"sd_{k.lower()}"] = _sd_from_sums(s, bat_sum2[p][k], g, floor=0.15)
        row["mu_hrr"] = row["mu_h"] + row["mu_r"] + row["mu_rbi"]
        row["sd_hrr"] = math.sqrt(row["sd_h"]**2 + row["sd_r"]**2 + row["sd_rbi"]**2)
        bat_rows.append(row)

    pit_rows = []
    for p, sums in pit_sum.items():
        g = max(1, pit_g[p])
        row = {"Player": p, "g": g}
        for k, s in sums.items():
            row[f"mu_{k.lower()}"] = s / g
            row[f"sd_{k.lower()}"] = _sd_from_sums(s, pit_sum2[p][k], g, floor=0.15)
        # derive win rate mean (already per game); keep as μ for Yes/No
        pit_rows.append(row)

    return pd.DataFrame(bat_rows), pd.DataFrame(pit_rows)

# t-draw helper for O/U
def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

# ---------------- UI 1: Date range ----------------
st.header("1) Date range")
col1, col2 = st.columns(2)
with col1:
    start_date = st.text_input("Start date (YYYY/MM/DD)", value="2025/09/01")
with col2:
    end_date = st.text_input("End date (YYYY/MM/DD)", value="2025/09/07")

# ---------------- Build projections ----------------
st.header("2) Build per-player projections from MLB StatsAPI")
if st.button("📥 Build MLB projections"):
    bat_agg, pit_agg = build_mlb_averages_statsapi(start_date, end_date)
    if bat_agg.empty and pit_agg.empty:
        st.error("No data returned from MLB StatsAPI for this date range.")
    else:
        st.success(f"Built projections. Batters: {len(bat_agg)} | Pitchers: {len(pit_agg)}")
        st.session_state["bat_proj"] = bat_agg
        st.session_state["pit_proj"] = pit_agg

        # Preview tables (raw per-game μ for QC)
        with st.expander("Preview — Batters (per-game μ / σ)", expanded=False):
            cols = ["Player","g","mu_tb","sd_tb","mu_h","sd_h","mu_hr","sd_hr","mu_r","sd_r","mu_rbi","sd_rbi","mu_hrr","sd_hrr"]
            show = bat_agg[cols].sort_values("mu_tb", ascending=False).head(30) if not bat_agg.empty else pd.DataFrame()
            st.dataframe(show, use_container_width=True)
        with st.expander("Preview — Pitchers (per-game μ / σ)", expanded=False):
            cols = ["Player","g","mu_so_pitch","sd_so_pitch","mu_h_allowed","sd_h_allowed","mu_bb_allowed","sd_bb_allowed","mu_er","sd_er","mu_outs","sd_outs","mu_win"]
            dfp = pit_agg.copy()
            if "mu_win" not in dfp.columns and "mu_win" in dfp.columns:
                pass
            if not dfp.empty and "mu_win" not in dfp.columns:
                dfp["mu_win"] = dfp.get("mu_win", pd.Series([0.0]*len(dfp)))
            showp = dfp[cols].sort_values("mu_so_pitch", ascending=False).head(30) if not dfp.empty else pd.DataFrame()
            st.dataframe(showp, use_container_width=True)

# ---------------- Odds API selection ----------------
st.header("3) Pick a game & markets from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets to fetch", VALID_MARKETS, default=["batter_hits","batter_total_bases","pitcher_strikeouts"])

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

    # quick lookup frames (by normalized name)
    if not bat_proj.empty:
        bat_proj = bat_proj.copy()
        bat_proj["PN"] = bat_proj["Player"].apply(_norm_name)
        bat_idx = bat_proj.set_index("PN")
    else:
        bat_idx = pd.DataFrame()

    if not pit_proj.empty:
        pit_proj = pit_proj.copy()
        pit_proj["PN"] = pit_proj["Player"].apply(_norm_name)
        pit_idx = pit_proj.set_index("PN")
    else:
        pit_idx = pd.DataFrame()

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
                name = _norm_name(o.get("description"))
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

    out = []

    for _, r in props.iterrows():
        mkt, player, side, line = r["market"], r["player"], r["side"], r["line"]

        def sim_cont(mu, sd, line):
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            return p_over if side == "Over" else 1.0 - p_over

        # BATTERS
        if not bat_idx.empty and player in bat_idx.index:
            b = bat_idx.loc[player]
            mu = sd = None

            if mkt == "batter_hits":
                mu, sd = float(b["mu_h"]), float(b["sd_h"])
            elif mkt == "batter_total_bases":
                mu, sd = float(b["mu_tb"]), float(b["sd_tb"])
            elif mkt == "batter_home_runs":
                mu, sd = float(b["mu_hr"]), float(b["sd_hr"])
            elif mkt == "batter_rbis":
                mu, sd = float(b["mu_rbi"]), float(b["sd_rbi"])
            elif mkt == "batter_runs_scored":
                mu, sd = float(b["mu_r"]), float(b["sd_r"])
            elif mkt == "batter_hits_runs_rbis":
                mu, sd = float(b["mu_hrr"]), float(b["sd_hrr"])
            elif mkt == "batter_singles":
                mu, sd = float(b["mu_1b"]), float(b["sd_1b"])
            elif mkt == "batter_doubles":
                mu, sd = float(b["mu_2b"]), float(b["sd_2b"])
            elif mkt == "batter_triples":
                mu, sd = float(b["mu_3b"]), float(b["sd_3b"])
            elif mkt == "batter_walks":
                mu, sd = float(b["mu_bb"]), float(b["sd_bb"])
            elif mkt == "batter_strikeouts":
                mu, sd = float(b["mu_so"]), float(b["sd_so"])
            elif mkt == "batter_stolen_bases":
                mu, sd = float(b["mu_sb"]), float(b["sd_sb"])
            else:
                continue

            if pd.isna(line) or pd.isna(mu) or pd.isna(sd):
                continue
            p = sim_cont(mu, sd, line)
            out.append({
                "market": mkt, "player": player, "side": side,
                "line": round(float(line),2),
                "μ (per-game)": round(mu,2),
                "σ (per-game)": round(sd,2),
                "Win Prob %": round(100*p,2),
                "books": int(r["n_books"]),
            })
            continue

        # PITCHERS
        if not pit_idx.empty and player in pit_idx.index:
            p = pit_idx.loc[player]
            mu = sd = None
            if mkt == "pitcher_strikeouts":
                mu, sd = float(p["mu_so_pitch"]), float(p["sd_so_pitch"])
            elif mkt == "pitcher_hits_allowed":
                mu, sd = float(p["mu_h_allowed"]), float(p["sd_h_allowed"])
            elif mkt == "pitcher_walks":
                mu, sd = float(p["mu_bb_allowed"]), float(p["sd_bb_allowed"])
            elif mkt == "pitcher_earned_runs":
                mu, sd = float(p["mu_er"]), float(p["sd_er"])
            elif mkt == "pitcher_outs":
                mu, sd = float(p["mu_outs"]), float(p["sd_outs"])
            elif mkt == "pitcher_record_a_win":
                # Yes/No: empirical win rate per appearance
                mu = float(p.get("mu_win", 0.0))
                p_yes = float(np.clip(mu, 0.0, 1.0))
                pprob = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
                out.append({
                    "market": mkt, "player": player, "side": side,
                    "line": None, "μ (per-game)": round(mu,3), "σ (per-game)": None,
                    "Win Prob %": round(100*pprob,2), "books": int(r["n_books"])
                })
                continue
            else:
                continue

            if pd.isna(line) or pd.isna(mu) or pd.isna(sd):
                continue
            pprob = sim_cont(mu, sd, line)
            out.append({
                "market": mkt, "player": player, "side": side,
                "line": round(float(line),2),
                "μ (per-game)": round(mu,2),
                "σ (per-game)": round(sd,2),
                "Win Prob %": round(100*pprob,2),
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
