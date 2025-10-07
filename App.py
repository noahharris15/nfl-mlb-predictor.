# NBA / MLB Player Props — Odds API + ESPN (per-game averages)
# This is a Streamlit subpage meant to live in pages/ under your NFL app.
# DO NOT call st.set_page_config here; app.py already does that.

import re, unicodedata
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.title("🏀⚾ Player Props — NBA & MLB (Odds API + ESPN)")

SIM_TRIALS = 10_000  # 10k simulations, same as your NFL page

LEAGUES = {
    "NBA": {
        "espn_path": ("basketball", "nba"),
        # Common Odds API NBA markets
        "valid_markets": [
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_three_points_made",
        ],
        "market_labels": ["Points", "Rebounds", "Assists", "3PM"],
        "odds_sport": "basketball_nba",
    },
    "MLB": {
        "espn_path": ("baseball", "mlb"),
        # Common Odds API MLB markets
        "valid_markets": [
            "player_hits",
            "player_total_bases",
            "player_home_runs",
            "player_rbis",
            "player_runs_scored",
        ],
        "market_labels": ["Hits", "Total Bases", "Home Runs", "RBIs", "Runs"],
        "odds_sport": "baseball_mlb",
    },
}

# ------------------ Utils (names local to this page) ------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

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
    gv = int(g_val)
    if gv <= 1: return np.nan
    mean = sum_x / gv
    var = (sum_x2 / gv) - (mean**2)
    var = var * (gv / (gv - 1))  # Bessel
    return float(np.sqrt(max(var, 1e-6)))

# ------------------ ESPN endpoints ------------------
def espn_urls(league_key: str):
    sport, path = LEAGUES[league_key]["espn_path"]
    base = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{path}"
    return {
        "scoreboard": f"{base}/scoreboard",
        "summary": f"{base}/summary",
    }

def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def list_week_event_ids(league_key: str, year: int, week: int, seasontype: int) -> List[str]:
    urls = espn_urls(league_key)
    js = http_get(urls["scoreboard"], params={"year": year, "week": week, "seasontype": seasontype})
    if not js: return []
    return [str(e.get("id")) for e in js.get("events", []) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(league_key: str, event_id: str) -> Optional[dict]:
    urls = espn_urls(league_key)
    return http_get(urls["summary"], params={"event": event_id})

def _extract_team_players(box: dict) -> List[dict]:
    """Return raw per-player sections with category labels when available."""
    out = []
    try:
        sec = box.get("boxscore", {}).get("players", [])
        for team in sec:
            tname = team.get("team", {}).get("shortDisplayName")
            for p in team.get("statistics", []):
                label = (p.get("name") or "").lower()
                labels = p.get("labels", []) or p.get("descriptions", [])  # ESPN varies
                for a in p.get("athletes", []):
                    out.append({
                        "team": tname,
                        "name": a.get("athlete", {}).get("displayName"),
                        "cat": label,
                        "labels": labels,
                        "vals": a.get("stats") or [],
                    })
    except Exception:
        pass
    return out

# --------- NBA parser (PTS, REB, AST, 3PM) ----------
def parse_boxscore_players_nba(box: dict) -> pd.DataFrame:
    rows = _extract_team_players(box)
    keep = {}
    for r in rows:
        nm = normalize_name(r.get("name"))
        labels = [str(x).upper() for x in (r.get("labels") or [])]
        vals = r.get("vals", [])
        idx = {lab: i for i, lab in enumerate(labels)}

        def grab(label, alt_keys=()):
            for lab in (label, *alt_keys):
                i = idx.get(lab)
                if i is None or i >= len(vals):
                    continue
                v = vals[i]
                try:
                    return float(v)
                except Exception:
                    if isinstance(v, str) and "-" in v:
                        try:
                            return float(v.split("-")[0])  # parse "3-8" → 3
                        except Exception:
                            return np.nan
                    return np.nan
            return np.nan

        rec = keep.setdefault(nm, {"Player": nm, "pts":0.0, "reb":0.0, "ast":0.0, "tpm":0.0})

        # Points (PTS or end-of-row)
        pts = grab("PTS")
        if np.isnan(pts) and "POINTS" in idx:
            try: pts = float(vals[-1])
            except Exception: pts = np.nan
        if not np.isnan(pts): rec["pts"] += pts

        # Rebounds (REB or OREB + DREB)
        reb = grab("REB")
        if np.isnan(reb):
            oreb = grab("OREB"); dreb = grab("DREB")
            if not np.isnan(oreb) or not np.isnan(dreb):
                reb = (0 if np.isnan(oreb) else oreb) + (0 if np.isnan(dreb) else dreb)
        if not np.isnan(reb): rec["reb"] += reb

        ast = grab("AST")
        if not np.isnan(ast): rec["ast"] += ast

        tpm = grab("3PM", alt_keys=("3-PT", "3PTM", "3P MADE", "3PM-3PA"))
        if isinstance(tpm, float) and not np.isnan(tpm):
            rec["tpm"] += tpm
    if not keep:
        return pd.DataFrame(columns=["Player","pts","reb","ast","tpm"])
    return pd.DataFrame(list(keep.values()))

# --------- MLB parser (H, TB, HR, RBI, R) ----------
def parse_boxscore_players_mlb(box: dict) -> pd.DataFrame:
    rows = _extract_team_players(box)
    keep = {}
    for r in rows:
        nm = normalize_name(r.get("name"))
        labels = [str(x).upper() for x in (r.get("labels") or [])]
        vals = r.get("vals", [])
        idx = {lab: i for i, lab in enumerate(labels)}

        def grab(label):
            i = idx.get(label)
            if i is not None and i < len(vals):
                v = vals[i]
                try:
                    return float(v)
                except Exception:
                    if isinstance(v, str) and "-" in v:
                        try:
                            return float(v.split("-")[0])
                        except Exception:
                            return np.nan
            return np.nan

        rec = keep.setdefault(nm, {"Player": nm, "H":0.0, "TB":0.0, "HR":0.0, "RBI":0.0, "R":0.0})
        for lab, key in [("H","H"), ("TB","TB"), ("HR","HR"), ("RBI","RBI"), ("R","R")]:
            v = grab(lab)
            if not np.isnan(v): rec[key] += v

    if not keep:
        return pd.DataFrame(columns=["Player","H","TB","HR","RBI","R"])
    return pd.DataFrame(list(keep.values()))

# --------- Aggregate a season/week range ----------
@st.cache_data(show_spinner=True)
def build_espn_season_agg(league_key: str, year: int, weeks: List[int], seasontype: int) -> pd.DataFrame:
    events = []
    for wk in weeks:
        events.extend(list_week_event_ids(league_key, year, wk, seasontype))
    if not events:
        return pd.DataFrame()

    if league_key == "NBA":
        stat_keys = ["pts","reb","ast","tpm"]
        parser = parse_boxscore_players_nba
    else:
        stat_keys = ["H","TB","HR","RBI","R"]
        parser = parse_boxscore_players_mlb

    totals, sumsqs, games = {}, {}, {}
    def init_p(p: str):
        if p not in totals:
            totals[p] = {k:0.0 for k in stat_keys}
            sumsqs[p] = {k:0.0 for k in stat_keys}
            games[p]  = 0

    prog = st.progress(0.0, text=f"Crawling {len(events)} games…")
    for j, ev in enumerate(events, 1):
        box = fetch_boxscore_event(league_key, ev)
        if box:
            df = parser(box)
            for _, r in df.iterrows():
                p = r["Player"]; init_p(p)
                played = any([to_float(r.get(k, np.nan)) > 0 for k in stat_keys])
                if played: games[p] += 1
                for k in stat_keys:
                    v = to_float(r.get(k, np.nan))
                    if not np.isnan(v):
                        totals[p][k] += v
                        sumsqs[p][k] += v*v
        prog.progress(j/len(events))

    rows = []
    for p, stat in totals.items():
        g = max(1, int(games.get(p, 0)))
        row = {"Player": p, "g": g}
        for k in stat_keys:
            row[k] = stat[k]
            row[f"sq_{k}"] = sumsqs[p][k]
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------ UI — scope ------------------
st.markdown("### 1) League, season scope")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    league_key = st.selectbox("League", list(LEAGUES.keys()), index=0)
with c2:
    season = st.number_input("Season (year)", min_value=2015, max_value=2100, value=2025, step=1)
with c3:
    seasontype = st.selectbox("Season type", options=[("Preseason",1),("Regular",2),("Postseason",3)],
                              index=1, format_func=lambda x: x[0])[1]

# ESPN scoreboard uses "week" buckets for NBA/MLB as well; expose a broad slider.
if league_key == "NBA":
    wk = st.slider("Week buckets", 1, 30, (1, 30)) if seasontype==2 else (st.slider("Pre Weeks", 0, 4, (0,3)) if seasontype==1 else st.slider("Post Weeks", 1, 5, (1,5)))
else:
    wk = st.slider("Weeks", 1, 30, (1, 30)) if seasontype==2 else (st.slider("Pre Weeks", 0, 4, (0,3)) if seasontype==1 else st.slider("Post Weeks", 1, 5, (1,5)))
weeks = list(range(wk[0], wk[1] + 1))

# ------------------ Build projections ------------------
st.markdown("### 2) Build per-player projections from ESPN")
if st.button("📥 Build NBA/MLB projections"):
    season_df = build_espn_season_agg(league_key, season, weeks, seasontype)
    if season_df.empty:
        st.error("No data returned from ESPN.")
        st.stop()

    g = season_df["g"].clip(lower=1)

    if league_key == "NBA":
        # per-game means
        season_df["mu_pts_raw"] = season_df["pts"] / g
        season_df["mu_reb_raw"] = season_df["reb"] / g
        season_df["mu_ast_raw"] = season_df["ast"] / g
        season_df["mu_tpm_raw"] = season_df["tpm"] / g

        # SDs
        sd_pts = season_df.apply(lambda r: sample_sd(r["pts"], r["sq_pts"], r["g"]), axis=1)
        sd_reb = season_df.apply(lambda r: sample_sd(r["reb"], r["sq_reb"], r["g"]), axis=1)
        sd_ast = season_df.apply(lambda r: sample_sd(r["ast"], r["sq_ast"], r["g"]), axis=1)
        sd_tpm = season_df.apply(lambda r: sample_sd(r["tpm"], r["sq_tpm"], r["g"]), axis=1)

        SD_INFLATE = 1.10
        season_df["sd_pts"] = np.where(np.isnan(sd_pts), np.nan, np.maximum(3.0, sd_pts) * SD_INFLATE)
        season_df["sd_reb"] = np.where(np.isnan(sd_reb), np.nan, np.maximum(1.5, sd_reb) * SD_INFLATE)
        season_df["sd_ast"] = np.where(np.isnan(sd_ast), np.nan, np.maximum(1.5, sd_ast) * SD_INFLATE)
        season_df["sd_tpm"] = np.where(np.isnan(sd_tpm), np.nan, np.maximum(0.8, sd_tpm) * SD_INFLATE)

        # neutral scaling
        season_df["mu_pts"] = season_df["mu_pts_raw"]
        season_df["mu_reb"] = season_df["mu_reb_raw"]
        season_df["mu_ast"] = season_df["mu_ast_raw"]
        season_df["mu_tpm"] = season_df["mu_tpm_raw"]

        st.session_state["nba_proj"] = season_df[[
            "Player","g",
            "mu_pts_raw","mu_pts","sd_pts",
            "mu_reb_raw","mu_reb","sd_reb",
            "mu_ast_raw","mu_ast","sd_ast",
            "mu_tpm_raw","mu_tpm","sd_tpm"
        ]]

        st.dataframe(st.session_state["nba_proj"].head(20), use_container_width=True)

    else:  # MLB
        season_df["mu_hits_raw"] = season_df["H"] / g
        season_df["mu_tb_raw"]   = season_df["TB"] / g
        season_df["mu_hr_raw"]   = season_df["HR"] / g
        season_df["mu_rbi_raw"]  = season_df["RBI"] / g
        season_df["mu_runs_raw"] = season_df["R"] / g

        sd_H  = season_df.apply(lambda r: sample_sd(r["H"],  r["sq_H"],  r["g"]), axis=1)
        sd_TB = season_df.apply(lambda r: sample_sd(r["TB"], r["sq_TB"], r["g"]), axis=1)
        sd_HR = season_df.apply(lambda r: sample_sd(r["HR"], r["sq_HR"], r["g"]), axis=1)
        sd_RBI= season_df.apply(lambda r: sample_sd(r["RBI"],r["sq_RBI"],r["g"]), axis=1)
        sd_R  = season_df.apply(lambda r: sample_sd(r["R"],  r["sq_R"],  r["g"]), axis=1)

        SD_INFLATE = 1.10
        season_df["sd_hits"] = np.where(np.isnan(sd_H),  np.nan, np.maximum(0.8, sd_H)  * SD_INFLATE)
        season_df["sd_tb"]   = np.where(np.isnan(sd_TB), np.nan, np.maximum(1.2, sd_TB) * SD_INFLATE)
        season_df["sd_hr"]   = np.where(np.isnan(sd_HR), np.nan, np.maximum(0.3, sd_HR) * SD_INFLATE)
        season_df["sd_rbi"]  = np.where(np.isnan(sd_RBI),np.nan, np.maximum(0.8, sd_RBI)* SD_INFLATE)
        season_df["sd_runs"] = np.where(np.isnan(sd_R),  np.nan, np.maximum(0.8, sd_R)  * SD_INFLATE)

        # neutral scaling
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

valid_markets = LEAGUES.get(st.session_state.get("league_key_over"), {}).get("valid_markets")
# Build from current league selection instead:
league_key = st.session_state.get("league_key_current", None)
if league_key is None:
    # set on first render
    league_key = st.selectbox("Active League for Odds API", list(LEAGUES.keys()), index=0, key="league_key_current")
else:
    st.selectbox("Active League for Odds API", list(LEAGUES.keys()), index=list(LEAGUES.keys()).index(league_key), key="league_key_current")

valid_markets = LEAGUES[league_key]["valid_markets"]
markets = st.multiselect("Markets to fetch", valid_markets, default=valid_markets)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200: raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_events(league_key: str, api_key: str, lookahead_days: int, region: str):
    sport = LEAGUES[league_key]["odds_sport"]
    base = f"https://api.the-odds-api.com/v4/sports/{sport}/events"
    params = {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region}
    return odds_get(base, params)

def fetch_event_props(league_key: str, api_key: str, event_id: str, region: str, markets: List[str]):
    sport = LEAGUES[league_key]["odds_sport"]
    base = f"https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"}
    return odds_get(base, params)

events = []
if api_key:
    try:
        events = list_events(league_key, api_key, lookahead, region)
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
go = st.button("Fetch lines & simulate (NBA/MLB)")

if go:
    if league_key == "NBA":
        proj = st.session_state.get("nba_proj", pd.DataFrame())
    else:
        proj = st.session_state.get("mlb_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build ESPN projections first (Step 2).")
        st.stop()

    try:
        data = fetch_event_props(league_key, api_key, event_id, region, markets)
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
                if mkey not in LEAGUES[league_key]["valid_markets"] or not name or side not in ("Over","Under"):
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

    props_df = (pd.DataFrame(rows).groupby(["market","player_norm","side"], as_index=False)
                              .agg(line=("point","median"), n_books=("point","size")))

    out = []
    names = set(proj["Player"])
    for _, r in props_df.iterrows():
        market, player, point, side = r["market"], r["player_norm"], r["line"], r["side"]
        if player not in names or pd.isna(point): continue
        row = proj.loc[proj["Player"] == player].iloc[0]

        if league_key == "NBA":
            if market == "player_points":
                mu, sd = float(row["mu_pts"]), float(row["sd_pts"]);  mu_raw = float(row["mu_pts_raw"])
            elif market == "player_rebounds":
                mu, sd = float(row["mu_reb"]), float(row["sd_reb"]);  mu_raw = float(row["mu_reb_raw"])
            elif market == "player_assists":
                mu, sd = float(row["mu_ast"]), float(row["sd_ast"]);  mu_raw = float(row["mu_ast_raw"])
            elif market == "player_three_points_made":
                mu, sd = float(row["mu_tpm"]), float(row["sd_tpm"]);  mu_raw = float(row["mu_tpm_raw"])
            else:
                continue
        else:  # MLB
            if market == "player_hits":
                mu, sd = float(row["mu_hits"]), float(row["sd_hits"]); mu_raw = float(row["mu_hits_raw"])
            elif market == "player_total_bases":
                mu, sd = float(row["mu_tb"]), float(row["sd_tb"]);     mu_raw = float(row["mu_tb_raw"])
            elif market == "player_home_runs":
                mu, sd = float(row["mu_hr"]), float(row["sd_hr"]);     mu_raw = float(row["mu_hr_raw"])
            elif market == "player_rbis":
                mu, sd = float(row["mu_rbi"]), float(row["sd_rbi"]);   mu_raw = float(row["mu_rbi_raw"])
            elif market == "player_runs_scored":
                mu, sd = float(row["mu_runs"]), float(row["sd_runs"]); mu_raw = float(row["mu_runs_raw"])
            else:
                continue

        if np.isnan(mu) or np.isnan(sd): continue
        p_over = t_over_prob(mu, sd, float(point), SIM_TRIALS)
        p = p_over if side == "Over" else 1.0 - p_over

        out.append({
            "market": market,
            "player": player,
            "side": side,
            "line": round(float(point), 2),
            "Avg (raw)": round(mu_raw, 2),
            "μ (scaled)": round(mu, 2),
            "σ (per-game)": round(sd, 2),
            "Games": int(row["g"]),
            "Win Prob %": round(100*p, 2),
            "books": int(r["n_books"]),
        })

    if not out:
        st.warning("No props matched your projections.")
        st.stop()

    results = (pd.DataFrame(out)
               .drop_duplicates(subset=["market","player","side"])
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    st.subheader(f"Simulated probabilities — {league_key}")
    labels = LEAGUES[league_key]["market_labels"]
    tabs = st.tabs(["All", *labels])
    colcfg = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "Avg (raw)": st.column_config.NumberColumn("Avg (raw)", format="%.2f"),
        "μ (scaled)": st.column_config.NumberColumn("μ (scaled)", format="%.2f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
        "Games": st.column_config.NumberColumn("Games", width="small"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "books": st.column_config.NumberColumn("#Books", width="small"),
    }
    with tabs[0]:
        st.dataframe(results, use_container_width=True, hide_index=True, column_config=colcfg)

    label_to_key = dict(zip(labels, LEAGUES[league_key]["valid_markets"]))
    show_top_chart = st.toggle("Show Top Picks chart (per tab)", value=True, key=f"topchart_{league_key}")
    for i, label in enumerate(labels, start=1):
        mkt = label_to_key[label]
        with tabs[i]:
            sub = results[results["market"] == mkt].copy()
            st.dataframe(sub, use_container_width=True, hide_index=True, column_config=colcfg)
            if show_top_chart and not sub.empty:
                top = sub.sort_values("Win Prob %", ascending=False).head(12)
                st.bar_chart(top.set_index("player")["Win Prob %"], use_container_width=True)

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name=f"{league_key.lower()}_props_sim_results.csv",
        mime="text/csv",
    )
