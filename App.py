# Player Props — Odds API + ESPN (per-game averages + defense multipliers + t-dist)
# Run: streamlit run app.py

import re, math, unicodedata
from io import StringIO
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import process, fuzz

# ------------------ Page / theme ------------------
st.set_page_config(page_title="NFL Player Props — Odds API + ESPN", layout="wide")
st.title("📈 NFL Player Props — Odds API + ESPN (per-game averages)")

SIM_TRIALS = 10_000
VALID_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receiving_yds",  # correct Odds API key
    "player_receptions",
    "player_pass_tds",
]

# ------------------ Embedded 2025 defense EPA ------------------
DEFENSE_EPA_2025 = """Team,EPA_Pass,EPA_Rush,Comp_Pct
Minnesota Vikings,-0.37,0.06,0.6762
Jacksonville Jaguars,-0.17,-0.05,0.5962
Denver Broncos,-0.10,-0.12,0.5746
Los Angeles Chargers,-0.17,0.01,0.5938
Detroit Lions,0.00,-0.22,0.6271
Philadelphia Eagles,-0.11,-0.04,0.5693
Houston Texans,-0.16,0.04,0.5714
Los Angeles Rams,-0.12,0.00,0.6640
Seattle Seahawks,0.00,-0.19,0.6645
San Francisco 49ers,-0.09,-0.03,0.6829
Tampa Bay Buccaneers,-0.02,-0.11,0.6429
Atlanta Falcons,-0.13,0.05,0.5769
Cleveland Browns,0.06,-0.17,0.6442
Indianapolis Colts,-0.04,-0.05,0.6643
Kansas City Chiefs,-0.09,0.09,0.6694
Arizona Cardinals,0.06,-0.14,0.6369
Las Vegas Raiders,0.14,-0.22,0.6565
Green Bay Packers,0.03,-0.07,0.6815
Chicago Bears,0.01,0.00,0.7368
Buffalo Bills,-0.06,0.10,0.6214
Carolina Panthers,0.03,0.05,0.6239
Pittsburgh Steelers,0.11,-0.05,0.6957
Washington Commanders,0.18,-0.12,0.6098
New England Patriots,0.19,-0.15,0.7120
New York Giants,-0.01,0.19,0.6375
New Orleans Saints,0.20,-0.06,0.7117
Cincinnati Bengals,0.13,0.04,0.6536
New York Jets,0.23,-0.03,0.6577
Tennessee Titans,0.16,0.07,0.6984
Baltimore Ravens,0.14,0.12,0.6667
Dallas Cowboys,0.40,0.06,0.7333
Miami Dolphins,0.34,0.12,0.7757
"""
@st.cache_data(show_spinner=False)
def load_defense_table() -> pd.DataFrame:
    df = pd.read_csv(StringIO(DEFENSE_EPA_2025))
    for c in ["EPA_Pass","EPA_Rush","Comp_Pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    def adj_from_epa(s: pd.Series, scale: float) -> pd.Series:
        x = s.fillna(0.0)
        adj = 1.0 - scale * x
        return adj.clip(0.7, 1.3)
    pass_adj = adj_from_epa(df["EPA_Pass"], 0.8)
    rush_adj = adj_from_epa(df["EPA_Rush"], 0.8)
    comp = df["Comp_Pct"].clip(0.45, 0.80).fillna(df["Comp_Pct"].mean())
    comp_adj = (1.0 + (comp - comp.mean()) * 0.6).clip(0.7, 1.3)
    recv_adj = (0.7 * pass_adj + 0.3 * comp_adj).clip(0.7, 1.3)
    return pd.DataFrame({"Team": df["Team"], "pass_adj": pass_adj, "rush_adj": rush_adj, "recv_adj": recv_adj})
DEF_TABLE = load_defense_table()

# ------------------ Utils ------------------
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
def clamp(x, lo=0.85, hi=1.15): return float(np.clip(x, lo, hi))

# ------------------ ESPN crawl ------------------
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY   = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def list_week_event_ids(year: int, week: int, seasontype: int = 2) -> List[str]:
    js = http_get(SCOREBOARD, params={"year": year, "week": week, "seasontype": seasontype})
    if not js: return []
    return [str(e.get("id")) for e in js.get("events", []) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id: str) -> Optional[dict]:
    return http_get(SUMMARY, params={"event": event_id})

def _extract_team_players(box: dict) -> List[dict]:
    players = []
    try:
        sec = box.get("boxscore", {}).get("players", [])
        for team in sec:
            for p in team.get("statistics", []):
                label = (p.get("name") or "").lower()
                for a in p.get("athletes", []):
                    players.append({
                        "team": team.get("team", {}).get("shortDisplayName"),
                        "name": a.get("athlete", {}).get("displayName"),
                        "cat": label,
                        "vals": a.get("stats") or []
                    })
    except Exception:
        pass
    return players

def parse_boxscore_players(box: dict) -> pd.DataFrame:
    rows = _extract_team_players(box)
    out = []
    for r in rows:
        nm = normalize_name(r.get("name"))
        cat = r.get("cat"); vals = r.get("vals", [])
        try:
            if "passing" in cat:
                yds = to_float(vals[1]) if len(vals) > 1 else np.nan
                tds = to_float(vals[2]) if len(vals) > 2 else np.nan
                out.append({"Player": nm, "pass_yds": yds, "pass_tds": tds,
                            "rush_yds": 0.0, "rec_yds": 0.0, "rec": 0.0})
            elif "rushing" in cat:
                yds = to_float(vals[1]) if len(vals) > 1 else np.nan
                out.append({"Player": nm, "pass_yds": 0.0, "pass_tds": 0.0,
                            "rush_yds": yds, "rec_yds": 0.0, "rec": 0.0})
            elif "receiving" in cat:
                recs = to_float(vals[0]) if len(vals) > 0 else np.nan
                rec_yds = to_float(vals[1]) if len(vals) > 1 else np.nan
                out.append({"Player": nm, "pass_yds": 0.0, "pass_tds": 0.0,
                            "rush_yds": 0.0, "rec_yds": rec_yds, "rec": recs})
        except Exception:
            continue
    if not out:
        return pd.DataFrame(columns=["Player","pass_yds","pass_tds","rush_yds","rec_yds","rec"])
    return pd.DataFrame(out).groupby("Player", as_index=False).sum(numeric_only=True)

@st.cache_data(show_spinner=True)
def build_espn_season_agg(year: int, weeks: List[int], seasontype: int) -> pd.DataFrame:
    """
    Crawl ESPN boxscores and aggregate totals + sum of squares,
    and count games where a player produced any stat (our games denominator).
    """
    totals, sumsqs, games = {}, {}, {}
    def init_p(p):
        if p not in totals:
            totals[p] = {"pass_yds":0.0,"pass_tds":0.0,"rush_yds":0.0,"rec_yds":0.0,"rec":0.0}
            sumsqs[p] = {"pass_yds":0.0,"rush_yds":0.0,"rec_yds":0.0,"rec":0.0}
            games[p]  = 0

    events = []
    for wk in weeks: events.extend(list_week_event_ids(year, wk, seasontype))
    if not events:
        return pd.DataFrame(columns=["Player","g","pass_yds","pass_tds","rush_yds","rec_yds","rec",
                                     "sq_pass_yds","sq_rush_yds","sq_rec_yds","sq_rec"])

    prog = st.progress(0.0, text=f"Crawling {len(events)} games...")
    for j, ev in enumerate(events, 1):
        box = fetch_boxscore_event(ev)
        if box:
            df = parse_boxscore_players(box)
            for _, r in df.iterrows():
                p = r["Player"]; init_p(p)

                py = to_float(r.get("pass_yds", np.nan))
                pt = to_float(r.get("pass_tds", np.nan))
                ry = to_float(r.get("rush_yds", np.nan))
                rcy= to_float(r.get("rec_yds",  np.nan))
                rc = to_float(r.get("rec",      np.nan))

                # count as a game if any tracked stat > 0
                if any([(not np.isnan(v) and v > 0) for v in [py, pt, ry, rcy, rc]]):
                    games[p] += 1

                for k, v in [("pass_yds", py), ("pass_tds", pt), ("rush_yds", ry), ("rec_yds", rcy), ("rec", rc)]:
                    if not np.isnan(v): totals[p][k] += v

                for k, v in [("pass_yds", py), ("rush_yds", ry), ("rec_yds", rcy), ("rec", rc)]:
                    if not np.isnan(v): sumsqs[p][k] += v*v
        prog.progress(j/len(events))

    rows = []
    for p, stat in totals.items():
        g = max(1, int(games.get(p, 0)))
        rows.append({"Player": p, "g": g, **stat,
                     "sq_pass_yds": sumsqs[p]["pass_yds"], "sq_rush_yds": sumsqs[p]["rush_yds"],
                     "sq_rec_yds": sumsqs[p]["rec_yds"], "sq_rec": sumsqs[p]["rec"]})
    return pd.DataFrame(rows)

# ------------------ UI Step 1 ------------------
st.markdown("### 1) Season scope & opponent defense")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    season = st.number_input("Season", min_value=2015, max_value=2100, value=2025, step=1)
with c2:
    seasontype = st.selectbox("Season type", options=[("Preseason",1),("Regular",2),("Postseason",3)],
                              index=1, format_func=lambda x: x[0])[1]
with c3:
    week_range = st.slider("Weeks to include", 1, 18, (1, 18)) if seasontype==2 else (st.slider("Preseason Weeks", 0, 4, (0,3)) if seasontype==1 else st.slider("Postseason Weeks", 1, 5, (1,5)))
weeks = list(range(week_range[0], week_range[1] + 1))
opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()

# ------------------ Build projections (per-game means) ------------------
st.markdown("### 2) Build per-player projections from ESPN")

if st.button("📥 Build projections"):
    season_df = build_espn_season_agg(season, weeks, seasontype)
    if season_df.empty:
        st.error("No data returned from ESPN."); st.stop()

    # ------- PURE PER-GAME AVERAGES across all counted games -------
    g = season_df["g"].clip(lower=1)
    mu_pass_yds_pg = season_df["pass_yds"] / g
    mu_pass_tds_pg = season_df["pass_tds"] / g
    mu_rush_yds_pg = season_df["rush_yds"] / g
    mu_rec_yds_pg  = season_df["rec_yds"]  / g
    mu_recs_pg     = season_df["rec"]      / g

    # Keep raw (pre-defense) for display
    season_df["mu_pass_yds_raw"]   = mu_pass_yds_pg
    season_df["mu_pass_tds_raw"]   = mu_pass_tds_pg
    season_df["mu_rush_yds_raw"]   = mu_rush_yds_pg
    season_df["mu_rec_yds_raw"]    = mu_rec_yds_pg
    season_df["mu_receptions_raw"] = mu_recs_pg

    # Sample SD from sums & sum of squares (Bessel-corrected) using same g
    def sample_sd(sum_x, sum_x2, g_val):
        gv = int(g_val)
        if gv <= 1: return np.nan
        mean = sum_x / gv
        var  = (sum_x2 / gv) - (mean**2)
        var  = var * (gv / (gv - 1))
        return float(np.sqrt(max(var, 1e-6)))

    sd_pass_yds = season_df.apply(lambda r: sample_sd(r["pass_yds"], r["sq_pass_yds"], r["g"]), axis=1)
    sd_rush_yds = season_df.apply(lambda r: sample_sd(r["rush_yds"], r["sq_rush_yds"], r["g"]), axis=1)
    sd_rec_yds  = season_df.apply(lambda r: sample_sd(r["rec_yds"],  r["sq_rec_yds"],  r["g"]), axis=1)
    sd_recs     = season_df.apply(lambda r: sample_sd(r["rec"],      r["sq_rec"],      r["g"]), axis=1)

    # Slight floor to avoid degenerate SDs; no extra % knockback applied
    SD_INFLATE = 1.10
    season_df["sd_pass_yds"]   = np.where(np.isnan(sd_pass_yds), np.nan, np.maximum(25.0, sd_pass_yds) * SD_INFLATE)
    season_df["sd_rush_yds"]   = np.where(np.isnan(sd_rush_yds), np.nan, np.maximum(12.0, sd_rush_yds) * SD_INFLATE)
    season_df["sd_rec_yds"]    = np.where(np.isnan(sd_rec_yds),  np.nan, np.maximum(10.0, sd_rec_yds)  * SD_INFLATE)
    season_df["sd_receptions"] = np.where(np.isnan(sd_recs),     np.nan, np.maximum(1.0,  sd_recs)     * SD_INFLATE)

    # Apply ONLY defense scaling (no additional percentage knockback)
    season_df["mu_pass_yds"]   = mu_pass_yds_pg * clamp(scalers["pass_adj"])
    season_df["mu_pass_tds"]   = mu_pass_tds_pg * clamp(scalers["pass_adj"])
    season_df["mu_rush_yds"]   = mu_rush_yds_pg * clamp(scalers["rush_adj"])
    season_df["mu_rec_yds"]    = mu_rec_yds_pg  * clamp(scalers["recv_adj"])
    season_df["mu_receptions"] = mu_recs_pg     * clamp(scalers["recv_adj"])

    # Save slim projections (+ raw means + games) to session
    st.session_state["qb_proj"] = season_df[[
        "Player","g",
        "mu_pass_yds_raw","mu_pass_yds","sd_pass_yds",
        "mu_pass_tds_raw","mu_pass_tds"
    ]].dropna(how="all")

    st.session_state["rb_proj"] = season_df[[
        "Player","g",
        "mu_rush_yds_raw","mu_rush_yds","sd_rush_yds"
    ]].dropna(how="all")

    st.session_state["wr_proj"] = season_df[[
        "Player","g",
        "mu_receptions_raw","mu_receptions","sd_receptions",
        "mu_rec_yds_raw","mu_rec_yds","sd_rec_yds"
    ]].dropna(how="all")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("QB (per-game from ESPN)")
        st.dataframe(st.session_state["qb_proj"].head(12), use_container_width=True)
    with c2:
        st.subheader("RB (per-game from ESPN)")
        st.dataframe(st.session_state["rb_proj"].head(12), use_container_width=True)
    with c3:
        st.subheader("WR/TE (per-game from ESPN)")
        st.dataframe(st.session_state["wr_proj"].head(12), use_container_width=True)

# ------------------ Odds API ------------------
st.markdown("### 3) Pick a game & markets from The Odds API (event endpoint)")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets to fetch", VALID_MARKETS, default=VALID_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()
def list_nfl_events(api_key: str, lookahead_days: int, region: str):
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
    params = {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region}
    return odds_get(base, params)
def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    base = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"}
    return odds_get(base, params)

events = []
if api_key:
    try:
        events = list_nfl_events(api_key, lookahead, region)
    except Exception as e:
        st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list upcoming games.")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ------------------ Fetch props & simulate ----------------
st.markdown("### 4) Fetch props for this event and simulate")
go = st.button("Fetch lines & simulate")
if go:
    qb_proj = st.session_state.get("qb_proj", pd.DataFrame())
    rb_proj = st.session_state.get("rb_proj", pd.DataFrame())
    wr_proj = st.session_state.get("wr_proj", pd.DataFrame())

    if qb_proj.empty and rb_proj.empty and wr_proj.empty:
        st.warning("Build ESPN projections first (Step 2).")
        st.stop()

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
                name = normalize_name(o.get("description"))  # player name
                side = o.get("name")                        # Over/Under
                point = o.get("point")
                if mkey not in VALID_MARKETS or not name or side not in ("Over","Under"):
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

    raw_props = pd.DataFrame(rows).drop_duplicates()
    props_df = (raw_props.groupby(["market","player_norm","side"], as_index=False)
                        .agg(line=("point","median"), n_books=("point","size")))

    out_rows = []
    qb_names = set(qb_proj["Player"])
    rb_names = set(rb_proj["Player"])
    wr_names = set(wr_proj["Player"])  # receiving & receptions

    for _, r in props_df.iterrows():
        market = r["market"]; player = r["player_norm"]; point = r["line"]; side = r["side"]

        if market == "player_pass_tds" and player in qb_names:
            row = qb_proj.loc[qb_proj["Player"] == player].iloc[0]
            lam   = float(row["mu_pass_tds"])         # per-game after defense scaling
            mu_raw= float(row["mu_pass_tds_raw"])     # raw per-game avg
            g_used= int(row["g"])
            if pd.isna(point): continue
            p_over = float((np.random.poisson(lam=lam, size=SIM_TRIALS) > float(point)).mean())
            p = p_over if side == "Over" else 1.0 - p_over
            mu, sd = lam, float("nan")

        elif market == "player_pass_yds" and player in qb_names:
            row = qb_proj.loc[qb_proj["Player"] == player].iloc[0]
            mu, sd = float(row["mu_pass_yds"]), float(row["sd_pass_yds"])
            mu_raw = float(row["mu_pass_yds_raw"]); g_used = int(row["g"])
            if np.isnan(mu) or np.isnan(sd) or pd.isna(point): continue
            p_over = t_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_rush_yds" and player in rb_names:
            row = rb_proj.loc[rb_proj["Player"] == player].iloc[0]
            mu, sd = float(row["mu_rush_yds"]), float(row["sd_rush_yds"])
            mu_raw = float(row["mu_rush_yds_raw"]); g_used = int(row["g"])
            if np.isnan(mu) or np.isnan(sd) or pd.isna(point): continue
            p_over = t_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_receptions" and player in wr_names:
            row = wr_proj.loc[wr_proj["Player"] == player].iloc[0]
            mu, sd = float(row["mu_receptions"]), float(row["sd_receptions"])
            mu_raw = float(row["mu_receptions_raw"]); g_used = int(row["g"])
            if np.isnan(mu) or np.isnan(sd) or pd.isna(point): continue
            p_over = t_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_receiving_yds" and player in wr_names:
            row = wr_proj.loc[wr_proj["Player"] == player].iloc[0]
            mu, sd = float(row["mu_rec_yds"]), float(row["sd_rec_yds"])
            mu_raw = float(row["mu_rec_yds_raw"]); g_used = int(row["g"])
            if np.isnan(mu) or np.isnan(sd) or pd.isna(point): continue
            p_over = t_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        else:
            continue

        out_rows.append({
            "market": market,
            "player": player,
            "side": side,
            "line": None if pd.isna(point) else round(float(point), 2),
            "Avg (raw)": round(mu_raw, 2),        # true per-game average (pre-defense)
            "μ (scaled)": None if np.isnan(mu) else round(float(mu), 2),  # after defense scaling
            "σ (per-game)": None if (isinstance(sd, float) and np.isnan(sd)) else (None if pd.isna(sd) else round(float(sd), 2)),
            "Games": g_used,
            "Win Prob %": round(100*p, 2),
            "books": int(r["n_books"]),
        })

    if not out_rows:
        st.warning("No props matched your projections.")
        st.stop()

    results = (pd.DataFrame(out_rows)
                 .drop_duplicates(subset=["market","player","side"])
                 .sort_values(["market","Win Prob %"], ascending=[True, False])
                 .reset_index(drop=True))

    st.subheader("Simulated probabilities (per-game averages + defense multipliers)")
    tabs = st.tabs(["All", "Passing Yards", "Rushing Yards", "Receptions", "Receiving Yards", "Passing TDs"])
    market_map = {
        "Passing Yards": "player_pass_yds",
        "Rushing Yards": "player_rush_yds",
        "Receptions": "player_receptions",
        "Receiving Yards": "player_receiving_yds",
        "Passing TDs": "player_pass_tds",
    }
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

    show_top_chart = st.toggle("Show Top Picks chart (per tab)", value=True, key="topchart")
    for i, label in enumerate(["Passing Yards", "Rushing Yards", "Receptions", "Receiving Yards", "Passing TDs"], start=1):
        mkt = market_map[label]
        with tabs[i]:
            sub = results[results["market"] == mkt].copy()
            st.dataframe(sub, use_container_width=True, hide_index=True, column_config=colcfg)
            if show_top_chart and not sub.empty:
                top = sub.sort_values("Win Prob %", ascending=False).head(12)
                st.bar_chart(top.set_index("player")["Win Prob %"], use_container_width=True)

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="props_sim_results.csv",
        mime="text/csv",
    )
