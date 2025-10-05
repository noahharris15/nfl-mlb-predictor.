# Player Props — Odds API + ESPN (crawl every 2025 game)
# Run: streamlit run app.py

import math, re, unicodedata, time
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import fuzz, process

st.set_page_config(page_title="NFL Player Props — Odds API + ESPN", layout="wide")
st.title("📈 NFL Player Props — Odds API + ESPN (full-season box scores)")

SIM_TRIALS = 10_000
VALID_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    "player_anytime_td",
    "player_pass_tds",
]

# ---------------- Embedded 2025 defense EPA ----------------
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
st.caption("Defense multipliers (1.0 = neutral) are embedded from your 2025 EPA sheet.")

# ---------------- Utilities ----------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = strip_accents(n)
    parts = n.split()
    if len(parts) >= 2 and len(parts[0]) == 1:
        n = " ".join(parts[1:])
    return n
def fuzzy_pick(name: str, candidates: List[str], cutoff=60) -> Optional[str]:
    if not candidates: return None
    res = process.extractOne(name, candidates, scorer=fuzz.WRatio)
    return res[0] if res and res[1] >= cutoff else None
def norm_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    return float((np.random.normal(mu, sd, size=trials) > line).mean())
def anytime_yes_prob_poisson(lam: float) -> float:
    lam = max(1e-6, float(lam))
    return 1.0 - math.exp(-lam)
def to_float(x) -> float:
    try: return float(x)
    except Exception: return float("nan")

# ---------------- ESPN scraping (scoreboard -> box score) ----------------
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def list_week_event_ids(year: int, week: int, seasontype: int = 2) -> List[str]:
    # seasontype: 1=pre, 2=reg, 3=post
    js = http_get(SCOREBOARD, params={"year": year, "week": week, "seasontype": seasontype})
    if not js: return []
    events = js.get("events") or []
    return [str(e.get("id")) for e in events if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id: str) -> Optional[dict]:
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
    return http_get(url, params={"event": event_id})

def _extract_team_players(box: dict) -> List[dict]:
    players = []
    try:
        sec = box.get("boxscore", {}).get("players", [])
        for team in sec:
            for p in team.get("statistics", []):
                # p["athletes"] is a list of players with that stat category (passing/rushing/receiving)
                label = (p.get("name") or "").lower()
                for a in p.get("athletes", []):
                    row = {"team": team.get("team", {}).get("shortDisplayName"), "name": a.get("athlete", {}).get("displayName")}
                    stats = a.get("stats") or []
                    # stats is a list of strings like "20-31", "245", "2", etc depending on category
                    # We'll map by category label
                    row["cat"] = label
                    row["vals"] = stats
                    players.append(row)
    except Exception:
        pass
    return players

def parse_boxscore_players(box: dict) -> pd.DataFrame:
    rows = _extract_team_players(box)
    # Convert category rows into numeric measures we need
    out = []
    for r in rows:
        nm = normalize_name(r.get("name"))
        cat = r.get("cat")
        vals = r.get("vals", [])
        try:
            if "passing" in cat:
                # common layout: ["CMP-ATT","YDS","TD","INT","SACKS-YDS","QBR","RTG"]
                yds = to_float(vals[1]) if len(vals) > 1 else np.nan
                tds = to_float(vals[2]) if len(vals) > 2 else np.nan
                out.append({"Player": nm, "pass_yds": yds, "pass_tds": tds, "rush_yds": 0.0, "rush_tds": 0.0, "rec": 0.0, "rec_tds": 0.0})
            elif "rushing" in cat:
                # layout: ["CAR","YDS","TD","LG"]
                yds = to_float(vals[1]) if len(vals) > 1 else np.nan
                tds = to_float(vals[2]) if len(vals) > 2 else np.nan
                out.append({"Player": nm, "pass_yds": 0.0, "pass_tds": 0.0, "rush_yds": yds, "rush_tds": tds, "rec": 0.0, "rec_tds": 0.0})
            elif "receiving" in cat:
                # layout: ["REC","YDS","TD","LG","TGT"]
                recs = to_float(vals[0]) if len(vals) > 0 else np.nan
                tds = to_float(vals[2]) if len(vals) > 2 else np.nan
                out.append({"Player": nm, "pass_yds": 0.0, "pass_tds": 0.0, "rush_yds": 0.0, "rush_tds": 0.0, "rec": recs, "rec_tds": tds})
        except Exception:
            continue
    if not out:
        return pd.DataFrame(columns=["Player","pass_yds","pass_tds","rush_yds","rush_tds","rec","rec_tds"])
    df = pd.DataFrame(out)
    # aggregate duplicates (same player across categories/teams record)
    df = df.groupby("Player", as_index=False).sum(numeric_only=True)
    return df

@st.cache_data(show_spinner=True)
def build_espn_season_agg(year: int, weeks: List[int], seasontype: int) -> pd.DataFrame:
    """
    Crawl every requested week, pull all events, parse box scores, and build
    per-player totals + games played.
    """
    totals = {}
    games = {}
    prog = st.progress(0.0, text=f"Fetching ESPN box scores for {year} (weeks {weeks[0]}–{weeks[-1]})")
    all_events = []
    for i, wk in enumerate(weeks, 1):
        ids = list_week_event_ids(year, wk, seasontype)
        all_events.extend(ids)
        prog.progress(i/len(weeks), text=f"Week {wk}: {len(ids)} games")
        time.sleep(0.01)
    if not all_events:
        return pd.DataFrame(columns=["Player","g","pass_yds","pass_tds","rush_yds","rush_tds","rec","rec_tds"])

    for j, ev in enumerate(all_events, 1):
        box = fetch_boxscore_event(ev)
        if not box: 
            continue
        df = parse_boxscore_players(box)
        if df.empty:
            continue
        for _, r in df.iterrows():
            p = r["Player"]
            if p not in totals:
                totals[p] = {"pass_yds":0.0,"pass_tds":0.0,"rush_yds":0.0,"rush_tds":0.0,"rec":0.0,"rec_tds":0.0}
                games[p] = 0
            # increment games if player recorded any valid stat line in this event
            played = any(to_float(r[k])>0 for k in ["pass_yds","pass_tds","rush_yds","rush_tds","rec","rec_tds"] if k in r)
            if played:
                games[p] += 1
            for k in totals[p]:
                v = to_float(r.get(k, 0))
                if not np.isnan(v):
                    totals[p][k] += v
        if j % 10 == 0:
            st.caption(f"Parsed {j}/{len(all_events)} games...")

    rows = []
    for p, stat in totals.items():
        g = max(1, int(games.get(p, 0)))
        rows.append({"Player": p, "g": g, **stat})
    return pd.DataFrame(rows)

# ---------------- UI: season & weeks ----------------
st.header("1) Season scope & opponent defense")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    season = st.number_input("Season", min_value=2015, max_value=2100, value=2025, step=1)
with c2:
    seasontype = st.selectbox("Season type", options=[("Preseason",1),("Regular",2),("Postseason",3)], index=1, format_func=lambda x: x[0])[1]
with c3:
    if seasontype == 2:
        week_range = st.slider("Weeks (regular season)", 1, 18, (1, 18))
    elif seasontype == 1:
        week_range = st.slider("Weeks (preseason)", 0, 4, (0, 3))
    else:
        week_range = st.slider("Weeks (postseason)", 1, 5, (1, 5))
weeks = list(range(week_range[0], week_range[1]+1))

opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()

# ---------------- Build season projections from ESPN ----------------
st.header("2) Build per-player projections from ESPN box scores")
if st.button("Build 2025 projections from ESPN"):
    season_df = build_espn_season_agg(season, weeks, seasontype)
    if season_df.empty:
        st.error("No data returned from ESPN. Try adjusting weeks/season type.")
        st.stop()

    # Per-game means + SD heuristics
    season_df["mu_pass_yds"] = (season_df["pass_yds"] / season_df["g"]) * scalers["pass_adj"]
    season_df["mu_pass_tds"] = (season_df["pass_tds"] / season_df["g"]) * scalers["pass_adj"]
    season_df["mu_rush_yds"] = (season_df["rush_yds"] / season_df["g"]) * scalers["rush_adj"]
    season_df["mu_receptions"] = (season_df["rec"] / season_df["g"]) * scalers["recv_adj"]

    season_df["sd_pass_yds"] = season_df["mu_pass_yds"].apply(lambda x: np.nan if np.isnan(x) else max(25.0, 0.20*x))
    season_df["sd_pass_tds"] = season_df["mu_pass_tds"].apply(lambda x: np.nan if np.isnan(x) else max(0.25, 0.60*x))
    season_df["sd_rush_yds"] = season_df["mu_rush_yds"].apply(lambda x: np.nan if np.isnan(x) else max(6.0, 0.22*x))
    season_df["sd_receptions"] = season_df["mu_receptions"].apply(lambda x: np.nan if np.isnan(x) else max(1.0, 0.45*x))

    # Anytime TD lambdas
    season_df["lam_any_wr"] = (season_df["rec_tds"] / season_df["g"]) * scalers["recv_adj"]
    season_df["lam_any_rb"] = ((season_df["rec_tds"] + season_df["rush_tds"]) / season_df["g"]) * scalers["rush_adj"]

    # Split role tables (we don't have positions from scoreboard; treat everyone as potential)
    qb_proj = season_df[["Player","mu_pass_yds","sd_pass_yds","mu_pass_tds","sd_pass_tds"]].dropna(how="all")
    rb_proj = season_df[["Player","mu_rush_yds","sd_rush_yds","lam_any_rb"]].dropna(how="all")
    wr_proj = season_df[["Player","mu_receptions","sd_receptions","lam_any_wr"]].dropna(how="all")

    st.session_state["qb_proj"] = qb_proj
    st.session_state["rb_proj"] = rb_proj
    st.session_state["wr_proj"] = wr_proj

    c1, c2, c3 = st.columns(3)
    with c1: st.subheader("QB (derived)"); st.dataframe(qb_proj.head(12), use_container_width=True)
    with c2: st.subheader("RB (derived)"); st.dataframe(rb_proj.head(12), use_container_width=True)
    with c3: st.subheader("WR/TE (derived)"); st.dataframe(wr_proj.head(12), use_container_width=True)

# ---------------- Odds API ----------------
st.header("3) Choose an NFL game & markets from The Odds API")
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

# ---------------- Simulate ----------------
st.header("4) Fetch props for this event and simulate")
go = st.button("Fetch lines & simulate")
if go:
    qb_proj = st.session_state.get("qb_proj", pd.DataFrame())
    rb_proj = st.session_state.get("rb_proj", pd.DataFrame())
    wr_proj = st.session_state.get("wr_proj", pd.DataFrame())
    if qb_proj.empty and rb_proj.empty and wr_proj.empty:
        st.warning("Build the ESPN projections first (step 2).")
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
                name = normalize_name(o.get("description"))
                side = o.get("name")
                point = o.get("point")
                if mkey not in VALID_MARKETS or not name or not side:
                    continue
                rows.append({"market": mkey, "player_raw": name, "side": side, "point": (None if point is None else float(point))})
    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()
    props_df = pd.DataFrame(rows).drop_duplicates()

    out_rows = []
    qb_names = qb_proj["Player"].tolist()
    rb_names = rb_proj["Player"].tolist()
    wr_names = wr_proj["Player"].tolist()

    for _, r in props_df.iterrows():
        market = r["market"]; player = r["player_raw"]; point = r["point"]; side = r["side"]

        if market == "player_pass_tds" and not qb_proj.empty:
            match = fuzzy_pick(player, qb_names, cutoff=60)
            if not match: continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row.get("mu_pass_tds", np.nan)), float(row.get("sd_pass_tds", np.nan))
            if np.isnan(mu) or np.isnan(sd) or point is None: continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_pass_yds" and not qb_proj.empty:
            match = fuzzy_pick(player, qb_names, cutoff=60)
            if not match: continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row.get("mu_pass_yds", np.nan)), float(row.get("sd_pass_yds", np.nan))
            if np.isnan(mu) or np.isnan(sd) or point is None: continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_rush_yds" and not rb_proj.empty:
            match = fuzzy_pick(player, rb_names, cutoff=60)
            if not match: continue
            row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
            mu, sd = float(row.get("mu_rush_yds", np.nan)), float(row.get("sd_rush_yds", np.nan))
            if np.isnan(mu) or np.isnan(sd) or point is None: continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_receptions" and not wr_proj.empty:
            match = fuzzy_pick(player, wr_names, cutoff=60)
            if not match: continue
            row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
            mu, sd = float(row.get("mu_receptions", np.nan)), float(row.get("sd_receptions", np.nan))
            if np.isnan(mu) or np.isnan(sd) or point is None: continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_anytime_td":
            match = None; src = None
            m = fuzzy_pick(player, wr_names, cutoff=60)
            if m: match, src = m, "WR"
            if match is None:
                m = fuzzy_pick(player, rb_names, cutoff=60)
                if m: match, src = m, "RB"
            if match is None: continue
            if src == "WR":
                row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
                lam = float(row.get("lam_any_wr", np.nan))
            else:
                row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
                lam = float(row.get("lam_any_rb", np.nan))
            if np.isnan(lam): continue
            p_yes = anytime_yes_prob_poisson(lam)
            if side in ("Yes","No"):
                p = p_yes if side == "Yes" else (1.0 - p_yes)
            elif side in ("Over","Under"):
                p = p_yes if side == "Over" else (1.0 - p_yes)
            else:
                continue
            mu, sd = lam, float("nan")
            point = (0.5 if side in ("Over","Under") else None)

        else:
            continue

        out_rows.append({
            "market": market,
            "player": match,
            "side": side,
            "line": (None if point is None else round(float(point), 3)),
            "mu": (None if (isinstance(mu, float) and math.isnan(mu)) else round(float(mu), 3)),
            "sd": (None if (isinstance(sd, float) and math.isnan(sd)) else round(float(sd), 3)),
            "prob": round(100*p, 2),
            "opp_def": opp_team,
            "pass_adj": round(scalers["pass_adj"], 3),
            "rush_adj": round(scalers["rush_adj"], 3),
            "recv_adj": round(scalers["recv_adj"], 3),
        })

    if not out_rows:
        st.warning("No props matched your ESPN-derived projections.")
        st.stop()

    results = pd.DataFrame(out_rows).sort_values("prob", ascending=False).reset_index(drop=True)
    st.subheader("Simulated probabilities (Normal for yards/TDs; Poisson for Anytime TD)")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="props_sim_results.csv",
        mime="text/csv",
    )
