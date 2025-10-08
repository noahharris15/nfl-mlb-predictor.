# NFL Player Props — Odds API + ESPN (averages per game; defense-scaled)
# Run: streamlit run App.py   (or pages/0_NFL.py if multipage)

import re, math, unicodedata
from io import StringIO
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- UI / App ----------------
st.set_page_config(page_title="NFL Player Props — Odds API + ESPN", layout="wide")
st.title("📈 NFL Player Props — Odds API + ESPN (per-game averages + defense scaling)")

SIM_TRIALS = 10_000

# ✅ Markets you requested (keep these literal keys)
VALID_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    "player_pass_tds",
    "player_rush_reception_yds",
    "player_rush_attempts",
    "player_reception_yds",
    "player_pass_completions",
    "player_pass_attempts",
    "player_field_goals",
]

# ---------------- Embedded 2025 defense EPA (updated) ----------------
DEFENSE_EPA_2025 = """Team,EPA_Pass,EPA_Rush,Comp_Pct
Minnesota Vikings,-0.30,0.05,0.6522
Houston Texans,-0.19,-0.01,0.5882
Detroit Lions,-0.05,-0.18,0.6329
Denver Broncos,-0.07,-0.11,0.5814
Indianapolis Colts,-0.09,-0.06,0.6705
Jacksonville Jaguars,-0.12,0.04,0.6193
Atlanta Falcons,-0.13,0.05,0.5769
Philadelphia Eagles,-0.07,-0.02,0.5795
Los Angeles Chargers,-0.07,0.01,0.5909
San Francisco 49ers,0.02,-0.11,0.6706
Cleveland Browns,0.12,-0.22,0.6691
Los Angeles Rams,-0.02,-0.03,0.6667
Arizona Cardinals,0.06,-0.16,0.6184
New Orleans Saints,0.04,-0.06,0.6954
Kansas City Chiefs,-0.08,0.09,0.6781
Green Bay Packers,0.03,-0.07,0.6815
Chicago Bears,0.01,0.00,0.7368
Washington Commanders,0.06,-0.07,0.6410
Seattle Seahawks,0.12,-0.18,0.7027
Buffalo Bills,-0.02,0.07,0.6466
Tampa Bay Buccaneers,0.11,-0.10,0.6813
Pittsburgh Steelers,0.11,-0.05,0.6957
Carolina Panthers,0.09,-0.01,0.6536
New England Patriots,0.18,-0.14,0.7115
Las Vegas Raiders,0.19,-0.14,0.6564
New York Giants,0.04,0.10,0.6458
Tennessee Titans,0.13,0.02,0.7025
Cincinnati Bengals,0.18,0.01,0.6780
New York Jets,0.26,0.02,0.6500
Baltimore Ravens,0.21,0.12,0.6895
Dallas Cowboys,0.30,0.03,0.7238
Miami Dolphins,0.26,0.17,0.7445
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

    pass_adj = adj_from_epa(df["EPA_Pass"], 0.80)
    rush_adj = adj_from_epa(df["EPA_Rush"], 0.80)
    comp = df["Comp_Pct"].clip(0.45, 0.80).fillna(df["Comp_Pct"].mean())
    comp_adj = (1.0 + (comp - comp.mean()) * 0.60).clip(0.7, 1.3)
    recv_adj = (0.7 * pass_adj + 0.3 * comp_adj).clip(0.7, 1.3)

    out = pd.DataFrame({"Team": df["Team"], "pass_adj": pass_adj, "rush_adj": rush_adj, "recv_adj": recv_adj})
    return out

DEF_TABLE = load_defense_table()
st.caption("Defense multipliers (1.0 = neutral) are embedded from your 2025 EPA sheet.")

# ---------------- Utils ----------------
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

# ---------------- ESPN endpoints ----------------
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

# -------- Parse ESPN boxscore into per-game stats we need --------
def _extract_team_sections(box: dict) -> List[dict]:
    """Return list of team sections with labeled stat groups + athletes."""
    try:
        return box.get("boxscore", {}).get("players", []) or []
    except Exception:
        return []

def parse_boxscore_players(box: dict) -> pd.DataFrame:
    """
    Build one row per player with all metrics from this single game:
    - pass yards, tds, attempts, completions
    - rush yards, attempts
    - rec, receiving yards, receiving tds
    - field goals made
    """
    out = []
    for team in _extract_team_sections(box):
        for group in team.get("statistics", []):
            label = (group.get("name") or "").lower()
            for a in group.get("athletes", []):
                nm = normalize_name(a.get("athlete", {}).get("displayName"))
                vals = a.get("stats") or []

                # Passing section
                if "passing" in label:
                    # ESPN often uses "CMP-ATT" at index 0, YDS at 1, TD at 2
                    cmp_att = str(vals[0]) if len(vals) > 0 else ""
                    yds = float(vals[1]) if len(vals) > 1 and str(vals[1]).replace(".","",1).isdigit() else np.nan
                    tds = float(vals[2]) if len(vals) > 2 and str(vals[2]).replace(".","",1).isdigit() else np.nan
                    try:
                        cmp_, att_ = cmp_att.split("-")
                        cmp_ = float(cmp_)
                        att_ = float(att_)
                    except Exception:
                        cmp_, att_ = np.nan, np.nan
                    out.append({
                        "Player": nm, "pass_yds": yds, "pass_tds": tds,
                        "pass_cmp": cmp_, "pass_att": att_,
                        "rush_yds": 0.0, "rush_att": 0.0,
                        "rec": 0.0, "rec_yds": 0.0,
                        "fgm": 0.0,
                    })

                # Rushing section
                elif "rushing" in label:
                    # CAR at 0, YDS at 1, TD at 2
                    try:
                        carr = float(vals[0]) if len(vals) > 0 else np.nan
                    except Exception:
                        carr = np.nan
                    yds = float(vals[1]) if len(vals) > 1 and str(vals[1]).replace(".","",1).isdigit() else np.nan
                    out.append({
                        "Player": nm, "pass_yds": 0.0, "pass_tds": 0.0,
                        "pass_cmp": 0.0, "pass_att": 0.0,
                        "rush_yds": yds, "rush_att": carr,
                        "rec": 0.0, "rec_yds": 0.0,
                        "fgm": 0.0,
                    })

                # Receiving section
                elif "receiving" in label:
                    # REC at 0, YDS at 1, TD at 2
                    recs = float(vals[0]) if len(vals) > 0 and str(vals[0]).replace(".","",1).isdigit() else np.nan
                    ryds = float(vals[1]) if len(vals) > 1 and str(vals[1]).replace(".","",1).isdigit() else np.nan
                    out.append({
                        "Player": nm, "pass_yds": 0.0, "pass_tds": 0.0,
                        "pass_cmp": 0.0, "pass_att": 0.0,
                        "rush_yds": 0.0, "rush_att": 0.0,
                        "rec": recs, "rec_yds": ryds,
                        "fgm": 0.0,
                    })

                # Kicking (field goals made)
                elif "kicking" in label:
                    # FG at index 0 like "1-2" (made-attempts)
                    fg = str(vals[0]) if len(vals) > 0 else ""
                    try:
                        made, _att = fg.split("-")
                        made = float(made)
                    except Exception:
                        made = 0.0
                    out.append({
                        "Player": nm, "pass_yds": 0.0, "pass_tds": 0.0,
                        "pass_cmp": 0.0, "pass_att": 0.0,
                        "rush_yds": 0.0, "rush_att": 0.0,
                        "rec": 0.0, "rec_yds": 0.0,
                        "fgm": made,
                    })

    if not out:
        return pd.DataFrame(columns=["Player","pass_yds","pass_tds","pass_cmp","pass_att","rush_yds","rush_att","rec","rec_yds","fgm"])
    # Sum within a game per player (sometimes players appear in multiple stat groups)
    gdf = pd.DataFrame(out).groupby("Player", as_index=False).sum(numeric_only=True)
    return gdf

# -------- Crawl an entire season-to-date and compute per-game averages --------
@st.cache_data(show_spinner=True)
def build_espn_season_agg(year: int, weeks: List[int], seasontype: int) -> pd.DataFrame:
    events = []
    for wk in weeks:
        events.extend(list_week_event_ids(year, wk, seasontype))
    if not events:
        return pd.DataFrame()

    totals, sumsqs, games = {}, {}, {}

    def init(p):
        if p not in totals:
            totals[p] = {"pass_yds":0.0,"pass_tds":0.0,"pass_cmp":0.0,"pass_att":0.0,
                         "rush_yds":0.0,"rush_att":0.0,"rec":0.0,"rec_yds":0.0,"fgm":0.0}
            sumsqs[p] = {"pass_yds":0.0,"rush_yds":0.0,"rec":0.0,"rec_yds":0.0,
                         "pass_cmp":0.0,"pass_att":0.0,"rush_att":0.0}
            games[p]  = 0

    prog = st.progress(0.0, text=f"Crawling {len(events)} NFL games from ESPN…")
    for j, ev in enumerate(events, 1):
        box = fetch_boxscore_event(ev)
        if box:
            one = parse_boxscore_players(box)
            for _, r in one.iterrows():
                p = r["Player"]; init(p)
                played = any(float(r[k]) > 0 for k in totals[p].keys())
                if played: games[p] += 1
                # sums
                for k in totals[p]:
                    v = float(r.get(k, 0.0)) if not pd.isna(r.get(k, np.nan)) else 0.0
                    totals[p][k] += v
                # sums of squares for SD (continuous / counts we model with t/normal)
                for k in ["pass_yds","rush_yds","rec","rec_yds","pass_cmp","pass_att","rush_att"]:
                    v = float(r.get(k, 0.0)) if not pd.isna(r.get(k, np.nan)) else 0.0
                    sumsqs[p][k] += v*v
        prog.progress(j/len(events))

    rows = []
    for p, stat in totals.items():
        g = max(1, int(games.get(p, 0)))
        rows.append({"Player": p, "g": g, **stat,
                     **{f"sq_{k}": sumsqs[p][k] for k in sumsqs[p]}})
    return pd.DataFrame(rows)

def sample_sd(sum_x: float, sum_x2: float, g: int, floor: float) -> float:
    g = int(g)
    if g <= 1: return float("nan")
    mean = sum_x / g
    var  = (sum_x2 / g) - (mean**2)
    var  = var * (g / (g - 1))
    return float(max(floor, math.sqrt(max(var, 1e-6))))

def clamp(x, lo=0.85, hi=1.15) -> float:
    return float(np.clip(x, lo, hi))

# ---------------- Step 1: season scope + defense ----------------
st.markdown("### 1) Season scope & opponent defense")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    season = st.number_input("Season", min_value=2015, max_value=2100, value=2025, step=1)
with c2:
    seasontype = st.selectbox("Season type", options=[("Preseason",1),("Regular",2),("Postseason",3)],
                              index=1, format_func=lambda x: x[0])[1]
with c3:
    week_range = st.slider("Weeks", 1, 18, (1, 18)) if seasontype==2 else (
                 st.slider("Preseason Weeks", 0, 4, (0,3)) if seasontype==1 else
                 st.slider("Postseason Weeks", 1, 5, (1,5)))
weeks = list(range(week_range[0], week_range[1] + 1))

opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()  # pass_adj, rush_adj, recv_adj

# ---------------- Step 2: build player projections (per-game means) ----------------
st.markdown("### 2) Build per-player projections from ESPN (uses averages for games played)")
if st.button("📥 Build NFL projections"):
    season_df = build_espn_season_agg(season, weeks, seasontype)
    if season_df.empty:
        st.error("No data returned from ESPN."); st.stop()

    g = season_df["g"].clip(lower=1)

    # Per-game means (NO shrinkage — exactly as requested)
    season_df["mu_pass_yds"]        = season_df["pass_yds"] / g
    season_df["mu_pass_tds"]        = season_df["pass_tds"] / g
    season_df["mu_pass_completions"]= season_df["pass_cmp"] / g
    season_df["mu_pass_attempts"]   = season_df["pass_att"] / g

    season_df["mu_rush_yds"]        = season_df["rush_yds"] / g
    season_df["mu_rush_attempts"]   = season_df["rush_att"] / g

    season_df["mu_receptions"]      = season_df["rec"] / g
    season_df["mu_rec_yds"]         = season_df["rec_yds"] / g

    season_df["mu_fg_made"]         = season_df["fgm"] / g

    # Sample SDs (Bessel) with small floors
    season_df["sd_pass_yds"]        = season_df.apply(lambda r: sample_sd(r["pass_yds"], r["sq_pass_yds"], r["g"], 30.0), axis=1)
    season_df["sd_rush_yds"]        = season_df.apply(lambda r: sample_sd(r["rush_yds"], r["sq_rush_yds"], r["g"], 15.0), axis=1)
    season_df["sd_receptions"]      = season_df.apply(lambda r: sample_sd(r["rec"],      r["sq_rec"],      r["g"], 1.2), axis=1)
    season_df["sd_rec_yds"]         = season_df.apply(lambda r: sample_sd(r["rec_yds"],  r["sq_rec_yds"],  r["g"], 15.0), axis=1)
    season_df["sd_pass_completions"]= season_df.apply(lambda r: sample_sd(r["pass_cmp"],  r["sq_pass_cmp"], r["g"], 2.0), axis=1)
    season_df["sd_pass_attempts"]   = season_df.apply(lambda r: sample_sd(r["pass_att"],  r["sq_pass_att"], r["g"], 2.5), axis=1)
    season_df["sd_rush_attempts"]   = season_df.apply(lambda r: sample_sd(r["rush_att"],  r["sq_rush_att"], r["g"], 2.0), axis=1)

    # Composite (rush + rec yards) mean & sd (assume weak correlation -> sum of vars)
    season_df["mu_rush_rec_yds"] = season_df["mu_rush_yds"] + season_df["mu_rec_yds"]
    season_df["sd_rush_rec_yds"] = np.sqrt(season_df["sd_rush_yds"]**2 + season_df["sd_rec_yds"]**2)

    # Apply defense scaling to MEANS only
    season_df["mu_pass_yds"]        *= clamp(scalers["pass_adj"])
    season_df["mu_pass_tds"]        *= clamp(scalers["pass_adj"])
    season_df["mu_pass_completions"]*= clamp(scalers["pass_adj"])
    season_df["mu_pass_attempts"]   *= clamp(scalers["pass_adj"])

    season_df["mu_rush_yds"]        *= clamp(scalers["rush_adj"])
    season_df["mu_rush_attempts"]   *= clamp(scalers["rush_adj"])

    season_df["mu_receptions"]      *= clamp(scalers["recv_adj"])
    season_df["mu_rec_yds"]         *= clamp(scalers["recv_adj"])
    season_df["mu_rush_rec_yds"]    *= clamp(0.5*scalers["rush_adj"] + 0.5*scalers["recv_adj"])

    # Slim tables stored in session
    keep_cols = [
        "Player","g",
        "mu_pass_yds","sd_pass_yds","mu_pass_tds",
        "mu_pass_completions","sd_pass_completions",
        "mu_pass_attempts","sd_pass_attempts",
        "mu_rush_yds","sd_rush_yds","mu_rush_attempts","sd_rush_attempts",
        "mu_receptions","sd_receptions","mu_rec_yds","sd_rec_yds",
        "mu_rush_rec_yds","sd_rush_rec_yds",
        "mu_fg_made"
    ]
    st.session_state["proj"] = season_df[keep_cols].copy()

    st.success("Built projections from ESPN.")
    st.dataframe(st.session_state["proj"].head(18), use_container_width=True)

# ---------------- Odds API (events + props) ----------------
st.markdown("### 3) Pick a game & markets (Odds API)")
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

# ---------------- Simulate with per-game averages ----------------
st.markdown("### 4) Fetch props for this event and simulate (10k draws)")
go = st.button("🎲 Fetch lines & simulate")

if go:
    proj = st.session_state.get("proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build NFL projections first (Step 2)."); st.stop()

    # normalized set of player names for fast lookups
    proj = proj.copy()
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    idx = proj.set_index("player_norm")

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
                rows.append({
                    "market": mkey,
                    "player_norm": name,
                    "side": side,
                    "point": (None if point is None else float(point)),
                })

    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    raw_props = pd.DataFrame(rows)
    props_df = (raw_props.groupby(["market","player_norm","side"], as_index=False)
                        .agg(line=("point","median"), n_books=("point","size")))

    out_rows = []
    for _, r in props_df.iterrows():
        mkt    = r["market"]
        name   = r["player_norm"]
        side   = r["side"]
        line   = r["line"]

        if name not in idx.index or pd.isna(line):
            continue

        row = idx.loc[name]

        # Map each market to (mu, sd) from our per-game averages
        mu = sd = None

        if mkt == "player_pass_tds":
            lam = float(row["mu_pass_tds"])
            # Poisson → probability Over(line)
            p_over = float((np.random.poisson(lam=max(1e-6,lam), size=SIM_TRIALS) > float(line)).mean())
            p = p_over if side == "Over" else 1.0 - p_over
            mu_display, sd_display = lam, None

        else:
            if mkt == "player_pass_yds":
                mu, sd = float(row["mu_pass_yds"]), float(row["sd_pass_yds"])
            elif mkt == "player_rush_yds":
                mu, sd = float(row["mu_rush_yds"]), float(row["sd_rush_yds"])
            elif mkt == "player_receptions":
                mu, sd = float(row["mu_receptions"]), float(row["sd_receptions"])
            elif mkt == "player_reception_yds":
                mu, sd = float(row["mu_rec_yds"]), float(row["sd_rec_yds"])
            elif mkt == "player_rush_attempts":
                mu, sd = float(row["mu_rush_attempts"]), float(row["sd_rush_attempts"])
            elif mkt == "player_rush_reception_yds":
                mu, sd = float(row["mu_rush_rec_yds"]), float(row["sd_rush_rec_yds"])
            elif mkt == "player_pass_completions":
                mu, sd = float(row["mu_pass_completions"]), float(row["sd_pass_completions"])
            elif mkt == "player_pass_attempts":
                mu, sd = float(row["mu_pass_attempts"]), float(row["sd_pass_attempts"])
            elif mkt == "player_field_goals":
                # treat FGs as Poisson mean; simulate Poisson
                lam = float(row["mu_fg_made"])
                p_over = float((np.random.poisson(lam=max(1e-6,lam), size=SIM_TRIALS) > float(line)).mean())
                p = p_over if side == "Over" else 1.0 - p_over
                mu_display, sd_display = lam, None
                out_rows.append({
                    "market": mkt, "player": row["Player"], "side": side,
                    "line": round(float(line),2),
                    "μ (per-game)": round(mu_display,2),
                    "σ (per-game)": None,
                    "Win Prob %": round(100*p,2),
                    "#Books": int(r["n_books"]),
                })
                continue
            else:
                continue

            if np.isnan(mu) or np.isnan(sd):
                continue

            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over
            mu_display, sd_display = mu, sd

        out_rows.append({
            "market": mkt, "player": row["Player"], "side": side,
            "line": round(float(line), 2),
            "μ (per-game)": None if mu_display is None else round(float(mu_display), 2),
            "σ (per-game)": None if (sd_display is None or (isinstance(sd_display, float) and np.isnan(sd_display))) else round(float(sd_display), 2),
            "Win Prob %": round(100*p, 2),
            "#Books": int(r["n_books"]),
        })

    if not out_rows:
        st.warning("No props matched projections."); st.stop()

    results = (pd.DataFrame(out_rows)
               .drop_duplicates(subset=["market","player","side"])
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    # ---- Display
    st.subheader("Results")
    colcfg = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.2f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "#Books": st.column_config.NumberColumn("#Books", width="small"),
    }
    st.dataframe(results, use_container_width=True, hide_index=True, column_config=colcfg)

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="nfl_props_results.csv",
        mime="text/csv",
    )
