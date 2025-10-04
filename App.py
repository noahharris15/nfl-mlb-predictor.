import time, math, json, gzip, io
from typing import Dict, Any, List, Tuple
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PrizePicks Simulator — Real Baselines", page_icon="📈", layout="wide")

# -------------------------------
# Config / constants
# -------------------------------
PP_ENDPOINTS = [
    # 1) "site" endpoint (works from Streamlit Cloud most of the time)
    ("https://site.api.prizepicks.io/api/v1/projections", {"per_page": 250, "single_stat": "true"}),
    # 2) classic endpoint (sometimes blocked; we keep it as a fallback)
    ("https://api.prizepicks.com/projections", {"per_page": 250, "single_stat": "true"}),
]

LEAGUES = ["NFL", "NBA"]  # add more once we wire baselines

# Market name normalization -> canonical market key
MARKET_MAP = {
    # NBA
    "Points": "points", "Rebounds": "rebounds", "Assists": "assists",
    "Pts+Rebs+Asts": "pra", "Rebs+Asts": "ra", "3-PT Made": "threes",
    "Fantasy Score": "fantasy",

    # NFL
    "Pass Yds": "pass_yards", "Rush Yds": "rush_yards", "Rec Yds": "rec_yards",
    "Receptions": "receptions", "Rush+Rec Yds": "rush_rec_yards",
    "Pass+Rush+Rec Yds": "prr_yards", "Pass TDs": "pass_tds",
}

# Per-market default SD as % of baseline (used if we can’t derive per-player variance)
DEFAULT_SD_PCT = {
    # NBA
    "points": 0.22, "rebounds": 0.30, "assists": 0.32, "pra": 0.20, "ra": 0.26,
    "threes": 0.55, "fantasy": 0.18,

    # NFL
    "pass_yards": 0.16, "rush_yards": 0.28, "rec_yards": 0.25, "receptions": 0.35,
    "rush_rec_yards": 0.22, "prr_yards": 0.18, "pass_tds": 0.65,
}
FLOOR_SD = 0.35  # don’t allow super tiny SDs

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.prizepicks.com/",
}

# -------------------------------
# Utility
# -------------------------------
def normal_cdf(x, mu, sigma):
    if sigma <= 0:  # safety
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

@st.cache_data(ttl=120)
def fetch_prizepicks(league: str) -> Dict[str, Any] | None:
    """Try 2 endpoints w/ backoff, return JSON or None."""
    params_extra = {"league": league.lower()}
    backoff = [1, 2, 4, 8]
    for base_url, base_params in PP_ENDPOINTS:
        params = base_params.copy()
        params.update(params_extra)
        for i, wait in enumerate([0]+backoff):
            if wait:
                st.info(f"⏳ Network hiccup: retrying in {wait}s…")
                time.sleep(wait)
            try:
                r = requests.get(base_url, params=params, headers=HEADERS, timeout=20)
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    return r.json()
                if r.status_code in (403, 429, 520, 522, 525):
                    # try next backoff cycle / endpoint
                    continue
            except requests.RequestException:
                continue
    return None

def parse_pp(json_payload: Dict[str, Any], league: str) -> pd.DataFrame:
    """Strictly keep rows where attributes.league == league."""
    if not json_payload:
        return pd.DataFrame()

    data = json_payload.get("data", [])
    included = json_payload.get("included", [])

    # players map
    players: Dict[str, Tuple[str, str]] = {}
    for inc in included:
        if inc.get("type") in ("players", "Player", "player"):
            pid = inc.get("id")
            attr = inc.get("attributes", {})
            name = attr.get("name") or attr.get("full_name") or attr.get("display_name") or ""
            team = attr.get("team") or attr.get("injury_team") or attr.get("league_team") or ""
            if pid:
                players[str(pid)] = (name, team)

    rows = []
    for item in data:
        if item.get("type") not in ("projection", "projections"): 
            continue
        attr = item.get("attributes", {})
        lg = (attr.get("league") or "").upper()
        if lg != league.upper():
            continue

        try:
            line = float(attr.get("line_score"))
        except (TypeError, ValueError):
            continue

        market_pp = attr.get("market_type") or attr.get("display_stat") or attr.get("stat_type") or ""
        market = MARKET_MAP.get(market_pp, market_pp.strip().lower())
        pid = str(attr.get("player_id") or "")

        name, team = players.get(pid, ("", ""))

        rows.append({
            "league": lg, "player": name, "team": team,
            "market_pp": market_pp, "market": market,
            "line": line, "player_id": pid
        })

    df = pd.DataFrame(rows)
    df = df[(df["player"].notna()) & (df["player"] != "")]
    return df.reset_index(drop=True)

# -------------------------------
# NBA baselines (balldontlie)
# -------------------------------
def nba_search_ids(names: List[str]) -> Dict[str, int]:
    """Search balldontlie by name -> id (crude but works well enough)."""
    ids: Dict[str, int] = {}
    for nm in sorted(set(names)):
        try:
            r = requests.get("https://www.balldontlie.io/api/v1/players", params={"search": nm, "per_page": 5}, timeout=12)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    # naive pick: best full-name match, else first
                    best = sorted(data, key=lambda x: (x.get("first_name","")+" "+x.get("last_name","")!=nm))
                    ids[nm] = best[0]["id"]
        except requests.RequestException:
            continue
    return ids

def nba_season_averages(year: int, ids: List[int]) -> pd.DataFrame:
    """Batch up to ~100 ids per request."""
    if not ids:
        return pd.DataFrame()
    all_rows = []
    batch = 75
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        params = [("season", year)] + [("player_ids[]", pid) for pid in chunk]
        try:
            r = requests.get("https://www.balldontlie.io/api/v1/season_averages", params=params, timeout=15)
            if r.status_code == 200:
                rows = r.json().get("data", [])
                all_rows.extend(rows)
        except requests.RequestException:
            continue
    df = pd.DataFrame(all_rows)
    # keep the main stats we need
    cols = {
        "player_id": "balldontlie_id",
        "pts": "points", "reb": "rebounds", "ast": "assists",
        "fg3m": "threes",
        # fantasy/pra etc can be built from these if needed
    }
    df = df.rename(columns=cols)
    return df

def build_nba_baselines(df_pp: pd.DataFrame, season: int) -> pd.DataFrame:
    names = df_pp["player"].dropna().unique().tolist()
    id_map = nba_search_ids(names)
    if not id_map:
        return pd.DataFrame()

    # attach ids to PP table
    df_pp2 = df_pp.copy()
    df_pp2["balldontlie_id"] = df_pp2["player"].map(id_map).astype("Int64")

    # pull season averages
    id_list = [i for i in df_pp2["balldontlie_id"].dropna().astype(int).unique().tolist()]
    df_avg = nba_season_averages(season, id_list)
    if df_avg.empty:
        return pd.DataFrame()

    # merge to PP rows
    merged = df_pp2.merge(df_avg, on="balldontlie_id", how="left")

    # produce baseline per row based on market
    def baseline_row(r):
        m = r["market"]
        if m == "points": return r.get("points", np.nan)
        if m == "rebounds": return r.get("rebounds", np.nan)
        if m == "assists": return r.get("assists", np.nan)
        if m == "threes": return r.get("threes", np.nan)
        if m == "pra":
            return (r.get("points", np.nan) or 0) + (r.get("rebounds", np.nan) or 0) + (r.get("assists", np.nan) or 0)
        if m == "ra":
            return (r.get("rebounds", np.nan) or 0) + (r.get("assists", np.nan) or 0)
        # fantasy not in balldontlie; use simple proxy
        if m == "fantasy":
            # very rough FanDuel-like: Pts + 1.2*Reb + 1.5*Ast
            return (r.get("points", 0) or 0) + 1.2*(r.get("rebounds", 0) or 0) + 1.5*(r.get("assists", 0) or 0)
        return np.nan

    merged["baseline"] = merged.apply(baseline_row, axis=1)
    return merged

# -------------------------------
# NFL baselines (nflfastR season stats)
# -------------------------------
@st.cache_data(ttl=6*3600)
def load_nflfastR_player_stats(season: int) -> pd.DataFrame:
    # Official mirrors host a gz CSV per season; no key needed
    url = f"https://github.com/nflverse/nflfastR-data/releases/download/player_stats/player_stats_{season}.csv.gz"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    df = pd.read_csv(buf, low_memory=False)
    return df

def build_nfl_baselines(df_pp: pd.DataFrame, season: int) -> pd.DataFrame:
    df_stats = load_nflfastR_player_stats(season)
    # build simple per-game averages by stat group
    # columns vary; we map a few common items
    keep = [
        "player_display_name", "recent_team", "season", "games",
        "passing_yards", "passing_tds",
        "rushing_yards", "rushing_tds",
        "receiving_yards", "receptions",
    ]
    df_stats = df_stats[[c for c in keep if c in df_stats.columns]].copy()
    df_stats = df_stats[df_stats["season"] == season]
    df_stats["games"] = df_stats["games"].replace(0, np.nan)

    # per-game
    for src, out in [
        ("passing_yards", "avg_pass_yards"),
        ("passing_tds", "avg_pass_tds"),
        ("rushing_yards", "avg_rush_yards"),
        ("receiving_yards", "avg_rec_yards"),
        ("receptions", "avg_receptions"),
    ]:
        if src in df_stats.columns:
            df_stats[out] = df_stats[src] / df_stats["games"]

    # Merge to PP rows by display name (simple exact match; can add fuzzy if needed)
    df_pp2 = df_pp.copy()
    merged = df_pp2.merge(
        df_stats, left_on="player", right_on="player_display_name", how="left", suffixes=("","_nfl")
    )

    def baseline_row(r):
        m = r["market"]
        if m == "pass_yards": return r.get("avg_pass_yards", np.nan)
        if m == "pass_tds": return r.get("avg_pass_tds", np.nan)
        if m == "rush_yards": return r.get("avg_rush_yards", np.nan)
        if m == "rec_yards": return r.get("avg_rec_yards", np.nan)
        if m == "receptions": return r.get("avg_receptions", np.nan)
        if m == "rush_rec_yards":
            a = (r.get("avg_rush_yards", np.nan) or 0) + (r.get("avg_rec_yards", np.nan) or 0)
            return a
        if m == "prr_yards":
            a = (r.get("avg_pass_yards", np.nan) or 0) + (r.get("avg_rush_yards", np.nan) or 0) + (r.get("avg_rec_yards", np.nan) or 0)
            return a
        return np.nan

    merged["baseline"] = merged.apply(baseline_row, axis=1)
    return merged

# -------------------------------
# Simulation (normal model)
# -------------------------------
def pick_sd(market: str, baseline: float) -> float:
    pct = DEFAULT_SD_PCT.get(market, 0.30)
    if baseline is None or pd.isna(baseline):
        # if baseline missing, use modest line-based SD later
        return np.nan
    return max(FLOOR_SD, pct * abs(float(baseline)))

def simulate(df: pd.DataFrame) -> pd.DataFrame:
    mu = df["baseline"].astype(float)
    sd = df["sd_est"].astype(float)

    # if baseline missing, use line as mean and default sd on line
    mask_missing = mu.isna()
    mu.loc[mask_missing] = df.loc[mask_missing, "line"].astype(float)
    sd.loc[mask_missing] = df.loc[mask_missing].apply(
        lambda r: max(FLOOR_SD, DEFAULT_SD_PCT.get(r["market"], 0.30) * abs(float(r["line"]))), axis=1
    )

    p_under = [normal_cdf(x, m, s) for x, m, s in zip(df["line"], mu, sd)]
    p_over = [1.0 - p for p in p_under]

    # “edge” is how sensitive prob is to 0.5*sd shift in mean
    shift = 0.5 * sd
    p_over_up = [1.0 - normal_cdf(x, m + d, s) for x, m, s, d in zip(df["line"], mu, sd, shift)]
    p_over_dn = [1.0 - normal_cdf(x, m - d, s) for x, m, s, d in zip(df["line"], mu, sd, shift)]
    edge = np.abs(np.array(p_over_up) - np.array(p_over_dn)) * 0.5

    out = df.copy()
    out["baseline_used"] = mu
    out["model_sd"] = sd
    out["P(over)"] = p_over
    out["P(under)"] = p_under
    out["edge"] = edge
    return out

# -------------------------------
# UI
# -------------------------------
st.title("📈 PrizePicks Simulator — Real Baselines (NFL + NBA)")
st.caption("Live board → real player baselines → quick simulation. No manual CSV uploads.")

col1, col2 = st.columns([1,1])
with col1:
    league = st.selectbox("League", LEAGUES, index=0)
with col2:
    season = st.number_input("Season", min_value=2018, max_value=2030, value=2024, step=1)

# Pull board
payload = fetch_prizepicks(league)
if not payload:
    st.error("Could not reach PrizePicks endpoints right now (DNS / rate limits). Give it ~30–60s and rerun.")
    st.stop()

df_pp = parse_pp(payload, league)
if df_pp.empty:
    st.warning("No markets parsed for this league right now.")
    st.stop()

st.success(f"Loaded {len(df_pp)} lines for {league}.")

# Build baselines
if league == "NBA":
    with st.spinner("Building NBA season-average baselines (balldontlie)…"):
        df_base = build_nba_baselines(df_pp, season)
elif league == "NFL":
    with st.spinner("Building NFL season-average baselines (nflfastR)…"):
        df_base = build_nfl_baselines(df_pp, season)
else:
    df_base = pd.DataFrame()

if df_base.empty:
    st.error("Baseline build returned empty (API quiet / name mismatches). We’ll still simulate with line-as-mean.")
    df_base = df_pp.copy()
    df_base["baseline"] = np.nan

# SD estimate (market defaults based on baseline)
df_base["sd_est"] = df_base.apply(lambda r: pick_sd(r["market"], r["baseline"]), axis=1)

# Run sim
df_sim = simulate(df_base)

# Present
st.subheader("Simulated probabilities")
edge_min = st.slider("Minimum edge (percent)", 0.0, 15.0, 2.0, 0.5)

view = df_sim.copy()
view = view.sort_values("edge", ascending=False)
view = view[view["edge"] >= (edge_min/100.0)]

def pct(x): return f"{x*100:0.1f}%"

show = view[[
    "player","team","market","market_pp","line","baseline_used","model_sd","P(over)","P(under)","edge"
]].copy()
show["line"] = show["line"].round(2)
show["baseline_used"] = show["baseline_used"].round(2)
show["model_sd"] = show["model_sd"].round(3)
show["P(over)"] = show["P(over)"].map(pct)
show["P(under)"] = show["P(under)"].map(pct)
show["edge"] = show["edge"].map(pct)

st.dataframe(show.reset_index(drop=True), use_container_width=True)

csv = show.to_csv(index=False)
st.download_button("Download table as CSV", data=csv, file_name=f"{league.lower()}_{season}_sim.csv", mime="text/csv")

with st.expander("Notes"):
    st.markdown(
        "- **NBA baselines**: season averages from balldontlie (free public API). "
        "We batch player IDs to avoid rate limits.\n"
        "- **NFL baselines**: per-season player stats from nflfastR (public GitHub release). "
        "We convert to per-game averages.\n"
        "- **SDs**: market-specific defaults (% of baseline) with a small floor. "
        "If a player baseline is missing, we fall back to **line-as-mean** for that row.\n"
        "- **PrizePicks fetch**: two endpoints + backoff. If you see DNS/429/403, wait 30–60s and rerun.\n"
        "- Want **MLB/NHL** added? Say the word — both have solid public stat APIs we can wire in the same way."
    )
