# Player Props Simulator — Odds API + nfl_data_py (no CSVs; with GitHub fallback)
# Run: streamlit run app.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO
from typing import List, Optional, Dict, Tuple
from rapidfuzz import process, fuzz
import nfl_data_py as nfl  # pip install nfl_data_py

st.set_page_config(page_title="NFL Player Props — Odds API + nfl_data_py", layout="wide")
st.title("📈 NFL Player Props — Odds API + nfl_data_py (defense EPA embedded)")

SIM_TRIALS = 10_000
VALID_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    "player_anytime_td",
    "player_pass_tds",
]

# ---------------- Embedded 2025 defense EPA (your table) ----------------
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

# ---------------- Helpers ----------------
def fuzzy_pick(name: str, candidates: List[str], cutoff=85) -> Optional[str]:
    if not candidates: return None
    res = process.extractOne(name, candidates, scorer=fuzz.token_sort_ratio)
    return res[0] if res and res[1] >= cutoff else None

def norm_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    return float((np.random.normal(mu, sd, size=trials) > line).mean())

def anytime_yes_prob_poisson(lam: float) -> float:
    lam = max(1e-6, float(lam))
    return 1.0 - math.exp(-lam)

# ---------------- nfl_data_py loaders ----------------
st.header("1) Pull season/weekly stats from nfl_data_py (no CSVs)")

c1, c2 = st.columns([1,1])
with c1:
    season = st.number_input("Season", min_value=2010, max_value=2100, value=2025, step=1)
with c2:
    week_max = st.number_input("Use games through week (inclusive)", min_value=1, max_value=22, value=5, step=1)

@st.cache_data(show_spinner=False)
def load_weekly_player_stats(season: int, week_max: int) -> pd.DataFrame:
    """
    Try nfl_data_py (works locally). If blocked (e.g., Streamlit Cloud),
    fall back to NFLVerse GitHub raw CSV (no uploads needed).
    """
    # ---- Attempt 1: nfl_data_py (local-friendly) ----
    try:
        df = nfl.import_weekly_data([season])
        df = df[df["week"].astype(int) <= int(week_max)].copy()
        st.info("Loaded weekly data via nfl_data_py.")
    except Exception as e:
        # ---- Attempt 2: GitHub raw CSV fallback (cloud-friendly) ----
        st.warning(f"nfl_data_py failed: {e}. Falling back to NFLVerse GitHub CSV.")
        urls = [
            f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/player_stats/player_stats_{season}.csv.gz",
            f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/player_stats/player_stats_{season}.csv",
        ]
        df = None
        last_err = None
        for url in urls:
            try:
                df_try = pd.read_csv(url, low_memory=False)
                df = df_try
                st.info(f"Loaded weekly data from {url.split('/')[-1]}")
                break
            except Exception as ee:
                last_err = ee
                continue
        if df is None:
            raise RuntimeError(f"Could not fetch weekly data from GitHub. Last error: {last_err}")

        # player_stats_{season} has all weeks; filter to <= week_max
        if "week" in df.columns:
            df = df[df["week"].astype(int) <= int(week_max)].copy()
        else:
            # some mirrors label week as 'game_week' — handle defensively
            if "game_week" in df.columns:
                df["week"] = df["game_week"]
                df = df[df["week"].astype(int) <= int(week_max)].copy()
            else:
                st.warning("No 'week' column found; using full season totals.")

    # ---- Standardize columns we need ----
    need = [
        "season","week","player_name","position","recent_team",
        "passing_yards","passing_tds",
        "rushing_yards","rushing_tds",
        "receptions","receiving_tds",
    ]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan

    return df[need].copy()

weekly = load_weekly_player_stats(season, week_max)

st.header("2) Choose opponent defense to apply")
opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()  # pass_adj, rush_adj, recv_adj

@st.cache_data(show_spinner=False)
def build_player_projections(weekly: pd.DataFrame, scalers: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if weekly.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    w = weekly.copy()
    w["player_name"] = w["player_name"].astype(str).str.strip()
    w["position"] = w["position"].astype(str).str.upper().str.strip()

    gp = w.groupby("player_name", dropna=False)["week"].nunique().rename("g").to_frame()

    agg = w.groupby(["player_name","position"], dropna=False).agg(
        pass_yds=("passing_yards","sum"),
        pass_tds=("passing_tds","sum"),
        rush_yds=("rushing_yards","sum"),
        rush_tds=("rushing_tds","sum"),
        recs=("receptions","sum"),
        rec_tds=("receiving_tds","sum"),
    ).reset_index()

    agg = agg.merge(gp, left_on="player_name", right_index=True, how="left")
    agg["g"] = agg["g"].clip(lower=1)

    agg["mu_pass_yds"] = (agg["pass_yds"] / agg["g"]) * scalers["pass_adj"]
    agg["mu_pass_tds"] = (agg["pass_tds"] / agg["g"]) * scalers["pass_adj"]
    agg["mu_rush_yds"] = (agg["rush_yds"] / agg["g"]) * scalers["rush_adj"]
    agg["mu_receptions"] = (agg["recs"] / agg["g"]) * scalers["recv_adj"]

    # Poisson λ for Anytime TD:
    agg["lam_any_rb"] = ((agg["rush_tds"] + agg["rec_tds"]) / agg["g"]) * scalers["rush_adj"]
    agg["lam_any_wr"] = (agg["rec_tds"] / agg["g"]) * scalers["recv_adj"]

    qb = agg[agg["position"] == "QB"].copy()
    rb = agg[agg["position"] == "RB"].copy()
    wr = agg[agg["position"].isin(["WR","TE"])].copy()

    # ---- SD heuristics ----
    qb["sd_pass_yds"] = (qb["mu_pass_yds"] * 0.20).clip(lower=25.0)
    qb["sd_pass_tds"] = (qb["mu_pass_tds"] * 0.60).clip(lower=0.25)
    rb["sd_rush_yds"] = (rb["mu_rush_yds"] * 0.22).clip(lower=6.0)
    wr["sd_receptions"] = (wr["mu_receptions"] * 0.45).clip(lower=1.0)

    qb.rename(columns={"player_name":"Player"}, inplace=True)
    rb.rename(columns={"player_name":"Player"}, inplace=True)
    wr.rename(columns={"player_name":"Player"}, inplace=True)

    for df in (qb, rb, wr):
        df["Player"] = df["Player"].astype(str)

    return qb, rb, wr

qb_proj, rb_proj, wr_proj = build_player_projections(weekly, scalers)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("QB (derived)")
    st.dataframe(qb_proj[["Player","mu_pass_yds","sd_pass_yds","mu_pass_tds","sd_pass_tds"]].head(12), use_container_width=True)
with c2:
    st.subheader("RB (derived)")
    st.dataframe(rb_proj[["Player","mu_rush_yds","sd_rush_yds","lam_any_rb"]].head(12), use_container_width=True)
with c3:
    st.subheader("WR/TE (derived)")
    st.dataframe(wr_proj[["Player","mu_receptions","sd_receptions","lam_any_wr"]].head(12), use_container_width=True)

# ---------------- Odds API (Event endpoint) ----------------
st.header("3) Choose an NFL game & markets from The Odds API (event endpoint)")
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

# ---------------- Fetch props & simulate ----------------
st.header("4) Fetch props for this event and simulate")
go = st.button("Fetch lines & simulate")
if go:
    if not markets:
        st.warning("Pick at least one market.")
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
                name = o.get("description")
                side = o.get("name")         # "Over"/"Under" or "Yes"/"No"
                point = o.get("point")       # may be None for anytime TD
                if mkey not in VALID_MARKETS or name is None or side is None:
                    continue
                rows.append({
                    "market": mkey,
                    "player_raw": name,
                    "side": side,
                    "point": (None if point is None else float(point)),
                })

    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    df = pd.DataFrame(rows).drop_duplicates()

    out_rows = []
    qb_names = qb_proj["Player"].tolist()
    rb_names = rb_proj["Player"].tolist()
    wr_names = wr_proj["Player"].tolist()

    for _, r in df.iterrows():
        market = r["market"]; player = r["player_raw"]; point = r["point"]; side = r["side"]

        if market == "player_pass_tds" and not qb_proj.empty:
            match = fuzzy_pick(player, qb_names, cutoff=82)
            if not match: 
                continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_pass_tds"]), float(row["sd_pass_tds"])
            if point is None: 
                continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_pass_yds" and not qb_proj.empty:
            match = fuzzy_pick(player, qb_names, cutoff=82)
            if not match:
                continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_pass_yds"]), float(row["sd_pass_yds"])
            if point is None:
                continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_rush_yds" and not rb_proj.empty:
            match = fuzzy_pick(player, rb_names, cutoff=82)
            if not match:
                continue
            row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_rush_yds"]), float(row["sd_rush_yds"])
            if point is None:
                continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_receptions" and not wr_proj.empty:
            match = fuzzy_pick(player, wr_names, cutoff=82)
            if not match:
                continue
            row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_receptions"]), float(row["sd_receptions"])
            if point is None:
                continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_anytime_td":
            match = None; src = None
            m = fuzzy_pick(player, wr_names, cutoff=82)
            if m is not None:
                match, src = m, "WR"
            if match is None:
                m = fuzzy_pick(player, rb_names, cutoff=82)
                if m is not None:
                    match, src = m, "RB"
            if match is None:
                continue

            if src == "WR":
                row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
                lam = float(row["lam_any_wr"])
            else:
                row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
                lam = float(row["lam_any_rb"])

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
        st.warning("No props matched your players.")
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
