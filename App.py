# app.py — NFL Player Props (All Games) with Odds API + Your CSVs + 2025 Defense EPA
# Run: streamlit run app.py

import time
import math
import random
from io import StringIO
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import process, fuzz

# ------------------------------ Streamlit UI ----------------------------------
st.set_page_config(page_title="NFL Player Props (All Games)", layout="wide")
st.title("📈 NFL Player Props — All Games (Odds API + Your CSVs + Defense EPA)")

SIM_TRIALS = 10000
REQUEST_TIMEOUT = 25
RETRY_ATTEMPTS = 3
RETRY_BASE_SLEEP = 0.8

# ---------------- Supported Player Prop Markets ----------------
# Note: player_rec_yds was invalid on Odds API; renamed to player_rec_yards_alt to handle it internally
MARKETS_ALL = [
    "player_pass_yds",        # Passing Yards
    "player_pass_tds",        # Passing Touchdowns
    "player_rush_yds",        # Rushing Yards
    "player_receptions",      # Receptions
    "player_anytime_td"       # Anytime Touchdown
]

# ---------------------- Embedded 2025 defense EPA multipliers ------------------
# From your table; converted to pass/rush/receiving multipliers inside loader.
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
    for c in ["EPA_Pass", "EPA_Rush", "Comp_Pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert EPA into multipliers; negative EPA (better defense) -> <1.0
    def to_adj(series: pd.Series, scale: float) -> pd.Series:
        x = series.fillna(0.0)
        adj = 1.0 - scale * x
        return adj.clip(0.7, 1.3)

    pass_adj = to_adj(df["EPA_Pass"], 0.8)
    rush_adj = to_adj(df["EPA_Rush"], 0.8)

    # receptions/rec yards: blend pass_adj with comp rate relative to league avg
    comp = df["Comp_Pct"].clip(0.45, 0.80).fillna(df["Comp_Pct"].mean())
    comp_adj = (1.0 + (comp - comp.mean()) * 0.6).clip(0.7, 1.3)
    recv_adj = (0.7 * pass_adj + 0.3 * comp_adj).clip(0.7, 1.3)

    out = pd.DataFrame({
        "team_name": df["Team"].astype(str),
        "pass_adj": pass_adj.astype(float),
        "rush_adj": rush_adj.astype(float),
        "recv_adj": recv_adj.astype(float),
    })
    return out

DEF_TABLE = load_defense_table()

# ----------------------------- Team name helpers ------------------------------
# Abbrev -> Full (so we can map your CSV "Team" to event participants)
ABBR_TO_FULL = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
    "LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","LV":"Las Vegas Raiders","MIA":"Miami Dolphins",
    "MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
    "NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks",
    "SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders",
}

FULL_TO_DEF = {t.lower(): t for t in DEF_TABLE["team_name"]}

def csv_team_to_full(team_val: str) -> Optional[str]:
    if not isinstance(team_val, str) or not team_val.strip():
        return None
    t = team_val.strip()
    if t.upper() in ABBR_TO_FULL:  # abbreviation
        return ABBR_TO_FULL[t.upper()]
    # already full or close — fuzzy match to defense table names
    cand = process.extractOne(t, list(FULL_TO_DEF.values()), scorer=fuzz.token_sort_ratio)
    return cand[0] if cand and cand[1] >= 85 else None

def defense_scalers_for_match(home: str, away: str, player_team_full: str) -> Tuple[str, float, float, float]:
    """
    Given event home/away full team names and player's full team name,
    return: opponent_def_team_name, pass_adj, rush_adj, recv_adj
    """
    home_key = process.extractOne(home, list(FULL_TO_DEF.values()), scorer=fuzz.token_sort_ratio)[0]
    away_key = process.extractOne(away, list(FULL_TO_DEF.values()), scorer=fuzz.token_sort_ratio)[0]
    player_key = process.extractOne(player_team_full, list(FULL_TO_DEF.values()), scorer=fuzz.token_sort_ratio)[0]

    if player_key == home_key:
        opp = away_key
    elif player_key == away_key:
        opp = home_key
    else:
        # Unknown mapping — neutral
        return "NEUTRAL", 1.0, 1.0, 1.0

    row = DEF_TABLE.loc[DEF_TABLE["team_name"] == opp].iloc[0]
    return opp, float(row["pass_adj"]), float(row["rush_adj"]), float(row["recv_adj"])

# -------------------------------- CSV helpers ---------------------------------
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False).str.replace("%","",regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_csv_any(uploaded) -> pd.DataFrame:
    raw = uploaded.read().decode("utf-8", errors="ignore")
    lines = raw.strip().splitlines()
    header_idx = 0
    for i, ln in enumerate(lines[:12]):
        low = ln.lower()
        if low.startswith("rk,player") or low.startswith("player,"):
            header_idx = i; break
    return pd.read_csv(StringIO("\n".join(lines[header_idx:])))

def fuzzy_pick(name: str, candidates: List[str], cutoff=84) -> Optional[str]:
    if not candidates: return None
    res = process.extractOne(name, candidates, scorer=fuzz.token_sort_ratio)
    return res[0] if res and res[1] >= cutoff else None

def over_prob_normal(mu: float, sd: float, line: float, trials: int=SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    return float((np.random.normal(mu, sd, size=trials) > line).mean())

# ----------------------------- 1) Upload CSVs ---------------------------------
st.subheader("1) Upload CSVs (QB / RB / WR)")
c1, c2, c3 = st.columns(3)
with c1: qb_file = st.file_uploader("QB CSV", type=["csv"])
with c2: rb_file = st.file_uploader("RB CSV", type=["csv"])
with c3: wr_file = st.file_uploader("WR CSV", type=["csv"])

qb_df = load_csv_any(qb_file) if qb_file else None
rb_df = load_csv_any(rb_file) if rb_file else None
wr_df = load_csv_any(wr_file) if wr_file else None

if qb_df is not None:
    qb_df = _coerce_numeric(qb_df, ["Y/G","Yds","TD","G","Att","Cmp"])
if rb_df is not None:
    rb_df = _coerce_numeric(rb_df, ["Y/G","Yds","TD","G","Att"])
if wr_df is not None:
    wr_df = _coerce_numeric(wr_df, ["Y/G","Yds","TD","G","Tgt","Rec"])

if not any([qb_df is not None, rb_df is not None, wr_df is not None]):
    st.info("Upload at least one CSV to proceed.")
    st.stop()

# Extract name lists for matching & a simple player->team (from your CSVs)
PLAYER_TO_TEAM: Dict[str, str] = {}
if qb_df is not None:
    for _, r in qb_df.iterrows():
        n = str(r.get("Player", r.get("player",""))).strip()
        t = str(r.get("Team", "")).strip()
        if n: PLAYER_TO_TEAM[n] = t
if rb_df is not None:
    for _, r in rb_df.iterrows():
        n = str(r.get("Player", r.get("player",""))).strip()
        t = str(r.get("Team", "")).strip()
        if n: PLAYER_TO_TEAM[n] = t
if wr_df is not None:
    for _, r in wr_df.iterrows():
        n = str(r.get("Player", r.get("player",""))).strip()
        t = str(r.get("Team", "")).strip()
        if n: PLAYER_TO_TEAM[n] = t

QB_NAMES = list(qb_df["Player"]) if qb_df is not None and "Player" in qb_df.columns else []
RB_NAMES = list(rb_df["Player"]) if rb_df is not None and "Player" in rb_df.columns else []
WR_NAMES = list(wr_df["Player"]) if wr_df is not None and "Player" in wr_df.columns else []

# -------------------- Build per-player baseline means/SDs from CSVs -----------
@st.cache_data(show_spinner=False)
def build_baselines(
    qb_df: Optional[pd.DataFrame],
    rb_df: Optional[pd.DataFrame],
    wr_df: Optional[pd.DataFrame],
):
    qb_stats, rb_stats, wr_stats = {}, {}, {}

    if qb_df is not None:
        for _, r in qb_df.iterrows():
            name = str(r.get("Player", r.get("player",""))).strip()
            g = float(r.get("G", 1) or 1)
            # Passing yards mu/sd
            try: py_mu = float(r.get("Y/G"))
            except Exception:
                try: py_mu = float(r.get("Yds")) / max(1.0, g)
                except Exception: py_mu = 230.0
            py_sd = max(18.0, 0.18 * py_mu)

            # Passing TDs for "anytime TD" not typical; we will not use QB for anytime TD
            qb_stats[name] = {"pass_yds_mu": py_mu, "pass_yds_sd": py_sd}

    if rb_df is not None:
        for _, r in rb_df.iterrows():
            name = str(r.get("Player", r.get("player",""))).strip()
            g = float(r.get("G", 1) or 1)
            try: ry_mu = float(r.get("Y/G"))
            except Exception:
                try: ry_mu = float(r.get("Yds")) / max(1.0, g)
                except Exception: ry_mu = 55.0
            ry_sd = max(6.0, 0.22 * ry_mu)

            # RB anytime TD baseline from TD/G
            try: r_td_mu = float(r.get("TD")) / max(1.0, g)
            except Exception: r_td_mu = 0.45
            r_td_sd = max(0.18, 0.60 * r_td_mu)
            rb_stats[name] = {"rush_yds_mu": ry_mu, "rush_yds_sd": ry_sd, "anytd_mu": r_td_mu, "anytd_sd": r_td_sd}

    if wr_df is not None:
        for _, r in wr_df.iterrows():
            name = str(r.get("Player", r.get("player",""))).strip()
            g = float(r.get("G", 1) or 1)
            try: recyd_mu = float(r.get("Y/G"))
            except Exception:
                try: recyd_mu = float(r.get("Yds")) / max(1.0, g)
                except Exception: recyd_mu = 56.0
            recyd_sd = max(8.0, 0.20 * recyd_mu)

            try: rec_mu = float(r.get("Rec")) / max(1.0, g)
            except Exception: rec_mu = 4.4
            rec_sd = max(1.0, 0.45 * rec_mu)

            # WR/TE anytime TD from TD/G
            try: w_td_mu = float(r.get("TD")) / max(1.0, g)
            except Exception: w_td_mu = 0.35
            w_td_sd = max(0.18, 0.65 * w_td_mu)

            wr_stats[name] = {
                "rec_yds_mu": recyd_mu, "rec_yds_sd": recyd_sd,
                "receptions_mu": rec_mu, "receptions_sd": rec_sd,
                "anytd_mu": w_td_mu, "anytd_sd": w_td_sd
            }

    return qb_stats, rb_stats, wr_stats

QB_BASE, RB_BASE, WR_BASE = build_baselines(qb_df, rb_df, wr_df)

# ---------------------- 2) Odds API settings (all games) ----------------------
st.subheader("2) Pick Odds API options")
api_key = st.text_input("Odds API Key", value="", type="password")
region = st.selectbox("Region", ["us", "us2", "eu", "uk"], index=0)
lookahead = st.slider("Lookahead days", 1, 7, value=2)
book_filter = st.multiselect(
    "Limit to bookmakers (optional; defaults = all returned)",
    ["draftkings","fanduel","betmgm","caesars","pointsbet","williamhill_us","barstool","betrivers"],
    default=[]
)

def http_get(url: str, params: dict) -> dict:
    last_err = None
    for i in range(RETRY_ATTEMPTS):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:250]}")
            return r.json()
        except Exception as e:
            last_err = e
            # jittered backoff
            time.sleep(RETRY_BASE_SLEEP * (1.6 ** i) + random.random() * 0.4)
    raise RuntimeError(last_err)

def list_events(api_key: str, lookahead: int, region: str) -> List[dict]:
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
    params = {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead, "regions": region}
    return http_get(url, params)

def fetch_event_props(event_id: str, api_key: str, region: str, markets: List[str]) -> dict:
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "includeLinks": "false",
    }
    return http_get(url, params)

if not api_key:
    st.info("Enter your Odds API key to fetch games.")
    st.stop()

# ---------------------- 3) Fetch all games & simulate -------------------------
st.subheader("3) Fetch props for ALL games & simulate")
events = []
try:
    events = list_events(api_key, lookahead, region)
    st.caption(f"Found **{len(events)}** events in the next {lookahead} day(s).")
except Exception as e:
    st.error(f"Could not list events: {e}")
    st.stop()

run_all = st.button("Fetch props for all games & simulate")
if not run_all:
    st.stop()

all_rows = []
progress = st.progress(0.0)

for idx, ev in enumerate(events):
    progress.progress((idx+1)/max(1,len(events)))
    home = ev.get("home_team","")
    away = ev.get("away_team","")
    eid = ev.get("id","")

    # Pull props for this event (only our 5 markets)
    try:
        data = fetch_event_props(eid, api_key, region, MARKETS_ALL)
    except Exception as e:
        st.error(f"Event {away} vs {home} fetch failed: {e}")
        continue

    # Optional bookmaker filter
    books = data.get("bookmakers", [])
    if book_filter:
        books = [b for b in books if b.get("key") in book_filter]

    # Flatten outcomes -> one row per (market, player, side, point), avg line across books
    tmp = []
    for bk in books:
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey not in MARKETS_ALL:  # extra guard
                continue
            for o in m.get("outcomes", []):
                name = o.get("description")  # player name
                side = o.get("name")        # "Over"/"Under" OR "Yes" for anytd on some books
                point = o.get("point")      # line (None for anytd on some books)
                if name is None:
                    continue
                # Normalize sides: anytd sometimes uses "Yes"
                if mkey == "player_anytime_td":
                    if side not in ("Yes","No","Over","Under",None):
                        continue
                    side_norm = "Yes" if side in (None,"Yes","Over") else "No"
                else:
                    if side not in ("Over","Under") or point is None:
                        continue
                    side_norm = side
                tmp.append({"market": mkey, "player": name, "side": side_norm, "point": point})

    if not tmp:
        # no props for this event — skip quietly
        continue

    props_df = pd.DataFrame(tmp).drop_duplicates()

    # Now: for each prop row, find player's team from your CSVs, map to opponent defense for THIS event
    for _, r in props_df.iterrows():
        market = r["market"]; player_raw = r["player"]; side = r["side"]; line = r["point"]

        # Which CSV pool to search?
        pool_names = []
        pool = None
        base = None

        if market == "player_pass_yds" and QB_BASE:
            pool_names = QB_NAMES; base = QB_BASE
        elif market == "player_rush_yds" and RB_BASE:
            pool_names = RB_NAMES; base = RB_BASE
        elif market == "player_rec_yds" and WR_BASE:
            pool_names = WR_NAMES; base = WR_BASE
        elif market == "player_receptions" and WR_BASE:
            pool_names = WR_NAMES; base = WR_BASE
        elif market == "player_anytime_td" and (WR_BASE or RB_BASE):
            # anytd across WR+RB pools (try WR first, then RB)
            pool_names = WR_NAMES + [n for n in RB_NAMES if n not in WR_NAMES]
            base = {**WR_BASE, **RB_BASE}
        else:
            continue

        match = fuzzy_pick(player_raw, pool_names, cutoff=84)
        if not match:
            continue

        # Player team -> full -> opponent scalers for this event
        csv_team = PLAYER_TO_TEAM.get(match, "")
        player_full = csv_team_to_full(csv_team) or ""  # may be empty if unknown
        opp_team, pass_adj, rush_adj, recv_adj = defense_scalers_for_match(home, away, player_full or home)

        # Baseline mu/sd from uploaded CSVs
        mu, sd = None, None
        if market == "player_pass_yds":
            mu = base[match]["pass_yds_mu"] * pass_adj
            sd = max(18.0, 0.18 * mu)
        elif market == "player_rush_yds":
            mu = base[match]["rush_yds_mu"] * rush_adj
            sd = max(6.0, 0.22 * mu)
        elif market == "player_rec_yds":
            mu = base[match]["rec_yds_mu"] * recv_adj
            sd = max(8.0, 0.20 * mu)
        elif market == "player_receptions":
            mu = base[match]["receptions_mu"] * recv_adj
            sd = max(1.0, 0.45 * mu)
        elif market == "player_anytime_td":
            any_base = base.get(match, {})
            mu = float(any_base.get("anytd_mu", 0.35))
            # scale by rush/recv depending on which pool the player came from originally
            if match in WR_BASE:
                mu *= recv_adj
                sd = max(0.18, 0.65 * mu)
            else:
                mu *= rush_adj
                sd = max(0.18, 0.60 * mu)

        # Compute probability
        if market == "player_anytime_td":
            # Interpret line as "0.5" if None; treat as over prob of 0.5 TDs with normal approx.
            line_val = 0.5 if line is None or (isinstance(line, float) and math.isnan(line)) else float(line)
            p_over = over_prob_normal(mu, sd, line_val, SIM_TRIALS)
            prob = p_over if side == "Yes" else 1.0 - p_over
        else:
            if line is None:
                continue
            line_val = float(line)
            p_over = over_prob_normal(mu, sd, line_val, SIM_TRIALS)
            prob = p_over if side == "Over" else 1.0 - p_over

        all_rows.append({
            "event": f"{away} @ {home}",
            "market": market,
            "player": match,
            "side": side,
            "line": round(line_val, 3),
            "mu": round(mu, 3),
            "sd": round(sd, 3),
            "prob_%": round(100*prob, 2),
            "player_team_from_csv": player_full or csv_team or "",
            "opp_def": opp_team,
            "pass_adj": round(pass_adj, 3),
            "rush_adj": round(rush_adj, 3),
            "recv_adj": round(recv_adj, 3),
        })

    # small pause to be gentle with the API
    time.sleep(0.5 + random.random() * 0.3)

progress.empty()

if not all_rows:
    st.warning("No props matched your uploaded players across the selected games.")
    st.stop()

results = pd.DataFrame(all_rows).sort_values(["prob_%","event"], ascending=[False, True]).reset_index(drop=True)

st.subheader("4) Results — All Games Combined")
st.caption("Conservative normal model; defense multipliers from your 2025 EPA sheet applied per opponent.")
st.dataframe(results, use_container_width=True, height=560)

st.download_button(
    "⬇️ Download CSV (all props, all games)",
    results.to_csv(index=False).encode("utf-8"),
    file_name="props_sim_results_all_games.csv",
    mime="text/csv",
)

# Quick filters
with st.expander("Filters"):
    mkt = st.multiselect("Market filter", MARKETS_ALL, default=MARKETS_ALL)
    min_prob = st.slider("Minimum probability %", 0, 100, 55)
    view = results[(results["market"].isin(mkt)) & (results["prob_%"] >= min_prob)]
    st.dataframe(view, use_container_width=True, height=380)
