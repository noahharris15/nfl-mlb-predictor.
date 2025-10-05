# Player Props Simulator — Odds API + ESPN JSON (no CSVs)
# Run: streamlit run app.py

import re
import math
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO
from typing import List, Optional, Dict, Tuple
from rapidfuzz import process, fuzz

st.set_page_config(page_title="NFL Player Props — Odds API + ESPN", layout="wide")
st.title("📈 NFL Player Props — Odds API + ESPN (defense EPA embedded)")

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
    try:
        return float(x)
    except Exception:
        return float("nan")

def per_game(val, games):
    try:
        g = max(1.0, float(games or 0))
        v = float(val)
        if np.isnan(v): return np.nan
        return v / g
    except Exception:
        return np.nan

# ---------------- Name normalization for ESPN search ----------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]           # drop " (Team)" suffix
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)  # remove dots/commas/apostrophes
    n = re.sub(r"\s+", " ", n).strip()
    n = strip_accents(n)
    # remove leading single-letter initials like "D Carr"
    parts = n.split()
    if len(parts) >= 2 and len(parts[0]) == 1:
        n = " ".join(parts[1:])
    return n

def name_variants(n: str) -> List[str]:
    n = normalize_name(n)
    parts = n.split()
    variants = [n]
    if len(parts) >= 2:
        variants.append(f"{parts[0]} {parts[-1]}")  # first last
        variants.append(parts[-1])                  # last name only
    return list(dict.fromkeys([v.lower() for v in variants]))

# ---------------- ESPN lookups (no keys) ----------------
ESPN_SEARCH_URL = "https://site.api.espn.com/apis/search/v2"
ESPN_STATS_URL_SEASON = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/{athlete_id}/statistics/{season}"
ESPN_STATS_URL_GENERIC = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/{athlete_id}/statistics"

@st.cache_data(show_spinner=False)
def espn_find_athlete_id(odds_name: str) -> Optional[str]:
    """Resolve ESPN athlete id using multiple query variants + fuzzy pick on last name."""
    variants = name_variants(odds_name)
    last = variants[-1] if variants else ""
    # try each variant directly
    for q in variants:
        try:
            r = requests.get(ESPN_SEARCH_URL, params={"query": q}, timeout=15)
            r.raise_for_status()
            js = r.json()
            candidates = []
            for cat in js.get("results", []):
                for item in cat.get("items", []):
                    if item.get("type") == "athlete":
                        leagues = item.get("leagues") or []
                        if any((lg.get("abbr") or "").upper() == "NFL" for lg in leagues):
                            # label is usually the athlete's full name
                            label = (item.get("name") or item.get("displayName") or item.get("fullName") or "").strip()
                            aid = str(item.get("id") or "")
                            uid = item.get("uid", "")
                            if not aid.isdigit() and "a:" in uid:
                                try: aid = uid.split("a:")[1].split("~")[0]
                                except Exception: pass
                            if aid and aid.isdigit():
                                candidates.append((label, aid))
            if candidates:
                labels = [c[0] for c in candidates]
                pick = fuzzy_pick(normalize_name(odds_name).lower(), [normalize_name(x).lower() for x in labels], cutoff=60)
                if pick:
                    for lbl, aid in candidates:
                        if normalize_name(lbl).lower() == normalize_name(pick).lower():
                            return aid
        except Exception:
            continue
    # fallback: last name broad search → pick best
    try:
        r = requests.get(ESPN_SEARCH_URL, params={"query": last}, timeout=15)
        r.raise_for_status()
        js = r.json()
        pool = []
        for cat in js.get("results", []):
            for item in cat.get("items", []):
                if item.get("type") == "athlete":
                    leagues = item.get("leagues") or []
                    if any((lg.get("abbr") or "").upper() == "NFL" for lg in leagues):
                        label = (item.get("name") or item.get("displayName") or item.get("fullName") or "").strip()
                        aid = str(item.get("id") or "")
                        uid = item.get("uid", "")
                        if not aid.isdigit() and "a:" in uid:
                            try: aid = uid.split("a:")[1].split("~")[0]
                            except Exception: pass
                        if aid and aid.isdigit():
                            pool.append((label, aid))
        if pool:
            labels = [normalize_name(x[0]).lower() for x in pool]
            pick = fuzzy_pick(normalize_name(odds_name).lower(), labels, cutoff=58)
            if pick:
                for (lbl, aid) in pool:
                    if normalize_name(lbl).lower() == pick:
                        return aid
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def espn_fetch_athlete_season_stats(athlete_id: str, season: int) -> Dict[str, float]:
    """
    Returns totals & games for season.
    Keys: pass_yds, pass_tds, rush_yds, rush_tds, rec, rec_tds, games
    """
    out = {"pass_yds": np.nan, "pass_tds": np.nan, "rush_yds": np.nan, "rush_tds": np.nan,
           "rec": np.nan, "rec_tds": np.nan, "games": np.nan}
    def parse_payload(js):
        # ESPN stats payload has categories with name + stats [{name,value}]
        cats = js.get("categories") or []
        for cat in cats:
            label = (cat.get("name") or "").lower()
            stats = {s.get("name"): s.get("value") for s in (cat.get("stats") or [])}
            if "pass" in label:
                out["pass_yds"] = stats.get("yards", out["pass_yds"])
                out["pass_tds"] = stats.get("touchdowns", out["pass_tds"])
                out["games"] = max(out["games"] or 0, stats.get("gamesPlayed") or 0)
            elif "rush" in label:
                out["rush_yds"] = stats.get("yards", out["rush_yds"])
                out["rush_tds"] = stats.get("touchdowns", out["rush_tds"])
                out["games"] = max(out["games"] or 0, stats.get("gamesPlayed") or 0)
            elif "receive" in label:
                out["rec"] = stats.get("receptions", out["rec"])
                out["rec_tds"] = stats.get("touchdowns", out["rec_tds"])
                out["games"] = max(out["games"] or 0, stats.get("gamesPlayed") or 0)

    # Try season-specific URL first
    try:
        url = ESPN_STATS_URL_SEASON.format(athlete_id=athlete_id, season=season)
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            parse_payload(r.json())
        else:
            # generic then (ESPN sometimes nests all seasons there)
            url2 = ESPN_STATS_URL_GENERIC.format(athlete_id=athlete_id)
            r2 = requests.get(url2, timeout=20)
            if r2.status_code == 200:
                parse_payload(r2.json())
    except Exception:
        pass
    return out

# ---------------- Inputs ----------------
st.header("1) Season & opponent defense")
c1, c2 = st.columns([1,1])
with c1:
    season = st.number_input("Season", min_value=2015, max_value=2100, value=2025, step=1)
with c2:
    opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()  # pass_adj, rush_adj, recv_adj

# ---------------- Odds API (Event endpoint) ----------------
st.header("2) Choose an NFL game & markets from The Odds API")
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
st.header("3) Fetch props for this event and simulate (ESPN stats)")
go = st.button("Fetch lines & simulate")
if go:
    if not markets:
        st.warning("Pick at least one market.")
        st.stop()

    # 1) Get props from Odds API
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
                name = o.get("description")  # player name
                side = o.get("name")         # "Over"/"Under" or "Yes"/"No"
                point = o.get("point")       # None for anytime TD on some books
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

    props_df = pd.DataFrame(rows).drop_duplicates()
    st.caption(f"Fetched {len(props_df)} player outcomes from Odds API for the selected game.")

    # 2) Resolve ESPN athlete IDs only for players we need
    unique_players = sorted(props_df["player_raw"].unique().tolist())

    player_to_espn: Dict[str, Optional[str]] = {}
    for name in unique_players:
        player_to_espn[name] = espn_find_athlete_id(name)

    matched = {p: a for p, a in player_to_espn.items() if a}
    missed  = [p for p, a in player_to_espn.items() if not a]

    with st.expander("ESPN ID matches"):
        st.write(matched)
    if missed:
        with st.expander("Players that failed ESPN lookup"):
            st.write(missed)

    # 3) For each matched player, fetch ESPN season stats and produce mu/sd per market
    qb_proj_rows, rb_proj_rows, wr_proj_rows = [], [], []
    for player, aid in matched.items():
        stats = espn_fetch_athlete_season_stats(aid, season)
        games = to_float(stats.get("games"))

        mu_pass_yds = per_game(stats.get("pass_yds"), games)
        mu_pass_tds = per_game(stats.get("pass_tds"), games)
        mu_rush_yds = per_game(stats.get("rush_yds"), games)
        mu_recs     = per_game(stats.get("rec"), games)
        mu_rec_tds  = per_game(stats.get("rec_tds"), games)
        mu_rush_tds = per_game(stats.get("rush_tds"), games)

        # apply defense scalers
        if not np.isnan(mu_pass_yds): mu_pass_yds *= scalers["pass_adj"]
        if not np.isnan(mu_pass_tds): mu_pass_tds *= scalers["pass_adj"]
        if not np.isnan(mu_rush_yds): mu_rush_yds *= scalers["rush_adj"]
        if not np.isnan(mu_recs):     mu_recs     *= scalers["recv_adj"]
        if not np.isnan(mu_rec_tds):  mu_rec_tds  *= scalers["recv_adj"]
        if not np.isnan(mu_rush_tds): mu_rush_tds *= scalers["rush_adj"]

        # SD heuristics
        sd_pass_yds = max(25.0, (mu_pass_yds * 0.20)) if not np.isnan(mu_pass_yds) else np.nan
        sd_pass_tds = max(0.25, (mu_pass_tds * 0.60)) if not np.isnan(mu_pass_tds) else np.nan
        sd_rush_yds = max(6.0,  (mu_rush_yds * 0.22)) if not np.isnan(mu_rush_yds) else np.nan
        sd_recs     = max(1.0,  (mu_recs * 0.45))     if not np.isnan(mu_recs) else np.nan

        qb_proj_rows.append({
            "Player": player,
            "mu_pass_yds": mu_pass_yds, "sd_pass_yds": sd_pass_yds,
            "mu_pass_tds": mu_pass_tds, "sd_pass_tds": sd_pass_tds,
        })
        rb_proj_rows.append({
            "Player": player,
            "mu_rush_yds": mu_rush_yds, "sd_rush_yds": sd_rush_yds,
            "lam_any_rb": (0 if np.isnan(mu_rec_tds) else mu_rec_tds) + (0 if np.isnan(mu_rush_tds) else mu_rush_tds),
        })
        wr_proj_rows.append({
            "Player": player,
            "mu_receptions": mu_recs, "sd_receptions": sd_recs,
            "lam_any_wr": mu_rec_tds,
        })

    qb_proj = pd.DataFrame(qb_proj_rows).drop_duplicates(subset=["Player"])
    rb_proj = pd.DataFrame(rb_proj_rows).drop_duplicates(subset=["Player"])
    wr_proj = pd.DataFrame(wr_proj_rows).drop_duplicates(subset=["Player"])

    c1, c2, c3 = st.columns(3)
    with c1:
        if not qb_proj.empty: st.dataframe(qb_proj.head(12), use_container_width=True)
    with c2:
        if not rb_proj.empty: st.dataframe(rb_proj.head(12), use_container_width=True)
    with c3:
        if not wr_proj.empty: st.dataframe(wr_proj.head(12), use_container_width=True)

    if qb_proj.empty and rb_proj.empty and wr_proj.empty:
        st.warning("No ESPN-based projections built. Check the 'failed lookup' expander for player names.")
        st.stop()

    # 4) Simulate for each prop outcome using ESPN-derived projections
    out_rows = []
    qb_names = qb_proj["Player"].tolist() if not qb_proj.empty else []
    rb_names = rb_proj["Player"].tolist() if not rb_proj.empty else []
    wr_names = wr_proj["Player"].tolist() if not wr_proj.empty else []

    for _, r in props_df.iterrows():
        market = r["market"]; player = r["player_raw"]; point = r["point"]; side = r["side"]

        if market == "player_pass_tds" and not qb_proj.empty:
            match = fuzzy_pick(player, qb_names, cutoff=60)
            if not match: 
                continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row.get("mu_pass_tds", np.nan)), float(row.get("sd_pass_tds", np.nan))
            if np.isnan(mu) or np.isnan(sd) or point is None: 
                continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_pass_yds" and not qb_proj.empty:
            match = fuzzy_pick(player, qb_names, cutoff=60)
            if not match:
                continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row.get("mu_pass_yds", np.nan)), float(row.get("sd_pass_yds", np.nan))
            if np.isnan(mu) or np.isnan(sd) or point is None:
                continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_rush_yds" and not rb_proj.empty:
            match = fuzzy_pick(player, rb_names, cutoff=60)
            if not match:
                continue
            row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
            mu, sd = float(row.get("mu_rush_yds", np.nan)), float(row.get("sd_rush_yds", np.nan))
            if np.isnan(mu) or np.isnan(sd) or point is None:
                continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_receptions" and not wr_proj.empty:
            match = fuzzy_pick(player, wr_names, cutoff=60)
            if not match:
                continue
            row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
            mu, sd = float(row.get("mu_receptions", np.nan)), float(row.get("sd_receptions", np.nan))
            if np.isnan(mu) or np.isnan(sd) or point is None:
                continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif market == "player_anytime_td":
            match = None; src = None
            if not wr_proj.empty:
                m = fuzzy_pick(player, wr_names, cutoff=60)
                if m: match, src = m, "WR"
            if match is None and not rb_proj.empty:
                m = fuzzy_pick(player, rb_names, cutoff=60)
                if m: match, src = m, "RB"
            if match is None:
                continue

            if src == "WR":
                row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
                lam = float(row.get("lam_any_wr", np.nan))
            else:
                row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
                lam = float(row.get("lam_any_rb", np.nan))

            if np.isnan(lam): 
                continue
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
        st.warning("No props could be simulated after ESPN lookups. Check 'failed lookup' list above.")
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
