# app.py — Single-game NFL Player Props (Odds API + your CSVs + embedded defense EPA)
# ------------------------------------------------------------
# What it does (one game at a time):
# 1) You upload any of QB / RB / WR CSVs (same format you used before).
# 2) Pick a lookahead and game from The Odds API.
# 3) App fetches player props for that event and simulates O/U % on:
#    - player_pass_yds
#    - player_pass_tds
#    - player_rush_yds
#    - player_receptions
#    - player_receiving_yards (also accepts player_rec_yds)
#    - player_anytime_td (derived from CSV stats)
#
# Notes:
# - We normalize “receiving yards” keys (player_rec_yds → player_receiving_yards).
# - Defense scaling is opponent-based (right side for home/away).
# - Robust fuzzy matching for player names (no crashes on blanks).

import math, time, unicodedata, requests
from io import StringIO
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

st.set_page_config(page_title="NFL Player Props — Single Game (Odds API + CSVs + EPA)", layout="wide")
st.title("📈 NFL Player Props — Single Game (Odds API + Your CSVs + Defense EPA)")

SIM_TRIALS = 10000
HTTP_TIMEOUT = 25
NAME_CUTOFF_MAIN = 74
NAME_CUTOFF_STRONG = 86

# --------------------------------- Defense EPA (embedded) ---------------------------------
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
    def scale_from_epa(s: pd.Series, k: float) -> pd.Series:
        x = s.fillna(0.0)
        m = 1.0 - k * x  # negative EPA → <1 (tougher), positive → >1 (softer)
        return m.clip(0.7, 1.3)
    pass_adj = scale_from_epa(df["EPA_Pass"], 0.8)
    rush_adj = scale_from_epa(df["EPA_Rush"], 0.8)
    comp = df["Comp_Pct"].clip(0.45, 0.80).fillna(df["Comp_Pct"].mean())
    comp_adj = (1.0 + (comp - comp.mean()) * 0.6).clip(0.7, 1.3)
    recv_adj = (0.7 * pass_adj + 0.3 * comp_adj).clip(0.7, 1.3)
    out = pd.DataFrame({
        "Team": df["Team"].astype(str),
        "team_clean": df["Team"].str.lower().str.replace("-", " ").str.strip(),
        "pass_adj": pass_adj.astype(float),
        "rush_adj": rush_adj.astype(float),
        "recv_adj": recv_adj.astype(float),
    })
    return out

DEF = load_defense_table()
st.caption("Defense multipliers (1.0 = neutral) are embedded from your 2025 EPA sheet.")

# --------------------------------- Name cleaning/matching (robust) ---------------------------------
SUFFIXES = (" jr", " sr", " ii", " iii", " iv", " v")
ALIASES = {
    "aj": "a. j.", "dj": "d. j.", "cj": "c. j.",
    "pat mahomes": "patrick mahomes",
    "gabe davis": "gabriel davis",
    "juju smith schuster": "juju smith-schuster",
    "joshua dobbs": "josh dobbs",
    "odell beckham jr": "odell beckham",
}

def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def clean_name(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)): return ""
    s = str(s)
    if "(" in s and s.endswith(")"):
        s = s[:s.rfind("(")]
    s = _strip_accents(s).lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = " ".join(s.split())
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    s = ALIASES.get(s, s)
    return " ".join(s.split())

def safe_last_token(s: str) -> Optional[str]:
    s = clean_name(s)
    if not s: return None
    parts = s.split()
    return parts[-1] if parts else None

def best_match(target: str, candidates: List[str]) -> Tuple[Optional[str], int]:
    t = clean_name(target)
    if not t: return (None, 0)

    for c in candidates:
        if clean_name(c) == t:
            return (c, 100)

    t_last = safe_last_token(t)
    if t_last:
        last_hits = []
        for c in candidates:
            c_last = safe_last_token(c)
            if c_last and c_last == t_last:
                last_hits.append(c)
        if len(last_hits) == 1:
            return (last_hits[0], 95)

    pool = [c for c in candidates if clean_name(c)]
    if not pool: return (None, 0)
    m1 = process.extractOne(t, pool, scorer=fuzz.token_sort_ratio)
    if m1 and m1[1] >= NAME_CUTOFF_STRONG:
        return (m1[0], int(m1[1]))
    m2 = process.extractOne(t, pool, scorer=fuzz.token_set_ratio)
    if m2 and m2[1] >= NAME_CUTOFF_MAIN:
        return (m2[0], int(m2[1]))
    return (None, int(m2[1] if m2 else 0))

def clean_team(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)): return ""
    return " ".join(_strip_accents(str(s)).lower().replace("-", " ").split())

# --------------------------------- CSV loading & projections ---------------------------------
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",","",regex=False).str.replace("%","",regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_any_csv(uploaded) -> pd.DataFrame:
    raw = uploaded.read().decode("utf-8", errors="ignore")
    lines = raw.strip().splitlines()
    header_idx = 0
    for i, ln in enumerate(lines[:10]):
        if ln.lower().startswith("rk,player") or ln.lower().startswith("player,"):
            header_idx = i; break
    return pd.read_csv(StringIO("\n".join(lines[header_idx:])))

def build_player_team_map(*dfs: pd.DataFrame) -> Dict[str, str]:
    names = {}
    for df in dfs:
        if df is None or "Team" not in df.columns: continue
        for _, r in df.iterrows():
            p = str(r.get("Player", r.get("player","")))
            if not clean_name(p): continue
            raw_team = str(r.get("Team",""))
            if not raw_team: continue
            m = process.extractOne(clean_team(raw_team), DEF["team_clean"].tolist(), scorer=fuzz.token_sort_ratio)
            if m and m[1] >= 88:
                idx = DEF.index[DEF["team_clean"] == m[0]][0]
                names[p] = DEF.loc[idx, "Team"]
    return names

def qb_proj_from_csv(df: pd.DataFrame, pass_scale: float=1.0) -> Optional[pd.DataFrame]:
    if df is None: return None
    rows = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player",""))); 
        if not clean_name(name): continue
        g = float(r.get("G", 1) or 1)
        # TDs
        td = r.get("TD", np.nan)
        try: td_mu = float(td) / max(1.0, g)
        except Exception: td_mu = 1.2
        td_mu *= pass_scale
        td_sd = max(0.25, 0.60 * td_mu)
        # Pass yards
        ypg = r.get("Y/G", np.nan); yds = r.get("Yds", np.nan)
        try: py_mu = float(ypg) if pd.notna(ypg) else float(yds) / max(1.0, g)
        except Exception: py_mu = 235.0
        py_mu *= pass_scale
        py_sd = max(20.0, 0.18 * py_mu)
        rows.append({"Player": name, "mu_pass_tds": td_mu, "sd_pass_tds": td_sd,
                     "mu_pass_yds": py_mu, "sd_pass_yds": py_sd})
    return pd.DataFrame(rows)

def rb_proj_from_csv(df: pd.DataFrame, rush_scale: float=1.0) -> Optional[pd.DataFrame]:
    if df is None: return None
    rows = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player",""))); 
        if not clean_name(name): continue
        g = float(r.get("G", 1) or 1)
        ypg = r.get("Y/G", np.nan); yds = r.get("Yds", np.nan)
        try: rush_mu = float(ypg) if pd.notna(ypg) else float(yds) / max(1.0, g)
        except Exception: rush_mu = 55.0
        rush_mu *= rush_scale
        rush_sd = max(6.0, 0.22 * rush_mu)
        rows.append({"Player": name, "mu_rush_yds": rush_mu, "sd_rush_yds": rush_sd})
    return pd.DataFrame(rows)

def wr_proj_from_csv(df: pd.DataFrame, recv_scale: float=1.0) -> Optional[pd.DataFrame]:
    if df is None: return None
    rows = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player",""))); 
        if not clean_name(name): continue
        g = float(r.get("G", 1) or 1)
        # Rec yards
        ypg = r.get("Y/G", np.nan); yds = r.get("Yds", np.nan)
        try: rec_yds_mu = float(ypg) if pd.notna(ypg) else float(yds) / max(1.0, g)
        except Exception: rec_yds_mu = 55.0
        rec_yds_mu *= recv_scale
        rec_yds_sd = max(8.0, 0.22 * rec_yds_mu)
        # Receptions
        rec = r.get("Rec", np.nan)
        try: rec_mu = float(rec) / max(1.0, g)
        except Exception: rec_mu = 4.5
        rec_mu *= recv_scale
        rec_sd = max(1.0, 0.45 * rec_mu)
        rows.append({"Player": name,
                     "mu_rec_yards": rec_yds_mu, "sd_rec_yards": rec_yds_sd,
                     "mu_receptions": rec_mu,   "sd_receptions": rec_sd})
    return pd.DataFrame(rows)

def norm_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    return float((np.random.normal(mu, sd, size=trials) > line).mean())

# --------------------------------- UI — CSVs ---------------------------------
st.markdown("### 1) Upload CSVs (QB / RB / WR)")
c1,c2,c3 = st.columns(3)
with c1: qb_file = st.file_uploader("QB CSV", type=["csv"])
with c2: rb_file = st.file_uploader("RB CSV", type=["csv"])
with c3: wr_file = st.file_uploader("WR CSV", type=["csv"])

qb_df = _load_any_csv(qb_file) if qb_file else None
rb_df = _load_any_csv(rb_file) if rb_file else None
wr_df = _load_any_csv(wr_file) if wr_file else None

if qb_df is not None: qb_df = _coerce_numeric(qb_df, ["Y/G","Yds","TD","G","Att","Cmp"])
if rb_df is not None: rb_df = _coerce_numeric(rb_df, ["Y/G","Yds","TD","G","Att"])
if wr_df is not None: wr_df = _coerce_numeric(wr_df, ["Y/G","Yds","TD","G","Tgt","Rec"])

qb_proj = qb_proj_from_csv(qb_df)
rb_proj = rb_proj_from_csv(rb_df)
wr_proj = wr_proj_from_csv(wr_df)

PLAYER_TO_TEAM = build_player_team_map(qb_df, rb_df, wr_df)

# --------------------------------- Odds API (events + event odds) ---------------------------------
def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()

def list_events(api_key: str, lookahead_days: int, region: str) -> List[dict]:
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
    params = {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region}
    return odds_get(base, params)

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str], bookmakers: List[str]|None):
    base = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": region, "oddsFormat": "american"}
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    # Try sending all; if a market is invalid, drop it and retry.
    remain = markets[:]
    last_err = None
    for _ in range(6):
        if not remain: break
        params["markets"] = ",".join(remain)
        try:
            return odds_get(base, params)
        except requests.HTTPError as e:
            msg = str(e); last_err = msg
            bad = None
            for m in list(remain):
                if m in msg:
                    bad = m; break
            if bad:
                remain.remove(bad)
                continue
            else:
                raise
    if last_err:
        raise requests.HTTPError(last_err)
    return {"bookmakers": []}

MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_receptions",
    "player_receiving_yards",  # we also accept player_rec_yds and map to this
    "player_rec_yds",
    "player_anytime_td",
]

def collect_outcomes(event_json: dict) -> pd.DataFrame:
    rows = []
    for bk in event_json.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = o.get("description")
                side = o.get("name")
                point = o.get("point")
                if name is None or point is None or side not in ("Over","Under"):
                    continue
                try:
                    pt = float(point)
                except Exception:
                    continue
                rows.append({"market": mkey, "player_raw": name, "side": side, "point": pt})
    if not rows:
        return pd.DataFrame(columns=["market","player_raw","side","point"])
    df = pd.DataFrame(rows).drop_duplicates()
    df["market"] = df["market"].replace({"player_rec_yds":"player_receiving_yards"})
    return df

# --------------------------------- Defense scaling per opponent ---------------------------------
def multipliers_for_opponent(team_full: str) -> Dict[str,float]:
    row = DEF.loc[DEF["Team"] == team_full]
    if row.empty:
        return {"pass":1.0, "rush":1.0, "recv":1.0}
    r = row.iloc[0]
    return {"pass":float(r["pass_adj"]), "rush":float(r["rush_adj"]), "recv":float(r["recv_adj"])}

def apply_defense_scaling(player: str, market: str, home_team: str, away_team: str) -> float:
    p_team = PLAYER_TO_TEAM.get(player)
    if not p_team:  # unknown → neutral
        return 1.0
    ch, ca, cp = clean_team(home_team), clean_team(away_team), clean_team(p_team)
    if cp == ch:
        opp = away_team
    elif cp == ca:
        opp = home_team
    else:
        return 1.0
    m = multipliers_for_opponent(opp)
    if market in ("player_pass_tds","player_pass_yds"):
        return m["pass"]
    if market == "player_rush_yds":
        return m["rush"]
    if market in ("player_receptions","player_receiving_yards"):
        return m["recv"]
    # anytime TD: blend rush/recv
    return 0.5*m["rush"] + 0.5*m["recv"]

def simulate_from_row(market: str, base_row: pd.Series, line: float, scale: float) -> Tuple[float,float]:
    if market == "player_pass_tds":
        mu, sd = base_row["mu_pass_tds"], base_row["sd_pass_tds"]
    elif market == "player_pass_yds":
        mu, sd = base_row["mu_pass_yds"], base_row["sd_pass_yds"]
    elif market == "player_rush_yds":
        mu, sd = base_row["mu_rush_yds"], base_row["sd_rush_yds"]
    elif market == "player_receptions":
        mu, sd = base_row["mu_receptions"], base_row["sd_receptions"]
    elif market == "player_receiving_yards":
        mu, sd = base_row["mu_rec_yards"], base_row["sd_rec_yards"]
    else:  # anytime TD (derived)
        if "mu_receptions" in base_row:
            mu = 0.10 * float(base_row["mu_receptions"])
        elif "mu_rush_yds" in base_row:
            mu = 0.06 * float(base_row["mu_rush_yds"])
        else:
            mu = 0.35
        sd = max(0.20, 0.60 * mu)

    mu *= scale
    sd *= max(0.9, min(1.1, scale))
    p_over = norm_over_prob(mu, sd, line, SIM_TRIALS)
    return mu, p_over

# --------------------------------- API inputs & game picker ---------------------------------
st.markdown("### 2) Pick a game from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept in session only)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
books = st.multiselect("Limit to bookmakers (optional)", 
                       ["draftkings","fanduel","betmgm","caesars","pointsbet","williamhill_us","bet365","barstool"])

if not api_key:
    st.info("Enter your Odds API key.")
    st.stop()

try:
    events = list_events(api_key, lookahead, region)
except Exception as e:
    st.error(f"Events fetch error: {e}")
    st.stop()

if not events:
    st.warning("No events returned.")
    st.stop()

labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", labels, index=0)
event = events[labels.index(pick)]
home, away, eid = event["home_team"], event["away_team"], event["id"]

# --------------------------------- Fetch & simulate for this game ---------------------------------
st.markdown("### 3) Fetch Odds API props for this event and simulate")
go = st.button("Fetch & simulate this event")
if go:
    if all(x is None for x in [qb_proj, rb_proj, wr_proj]):
        st.warning("Upload at least one of QB / RB / WR CSVs first.")
        st.stop()

    try:
        data = fetch_event_props(api_key, eid, region, MARKETS, books)
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    df = collect_outcomes(data)
    if df.empty:
        st.warning("No player outcomes returned for this event.")
        st.stop()

    qb_names = qb_proj["Player"].dropna().astype(str).tolist() if qb_proj is not None else []
    rb_names = rb_proj["Player"].dropna().astype(str).tolist() if rb_proj is not None else []
    wr_names = wr_proj["Player"].dropna().astype(str).tolist() if wr_proj is not None else []

    rows, um = [], []
    for _, r in df.iterrows():
        mkt, raw, side, line = r["market"], r["player_raw"], r["side"], r["point"]

        base, match, score = None, None, 0
        if mkt in ("player_pass_tds","player_pass_yds") and qb_proj is not None:
            match, score = best_match(raw, qb_names)
            if match is not None:
                base = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
        elif mkt == "player_rush_yds" and rb_proj is not None:
            match, score = best_match(raw, rb_names)
            if match is not None:
                base = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
        elif mkt in ("player_receptions","player_receiving_yards") and wr_proj is not None:
            match, score = best_match(raw, wr_names)
            if match is not None:
                base = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
        elif mkt == "player_anytime_td":
            # try WR, then RB, then QB
            if wr_proj is not None:
                match, score = best_match(raw, wr_names)
                if match is not None:
                    base = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
            if base is None and rb_proj is not None:
                match, score = best_match(raw, rb_names)
                if match is not None:
                    base = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
            if base is None and qb_proj is not None:
                match, score = best_match(raw, qb_names)
                if match is not None:
                    base = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
        else:
            continue

        if base is None or match is None:
            um.append((mkt, raw, score))
            continue

        scale = apply_defense_scaling(match, mkt, home, away)
        mu, p_over = simulate_from_row(mkt, base, line, scale)
        p = p_over if side == "Over" else 1.0 - p_over

        rows.append({
            "market": mkt,
            "player": match,
            "side": side,
            "line": round(line, 3),
            "mu": round(float(mu), 3),
            "sd": np.nan,  # not strictly needed for display; mu drives the sim
            "prob": round(100*float(p), 2),
            "opp_def": (away if PLAYER_TO_TEAM.get(match) == home else home),
            "pass_adj": round(multipliers_for_opponent(away if PLAYER_TO_TEAM.get(match) == home else home)["pass"], 3),
            "rush_adj": round(multipliers_for_opponent(away if PLAYER_TO_TEAM.get(match) == home else home)["rush"], 3),
            "recv_adj": round(multipliers_for_opponent(away if PLAYER_TO_TEAM.get(match) == home else home)["recv"], 3),
        })

    if not rows:
        st.warning("No props matched your uploaded players for this event.")
        if um:
            st.markdown("**Unmatched (for debugging)**")
            st.dataframe(pd.DataFrame(um, columns=["market","book_player","fuzzy_score"]), use_container_width=True)
        st.stop()

    results = pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)
    st.subheader("Simulated props for this game (conservative normal model)")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="props_sim_results.csv",
        mime="text/csv",
    )

    if um:
        st.markdown("#### Unmatched props (debug)")
        st.dataframe(pd.DataFrame(um, columns=["market","book_player","fuzzy_score"])
                     .sort_values("fuzzy_score", ascending=False),
                     use_container_width=True, height=240)
