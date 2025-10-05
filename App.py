# app.py — NFL Player Props (Odds API + CSVs + Defense EPA), all-in-one

import numpy as np
import pandas as pd
import streamlit as st
import requests, unicodedata, time
from io import StringIO
from typing import Optional, List, Dict, Tuple
from rapidfuzz import process, fuzz

# ------------------------------------------------------------------------------
# Streamlit basics
# ------------------------------------------------------------------------------
st.set_page_config(page_title="NFL Player Props — Odds API + CSVs (EPA)", layout="wide")
st.title("📈 NFL Player Props — Odds API + Your CSVs (Defense EPA embedded)")

SIM_TRIALS = 10000
HTTP_TIMEOUT = 25
NAME_CUTOFF_MAIN = 74
NAME_CUTOFF_STRONG = 86

SUFFIXES = (" jr", " sr", " ii", " iii", " iv", " v")
ALIASES = {
    "aj": "a. j.", "dj": "d. j.", "cj": "c. j.",
    "pat mahomes": "patrick mahomes",
    "gabe davis": "gabriel davis",
    "juju smith schuster": "juju smith-schuster",
    "joshua dobbs": "josh dobbs",
    "odell beckham jr": "odell beckham",
}

# ——— Defense EPA table (2025) ———
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

# ------------------------------------------------------------------------------
# Helpers: name cleaning / matching
# ------------------------------------------------------------------------------
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def clean_name(s: str) -> str:
    if not s: return ""
    s = str(s)
    # Drop team tags like " (MIN)"
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

def best_match(target: str, candidates: List[str]) -> Tuple[Optional[str], int]:
    t = clean_name(target)
    if not t: return (None, 0)

    # Exact after cleaning
    for c in candidates:
        if clean_name(c) == t:
            return (c, 100)

    # Unique last name
    t_last = t.split()[-1]
    last_hits = [c for c in candidates if clean_name(c).split()[-1] == t_last]
    if len(last_hits) == 1:
        return (last_hits[0], 95)

    # Fuzzy: strong then relaxed
    choices = list(candidates)
    m1 = process.extractOne(t, choices, scorer=fuzz.token_sort_ratio)
    if m1 and m1[1] >= NAME_CUTOFF_STRONG:
        return (m1[0], int(m1[1]))
    m2 = process.extractOne(t, choices, scorer=fuzz.token_set_ratio)
    if m2 and m2[1] >= NAME_CUTOFF_MAIN:
        return (m2[0], int(m2[1]))
    return (None, int(m2[1] if m2 else 0))

def clean_team(s: str) -> str:
    return " ".join(_strip_accents(str(s)).lower().replace("-", " ").split())

# ------------------------------------------------------------------------------
# Defense table → multipliers
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_defense_table() -> pd.DataFrame:
    df = pd.read_csv(StringIO(DEFENSE_EPA_2025))
    for c in ["EPA_Pass","EPA_Rush","Comp_Pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    def scale_from_epa(s: pd.Series, k: float) -> pd.Series:
        x = s.fillna(0.0)
        m = 1.0 - k * x  # negative => <1 (tougher), positive => >1 (softer)
        return m.clip(0.7, 1.3)
    pass_adj = scale_from_epa(df["EPA_Pass"], 0.8)
    rush_adj = scale_from_epa(df["EPA_Rush"], 0.8)
    comp = df["Comp_Pct"].clip(0.45, 0.80).fillna(df["Comp_Pct"].mean())
    comp_adj = (1.0 + (comp - comp.mean()) * 0.6).clip(0.7, 1.3)
    recv_adj = (0.7 * pass_adj + 0.3 * comp_adj).clip(0.7, 1.3)
    out = pd.DataFrame({
        "Team": df["Team"].astype(str),
        "team_clean": df["Team"].apply(clean_team),
        "pass_adj": pass_adj.astype(float),
        "rush_adj": rush_adj.astype(float),
        "recv_adj": recv_adj.astype(float),
    })
    return out

DEF = load_defense_table()
DEF_TEAMS = DEF["Team"].tolist()

# ------------------------------------------------------------------------------
# CSV loading + projections
# ------------------------------------------------------------------------------
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
    """Map player -> team (full name from DEF table) using CSV 'Team' if present."""
    names = {}
    for df in dfs:
        if df is None: continue
        if "Team" not in df.columns: continue
        for _, r in df.iterrows():
            p = str(r.get("Player", r.get("player","")))
            if not p: continue
            raw_team = str(r.get("Team",""))
            if not raw_team: continue
            # Fuzzy team match to DEF names
            m = process.extractOne(clean_team(raw_team), DEF["team_clean"].tolist(), scorer=fuzz.token_sort_ratio)
            if m and m[1] >= 88:
                team_full = DEF.iloc[[i for i,t in enumerate(DEF["team_clean"]) if t == m[0]][0]]["Team"]
                names[p] = team_full
    return names

def qb_proj_from_csv(df: pd.DataFrame, pass_scale: float) -> Optional[pd.DataFrame]:
    if df is None: return None
    rows = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player","")))
        g = float(r.get("G", 1) or 1)
        td = r.get("TD", np.nan)
        try: td_mu = float(td) / max(1.0, g)
        except Exception: td_mu = 1.2
        td_mu *= pass_scale
        td_sd = max(0.25, 0.60 * td_mu)
        # OPTIONAL: pass yards from Y/G if you want
        ypg = r.get("Y/G", np.nan); yds = r.get("Yds", np.nan)
        try: py_mu = float(ypg) if pd.notna(ypg) else float(yds) / max(1.0, g)
        except Exception: py_mu = 235.0
        py_mu *= pass_scale
        py_sd = max(20.0, 0.18 * py_mu)
        rows.append({
            "Player": name,
            "mu_pass_tds": td_mu, "sd_pass_tds": td_sd,
            "mu_pass_yds": py_mu, "sd_pass_yds": py_sd,
        })
    return pd.DataFrame(rows)

def rb_proj_from_csv(df: pd.DataFrame, rush_scale: float) -> Optional[pd.DataFrame]:
    if df is None: return None
    rows = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player","")))
        g = float(r.get("G", 1) or 1)
        ypg = r.get("Y/G", np.nan); yds = r.get("Yds", np.nan)
        try: rush_mu = float(ypg) if pd.notna(ypg) else float(yds) / max(1.0, g)
        except Exception: rush_mu = 55.0
        rush_mu *= rush_scale
        rush_sd = max(6.0, 0.22 * rush_mu)
        rows.append({"Player": name, "mu_rush_yds": rush_mu, "sd_rush_yds": rush_sd})
    return pd.DataFrame(rows)

def wr_proj_from_csv(df: pd.DataFrame, recv_scale: float) -> Optional[pd.DataFrame]:
    if df is None: return None
    rows = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player","")))
        g = float(r.get("G", 1) or 1)
        # Receiving yards
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
        rows.append({
            "Player": name,
            "mu_rec_yards": rec_yds_mu, "sd_rec_yards": rec_yds_sd,
            "mu_receptions": rec_mu,   "sd_receptions": rec_sd,
        })
    return pd.DataFrame(rows)

def norm_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    return float((np.random.normal(mu, sd, size=trials) > line).mean())

# ------------------------------------------------------------------------------
# UI — CSV uploads
# ------------------------------------------------------------------------------
st.markdown("### 1) Upload CSVs (any/each of QB / RB / WR)")
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

# Start with neutral (will be replaced by event-specific opponent later)
qb_proj = qb_proj_from_csv(qb_df, 1.0) if qb_df is not None else None
rb_proj = rb_proj_from_csv(rb_df, 1.0) if rb_df is not None else None
wr_proj = wr_proj_from_csv(wr_df, 1.0) if wr_df is not None else None

# Precompute player→team (from CSV 'Team' column)
PLAYER_TO_TEAM = build_player_team_map(qb_df, rb_df, wr_df)

# ------------------------------------------------------------------------------
# Odds API bits
# ------------------------------------------------------------------------------
def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()

def list_events(api_key: str, lookahead_days: int, region: str) -> List[dict]:
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
    params = {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region}
    return odds_get(base, params)

def fetch_event_with_markets(api_key: str, event_id: str, region: str, markets: List[str], bookmakers: List[str]|None):
    base = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": region, "oddsFormat": "american"}
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    # Some accounts don’t support certain market keys. We iteratively drop the offender on 422.
    remaining = markets[:]
    last_err = None
    for _ in range(6):
        if not remaining: break
        params["markets"] = ",".join(remaining)
        try:
            return odds_get(base, params)
        except requests.HTTPError as e:
            msg = str(e)
            last_err = msg
            # try to parse offending 'invalid market'
            bad = None
            for m in remaining:
                if m in msg:
                    bad = m; break
            if bad:
                remaining.remove(bad)
                continue
            else:
                raise
    if last_err:
        raise requests.HTTPError(last_err)
    return {"bookmakers": []}

# Preferred market keys (we’ll request all; backend will drop invalids)
MARKETS_ALL = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_receptions",
    # try several receiving-yards spellings; the fetch function will drop invalid ones
    "player_receiving_yards",
    "player_rec_yds",
    "player_anytime_td",
]

# ------------------------------------------------------------------------------
# UI — API inputs
# ------------------------------------------------------------------------------
st.markdown("### 2) Odds API inputs")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=2)
limit_books = st.multiselect("Limit to bookmakers (optional)",
                             ["draftkings","fanduel","betmgm","caesars","pointsbet","williamhill_us","bet365","barstool"])

if not api_key:
    st.info("Add your Odds API key to proceed.")
    st.stop()

# ------------------------------------------------------------------------------
# Fetch events
# ------------------------------------------------------------------------------
try:
    events = list_events(api_key, lookahead, region)
except Exception as e:
    st.error(f"Event list error: {e}")
    st.stop()

st.caption(f"Found **{len(events)}** events in the next **{lookahead}** day(s).")

# ------------------------------------------------------------------------------
# Core: simulate all events
# ------------------------------------------------------------------------------
def multipliers_for_opponent(opponent_team_full: str) -> Dict[str,float]:
    row = DEF.loc[DEF["Team"] == opponent_team_full]
    if row.empty:
        return {"pass":1.0, "rush":1.0, "recv":1.0}
    r = row.iloc[0]
    return {"pass":float(r["pass_adj"]), "rush":float(r["rush_adj"]), "recv":float(r["recv_adj"])}

def apply_defense_scaling(player: str, market: str, home_team: str, away_team: str) -> Tuple[float,float]:
    """
    Returns (mu_scale, sd_scale) for the player in this event using his TEAM from CSV,
    so we can pick the correct **opponent** multipliers.
    """
    # Which team is the player on?
    p_team = PLAYER_TO_TEAM.get(player)
    if not p_team:
        return (1.0, 1.0)
    # Figure opponent
    cleaned_home = clean_team(home_team)
    cleaned_away = clean_team(away_team)
    cleaned_player_team = clean_team(p_team)
    if cleaned_player_team == cleaned_home:
        opp = away_team
    elif cleaned_player_team == cleaned_away:
        opp = home_team
    else:
        # couldn’t tell; neutral
        return (1.0, 1.0)

    m = multipliers_for_opponent(opp)
    if market in ("player_pass_tds","player_pass_yds"):
        sc = m["pass"]
    elif market in ("player_rush_yds",):
        sc = m["rush"]
    elif market in ("player_receptions","player_receiving_yards","player_rec_yds"):
        sc = m["recv"]
    else:
        # anytime TD: blend by guess
        sc = 0.5*m["rush"] + 0.5*m["recv"]
    # very mild SD scaling
    return (sc, max(0.9, min(1.1, sc)))

def simulate_from_row(market: str, player: str, side: str, line: float,
                      base_row: pd.Series, home: str, away: str) -> Optional[dict]:
    # get base mu/sd from projections
    if market == "player_pass_tds":
        mu, sd = base_row["mu_pass_tds"], base_row["sd_pass_tds"]
    elif market == "player_pass_yds":
        mu, sd = base_row["mu_pass_yds"], base_row["sd_pass_yds"]
    elif market == "player_rush_yds":
        mu, sd = base_row["mu_rush_yds"], base_row["sd_rush_yds"]
    elif market in ("player_receptions",):
        mu, sd = base_row["mu_receptions"], base_row["sd_receptions"]
    elif market in ("player_receiving_yards","player_rec_yds"):
        mu, sd = base_row["mu_rec_yards"], base_row["sd_rec_yards"]
    else:  # anytime TD (heuristic: use receptions or rush yards proxy already pre-scaled later)
        # basic fallback: convert a usage metric to TD rate (very conservative)
        if "mu_receptions" in base_row:
            mu = 0.10 * float(base_row["mu_receptions"])
        elif "mu_rush_yds" in base_row:
            mu = 0.06 * float(base_row["mu_rush_yds"])
        else:
            return None
        sd = max(0.20, 0.60 * mu)

    # apply defense scaling by opponent
    sc_mu, sc_sd = apply_defense_scaling(player, market, home, away)
    mu *= sc_mu
    sd *= sc_sd
    p_over = norm_over_prob(mu, sd, line, SIM_TRIALS)
    p = p_over if side == "Over" else 1.0 - p_over

    return {
        "market": market, "player": player, "side": side,
        "line": round(line, 3), "mu": round(float(mu), 3), "sd": round(float(sd), 3),
        "prob": round(100*float(p), 2),
        "home": home, "away": away,
        "opp_scale": round(sc_mu, 3),
    }

def collect_book_outcomes(event_json: dict) -> pd.DataFrame:
    rows = []
    for bk in event_json.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = o.get("description")  # player name for player_* markets
                side = o.get("name")
                point = o.get("point")
                if name is None or point is None or side not in ("Over","Under"):
                    continue
                rows.append({"market": mkey, "player_raw": name, "side": side, "point": float(point)})
    if not rows:
        return pd.DataFrame(columns=["market","player_raw","side","point"])
    df = pd.DataFrame(rows).drop_duplicates()
    # normalize receiving yard keys
    df["market"] = df["market"].replace({"player_rec_yds":"player_receiving_yards"})
    return df

def simulate_event(event: dict, markets: List[str], api_key: str, region: str, limit_books: List[str]|None,
                   qb_proj: pd.DataFrame|None, rb_proj: pd.DataFrame|None, wr_proj: pd.DataFrame|None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    home, away, eid = event["home_team"], event["away_team"], event["id"]
    # fetch props (dropping unsupported markets as needed)
    data = fetch_event_with_markets(api_key, eid, region, markets, limit_books)
    df = collect_book_outcomes(data)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Prepare projection name lists
    qb_names = qb_proj["Player"].tolist() if qb_proj is not None else []
    rb_names = rb_proj["Player"].tolist() if rb_proj is not None else []
    wr_names = wr_proj["Player"].tolist() if wr_proj is not None else []

    out_rows, unmatched = [], []
    for _, r in df.iterrows():
        mkt, raw, side, line = r["market"], r["player_raw"], r["side"], r["point"]
        # Choose the correct pool to match
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
            unmatched.append((mkt, raw, score, home, away))
            continue

        row = simulate_from_row(mkt, match, side, line, base, home, away)
        if row is not None:
            out_rows.append(row)

    return pd.DataFrame(out_rows), pd.DataFrame(unmatched, columns=["market","book_player","fuzzy_score","home","away"])

# ------------------------------------------------------------------------------
# Run for ALL games
# ------------------------------------------------------------------------------
st.markdown("### 3) Fetch props for ALL games & simulate")
go_all = st.button("Fetch props for all games & simulate")
if go_all:
    if all(x is None for x in [qb_proj, rb_proj, wr_proj]):
        st.warning("Upload at least one of QB / RB / WR CSVs first.")
        st.stop()

    all_rows, all_unmatched = [], []
    prog = st.progress(0.0)
    for i, ev in enumerate(events):
        try:
            sim, um = simulate_event(ev, MARKETS_ALL, api_key, region, limit_books, qb_proj, rb_proj, wr_proj)
            if not sim.empty:
                # include event meta
                sim["event"] = f'{ev["away_team"]} @ {ev["home_team"]}'
                all_rows.append(sim)
            if not um.empty:
                all_unmatched.append(um)
        except Exception as e:
            st.error(f'Event {ev["away_team"]} vs {ev["home_team"]} fetch failed: {e}')
        prog.progress((i+1)/max(1,len(events)))
        time.sleep(0.05)

    prog.empty()
    if not all_rows:
        st.warning("No props matched your uploaded players across the selected games.")
        st.stop()

    results = pd.concat(all_rows, ignore_index=True).sort_values(["prob","event"], ascending=[False, True])
    st.subheader("Simulated probabilities (all games)")
    st.dataframe(results, use_container_width=True)

    # Download
    st.download_button(
        "⬇️ Download all-game results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="props_sim_results_all_games.csv",
        mime="text/csv",
    )

    # Show unmatched (debug)
    if all_unmatched:
        dbg = pd.concat(all_unmatched, ignore_index=True).sort_values(["market","fuzzy_score"], ascending=[True, False])
        st.markdown("#### Unmatched props (debug)")
        st.dataframe(dbg, use_container_width=True, height=300)

# ------------------------------------------------------------------------------
# Quick single-event runner (optional)
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### (Optional) Run a single event")
labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Event", labels, index=0)
if st.button("Fetch & simulate this event"):
    ev = events[labels.index(pick)]
    sim, um = simulate_event(ev, MARKETS_ALL, api_key, region, limit_books, qb_proj, rb_proj, wr_proj)
    if sim.empty:
        st.warning("No matches for this event.")
    else:
        st.dataframe(sim.sort_values("prob", ascending=False).reset_index(drop=True), use_container_width=True)
        st.download_button(
            "⬇️ Download event results CSV",
            sim.to_csv(index=False).encode("utf-8"),
            file_name="props_sim_results_event.csv",
            mime="text/csv",
        )
    if not um.empty:
        st.markdown("#### Unmatched for this event (debug)")
        st.dataframe(um.sort_values("fuzzy_score", ascending=False), use_container_width=True, height=240)
