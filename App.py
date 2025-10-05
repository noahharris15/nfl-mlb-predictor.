# Player Props Simulator — Odds API (event endpoint) + Your CSVs or Pasted CSVs
# Single page; embedded 2025 defense EPA scalers
# Run: streamlit run app.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO
from typing import List, Optional, Dict, Tuple
from rapidfuzz import process, fuzz

st.set_page_config(page_title="NFL Player Props (Odds API + CSV + Defense EPA)", layout="wide")
st.title("📈 NFL Player Props — Odds API + Your CSVs (defense EPA embedded)")

SIM_TRIALS = 10000

# ---------------------------------------------------------------------
# ✅ EXACT FIVE MARKETS YOU ASKED FOR
# ---------------------------------------------------------------------
VALID_MARKETS = [
    "player_pass_yds",     # QB passing yards
    "player_rush_yds",     # RB rushing yards
    "player_receptions",   # WR/TE receptions
    "player_anytime_td",   # WR/TE/RB anytime TD
    "player_pass_tds"      # QB passing TDs
]

# ---------------- Embedded 2025 defense EPA (from your sheet) ----------------
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

# ---------------------------------------------------------------------
# ✅ PASTE-IN CSVs (taken from your screenshots) — these take priority
#    Keep only columns we actually need. You can replace the sample rows
#    with your full CSV dump text.
# ---------------------------------------------------------------------
QB_PASTE_HEADER = (
    "Rk,Player,Cmp,Att,Yds,Cmp%,TD,Int,Sk,Rate,Sk%,Y/A,AY/A,ANY/A,Y/C,Y/G,Succ%,W,L,T,4QC,GWD,Pos,-9999"
)
QB_PASTE_SAMPLE = """1,Bryce Young,87,144,753,60.4,5,2,17,77.1,6.5,5.2,4.2,7.0,188.3,42.7,1,3,0,0,0,0,QB,YoungBr01
2,Zach Wilson,35,32,62,5.0,0,0,0,0.0,0.0,4.0,0.0,4.0,20.5,26.0,38.5,5.2,3,0,0,0,0,QB,WilsZa00
3,Russell Wilson,66,111,786,59.5,5,3,2,7.8,96.2,5.3,7.2,7.8,9.6,196.5,39.5,0,3,0,0,0,0,QB,WilsRu00
4,Caleb Williams,81,130,927,62.3,8,3,1,5.1,97.6,5.1,7.1,7.1,9.0,231.8,41.6,2,2,0,1,0,0,QB,WillCa03
5,Carson Wentz,44,66,523,66.7,4,3,0,98.2,12.0,7.9,4.6,7.7,20.7,130.8,29.7,0,3,0,0,0,0,QB,WentCa00
"""

RB_PASTE_HEADER = (
    "Rk,Player,Att,Yds,Y/A,TD,Y/G,1D,Succ%,Season,Age,Team,G,GS,Att,Yds,Y/A,TD,Y/G,1D,Succ%,Pos,-9999"
)
RB_PASTE_SAMPLE = """1,Jacardia Wright,5,20,4.0,0,20.0,1,40.0,2025,25,SEA,1,0,5,20,4.0,0,20.0,1,40.0,RB,WrigJa05
2,Emanuel Wilson,15,73,4.9,0,18.3,4,53.3,2025,26,GNB,4,0,15,73,4.9,0,18.3,4,53.3,RB,WilEma00
3,Kyren Williams,68,303,4.5,1,75.8,19,61.8,2025,25,LAR,4,4,68,303,4.5,1,75.8,19,61.8,RB,WillKy02
4,Javonte Williams,63,312,5.0,4,78.0,19,65.1,2025,25,DAL,4,4,63,312,5.0,4,78.0,19,65.1,RB,WillJa10
5,Zamir White,10,25,2.5,0,8.3,3,20.0,2025,26,LVR,3,0,10,25,2.5,0,8.3,3,20.0,RB,WhitZa01
"""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",","",regex=False).str.replace("%","",regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

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

# ---------------------------------------------------------------------
# Uploads OR Pasted CSVs (pasted takes priority)
# ---------------------------------------------------------------------
st.header("1) QB / RB / WR data")
c1, c2, c3 = st.columns(3)
with c1: qb_file = st.file_uploader("QB CSV (optional if pasting)", type=["csv"])
with c2: rb_file = st.file_uploader("RB CSV (optional if pasting)", type=["csv"])
with c3: wr_file = st.file_uploader("WR/TE CSV (for receptions/anytime)", type=["csv"])

st.subheader("Paste QB CSV (priority)")
qb_paste = st.text_area("Paste QB CSV text here (headers included)", value="\n".join([QB_PASTE_HEADER, QB_PASTE_SAMPLE]), height=170)
st.subheader("Paste RB CSV (priority)")
rb_paste = st.text_area("Paste RB CSV text here (headers included)", value="\n".join([RB_PASTE_HEADER, RB_PASTE_SAMPLE]), height=170)

def parse_csv_from_any(pasted: str, uploaded) -> Optional[pd.DataFrame]:
    pasted = (pasted or "").strip()
    if pasted and "," in pasted and "\n" in pasted:
        try:
            return pd.read_csv(StringIO(pasted))
        except Exception:
            st.warning("Could not parse pasted CSV; falling back to uploader.")
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
    return None

qb_df = parse_csv_from_any(qb_paste, qb_file)
rb_df = parse_csv_from_any(rb_paste, rb_file)
wr_df = pd.read_csv(wr_file) if wr_file else None  # WR still via file (unchanged)

# Normalize types for the few columns we need
if qb_df is not None: qb_df = _coerce_numeric(qb_df, ["Y/G","Yds","TD","G","Att","Cmp"])
if rb_df is not None: rb_df = _coerce_numeric(rb_df, ["Y/G","Yds","TD","G","Att"])
if wr_df is not None: wr_df = _coerce_numeric(wr_df, ["Rec","TD","G"])

# ---------------------------------------------------------------------
# Opponent defense
# ---------------------------------------------------------------------
st.header("2) Opponent defense")
opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()  # pass_adj, rush_adj, recv_adj

# ---------------------------------------------------------------------
# Build projections (only using fields we need)
# ---------------------------------------------------------------------
def qb_proj_from_csv(df: pd.DataFrame, pass_scale: float) -> Optional[pd.DataFrame]:
    if df is None: return None
    out = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player","")))
        g = float(r.get("G", 1) or 1)

        # Passing TDs
        td = r.get("TD", np.nan)
        try: td_mu = float(td) / max(1.0, g)
        except Exception: td_mu = 1.2
        td_mu *= pass_scale
        td_sd = max(0.25, 0.60 * td_mu)

        # Passing Yards
        ypg = r.get("Y/G", np.nan); yds_total = r.get("Yds", np.nan)
        try:
            py_mu = float(ypg) if pd.notna(ypg) else float(yds_total) / max(1.0, g)
        except Exception:
            py_mu = 235.0
        py_mu *= pass_scale
        py_sd = max(25.0, 0.20 * py_mu)

        out.append({"Player": name, "mu_pass_tds": td_mu, "sd_pass_tds": td_sd,
                    "mu_pass_yds": py_mu, "sd_pass_yds": py_sd})
    return pd.DataFrame(out)

def rb_proj_from_csv(df: pd.DataFrame, rush_scale: float) -> Optional[pd.DataFrame]:
    if df is None: return None
    out = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player","")))
        g = float(r.get("G", 1) or 1)

        # Rushing yards
        ypg = r.get("Y/G", np.nan); yds_total = r.get("Yds", np.nan)
        try:
            rush_mu = float(ypg) if pd.notna(ypg) else float(yds_total) / max(1.0, g)
        except Exception:
            # derive from Att and Y/A if present
            try:
                att_total = float(r.get("Att", np.nan))
                ypa = float(r.get("Y/A", np.nan))
                rush_mu = (att_total / max(1.0, g)) * ypa
            except Exception:
                rush_mu = 55.0
        rush_mu *= rush_scale
        rush_sd = max(6.0, 0.22 * rush_mu)

        # Anytime TD lambda from RB TD per game
        rb_td = r.get("TD", np.nan)
        try: td_mu = float(rb_td) / max(1.0, g)
        except Exception: td_mu = 0.45
        td_mu *= rush_scale
        td_mu = max(0.02, td_mu)

        out.append({"Player": name, "mu_rush_yds": rush_mu, "sd_rush_yds": rush_sd,
                    "mu_any_td": td_mu})
    return pd.DataFrame(out)

def wr_proj_from_csv(df: pd.DataFrame, recv_scale: float) -> Optional[pd.DataFrame]:
    if df is None: return None
    out = []
    for _, r in df.iterrows():
        name = str(r.get("Player", r.get("player","")))
        g = float(r.get("G", 1) or 1)
        rec_total = r.get("Rec", np.nan)
        try: rec_mu = float(rec_total) / max(1.0, g)
        except Exception: rec_mu = 4.5
        rec_mu *= recv_scale
        rec_sd = max(1.0, 0.45 * rec_mu)

        wr_td = r.get("TD", np.nan)
        try: td_mu = float(wr_td) / max(1.0, g)
        except Exception: td_mu = 0.35
        td_mu *= recv_scale
        td_mu = max(0.02, td_mu)

        out.append({"Player": name, "mu_receptions": rec_mu, "sd_receptions": rec_sd,
                    "mu_any_td": td_mu})
    return pd.DataFrame(out)

qb_proj = qb_proj_from_csv(qb_df, scalers["pass_adj"])
rb_proj = rb_proj_from_csv(rb_df, scalers["rush_adj"])
wr_proj = wr_proj_from_csv(wr_df, scalers["recv_adj"]) if wr_df is not None else None

c1, c2, c3 = st.columns(3)
with c1:
    if qb_proj is not None: st.dataframe(qb_proj.head(12), use_container_width=True)
with c2:
    if rb_proj is not None: st.dataframe(rb_proj.head(12), use_container_width=True)
with c3:
    if wr_proj is not None: st.dataframe(wr_proj.head(12), use_container_width=True)

# ---------------------------------------------------------------------
# Odds API
# ---------------------------------------------------------------------
st.header("3) Select game & markets (Odds API event endpoint)")
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

# ---------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------
st.header("4) Fetch props for this event and simulate")
go = st.button("Fetch lines & simulate")
if go:
    if not markets:
        st.warning("Pick at least one market."); st.stop()
    if qb_proj is None and rb_proj is None and wr_proj is None:
        st.warning("Provide at least one of QB / RB / WR tables (paste or upload)."); st.stop()

    try:
        data = fetch_event_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}"); st.stop()

    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = o.get("description")
                side = o.get("name")         # "Over"/"Under" or "Yes"/"No"
                point = o.get("point")       # None for anytime TD sometimes
                if mkey not in VALID_MARKETS or name is None or side is None:
                    continue
                rows.append({"market": mkey, "player_raw": name, "side": side,
                            "point": (None if point is None else float(point))})

    if not rows:
        st.warning("No player outcomes returned for selected markets."); st.stop()

    df = pd.DataFrame(rows).drop_duplicates()

    out_rows = []
    qb_names = qb_proj["Player"].tolist() if qb_proj is not None else []
    rb_names = rb_proj["Player"].tolist() if rb_proj is not None else []
    wr_names = wr_proj["Player"].tolist() if wr_proj is not None else []

    for _, r in df.iterrows():
        market = r["market"]; player = r["player_raw"]; point = r["point"]; side = r["side"]

        # QB passing TDs
        if market == "player_pass_tds" and qb_proj is not None:
            match = fuzzy_pick(player, qb_names, cutoff=82)
            if not match: continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_pass_tds"]), float(row["sd_pass_tds"])
            if point is None: continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        # QB passing yards
        elif market == "player_pass_yds" and qb_proj is not None:
            match = fuzzy_pick(player, qb_names, cutoff=82)
            if not match: continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_pass_yds"]), float(row["sd_pass_yds"])
            if point is None: continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        # RB rushing yards
        elif market == "player_rush_yds" and rb_proj is not None:
            match = fuzzy_pick(player, rb_names, cutoff=82)
            if not match: continue
            row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_rush_yds"]), float(row["sd_rush_yds"])
            if point is None: continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        # WR/TE receptions
        elif market == "player_receptions" and wr_proj is not None:
            match = fuzzy_pick(player, wr_names, cutoff=82)
            if not match: continue
            row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_receptions"]), float(row["sd_receptions"])
            if point is None: continue
            p_over = norm_over_prob(mu, sd, float(point), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        # Anytime TD (WR/TE/RB)
        elif market == "player_anytime_td":
            match = None; src = None
            if wr_proj is not None:
                m = fuzzy_pick(player, wr_names, cutoff=82)
                if m: match, src = m, "WR"
            if match is None and rb_proj is not None:
                m = fuzzy_pick(player, rb_names, cutoff=82)
                if m: match, src = m, "RB"
            if match is None: continue
            row = (wr_proj if src == "WR" else rb_proj).loc[(wr_proj if src == "WR" else rb_proj)["Player"] == match].iloc[0]
            lam = float(row["mu_any_td"])
            p_yes = anytime_yes_prob_poisson(lam)
            if side in ("Yes","No"):
                p = p_yes if side == "Yes" else (1.0 - p_yes)
            elif side in ("Over","Under"):   # books that show O/U 0.5
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
            "mu": (None if isinstance(mu, float) and math.isnan(mu) else round(float(mu), 3)),
            "sd": (None if isinstance(sd, float) and math.isnan(sd) else round(float(sd), 3)),
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
