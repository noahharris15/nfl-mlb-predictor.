# Player Props Simulator — Odds API (event endpoint) + Your CSVs
# Single page; embedded 2025 defense EPA scalers
# Run: streamlit run app.py

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
VALID_MARKETS = ["player_pass_tds", "player_rush_yds", "player_receptions"]

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
    # Coerce numeric
    for c in ["EPA_Pass","EPA_Rush","Comp_Pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Convert EPA to multipliers (negative EPA => tougher => <1.0)
    def adj_from_epa(s: pd.Series, scale: float) -> pd.Series:
        x = s.fillna(0.0)
        adj = 1.0 - scale * x
        return adj.clip(0.7, 1.3)
    pass_adj = adj_from_epa(df["EPA_Pass"], 0.8)
    rush_adj = adj_from_epa(df["EPA_Rush"], 0.8)
    comp = df["Comp_Pct"].clip(0.45, 0.80).fillna(df["Comp_Pct"].mean())
    comp_adj = (1.0 + (comp - comp.mean()) * 0.6).clip(0.7, 1.3)
    recv_adj = (0.7 * pass_adj + 0.3 * comp_adj).clip(0.7, 1.3)
    out = pd.DataFrame({
        "Team": df["Team"].astype(str),
        "pass_adj": pass_adj.astype(float),
        "rush_adj": rush_adj.astype(float),
        "recv_adj": recv_adj.astype(float),
    })
    return out

DEF_TABLE = load_defense_table()
st.caption("Defense multipliers (1.0 = neutral) are embedded from your 2025 EPA sheet.")

# ---------------- Helpers ----------------
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

def fuzzy_pick(name: str, candidates: List[str], cutoff=85) -> Optional[str]:
    if not candidates: return None
    res = process.extractOne(name, candidates, scorer=fuzz.token_sort_ratio)
    return res[0] if res and res[1] >= cutoff else None

def norm_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    return float((np.random.normal(mu, sd, size=trials) > line).mean())

# ---------------- Upload CSVs ----------------
st.markdown("### 1) Upload CSVs (any of QB / RB / WR)")
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

# ---------------- Choose opponent defense to apply ----------------
st.markdown("### 2) Choose the opponent defense to apply to your projections")
opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()  # pass_adj, rush_adj, recv_adj

# ---------------- Build defense-adjusted projections ----------------
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
        rows.append({"Player": name, "mu_pass_tds": td_mu, "sd_pass_tds": td_sd})
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
        rec = r.get("Rec", np.nan)
        try: rec_mu = float(rec) / max(1.0, g)
        except Exception: rec_mu = 4.5
        rec_mu *= recv_scale
        rec_sd = max(1.0, 0.45 * rec_mu)
        rows.append({"Player": name, "mu_receptions": rec_mu, "sd_receptions": rec_sd})
    return pd.DataFrame(rows)

qb_proj = qb_proj_from_csv(qb_df, scalers["pass_adj"])
rb_proj = rb_proj_from_csv(rb_df, scalers["rush_adj"])
wr_proj = wr_proj_from_csv(wr_df, scalers["recv_adj"])

if qb_proj is not None: st.dataframe(qb_proj.head(12), use_container_width=True)
if rb_proj is not None: st.dataframe(rb_proj.head(12), use_container_width=True)
if wr_proj is not None: st.dataframe(wr_proj.head(12), use_container_width=True)

# ---------------- Odds API (Event endpoint) ----------------
st.markdown("### 3) Choose an NFL game & markets from The Odds API (event endpoint)")
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
st.markdown("### 4) Fetch props for this event and simulate")
go = st.button("Fetch lines & simulate")
if go:
    if not markets:
        st.warning("Pick at least one market.")
        st.stop()
    if all(x is None for x in [qb_proj, rb_proj, wr_proj]):
        st.warning("Upload at least one of QB / RB / WR CSVs first.")
        st.stop()

    try:
        data = fetch_event_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    # Aggregate bookmaker outcomes → one row per (market, player, point, side) averaged over books
    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = o.get("description")  # player name for player_* markets
                side = o.get("name")         # "Over" or "Under"
                point = o.get("point")
                if mkey not in VALID_MARKETS or name is None or point is None or side not in ("Over","Under"):
                    continue
                rows.append({
                    "market": mkey,
                    "player_raw": name,
                    "side": side,
                    "point": float(point),
                })

    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    df = pd.DataFrame(rows).drop_duplicates()

    # Fuzzy-match to projections, compute probabilities
    out_rows = []
    qb_names = qb_proj["Player"].tolist() if qb_proj is not None else []
    rb_names = rb_proj["Player"].tolist() if rb_proj is not None else []
    wr_names = wr_proj["Player"].tolist() if wr_proj is not None else []

    for _, r in df.iterrows():
        market = r["market"]; player = r["player_raw"]; point = r["point"]; side = r["side"]

        if market == "player_pass_tds" and qb_proj is not None:
            match = fuzzy_pick(player, qb_names, cutoff=82)
            if match:
                row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
                mu, sd = float(row["mu_pass_tds"]), float(row["sd_pass_tds"])
            else: 
                continue
        elif market == "player_rush_yds" and rb_proj is not None:
            match = fuzzy_pick(player, rb_names, cutoff=82)
            if match:
                row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
                mu, sd = float(row["mu_rush_yds"]), float(row["sd_rush_yds"])
            else: 
                continue
        elif market == "player_receptions" and wr_proj is not None:
            match = fuzzy_pick(player, wr_names, cutoff=82)
            if match:
                row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
                mu, sd = float(row["mu_receptions"]), float(row["sd_receptions"])
            else: 
                continue
        else:
            continue

        p_over = norm_over_prob(mu, sd, point, SIM_TRIALS)
        p = p_over if side == "Over" else 1.0 - p_over

        out_rows.append({
            "market": market,
            "player": match,
            "side": side,
            "line": round(point, 3),
            "mu": round(mu, 3),
            "sd": round(sd, 3),
            "prob": round(100*p, 2),
            "opp_def": opp_team,
            "pass_adj": round(scalers["pass_adj"], 3),
            "rush_adj": round(scalers["rush_adj"], 3),
            "recv_adj": round(scalers["recv_adj"], 3),
        })

    if not out_rows:
        st.warning("No props matched your uploaded players.")
        st.stop()

    results = pd.DataFrame(out_rows).sort_values("prob", ascending=False).reset_index(drop=True)
    st.subheader("Simulated probabilities (conservative normal model)")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="props_sim_results.csv",
        mime="text/csv",
    )
