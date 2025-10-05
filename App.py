# Player Props Simulator — Odds API (event endpoint) + Your CSVs
# Single page; embedded 2025 defense EPA scalers
# Run: streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO
from typing import List, Optional
from rapidfuzz import process, fuzz
import re

st.set_page_config(page_title="NFL Player Props (Odds API + CSV + Defense EPA)", layout="wide")
st.title("📈 NFL Player Props — Odds API + Your CSVs (defense EPA embedded)")

SIM_TRIALS = 10000

# ---- You asked for these FIVE props (receiving YARDS excluded on purpose) ----
# We'll accept either 'player_anytime_td' or 'player_tds' for anytime TDs.
PRIMARY_ANYTIME_KEY = "player_anytime_td"
ALT_ANYTIME_KEY     = "player_tds"

VALID_MARKETS = [
    "player_pass_tds",
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    PRIMARY_ANYTIME_KEY,  # if API doesn’t support this, we’ll also look for ALT_ANYTIME_KEY
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

    return pd.DataFrame({
        "Team": df["Team"].astype(str),
        "pass_adj": pass_adj.astype(float),
        "rush_adj": rush_adj.astype(float),
        "recv_adj": recv_adj.astype(float),
    })

DEF_TABLE = load_defense_table()
st.caption("Defense multipliers (1.0 = neutral) are embedded from your 2025 EPA sheet.")

# ---------------- Helpers ----------------
def _load_any_csv(uploaded) -> pd.DataFrame:
    """Robust loader for PFR-style CSVs that sometimes repeat headers."""
    raw = uploaded.read().decode("utf-8", errors="ignore")
    lines = [ln for ln in raw.splitlines() if ln.strip()]

    header_idx = 0
    for i, ln in enumerate(lines[:30]):
        l = ln.lower()
        if l.startswith("rk,player") or l.startswith("player,"):
            header_idx = i
            break

    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _coerce_numeric(df: pd.DataFrame, bases: List[str]) -> pd.DataFrame:
    if df is None:
        return df
    cols = list(df.columns)
    for base in bases:
        for c in cols:
            if c == base or c.startswith(base + "."):
                df[c] = (
                    df[c].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                )
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Pick best column among duplicates; prefer the last repeated block (e.g., Y/G.1)."""
    if df is None:
        return None
    cols = list(df.columns)
    for base in candidates:
        options = [c for c in cols if c == base or re.fullmatch(re.escape(base) + r"\.\d+", c)]
        if options:
            return sorted(options)[-1]
    return None

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

if qb_df is not None: qb_df = _coerce_numeric(qb_df, ["G","Y/G","Yds","TD","Att","Cmp"])
if rb_df is not None: rb_df = _coerce_numeric(rb_df, ["G","Y/G","Yds","TD","Att"])
if wr_df is not None: wr_df = _coerce_numeric(wr_df, ["G","Y/G","Yds","TD","Tgt","Rec"])

# ---------------- Choose opponent defense to apply ----------------
st.markdown("### 2) Choose the opponent defense to apply to your projections")
opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()  # pass_adj, rush_adj, recv_adj

# ---------------- Build defense-adjusted projections ----------------
def qb_proj_from_csv(df: pd.DataFrame, pass_scale: float) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    player_col = pick_col(df, "Player", "player")
    g_col      = pick_col(df, "G")
    ypg_col    = pick_col(df, "Y/G")
    yds_col    = pick_col(df, "Yds")
    td_col     = pick_col(df, "TD")

    rows = []
    for _, r in df.iterrows():
        name = str(r.get(player_col, "")).strip()
        if not name:
            continue
        g = float(r.get(g_col, 1) or 1)

        # Passing TDs / game
        try:
            td_mu = float(r.get(td_col, np.nan)) / max(1.0, g)
        except Exception:
            td_mu = 1.2
        td_mu *= pass_scale
        td_sd = max(0.25, 0.60 * td_mu)

        # Passing yards / game
        py_mu = np.nan
        if ypg_col and pd.notna(r.get(ypg_col)):
            py_mu = float(r.get(ypg_col))
        elif yds_col and pd.notna(r.get(yds_col)):
            py_mu = float(r.get(yds_col)) / max(1.0, g)
        if not np.isfinite(py_mu):
            py_mu = 225.0
        py_mu *= pass_scale
        py_sd = max(18.0, 0.18 * py_mu)

        rows.append({
            "Player": name,
            "mu_pass_tds": td_mu, "sd_pass_tds": td_sd,
            "mu_pass_yds": py_mu, "sd_pass_yds": py_sd,
        })

    return pd.DataFrame(rows)

def rb_proj_from_csv(df: pd.DataFrame, rush_scale: float, anytd_scale: float) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    player_col = pick_col(df, "Player", "player")
    g_col      = pick_col(df, "G")
    ypg_col    = pick_col(df, "Y/G")
    yds_col    = pick_col(df, "Yds")
    td_col     = pick_col(df, "TD")

    rows = []
    for _, r in df.iterrows():
        name = str(r.get(player_col, "")).strip()
        if not name:
            continue
        g = float(r.get(g_col, 1) or 1)

        # Rush yards / game
        ry_mu = np.nan
        if ypg_col and pd.notna(r.get(ypg_col)):
            ry_mu = float(r.get(ypg_col))
        elif yds_col and pd.notna(r.get(yds_col)):
            ry_mu = float(r.get(yds_col)) / max(1.0, g)
        if not np.isfinite(ry_mu):
            ry_mu = 55.0
        ry_mu *= rush_scale
        ry_sd = max(6.0, 0.22 * ry_mu)

        # Anytime TD rate (RB): use TDs/game as lambda for Poisson >=1
        try:
            atd_mu = float(r.get(td_col, np.nan)) / max(1.0, g)
        except Exception:
            atd_mu = 0.35
        atd_mu *= anytd_scale  # modest scaling by mix of rush/recv defenses

        rows.append({
            "Player": name,
            "mu_rush_yds": ry_mu, "sd_rush_yds": ry_sd,
            "mu_anytd": atd_mu
        })

    return pd.DataFrame(rows)

def wr_proj_from_csv(df: pd.DataFrame, recv_scale: float, anytd_scale: float) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    player_col = pick_col(df, "Player", "player")
    g_col      = pick_col(df, "G")
    rec_col    = pick_col(df, "Rec")
    td_col     = pick_col(df, "TD")

    rows = []
    for _, r in df.iterrows():
        name = str(r.get(player_col, "")).strip()
        if not name:
            continue
        g = float(r.get(g_col, 1) or 1)

        # Receptions / game
        rec_mu = np.nan
        if rec_col and pd.notna(r.get(rec_col)):
            rec_mu = float(r.get(rec_col)) / max(1.0, g)
        if not np.isfinite(rec_mu):
            rec_mu = 4.5
        rec_mu *= recv_scale
        rec_sd = max(1.0, 0.45 * rec_mu)

        # Anytime TD rate (WR): TDs/game
        try:
            atd_mu = float(r.get(td_col, np.nan)) / max(1.0, g)
        except Exception:
            atd_mu = 0.25
        atd_mu *= anytd_scale

        rows.append({
            "Player": name,
            "mu_receptions": rec_mu, "sd_receptions": rec_sd,
            "mu_anytd": atd_mu
        })

    return pd.DataFrame(rows)

anytd_scale = 0.5 * scalers["rush_adj"] + 0.5 * scalers["recv_adj"]
qb_proj = qb_proj_from_csv(qb_df, scalers["pass_adj"])
rb_proj = rb_proj_from_csv(rb_df, scalers["rush_adj"], anytd_scale)
wr_proj = wr_proj_from_csv(wr_df, scalers["recv_adj"], anytd_scale)

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

    # Aggregate bookmaker outcomes → one row per (market, player, point, side)
    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            # Accept either primary or alternate key for anytime TDs
            if mkey == ALT_ANYTIME_KEY and PRIMARY_ANYTIME_KEY not in markets:
                mkey_out = ALT_ANYTIME_KEY
            else:
                mkey_out = mkey

            for o in m.get("outcomes", []):
                name = o.get("description")  # player name for player_* markets
                side = o.get("name")         # "Over" or "Under" (Yes/No for anytime show as Over/Under 0.5)
                point = o.get("point")
                if mkey_out not in (VALID_MARKETS + [ALT_ANYTIME_KEY]):
                    continue
                if name is None or side not in ("Over","Under"):
                    continue
                if point is None:
                    # Some books omit 'point' for anytime TD; default to 0.5
                    point = 0.5
                rows.append({
                    "market": mkey_out,
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

    # Build anytime map (RB + WR; QBs not included because ‘anytime’ excludes passing TDs)
    any_map = {}
    if rb_proj is not None:
        any_map.update({n: float(mu) for n, mu in zip(rb_proj["Player"], rb_proj["mu_anytd"])})
    if wr_proj is not None:
        for n, mu in zip(wr_proj["Player"], wr_proj["mu_anytd"]):
            any_map[n] = float(mu)

    for _, r in df.iterrows():
        market = r["market"]; player = r["player_raw"]; point = r["point"]; side = r["side"]

        # Passing TDs
        if market == "player_pass_tds" and qb_proj is not None:
            match = fuzzy_pick(player, qb_names, cutoff=82)
            if not match: 
                continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_pass_tds"]), float(row["sd_pass_tds"])
            p_over = norm_over_prob(mu, sd, point, SIM_TRIALS)
        # Passing yards
        elif market == "player_pass_yds" and qb_proj is not None:
            match = fuzzy_pick(player, qb_names, cutoff=82)
            if not match: 
                continue
            row = qb_proj.loc[qb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_pass_yds"]), float(row["sd_pass_yds"])
            p_over = norm_over_prob(mu, sd, point, SIM_TRIALS)
        # Rushing yards
        elif market == "player_rush_yds" and rb_proj is not None:
            match = fuzzy_pick(player, rb_names, cutoff=82)
            if not match: 
                continue
            row = rb_proj.loc[rb_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_rush_yds"]), float(row["sd_rush_yds"])
            p_over = norm_over_prob(mu, sd, point, SIM_TRIALS)
        # Receptions
        elif market == "player_receptions" and wr_proj is not None:
            match = fuzzy_pick(player, wr_names, cutoff=82)
            if not match: 
                continue
            row = wr_proj.loc[wr_proj["Player"] == match].iloc[0]
            mu, sd = float(row["mu_receptions"]), float(row["sd_receptions"])
            p_over = norm_over_prob(mu, sd, point, SIM_TRIALS)
        # Anytime TD (either key)
        elif market in (PRIMARY_ANYTIME_KEY, ALT_ANYTIME_KEY) and any_map:
            # Try RB names first, then WR names, then union
            match = (fuzzy_pick(player, list(any_map.keys()), cutoff=82) if any_map else None)
            if not match:
                continue
            lam = max(0.01, float(any_map[match]))  # lambda for Poisson
            # P(score ≥ 1 TD) = 1 - e^{-λ}
            p_any = 1.0 - np.exp(-lam)
            # market is framed like Over/Under 0.5 TDs
            p_over = p_any
        else:
            continue

        p = p_over if side == "Over" else (1.0 - p_over)

        out_rows.append({
            "market": market,
            "player": match,
            "side": side,
            "line": round(point, 3),
            "mu": round(mu if market not in (PRIMARY_ANYTIME_KEY, ALT_ANYTIME_KEY) else lam, 3),
            "sd": round(sd, 3) if market not in (PRIMARY_ANYTIME_KEY, ALT_ANYTIME_KEY) else None,
            "prob": round(100*p, 2),
            "opp_def": opp_team,
            "pass_adj": round(scalers["pass_adj"], 3),
            "rush_adj": round(scalers["rush_adj"], 3),
            "recv_adj": round(scalers["recv_adj"], 3),
        })

    if not out_rows:
        st.warning("No props matched your uploaded players.")
        st.stop()

    results = pd.DataFrame(out_rows).sort_values(["prob","market"], ascending=[False, True]).reset_index(drop=True)
    st.subheader("Simulated probabilities (conservative normal / Poisson for Anytime TD)")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="props_sim_results.csv",
        mime="text/csv",
    )
