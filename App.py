# app_cfb.py — College Football (stats only) from CFBD + Poisson sim
# Requirements (add to requirements.txt): streamlit requests pandas numpy
# In Streamlit "Secrets", set: CFBD_API_KEY = "your_key_here"

from __future__ import annotations
import math
from typing import Tuple, Optional
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------ UI config ------------------------
st.set_page_config(page_title="College Football — (auto from CFBD)", layout="wide")

st.title("🏈🎓 College Football — 2025 (auto from CFBD)")

st.caption(
    "This page fetches current-season **team scoring** from CollegeFootballData and "
    "runs a simple Poisson simulation for win probabilities. "
    "If the API is unavailable or rate-limited, you can upload a CSV fallback "
    "with columns: `team, off_ppg, def_ppg`."
)

# ------------------------ constants ------------------------
SIM_TRIALS = 10000
EPS = 1e-9

# ------------------------ helpers --------------------------
def poisson_sim(mu_home: float, mu_away: float, trials: int = SIM_TRIALS) -> Tuple[float, float, float, float]:
    """Return (p_home, p_away, mean_home, mean_away)."""
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        # tiny home OT edge, college-ish
        wins_home[ties] = 0.52
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

def _api_key_ok() -> bool:
    try:
        key = st.secrets.get("CFBD_API_KEY", "")
        return isinstance(key, str) and len(key.strip()) >= 8
    except Exception:
        return False

def _headers() -> dict:
    return {"Authorization": f"Bearer {st.secrets['CFBD_API_KEY']}"}

@st.cache_data(show_spinner=False, ttl=900)
def cfb_team_ppg_from_games(year: int, start_week: Optional[int], end_week: Optional[int]) -> Tuple[pd.DataFrame, dict]:
    """
    Build team Off PPG and Def PPG by aggregating /games/teams endpoint.
    Returns (df, meta) where df has columns [team, off_ppg, def_ppg].
    """
    params = {"year": year}
    if start_week: params["startWeek"] = int(start_week)
    if end_week: params["endWeek"] = int(end_week)

    url = "https://api.collegefootballdata.com/games/teams"
    meta = {"url": url, "params": params}

    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    meta["status"] = resp.status_code

    # Try to detect non-JSON (e.g., HTML error page)
    try:
        data = resp.json()
    except Exception:
        meta["error"] = f"Non-JSON response (first 200 chars): {resp.text[:200]!r}"
        return pd.DataFrame(), meta

    if resp.status_code != 200:
        meta["error"] = f"HTTP {resp.status_code}: {data}"
        return pd.DataFrame(), meta

    # data is a list of team-game dicts. Each has 'team', 'points', 'opponent', 'opponentPoints'
    rows = []
    for g in data:
        t = str(g.get("team", "")).strip()
        pts_for = g.get("points", None)
        pts_against = g.get("opponentPoints", None)
        if not t or pts_for is None or pts_against is None:
            continue
        try:
            rows.append({"team": t, "pts_for": float(pts_for), "pts_against": float(pts_against)})
        except Exception:
            continue

    if not rows:
        meta["error"] = "returned empty team list (after parse)."
        return pd.DataFrame(), meta

    games = pd.DataFrame(rows)
    agg = games.groupby("team", as_index=False).agg(
        off_ppg=("pts_for", "mean"),
        def_ppg=("pts_against", "mean"),
        games=("pts_for", "size"),
        off_pts=("pts_for", "sum"),
        def_pts=("pts_against", "sum"),
    ).sort_values("team").reset_index(drop=True)

    # Light shrink toward league average if few games
    if not agg.empty:
        league_for = float(agg["off_ppg"].mean())
        league_against = float(agg["def_ppg"].mean())
        w = np.clip(1.0 - agg["games"] / 4.0, 0.0, 1.0)
        agg["off_ppg"] = (1 - w) * agg["off_ppg"] + w * league_for
        agg["def_ppg"] = (1 - w) * agg["def_ppg"] + w * league_against

    return agg[["team", "off_ppg", "def_ppg"]], meta

def _csv_fallback_uploader() -> pd.DataFrame:
    up = st.file_uploader("Upload CFB CSV fallback (team, off_ppg, def_ppg)", type=["csv", "xlsx"])
    if up is None:
        return pd.DataFrame()
    name = up.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(up)
    else:
        df = pd.read_csv(up)
    cols = {c.strip().lower(): c for c in df.columns}
    tcol = next((cols[k] for k in ("team", "school") if k in cols), None)
    ofc = next((cols[k] for k in ("off_ppg","offense_ppg","points_for_pg") if k in cols), None)
    dfc = next((cols[k] for k in ("def_ppg","defense_ppg","points_against_pg") if k in cols), None)
    if not (tcol and ofc and dfc):
        st.error("CSV needs columns: team, off_ppg, def_ppg (names are flexible).")
        return pd.DataFrame()
    out = df[[tcol, ofc, dfc]].copy()
    out.columns = ["team", "off_ppg", "def_ppg"]
    return out

# ------------------------ controls -------------------------
year_default = datetime.utcnow().year
# If new season hasn't fully populated, 2024 is safer than 2025 early in season
year = st.number_input("Season year", min_value=2014, max_value=2100, value=max(2024, year_default), step=1)
c1, c2, c3 = st.columns([1,1,2])
with c1:
    start_week = st.number_input("Start week (optional)", min_value=1, max_value=20, value=1, step=1)
with c2:
    end_week = st.number_input("End week (optional)", min_value=1, max_value=20, value=start_week, step=1)

st.divider()

# ------------------------ data fetch -----------------------
df = pd.DataFrame()
meta = {}

if _api_key_ok():
    with st.spinner("Fetching team scoring from CFBD…"):
        df, meta = cfb_team_ppg_from_games(int(year), int(start_week) if start_week else None, int(end_week) if end_week else None)

    with st.expander("Debug (CFBD request)"):
        st.write({"key_present": True, "url": meta.get("url"), "params": meta.get("params"),
                  "status": meta.get("status"), "error": meta.get("error")})

    if meta.get("error"):
        st.error("CFBD request failed or returned no data. You can still use a CSV fallback.")
        df_csv = _csv_fallback_uploader()
        if not df_csv.empty:
            df = df_csv.copy()
else:
    st.warning("CFBD_API_KEY not found in Secrets. Using CSV fallback.")
    df_csv = _csv_fallback_uploader()
    if not df_csv.empty:
        df = df_csv.copy()

if df.empty:
    st.stop()

# ------------------------ matchup UI -----------------------
teams = df["team"].tolist()
h = st.selectbox("Home team", teams, index=0)
a = st.selectbox("Away team", [t for t in teams if t != h], index=min(1, len(teams)-1))

H = df.loc[df["team"] == h].iloc[0]
A = df.loc[df["team"] == a].iloc[0]

mu_home = max(EPS, (H["off_ppg"] + A["def_ppg"]) / 2.0)
mu_away = max(EPS, (A["off_ppg"] + H["def_ppg"]) / 2.0)

pH, pA, mH, mA = poisson_sim(mu_home, mu_away)

st.markdown(
    f"**{h}** vs **{a}** — Expected points: **{mH:.1f}–{mA:.1f}**  "
    f"· P({h} win) = **{100*pH:.1f}%**, P({a} win) = **{100*pA:.1f}%**"
)

with st.expander("Show team table"):
    st.dataframe(df.sort_values("team").reset_index(drop=True))
