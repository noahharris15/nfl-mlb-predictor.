# app_cfb.py — College Football (auto from CFBD /games)
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="College Football — stats from CFBD", layout="wide")
st.title("🏈🎓 College Football — 2025 (auto from CFBD)")

# ------------------ Config ------------------
BASE_URL = "https://api.collegefootballdata.com"
HOME_EDGE_CFB = 1.8          # small home bump in expected points
SIM_TRIALS = 10000
TIMEOUT = 25

def _poisson_sim(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    p_home = float((h > a).mean())
    ties = (h == a)
    if ties.any():
        # slight home edge on ties
        p_home = float(((h > a) | (ties & (np.random.rand(ties.size) < 0.55))).mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

# ------------------ CFBD key ------------------
def _get_cfbd_key() -> str | None:
    # prefer Streamlit secrets; fall back to env var if you set it locally
    key = st.secrets.get("CFBD_API_KEY") if "CFBD_API_KEY" in st.secrets else os.getenv("CFBD_API_KEY")
    return (key or "").strip() or None

CFBD_KEY = _get_cfbd_key()
if not CFBD_KEY:
    st.error("Add your CFBD key in **Secrets** as `CFBD_API_KEY = \"...\"`.")
    st.stop()

headers = {"Authorization": f"Bearer {CFBD_KEY}"}

# ------------------ Data pull (from /games) ------------------
@st.cache_data(show_spinner=True, ttl=60 * 30)  # cache 30 minutes
def cfb_team_stats_from_games(year: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Build per-team Off/Def PPG using the CFBD /games endpoint.
    Works even when /stats endpoints are empty.
    """
    url = f"{BASE_URL}/games"
    params = {"year": year, "seasonType": season_type}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        games = r.json()
    except Exception as e:
        return pd.DataFrame({"__error__": [f"CFBD request failed: {e}"]})

    # No games -> return empty
    if not games:
        return pd.DataFrame()

    buckets: dict[str, dict[str, list[float]]] = {}
    for g in games:
        hp, ap = g.get("home_points"), g.get("away_points")
        if hp is None or ap is None:
            # skip unfinished or missing-score games
            continue
        h, a = g.get("home_team"), g.get("away_team")
        if not h or not a:
            continue
        for team, pts_for, pts_against in [(h, hp, ap), (a, ap, hp)]:
            if team not in buckets:
                buckets[team] = {"pf": [], "pa": []}
            buckets[team]["pf"].append(float(pts_for))
            buckets[team]["pa"].append(float(pts_against))

    if not buckets:
        return pd.DataFrame()

    rows = []
    for team, d in buckets.items():
        rows.append({
            "team": team,
            "off_ppg": float(np.mean(d["pf"])),
            "def_ppg": float(np.mean(d["pa"])),
            "games": int(len(d["pf"])),
        })
    df = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    return df

# ------------------ UI ------------------
c1, c2, c3 = st.columns([1,1,2])
with c1:
    year = st.number_input("Season", value=2025, step=1, min_value=2013, max_value=2035)
with c2:
    season_type = st.selectbox("Season type", ["regular", "postseason"])

df = cfb_team_stats_from_games(int(year), season_type)

# Show helpful diagnostics
with st.expander("Debug (CFBD request)"):
    if "__error__" in df.columns:
        st.error(df["__error__"].iloc[0])
    elif df.empty:
        st.warning("No games returned for this request.")
    else:
        st.write(f"Fetched **{len(df)} teams** with scoring data (min 1 finished game).")
        st.dataframe(df)

# Require data for simulation
if df.empty or "__error__" in df.columns:
    st.stop()

teams = df["team"].tolist()
h_team = st.selectbox("Home team", teams, index=0)
a_team = st.selectbox("Away team", teams, index=min(1, len(teams)-1))

H = df.loc[df["team"] == h_team].iloc[0]
A = df.loc[df["team"] == a_team].iloc[0]

# Expected points using Off PPG vs Opp Def PPG (simple blend) + home bump
mu_home = (float(H["off_ppg"]) + float(A["def_ppg"])) / 2.0 + HOME_EDGE_CFB
mu_away = (float(A["off_ppg"]) + float(H["def_ppg"])) / 2.0

pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)

st.markdown(
    f"**{h_team}** vs **{a_team}** — Expected points: {mH:.1f}–{mA:.1f} · "
    f"P({h_team} win) = **{100*pH:.1f}%**, P({a_team} win) = **{100*pA:.1f}%**"
)
