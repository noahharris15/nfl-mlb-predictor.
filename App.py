# app_cfb_only.py — College Football Predictor (CFBD, stats-only)
# Requires Streamlit secret: CFBD_API_KEY = "..."
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import requests
from typing import Tuple, Optional

st.set_page_config(page_title="College Football — Predictor (CFBD)", layout="centered")
st.title("🏈 College Football — Predictor (CFBD)")

# -------------------------------------------------------------
# Config / Secrets
# -------------------------------------------------------------
API_KEY = st.secrets.get("CFBD_API_KEY", "")
if not API_KEY:
    st.error("Add your CFBD API key in Streamlit **Secrets** as `CFBD_API_KEY = \"...\"`.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE = "https://api.collegefootballdata.com"

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _safe_json(resp: requests.Response) -> Optional[object]:
    try:
        return resp.json()
    except Exception:
        return None

def _http_get(url: str, params: dict | None = None) -> Tuple[int, Optional[object], str]:
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
        data = _safe_json(r)
        text = r.text if data is None else ""
        return r.status_code, data, text
    except requests.RequestException as e:
        return 0, None, f"request exception: {e}"

def _pivot_stats_to_ppg(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    df_raw has columns: ['team','statName','statValue', ...]
    We pivot and try to find offense & defense PPG regardless of exact labels.
    """
    if df_raw.empty:
        return pd.DataFrame()

    # Robust column normalization
    cols = {c: c for c in df_raw.columns}
    # Required minimum columns to pivot
    needed = {"team", "statName", "statValue"}
    if not needed.issubset(set(df_raw.columns)):
        return pd.DataFrame()

    # Pivot: each row = team, columns = statName
    pt = pd.pivot_table(
        df_raw,
        index="team",
        columns="statName",
        values="statValue",
        aggfunc="mean",
    )

    # Canonical names used by CFBD today:
    #   - pointsPerGame                  (offense)
    #   - opponentPointsPerGame         (defense)
    # Sometimes names vary; add aliases here if CFBD changes again:
    OFF_KEYS = [
        "pointsPerGame",
        "Points Per Game",
        "points_per_game",
        "points per game",
    ]
    DEF_KEYS = [
        "opponentPointsPerGame",
        "Opponent Points Per Game",
        "opponent_points_per_game",
        "opponent points per game",
    ]

    def _first_present(candidates: list[str]) -> Optional[str]:
        for k in candidates:
            if k in pt.columns:
                return k
        return None

    off_key = _first_present(OFF_KEYS)
    def_key = _first_present(DEF_KEYS)

    # If still missing, try to infer by looking for the “opponent” pattern
    if off_key is None:
        for c in pt.columns:
            if "points" in str(c).lower() and "opponent" not in str(c).lower():
                off_key = c
                break
    if def_key is None:
        for c in pt.columns:
            if "points" in str(c).lower() and "opponent" in str(c).lower():
                def_key = c
                break

    if off_key is None or def_key is None:
        # Feed back what the API actually returned so you can see the names
        st.error("❌ Could not locate PPG columns in CFBD data.")
        with st.expander("Show available stat columns returned by CFBD"):
            st.write(sorted([str(c) for c in pt.columns]))
        return pd.DataFrame()

    out = pd.DataFrame(index=pt.index).assign(
        off_ppg=pd.to_numeric(pt[off_key], errors="coerce"),
        def_ppg=pd.to_numeric(pt[def_key], errors="coerce"),
    ).reset_index(names="team")

    out = out.dropna(subset=["off_ppg", "def_ppg"])
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def get_cfb_team_stats(season: int) -> pd.DataFrame:
    """
    Primary: CFBD /stats/season (provides offense & defense points per game).
    Falls back to empty DataFrame if missing.
    """
    url = f"{BASE}/stats/season"
    code, data, raw = _http_get(url, params={"year": season})

    if code != 200 or data is None:
        if code == 200 and data is None:
            st.error("❌ JSON decode failed from CFBD.")
        else:
            st.error(f"CFBD error: HTTP {code}.")
            if raw:
                with st.expander("Show raw response text"):
                    st.code(raw)
        return pd.DataFrame()

    if not isinstance(data, list) or len(data) == 0:
        st.warning(f"No data returned for {season}.")
        return pd.DataFrame()

    df_raw = pd.DataFrame(data)
    # Keep only the columns we need if present
    keep_cols = [c for c in ["team", "statName", "statValue", "season", "category"] if c in df_raw.columns]
    df_raw = df_raw[keep_cols].copy()

    # Ensure numeric
    if "statValue" in df_raw.columns:
        df_raw["statValue"] = pd.to_numeric(df_raw["statValue"], errors="coerce")

    df_ppg = _pivot_stats_to_ppg(df_raw)
    return df_ppg

def simulate_matchup(home: str, away: str, df_ppg: pd.DataFrame, trials: int = 10000):
    H = df_ppg.loc[df_ppg["team"] == home]
    A = df_ppg.loc[df_ppg["team"] == away]
    if H.empty or A.empty:
        raise ValueError("Selected teams not found in the dataset.")
    h, a = H.iloc[0], A.iloc[0]
    mu_home = max(0.1, (h["off_ppg"] + a["def_ppg"]) / 2.0 + 0.4)  # small home tilt
    mu_away = max(0.1, (a["off_ppg"] + h["def_ppg"]) / 2.0)

    home_scores = np.random.poisson(mu_home, trials)
    away_scores = np.random.poisson(mu_away, trials)

    p_home = float((home_scores > away_scores).mean())
    p_away = float((away_scores > home_scores).mean())
    return float(home_scores.mean()), float(away_scores.mean()), p_home, p_away

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.subheader("Select Season & Teams")
year = st.number_input("Season", min_value=2018, max_value=2025, value=2024, step=1)

with st.spinner("Loading CFBD team PPG…"):
    df = get_cfb_team_stats(int(year))

if df.empty:
    st.warning("No CFBD team PPG available. You can upload a CSV fallback below.")
    up = st.file_uploader("Upload CFB CSV fallback (columns: team, off_ppg, def_ppg)", type=["csv"])
    if up:
        try:
            df = pd.read_csv(up)
            # basic sanity check
            need = {"team", "off_ppg", "def_ppg"}
            if not need.issubset(set(df.columns)):
                st.error("CSV missing required columns: team, off_ppg, def_ppg")
                st.stop()
            df["off_ppg"] = pd.to_numeric(df["off_ppg"], errors="coerce")
            df["def_ppg"] = pd.to_numeric(df["def_ppg"], errors="coerce")
            df = df.dropna(subset=["team","off_ppg","def_ppg"])
        except Exception as e:
            st.error(f"Couldn't read CSV: {e}")
            st.stop()
    else:
        st.stop()

teams = sorted(df["team"].unique().tolist())
home = st.selectbox("🏠 Home team", teams, index=0 if teams else None)
away = st.selectbox("✈️ Away team", [t for t in teams if t != home], index=1 if len(teams) > 1 else 0)

if home and away:
    try:
        mH, mA, pH, pA = simulate_matchup(home, away, df)
        st.markdown(
            f"### {home} vs {away}\n"
            f"**Expected points:** {mH:.1f} – {mA:.1f}  \n"
            f"**Win prob:** {home} **{100*pH:.1f}%** · {away} **{100*pA:.1f}%**"
        )
    except Exception as e:
        st.error(str(e))

with st.expander("📊 Show team table used"):
    if not df.empty:
        st.dataframe(df.sort_values("off_ppg", ascending=False).reset_index(drop=True))
