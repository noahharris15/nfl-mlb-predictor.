# app.py — College Football (2025, stats-only) — auto from CFBD
# - Reads CFBD_API_KEY from Streamlit Secrets
# - Shows diagnostics (key preview + HTTP status/details)
# - Caches responses (30 min)
# - Falls back to CSV upload (team, off_ppg, def_ppg) if API fails

from __future__ import annotations
import json
from typing import Tuple
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------- App config --------------------------------------
st.set_page_config(page_title="College Football — 2025 (auto from CFBD)", layout="wide")
st.title("🏈🎓 College Football — 2025 (auto from CFBD)")

CFB_SEASON = 2025
EPS = 1e-9

# ---------------------------- Helpers -----------------------------------------
def _cfbd_key_info() -> Tuple[str, bool, str]:
    """Return (key, has_key, safe_preview)."""
    try:
        key = st.secrets.get("CFBD_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        return "", False, "No key in st.secrets"
    preview = f"{key[:3]}…{key[-3:]} (len={len(key)})"
    return key, True, preview

@st.cache_data(ttl=1800, show_spinner=False)  # cache 30 min
def _cfbd_request(path: str, params: dict | None = None) -> tuple[int, dict | list | str, dict]:
    """GET CFBD path → (status_code, json_or_text, headers)."""
    base = "https://api.collegefootballdata.com"
    key = st.secrets.get("CFBD_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": "streamlit-cfb-ppg/1.0",
    }
    url = f"{base}{path}"
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=15)
        ct = r.headers.get("Content-Type", "")
        if "application/json" in ct:
            try:
                return r.status_code, r.json(), dict(r.headers)
            except Exception:
                return r.status_code, {"parse_error": True, "text": r.text}, dict(r.headers)
        else:
            return r.status_code, r.text, dict(r.headers)
    except requests.RequestException as e:
        return 0, {"error": str(e)}, {}

@st.cache_data(ttl=1800, show_spinner=False)
def cfb_team_stats_2025() -> pd.DataFrame:
    """
    Build a simple team table with off_ppg and def_ppg for the season
    using CFBD /stats/season (teamType=both to include FBS+FCS).
    """
    code_off, off, hdr_off = _cfbd_request(
        "/stats/season", {"year": CFB_SEASON, "teamType": "both", "side": "offense"}
    )
    code_def, deff, hdr_def = _cfbd_request(
        "/stats/season", {"year": CFB_SEASON, "teamType": "both", "side": "defense"}
    )

    if code_off != 200 or code_def != 200:
        raise RuntimeError(json.dumps({
            "off_status": code_off, "def_status": code_def,
            "off_sample": (off[:1] if isinstance(off, list) else off),
            "def_sample": (deff[:1] if isinstance(deff, list) else deff),
            "off_headers": hdr_off, "def_headers": hdr_def,
        }, default=str))

    def _ppg_from_rows(rows):
        out = {}
        for r in rows or []:
            team = r.get("team")
            # ppg may appear as flat or nested depending on CFBD endpoint version
            ppg = (r.get("pointsPerGame")
                   or r.get("stat", {}).get("pointsPerGame")
                   or r.get("ppa", {}).get("pointsPerGame")
                   or r.get("scoring", {}).get("pointsPerGame")
                   or r.get("scoring", {}).get("ptsPerGame"))
            if team and ppg is not None:
                try:
                    out[team] = float(ppg)
                except Exception:
                    pass
        return out

    off_map = _ppg_from_rows(off if isinstance(off, list) else [])
    def_map = _ppg_from_rows(deff if isinstance(deff, list) else [])

    teams = sorted(set(off_map) | set(def_map))
    if not teams:
        raise RuntimeError("CFBD returned empty team list (after parse).")

    df = pd.DataFrame({
        "team": teams,
        "off_ppg": [off_map.get(t, 28.0) for t in teams],
        "def_ppg": [def_map.get(t, 28.0) for t in teams],
    })
    # Gentle shrink toward 28 to stabilize early-season noise
    df["off_ppg"] = 0.9*df["off_ppg"] + 0.1*28.0
    df["def_ppg"] = 0.9*df["def_ppg"] + 0.1*28.0
    return df

def _poisson_sim(mu_home: float, mu_away: float, trials: int = 10000):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # tiny tiebreak toward home
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

def _cfb_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> tuple[float, float]:
    rH = rates.loc[rates["team"] == home]
    rA = rates.loc[rates["team"] == away]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown CFB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["off_ppg"] + A["def_ppg"]) / 2.0)
    mu_away = max(EPS, (A["off_ppg"] + H["def_ppg"]) / 2.0)
    return mu_home, mu_away

# ---------------------------- Diagnostics -------------------------------------
key, has_key, key_preview = _cfbd_key_info()
with st.expander("Diagnostics", expanded=False):
    st.write(f"Key present: **{has_key}**")
    if has_key:
        st.write(f"Key preview: `{key_preview}`")
    if st.button("Clear CFB cache"):
        st.cache_data.clear()
        st.success("Cleared CFB cache. Re-run the app.")

# ---------------------------- Load data / Fallback ----------------------------
if not has_key:
    st.error("No `CFBD_API_KEY` found in Secrets. Add it in your app’s Secrets and reload.")
    st.stop()

try:
    rates = cfb_team_stats_2025()
except Exception as e:
    st.error("CFBD request failed or returned no data. Details below.")
    with st.expander("Error details (for debugging)"):
        st.code(str(e))
    st.info("You can still use a CSV fallback with columns: team, off_ppg, def_ppg.")
    up = st.file_uploader("Upload CFB CSV fallback", type=["csv","xlsx"])
    if up is None:
        st.stop()
    df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    need = {"team", "off_ppg", "def_ppg"}
    lowcols = {c.lower(): c for c in df.columns}
    if not need.issubset(lowcols.keys()):
        st.error("CSV must contain: team, off_ppg, def_ppg.")
        st.stop()
    rates = df.rename(columns={
        lowcols["team"]: "team",
        lowcols["off_ppg"]: "off_ppg",
        lowcols["def_ppg"]: "def_ppg",
    })[["team", "off_ppg", "def_ppg"]]

# ---------------------------- UI ----------------------------------------------
home = st.selectbox("Home team", sorted(rates["team"].unique().tolist()))
away = st.selectbox("Away team", sorted([t for t in rates["team"].unique().tolist() if t != home]))

mu_h, mu_a = _cfb_matchup_mu(rates, home, away)
pH, pA, mH, mA = _poisson_sim(mu_h, mu_a)
st.markdown(
    f"**{home}** vs **{away}** — Expected points: {mH:.1f}–{mA:.1f} · "
    f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
)

with st.expander("Show team table"):
    st.dataframe(rates.sort_values("team").reset_index(drop=True))
