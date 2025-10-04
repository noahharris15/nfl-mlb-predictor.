# cfb_only_app.py — College Football (2025) — auto from CFBD + Poisson sim

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------- Config -------------------------------------------
YEAR = 2025
SEASON_TYPE = "regular"     # "regular" or "postseason"
HOME_EDGE = 1.8             # small points bump to home mean
TRIALS = 10000
EPS = 1e-9

# ------------------------ Secrets / API headers -------------------------------
def _get_cfbd_key() -> str | None:
    try:
        k = st.secrets.get("CFBD_API_KEY", "").strip()
        return k or None
    except Exception:
        return None

def _cfbd_headers() -> dict:
    key = _get_cfbd_key()
    return {"Authorization": f"Bearer {key}"} if key else {}

BASE = "https://api.collegefootballdata.com"

# ----------------------------- Fetchers ---------------------------------------
@st.cache_data(show_spinner=True)
def fetch_stats_season(year: int, season_type: str) -> pd.DataFrame:
    """
    Hit CFBD /stats/season and return a normalized DataFrame.
    CFBD sometimes returns:
      - a flat list of rows with 'team','conference','statName','statValue','category'
      - or nested dicts per team (rare in older mirrors)
    We normalize and then mine PPG fields by category.
    """
    url = f"{BASE}/stats/season"
    r = requests.get(url, headers=_cfbd_headers(), params={"year": year, "seasonType": season_type}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"CFBD {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not data:
        # Return explicit empty frame; caller will handle
        return pd.DataFrame()
    df = pd.json_normalize(data)
    return df

@st.cache_data(show_spinner=True)
def fetch_teams_list(year: int) -> list[str]:
    """
    Get a canonical list of FBS team names for selection dropdowns.
    """
    url = f"{BASE}/teams/fbs"
    r = requests.get(url, headers=_cfbd_headers(), params={"year": year}, timeout=30)
    if r.status_code != 200:
        return []
    js = r.json() or []
    names = []
    for row in js:
        name = str(row.get("school") or row.get("team") or "").strip()
        if name:
            names.append(name)
    return sorted(set(names))

# --------------------------- Mining PPG fields --------------------------------
def _mine_ppg(df: pd.DataFrame) -> pd.DataFrame:
    """
    From /stats/season DF, build a {team: off_ppg, def_ppg} table.

    We look for rows where:
      - 'category' is offense/Offense and statName indicates points per game
      - 'category' is defense/Defense and statName indicates (opponent) points per game

    We accept many shapes & names to be robust.
    """
    if df.empty:
        return pd.DataFrame(columns=["team", "off_ppg", "def_ppg"])

    # Normalize likely column names
    cols = {c.lower(): c for c in df.columns}
    def col(*cands):
        for c in cands:
            if c in cols: return cols[c]
        return None

    team_col = col("team", "school")
    cat_col  = col("category")
    name_col = col("statname", "stat_name", "name")
    val_col  = col("statvalue", "stat_value", "value")

    # If the flat shape isn't present, try a very generous fallback:
    if not all([team_col, val_col]):
        # try to find any numeric columns named like ppg
        lower_cols = [c.lower() for c in df.columns]
        off_like = [i for i,c in enumerate(lower_cols) if "ppg" in c and "def" not in c and "opp" not in c]
        def_like = [i for i,c in enumerate(lower_cols) if ("def" in c or "opp" in c) and "ppg" in c]
        if off_like and def_like and team_col:
            off_col = df.columns[off_like[0]]
            def_col = df.columns[def_like[0]]
            out = df[[team_col, off_col, def_col]].copy()
            out.columns = ["team", "off_ppg", "def_ppg"]
            out["off_ppg"] = pd.to_numeric(out["off_ppg"], errors="coerce")
            out["def_ppg"] = pd.to_numeric(out["def_ppg"], errors="coerce")
            return out.dropna(subset=["team"]).drop_duplicates(subset=["team"]).reset_index(drop=True)

        # last-ditch: give empty
        return pd.DataFrame(columns=["team", "off_ppg", "def_ppg"])

    # Prepare helpers
    def is_ppg(s: str) -> bool:
        s = (s or "").lower().strip()
        return any(k in s for k in ["points per game", "ppg", "points_per_game", "pts/g", "pts per game"])

    def is_offense(s: str) -> bool:
        return (s or "").lower().strip() in ("offense", "offensive")

    def is_defense(s: str) -> bool:
        return (s or "").lower().strip() in ("defense", "defensive")

    rows = []
    for _, r in df.iterrows():
        t = r.get(team_col)
        if not t: 
            continue
        nm = str(r.get(name_col, ""))
        cat = str(r.get(cat_col, ""))
        try:
            val = float(r.get(val_col))
        except Exception:
            continue

        rows.append({"team": str(t), "name": nm, "cat": cat, "val": val})

    if not rows:
        return pd.DataFrame(columns=["team", "off_ppg", "def_ppg"])

    x = pd.DataFrame(rows)

    # candidate offense rows
    off = x[x["name"].apply(is_ppg) & x["cat"].apply(is_offense)].copy()
    # some APIs label simply 'points per game' without category; take the largest offense-ish value per team
    if off.empty:
        off = x[x["name"].apply(is_ppg)].copy()
    off = off.sort_values(["team", "val"], ascending=[True, False]).drop_duplicates("team")
    off = off[["team", "val"]].rename(columns={"val": "off_ppg"})

    # candidate defense rows (opp points per game)
    deff = x[x["name"].apply(is_ppg) & x["cat"].apply(is_defense)].copy()
    # if category missing, try anything with "opp" in name
    if deff.empty:
        deff = x[x["name"].str.lower().str.contains("opp") & x["name"].apply(is_ppg)].copy()
    deff = deff.sort_values(["team", "val"], ascending=[True, True]).drop_duplicates("team")
    deff = deff[["team", "val"]].rename(columns={"val": "def_ppg"})

    out = pd.merge(off, deff, on="team", how="outer")
    # prune wildly bad values and fill gentle defaults
    out["off_ppg"] = pd.to_numeric(out["off_ppg"], errors="coerce")
    out["def_ppg"] = pd.to_numeric(out["def_ppg"], errors="coerce")
    out["off_ppg"] = out["off_ppg"].clip(lower=5, upper=60)
    out["def_ppg"] = out["def_ppg"].clip(lower=5, upper=60)
    # Fill missing sides with league averages if necessary
    if not out.empty:
        out["off_ppg"] = out["off_ppg"].fillna(out["off_ppg"].mean())
        out["def_ppg"] = out["def_ppg"].fillna(out["def_ppg"].mean())
    return out.dropna(subset=["team"]).drop_duplicates(subset=["team"]).reset_index(drop=True)

# --------------------------- Simulation ---------------------------------------
def simulate_poisson(mu_home: float, mu_away: float, trials: int = TRIALS):
    mu_home = max(EPS, float(mu_home))
    mu_away = max(EPS, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    # No ties in CFB; if tie, tiny home advantage breaker
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.52
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean())

# --------------------------------- UI -----------------------------------------
st.set_page_config(page_title="CFB Poisson — auto from CFBD", layout="centered")
st.title("🏈🎓 College Football — 2025 (auto from CFBD)")

with st.expander("Key / cache tools", expanded=False):
    k = _get_cfbd_key()
    st.write("Key present:", bool(k))
    if k:
        st.code(f"{k[:2]}…{k[-3:]} (len={len(k)})", language="text")
    if st.button("Clear CFB cache"):
        fetch_stats_season.clear()
        fetch_teams_list.clear()
        st.success("Cleared CFB cache. Re-run the app.")

if not _get_cfbd_key():
    st.error("Add your CFBD API key in Streamlit **Secrets** as `CFBD_API_KEY = \"...\"`.")
    st.stop()

# Try to fetch
try:
    raw = fetch_stats_season(YEAR, SEASON_TYPE)
except Exception as e:
    st.error(f"CFBD request failed: {e}")
    st.stop()

if raw.empty:
    st.error("CFBD returned no data for this request (empty). Check year/seasonType or API limits.")
    st.stop()

team_table = _mine_ppg(raw)
if team_table.empty:
    st.error("Parsed an empty team table from CFBD response (no PPG fields found).")
    with st.expander("See first rows of raw CFBD data"):
        st.dataframe(raw.head(50))
    st.stop()

# Allow user to choose teams
teams = sorted(team_table["team"].unique().tolist())
home = st.selectbox("Home team", teams, index=0)
away = st.selectbox("Away team", teams, index=min(1, len(teams)-1))

if home == away:
    st.info("Pick two different teams.")
    st.stop()

H = team_table.loc[team_table["team"] == home].iloc[0]
A = team_table.loc[team_table["team"] == away].iloc[0]

mu_home = (float(H["off_ppg"]) + float(A["def_ppg"])) / 2.0 + HOME_EDGE
mu_away = (float(A["off_ppg"]) + float(H["def_ppg"])) / 2.0

pH, pA, mH, mA = simulate_poisson(mu_home, mu_away, TRIALS)

st.markdown(
    f"**{home}** vs **{away}** — "
    f"Expected points: **{mH:.1f}–{mA:.1f}** · "
    f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
)

with st.expander("Show team table (off_ppg / def_ppg)"):
    st.dataframe(team_table.sort_values("team").reset_index(drop=True))
