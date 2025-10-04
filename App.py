# player_props_prizepicks_csv.py
# Streamlit page: PrizePicks NFL props + optional player CSV baselines

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.subheader("🎯 NFL Player Props — PrizePicks + Your CSV Baselines")

# ---------------------- Config ----------------------
PP_URL = "https://api.prizepicks.com/projections"
PP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; Streamlit/1.0)",
    "Origin": "https://www.prizepicks.com",
    "Referer": "https://www.prizepicks.com/",
}

SPORT_SLUG = st.selectbox("Sport board", ["nfl","nba","mlb","nhl"], index=0)
N_SIMS = st.slider("Sim runs per prop", 2000, 20000, 8000, step=1000)

# Normalize market names from PrizePicks to our keys
MARKET_MAP = {
    "passing yards": "pass_yds",
    "passing attempts": "pass_att",
    "completions": "pass_cmp",
    "passing tds": "pass_td",
    "rushing yards": "rush_yds",
    "rushing attempts": "rush_att",
    "rushing tds": "rush_td",
    "receiving yards": "rec_yds",
    "receptions": "rec_rec",
    "receiving tds": "rec_td",
    "longest reception": "long_rec",
    "longest rush": "long_rush",
    "fantasy score": "fantasy",
}

# Default sigma/prior when CSV doesn't supply sd
NORMAL_SIGMA = {
    "pass_yds": 55.0,
    "rush_yds": 22.0,
    "rec_yds": 24.0,
    "pass_att": 4.5,
    "pass_cmp": 3.5,
    "rush_att": 3.0,
    "long_rec": 6.0,
    "long_rush": 6.0,
    "fantasy": 8.0,
}
POISSON_KEYS = {"pass_td","rush_td","rec_td"}

MEAN_SHRINK = st.slider("Mean shrink toward PP line (safer)", 0.0, 0.5, 0.10, 0.05)

# ---------------------- Helpers ----------------------
@st.cache_data(ttl=60)
def fetch_prizepicks_board(league_slug: str) -> dict:
    params = {"per_page": 1000, "single_stat": "true", "league": league_slug}
    r = requests.get(PP_URL, params=params, headers=PP_HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def normalize_market(label: str) -> str:
    s = (label or "").lower()
    for k, v in MARKET_MAP.items():
        if k in s: return v
    return s[:30]  # fallback

def parse_board(payload: dict) -> pd.DataFrame:
    players, leagues = {}, {}
    for inc in payload.get("included", []):
        t = inc.get("type")
        a = inc.get("attributes", {}) or {}
        _id = str(inc.get("id"))
        if t in ("new_player","players"):
            name = a.get("name") or f"{a.get('first_name','').strip()} {a.get('last_name','').strip()}".strip()
            players[_id] = {
                "name": name or "Unknown",
                "team": a.get("team") or a.get("team_name"),
                "pos": a.get("position"),
            }

    rows = []
    for d in payload.get("data", []):
        a = d.get("attributes", {}) or {}
        rels = d.get("relationships", {}) or {}
        proj_type = a.get("projection_type") or a.get("stat_type") or ""
        line = a.get("line_score") or a.get("value") or a.get("line")
        pid = None
        for key in ("new_player","player","athlete"):
            if key in rels and isinstance(rels[key].get("data"), dict):
                pid = str(rels[key]["data"].get("id"))
                break
        pl = players.get(pid, {})
        rows.append({
            "player": pl.get("name"),
            "team": pl.get("team"),
            "pos": pl.get("pos"),
            "raw_market": (proj_type or "").strip(),
            "market": normalize_market(proj_type),
            "line": float(line) if line not in (None,"") else np.nan,
            "posted_at": a.get("updated_at") or a.get("created_at"),
        })
    df = pd.DataFrame(rows).dropna(subset=["player","market","line"]).reset_index(drop=True)
    return df

# ---------------------- CSV handling ----------------------
st.markdown(
    """
**Upload your NFL player baselines (optional)**

- **Wide format** (example columns):  
  `player, team, pos, pass_yds_mean, pass_yds_sd, rush_yds_mean, rush_yds_sd, rec_yds_mean, rec_yds_sd, rec_rec_mean, rec_rec_sd, pass_td_mean, ...`

- **Long format**:  
  `player, market, mean, sd`  where `market` uses keys like `pass_yds, rush_yds, rec_rec, pass_td, ...`
"""
)

csv_file = st.file_uploader("Upload player CSV (optional)", type=["csv"])
csv_long = None
csv_wide = None

if csv_file is not None:
    try:
        raw = pd.read_csv(csv_file)
        cols = [c.lower() for c in raw.columns]
        raw.columns = cols

        if set(["player","market","mean"]).issubset(cols):
            # long format; sd optional
            if "sd" not in raw.columns:
                raw["sd"] = np.nan
            csv_long = raw.copy()
            st.success(f"Loaded long-format CSV: {len(csv_long)} rows.")
            st.dataframe(csv_long.head(10), use_container_width=True)
        else:
            # assume wide format: player row with per-market mean/sd columns
            csv_wide = raw.copy()
            st.success(f"Loaded wide-format CSV: {len(csv_wide)} players.")
            st.dataframe(csv_wide.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"CSV parse error: {e}")

# ---------------------- Simulation ----------------------
rng = np.random.default_rng(17)

def lookup_mean_sd(player: str, market: str, line: float):
    """
    Returns (mu, sd) using uploaded CSV if present.
    - Long format exact match on (player, market)
    - Wide format looks for columns f"{market}_mean" and optional f"{market}_sd"
    - Fallback: mean = (1-shrink)*CSV_or_default + shrink*line; sd from NORMAL_SIGMA or 0.25*mean
    """
    mu = None
    sd = None

    if csv_long is not None:
        m = csv_long[(csv_long["player"].str.lower()==player.lower()) &
                     (csv_long["market"].str.lower()==market.lower())]
        if not m.empty:
            mu = float(m.iloc[0]["mean"])
            sd = m.iloc[0]["sd"]
            sd = None if pd.isna(sd) else float(sd)

    if mu is None and csv_wide is not None:
        # exact player row
        pw = csv_wide[csv_wide["player"].str.lower()==player.lower()]
        if not pw.empty:
            mcol = f"{market}_mean"
            scol = f"{market}_sd"
            if mcol in pw.columns:
                mu = float(pw.iloc[0][mcol])
            if scol in pw.columns and not pd.isna(pw.iloc[0][scol]):
                sd = float(pw.iloc[0][scol])

    # choose defaults when missing
    if mu is None:
        mu = float(line)  # base at the market line
    if sd is None:
        if market in POISSON_KEYS:
            sd = None  # not used for Poisson
        else:
            sd = NORMAL_SIGMA.get(market, max(2.5, 0.08*max(1.0, mu)))

    # shrink mean toward PP line (safety)
    mu = (1.0 - MEAN_SHRINK) * mu + MEAN_SHRINK * float(line)
    return mu, sd

def simulate_row(row):
    market = row["market"]
    L = float(row["line"])
    mu, sd = lookup_mean_sd(row["player"], market, L)

    if market in POISSON_KEYS:
        lam = max(mu, 0.01)
        sims = rng.poisson(lam, size=N_SIMS)
    else:
        sims = rng.normal(loc=mu, scale=max(1e-6, sd), size=N_SIMS)
        sims = np.clip(sims, 0, None)

    p_over = float((sims > L).mean())
    return pd.Series({"mu_used": mu, "sd_used": sd if sd is not None else np.nan,
                      "p_over": p_over, "p_under": 1.0 - p_over,
                      "edge": abs(p_over - 0.5)})

# ---------------------- Run ----------------------
with st.spinner("Pulling PrizePicks board…"):
    try:
        payload = fetch_prizepicks_board(SPORT_SLUG)
        board = parse_board(payload)
    except Exception as e:
        st.error(f"Error loading PrizePicks board: {e}")
        st.stop()

if board.empty:
    st.warning("No props found right now. Try again later.")
    st.stop()

# Filters
c1, c2 = st.columns([2,2])
with c1:
    markets = ["All"] + sorted(board["market"].unique().tolist())
    pick_market = st.selectbox("Market", markets, index=0)
with c2:
    q = st.text_input("Search player/team", "")

view = board.copy()
if pick_market != "All":
    view = view[view["market"] == pick_market]
if q.strip():
    ql = q.lower()
    view = view[view["player"].str.lower().str.contains(ql, na=False) |
                view["team"].astype(str).str.lower().str.contains(ql, na=False)]

st.caption(f"{len(view)} props on board")
with st.spinner("Simulating…"):
    sims = view.apply(simulate_row, axis=1)

out = pd.concat([view.reset_index(drop=True), sims], axis=1)
out["pick"] = np.where(out["p_over"]>=0.5, "Over", "Under")
out["confidence"] = np.where(out["p_over"]>=0.5, out["p_over"], out["p_under"])
out = out.sort_values(["edge","player"], ascending=[False, True]).reset_index(drop=True)

st.dataframe(
    out[["player","team","pos","market","line","mu_used","sd_used","pick","confidence","p_over","p_under","posted_at"]],
    use_container_width=True,
)

st.download_button(
    "⬇️ Download results (CSV)",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name=f"prizepicks_{SPORT_SLUG}_props_sim.csv",
    mime="text/csv",
)

st.info(
    "CSV overrides are used when present (exact player+market). "
    "Provide either a long table (player, market, mean, sd) or a wide table with per-market columns like "
    "`rush_yds_mean, rush_yds_sd`. Missing markets fall back to conservative league priors."
)
