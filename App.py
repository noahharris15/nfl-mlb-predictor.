import time, math, json, random
from typing import Dict, Any, List, Tuple
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="All-Sports PrizePicks Simulator", page_icon="📊", layout="wide")

# ----------------------------
# Helpers
# ----------------------------

PP_URL = "https://site.api.prizepicks.io/api/v1/projections"

LEAGUE_CHOICES = [
    "NFL", "NBA", "MLB", "NHL", "WNBA", "NCAAF", "NCAAB", "SOC", "LOL", "CS2", "VALORANT"
]

# Some PrizePicks boards use funky market labels; normalize to a core "market" string
MARKET_NORMALIZER = {
    # NBA
    "Points": "points", "Rebounds": "rebounds", "Assists": "assists",
    "Pts+Rebs+Asts": "pra", "Rebs+Asts": "rebounds_assists",
    "3-PT Made": "threes_made", "Fantasy Score": "fantasy",

    # NFL
    "Pass Yds": "pass_yards", "Rush Yds": "rush_yards", "Rec Yds": "rec_yards",
    "Receptions": "receptions", "Pass+Rush+Rec Yds": "prr_yards",
    "Pass TDs": "pass_tds", "Rush+Rec Yds": "rush_rec_yards",
    "Kicking Pts": "kicking_pts",

    # MLB
    "Hits+Runs+RBIs": "hrr", "Total Bases": "total_bases",
    "Strikeouts": "strikeouts", "Pitching Outs": "pitching_outs",
    "Walks Allowed": "walks_allowed",

    # NHL
    "SOG": "shots_on_goal", "Points (Hockey)": "points_hky",

    # Generic catch-alls we preserve
}

def normal_cdf(x, mu, sigma):
    # numerically-stable normal CDF without SciPy
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

@st.cache_data(ttl=120)
def fetch_prizepicks_json(league: str) -> Dict[str, Any] | None:
    """
    Pull PrizePicks 'site' API (works in Streamlit). We request only one league
    and do a couple of retries for 429/403 style failures.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.prizepicks.com/",
    }
    params = {"per_page": 250, "single_stat": "true", "league": league.lower()}
    backoff = [1.0, 2.0, 4.0]
    for i in range(len(backoff) + 1):
        try:
            r = requests.get(PP_URL, headers=headers, params=params, timeout=15)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json()
            # If blocked or rate-limited, wait and retry
            if r.status_code in (403, 429):
                if i < len(backoff):
                    st.info(f"⏳ PrizePicks is throttling ({r.status_code}). Retrying in {backoff[i]}s…")
                    time.sleep(backoff[i])
                    continue
            # Any other non-200: break with None
            st.error(f"PrizePicks fetch failed (HTTP {r.status_code}).")
            return None
        except requests.RequestException as e:
            if i < len(backoff):
                st.info(f"⏳ Network hiccup: {e}. Retrying in {backoff[i]}s…")
                time.sleep(backoff[i])
                continue
            st.error(f"PrizePicks request failed: {e}")
            return None
    return None

def parse_pp(json_payload: Dict[str, Any], league: str) -> pd.DataFrame:
    """
    Convert the PP JSON into a clean DataFrame with one row per projection line.
    We strictly filter to the league you selected.
    """
    if not json_payload:
        return pd.DataFrame()

    data = json_payload.get("data", [])
    included = json_payload.get("included", [])

    # Map player_id -> (name, team)
    players: Dict[str, Tuple[str, str]] = {}
    for inc in included:
        if inc.get("type") in ("players", "Player", "player"):
            pid = inc.get("id")
            attr = inc.get("attributes", {})
            name = attr.get("name") or attr.get("full_name") or attr.get("display_name") or "Unknown"
            team = attr.get("team") or attr.get("injury_team") or attr.get("league_team") or ""
            players[pid] = (name, team)

    rows = []
    for item in data:
        if item.get("type") not in ("projection", "projections"):
            continue

        attr = item.get("attributes", {})
        lge = (attr.get("league") or "").upper()
        if lge != league.upper():
            continue  # hard filter — avoid mixed leagues

        line_score = attr.get("line_score")
        try:
            line = float(line_score)
        except (TypeError, ValueError):
            continue

        stat = attr.get("stat_type") or attr.get("stat") or attr.get("label") or ""
        market_pp = attr.get("market_type") or attr.get("display_stat") or stat
        market_norm = MARKET_NORMALIZER.get(market_pp, market_pp.strip().lower())

        pid = str(attr.get("player_id") or item.get("relationships", {}).get("new_player", {}).get("data", {}).get("id") or "")
        name, team = players.get(pid, ("Unknown", ""))

        rows.append({
            "league": lge,
            "player": name,
            "team": team,
            "market_pp": market_pp,
            "market": market_norm,
            "line": line,
            "projection_id": item.get("id"),
            "game_time": attr.get("start_time") or attr.get("start_time_tbd") or "",
        })

    df = pd.DataFrame(rows)
    # Remove any obvious junk / null names
    df = df[(df["player"].notna()) & (df["player"] != "Unknown")]
    return df.reset_index(drop=True)

def estimate_sd(df: pd.DataFrame) -> pd.Series:
    """
    We need a reasonable SD to simulate. With no historical feed, we estimate:
      1) Use dispersion of all current lines in the same market as a proxy
      2) Fallback to market-specific default percentage of mean line
    """
    defaults = {
        # NBA
        "points": 0.22, "rebounds": 0.30, "assists": 0.32, "pra": 0.20, "rebounds_assists": 0.26,
        "threes_made": 0.55, "fantasy": 0.18,

        # NFL
        "pass_yards": 0.16, "rush_yards": 0.28, "rec_yards": 0.25, "receptions": 0.35,
        "prr_yards": 0.18, "rush_rec_yards": 0.22, "pass_tds": 0.65, "kicking_pts": 0.30,

        # MLB
        "hrr": 0.60, "total_bases": 0.75, "strikeouts": 0.35, "pitching_outs": 0.18, "walks_allowed": 0.50,

        # NHL
        "shots_on_goal": 0.45, "points_hky": 0.65,
    }

    # market-level empirical SD from the board itself
    sd_emp = df.groupby("market")["line"].transform(lambda x: float(np.std(x, ddof=1)) if len(x) >= 6 else np.nan)

    # fallback default SD (percent of line)
    pct = df["market"].map(defaults).fillna(0.30)
    sd_fallback = (df["line"].abs() * pct).clip(lower=0.35)  # keep a floor so tails aren't degenerate

    # choose empirical when reasonable, else fallback
    sd = sd_emp.fillna(sd_fallback)
    # Final safety: no zero/NaN
    sd = sd.replace([0, np.nan, np.inf, -np.inf], np.nan).fillna(sd_fallback)
    return sd

def simulate_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normal-model approximation:
      - mean := line (board-implied fair line, since we have no historical per-player feed here)
      - sd   := estimated (see above)
      - P(over) := 1 - CDF(line | mean, sd)
      - P(under) := CDF(line | mean, sd)
      - 'edge' shows how far from 50% we are if you nudge mean slightly by 0.5*sd
        (a softer heuristic to surface interesting spots on the board)
    """
    # baseline mean anchored at the current line (fair price ≈ 50/50)
    mu = df["line"].astype(float)
    sd = estimate_sd(df)

    # nudge the mean slightly with a tiny random jitter to break ties visually
    jitter = np.random.normal(loc=0.0, scale=0.02, size=len(mu))
    mu_adj = mu + jitter

    p_under = [normal_cdf(x, m, s) for x, m, s in zip(df["line"], mu_adj, sd)]
    p_over = [1.0 - p for p in p_under]

    # Heuristic "edge" (how sensitive the probability is if the line is mispriced by ~0.5*sd)
    shift = 0.5 * sd
    p_over_up = [1.0 - normal_cdf(x, m + d, s) for x, m, s, d in zip(df["line"], mu_adj, sd, shift)]
    p_over_dn = [1.0 - normal_cdf(x, m - d, s) for x, m, s, d in zip(df["line"], mu_adj, sd, shift)]
    edge = np.abs(np.array(p_over_up) - np.array(p_over_dn)) * 0.5  # symmetrical sensitivity

    out = df.copy()
    out["_mean"] = mu_adj
    out["_sd"] = sd
    out["P(over)"] = p_over
    out["P(under)"] = p_under
    out["edge"] = edge
    return out

# ----------------------------
# UI
# ----------------------------

st.title("📊 All-Sports PrizePicks Simulator (auto baselines, fixed parsing)")
st.caption("Live board → clean league filter → quick normal-model probabilities. No CSVs needed.")

colL, colR = st.columns([1, 2])
with colL:
    league = st.selectbox("League", options=LEAGUE_CHOICES, index=LEAGUE_CHOICES.index("NFL"))

with st.expander("How this works", False):
    st.markdown(
        "- We fetch the live board from the **PrizePicks site API** (works on Streamlit Cloud).\n"
        "- We **hard-filter** to the league you picked to avoid mixed names.\n"
        "- Without historical per-player stats, we treat the current line as the fair mean and "
        "estimate standard deviation from the board + market defaults.\n"
        "- The result is a fast **P(over)/P(under)** estimate and a sortable **edge** heuristic to surface interesting spots.\n"
        "- If you later want true player baselines, we can wire those in per league (ESPN/Stats APIs) — this scaffold already supports dropping them in."
    )

# Fetch & parse
payload = fetch_prizepicks_json(league)
if not payload:
    st.error("Could not fetch PrizePicks data. Network or temporary block. Try again in ~30–60 seconds.")
    st.stop()

df_raw = parse_pp(payload, league)
if df_raw.empty:
    st.warning("No projections parsed for this league right now. (Board could be light / off-cycling.)")
    st.stop()

# Simulate
df_sim = simulate_probs(df_raw)

# Format & display
display_cols = ["league", "player", "team", "market", "market_pp", "line", "P(over)", "P(under)", "edge"]
df_view = df_sim[display_cols].copy()

# nicer formatting
for col in ["line"]:
    df_view[col] = df_view[col].map(lambda x: round(float(x), 2))
for col in ["P(over)", "P(under)", "edge"]:
    df_view[col] = (df_view[col] * 100.0).map(lambda x: f"{x:0.1f}%")

st.subheader("Simulated edges (normal model)")
min_edge = st.slider("Minimum edge to show (percent)", 0.0, 10.0, 2.0, 0.5)
df_sort = df_sim.copy()
df_sort = df_sort[df_sort["edge"] >= (min_edge / 100.0)]
df_sort = df_sort.sort_values("edge", ascending=False)

nice = df_sort[["league", "player", "team", "market", "market_pp", "line", "P(over)", "P(under)", "edge"]].copy()
nice["line"] = nice["line"].round(2)
nice["P(over)"] = (nice["P(over)"] * 100).map(lambda x: f"{x:0.1f}%")
nice["P(under)"] = (nice["P(under)"] * 100).map(lambda x: f"{x:0.1f}%")
nice["edge"] = (nice["edge"] * 100).map(lambda x: f"{x:0.1f}%")

st.dataframe(nice.reset_index(drop=True), use_container_width=True)

# Download
csv = nice.to_csv(index=False)
st.download_button("Download table as CSV", data=csv, file_name=f"prizepicks_{league.lower()}_simulated.csv", mime="text/csv")

st.caption("Tip: if you see throttling, wait 20–60s and hit the **Rerun** button in the top-right menu.")
