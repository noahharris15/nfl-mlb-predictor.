import os
import json
import time
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="NFL/MLB + Player Props (Odds API)", layout="wide")

# ---------- Shared helpers ----------
def conservative_sd(mean_val: float, min_sd: float = 0.75, frac: float = 0.30) -> float:
    try:
        m = float(mean_val)
    except Exception:
        return 1.25
    if m <= 0:
        return 1.0
    return max(min_sd, frac * m)

def prob_over_under(mean_val: float, line_val: float, sd_val: Optional[float] = None):
    sd = sd_val if (sd_val and sd_val > 0) else conservative_sd(mean_val)
    p_over = float(1 - norm.cdf(line_val, loc=mean_val, scale=sd))
    p_under = float(norm.cdf(line_val, loc=mean_val, scale=sd))
    return round(p_over * 100, 2), round(p_under * 100, 2), round(sd, 3)

def clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").strip().lower()

def try_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# ---------- NAV ----------
page = st.sidebar.radio("Pick a page", ["NFL", "MLB", "Player Props"])

# =======================================================================
# NFL PAGE  — team vs team using your CSV (off_ppg, def_ppg)
# =======================================================================
if page == "NFL":
    st.title("🏈 NFL — Team Simulator (CSV stats)")
    st.write("Upload a CSV with columns: **team, off_ppg, def_ppg**")

    up = st.file_uploader("Upload NFL team CSV", type=["csv"])
    if not up:
        st.stop()

    teams = pd.read_csv(up)
    req_cols = {"team", "off_ppg", "def_ppg"}
    if not req_cols.issubset(set(teams.columns)):
        st.error(f"CSV must include columns: {sorted(req_cols)}")
        st.stop()

    teams["off_ppg"] = pd.to_numeric(teams["off_ppg"], errors="coerce")
    teams["def_ppg"] = pd.to_numeric(teams["def_ppg"], errors="coerce")
    teams = teams.dropna(subset=["team", "off_ppg", "def_ppg"])

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("Home team", sorted(teams["team"].unique()))
    with c2:
        away = st.selectbox("Away team", sorted(teams["team"].unique()))

    if home == away:
        st.warning("Pick two different teams.")
        st.stop()

    h = teams.loc[teams["team"] == home].iloc[0]
    a = teams.loc[teams["team"] == away].iloc[0]

    # Simple rate blend
    home_exp = 0.6 * h["off_ppg"] + 0.4 * (a["def_ppg"])
    away_exp = 0.6 * a["off_ppg"] + 0.4 * (h["def_ppg"])

    st.markdown(
        f"**{home} vs {away}** — expected points: **{home_exp:.1f}–{away_exp:.1f}**"
    )

    sims = st.slider("Simulations", 1000, 50000, 10000, step=1000)
    rng = np.random.default_rng(123)
    home_scores = rng.poisson(lam=max(home_exp, 0.1), size=sims)
    away_scores = rng.poisson(lam=max(away_exp, 0.1), size=sims)

    p_home = float((home_scores > away_scores).mean() * 100)
    p_away = float((away_scores > home_scores).mean() * 100)
    p_tie  = round(100 - p_home - p_away, 2)

    st.write(f"**P({home} win)** = {p_home:.1f}% · **P({away} win)** = {p_away:.1f}% · **P(Tie)** = {p_tie:.1f}%")
    st.dataframe(
        pd.DataFrame({"home_score": home_scores, "away_score": away_scores}).describe().T,
        use_container_width=True
    )

# =======================================================================
# MLB PAGE — team vs team using your CSV (off_rpg, def_rpg) or off_ppg naming
# =======================================================================
elif page == "MLB":
    st.title("⚾ MLB — Team Simulator (CSV stats)")
    st.write("Upload a CSV with columns: **team, off_rpg (or off_ppg), def_rpg (or def_ppg)**")

    up = st.file_uploader("Upload MLB team CSV", type=["csv"])
    if not up:
        st.stop()

    teams = pd.read_csv(up)
    # allow both naming conventions
    col_map = {
        "off_rpg": None, "off_ppg": None, "def_rpg": None, "def_ppg": None
    }
    for c in list(teams.columns):
        if c in col_map:
            col_map[c] = c
    # normalize
    teams["off"] = pd.to_numeric(
        teams[col_map["off_rpg"] or "off_ppg"], errors="coerce"
        if (col_map["off_rpg"] is None and "off_ppg" in teams.columns)
        else teams[col_map["off_rpg"]], errors="coerce"
    )

    teams["off"] = pd.to_numeric(
        teams[col_map["off_rpg"] or "off_ppg"], errors="coerce"
    ) if (col_map["off_rpg"] or "off_ppg") else np.nan

    teams["def"] = pd.to_numeric(
        teams[col_map["def_rpg"] or "def_ppg"], errors="coerce"
    ) if (col_map["def_rpg"] or "def_ppg") else np.nan

    if "team" not in teams.columns or teams["off"].isna().all() or teams["def"].isna().all():
        st.error("CSV must include team + off_rpg/off_ppg and def_rpg/def_ppg.")
        st.stop()

    teams = teams.dropna(subset=["team", "off", "def"])

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("Home team", sorted(teams["team"].unique()))
    with c2:
        away = st.selectbox("Away team", sorted(teams["team"].unique()))
    if home == away:
        st.warning("Pick two different teams.")
        st.stop()

    h = teams.loc[teams["team"] == home].iloc[0]
    a = teams.loc[teams["team"] == away].iloc[0]

    home_exp = 0.55 * h["off"] + 0.45 * a["def"]
    away_exp = 0.55 * a["off"] + 0.45 * h["def"]

    st.markdown(f"**{home} vs {away}** — expected runs: **{home_exp:.1f}–{away_exp:.1f}**")

    sims = st.slider("Simulations", 1000, 50000, 10000, step=1000)
    rng = np.random.default_rng(123)
    home_scores = rng.poisson(lam=max(home_exp, 0.1), size=sims)
    away_scores = rng.poisson(lam=max(away_exp, 0.1), size=sims)

    p_home = float((home_scores > away_scores).mean() * 100)
    p_away = float((away_scores > home_scores).mean() * 100)
    p_tie  = round(100 - p_home - p_away, 2)
    st.write(f"**P({home} win)** = {p_home:.1f}% · **P({away} win)** = {p_away:.1f}% · **P(Tie)** = {p_tie:.1f}%")

# =======================================================================
# PLAYER PROPS — Odds API lines + your uploaded player CSV
# =======================================================================
else:
    st.title("🎯 Player Props — Odds API lines + your CSV stats")

    # Odds API key from secrets or textbox
    secret_key = st.secrets.get("ODDS_API_KEY") if "ODDS_API_KEY" in st.secrets else ""
    api_key = st.text_input("Odds API Key", type="password", value=secret_key or "")
    if not api_key:
        st.info("Enter your Odds API key to fetch lines automatically.")
    league_choice = st.selectbox("League (for props)", ["NFL", "MLB", "NBA"])

    # Default market → CSV column mapping (you can edit/add)
    default_mapping = {
        # NFL
        "Passing Yards": "passing_yards",
        "Rush Yards": "rushing_yards",
        "Receiving Yards": "receiving_yards",
        "Receptions": "receptions",
        "Pass TDs": "passing_tds",
        "Rush TDs": "rushing_tds",
        "Receiving TDs": "receiving_tds",
        # MLB
        "Pitcher Strikeouts": "strikeouts",
        "Hits": "hits",
        "Home Runs": "home_runs",
        "RBIs": "rbi",
        "Total Bases": "total_bases",
        # NBA
        "Points": "points",
        "Rebounds": "rebounds",
        "Assists": "assists",
        "3-PT Made": "threes",
        "Pts+Reb+Ast": "pra",
    }
    st.caption("If your CSV uses different column names, change the mapping below.")
    mapping_json = st.text_area(
        "Market → CSV column mapping (JSON)",
        value=json.dumps(default_mapping, indent=2),
        height=220
    )
    try:
        market_to_col: Dict[str, str] = json.loads(mapping_json)
    except Exception as e:
        st.error(f"Mapping JSON error: {e}")
        st.stop()

    # Upload your per-player CSV (averages; optional *_sd columns)
    st.subheader("Upload your player stats CSV")
    st.write("Required column: **player**. Then one column per stat in your mapping (plus optional `<col>_sd`).")
    up = st.file_uploader("Upload player stats CSV", type=["csv"])
    if not up:
        st.stop()

    stats = pd.read_csv(up)
    if "player" not in stats.columns:
        st.error("CSV must include a 'player' column.")
        st.stop()
    # Normalize names
    stats["player_clean"] = stats["player"].apply(clean_name)

    # ------------- fetch Odds API lines -------------
    sport_keys = {
        "NFL": "americanfootball_nfl",
        "MLB": "baseball_mlb",
        "NBA": "basketball_nba",
    }
    sport = sport_keys[league_choice]

    # Odds API markets for props (you can tweak)
    # (The Odds API uses market ids per book; we will pull 'player props' group by passing a list)
    # Reference markets commonly available:
    market_candidates = {
        "NFL": ["player_passing_yards", "player_rushing_yards", "player_receiving_yards",
                "player_receptions", "player_passing_tds", "player_rushing_tds", "player_receiving_tds"],
        "MLB": ["player_hits", "player_home_runs", "player_rbis", "player_total_bases", "player_strikeouts"],
        "NBA": ["player_points", "player_rebounds", "player_assists", "player_three_points_made", "player_points_rebounds_assists"],
    }
    chosen_markets = market_candidates[league_choice]

    def fetch_props_from_oddsapi(api_key: str, sport_key: str, markets: List[str], region: str = "us") -> pd.DataFrame:
        """
        Returns DataFrame with columns: player, market_api, bookmaker, team(optional), line
        """
        if not api_key:
            return pd.DataFrame()
        url = "https://api.the-odds-api.com/v4/sports/{sport}/odds".format(sport=sport_key)
        params = {
            "apiKey": api_key,
            "regions": region,
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "unix",
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:240]}")
        events = r.json()
        rows = []
        for ev in events:
            home, away = ev.get("home_team"), ev.get("away_team")
            for bk in ev.get("bookmakers", []):
                bk_name = bk.get("title")
                for m in bk.get("markets", []):
                    m_key = m.get("key")  # e.g., player_points
                    for out in m.get("outcomes", []):
                        name = out.get("name")  # player name
                        point = out.get("point")  # the line
                        if name is None or point is None:
                            continue
                        rows.append({
                            "event_home": home,
                            "event_away": away,
                            "bookmaker": bk_name,
                            "market_api": m_key,
                            "player": name,
                            "line": try_float(point),
                        })
        df = pd.DataFrame(rows)
        return df.dropna(subset=["player", "line"])

    with st.expander("Fetch (debug)"):
        st.code(f"sport={sport} markets={chosen_markets}")

    lines_df = pd.DataFrame()
    if api_key:
        try:
            lines_df = fetch_props_from_oddsapi(api_key, sport, chosen_markets)
        except Exception as e:
            st.error(f"Fetching Odds API props failed: {e}")

    if lines_df.empty:
        st.warning("No props pulled from Odds API (rate limit / plan / market availability). You can upload a fallback.")
        up_props = st.file_uploader("Upload props fallback CSV (player, market_api, line)", type=["csv"], key="props_fallback")
        if up_props:
            lines_df = pd.read_csv(up_props)
    if lines_df.empty:
        st.stop()

    # ------------- map market_api -> readable market (to match your CSV columns) -------------
    # Simple translation table from Odds API market keys to display names used in mapping
    api_to_display = {
        # NFL
        "player_passing_yards": "Passing Yards",
        "player_rushing_yards": "Rush Yards",
        "player_receiving_yards": "Receiving Yards",
        "player_receptions": "Receptions",
        "player_passing_tds": "Pass TDs",
        "player_rushing_tds": "Rush TDs",
        "player_receiving_tds": "Receiving TDs",
        # MLB
        "player_strikeouts": "Pitcher Strikeouts",
        "player_hits": "Hits",
        "player_home_runs": "Home Runs",
        "player_rbis": "RBIs",
        "player_total_bases": "Total Bases",
        # NBA
        "player_points": "Points",
        "player_rebounds": "Rebounds",
        "player_assists": "Assists",
        "player_three_points_made": "3-PT Made",
        "player_points_rebounds_assists": "Pts+Reb+Ast",
    }
    lines_df["market"] = lines_df["market_api"].map(api_to_display).fillna(lines_df["market_api"])
    lines_df["player_clean"] = lines_df["player"].apply(clean_name)

    # ------------- join lines to your stats -------------
    # We will left-join on cleaned player names; allow user to adjust a small tolerance by merging suggestions
    merged = pd.merge(
        lines_df,
        stats,
        left_on="player_clean",
        right_on="player_clean",
        how="left",
        suffixes=("", "_stat")
    )

    # compute avg + sd per row based on the mapping
    means, sds = [], []
    for _, r in merged.iterrows():
        mkt = r.get("market")
        col = market_to_col.get(str(mkt), None)
        if col is None or col not in stats.columns:
            means.append(np.nan); sds.append(np.nan); continue
        mean_val = try_float(r.get(col))
        sd_val = try_float(r.get(f"{col}_sd"))
        means.append(mean_val)
        sds.append(sd_val)
    merged["avg"] = means
    merged["sd"] = sds

    # drop rows without a mean
    merged = merged.dropna(subset=["avg"])
    if merged.empty:
        st.error("Mapped zero rows. Check your Market→CSV mapping and your CSV column names.")
        st.stop()

    # ------------- simulate -------------
    p_over_list, p_under_list, used_sd_list = [], [], []
    for _, r in merged.iterrows():
        p_over, p_under, used_sd = prob_over_under(r["avg"], r["line"], r.get("sd"))
        p_over_list.append(p_over)
        p_under_list.append(p_under)
        used_sd_list.append(used_sd)

    merged["P(Over)"] = p_over_list
    merged["P(Under)"] = p_under_list
    merged["model_sd"] = used_sd_list

    show_cols = ["bookmaker", "player", "market", "line", "avg", "model_sd", "P(Over)", "P(Under)"]
    st.subheader("Results")
    st.dataframe(merged[show_cols].sort_values("P(Over)", ascending=False), use_container_width=True)

    st.download_button(
        "Download results CSV",
        merged[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"{league_choice.lower()}_props_sim.csv",
        mime="text/csv"
    )

    with st.expander("What matched / debug"):
        st.write("Pulled props:", len(lines_df))
        st.write("Matched rows:", len(merged))
        st.dataframe(lines_df.head(30), use_container_width=True)
        st.dataframe(stats.head(30), use_container_width=True)
