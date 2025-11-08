# NBA Player Props — Odds API + NBA Stats (season averages) + 10k sims
# Place this file at: pages/1_NBA.py

import re, unicodedata, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# NBA API
from nba_api.stats.endpoints import leaguedashplayerstats  # season averages (fast)
from nba_api.stats.static import players as nba_players

st.title("🏀 NBA Player Props — 10k Sims + Defense Adjustments (Season Averages)")

SIM_TRIALS = 10_000
SEASON = "2025-26"   # <- only season used (no fallbacks)

# ---------------- DEFENSE RATINGS ----------------
DEF_RATINGS = {
    "Oklahoma City Thunder": 1.031,
    "San Antonio Spurs": 1.053,
    "Portland Trail Blazers": 1.073,
    "Miami Heat": 1.073,
    "Denver Nuggets": 1.074,
    "Detroit Pistons": 1.076,
    "Cleveland Cavaliers": 1.084,
    "Dallas Mavericks": 1.093,
    "Boston Celtics": 1.097,
    "Orlando Magic": 1.100,
    "Houston Rockets": 1.106,
    "Golden State Warriors": 1.109,
    "Indiana Pacers": 1.112,
    "Philadelphia 76ers": 1.116,
    "Chicago Bulls": 1.122,
    "Atlanta Hawks": 1.123,
    "Los Angeles Lakers": 1.127,
    "Milwaukee Bucks": 1.133,
    "Minnesota Timberwolves": 1.135,
    "Phoenix Suns": 1.137,
    "New York Knicks": 1.138,
    "Los Angeles Clippers": 1.141,
    "Memphis Grizzlies": 1.147,
    "Charlotte Hornets": 1.149,
    "Utah Jazz": 1.150,
    "Toronto Raptors": 1.152,
    "Sacramento Kings": 1.153,
    "Washington Wizards": 1.167,
    "New Orleans Pelicans": 1.226,
    "Brooklyn Nets": 1.249,
}

# -------------------- VALID MARKETS --------------------
VALID_MARKETS = [
    "player_points","player_rebounds","player_assists","player_threes",
    "player_blocks","player_steals","player_blocks_steals","player_turnovers",
    "player_points_rebounds_assists","player_points_rebounds",
    "player_points_assists","player_rebounds_assists",
    "player_field_goals","player_frees_made","player_frees_attempts",
]

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket","player_first_team_basket","player_double_double",
    "player_triple_double","player_points_q1","player_rebounds_q1","player_assists_q1",
}

ODDS_SPORT = "basketball_nba"

# ------------------- UTILITIES -------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n).lower()

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS):
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean()), draws

# ------------------- NAME ↔ ID INDEX (optional; for normalization help) -------------------
@st.cache_data(show_spinner=False)
def _players_index():
    df = pd.DataFrame(nba_players.get_players())
    df["name_norm"] = df["full_name"].apply(normalize_name)
    return df[["id","full_name","name_norm"]]

# ------------------- SEASON AVERAGES (FAST, SINGLE CALL) -------------------
@st.cache_data(show_spinner=False)
def fetch_season_averages(season: str) -> pd.DataFrame:
    """
    League-wide per-game season averages for the given season.
    No fallbacks. If a player isn't in here, we skip them.
    """
    time.sleep(0.25 + random.random()*0.15)  # gentle rate-limit buffer
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, per_mode_detailed="PerGame", timeout=20
    ).get_data_frames()[0]

    # Standardize columns we'll use
    # Typical columns: PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME,
    # GP, MIN, PTS, REB, AST, STL, BLK, TOV, FGM, FG3M, FTM, FTA, ...
    df = resp.copy()

    # Keep only required columns
    keep = ["PLAYER_ID","PLAYER_NAME","TEAM_NAME",
            "PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA","GP"]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan

    # Normalize and set keys
    df["player_norm"] = df["PLAYER_NAME"].apply(normalize_name)
    # Compute simple SD proxy from per-game variance proxy (approx):
    # We don't have per-game square sum; use a conservative sd floor from per-game dispersion:
    # sd ≈ max(0.75, 0.3 * mean) — avoids zero-variance props. Feel free to tweak.
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        mu = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["mu_"+col] = mu
        df["sd_"+col] = np.maximum(0.75, 0.30 * mu)

    return df[["player_norm","PLAYER_NAME","TEAM_NAME","GP"] +
              [f"mu_{c}" for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]] +
              [f"sd_{c}" for c in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]]
            ].reset_index(drop=True)

# ------------------- ODDS API -------------------
def odds_get(url: str, params: dict):
    try:
        r = requests.get(url, params=params, timeout=25)
        return r.json()
    except:
        return {}

def list_events(api_key, lookahead, region):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "regions": region, "daysFrom": 0, "daysTo": lookahead if lookahead > 0 else 1}
    )

def fetch_event_props(api_key, event_id, region, markets):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
    )

# ------------------- UI SELECTION -------------------
st.markdown(f"### 1) Season locked to {SEASON} (no fallbacks)")
colA, colB = st.columns([1,1], gap="small")
with colA:
    api_key = st.text_input("Odds API Key", type="password")
with colB:
    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared — reloading…")
        st.experimental_rerun()

st.markdown("### 2) Select region, lookahead & markets")
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 1, 7, 1)
markets = st.multiselect("Markets", VALID_MARKETS, default=VALID_MARKETS)

# Preload season table once (fast & reused)
if "season_df" not in st.session_state:
    st.session_state["season_df"] = fetch_season_averages(SEASON)

season_df = st.session_state["season_df"]

# ------------------- LOAD GAMES -------------------
events = list_events(api_key, lookahead, region) if api_key else []

if not events:
    st.stop()

# robust label build
event_labels = []
for e in events:
    away = e.get("away_team") or (e.get("teams",[None,None])[0] or "Away")
    home = e.get("home_team") or (e.get("teams",[None,None])[1] or "Home")
    date = e.get("commence_time","")
    event_labels.append(f"{away} @ {home} — {date}")

pick = st.selectbox("Select Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]
home_team = event.get("home_team")
away_team = event.get("away_team")

# ****************** 3) BUILD PLAYER PROJECTIONS (SEASON AVERAGES) ******************
st.markdown("### 3) Build Player Projections (Season Averages)")
if st.button("📥 Build Projections"):
    prop_preview = fetch_event_props(api_key, event_id, region, markets)
    player_names = set()

    for bk in prop_preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE:
                continue
            for o in m.get("outcomes", []) or []:
                nm = normalize_name(o.get("description"))
                if nm:
                    player_names.add(nm)

    # Join player list to season averages by normalized name (no fallbacks)
    joined = season_df.merge(
        pd.DataFrame({"player_norm": sorted(player_names)}),
        on="player_norm",
        how="inner"
    )

    if joined.empty:
        st.warning("No matching season-average players found for this game with the selected markets.")
    else:
        # Keep only what we need & display
        out = joined.rename(columns={"PLAYER_NAME":"Player","TEAM_NAME":"Team"})
        st.session_state["proj"] = out
        st.dataframe(out[["Player","Team","GP","mu_PTS","mu_REB","mu_AST","mu_STL","mu_BLK","mu_TOV","mu_FG3M","mu_FGM","mu_FTM","mu_FTA"]],
                     use_container_width=True)

# ****************** 4) SIMULATE GAME PROPS ******************
st.markdown("### 4) Run Simulation (10,000 Sims)")
if st.button("🎯 Simulate"):
    proj = st.session_state.get("proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build projections first.")
        st.stop()

    props = fetch_event_props(api_key, event_id, region, markets)
    proj = proj.set_index("player_norm")

    sim_rows = []

    for bk in props.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey not in VALID_MARKETS:  # ignore unsupported/hidden
                continue

            for o in m.get("outcomes", []) or []:
                name = normalize_name(o.get("description"))
                if name not in proj.index:
                    continue
                row = proj.loc[name]
                team = row["Team"]

                # Determine opponent (no fallback)
                if team == home_team:
                    opponent = away_team
                elif team == away_team:
                    opponent = home_team
                else:
                    opponent = "UNKNOWN"  # as requested, do not fallback/guess

                def_mult = DEF_RATINGS.get(opponent, 1.0)

                # Map mu/sd by market
                def grab(stat):
                    return float(row[f"mu_{stat}"]), float(row[f"sd_{stat}"])

                line = float(o["point"])
                side = o["name"]

                if   mkey=="player_points":   mu,sd = grab("PTS")
                elif mkey=="player_rebounds": mu,sd = grab("REB")
                elif mkey=="player_assists":  mu,sd = grab("AST")
                elif mkey=="player_threes":   mu,sd = grab("FG3M")
                elif mkey=="player_blocks":   mu,sd = grab("BLK")
                elif mkey=="player_steals":   mu,sd = grab("STL")
                elif mkey=="player_turnovers":mu,sd = grab("TOV")
                elif mkey=="player_field_goals": mu,sd = grab("FGM")
                elif mkey=="player_frees_made":  mu,sd = grab("FTM")
                elif mkey=="player_frees_attempts": mu,sd = grab("FTA")
                elif mkey=="player_points_rebounds_assists":
                    mu = float(row["mu_PTS"] + row["mu_REB"] + row["mu_AST"])
                    sd = float(np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2))
                elif mkey=="player_points_rebounds":
                    mu = float(row["mu_PTS"] + row["mu_REB"])
                    sd = float(np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2))
                elif mkey=="player_points_assists":
                    mu = float(row["mu_PTS"] + row["mu_AST"])
                    sd = float(np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2))
                elif mkey=="player_rebounds_assists":
                    mu = float(row["mu_REB"] + row["mu_AST"])
                    sd = float(np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2))
                elif mkey=="player_blocks_steals":
                    mu = float(row["mu_BLK"] + row["mu_STL"])
                    sd = float(np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2))
                else:
                    continue

                # Defense adjustment on mean only
                mu_adj = mu * def_mult

                p_over, draws = t_over_prob(mu_adj, sd, line, SIM_TRIALS)
                proj_val = float(np.median(draws))
                win_prob = p_over if side=="Over" else (1 - p_over)

                sim_rows.append({
                    "Player": row["Player"],
                    "Team": team,
                    "Opponent": opponent,
                    "Market": mkey,
                    "Side": side,
                    "Line": round(line,2),
                    "Defense Mult": def_mult,
                    "Model Projection": round(proj_val,2),   # sim’s expected value (median)
                    "Win Prob %": round(win_prob*100,2),
                })

    results = pd.DataFrame(sim_rows)
    if results.empty:
        st.warning("No results returned for this game/markets.")
    else:
        results = results.sort_values(["Market","Win Prob %"], ascending=[True, False]).reset_index(drop=True)
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "⬇️ Download results CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="nba_props_sim_results.csv",
            mime="text/csv",
        )

# ****************** 5) FULL SLATE MODE — BEST VALUE ACROSS ALL GAMES ******************
st.markdown("### 5) Full Slate: Best Value Across All Games")
if st.button("📊 Run Full Slate"):
    all_events = list_events(api_key, lookahead, region)
    if not all_events:
        st.error("No games returned.")
        st.stop()

    master = []

    for ev in all_events:
        eid = ev["id"]
        h = ev.get("home_team")
        a = ev.get("away_team")

        props = fetch_event_props(api_key, eid, region, markets)
        if not props:
            continue

        # Gather players for this game from props
        names = set()
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") in UNSUPPORTED_MARKETS_HIDE:
                    continue
                for o in m.get("outcomes", []) or []:
                    nm = normalize_name(o.get("description"))
                    if nm:
                        names.add(nm)

        # Join to season averages
        proj = season_df.merge(pd.DataFrame({"player_norm": sorted(names)}),
                               on="player_norm", how="inner")
        if proj.empty:
            continue
        proj = proj.rename(columns={"PLAYER_NAME":"Player","TEAM_NAME":"Team"}).set_index("player_norm")

        # Simulate props
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                mkey = m.get("key")
                if mkey not in VALID_MARKETS:
                    continue

                for o in m.get("outcomes", []) or []:
                    nm = normalize_name(o.get("description"))
                    if nm not in proj.index or pd.isna(o.get("point")):
                        continue

                    row = proj.loc[nm]
                    team = row["Team"]
                    if team == h:
                        opp = a
                    elif team == a:
                        opp = h
                    else:
                        opp = "UNKNOWN"  # no fallback

                    def_mult = DEF_RATINGS.get(opp, 1.0)
                    line = float(o["point"])

                    def grab(stat):
                        return float(row[f"mu_{stat}"]), float(row[f"sd_{stat}"])

                    if   mkey=="player_points":   mu,sd = grab("PTS")
                    elif mkey=="player_rebounds": mu,sd = grab("REB")
                    elif mkey=="player_assists":  mu,sd = grab("AST")
                    elif mkey=="player_threes":   mu,sd = grab("FG3M")
                    elif mkey=="player_blocks":   mu,sd = grab("BLK")
                    elif mkey=="player_steals":   mu,sd = grab("STL")
                    elif mkey=="player_turnovers":mu,sd = grab("TOV")
                    elif mkey=="player_field_goals": mu,sd = grab("FGM")
                    elif mkey=="player_frees_made":  mu,sd = grab("FTM")
                    elif mkey=="player_frees_attempts": mu,sd = grab("FTA")
                    elif mkey=="player_points_rebounds_assists":
                        mu = float(row["mu_PTS"] + row["mu_REB"] + row["mu_AST"])
                        sd = float(np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2))
                    elif mkey=="player_points_rebounds":
                        mu = float(row["mu_PTS"] + row["mu_REB"])
                        sd = float(np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2))
                    elif mkey=="player_points_assists":
                        mu = float(row["mu_PTS"] + row["mu_AST"])
                        sd = float(np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2))
                    elif mkey=="player_rebounds_assists":
                        mu = float(row["mu_REB"] + row["mu_AST"])
                        sd = float(np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2))
                    elif mkey=="player_blocks_steals":
                        mu = float(row["mu_BLK"] + row["mu_STL"])
                        sd = float(np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2))
                    else:
                        continue

                    mu_adj = mu * def_mult
                    p_over, draws = t_over_prob(mu_adj, sd, line, SIM_TRIALS)
                    proj_val = float(np.median(draws))
                    edge = proj_val - line

                    master.append({
                        "Game": f"{a} @ {h}",
                        "Player": row["Player"],
                        "Team": team,
                        "Opponent": opp,
                        "Market": mkey,
                        "Line": round(line,2),
                        "Model Projection": round(proj_val,2),
                        "Edge": round(edge,2),
                        "Defense Mult": def_mult,
                    })

    full = pd.DataFrame(master)
    if full.empty:
        st.warning("No full-slate results.")
    else:
        full = full.sort_values("Edge", ascending=False).reset_index(drop=True)
        st.dataframe(full.head(50), use_container_width=True)
        st.download_button(
            "⬇️ Download full-slate CSV",
            full.to_csv(index=False).encode("utf-8"),
            file_name="nba_full_slate_edges.csv",
            mime="text/csv",
        )
        st.markdown("### 📈 Biggest Edges (Top 20)")
        st.bar_chart(full.head(20).set_index("Player")["Edge"])
