# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 10k sims
# EXACT SAME CODE — only sharper sim added

import re, unicodedata, datetime as dt, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Player Props — Odds API + NBA Stats (live)")

SIM_TRIALS = 10000

VALID_MARKETS = [
    "player_points","player_rebounds","player_assists","player_threes",
    "player_blocks","player_steals","player_blocks_steals","player_turnovers",
    "player_points_rebounds_assists","player_points_rebounds",
    "player_points_assists","player_rebounds_assists","player_field_goals",
    "player_frees_made","player_frees_attempts",
]

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket","player_first_team_basket","player_double_double",
    "player_triple_double","player_points_q1","player_rebounds_q1","player_assists_q1",
}

ODDS_SPORT = "basketball_nba"

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n):
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n).lower()

def sample_sd(sum_x, sum_x2, g, floor=0.0):
    if g <= 1: return float("nan")
    mean = sum_x / g
    var  = (sum_x2 / g) - (mean**2)
    var  = var * (g / (g - 1))
    return float(max(np.sqrt(max(var, 1e-9)), floor))

@st.cache_data(show_spinner=False)
def _players_index():
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    df["name_norm"]  = df["full_name"].apply(normalize_name)
    df["first_norm"] = df["first_name"].apply(normalize_name)
    df["last_norm"]  = df["last_name"].apply(normalize_name)
    return df[["id","full_name","name_norm","first_norm","last_norm"]]

def find_player_id_by_name(name):
    df = _players_index()
    n = normalize_name(name)
    parts = n.split()

    hit = df.loc[df["name_norm"] == n]
    if not hit.empty: return int(hit.iloc[0]["id"])

    if len(parts) == 2:
        first, last = parts
        cand = df.loc[df["last_norm"].str.startswith(last)]
        if cand.empty: cand = df.loc[df["last_norm"].str.contains(last)]
        if not cand.empty:
            cand = cand.loc[cand["first_norm"].str.startswith(first[:1]) | cand["first_norm"].str.contains(first)]
            if not cand.empty: return int(cand.iloc[0]["id"])

    last = parts[-1] if parts else n
    cand = df.loc[df["name_norm"].str.contains(last)]
    if not cand.empty: return int(cand.iloc[0]["id"])
    return None

def fetch_player_gamelog_df(pid, season, season_type):
    time.sleep(0.25 + random.random()*0.15)
    df = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star=season_type).get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def agg_full_season(df):
    g = len(df)
    if g == 0: return {"g":0}

    def s(col):
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        return float(x.sum()), float((x**2).sum())

    sums = {}; out = {"g":g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        sums[col], sums["sq_"+col] = s(col)
        out["mu_"+col] = sums[col] / g
        out["sd_"+col] = sample_sd(sums[col], sums["sq_"+col], g)

    ### 🔥 Add last-5 form
    last5 = df.sort_values("GAME_DATE").tail(5)
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        out["mu_last5_"+col] = pd.to_numeric(last5[col], errors="coerce").fillna(0).mean()

    return out

st.markdown("### 1) Season Locked to 2025-26")
season_locked = "2025-26"

api_key = st.text_input("Odds API Key", value="", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets_pickable = VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE))
markets = st.multiselect("Markets to fetch", markets_pickable, default=VALID_MARKETS)

def odds_get(url, params): return requests.get(url, params=params, timeout=25).json()

def list_events():
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "daysFrom":0, "daysTo":lookahead, "regions":region}
    )

if not api_key: st.stop()
events = list_events()
if not events: st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]}' for e in events]
pick = st.selectbox("Game", event_labels)
event = next(e for e in events if f'{e["away_team"]} @ {e["home_team"]}' == pick)
event_id = event["id"]

def fetch_event_props(eid, mkts):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
        {"apiKey": api_key, "regions":region, "markets":",".join(mkts)}
    )

st.markdown("### 3) Build per-player season averages")
build = st.button("📥 Build NBA projections")

if build:
    preview = fetch_event_props(event_id, list(set(markets)))
    player_names = set()

    for bk in preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm: player_names.add(nm)

    rows = []
    for pn in sorted(player_names):
        pid = find_player_id_by_name(pn)
        if not pid: continue
        df = fetch_player_gamelog_df(pid, season_locked, "Regular Season")
        stats = agg_full_season(df)

        if stats["g"] == 0:
            df = fetch_player_gamelog_df(pid, "2024-25", "Regular Season")
            stats = agg_full_season(df)

        if stats["g"] == 0: continue

        rows.append({"Player":pn, **stats})

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    st.session_state["nba_proj"] = proj
    st.dataframe(proj.head(30))

st.markdown("### 4) Fetch lines & simulate (sharper)")
go = st.button("Run Simulation")

if go:
    proj = st.session_state.get("nba_proj", pd.DataFrame())
    if proj.empty: st.stop()

    data = fetch_event_props(event_id, list(set(markets)))
    proj = proj.set_index("player_norm")

    results = []

    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey in UNSUPPORTED_MARKETS_HIDE or mkey not in VALID_MARKETS: continue
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                if name not in proj.index: continue

                row = proj.loc[name]
                line = float(o["point"])
                side = o["name"]

                # map stats
                def get(col):
                    return row[f"mu_{col}"], row[f"sd_{col}"], row[f"mu_last5_{col}"]

                if mkey=="player_points":      mu,sd,last5 = get("PTS")
                elif mkey=="player_rebounds":  mu,sd,last5 = get("REB")
                elif mkey=="player_assists":   mu,sd,last5 = get("AST")
                elif mkey=="player_threes":    mu,sd,last5 = get("FG3M")
                elif mkey=="player_blocks":    mu,sd,last5 = get("BLK")
                elif mkey=="player_steals":    mu,sd,last5 = get("STL")
                elif mkey=="player_turnovers": mu,sd,last5 = get("TOV")
                elif mkey=="player_field_goals": mu,sd,last5 = get("FGM")
                elif mkey=="player_frees_made": mu,sd,last5 = get("FTM")
                elif mkey=="player_frees_attempts": mu,sd,last5 = get("FTA")

                elif mkey=="player_points_rebounds_assists":
                    mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_REB"]+row["mu_last5_AST"]

                elif mkey=="player_points_rebounds":
                    mu = row["mu_PTS"]+row["mu_REB"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_REB"]

                elif mkey=="player_points_assists":
                    mu = row["mu_PTS"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_AST"]

                elif mkey=="player_rebounds_assists":
                    mu = row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_REB"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_REB"]+row["mu_last5_AST"]

                else:
                    continue

                ### 🔥 Sharpen the projection
                mu = (mu * 0.70) + (last5 * 0.30)

                draws = mu + sd * np.random.standard_t(df=5, size=SIM_TRIALS)
                proj_val = float(np.median(draws))  # sharper than mean
                win_prob = float((draws > line).mean()) if side=="Over" else float((draws < line).mean())

                results.append({
                    "Player":row["Player"],
                    "Market":mkey,
                    "Side":side,
                    "Line":line,
                    "Model Projection":round(proj_val,2),
                    "Win %":round(win_prob*100,2)
                })

    df = pd.DataFrame(results).sort_values(["Market","Win %"], ascending=[True,False])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "nba_sharp_results.csv")
