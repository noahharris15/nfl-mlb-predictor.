# NBA Player Props — Odds API + NBA Stats — SHARP MODEL VERSION
# SAME UI — SAME STRUCTURE — JUST BETTER PROJECTIONS

import re, unicodedata, datetime as dt, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Player Props — Sharpened Simulation (30K sims + Last-5 blend)")

SIM_TRIALS = 30000  # sharper sim than 10k

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

# ------------------ Utilities ------------------

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining())

def normalize_name(n: str) -> str:
    n = str(n or "").split("(")[0]
    n = re.sub(r"[.,']", " ", n).replace("-", " ").strip()
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n).lower()

def t_sim(mu, sd, line):
    sd = max(sd, 1e-6)
    draws = mu + sd * np.random.standard_t(df=5, size=SIM_TRIALS)
    p_over = float((draws > line).mean())
    proj = float(np.median(draws))  # median = winner
    return proj, p_over

def sample_sd(sum_x, sum_x2, g, floor=0.0):
    if g <= 1: return 0.0001
    mean = sum_x / g
    var = (sum_x2 / g) - (mean**2)
    var = var * (g / (g - 1))
    return float(max(np.sqrt(max(var, 1e-9)), floor))

# ------------------ NBA Stats ------------------

@st.cache_data(show_spinner=False)
def _players_index():
    df = pd.DataFrame(nba_players.get_players())
    df["name_norm"] = df["full_name"].apply(normalize_name)
    df["first_norm"] = df["first_name"].apply(normalize_name)
    df["last_norm"] = df["last_name"].apply(normalize_name)
    return df

def find_player_id_by_name(name: str):
    df = _players_index()
    n = normalize_name(name)
    parts = n.split()

    hit = df.loc[df["name_norm"] == n]
    if not hit.empty: return int(hit.iloc[0]["id"])

    if len(parts) == 2:
        first, last = parts
        cand = df.loc[df["last_norm"].str.contains(last)]
        if not cand.empty:
            cand = cand.loc[cand["first_norm"].str.contains(first[:1])]
            if not cand.empty: return int(cand.iloc[0]["id"])

    cand = df.loc[df["name_norm"].str.contains(parts[-1])]
    if not cand.empty: return int(cand.iloc[0]["id"])

    return None

def fetch_player_gamelog_df(pid, season):
    time.sleep(0.25 + random.random()*0.15)
    df = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def agg_full_season(df):
    g = len(df)
    if g == 0: return {"g":0}

    stats = {"g":g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        xs = pd.to_numeric(df[col], errors="coerce").fillna(0)
        stats[f"mu_{col}"] = xs.mean()
        stats[f"sd_{col}"] = sample_sd(xs.sum(), (xs**2).sum(), g)

    last5 = df.sort_values("GAME_DATE").tail(5)
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        stats[f"mu_last5_{col}"] = pd.to_numeric(last5[col], errors="coerce").fillna(0).mean()

    return stats

# ------------------ UI ------------------

st.markdown("### 1) Season: 2025-26 + auto fallback to 2024-25")

api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, 1)
markets = st.multiselect("Props", VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE)), VALID_MARKETS)

def odds_get(url, p): return requests.get(url, params=p, timeout=20).json()

def list_events():
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "daysFrom":0, "daysTo":lookahead, "regions":region}
    )

if not api_key: st.stop()

events = list_events()
if not events: st.stop()


def fetch_event_props(eid, mkts):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
        {"apiKey": api_key, "regions":region, "markets":",".join(mkts)}
    )

# ------------------ Build player averages ------------------

if st.button("📥 Build NBA projections"):
    preview = fetch_event_props(event_id, list(set(markets)))
    player_names = set()

    for bk in preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm: player_names.add(nm)

    rows = []
    for pn in sorted(player_names):
        pid = find_player_id_by_name(pn)
        if not pid: continue

        df = fetch_player_gamelog_df(pid, "2025-26")
        stats = agg_full_season(df)

        if stats["g"] == 0:
            df = fetch_player_gamelog_df(pid, "2024-25")
            stats = agg_full_season(df)

        if stats["g"] == 0: continue

        rows.append({"Player":pn, **stats})

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    st.session_state["nba_proj"] = proj
    st.dataframe(proj)

# ------------------ SIM ------------------

if st.button("Run Simulation (Sharper)"):
    proj = st.session_state.get("nba_proj", pd.DataFrame())
    if proj.empty: st.stop()

    data = fetch_event_props(event_id, list(set(markets)))

    proj = proj.set_index("player_norm")
    out = []

    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m["key"]
            if mkey in UNSUPPORTED_MARKETS_HIDE or mkey not in VALID_MARKETS: continue
            for o in m.get("outcomes", []):
                name = normalize_name(o["description"])
                if name not in proj.index: continue

                row = proj.loc[name]
                line = float(o["point"])
                side = o["name"]

                # map stat
                smap = {
                    "player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
                    "player_threes":"FG3M","player_blocks":"BLK","player_steals":"STL",
                    "player_turnovers":"TOV"
                }

                if mkey in smap:
                    col = smap[mkey]
                    mu = row[f"mu_{col}"]
                    sd = row[f"sd_{col}"]
                    last5 = row[f"mu_last5_{col}"]

                elif mkey == "player_points_rebounds_assists":
                    mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_REB"]+row["mu_last5_AST"]

                elif mkey == "player_points_rebounds":
                    mu = row["mu_PTS"]+row["mu_REB"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_REB"]

                elif mkey == "player_points_assists":
                    mu = row["mu_PTS"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_AST"]

                elif mkey == "player_rebounds_assists":
                    mu = row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_REB"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_REB"]+row["mu_last5_AST"]

                else:
                    continue

                # ✅ Sharper model: blend recent form
                mu = (mu * 0.70) + (last5 * 0.30)

                proj_val, p_over = t_sim(mu, sd, line)
                win = p_over if side=="Over" else 1-p_over

                out.append({
                    "Player":row["Player"],
                    "Market":mkey,
                    "Side":side,
                    "Line":line,
                    "Model Projection":round(proj_val,2),
                    "Win %":round(win*100,2),
                })

    df = pd.DataFrame(out).sort_values(["Market","Win %"], ascending=[True,False])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "nba_sharp_results.csv")
