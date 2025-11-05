# NBA Player Props — Sharpened Model (30K Sims + Recent-Form Weighting)

import re, unicodedata, time, random
import numpy as np
import pandas as pd
import requests
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Prop Model — Sharpened Simulation (30K Sims)")

SIM_TRIALS = 30_000

VALID_MARKETS = [
    "player_points","player_rebounds","player_assists","player_threes",
    "player_blocks","player_steals","player_blocks_steals","player_turnovers",
    "player_points_rebounds_assists","player_points_rebounds","player_points_assists",
    "player_rebounds_assists","player_field_goals","player_frees_made","player_frees_attempts"
]

HIDE_MARKETS = {
    "player_first_basket","player_first_team_basket","player_double_double",
    "player_triple_double","player_points_q1","player_rebounds_q1","player_assists_q1"
}

SPORT_KEY = "basketball_nba"

# ------------------ HELPERS ------------------

def normalize_name(n: str) -> str:
    if not n: return ""
    n = str(n).split("(")[0]
    n = re.sub(r"[.,']", " ", n).replace("-", " ").strip()
    return "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining()).lower()

def sample_sd(sum_x, sum_x2, g):
    if g <= 1: return 0.0001
    mean = sum_x / g
    var = (sum_x2 / g) - mean**2
    var = var * (g / (g - 1))
    return max(np.sqrt(max(var, 1e-9)), 0.0001)

def fetch_games(pid, season):
    time.sleep(0.25)
    data = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season")
    return data.get_data_frames()[0]

def build_stats(df):
    g = len(df)
    if g == 0: return {"g":0}

    stats = {"g":g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        stats[f"mu_{col}"] = x.mean()
        stats[f"sd_{col}"] = sample_sd(x.sum(), (x**2).sum(), g)

    last5 = df.sort_values("GAME_DATE").tail(5)
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        stats[f"mu_last5_{col}"] = pd.to_numeric(last5[col], errors="coerce").fillna(0).mean()

    return stats

@st.cache_data(show_spinner=False)
def _players_index():
    df = pd.DataFrame(nba_players.get_players())
    df["norm"] = df["full_name"].apply(normalize_name)
    return df[["id","full_name","norm"]]

def find_player_id(n):
    df = _players_index()
    n = normalize_name(n)
    row = df[df["norm"] == n]
    return int(row.iloc[0]["id"]) if not row.empty else None

def sim(mu, sd, line):
    sd = max(sd, 1e-6)
    draws = mu + sd * np.random.standard_t(df=5, size=SIM_TRIALS)
    proj = float(np.median(draws))
    win_over = float((draws > line).mean())
    return proj, win_over

# ------------------ UI ------------------

api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead Days", 0, 7, 1)
markets = st.multiselect("Markets", VALID_MARKETS + sorted(list(HIDE_MARKETS)), VALID_MARKETS)

def api(url, params): return requests.get(url, params=params, timeout=20).json()

def list_games():
    return api(
        f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events",
        {"apiKey": api_key, "daysFrom":0, "daysTo":lookahead, "regions":region}
    )

def get_props(eid, mkts):
    return api(
        f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{eid}/odds",
        {"apiKey": api_key, "regions":region, "markets":",".join(mkts)}
    )

if not api_key: st.stop()

events = list_games()
if not events: st.stop()

ev_labels = [f'{e["away_team"]} @ {e["home_team"]}' for e in events]
pick = st.selectbox("Select Game", ev_labels)
event = events[ev_labels.index(pick)]

# ------------------ BUILD PLAYER DATA ------------------

if st.button("Build Player Stats"):
    preview = get_props(event["id"], markets)
    players = set()

    for bk in preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] in HIDE_MARKETS: continue
            for o in m.get("outcomes", []):
                p = normalize_name(o.get("description"))
                if p and len(p) > 2: players.add(p)

    if not players:
        st.error("No player props found — select core props like points/rebounds.")
        st.stop()

    rows = []
    for name in sorted(players):
        pid = find_player_id(name)
        if not pid: continue

        df = fetch_games(pid, "2025-26")
        stats = build_stats(df)

        if stats["g"] == 0:
            df = fetch_games(pid, "2024-25")
            stats = build_stats(df)

        if stats["g"] == 0: continue

        rows.append({"Player":name, **stats})

    proj = pd.DataFrame(rows)
    proj["norm"] = proj["Player"].apply(normalize_name)

    st.session_state["proj"] = proj
    st.dataframe(proj)
    st.success("✅ Player data loaded")

# ------------------ SIMULATE ------------------

if st.button("Run Model (30K Sims)"):

    proj = st.session_state.get("proj")
    if proj is None or proj.empty: st.stop()

    data = get_props(event["id"], markets)
    proj = proj.set_index("norm")

    results = []

    for bk in data.get("bookmakers",[]):
        for m in bk.get("markets",[]):
            if m["key"] in HIDE_MARKETS or m["key"] not in VALID_MARKETS: continue

            for o in m.get("outcomes",[]):
                nm = normalize_name(o.get("description"))
                if nm not in proj.index: continue

                line = float(o["point"])
                side = o["name"]
                row = proj.loc[nm]

                # ----- Market-stat mapping -----
                smap = {
                    "player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
                    "player_threes":"FG3M","player_blocks":"BLK","player_steals":"STL",
                    "player_turnovers":"TOV"
                }

                if m["key"] in smap:
                    mu = row[f"mu_{smap[m['key']]}"]
                    sd = row[f"sd_{smap[m['key']]}"]
                    last5 = row[f"mu_last5_{smap[m['key']]}"]
                elif m["key"] == "player_points_rebounds_assists":
                    mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_REB"]+row["mu_last5_AST"]
                elif m["key"] == "player_points_rebounds":
                    mu = row["mu_PTS"]+row["mu_REB"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_REB"]
                elif m["key"] == "player_points_assists":
                    mu = row["mu_PTS"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_PTS"]+row["mu_last5_AST"]
                elif m["key"] == "player_rebounds_assists":
                    mu = row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_REB"]**2+row["sd_AST"]**2)
                    last5 = row["mu_last5_REB"]+row["mu_last5_AST"]
                else:
                    continue

                # ✅ Sharpen projection using recent form (30% weight)
                mu = (mu * 0.70) + (last5 * 0.30)

                proj_val, over_prob = sim(mu, sd, line)
                win_prob = over_prob if side=="Over" else 1-over_prob

                results.append({
                    "Player":row["Player"],
                    "Market":m["key"],
                    "Side":side,
                    "Line":line,
                    "Model Projection":round(proj_val,2),
                    "Win %":round(win_prob*100,2)
                })

    df = pd.DataFrame(results).sort_values(["Market","Win %"], ascending=[True,False])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "nba_results.csv")
