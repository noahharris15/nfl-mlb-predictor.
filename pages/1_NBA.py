# NBA Player Props — Sharpened Model (30,000 sims + Last 5-game weighting)

import re, unicodedata, time, random
import numpy as np
import pandas as pd
import requests
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Prop Model — Sharpened Simulation")

SIM_TRIALS = 30_000

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

def fetch_player_gamelog_df(pid, season):
    time.sleep(0.25)
    gl = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season")
    return gl.get_data_frames()[0]

def agg_stats(df):
    g = len(df)
    if g == 0: return {"g":0}
    stats = {"g":g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        stats["mu_"+col] = x.mean()
        stats["sd_"+col] = sample_sd(x.sum(), (x**2).sum(), g)
    # Last 5 game means
    last5 = df.sort_values("GAME_DATE").tail(5)
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        stats["mu_last5_"+col] = pd.to_numeric(last5[col], errors="coerce").fillna(0).mean()
    return stats

@st.cache_data(show_spinner=False)
def _players_index():
    df = pd.DataFrame(nba_players.get_players())
    df["norm"] = df["full_name"].apply(normalize_name)
    return df[["id","full_name","norm"]]

def find_player_id(n):
    df = _players_index()
    n = normalize_name(n)
    hit = df[df["norm"]==n]
    if not hit.empty: return int(hit.iloc[0]["id"])
    parts = n.split()
    if len(parts)==2:
        cand = df[df["norm"].str.contains(parts[-1])]
        if not cand.empty: return int(cand.iloc[0]["id"])
    return None

def t_sim(mu, sd, line, trials):
    sd = max(sd, 1e-6)
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    win_over = float((draws > line).mean())
    proj_med = float(np.median(draws))
    return win_over, proj_med

# ------------------ UI ------------------

api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead Days", 0, 7, 1)
markets = st.multiselect("Markets", VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE)), VALID_MARKETS)

def api_get(url, p): return requests.get(url, params=p, timeout=20).json()

def list_events():
    return api_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
                   {"apiKey": api_key, "daysFrom":0, "daysTo":lookahead, "regions":region})

def props_for(eid, mkts):
    return api_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
                   {"apiKey": api_key, "regions":region, "markets":",".join(mkts), "oddsFormat":"american"})

if not api_key:
    st.stop()

events = list_events()
if not events:
    st.error("No games found yet.")
    st.stop()

ev_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e["commence_time"]}' for e in events]
pick = st.selectbox("Select Game", ev_labels)
event = events[ev_labels.index(pick)]

# ------------------ BUILD PLAYER STATS ------------------

if st.button("📥 Build Player Averages"):

    preview = props_for(event["id"], markets)
    player_names = set()

    for bk in preview.get("bookmakers", []) or []:
        for m in bk.get("markets", []) or []:
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []) or []:
                desc = o.get("description")
                if isinstance(desc, str):
                    nm = normalize_name(desc)
                    if nm and len(nm) > 2:
                        player_names.add(nm)

    if not player_names:
        st.error("No player props found. Try points/assists/rebounds.")
        st.stop()

    rows = []
    for pn in sorted(player_names):
        pid = find_player_id(pn)
        if not pid: continue

        df = fetch_player_gamelog_df(pid, "2025-26")
        stats = agg_stats(df)
        if stats["g"] == 0:
            df = fetch_player_gamelog_df(pid, "2024-25")
            stats = agg_stats(df)
        if stats["g"] == 0: continue

        rows.append({"Player":pn, **stats})

    proj = pd.DataFrame(rows)
    proj["norm"] = proj["Player"].apply(normalize_name)

    st.session_state["proj"] = proj
    st.success("✅ Player averages and last-5 form loaded")
    st.dataframe(proj)

# ------------------ SIM ------------------

if st.button("▶️ Run 30K Sims"):

    proj = st.session_state.get("proj")
    if proj is None or proj.empty: st.stop()

    data = props_for(event["id"], markets)
    proj = proj.set_index("norm")

    sim_rows=[]
    for bk in data.get("bookmakers",[]):
        for m in bk.get("markets",[]):
            if m["key"] in UNSUPPORTED_MARKETS_HIDE or m["key"] not in VALID_MARKETS:
                continue
            for o in m.get("outcomes",[]):
                nm = normalize_name(o.get("description"))
                if nm in proj.index:
                    sim_rows.append({
                        "market":m["key"], "player":nm,
                        "side":o["name"], "line":float(o["point"])
                    })

    results=[]
    for r in sim_rows:
        row = proj.loc[r["player"]]
        mkt, side, line = r["market"], r["side"], r["line"]

        smap = {
            "player_points":"PTS","player_rebounds":"REB","player_assists":"AST",
            "player_threes":"FG3M","player_blocks":"BLK","player_steals":"STL",
            "player_turnovers":"TOV"
        }

        # get base season averages
        if mkt in smap:
            raw_mu = row["mu_"+smap[mkt]]
            raw_sd = row["sd_"+smap[mkt]]
            last5_mu = row["mu_last5_"+smap[mkt]]
        elif mkt=="player_points_rebounds_assists":
            raw_mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
            raw_sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2+row["sd_AST"]**2)
            last5_mu = row["mu_last5_PTS"]+row["mu_last5_REB"]+row["mu_last5_AST"]
        elif mkt=="player_points_rebounds":
            raw_mu = row["mu_PTS"]+row["mu_REB"]
            raw_sd = np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2)
            last5_mu = row["mu_last5_PTS"]+row["mu_last5_REB"]
        elif mkt=="player_points_assists":
            raw_mu = row["mu_PTS"]+row["mu_AST"]
            raw_sd = np.sqrt(row["sd_PTS"]**2+row["sd_AST"]**2)
            last5_mu = row["mu_last5_PTS"]+row["mu_last5_AST"]
        elif mkt=="player_rebounds_assists":
            raw_mu = row["mu_REB"]+row["mu_AST"]
            raw_sd = np.sqrt(row["sd_REB"]**2+row["sd_AST"]**2)
            last5_mu = row["mu_last5_REB"]+row["mu_last5_AST"]
        else:
            continue

        # ✅ Sharpened projection: blend season & last-5
        mu = (raw_mu * 0.70) + (last5_mu * 0.30)
        sd = raw_sd

        p_over, proj_med = t_sim(mu, sd, line, SIM_TRIALS)
        win = p_over if side=="Over" else 1-p_over

        results.append({
            "Player":row["Player"],
            "Market":mkt,
            "Side":side,
            "Line":line,
            "Model Projection":round(proj_med,2),
            "Win %":round(win*100,2),
        })

    df = pd.DataFrame(results).sort_values(["Market","Win %"], ascending=[True,False])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "nba_sharp_results.csv")
