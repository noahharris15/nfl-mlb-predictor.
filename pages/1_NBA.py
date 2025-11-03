# ------------------ NBA Player Props Live Model w/ Minutes + Defense Adjust ------------------

import re, unicodedata, datetime as dt, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Player Props Model — Projections, Defense & Injury Minutes")

SIM_TRIALS = 10_000

# Valid markets
VALID_MARKETS = [
    "player_points","player_rebounds","player_assists","player_threes",
    "player_blocks","player_steals","player_blocks_steals","player_turnovers",
    "player_points_rebounds_assists","player_points_rebounds",
    "player_points_assists","player_rebounds_assists",
    "player_field_goals","player_frees_made","player_frees_attempts"
]

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket","player_first_team_basket","player_double_double",
    "player_triple_double","player_points_q1","player_rebounds_q1","player_assists_q1"
}

ODDS_SPORT = "basketball_nba"

# -------- Helpers --------
def strip_accents(s): return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n):
    n = str(n or "").split("(")[0]
    n = re.sub(r"[.,'-]", "", n).replace("-", " ").strip()
    return strip_accents(n.lower())

def t_over_prob(mu, sd, line, trials=SIM_TRIALS):
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def sample_sd(sum_x, sum_x2, g, floor=0.0):
    if g<=1: return float("nan")
    mean = sum_x / g
    var = (sum_x2/g) - mean**2
    var = var * (g/(g-1))
    return float(max(np.sqrt(max(var,1e-9)), floor))

# -------- Players index --------
@st.cache_data(show_spinner=False)
def _players_index():
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    df["name_norm"] = df["full_name"].apply(normalize_name)
    return df[["id","full_name","name_norm"]]

def find_player_id_by_name(name):
    df = _players_index()
    name = normalize_name(name)
    hit = df[df["name_norm"]==name]
    if not hit.empty: return int(hit.iloc[0]["id"])
    part = name.split()
    if len(part)==2:
        cand = df[df["name_norm"].str.contains(part[-1])]
        if not cand.empty: return int(cand.iloc[0]["id"])
    return None

# -------- Fetch NBA Game Logs --------
def fetch_game_log(pid, season):
    time.sleep(0.25)
    df = playergamelog.PlayerGameLog(pid, season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def agg_season(df):
    g = df.shape[0]
    if g==0: return {"g":0}
    def s(c):
        x = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
        return float(x.sum()), float((x**2).sum())
    out={"g":g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA","MIN"]:
        sx, sx2 = s(col)
        out["mu_"+col] = sx/g
        out["sd_"+col] = sample_sd(sx, sx2, g, floor=0.0)
    return out

# -------- Defense vs Position (FantasyPros) --------
@st.cache_data(ttl=3600)
def get_defense_table():
    url="https://www.fantasypros.com/nba/defense-vs-position.php"
    try:
        df = pd.read_html(url)[0]
        df.columns = ["Team","PG","SG","SF","PF","C","Last3"]
        for p in ["PG","SG","SF","PF","C"]:
            df[p] = pd.to_numeric(df[p],errors="coerce")
            df[p] = (df[p]-df[p].mean())/df[p].std() * 0.15  # +/-15% scale
            df[p] = 1 + df[p]
        return df.set_index("Team")
    except:
        return pd.DataFrame(columns=["PG","SG","SF","PF","C"])

def get_position(name):
    try:
        url = f"https://www.basketball-reference.com/search/search.fcgi?search={name.replace(' ','+')}"
        txt = requests.get(url,timeout=6).text
        m = re.search(r"Position:\s([A-Z-]+)", txt)
        if m: return m.group(1).split("-")[0]
    except: pass
    return "SG"

# -------- Injury & minute trend --------
def fetch_recent_minutes(name):
    try:
        url=f"https://www.basketball-reference.com/search/search.fcgi?search={name.replace(' ','+')}"
        txt=requests.get(url,timeout=6).text
        # find player code
        idmatch=re.search(r"/players/[a-z]/([a-z0-9]+)\.html",txt)
        if not idmatch: return None
        pid = idmatch.group(1)
        gpage = requests.get(f"https://www.basketball-reference.com/players/{pid[0]}/{pid}.html", timeout=6).text
        tbl = pd.read_html(gpage, match="Game Logs")[0]
        recent = tbl.tail(5)
        mins = pd.to_numeric(recent["MP"],errors="coerce").mean()
        return mins
    except:
        return None

def injury_minute_scale(name, season_avg_min):
    recent = fetch_recent_minutes(name)
    if not recent: return 1
    if season_avg_min<5: return 1
    scale = recent/season_avg_min
    return max(0.50, min(1.15, scale)) # cap to avoid extremes

# -------- UI --------
st.subheader("Season: **2025-26** (auto fallback 24-25 if no games)")
api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)

# -------- Odds API --------
def odds_get(url,params):
    r=requests.get(url,params=params,timeout=25)
    return r.json()

def list_events(): 
    return odds_get(f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
                    {"apiKey":api_key,"daysFrom":0,"daysTo":lookahead,"regions":region})

def get_event_props(eid, mkts):
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
        {"apiKey":api_key,"regions":region,"markets":",".join(mkts),"oddsFormat":"american"}
    )

if not api_key:
    st.info("Enter Odds API key to load games.")
    st.stop()

events=list_events()
if not events:
    st.error("No games found")
    st.stop()

label=[f"{e['away_team']} @ {e['home_team']} — {e['commence_time']}" for e in events]
pick = st.selectbox("Select Game", label)
event = events[label.index(pick)]

# -------- Step 3: Build Season Stats --------
if st.button("Build Player Stats"):
    try:
        sample_props = get_event_props(event["id"], VALID_MARKETS)
    except:
        st.error("Could not fetch props"); st.stop()

    names=set()
    for bk in sample_props.get("bookmakers",[]):
        for m in bk.get("markets",[]):
            if m.get("key") not in VALID_MARKETS: continue
            for o in m.get("outcomes",[]) or []:
                names.add(normalize_name(o.get("description")))

    rows=[]
    for n in names:
        pid=find_player_id_by_name(n)
        if not pid: continue
        try:
            gl=fetch_game_log(pid,"2025-26"); ag=agg_season(gl)
            if ag["g"]==0:
                gl2=fetch_game_log(pid,"2024-25"); ag=agg_season(gl2)
        except: continue
        if ag["g"]==0: continue
        rows.append({"Player":n, **ag})

    df=pd.DataFrame(rows)
    st.session_state["proj"]=df
    st.success("✅ Player season projections loaded!")
    st.dataframe(df)

# -------- Step 4: Simulate --------
if st.button("Run Model"):
    proj=st.session_state.get("proj")
    if proj is None or proj.empty:
        st.warning("Run 'Build Player Stats' first"); st.stop()

    proj["norm"]=proj["Player"].apply(normalize_name)
    idx=proj.set_index("norm")

    try:
        data=get_event_props(event["id"], VALID_MARKETS)
    except:
        st.error("Props unavailable"); st.stop()

    def_table=get_defense_table()

    props=[]
    for bk in data.get("bookmakers",[]):
        for m in bk.get("markets",[]):
            if m.get("key") not in VALID_MARKETS: continue
            for o in m.get("outcomes",[]) or []:
                props.append({
                    "market":m["key"],
                    "player":normalize_name(o["description"]),
                    "side":o["name"],
                    "line":float(o["point"])
                })

    out=[]
    for r in props:
        name=r["player"]
        if name not in idx.index: continue
        row=idx.loc[name]

        mu_map={"player_points":"PTS","player_rebounds":"REB","player_assists":"AST","player_threes":"FG3M",
                "player_blocks":"BLK","player_steals":"STL","player_turnovers":"TOV"}
        if r["market"] in mu_map:
            stat=mu_map[r["market"]]
            mu=row["mu_"+stat]; sd=row["sd_"+stat]
        else:
            if r["market"]=="player_points_rebounds_assists":
                mu=row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                sd=np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2+row["sd_AST"]**2)
            elif r["market"]=="player_points_rebounds":
                mu=row["mu_PTS"]+row["mu_REB"]; sd=np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2)
            elif r["market"]=="player_points_assists":
                mu=row["mu_PTS"]+row["mu_AST"]; sd=np.sqrt(row["sd_PTS"]**2+row["sd_AST"]**2)
            elif r["market"]=="player_rebounds_assists":
                mu=row["mu_REB"]+row["mu_AST"]; sd=np.sqrt(row["sd_REB"]**2+row["sd_AST"]**2)
            else: continue

        # Defense
        pos=get_position(row["Player"])
        if pos.startswith("G"): pos="PG"
        elif pos.startswith("F"): pos="SF"
        elif pos.startswith("C"): pos="C"
        else: pos="SG"

        team = event["home_team"] if event["away_team"].lower() in pick.lower() else event["away_team"]
        if team in def_table.index:
            mu = mu * float(def_table.loc[team].get(pos,1))

        # Minutes injury scale
        season_min=row["mu_MIN"]
        scale = injury_minute_scale(row["Player"], season_min)
        mu_adj = mu * scale

        p = t_over_prob(mu_adj, sd, r["line"])
        win = p if r["side"]=="Over" else 1-p

        out.append({
            "Player":row["Player"],
            "Market":r["market"],
            "Side":r["side"],
            "Line":r["line"],
            "Model Projection":round(mu_adj,2),
            "Win %":round(win*100,2)
        })

    result=pd.DataFrame(out).sort_values("Win %",ascending=False)
    st.dataframe(result,use_container_width=True)
    st.download_button("Download CSV",result.to_csv(index=False),file_name="nba_model_results.csv")
