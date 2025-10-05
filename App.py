# Player Props — Odds API + ESPN (pos-aware priors + dynamic shrink + sample SD + t-dist)
# Run: streamlit run app.py

import re, math, unicodedata
from io import StringIO
from typing import List, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import process, fuzz

# ------------------ Page / theme ------------------
st.set_page_config(page_title="NFL Player Props — Odds API + ESPN", layout="wide")
st.title("📈 NFL Player Props — Odds API + ESPN (pos-aware priors, calibrated)")

SIM_TRIALS = 10_000
VALID_MARKETS = [
    "player_pass_yds", "player_rush_yds",
    "player_rec_yds", "player_receptions", "player_pass_tds"
]

# ------------------ Embedded 2025 defense EPA ------------------
DEFENSE_EPA_2025 = """Team,EPA_Pass,EPA_Rush,Comp_Pct
Minnesota Vikings,-0.37,0.06,0.6762
Jacksonville Jaguars,-0.17,-0.05,0.5962
Denver Broncos,-0.10,-0.12,0.5746
Los Angeles Chargers,-0.17,0.01,0.5938
Detroit Lions,0.00,-0.22,0.6271
Philadelphia Eagles,-0.11,-0.04,0.5693
Houston Texans,-0.16,0.04,0.5714
Los Angeles Rams,-0.12,0.00,0.6640
Seattle Seahawks,0.00,-0.19,0.6645
San Francisco 49ers,-0.09,-0.03,0.6829
Tampa Bay Buccaneers,-0.02,-0.11,0.6429
Atlanta Falcons,-0.13,0.05,0.5769
Cleveland Browns,0.06,-0.17,0.6442
Indianapolis Colts,-0.04,-0.05,0.6643
Kansas City Chiefs,-0.09,0.09,0.6694
Arizona Cardinals,0.06,-0.14,0.6369
Las Vegas Raiders,0.14,-0.22,0.6565
Green Bay Packers,0.03,-0.07,0.6815
Chicago Bears,0.01,0.00,0.7368
Buffalo Bills,-0.06,0.10,0.6214
Carolina Panthers,0.03,0.05,0.6239
Pittsburgh Steelers,0.11,-0.05,0.6957
Washington Commanders,0.18,-0.12,0.6098
New England Patriots,0.19,-0.15,0.7120
New York Giants,-0.01,0.19,0.6375
New Orleans Saints,0.20,-0.06,0.7117
Cincinnati Bengals,0.13,0.04,0.6536
New York Jets,0.23,-0.03,0.6577
Tennessee Titans,0.16,0.07,0.6984
Baltimore Ravens,0.14,0.12,0.6667
Dallas Cowboys,0.40,0.06,0.7333
Miami Dolphins,0.34,0.12,0.7757
"""
@st.cache_data(show_spinner=False)
def load_defense_table():
    df = pd.read_csv(StringIO(DEFENSE_EPA_2025))
    for c in ["EPA_Pass","EPA_Rush","Comp_Pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    def adj_from_epa(s, scale):
        x = s.fillna(0.0)
        return (1.0 - scale * x).clip(0.7, 1.3)
    pass_adj = adj_from_epa(df["EPA_Pass"], 0.8)
    rush_adj = adj_from_epa(df["EPA_Rush"], 0.8)
    comp = df["Comp_Pct"].clip(0.45,0.80).fillna(df["Comp_Pct"].mean())
    comp_adj = (1.0 + (comp - comp.mean()) * 0.6).clip(0.7,1.3)
    recv_adj = (0.7*pass_adj + 0.3*comp_adj).clip(0.7,1.3)
    return pd.DataFrame({"Team":df["Team"],"pass_adj":pass_adj,"rush_adj":rush_adj,"recv_adj":recv_adj})
DEF_TABLE = load_defense_table()

# ------------------ Helpers ------------------
def strip_accents(s): return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
def normalize_name(n):
    n=str(n or ""); n=n.split("(")[0]; n=n.replace("-"," "); n=re.sub(r"[.,']"," ",n)
    return strip_accents(re.sub(r"\s+"," ",n).strip())
def t_over_prob(mu, sd, line, trials=SIM_TRIALS):
    sd=max(1e-6,float(sd))
    draws = mu + sd*np.random.standard_t(df=5, size=trials)
    return float((draws>line).mean())
def to_float(x):
    try: return float(x)
    except Exception: return float("nan")

# ------------------ ESPN API ------------------
SCOREBOARD="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY="https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

def http_get(url, params=None, timeout=25):
    try:
        r=requests.get(url,params=params,timeout=timeout)
        if r.status_code==200: return r.json()
    except Exception: pass
    return None

@st.cache_data(show_spinner=False)
def list_week_event_ids(year,week,seasontype=2):
    js=http_get(SCOREBOARD,params={"year":year,"week":week,"seasontype":seasontype})
    if not js: return []
    return [str(e.get("id")) for e in js.get("events",[]) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id): return http_get(SUMMARY,params={"event":event_id})

def parse_boxscore_players(box):
    out=[]
    try:
        for team in box.get("boxscore",{}).get("players",[]):
            for cat in team.get("statistics",[]):
                label=(cat.get("name") or "").lower()
                for a in cat.get("athletes",[]):
                    nm=normalize_name(a.get("athlete",{}).get("displayName"))
                    vals=a.get("stats") or []
                    if "passing" in label:
                        yds,tds=to_float(vals[1]),to_float(vals[2])
                        out.append({"Player":nm,"pass_yds":yds,"pass_tds":tds,
                                    "rush_yds":0,"rec_yds":0,"rec":0})
                    elif "rushing" in label:
                        yds=to_float(vals[1])
                        out.append({"Player":nm,"pass_yds":0,"pass_tds":0,
                                    "rush_yds":yds,"rec_yds":0,"rec":0})
                    elif "receiving" in label:
                        recs,yds=to_float(vals[0]),to_float(vals[1])
                        out.append({"Player":nm,"pass_yds":0,"pass_tds":0,
                                    "rush_yds":0,"rec_yds":yds,"rec":recs})
    except Exception: pass
    if not out: return pd.DataFrame(columns=["Player","pass_yds","pass_tds","rush_yds","rec_yds","rec"])
    return pd.DataFrame(out).groupby("Player",as_index=False).sum(numeric_only=True)

@st.cache_data(show_spinner=True)
def build_espn_season_agg(year,weeks,seasontype):
    totals,sumsqs,games={}, {}, {}
    def init_p(p):
        if p not in totals:
            totals[p]={"pass_yds":0,"pass_tds":0,"rush_yds":0,"rec_yds":0,"rec":0}
            sumsqs[p]={"pass_yds":0,"rush_yds":0,"rec_yds":0}
            games[p]=0
    events=[]
    for wk in weeks: events+=list_week_event_ids(year,wk,seasontype)
    prog=st.progress(0.0,text=f"Crawling {len(events)} games...")
    for j,ev in enumerate(events,1):
        box=fetch_boxscore_event(ev)
        if box is None: continue
        df=parse_boxscore_players(box)
        for _,r in df.iterrows():
            p=r["Player"]; init_p(p)
            played=any(to_float(r[k])>0 for k in ["pass_yds","rush_yds","rec_yds","rec"])
            if played: games[p]+=1
            for k in totals[p]:
                v=to_float(r.get(k,0))
                if not np.isnan(v): totals[p][k]+=v
            for k in sumsqs[p]:
                v=to_float(r.get(k,0))
                if not np.isnan(v): sumsqs[p][k]+=v*v
        prog.progress(j/len(events))
    rows=[]
    for p,stat in totals.items():
        g=max(1,int(games.get(p,0)))
        rows.append({"Player":p,"g":g,**stat,
                     "sq_pass_yds":sumsqs[p]["pass_yds"],
                     "sq_rush_yds":sumsqs[p]["rush_yds"],
                     "sq_rec_yds":sumsqs[p]["rec_yds"]})
    return pd.DataFrame(rows)

# ------------------ Step 1: Build projections ------------------
st.markdown("### 1) Build projections from ESPN")
season=st.number_input("Season",2015,2100,2025)
weeks=list(range(1,18+1))
opp_team=st.selectbox("Opponent",DEF_TABLE["Team"].tolist(),index=0)
scalers=DEF_TABLE.set_index("Team").loc[opp_team].to_dict()

if st.button("📥 Build projections"):
    df=build_espn_season_agg(season,weeks,2)
    if df.empty: st.stop()
    g=df["g"].clip(lower=1)
    df["rush_ypg"]=df["rush_yds"]/g
    df["rec_ypg"]=df["rec_yds"]/g
    df["rec_pg"]=df["rec"]/g
    df["pass_ypg"]=df["pass_yds"]/g
    df["pass_tdp"]=df["pass_tds"]/g
    rushers=df[df["rush_ypg"]>=15]; receivers=df[df["rec_pg"]>=1.5]; passers=df[df["pass_ypg"]>=100]
    def trimmed_mean(s,t=0.2):
        s=s.dropna().sort_values(); n=len(s); k=int(n*t)
        return float(s.iloc[k:n-k].mean() if n>2*k else s.mean())
    lg_mu_rush=trimmed_mean(rushers["rush_ypg"]) if not rushers.empty else df["rush_ypg"].median()
    lg_mu_recyds=trimmed_mean(receivers["rec_ypg"]) if not receivers.empty else df["rec_ypg"].median()
    lg_mu_recs=trimmed_mean(receivers["rec_pg"]) if not receivers.empty else df["rec_pg"].median()
    lg_mu_pass=trimmed_mean(passers["pass_ypg"]) if not passers.empty else df["pass_ypg"].median()
    lg_mu_passtd=trimmed_mean(passers["pass_tdp"]) if not passers.empty else df["pass_tdp"].median()
    K=np.where(g>=6,1.0,np.where(g>=4,2.0,5.0)); w=g/(g+K)
    df["mu_rush_yds"]=w*df["rush_ypg"]+(1-w)*lg_mu_rush
    df["mu_rec_yds"]=w*df["rec_ypg"]+(1-w)*lg_mu_recyds
    df["mu_receptions"]=w*df["rec_pg"]+(1-w)*lg_mu_recs
    df["mu_pass_yds"]=w*df["pass_ypg"]+(1-w)*lg_mu_pass
    df["mu_pass_tds"]=w*df["pass_tdp"]+(1-w)*lg_mu_passtd
    df["sd_rush_yds"]=df["rush_yds"].std()*0.3 if not df.empty else 10
    df["sd_rec_yds"]=df["rec_yds"].std()*0.25 if not df.empty else 8
    df["sd_receptions"]=df["rec"].std()*0.25 if not df.empty else 1
    df["sd_pass_yds"]=df["pass_yds"].std()*0.25 if not df.empty else 30
    df["sd_pass_tds"]=0.6
    df["mu_pass_yds"]*=scalers["pass_adj"]
    df["mu_pass_tds"]*=scalers["pass_adj"]
    df["mu_rush_yds"]*=scalers["rush_adj"]
    df["mu_rec_yds"]*=scalers["recv_adj"]
    df["mu_receptions"]*=scalers["recv_adj"]
    st.session_state["proj"]=df

# ------------------ Step 2: Odds API ------------------
st.markdown("### 2) Fetch & simulate Odds API props")
api_key=st.text_input("Odds API Key","",type="password")
region=st.selectbox("Region",["us","us2","eu","uk"],0)
lookahead=st.slider("Lookahead days",0,7,1)

def odds_get(url,params):
    r=requests.get(url,params=params,timeout=25)
    r.raise_for_status(); return r.json()
def list_nfl_events(api_key,lookahead,region):
    return odds_get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events",
                    {"apiKey":api_key,"daysFrom":0,"daysTo":lookahead,"regions":region})
def fetch_event_props(api_key,event_id,region,markets):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds",
                    {"apiKey":api_key,"regions":region,"markets":",".join(markets),"oddsFormat":"american"})

if api_key:
    try: events=list_nfl_events(api_key,lookahead,region)
    except Exception as e: st.error(e); st.stop()
    event_labels=[f'{e["away_team"]} @ {e["home_team"]}' for e in events]
    pick=st.selectbox("Game",event_labels)
    event_id=events[event_labels.index(pick)]["id"]
    if st.button("🎲 Simulate"):
        data=fetch_event_props(api_key,event_id,region,VALID_MARKETS)
        props=[]
        for bk in data.get("bookmakers",[]):
            for m in bk.get("markets",[]):
                key=m.get("key")
                for o in m.get("outcomes",[]):
                    nm=normalize_name(o.get("description")); side=o.get("name"); pt=o.get("point")
                    if key not in VALID_MARKETS or not nm or pt is None: continue
                    props.append({"market":key,"player":nm,"side":side,"line":float(pt)})
        if not props: st.warning("No props found."); st.stop()
        props_df=pd.DataFrame(props).groupby(["market","player","side"],as_index=False).agg(line=("line","median"))
        df=st.session_state["proj"]
        results=[]
        for _,r in props_df.iterrows():
            name,side,market,line=r["player"],r["side"],r["market"],r["line"]
            match=process.extractOne(name,df["Player"].tolist(),scorer=fuzz.token_sort_ratio)
            if not match or match[1]<85: continue
            p=match[0]; row=df[df["Player"]==p].iloc[0]
            mu_col,sd_col={
                "player_pass_yds":("mu_pass_yds","sd_pass_yds"),
                "player_pass_tds":("mu_pass_tds","sd_pass_tds"),
                "player_rush_yds":("mu_rush_yds","sd_rush_yds"),
                "player_receptions":("mu_receptions","sd_receptions"),
                "player_rec_yds":("mu_rec_yds","sd_rec_yds")
            }[market]
            mu,sd=float(row[mu_col]),float(row[sd_col])
            p_over=t_over_prob(mu,sd,line)
            prob=p_over if side=="Over" else 1-p_over
            results.append({"market":market,"player":p,"side":side,"line":line,"μ":mu,"σ":sd,"Win Prob %":round(prob*100,2)})
        res=pd.DataFrame(results).sort_values("Win Prob %",ascending=False)
        st.dataframe(res,use_container_width=True)
        st.bar_chart(res.head(15).set_index("player")["Win Prob %"])
        st.download_button("⬇️ Download results CSV",res.to_csv(index=False).encode("utf-8"),"props_sim_results.csv","text/csv")
