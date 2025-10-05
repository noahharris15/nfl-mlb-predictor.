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
    "player_pass_yds",
    "player_rush_yds",
    "player_rec_yds",
    "player_receptions",
    "player_pass_tds",
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
    for c in ["EPA_Pass", "EPA_Rush", "Comp_Pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    def adj_from_epa(s, scale):
        x = s.fillna(0.0)
        return (1.0 - scale * x).clip(0.7, 1.3)
    pass_adj = adj_from_epa(df["EPA_Pass"], 0.8)
    rush_adj = adj_from_epa(df["EPA_Rush"], 0.8)
    comp = df["Comp_Pct"].clip(0.45, 0.80).fillna(df["Comp_Pct"].mean())
    comp_adj = (1.0 + (comp - comp.mean()) * 0.6).clip(0.7, 1.3)
    recv_adj = (0.7 * pass_adj + 0.3 * comp_adj).clip(0.7, 1.3)
    return pd.DataFrame({
        "Team": df["Team"],
        "pass_adj": pass_adj,
        "rush_adj": rush_adj,
        "recv_adj": recv_adj
    })

DEF_TABLE = load_defense_table()

# ------------------ Helpers ------------------
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))

def normalize_name(n):
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    return strip_accents(re.sub(r"\s+", " ", n).strip())

def t_over_prob(mu, sd, line, trials=SIM_TRIALS):
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

# ------------------ ESPN crawl ------------------
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

def http_get(url, params=None, timeout=25):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def list_week_event_ids(year, week, seasontype=2):
    js = http_get(SCOREBOARD, params={"year": year, "week": week, "seasontype": seasontype})
    if not js:
        return []
    return [str(e.get("id")) for e in js.get("events", []) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_boxscore_event(event_id):
    return http_get(SUMMARY, params={"event": event_id})

def parse_boxscore_players(box):
    out = []
    try:
        for team in box.get("boxscore", {}).get("players", []):
            for cat in team.get("statistics", []):
                label = (cat.get("name") or "").lower()
                for a in cat.get("athletes", []):
                    nm = normalize_name(a.get("athlete", {}).get("displayName"))
                    vals = a.get("stats") or []
                    if "passing" in label:
                        yds = to_float(vals[1]) if len(vals) > 1 else np.nan
                        tds = to_float(vals[2]) if len(vals) > 2 else np.nan
                        out.append({
                            "Player": nm, "pass_yds": yds, "pass_tds": tds,
                            "rush_yds": 0.0, "rec_yds": 0.0, "rec": 0.0
                        })
                    elif "rushing" in label:
                        yds = to_float(vals[1]) if len(vals) > 1 else np.nan
                        out.append({
                            "Player": nm, "pass_yds": 0.0, "pass_tds": 0.0,
                            "rush_yds": yds, "rec_yds": 0.0, "rec": 0.0
                        })
                    elif "receiving" in label:
                        recs = to_float(vals[0]) if len(vals) > 0 else np.nan
                        yds = to_float(vals[1]) if len(vals) > 1 else np.nan
                        out.append({
                            "Player": nm, "pass_yds": 0.0, "pass_tds": 0.0,
                            "rush_yds": 0.0, "rec_yds": yds, "rec": recs
                        })
    except Exception:
        pass
    if not out:
        return pd.DataFrame(columns=[
            "Player", "pass_yds", "pass_tds",
            "rush_yds", "rec_yds", "rec"
        ])
    return pd.DataFrame(out).groupby("Player", as_index=False).sum(numeric_only=True)
    # ------------------ Aggregate season data ------------------
@st.cache_data(show_spinner=True)
def build_espn_season_agg(year: int, weeks: List[int], seasontype: int) -> pd.DataFrame:
    totals, sumsqs, games = {}, {}, {}
    def init_p(p):
        if p not in totals:
            totals[p] = {"pass_yds":0.0,"pass_tds":0.0,"rush_yds":0.0,"rec_yds":0.0,"rec":0.0}
            sumsqs[p] = {"pass_yds":0.0,"rush_yds":0.0,"rec_yds":0.0,"rec":0.0}
            games[p]  = 0
    events = []
    for wk in weeks:
        events.extend(list_week_event_ids(year, wk, seasontype))
    if not events:
        return pd.DataFrame()
    prog = st.progress(0.0, text=f"Crawling {len(events)} games...")
    for j, ev in enumerate(events, 1):
        box = fetch_boxscore_event(ev)
        if box is None: continue
        df = parse_boxscore_players(box)
        for _, r in df.iterrows():
            p = r["Player"]; init_p(p)
            played = any(to_float(r[k]) > 0 for k in ["pass_yds","rush_yds","rec_yds","rec"])
            if played: games[p] += 1
            for k in totals[p]:
                v = to_float(r.get(k, 0)); 
                if not np.isnan(v): totals[p][k] += v
            for k in sumsqs[p]:
                v = to_float(r.get(k, 0));
                if not np.isnan(v): sumsqs[p][k] += v*v
        prog.progress(j/len(events))
    rows = []
    for p, stat in totals.items():
        g = max(1, int(games.get(p, 0)))
        rows.append({"Player": p, "g": g, **stat,
                     "sq_pass_yds": sumsqs[p]["pass_yds"],
                     "sq_rush_yds": sumsqs[p]["rush_yds"],
                     "sq_rec_yds": sumsqs[p]["rec_yds"],
                     "sq_rec": sumsqs[p]["rec"]})
    return pd.DataFrame(rows)

# ------------------ UI: Build projections ------------------
st.markdown("### 1️⃣ Season scope & opponent defense")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    season = st.number_input("Season", min_value=2015, max_value=2100, value=2025)
with c2:
    seasontype = st.selectbox("Season type",
        [("Preseason",1),("Regular",2),("Postseason",3)], index=1,
        format_func=lambda x: x[0])[1]
with c3:
    week_range = st.slider("Weeks", 1, 18, (1,18)) if seasontype==2 else (1,4)
weeks = list(range(week_range[0], week_range[-1]+1))
opp_team = st.selectbox("Opponent (defense scaling)", DEF_TABLE["Team"].tolist(), index=0)
scalers = DEF_TABLE.set_index("Team").loc[opp_team].to_dict()

def clamp(x, lo=0.85, hi=1.15): return float(np.clip(x, lo, hi))

st.markdown("### 2️⃣ Build per-player projections from ESPN data")
if st.button("📥 Build projections"):
    season_df = build_espn_season_agg(season, weeks, seasontype)
    if season_df.empty:
        st.error("No ESPN data found."); st.stop()

    df = season_df.copy()
    df["rush_ypg"] = df["rush_yds"]/df["g"].clip(lower=1)
    df["rec_ypg"]  = df["rec_yds"]/df["g"].clip(lower=1)
    df["rec_pg"]   = df["rec"]/df["g"].clip(lower=1)
    df["pass_ypg"] = df["pass_yds"]/df["g"].clip(lower=1)
    df["pass_tdp"] = df["pass_tds"]/df["g"].clip(lower=1)

    rushers   = df[(df["g"]>=2)&(df["rush_ypg"]>=15)]
    receivers = df[(df["g"]>=2)&(df["rec_pg"]>=1.5)]
    passers   = df[(df["g"]>=2)&(df["pass_ypg"]>=100)]

    def trimmed_mean(s, trim=0.2):
        s=s.dropna().sort_values(); n=len(s); k=int(n*trim)
        return float(s.iloc[k:n-k].mean() if n>2*k else s.mean()) if n>0 else float("nan")

    lg_mu_rush_yds = trimmed_mean(rushers["rush_ypg"])
    lg_mu_rec_yds  = trimmed_mean(receivers["rec_ypg"])
    lg_mu_recs     = trimmed_mean(receivers["rec_pg"])
    lg_mu_pass_yds = trimmed_mean(passers["pass_ypg"])
    lg_mu_pass_tds = trimmed_mean(passers["pass_tdp"])

    g = season_df["g"].astype(float)
    K = np.where(g>=6,1.0,np.where(g>=4,2.0,5.0))
    w = g/(g+K)

    raw_mu_pass_yds = season_df["pass_yds"]/g.clip(lower=1)
    raw_mu_pass_tds = season_df["pass_tds"]/g.clip(lower=1)
    raw_mu_rush_yds = season_df["rush_yds"]/g.clip(lower=1)
    raw_mu_rec_yds  = season_df["rec_yds"]/g.clip(lower=1)
    raw_mu_recs     = season_df["rec"]/g.clip(lower=1)

    season_df["mu_pass_yds"]   = w*raw_mu_pass_yds+(1-w)*lg_mu_pass_yds
    season_df["mu_pass_tds"]   = w*raw_mu_pass_tds+(1-w)*lg_mu_pass_tds
    season_df["mu_rush_yds"]   = w*raw_mu_rush_yds+(1-w)*lg_mu_rush_yds
    season_df["mu_rec_yds"]    = w*raw_mu_rec_yds+(1-w)*lg_mu_rec_yds
    season_df["mu_receptions"] = w*raw_mu_recs+(1-w)*lg_mu_recs

    def sample_sd(sum_x, sum_x2, g_val):
        g_val=int(g_val)
        if g_val<=1: return np.nan
        mean=sum_x/g_val
        var=(sum_x2/g_val)-(mean**2)
        var=var*(g_val/(g_val-1))
        return float(np.sqrt(max(var,1e-6)))

    sd_pass = season_df.apply(lambda r: sample_sd(r["pass_yds"],r["sq_pass_yds"],r["g"]),axis=1)
    sd_rush = season_df.apply(lambda r: sample_sd(r["rush_yds"],r["sq_rush_yds"],r["g"]),axis=1)
    sd_recyds = season_df.apply(lambda r: sample_sd(r["rec_yds"],r["sq_rec_yds"],r["g"]),axis=1)
    sd_recs = season_df.apply(lambda r: sample_sd(r["rec"],r["sq_rec"],r["g"]),axis=1)

    SD_INFLATE=1.15
    season_df["sd_pass_yds"]   = np.maximum(30, sd_pass.fillna(30))*SD_INFLATE
    season_df["sd_rush_yds"]   = np.maximum(15, sd_rush.fillna(15))*SD_INFLATE
    season_df["sd_rec_yds"]    = np.maximum(20, sd_recyds.fillna(20))*SD_INFLATE
    season_df["sd_receptions"] = np.maximum(1.2, sd_recs.fillna(1.2))*SD_INFLATE

    season_df["mu_pass_yds"]   *= clamp(scalers["pass_adj"])
    season_df["mu_pass_tds"]   *= clamp(scalers["pass_adj"])
    season_df["mu_rush_yds"]   *= clamp(scalers["rush_adj"])
    season_df["mu_rec_yds"]    *= clamp(scalers["recv_adj"])
    season_df["mu_receptions"] *= clamp(scalers["recv_adj"])

    st.session_state["qb_proj"]=season_df[["Player","mu_pass_yds","sd_pass_yds","mu_pass_tds"]]
    st.session_state["rb_proj"]=season_df[["Player","mu_rush_yds","sd_rush_yds"]]
    st.session_state["wr_proj"]=season_df[["Player","mu_rec_yds","sd_rec_yds","mu_receptions","sd_receptions"]]

    c1,c2,c3=st.columns(3)
    with c1: st.dataframe(st.session_state["qb_proj"].head(10))
    with c2: st.dataframe(st.session_state["rb_proj"].head(10))
    with c3: st.dataframe(st.session_state["wr_proj"].head(10))

# ------------------ Odds API ------------------
st.markdown("### 3️⃣ Pick a game & markets")
api_key=(st.secrets.get("odds_api_key") if hasattr(st,"secrets") else None) or st.text_input("Odds API Key",type="password")
region=st.selectbox("Region",["us","us2","eu","uk"],index=0)
lookahead=st.slider("Lookahead days",0,7,value=1)
markets=st.multiselect("Markets to fetch",VALID_MARKETS,default=VALID_MARKETS)

def odds_get(url,params): r=requests.get(url,params=params,timeout=25); r.raise_for_status(); return r.json()
def list_nfl_events(api_key,lookahead,region):
    return odds_get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events",
                    {"apiKey":api_key,"daysFrom":0,"daysTo":lookahead,"regions":region})
def fetch_event_props(api_key,event_id,region,markets):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds",
                    {"apiKey":api_key,"regions":region,"markets":",".join(markets),"oddsFormat":"american"})

events=[]
if api_key:
    try: events=list_nfl_events(api_key,lookahead,region)
    except Exception as e: st.error(e)
if not events: st.stop()
pick=st.selectbox("Game",[f'{e["away_team"]} @ {e["home_team"]}' for e in events])
event_id=events[[f'{e["away_team"]} @ {e["home_team"]}' for e in events].index(pick)]["id"]

# ------------------ Simulate ------------------
st.markdown("### 4️⃣ Simulate props with median book line")
go=st.button("🎲 Simulate")
if go:
    qb_proj=st.session_state.get("qb_proj",pd.DataFrame())
    rb_proj=st.session_state.get("rb_proj",pd.DataFrame())
    wr_proj=st.session_state.get("wr_proj",pd.DataFrame())
    data=fetch_event_props(api_key,event_id,region,markets)
    rows=[]
    for bk in data.get("bookmakers",[]):
        for m in bk.get("markets",[]):
            mkey=m.get("key")
            for o in m.get("outcomes",[]):
                name=normalize_name(o.get("description"))
                side=o.get("name"); point=o.get("point")
                if mkey not in VALID_MARKETS or not name: continue
                rows.append({"market":mkey,"player":name,"side":side,"point":float(point) if point else None})
    props=pd.DataFrame(rows).groupby(["market","player","side"],as_index=False).agg(line=("point","median"))

    out=[]
    for _,r in props.iterrows():
        m=r["market"]; p=r["player"]; line=r["line"]; s=r["side"]
        if m=="player_pass_yds" and p in qb_proj["Player"].values:
            row=qb_proj[qb_proj["Player"]==p].iloc[0]
            prob=t_over_prob(row["mu_pass_yds"],row["sd_pass_yds"],line)
        elif m=="player_rush_yds" and p in rb_proj["Player"].values:
            row=rb_proj[rb_proj["Player"]==p].iloc[0]
            prob=t_over_prob(row["mu_rush_yds"],row["sd_rush_yds"],line)
        elif m=="player_rec_yds" and p in wr_proj["Player"].values:
            row=wr_proj[wr_proj["Player"]==p].iloc[0]
            prob=t_over_prob(row["mu_rec_yds"],row["sd_rec_yds"],line)
        elif m=="player_receptions" and p in wr_proj["Player"].values:
            row=wr_proj[wr_proj["Player"]==p].iloc[0]
            prob=t_over_prob(row["mu_receptions"],row["sd_receptions"],line)
        elif m=="player_pass_tds" and p in qb_proj["Player"].values:
            lam=qb_proj.loc[qb_proj["Player"]==p,"mu_pass_tds"].iloc[0]
            prob=float((np.random.poisson(lam,10_000)>line).mean())
        else: continue
        win=prob if s=="Over" else 1-prob
        out.append({"market":m,"player":p,"side":s,"line":line,"Win Prob %":round(100*win,2)})
    res=pd.DataFrame(out).sort_values(["market","Win Prob %"],ascending=[True,False])
    st.dataframe(res,use_container_width=True)
    st.bar_chart(res.groupby("market")["Win Prob %"].max())
    st.download_button("⬇️ Download CSV",res.to_csv(index=False).encode("utf-8"),"props_results.csv")
