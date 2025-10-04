# app.py — NFL + MLB + Player Props + College Football (2025, stats-only)
# - NFL & MLB team win pages use season scoring rates
# - Player Props uses your uploaded CSVs + embedded NFL defense factors
# - College Football auto-loads 2025 team stats from CollegeFootballData (no CSV)

from __future__ import annotations
import io, os, math, requests
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Libraries for NFL / MLB ----------------
import nfl_data_py as nfl
from pybaseball import schedule_and_record

# ---------------- Streamlit basics -----------------------
st.set_page_config(page_title="NFL + MLB + CFB + Props — 2025 (stats only)", layout="wide")
st.title("🏈⚾🎓 NFL + MLB + College Football + Player Props — 2025 (stats only)")
st.caption(
    "NFL & MLB pages use team scoring rates only. "
    "Player Props uses your CSV + embedded 2025 NFL defense. "
    "College Football auto-loads this season from CollegeFootballData (add CFBD_API_KEY in Secrets)."
)

# =============================================================================
# Embedded 2025 NFL defense table (EPA/play -> multiplicative factor)
# =============================================================================
# If you want to paste a new CSV in the future, put it inside DEFENSE_CSV_TEXT.
DEFENSE_CSV_TEXT = ""  # leave blank to use fallback dict below

DEF_EPA_2025_FALLBACK = {
    "MIN": -0.27,"JAX": -0.15,"GB": -0.13,"SF": -0.11,"ATL": -0.10,
    "IND": -0.08,"LAC": -0.08,"DEN": -0.08,"LAR": -0.07,"SEA": -0.07,
    "PHI": -0.06,"TB": -0.05,"CAR": -0.05,"ARI": -0.03,"CLE": -0.02,
    "WAS": -0.02,"HOU":  0.00,"KC": 0.01,"DET": 0.01,"LV": 0.03,
    "PIT": 0.05,"CIN": 0.05,"NO": 0.05,"BUF": 0.05,"CHI": 0.06,
    "NE": 0.09,"NYJ": 0.10,"TEN": 0.11,"BAL": 0.11,"NYG": 0.13,
    "DAL": 0.21,"MIA": 0.28
}
ALIAS_TO_STD = {
    "GNB":"GB","SFO":"SF","KAN":"KC","NWE":"NE","NOR":"NO","TAM":"TB","LVR":"LV","SDG":"LAC","STL":"LAR",
    "JAC":"JAX","WSH":"WAS","LA":"LAR","OAK":"LV"
}
def _norm_team(code:str)->str:
    c=(code or "").strip().upper()
    return ALIAS_TO_STD.get(c,c)

def _def_epa_from_df(df: pd.DataFrame) -> Dict[str,float]:
    if df.empty: return {}
    cols = {str(c).strip().lower(): c for c in df.columns}
    team_col = next((cols[k] for k in ["team","team_code","def_team","abbr","tm","opponent","opp","code"] if k in cols), None)
    epa_col  = next((cols[k] for k in ["epa/play","epa per play","epa_play","def_epa","def epa","epa"] if k in cols), None)
    if team_col is None or epa_col is None:
        # best-effort fallback
        if team_col is None:
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]): team_col=c; break
        if epa_col is None:
            nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            epa_col=nums[-1] if nums else None
    if team_col is None or epa_col is None: return {}
    out={}
    for _,r in df[[team_col,epa_col]].dropna().iterrows():
        try: out[_norm_team(str(r[team_col]))]=float(r[epa_col])
        except: pass
    return out

def _build_def_factor_map(epa: Dict[str,float])->Dict[str,float]:
    if not epa: return {}
    s=pd.Series(epa,dtype=float)
    mu=float(s.mean()); sd=float(s.std(ddof=0) or 1.0)
    z=(s-mu)/(sd if sd>1e-9 else 1.0)
    # Better D (lower EPA) -> lower factor; clamp ±2σ => ~0.85..1.15 around 1.0
    factors = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {k: float(v) for k,v in factors.items()}

def load_defense_table()->Tuple[Dict[str,float],Dict[str,float],str]:
    txt = DEFENSE_CSV_TEXT.strip()
    if txt:
        try:
            df=pd.read_csv(io.StringIO(txt))
            epa=_def_epa_from_df(df)
            if epa:
                return _build_def_factor_map(epa), epa, "Embedded CSV"
        except: pass
    return _build_def_factor_map(DEF_EPA_2025_FALLBACK), DEF_EPA_2025_FALLBACK, "Embedded fallback"

DEF_FACTORS, DEF_EPA_MAP, DEF_SRC = load_defense_table()

# =============================================================================
# Shared helpers
# =============================================================================
SIM_TRIALS = 10_000
HOME_EDGE_NFL = 0.6
EPS = 1e-9

MLB_TEAMS_2025: Dict[str,str] = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles",
    "BOS":"Boston Red Sox","CHC":"Chicago Cubs","CHW":"Chicago White Sox",
    "CIN":"Cincinnati Reds","CLE":"Cleveland Guardians","COL":"Colorado Rockies",
    "DET":"Detroit Tigers","HOU":"Houston Astros","KCR":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins",
    "MIL":"Milwaukee Brewers","MIN":"Minnesota Twins","NYM":"New York Mets",
    "NYY":"New York Yankees","OAK":"Oakland Athletics","PHI":"Philadelphia Phillies",
    "PIT":"Pittsburgh Pirates","SDP":"San Diego Padres","SEA":"Seattle Mariners",
    "SFG":"San Francisco Giants","STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays",
    "TEX":"Texas Rangers","TOR":"Toronto Blue Jays","WSN":"Washington Nationals"
}

def _poisson_game(mu_h:float, mu_a:float, trials:int=SIM_TRIALS):
    mu_h=max(0.1,float(mu_h)); mu_a=max(0.1,float(mu_a))
    h=np.random.poisson(mu_h,size=trials); a=np.random.poisson(mu_a,size=trials)
    w=(h>a).astype(np.float64); ties=(h==a)
    if ties.any(): w[ties]=0.53
    p_home=float(w.mean())
    return p_home,1-p_home,float(h.mean()),float(a.mean())

# =============================================================================
# NFL page
# =============================================================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])
    date_col=None
    for c in ("gameday","game_date","start_time"): 
        if c in sched.columns: date_col=c; break

    played=sched.dropna(subset=["home_score","away_score"])
    home=played.rename(columns={"home_team":"team","away_team":"opp","home_score":"pf","away_score":"pa"})[["team","opp","pf","pa"]]
    away=played.rename(columns={"away_team":"team","home_team":"opp","away_score":"pf","home_score":"pa"})[["team","opp","pf","pa"]]
    long=pd.concat([home,away],ignore_index=True)

    if long.empty:
        per=45/2
        teams32=["Arizona Cardinals","Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers","Chicago Bears",
                 "Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions","Green Bay Packers",
                 "Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs","Las Vegas Raiders",
                 "Los Angeles Chargers","Los Angeles Rams","Miami Dolphins","Minnesota Vikings","New England Patriots",
                 "New Orleans Saints","New York Giants","New York Jets","Philadelphia Eagles","Pittsburgh Steelers",
                 "San Francisco 49ers","Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Commanders"]
        rates=pd.DataFrame({"team":teams32,"PF_pg":per,"PA_pg":per})
    else:
        team=long.groupby("team",as_index=False).agg(games=("pf","size"),PF=("pf","sum"),PA=("pa","sum"))
        rates=pd.DataFrame({"team":team["team"],"PF_pg":team["PF"]/team["games"],"PA_pg":team["PA"]/team["games"]})
        league_total=float((long["pf"]+long["pa"]).mean()); prior=league_total/2
        shrink=np.clip(1-team["games"]/4.0,0.0,1.0)
        rates["PF_pg"]=(1-shrink)*rates["PF_pg"]+shrink*prior
        rates["PA_pg"]=(1-shrink)*rates["PA_pg"]+shrink*prior

    if {"home_team","away_team"}.issubset(sched.columns):
        filt=sched["home_score"].isna() & sched["away_score"].isna()
        upcoming=sched.loc[filt,["home_team","away_team"]].copy()
        upcoming["date"]=sched.loc[filt,date_col].astype(str) if date_col else ""
    else:
        upcoming=pd.DataFrame(columns=["home_team","away_team","date"])
    for c in ["home_team","away_team"]:
        if c in upcoming.columns: upcoming[c]=upcoming[c].astype(str).str.replace(r"\s+"," ",regex=True)
    return rates,upcoming

def nfl_matchup_mu(rates:pd.DataFrame, home:str, away:str)->Tuple[float,float]:
    rH=rates.loc[rates["team"].str.lower()==home.lower()]
    rA=rates.loc[rates["team"].str.lower()==away.lower()]
    if rH.empty or rA.empty: raise ValueError(f"Unknown teams: {home}, {away}")
    H,A=rH.iloc[0],rA.iloc[0]
    return max(EPS,(H["PF_pg"]+A["PA_pg"])/2 + HOME_EDGE_NFL), max(EPS,(A["PF_pg"]+H["PA_pg"])/2)

# =============================================================================
# MLB page
# =============================================================================
@st.cache_data(show_spinner=False)
def mlb_team_rates_2025()->pd.DataFrame:
    rows=[]
    for br,name in MLB_TEAMS_2025.items():
        try:
            sar=schedule_and_record(2025,br)
            sar=sar[pd.to_numeric(sar.get("R"),errors="coerce").notna()]
            sar=sar[pd.to_numeric(sar.get("RA"),errors="coerce").notna()]
            if sar.empty:
                RS_pg=RA_pg=4.5
            else:
                sar["R"]=sar["R"].astype(float); sar["RA"]=sar["RA"].astype(float)
                g=int(len(sar)); RS_pg=float(sar["R"].sum()/g); RA_pg=float(sar["RA"].sum()/g)
            rows.append({"team":name,"RS_pg":RS_pg,"RA_pg":RA_pg})
        except:
            rows.append({"team":name,"RS_pg":4.5,"RA_pg":4.5})
    df=pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs=float(df["RS_pg"].mean()); league_ra=float(df["RA_pg"].mean())
        df["RS_pg"]=0.9*df["RS_pg"]+0.1*league_rs
        df["RA_pg"]=0.9*df["RA_pg"]+0.1*league_ra
    return df

# =============================================================================
# Player Props page (CSV + embedded defense)
# =============================================================================
def _yard_col(df:pd.DataFrame, pos:str)->str:
    prefer=["Y/G","Yds/G","YDS/G","Yards/G","PY/G","RY/G","Rec Y/G","Yds","Yards","yds","yards"]
    low=[str(c).lower() for c in df.columns]
    for p in [x.lower() for x in prefer]:
        if p in low: return df.columns[low.index(p)]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return df.columns[-1]

def _name_col(df:pd.DataFrame)->str:
    for c in df.columns:
        if str(c).lower() in ("player","name"): return c
    return df.columns[0]

def _estimate_sd(mean:float,pos:str)->float:
    mean=float(mean)
    if pos=="QB": return max(35.0,0.60*mean)
    if pos=="RB": return max(20.0,0.75*mean)
    return max(22.0,0.85*mean)  # WR

def _read_any(up)->pd.DataFrame:
    if up is None: return pd.DataFrame()
    name=(up.name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"): return pd.read_excel(up)
    return pd.read_csv(up)

def _prop_sim(mean:float,line:float,sd:float)->Tuple[float,float]:
    sd=max(5.0,float(sd))
    z=(line-mean)/sd
    p_over=float(1.0 - 0.5*(1.0 + math.erf(z/ math.sqrt(2))))
    return float(np.clip(p_over,0,1)), float(np.clip(1-p_over,0,1))

# =============================================================================
# College Football (auto from CFBD)
# ==============================================================================
# College Football (CFBD API)
# ==============================================================================
import requests

@st.cache_data(show_spinner=False)
def cfb_team_stats_2025() -> pd.DataFrame:
    api_key = st.secrets.get("CFBD_API_KEY", "")
    if not api_key:
        st.error("Missing CFBD_API_KEY in Streamlit secrets")
        return pd.DataFrame()

    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://api.collegefootballdata.com/team/stats/season?year=2025"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            st.error(f"CFBD error {resp.status_code}: {resp.text}")
            return pd.DataFrame()

        raw = resp.json()
        rows = []
        for t in raw:
            team = t.get("team")
            # Look for points scored / allowed
            pts_for = next((s["stat"] for s in t.get("stats", []) if s["category"] == "points"), None)
            pts_against = next((s["stat"] for s in t.get("stats", []) if s["category"] == "opponentPoints"), None)
            games = next((s["stat"] for s in t.get("stats", []) if s["category"] == "games"), None)

            try:
                pts_for = float(pts_for)
                pts_against = float(pts_against)
                games = float(games)
                off_ppg = pts_for / games if games > 0 else 0.0
                def_ppg = pts_against / games if games > 0 else 0.0
                rows.append({"team": team, "off_ppg": off_ppg, "def_ppg": def_ppg})
            except Exception:
                continue

        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"CFBD request failed: {e}")
        return pd.DataFrame()

# --------------------- College Football Page ---------------------
elif page == "College Football":
    st.subheader("🏈🎓 College Football — 2025 (auto from CFBD)")
    df = cfb_team_stats_2025()
    if df.empty:
        st.warning("No data returned from CFBD. Check your API key or wait for season games.")
        st.stop()

    home = st.selectbox("Home team", df["team"].tolist())
    away = st.selectbox("Away team", [t for t in df["team"].tolist() if t != home])

    rowH = df.loc[df["team"] == home].iloc[0]
    rowA = df.loc[df["team"] == away].iloc[0]

    mu_home = (rowH["off_ppg"] + rowA["def_ppg"]) / 2.0 + 0.8  # home edge ~0.8 pts
    mu_away = (rowA["off_ppg"] + rowH["def_ppg"]) / 2.0
    pH, pA, mH, mA = _poisson_sim(mu_home, mu_away)

    st.markdown(
        f"**{home}** vs **{away}** — "
        f"Expected points: {mH:.1f}–{mA:.1f} · "
        f"P({home} win) = **{100*pH:.1f}%**, "
        f"P({away} win) = **{100*pA:.1f}%**"
    )

    with st.expander("Show team table"):
        st.dataframe(df.sort_values("off_ppg", ascending=False).reset_index(drop=True))

# =============================================================================
# UI router
# =============================================================================
page = st.radio("Pick a page", ["NFL","MLB","College Football","Player Props"], horizontal=True)

# ---------------- NFL ----------------
if page=="NFL":
    st.subheader("🏈 NFL — 2025 REG season")
    rates, upcoming = nfl_team_rates_2025()

    st.caption(f"Embedded NFL defense source: **{DEF_SRC}** (used on Player Props page)")
    if not upcoming.empty and {"home_team","away_team"}.issubset(upcoming.columns):
        labels=[f"{r['home_team']} vs {r['away_team']} — {r.get('date','')}" for _,r in upcoming.iterrows()]
        sel=st.selectbox("Select upcoming game", labels)
        try:
            teams_part=sel.split(" — ")[0]
            home,away=[t.strip() for t in teams_part.split(" vs ")]
        except:
            home=away=None
    else:
        st.info("No upcoming list available — pick any two teams below.")
        home=st.selectbox("Home team", rates["team"].tolist())
        away=st.selectbox("Away team", [t for t in rates["team"].tolist() if t!=home])

    if home and away:
        try:
            mu_h,mu_a = nfl_matchup_mu(rates, home, away)
            pH,pA,mH,mA=_poisson_game(mu_h,mu_a)
            st.markdown(f"**{home}** vs **{away}** — Expected points: **{mH:.1f}–{mA:.1f}** · "
                        f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**")
        except Exception as e:
            st.error(str(e))

# ---------------- MLB ----------------
elif page=="MLB":
    st.subheader("⚾ MLB — 2025 REG season (team scoring rates only)")
    rates=mlb_team_rates_2025()
    t1=st.selectbox("Home team", rates["team"].tolist())
    t2=st.selectbox("Away team", [t for t in rates["team"].tolist() if t!=t1])
    H=rates.loc[rates["team"]==t1].iloc[0]; A=rates.loc[rates["team"]==t2].iloc[0]
    mu_h=(H["RS_pg"]+A["RA_pg"])/2; mu_a=(A["RS_pg"]+H["RA_pg"])/2
    pH,pA,mH,mA=_poisson_game(mu_h,mu_a)
    st.markdown(f"**{t1}** vs **{t2}** — Expected runs: **{mH:.1f}–{mA:.1f}** · "
                f"P({t1} win) = **{100*pH:.1f}%**, P({t2} win) = **{100*pA:.1f}%**")

# --------------- College Football ---------------
elif page=="College Football":
    st.subheader("🏈🎓 College Football — 2025 (auto from CFBD)")
    if not CFBD_API_KEY:
        st.info('Add your CFBD key in Secrets as `CFBD_API_KEY="..."` to enable this page.')
        st.stop()
    df = cfb_team_stats_2025()
    if df.empty:
        st.warning("Could not load CFBD stats (rate limit, key issue, or no games yet).")
        st.stop()

    home=st.selectbox("Home team", df["team"].tolist())
    away=st.selectbox("Away team", [t for t in df["team"].tolist() if t!=home])

    H=df[df["team"]==home].iloc[0]; A=df[df["team"]==away].iloc[0]
    mu_h=(H["off_ppg"]+A["def_ppg"])/2; mu_a=(A["off_ppg"]+H["def_ppg"])/2
    pH,pA,mH,mA=_poisson_game(mu_h,mu_a)
    st.markdown(f"**{home}** vs **{away}** — Expected points: **{mH:.1f}–{mA:.1f}** · "
                f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**")

    with st.expander("📊 Show team table"):
        st.dataframe(df.reset_index(drop=True))

# ---------------- Player Props ----------------
else:
    st.subheader("🎯 Player Props — upload your CSVs (defense embedded)")
    st.caption(f"Defense factors source: **{DEF_SRC}**")

    c1,c2,c3=st.columns(3)
    with c1: qb_up=st.file_uploader("QB CSV", type=["csv","xlsx"], key="qb")
    with c2: rb_up=st.file_uploader("RB CSV", type=["csv","xlsx"], key="rb")
    with c3: wr_up=st.file_uploader("WR CSV", type=["csv","xlsx"], key="wr")

    dfs={}
    if qb_up: dfs["QB"]=_read_any(qb_up).copy()
    if rb_up: dfs["RB"]=_read_any(rb_up).copy()
    if wr_up: dfs["WR"]=_read_any(wr_up).copy()
    if not dfs:
        st.info("Upload at least one of QB/RB/WR CSVs to begin.")
        st.stop()

    pos=st.selectbox("Market", ["QB","RB","WR"])
    df=dfs.get(pos, pd.DataFrame())
    if df.empty:
        st.warning(f"No {pos} CSV uploaded yet."); st.stop()

    name_col=_name_col(df); yards_col=_yard_col(df,pos)

    # Clean player names for dropdown (strip ranks and commas if present)
    def _clean_name(x:str)->str:
        s=str(x).strip()
        # drop any leading ordinal like "12, " or "12."
        s=s.split(",",1)[-1].strip() if "," in s and s.split(",")[0].strip().isdigit() else s
        s=s.split(".",1)[-1].strip() if "." in s and s.split(".")[0].strip().isdigit() else s
        return s
    names = df[name_col].astype(str).map(_clean_name).tolist()
    # keep a map from clean name -> original row
    df["_clean_name"]=names

    player=st.selectbox("Player", names)
    opp_in=st.text_input("Opponent team code (e.g., DAL, PHI). Aliases (KAN/NOR/GNB/SFO) OK.", value="")
    opp=_norm_team(opp_in)

    row=df.loc[df["_clean_name"]==player].head(1)
    csv_mean=float(pd.to_numeric(row[yards_col], errors="coerce").fillna(0).mean()) if not row.empty else 0.0
    line=st.number_input("Yardage line", value=round(csv_mean,1), step=0.5)

    est_sd=_estimate_sd(csv_mean,pos)
    def_factor = DEF_FACTORS.get(opp,1.00) if opp else 1.00
    adj_mean=csv_mean*def_factor
    p_over,p_under=_prop_sim(adj_mean,line,est_sd)

    st.success(
        f"**{player} — {('Passing' if pos=='QB' else 'Rush' if pos=='RB' else 'Receiving')} Yards**  \n"
        f"CSV mean: **{csv_mean:.1f}** · Defense factor ({opp or 'AVG'}): **×{def_factor:.3f}** → "
        f"Adjusted mean: **{adj_mean:.1f}**  \n"
        f"Line: **{line:.1f}** → **P(over) = {100*p_over:.1f}%**, **P(under) = {100*p_under:.1f}%**"
    )

    with st.expander("Show player row used"):
        show=row.drop(columns=["_clean_name"]) if not row.empty else df.head(5).drop(columns=["_clean_name"], errors="ignore")
        st.dataframe(show)

    with st.expander("Embedded NFL defense (EPA/play → factor)"):
        table=pd.DataFrame({
            "TEAM": list(DEF_EPA_MAP.keys()),
            "EPA/play": [DEF_EPA_MAP[k] for k in DEF_EPA_MAP.keys()],
            "DEF_FACTOR": [DEF_FACTORS[k] for k in DEF_EPA_MAP.keys()],
        }).sort_values("DEF_FACTOR")
        st.dataframe(table.reset_index(drop=True))
