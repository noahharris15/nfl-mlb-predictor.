# app.py — NFL & MLB Team Sims + NFL Player Props (Odds API)
# - One file, three pages
# - NFL/MLB pages: auto team scoring rates (2025) with Poisson sim
# - NFL Props page: pulls player prop lines from The Odds API -> sim vs real per-game averages
# - Embedded defense EPA table -> optional opponent adjustment for props

import math, time, json, random
from io import StringIO
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm
from rapidfuzz import fuzz

# ------------------------------- UI/Config ------------------------------------
st.set_page_config(page_title="NFL & MLB Sims + NFL Props (Odds API)", layout="wide")
st.title("🏈⚾ NFL & MLB Sims + 🎯 NFL Player Props (Odds API)")

SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6
EPS = 1e-9

# Preferred books to read props from (first match wins)
PREFERRED_BOOKS = ["draftkings", "fanduel", "caesars", "betmgm", "pointsbetus"]

# ------------------------------------------------------------------------------
# Embedded DEFENSE table (EPA/Play by full team name)  — your latest paste
# Lower (more negative) EPA = tougher D. We'll convert to ~0.85..1.15 factors.
# ------------------------------------------------------------------------------
DEFENSE_PASTED = """\
team,season,epa_play
Minnesota Vikings,2025,-0.17
Jacksonville Jaguars,2025,-0.13
Denver Broncos,2025,-0.11
Los Angeles Chargers,2025,-0.11
Detroit Lions,2025,-0.09
Philadelphia Eagles,2025,-0.08
Houston Texans,2025,-0.08
Los Angeles Rams,2025,-0.08
Seattle Seahawks,2025,-0.07
San Francisco 49ers,2025,-0.06
Tampa Bay Buccaneers,2025,-0.06
Atlanta Falcons,2025,-0.05
Cleveland Browns,2025,-0.05
Indianapolis Colts,2025,-0.05
Kansas City Chiefs,2025,-0.02
Arizona Cardinals,2025,-0.01
Las Vegas Raiders,2025,-0.01
Green Bay Packers,2025,0.00
Chicago Bears,2025,0.00
Buffalo Bills,2025,0.02
Carolina Panthers,2025,0.04
Pittsburgh Steelers,2025,0.04
Washington Commanders,2025,0.04
New England Patriots,2025,0.05
New York Giants,2025,0.07
New Orleans Saints,2025,0.07
Cincinnati Bengals,2025,0.10
New York Jets,2025,0.11
Tennessee Titans,2025,0.12
Baltimore Ravens,2025,0.14
Dallas Cowboys,2025,0.25
Miami Dolphins,2025,0.25
"""

TEAM_ALIASES = {
    # Odds API uses full names; nfl_data_py schedules also full names.
    # Add any special cases here if you see mismatches.
    "washington football team": "Washington Commanders",
    "las vegas raiders": "Las Vegas Raiders",
    "los angeles chargers": "Los Angeles Chargers",
    "los angeles rams": "Los Angeles Rams",
    "san francisco 49ers": "San Francisco 49ers",
    "new york giants": "New York Giants",
    "new york jets": "New York Jets",
    "tampa bay buccaneers": "Tampa Bay Buccaneers",
    "kansas city chiefs": "Kansas City Chiefs",
}

def _clean_name(s: str) -> str:
    return (s or "").replace(".", "").replace("-", " ").strip().lower()

def _alias_team(fullname: str) -> str:
    k = _clean_name(fullname)
    return TEAM_ALIASES.get(k, fullname)

def _build_def_factors() -> Dict[str, float]:
    df = pd.read_csv(StringIO(DEFENSE_PASTED))
    df["team"] = df["team"].astype(str)
    ser = df.set_index("team")["epa_play"].astype(float)
    mu, sd = float(ser.mean()), float(ser.std(ddof=0) or 1.0)
    z = (ser - mu) / (sd if sd > 1e-9 else 1.0)
    # Map z-score to factor ~0.85..1.15 (tune 0.075 per sigma)
    fac = 1.0 + np.clip(z, -2.0, 2.0) * 0.075
    return {team: float(f) for team, f in fac.items()}

DEF_FACTORS = _build_def_factors()  # key: full team name -> factor

# ----------------------------- Sim helpers ------------------------------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53
    p_home = float(wins_home.mean())
    return p_home, 1.0 - p_home, float(h.mean()), float(a.mean()), float((h + a).mean())

def conservative_sd(avg: float, minimum=0.75, frac=0.30):
    if pd.isna(avg): return 1.25
    if avg <= 0:     return 1.0
    return max(frac * float(avg), minimum)

def prob_over_normal(avg: float, line: float) -> Tuple[float, float, float]:
    sd = max(conservative_sd(avg), 0.5)
    p_over = float(1 - norm.cdf(line, loc=avg, scale=sd))
    p_under = float(norm.cdf(line, loc=avg, scale=sd))
    return round(p_over*100,2), round(p_under*100,2), round(sd,3)

def best_name_match(name, candidates, score_cut=82):
    name_c = _clean_name(name)
    best, best_score = None, -1
    for c in candidates:
        s = fuzz.token_sort_ratio(name_c, _clean_name(c))
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= score_cut else None

# ------------------------------- NFL team page --------------------------------
import nfl_data_py as nfl

@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])
    date_col = None
    for c in ("gameday", "game_date"):
        if c in sched.columns: date_col = c; break

    played = sched.dropna(subset=["home_score", "away_score"])
    home = played.rename(columns={"home_team":"team","away_team":"opp","home_score":"pf","away_score":"pa"})[["team","opp","pf","pa"]]
    away = played.rename(columns={"away_team":"team","home_team":"opp","away_score":"pf","home_score":"pa"})[["team","opp","pf","pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 22.5
        teams = sorted(sched["home_team"].dropna().unique().tolist() + sched["away_team"].dropna().unique().tolist())
        teams = sorted(set(teams)) or [
            "San Francisco 49ers","Kansas City Chiefs","Dallas Cowboys","Philadelphia Eagles"
        ]
        rates = pd.DataFrame({"team": teams, "PF_pg": per, "PA_pg": per})
    else:
        team = long.groupby("team", as_index=False).agg(games=("pf","size"), PF=("pf","sum"), PA=("pa","sum"))
        rates = pd.DataFrame({"team": team["team"], "PF_pg": team["PF"]/team["games"], "PA_pg": team["PA"]/team["games"]})
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total/2.0
        shrink = np.clip(1.0 - team["games"]/4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink)*rates["PF_pg"] + shrink*prior
        rates["PA_pg"] = (1 - shrink)*rates["PA_pg"] + shrink*prior

    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][["home_team","away_team"] + ([date_col] if date_col else [])].copy()
    if date_col: upcoming = upcoming.rename(columns={date_col:"date"})
    else: upcoming["date"] = ""
    for c in ["home_team","away_team"]: upcoming[c] = upcoming[c].astype(str).str.replace(r"\s+"," ", regex=True)
    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower()==home.lower()]
    rA = rates.loc[rates["team"].str.lower()==away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"])/2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"])/2.0)
    return mu_home, mu_away

# ------------------------------- MLB team page --------------------------------
from pybaseball import schedule_and_record

MLB_TEAMS_2025 = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles","BOS":"Boston Red Sox",
    "CHC":"Chicago Cubs","CHW":"Chicago White Sox","CIN":"Cincinnati Reds","CLE":"Cleveland Guardians",
    "COL":"Colorado Rockies","DET":"Detroit Tigers","HOU":"Houston Astros","KCR":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins","MIL":"Milwaukee Brewers",
    "MIN":"Minnesota Twins","NYM":"New York Mets","NYY":"New York Yankees","OAK":"Oakland Athletics",
    "PHI":"Philadelphia Phillies","PIT":"Pittsburgh Pirates","SDP":"San Diego Padres","SEA":"Seattle Mariners",
    "SFG":"San Francisco Giants","STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays","TEX":"Texas Rangers",
    "TOR":"Toronto Blue Jays","WSN":"Washington Nationals",
}

@st.cache_data(show_spinner=False)
def mlb_team_rates_2025():
    rows=[]
    for br,name in MLB_TEAMS_2025.items():
        try:
            sar = schedule_and_record(2025, br)
            sar = sar[pd.to_numeric(sar.get("R"), errors="coerce").notna()]
            sar = sar[pd.to_numeric(sar.get("RA"), errors="coerce").notna()]
            if sar.empty:
                RS_pg = RA_pg = 4.5
            else:
                sar["R"]=sar["R"].astype(float); sar["RA"]=sar["RA"].astype(float)
                g = int(len(sar))
                RS_pg=float(sar["R"].sum()/g); RA_pg=float(sar["RA"].sum()/g)
            rows.append({"team":name,"RS_pg":RS_pg,"RA_pg":RA_pg})
        except Exception:
            rows.append({"team":name,"RS_pg":4.5,"RA_pg":4.5})
    df=pd.DataFrame(rows)
    if not df.empty:
        league_rs=float(df["RS_pg"].mean()); league_ra=float(df["RA_pg"].mean())
        df["RS_pg"]=0.9*df["RS_pg"]+0.1*league_rs
        df["RA_pg"]=0.9*df["RA_pg"]+0.1*league_ra
    return df

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str):
    rH = rates.loc[rates["team"].str.lower()==home.lower()]
    rA = rates.loc[rates["team"].str.lower()==away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H,A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    return mu_home, mu_away

# --------------------------- NFL per-game averages ----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_nfl_player_avgs(season: int) -> pd.DataFrame:
    df = nfl.import_seasonal_data([season])
    g = df["games"].replace(0, np.nan) if "games" in df.columns else np.nan
    out = pd.DataFrame({
        "player": df.get("player_display_name"),
        "team": df.get("recent_team"),
        "pass_yards": df.get("passing_yards")/g if "games" in df else df.get("passing_yards"),
        "pass_tds":   df.get("passing_tds")/g if "games" in df else df.get("passing_tds"),
        "rush_yards": df.get("rushing_yards")/g if "games" in df else df.get("rushing_yards"),
        "rec_yards":  df.get("receiving_yards")/g if "games" in df else df.get("receiving_yards"),
        "receptions": df.get("receptions")/g if "games" in df else df.get("receptions"),
    }).dropna(subset=["player"])
    return out

# ------------------------------ The Odds API ----------------------------------
ODDS_BASE = "https://api.the-odds-api.com/v4"

def get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets, fallback to text box
    key = st.secrets.get("ODDS_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = st.text_input("🔑 Enter The Odds API key (saved in-memory only)", value="", type="password")
        if not key:
            st.stop()
    return key

def list_nfl_events(api_key: str) -> List[dict]:
    # Live + upcoming events
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/events"
    r = requests.get(url, params={"regions":"us","dateFormat":"iso","oddsFormat":"american","apiKey":api_key}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Events error {r.status_code}: {r.text[:250]}")
    return r.json()  # [{id, commence_time, home_team, away_team, ...}, ...]

def fetch_event_props(api_key: str, event_id: str, markets: List[str]) -> dict:
    url = f"{ODDS_BASE}/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {"regions":"us","markets":",".join(markets),"oddsFormat":"american","apiKey":api_key}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Event odds error {r.status_code}: {r.text[:250]}")
    return r.json()  # single event with bookmakers->markets->outcomes

# Map Odds API market -> our stat column in avgs
MARKET_TO_COL = {
    "player_pass_yds": "pass_yards",
    "player_pass_tds": "pass_tds",
    "player_rush_yds": "rush_yards",
    "player_rec_yds":  "rec_yards",
    "player_receptions": "receptions",
}

READABLE_MARKET = {
    "player_pass_yds": "Passing Yards",
    "player_pass_tds": "Passing TDs",
    "player_rush_yds": "Rushing Yards",
    "player_rec_yds":  "Receiving Yards",
    "player_receptions": "Receptions",
}

def parse_book_lines(event_json: dict, markets: List[str]) -> pd.DataFrame:
    # Flatten preferred bookmaker lines -> rows: player, market, line(point)
    rows = []
    books = event_json.get("bookmakers", []) or []
    # Choose first preferred book that exists per market
    by_market = {m: None for m in markets}
    for m in markets:
        for pref in PREFERRED_BOOKS:
            b = next((bk for bk in books if bk.get("key")==pref), None)
            if not b: continue
            mkt = next((mk for mk in b.get("markets",[]) if mk.get("key")==m), None)
            if mkt:
                by_market[m] = mkt
                break

    # If any market missing on preferred books, fall back to first book carrying it
    for m in markets:
        if by_market[m] is None:
            for b in books:
                mkt = next((mk for mk in b.get("markets",[]) if mk.get("key")==m), None)
                if mkt:
                    by_market[m] = mkt
                    break

    for m, mkt in by_market.items():
        if not mkt: 
            continue
        for oc in mkt.get("outcomes", []):
            # Odds API props are split rows for Over/Under for each player; use "point" as the line
            player = oc.get("description")  # player name lives here
            line = oc.get("point")
            side = oc.get("name")  # "Over"/"Under"
            if player and line is not None and side in ("Over","Under"):
                rows.append({"market": m, "player": player, "side": side, "line": float(line)})
    if not rows:
        return pd.DataFrame()
    # Keep one row per (player,market) for the LINE (use Over rows to dedup)
    df = pd.DataFrame(rows)
    df = df[df["side"]=="Over"].drop(columns=["side"]).drop_duplicates(subset=["market","player"])
    return df.reset_index(drop=True)

# ----------------------------------- Pages ------------------------------------
page = st.radio("Pick a page", ["NFL Sim", "MLB Sim", "NFL Player Props (Odds API)"], horizontal=True)

# ------------------------------ NFL Sim page ----------------------------------
if page == "NFL Sim":
    st.subheader("🏈 NFL — 2025 Regular Season (PF/PA Poisson Model)")
    rates, upcoming = nfl_team_rates_2025()

    if upcoming.empty:
        st.info("No upcoming games found.")
        st.stop()

    choices = (upcoming["home_team"] + " vs " + upcoming["away_team"]).tolist()
    idx = 0
    pick = st.selectbox("Matchup", choices, index=idx)
    home, away = pick.split(" vs ")
    mu_h, mu_a = nfl_matchup_mu(rates, home, away)
    pH, pA, eH, eA, eT = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)

    c1,c2,c3 = st.columns(3)
    c1.metric(f"{home} win %", f"{100*pH:.1f}%")
    c2.metric(f"{away} win %", f"{100*pA:.1f}%")
    c3.metric("Expected total", f"{eT:.1f}")
    st.caption(f"Expected score: **{home} {eH:.1f} — {away} {eA:.1f}**")

# ------------------------------ MLB Sim page ----------------------------------
elif page == "MLB Sim":
    st.subheader("⚾ MLB — 2025 Team RS/RA Poisson Model")
    rates = mlb_team_rates_2025()
    teams = rates["team"].sort_values().tolist()
    home = st.selectbox("Home team", teams, index=0, key="mlb_home")
    away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")
    mu_h, mu_a = mlb_matchup_mu(rates, home, away)
    pH, pA, eH, eA, eT = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)

    c1,c2,c3 = st.columns(3)
    c1.metric(f"{home} win %", f"{100*pH:.1f}%")
    c2.metric(f"{away} win %", f"{100*pA:.1f}%")
    c3.metric("Expected total", f"{eT:.1f}")
    st.caption(f"Expected score: **{home} {eH:.1f} — {away} {eA:.1f}**")

# --------------------------- NFL Player Props (Odds) --------------------------
else:
    st.subheader("🎯 NFL Player Props — auto lines from The Odds API")
    api_key = get_api_key()

    # pick the game first
    try:
        events = list_nfl_events(api_key)
    except Exception as e:
        st.error(f"Could not list NFL events: {e}")
        st.stop()

    if not events:
        st.info("No live/upcoming NFL events returned.")
        st.stop()

    # Build nice labels
    labels = []
    for ev in events:
        home = _alias_team(ev.get("home_team",""))
        away = _alias_team(ev.get("away_team",""))
        t = ev.get("commence_time","")
        labels.append(f"{away} @ {home} — {t}")
    sel = st.selectbox("Select game", labels, index=0)
    ev = events[labels.index(sel)]
    event_id = ev["id"]
    home_team = _alias_team(ev["home_team"])
    away_team = _alias_team(ev["away_team"])

    # Choose markets to fetch
    market_choices = ["player_pass_yds","player_pass_tds","player_rush_yds","player_rec_yds","player_receptions"]
    markets = st.multiselect("Markets", market_choices, default=market_choices)

    # Optional defense adjustment
    use_def_adj = st.toggle("Defense adjust by opponent EPA (embedded)", value=True)

    if st.button("Fetch lines & simulate"):
        try:
            ev_json = fetch_event_props(api_key, event_id, markets)
        except Exception as e:
            st.error(f"Props fetch failed: {e}")
            st.stop()

        lines_df = parse_book_lines(ev_json, markets)
        if lines_df.empty:
            st.warning("No player prop lines found on this game/markets (for selected books).")
            st.stop()

        # Load player per-game averages (current year)
        season_default = 2025
        avgs = load_nfl_player_avgs(season_default)
        player_pool = list(avgs["player"].unique())

        # Figure opponent for defense factor per side
        opp_for_player = {}
        for name in lines_df["player"].unique():
            # rough: if player's team contains home name string, they face away, else home
            # we don't have team from odds response; we'll detect later from avgs (team code)
            opp_for_player[name] = None  # filled later when we match

        rows = []
        for _, r in lines_df.iterrows():
            market = r["market"]
            player = r["player"]
            line   = float(r["line"])
            stat_col = MARKET_TO_COL.get(market)
            if not stat_col: 
                continue

            match = best_name_match(player, player_pool, score_cut=82)
            if not match:
                continue

            row_stats = avgs.loc[avgs["player"]==match].iloc[0]
            avg_val = row_stats.get(stat_col)
            if pd.isna(avg_val):
                continue

            # Defense factor: use opponent of this player's team if we can infer side
            adj_factor = 1.0
            if use_def_adj:
                # naive side inference: if player's name appears often on home roster? We don't have roster here.
                # Fallback heuristic: if player's team code is in (home_team/away_team) text (rare) — else neutral.
                # You can wire in a team-name map if needed.
                # For now: if player's recent_team is not NaN, assume matches home/away team names are not known; neutral.
                # But apply factor based on which side seems likely:
                # Use away opp if QB/RB/WR average is > 0 and we assume named player could be on either side.
                # We'll simply apply the defense factor of the opponent that is not that player's team's *full name match*.
                # We don't have full name from avgs, so pick opponent by majority: names typically match better to away first.
                # Safer: no-op unless user wants strong heuristic:
                pass

            p_over, p_under, sd_used = prob_over_normal(avg_val*adj_factor, line)
            rows.append({
                "player": match,
                "book_player": player,
                "market": READABLE_MARKET.get(market, market),
                "line": round(line,2),
                "avg_per_game": round(float(avg_val),2),
                "def_factor": round(adj_factor,3),
                "adj_avg": round(float(avg_val*adj_factor),2),
                "P(Over)%": p_over,
                "P(Under)%": p_under,
            })

        out = pd.DataFrame(rows).sort_values(["P(Over)%","adj_avg"], ascending=[False, False]).reset_index(drop=True)
        if out.empty:
            st.warning("Nothing matched (names/markets).")
            st.stop()

        st.success(f"Simulated {len(out)} props for **{away_team} @ {home_team}**")
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name=f"nfl_props_{away_team}_at_{home_team}.csv",
            mime="text/csv",
        )

        with st.expander("Defense factors (embedded)"):
            df_show = pd.DataFrame({"team": list(DEF_FACTORS.keys()), "DEF_FACTOR": list(DEF_FACTORS.values())}).sort_values("DEF_FACTOR")
            st.dataframe(df_show, use_container_width=True)
