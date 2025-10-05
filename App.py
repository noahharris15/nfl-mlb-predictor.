# app.py — NFL Player Props (ALL GAMES) with per-player defense adjustment (EPA) + simulation
import time, math, json, random
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# ────────────────────────────── CONFIG ──────────────────────────────
st.set_page_config(page_title="NFL Player Props • All Games", layout="wide")
st.title("🏈 NFL Player Props — all games (Odds API + real stats + defense-adjusted sims)")

# Put your key here or in st.secrets["ODDS_API_KEY"]
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "").strip() or "PASTE_YOUR_ODDS_API_KEY_HERE"

LOOKAHEAD_DAYS = st.slider("Lookahead days for events", 0, 7, 1)
REGION = st.selectbox("Region", ["us","us2","eu","uk","au"], index=0)
SEASON = st.number_input("Season for per-game averages", 2018, 2025, value=2025, step=1)

MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receiving_yds",
    "player_receptions",
    "player_anytime_td",
]

st.caption("We fetch all NFL events in the lookahead window, pull **five** player markets per game from The Odds API, "
           "average book lines, estimate defense-adjusted μ (from real per-game averages), and compute Over/Under probabilities "
           "with a conservative normal model. Anytime TD is treated as a Bernoulli (λ≈TDs per game → p≈1−e^(−λ)).")

# ───────────────── DEFENSE TABLE (embedded; EPA → multipliers) ─────────────────
# Using your EPA snapshot. We transform to pass/rush/recv multipliers centered at 1.0.
# You can tweak the scaling knobs below if you want stronger/weaker adjustments.
_DEFENSE_EPA = """
Team,Season,EPA/Play,Total EPA,Success %,EPA/Pass,EPA/Rush,Pass Yards,Comp %,Pass TD,Rush Yards,Rush TD,ADoT,Sack %,Scramble %,Int %
Minnesota Vikings,2025,-0.17,-38.71,0.4174,-0.37,0.06,685,0.6762,3,521,4,5.8,0.0813,0.065,0.0163
Jacksonville Jaguars,2025,-0.13,-33.41,0.379,-0.17,-0.05,984,0.5962,7,331,1,8.15,0.0409,0.0468,0.0526
Denver Broncos,2025,-0.11,-26.37,0.3796,-0.1,-0.12,853,0.5746,2,397,2,9.94,0.0974,0.0325,0.0065
Los Angeles Chargers,2025,-0.11,-25.57,0.3667,-0.17,0.01,710,0.5938,3,445,3,7.69,0.0823,0.1076,0.019
Detroit Lions,2025,-0.09,-21.51,0.3783,0.0,-0.22,894,0.6271,7,376,4,10.44,0.1,0.0571,0.0214
Philadelphia Eagles,2025,-0.08,-20.49,0.4291,-0.11,-0.04,860,0.5693,5,504,3,8.43,0.0329,0.0658,0.0197
Houston Texans,2025,-0.08,-19.59,0.4149,-0.16,0.04,790,0.5714,3,409,4,8.3,0.0733,0.04,0.0133
Los Angeles Rams,2025,-0.08,-18.57,0.4042,-0.12,0.0,851,0.664,5,394,2,8.12,0.0952,0.0544,0.0204
Seattle Seahawks,2025,-0.07,-18.65,0.4075,0.0,-0.19,910,0.6645,6,359,0,7.03,0.069,0.0575,0.0402
San Francisco 49ers,2025,-0.06,-14.94,0.4344,-0.09,-0.03,690,0.6829,5,462,2,7.02,0.0368,0.0588,0.0
Tampa Bay Buccaneers,2025,-0.06,-13.91,0.4202,-0.02,-0.11,832,0.6429,6,340,3,6.54,0.0645,0.1226,0.0065
Atlanta Falcons,2025,-0.05,-11.24,0.4279,-0.13,0.05,602,0.5769,5,436,2,10.08,0.0806,0.0806,0.0242
Cleveland Browns,2025,-0.05,-10.73,0.379,0.06,-0.17,689,0.6442,8,281,2,7.95,0.0924,0.0336,0.0168
Indianapolis Colts,2025,-0.05,-11.02,0.4638,-0.04,-0.05,946,0.6643,8,384,2,6.69,0.0633,0.0506,0.0253
Kansas City Chiefs,2025,-0.02,-3.64,0.4487,-0.09,0.09,778,0.6694,4,508,4,8.27,0.0694,0.0903,0.0208
Arizona Cardinals,2025,-0.01,-2.38,0.4559,0.06,-0.14,1068,0.6369,5,384,2,8.06,0.0444,0.0222,0.0111
Las Vegas Raiders,2025,-0.01,-1.85,0.4315,0.14,-0.22,948,0.6565,5,411,4,7.98,0.0544,0.0544,0.0136
Green Bay Packers,2025,0.0,0.2,0.4094,0.03,-0.07,886,0.6815,6,310,3,6.87,0.0632,0.0345,0.0115
Chicago Bears,2025,0.0,0.89,0.4912,0.01,0.0,886,0.7368,10,658,4,6.75,0.0407,0.0325,0.0569
Buffalo Bills,2025,0.02,3.33,0.4208,-0.06,0.1,564,0.6214,6,657,5,6.87,0.0732,0.0894,0.0163
Carolina Panthers,2025,0.04,9.13,0.4133,0.03,0.05,802,0.6239,4,517,5,7.5,0.0155,0.0775,0.031
Pittsburgh Steelers,2025,0.04,11.09,0.461,0.11,-0.05,1131,0.6957,7,488,4,7.6,0.087,0.0559,0.0311
Washington Commanders,2025,0.04,10.44,0.4183,0.18,-0.12,1062,0.6098,7,430,3,10.83,0.0714,0.05,0.0071
New England Patriots,2025,0.05,12.43,0.4693,0.19,-0.15,1024,0.712,7,310,2,7.68,0.0725,0.0217,0.0217
New York Giants,2025,0.07,18.22,0.4613,-0.01,0.19,1021,0.6375,5,612,6,6.15,0.0562,0.0449,0.0169
New Orleans Saints,2025,0.07,17.94,0.4417,0.2,-0.06,884,0.7117,9,475,4,7.4,0.0853,0.0543,0.0078
Cincinnati Bengals,2025,0.1,27.12,0.4731,0.13,0.04,1089,0.6536,8,543,5,6.99,0.0366,0.0305,0.0305
New York Jets,2025,0.11,25.77,0.3959,0.23,-0.03,834,0.6577,7,522,4,6.11,0.0476,0.0714,0.0
Tennessee Titans,2025,0.12,30.47,0.4435,0.16,0.07,935,0.6984,6,566,7,6.82,0.0294,0.0441,0.0221
Baltimore Ravens,2025,0.14,39.54,0.4685,0.14,0.12,1084,0.6667,9,565,7,8.04,0.0233,0.0523,0.0058
Dallas Cowboys,2025,0.25,65.26,0.4943,0.4,0.06,1237,0.7333,10,493,6,9.19,0.034,0.0476,0.0068
Miami Dolphins,2025,0.25,59.66,0.5397,0.34,0.12,941,0.7757,7,632,5,6.15,0.0615,0.1154,0.0
"""

def load_defense_table() -> pd.DataFrame:
    df = pd.read_csv(StringIO(_DEFENSE_EPA))
    # Convert EPA per play to multipliers. Negative EPA (better defense) → <1 multiplier.
    # Knobs: larger K => stronger effect.
    K_PASS, K_RUSH, K_RECV = 0.9, 0.9, 0.7
    df["pass_adj"] = (1.0 - K_PASS * df["EPA/Pass"]).clip(0.6, 1.4)
    df["rush_adj"] = (1.0 - K_RUSH * df["EPA/Rush"]).clip(0.6, 1.4)
    # Use EPA/Pass for receiving context too (can blend with overall if you like)
    df["recv_adj"] = (1.0 - K_RECV * df["EPA/Pass"]).clip(0.6, 1.4)
    df["Team"] = df["Team"].astype(str)
    return df[["Team","pass_adj","rush_adj","recv_adj"]]

def_table = load_defense_table()

# ─────────────────────────── Helper: player → team ────────────────────────────
@st.cache_data(show_spinner=False)
def build_player_team_map(season: int) -> Dict[str,str]:
    import nfl_data_py as nfl
    rosters = nfl.import_rosters([season])
    # Prefer full franchise names if present; build a map from abbr to full via schedules
    sched = nfl.import_schedules([season])
    # derive mapping abbr->full from schedules (home/away columns are full names)
    teams_full = pd.unique(pd.concat([sched["home_team"], sched["away_team"]]).dropna())
    # crude abbr->full by best match
    def _best_full(abbr: str) -> Optional[str]:
        abbr = str(abbr).strip().upper()
        cand = [t for t in teams_full if abbr in t.upper() or t.upper().startswith(abbr)]
        return cand[0] if cand else None
    m = {}
    for _, r in rosters.iterrows():
        name = f"{r.get('first_name','').strip()} {r.get('last_name','').strip()}".strip()
        abbr = str(r.get("team")) if "team" in rosters.columns else str(r.get("team_abbr"))
        full = _best_full(abbr) or abbr
        m[name.lower()] = full
    return m

def resolve_player_team(player: str, roster_map: Dict[str,str]) -> Optional[str]:
    return roster_map.get(player.lower())

# ───────────────────────── Odds API wrappers (events + markets) ───────────────
EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
EVENT_ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"

@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=8))
def _get(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r

def fetch_events(days: int, region: str) -> List[dict]:
    params = {"apiKey": ODDS_API_KEY, "daysFrom": 0, "daysTo": days, "regions": region}
    r = _get(EVENTS_URL, params)
    return r.json()

def fetch_event_player_props(event_id: str, region: str, markets: List[str]) -> pd.DataFrame:
    url = f"{EVENT_ODDS_BASE}/{event_id}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"}
    r = _get(url, params)
    js = r.json()
    rows = []
    for bk in js.get("bookmakers", []):
        book_key = bk.get("key")
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                # player markets have description = player name
                rows.append({
                    "event_id": js.get("id"),
                    "commence_time": js.get("commence_time"),
                    "home_team": js.get("home_team"),
                    "away_team": js.get("away_team"),
                    "book": book_key,
                    "market": mkey,
                    "side": str(o.get("name")).title(),            # "Over"/"Under" or "Yes"/"No"
                    "player": o.get("description"),
                    "line": o.get("point"),
                    "price": o.get("price"),
                })
    df = pd.DataFrame(rows)
    # keep only real player rows
    if not df.empty:
        df = df[df["player"].notna()].copy()
    return df

# ─────────────────────────── Player averages & sims ────────────────────────────
@st.cache_data(show_spinner=False)
def nfl_player_avgs_per_game(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl
    df = nfl.import_seasonal_data([season])
    g = pd.to_numeric(df.get("games"), errors="coerce").replace(0, np.nan)
    out = pd.DataFrame({
        "player": df.get("player_display_name"),
        "pass_yds_pg": pd.to_numeric(df.get("passing_yards"), errors="coerce") / g,
        "rush_yds_pg": pd.to_numeric(df.get("rushing_yards"), errors="coerce") / g,
        "rec_yds_pg":  pd.to_numeric(df.get("receiving_yards"), errors="coerce") / g,
        "recs_pg":     pd.to_numeric(df.get("receptions"), errors="coerce") / g,
        "rush_tds_pg": pd.to_numeric(df.get("rushing_tds"), errors="coerce") / g,
        "rec_tds_pg":  pd.to_numeric(df.get("receiving_tds"), errors="coerce") / g,
    }).fillna(0.0)
    out["any_td_pg"] = out["rush_tds_pg"] + out["rec_tds_pg"]
    return out

def lookup_mu(avgs: pd.DataFrame, player: str, market: str) -> Optional[float]:
    row = avgs.loc[avgs["player"].str.lower() == player.lower()]
    if row.empty: return None
    r = row.iloc[0]
    return {
        "player_pass_yds": float(r["pass_yds_pg"]),
        "player_rush_yds": float(r["rush_yds_pg"]),
        "player_receiving_yds": float(r["rec_yds_pg"]),
        "player_receptions": float(r["recs_pg"]),
        "player_anytime_td": max(0.01, float(r["any_td_pg"])),
    }[market]

def sigma_for(market: str, mu: float) -> float:
    if market == "player_pass_yds": return max(15.0, 0.18*mu)
    if market in ("player_rush_yds","player_receiving_yds"): return max(10.0, 0.22*mu)
    if market == "player_receptions": return max(0.6, 0.35*mu)
    if market == "player_anytime_td": return 0.35*max(0.15, mu)
    return max(1.0, 0.25*mu)

def apply_defense(market: str, mu: float, pass_adj: float, rush_adj: float, recv_adj: float) -> float:
    if market == "player_pass_yds": return mu * pass_adj
    if market == "player_rush_yds": return mu * rush_adj
    if market in ("player_receiving_yds", "player_receptions"): return mu * recv_adj
    if market == "player_anytime_td": return mu * (0.5*rush_adj + 0.5*recv_adj)
    return mu

# ───────────────────────────── RUN (all games) ────────────────────────────────
run_btn = st.button("Fetch ALL games & simulate", type="primary")
if run_btn:
    if not ODDS_API_KEY or ODDS_API_KEY.startswith("PASTE_"):
        st.error("Add your Odds API key (st.secrets['ODDS_API_KEY'] or edit ODDS_API_KEY in this file).")
        st.stop()

    with st.spinner("Fetching event list…"):
        try:
            events = fetch_events(LOOKAHEAD_DAYS, REGION)
        except Exception as e:
            st.error(f"Events fetch failed: {e}")
            st.stop()

    if not events:
        st.warning("No events returned for the selected window.")
        st.stop()

    roster_map = build_player_team_map(SEASON)
    avgs = nfl_player_avgs_per_game(SEASON)

    all_rows = []
    progress = st.progress(0.0)
    for idx, ev in enumerate(events):
        progress.progress((idx+1)/len(events))
        try:
            df = fetch_event_player_props(ev["id"], REGION, MARKETS)
        except Exception as e:
            st.warning(f"Skipped event {ev.get('home_team')} vs {ev.get('away_team')}: {e}")
            continue
        if df.empty:
            continue

        # Average lines across books per (market,player,side)
        grp = df.groupby(["event_id","commence_time","home_team","away_team","market","player","side"], as_index=False).agg(
            line=("line","mean"),
            books=("book","nunique")
        )

        # prepare defense multipliers for home and away
        def _def_multipliers(team_full: str) -> Tuple[float,float,float]:
            r = def_table.loc[def_table["Team"].str.lower() == str(team_full).lower()]
            if r.empty: return (1.0,1.0,1.0)
            rr = r.iloc[0]
            return float(rr["pass_adj"]), float(rr["rush_adj"]), float(rr["recv_adj"])

        pH,rH,cH = _def_multipliers(ev["home_team"])
        pA,rA,cA = _def_multipliers(ev["away_team"])

        for _, r in grp.iterrows():
            market = r["market"]; player = r["player"]; side = r["side"]
            line = r["line"]; home = r["home_team"]; away = r["away_team"]

            # player team -> pick the correct opponent defense
            pteam = resolve_player_team(player, roster_map)
            if pteam and pteam.lower() == home.lower():
                opp = away; pass_adj, rush_adj, recv_adj = pA,rA,cA
            elif pteam and pteam.lower() == away.lower():
                opp = home; pass_adj, rush_adj, recv_adj = pH,rH,cH
            else:
                # fallback: assume away opp
                opp = away; pass_adj, rush_adj, recv_adj = pA,rA,cA

            mu0 = lookup_mu(avgs, player, market)
            if mu0 is None:
                continue

            mu = apply_defense(market, mu0, pass_adj, rush_adj, recv_adj)
            sd = sigma_for(market, mu)

            if market == "player_anytime_td":
                p_score = 1.0 - math.exp(-mu)  # Bernoulli approx
                prob = p_score*100.0 if side=="Over" else (1.0-p_score)*100.0
                used_line = 0.5
            else:
                if pd.isna(line):  # no line? skip
                    continue
                used_line = float(line)
                if side == "Over":
                    prob = (1.0 - norm.cdf(used_line, loc=mu, scale=sd)) * 100.0
                else:
                    prob = (norm.cdf(used_line, loc=mu, scale=sd)) * 100.0

            all_rows.append({
                "game": f"{home} @ {away}",
                "commence_time": r["commence_time"],
                "market": market,
                "player": player,
                "side": side,
                "line": round(used_line,2),
                "mu": round(mu,3),
                "sd": round(sd,3),
                "prob": round(float(prob),2),
                "opp_def": opp,
                "pass_adj": round(pass_adj,3),
                "rush_adj": round(rush_adj,3),
                "recv_adj": round(recv_adj,3),
                "books": int(r["books"]),
            })

        # small pause to be nice to the API
        time.sleep(0.25)

    progress.empty()
    results = pd.DataFrame(all_rows).sort_values(["market","prob"], ascending=[True, False]).reset_index(drop=True)

    if results.empty:
        st.warning("No player prop lines were returned by the API for the selected window/markets.")
        st.stop()

    st.subheader("All games — simulated props (defense-adjusted)")
    st.dataframe(results, use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="nfl_player_props_all_games.csv",
        mime="text/csv",
    )

    # Quick summaries
    st.markdown("#### Top Overs / Unders by market")
    for m in MARKETS:
        sub = results[results["market"] == m]
        if sub.empty: 
            continue
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"**{m} — Top 10 Overs**")
            st.dataframe(sub[sub["side"]=="Over"].nlargest(10, "prob")[["game","player","line","mu","prob"]], use_container_width=True)
        with c2:
            st.markdown(f"**{m} — Top 10 Unders**")
            st.dataframe(sub[sub["side"]=="Under"].nlargest(10, "prob")[["game","player","line","mu","prob"]], use_container_width=True)
