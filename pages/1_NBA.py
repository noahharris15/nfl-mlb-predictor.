# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 10k sims + Defense Scaling
# Place this file at: pages/1_NBA.py
# IMPORTANT: Do NOT call st.set_page_config here (it's in your main app).

import re, unicodedata, datetime as dt, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- NBA Stats ----------
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo

# ✅ Title
st.title("🏀 NBA Player Props — Advanced Model (with Defense Strength)")

SIM_TRIALS = 10000

# ✅ Defense multipliers you provided
DEF_RATINGS = {
    "Oklahoma City Thunder": 1.031,
    "San Antonio Spurs": 1.053,
    "Portland Trail Blazers": 1.073,
    "Miami Heat": 1.073,
    "Denver Nuggets": 1.074,
    "Detroit Pistons": 1.076,
    "Cleveland Cavaliers": 1.084,
    "Dallas Mavericks": 1.093,
    "Boston Celtics": 1.097,
    "Orlando Magic": 1.100,
    "Houston Rockets": 1.106,
    "Golden State Warriors": 1.109,
    "Indiana Pacers": 1.112,
    "Philadelphia 76ers": 1.116,
    "Chicago Bulls": 1.122,
    "Atlanta Hawks": 1.123,
    "Los Angeles Lakers": 1.127,
    "Milwaukee Bucks": 1.133,
    "Minnesota Timberwolves": 1.135,
    "Phoenix Suns": 1.137,
    "New York Knicks": 1.138,
    "Los Angeles Clippers": 1.141,
    "Memphis Grizzlies": 1.147,
    "Charlotte Hornets": 1.149,
    "Utah Jazz": 1.150,
    "Toronto Raptors": 1.152,
    "Sacramento Kings": 1.153,
    "Washington Wizards": 1.167,
    "New Orleans Pelicans": 1.226,
    "Brooklyn Nets": 1.249,
}

# -------------------- VALID MARKETS --------------------
VALID_MARKETS = [
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_blocks", "player_steals", "player_blocks_steals", "player_turnovers",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
    "player_field_goals", "player_frees_made", "player_frees_attempts",
]

UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket","player_first_team_basket","player_double_double",
    "player_triple_double","player_points_q1","player_rebounds_q1","player_assists_q1",
}

ODDS_SPORT = "basketball_nba"

# ------------------ Utilities ------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n).lower()

def t_sim(mu: float, sd: float, line: float, trials: int = SIM_TRIALS):
    sd = max(1e-6, sd)
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    win_prob = (draws > line).mean()
    projection = np.median(draws)
    return win_prob, projection, draws

def sample_sd(sum_x, sum_x2, g, floor=0.0):
    if g <= 1:
        return 0.0
    mean = sum_x / g
    var = (sum_x2 / g) - (mean**2)
    var = var * (g / (g - 1))
    return max(np.sqrt(max(var, 1e-9)), floor)

# ------------------ Player Index ------------------
@st.cache_data(show_spinner=False)
def _players_index():
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    df["name_norm"] = df["full_name"].apply(normalize_name)
    df["first_norm"] = df["first_name"].apply(normalize_name)
    df["last_norm"] = df["last_name"].apply(normalize_name)
    return df[["id","full_name","name_norm","first_norm","last_norm"]]

def find_player_id_by_name(name):
    df = _players_index()
    n = normalize_name(name)
    parts = n.split()

    hit = df[df["name_norm"] == n]
    if not hit.empty:
        return int(hit.iloc[0]["id"])

    if len(parts) == 2:
        f, l = parts
        cand = df[df["last_norm"].str.contains(l)]
        cand = cand[cand["first_norm"].str.contains(f[:1])]
        if not cand.empty:
            return int(cand.iloc[0]["id"])

    cand = df[df["name_norm"].str.contains(parts[-1])]
    if not cand.empty:
        return int(cand.iloc[0]["id"])

    return None

# ------------------ Game logs ------------------
def fetch_gamelog(player_id, season):
    time.sleep(0.2)
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star="Regular Season")
    df = gl.get_data_frames()[0]
    return df

def agg_full_season(df):
    g = len(df)
    if g == 0:
        return {"g":0}

    sums = {}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        sums[col] = x.sum()
        sums["sq_"+col] = (x**2).sum()

    out = {"g": g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        out["mu_"+col] = sums[col] / g
        out["sd_"+col] = sample_sd(sums[col], sums["sq_"+col], g)
    return out

# ------------------ UI ------------------
st.markdown("### 1) Season Locked to 2025-26")
SEASON = "2025-26"
FALLBACK = "2024-25"

st.markdown("### 2) Odds API Settings")
api_key = st.text_input("Odds API Key", "", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets = st.multiselect("Markets:", VALID_MARKETS + list(UNSUPPORTED_MARKETS_HIDE), VALID_MARKETS)

def odds_get(url, params):
    r = requests.get(url, params=params, timeout=25)
    return r.json()

def list_events():
    return odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "regions": region, "daysFrom": 0, "daysTo": lookahead if lookahead>0 else 1}
    )

if not api_key:
    st.info("Enter API key to load games.")
    st.stop()

events = list_events()

if not events:
    st.error("No games found.")
    st.stop()

# Build event labels
event_labels = []
for e in events:
    away = e.get("away_team") or "Away"
    home = e.get("home_team") or "Home"
    date = e.get("commence_time","")
    event_labels.append(f"{away} @ {home} — {date}")

pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

home_team = event.get("home_team","")
away_team = event.get("away_team","")

# ------------------ Build projections ------------------
st.markdown("### 3) Build Player Projections")
if st.button("📥 Build NBA projections"):

    props_preview = odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
    )

    player_names = set()
    for bk in props_preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE: continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm:
                    player_names.add(nm)

    rows = []

    for pn in sorted(player_names):
        pid = find_player_id_by_name(pn)
        if not pid:
            continue

        # ✅ Fetch player's actual team
        info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
        player_team = info["TEAM_NAME"].iloc[0]

        # ✅ Determine opponent correctly
        if player_team == home_team:
            opponent = away_team
        else:
            opponent = home_team

        # ✅ Get defense multiplier
        def_mult = DEF_RATINGS.get(opponent, 1.0)

        # ✅ Load logs
        df = fetch_gamelog(pid, SEASON)
        stats = agg_full_season(df)

        if stats["g"] == 0:
            df = fetch_gamelog(pid, FALLBACK)
            stats = agg_full_season(df)

        if stats["g"] == 0:
            continue

        stats["player"] = pn
        stats["team"] = player_team
        stats["opponent"] = opponent
        stats["def_multiplier"] = def_mult

        rows.append(stats)

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["player"].apply(normalize_name)

    st.session_state["proj"] = proj
    st.success("Projections built.")
    st.dataframe(proj)

# ------------------ Run Simulation ------------------
st.markdown("### 4) Simulate Props (10k Sims)")
if st.button("🎯 Run Simulation"):

    proj = st.session_state.get("proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build projections first.")
        st.stop()

    props = odds_get(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
    )

    proj = proj.set_index("player_norm")

    out_rows = []

    for bk in props.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey not in VALID_MARKETS:
                continue

            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                if name not in proj.index:
                    continue

                row = proj.loc[name]
                side = o["name"]
                line = float(o["point"])

                # ✅ Base means & SDs
                def get(col):
                    return row[f"mu_{col}"], row[f"sd_{col}"]

                # ✅ Determine μ and σ normally
                if mkey=="player_points":   mu,sd = get("PTS")
                elif mkey=="player_rebounds": mu,sd = get("REB")
                elif mkey=="player_assists":  mu,sd = get("AST")
                elif mkey=="player_threes":   mu,sd = get("FG3M")
                elif mkey=="player_blocks":   mu,sd = get("BLK")
                elif mkey=="player_steals":   mu,sd = get("STL")
                elif mkey=="player_turnovers": mu,sd = get("TOV")
                elif mkey=="player_field_goals": mu,sd = get("FGM")
                elif mkey=="player_frees_made": mu,sd = get("FTM")
                elif mkey=="player_frees_attempts": mu,sd = get("FTA")
                elif mkey=="player_points_rebounds_assists":
                    mu = row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)
                elif mkey=="player_points_rebounds":
                    mu = row["mu_PTS"]+row["mu_REB"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2)
                elif mkey=="player_points_assists":
                    mu = row["mu_PTS"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2)
                elif mkey=="player_rebounds_assists":
                    mu = row["mu_REB"]+row["mu_AST"]
                    sd = np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2)
                elif mkey=="player_blocks_steals":
                    mu = row["mu_BLK"]+row["mu_STL"]
                    sd = np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2)
                else:
                    continue

                # ✅ APPLY DEFENSE MULTIPLIER TO EVERYTHING
                mu = mu * row["def_multiplier"]

                # ✅ RUN SIM
                win_prob, projection, draws = t_sim(mu, sd, line)

                if side=="Under":
                    win_prob = 1 - win_prob

                out_rows.append({
                    "Player": row["player"],
                    "Team": row["team"],
                    "Opponent": row["opponent"],
                    "Defense Multiplier": row["def_multiplier"],
                    "Market": mkey,
                    "Side": side,
                    "Line": line,
                    "Model Projection": round(projection,2),
                    "Win Prob %": round(win_prob*100,2),
                })

    results = pd.DataFrame(out_rows)
    results = results.sort_values(["Market","Win Prob %"], ascending=[True,False])

    st.dataframe(results)

    st.download_button(
        "⬇️ Download CSV",
        results.to_csv(index=False).encode(),
        "nba_sim_results.csv",
        "text/csv"
    )

# ------------------ FULL SLATE VALUE REPORT ------------------
st.markdown("### 5) 🔥 Full-Slate Value Finder")

if st.button("📊 Run Full Slate Value Report"):

    all_events = list_events()
    master = []

    for ev in all_events:

        eid = ev["id"]
        homeT = ev.get("home_team","")
        awayT = ev.get("away_team","")

        props = odds_get(
            f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
            {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
        )

        # collect players
        pnames = set()
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") in UNSUPPORTED_MARKETS_HIDE: continue
                for o in m.get("outcomes", []):
                    nm = normalize_name(o.get("description"))
                    if nm: pnames.add(nm)

        rows = []
        # build projections
        for pn in sorted(pnames):

            pid = find_player_id_by_name(pn)
            if not pid:
                continue

            info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
            pteam = info["TEAM_NAME"].iloc[0]

            opp = awayT if pteam == homeT else homeT

            mult = DEF_RATINGS.get(opp, 1.0)

            df = fetch_gamelog(pid, SEASON)
            stx = agg_full_season(df)
            if stx["g"]==0:
                df = fetch_gamelog(pid, FALLBACK)
                stx = agg_full_season(df)
            if stx["g"] == 0:
                continue

            stx["player"]=pn
            stx["team"]=pteam
            stx["opponent"]=opp
            stx["mult"]=mult
            rows.append(stx)

        if not rows:
            continue

        proj = pd.DataFrame(rows)
        proj["player_norm"]=proj["player"].apply(normalize_name)
        proj = proj.set_index("player_norm")

        # simulate all props
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                mkey = m.get("key")
                if mkey not in VALID_MARKETS:
                    continue
                for o in m.get("outcomes", []):
                    nm = normalize_name(o.get("description"))
                    if nm not in proj.index:
                        continue
                    if pd.isna(o.get("point")):
                        continue

                    row = proj.loc[nm]
                    side = o["name"]
                    line = float(o["point"])

                    # get base μ, σ
                    def g(col):
                        return row[f"mu_{col}"], row[f"sd_{col}"]

                    if mkey=="player_points": mu,sd=g("PTS")
                    elif mkey=="player_rebounds": mu,sd=g("REB")
                    elif mkey=="player_assists": mu,sd=g("AST")
                    elif mkey=="player_threes": mu,sd=g("FG3M")
                    elif mkey=="player_blocks": mu,sd=g("BLK")
                    elif mkey=="player_steals": mu,sd=g("STL")
                    elif mkey=="player_turnovers": mu,sd=g("TOV")
                    elif mkey=="player_field_goals": mu,sd=g("FGM")
                    elif mkey=="player_frees_made": mu,sd=g("FTM")
                    elif mkey=="player_frees_attempts": mu,sd=g("FTA")
                    elif mkey=="player_points_rebounds_assists":
                        mu=row["mu_PTS"]+row["mu_REB"]+row["mu_AST"]
                        sd=np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2+row["sd_AST"]**2)
                    elif mkey=="player_points_rebounds":
                        mu=row["mu_PTS"]+row["mu_REB"]
                        sd=np.sqrt(row["sd_PTS"]**2+row["sd_REB"]**2)
                    elif mkey=="player_points_assists":
                        mu=row["mu_PTS"]+row["mu_AST"]
                        sd=np.sqrt(row["sd_PTS"]**2+row["sd_AST"]**2)
                    elif mkey=="player_rebounds_assists":
                        mu=row["mu_REB"]+row["mu_AST"]
                        sd=np.sqrt(row["sd_REB"]**2+row["sd_AST"]**2)
                    elif mkey=="player_blocks_steals":
                        mu=row["mu_BLK"]+row["mu_STL"]
                        sd=np.sqrt(row["sd_BLK"]**2+row["sd_STL"]**2)
                    else:
                        continue

                    # ✅ Apply defense scaling
                    mu = mu * row["mult"]

                    win_prob, projection, draws = t_sim(mu,sd,line)
                    edge = projection - line

                    master.append({
                        "Game": f"{awayT} @ {homeT}",
                        "Player": row["player"],
                        "Team": row["team"],
                        "Opponent": row["opponent"],
                        "Market": mkey,
                        "Side": side,
                        "Line": line,
                        "Projection": round(projection,2),
                        "Edge": round(edge,2),
                    })

    full = pd.DataFrame(master)
    full = full.sort_values("Edge", ascending=False)

    st.markdown("## 🔥 Top Value Plays (Full Slate)")
    st.dataframe(full.head(50))

    chart = full.head(20)[["Player","Edge"]].set_index("Player")
    st.bar_chart(chart)

    st.download_button(
        "⬇️ Download Full Slate CSV",
        full.to_csv(index=False).encode(),
        "full_slate_edges.csv",
        "text/csv"
    )
