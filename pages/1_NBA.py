# NBA Player Props — Odds API + NBA Stats (with Defense Scaling)
# FINAL VERSION — includes:
# ✅ Defense multipliers (your 2025 list)
# ✅ Accurate player team lookup
# ✅ Accurate opponent team detection
# ✅ Retry-safe NBA API calls
# ✅ Full slate mode
# ✅ Model projection (median of 10k sims)
# ✅ All markets included

import re, unicodedata, time, random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# NBA API
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo


# -------------------------------------------------
#  DEFENSE MULTIPLIERS YOU PROVIDED
# -------------------------------------------------
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

# -------------------------------------------------
#  SIM SETTINGS
# -------------------------------------------------
SIM_TRIALS = 10000
ODDS_SPORT = "basketball_nba"
SEASON = "2025-26"
FALLBACK_SEASON = "2024-25"


# -------------------------------------------------
#  VALID MARKETS
# -------------------------------------------------
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


# -------------------------------------------------
#  UTILITY FUNCTIONS
# -------------------------------------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(name: str) -> str:
    n = str(name or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n).lower()


# -------------------------------------------------
#  RETRY-SAFE GAMELOG FETCHER
# -------------------------------------------------
def fetch_gamelog(player_id: int, season: str, retries=3):
    """
    Safe NBA API gamelog fetch with retry + timeout protection.
    """
    for attempt in range(retries):
        try:
            time.sleep(0.25)
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season",
                timeout=10
            )
            df = gl.get_data_frames()[0]
            return df
        except:
            if attempt == retries - 1:
                return pd.DataFrame()
            time.sleep(0.5)
    return pd.DataFrame()


def sample_sd(sum_x, sum_x2, g):
    if g <= 1:
        return 0.0
    mean = sum_x / g
    var = (sum_x2 / g) - (mean * mean)
    var = var * (g / (g - 1))
    return max(np.sqrt(max(var, 1e-9)), 0.0001)


def agg_full_season(df: pd.DataFrame) -> Dict[str, float]:
    g = len(df)
    if g == 0:
        return {"g": 0}

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


def t_sim(mu, sd, line):
    sd = max(1e-6, sd)
    draws = mu + sd * np.random.standard_t(df=5, size=SIM_TRIALS)
    win_prob = (draws > line).mean()
    projection = np.median(draws)
    return win_prob, projection, draws


# -------------------------------------------------
#  PLAYER INDEX
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def get_player_index():
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    df["name_norm"] = df["full_name"].apply(normalize_name)
    return df[["id","full_name","name_norm"]]


def find_player_id(player_name: str) -> Optional[int]:
    df = get_player_index()
    n = normalize_name(player_name)
    hit = df[df["name_norm"] == n]
    if not hit.empty:
        return int(hit.iloc[0]["id"])
    return None


# -------------------------------------------------
#  STREAMLIT UI
# -------------------------------------------------
st.title("🏀 NBA Player Props — Advanced Simulation Model (w/ Defense Scaling)")

st.markdown("### 1) Season Locked to 2025-26")
st.caption("Stats come from 2025-26. If a player has 0 games, fallback is 2024-25.")

st.markdown("### 2) Odds API Settings")
api_key = st.text_input("Odds API Key", type="password")
region = st.selectbox("Region", ["us","us2","eu","uk"])
lookahead = st.slider("Lookahead days", 0, 7, 1)

markets = st.multiselect("Markets to fetch", VALID_MARKETS + list(UNSUPPORTED_MARKETS_HIDE), default=VALID_MARKETS)


# -------------------------------------------------
#  ODDS API HELPERS
# -------------------------------------------------
def odds_json(url, params):
    r = requests.get(url, params=params, timeout=20)
    return r.json()


def list_events():
    return odds_json(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events",
        {"apiKey": api_key, "regions": region, "daysFrom": 0, "daysTo": max(1,lookahead)}
    )


if not api_key:
    st.stop()

events = list_events()
if not events:
    st.error("No games found.")
    st.stop()

# Build label list
labels = []
for e in events:
    away = e.get("away_team","Away")
    home = e.get("home_team","Home")
    t = e.get("commence_time","")
    labels.append(f"{away} @ {home} — {t}")

pick = st.selectbox("Choose Game", labels)
event = events[labels.index(pick)]

event_id = event["id"]
home_team = event.get("home_team","")
away_team = event.get("away_team","")


# -------------------------------------------------
#  STEP 3 — BUILD PROJECTIONS
# -------------------------------------------------
st.markdown("### 3) Build Player Projections")

if st.button("📥 Build Projections"):
    props = odds_json(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
    )

    player_names = set()
    for bk in props.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") in UNSUPPORTED_MARKETS_HIDE: 
                continue
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm:
                    player_names.add(nm)

    rows = []

    for pn in sorted(player_names):
        pid = find_player_id(pn)
        if not pid:
            continue

        # ✅ Player team lookup
        info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
        team = info["TEAM_NAME"].iloc[0]

        # ✅ Opponent detection (no fallback)
        opp = away_team if team == home_team else home_team

        # ✅ Defense multiplier
        mult = DEF_RATINGS.get(opp, 1.0)

        # ✅ Game logs
        df = fetch_gamelog(pid, SEASON)
        stats = agg_full_season(df)

        if stats["g"] == 0:
            df = fetch_gamelog(pid, FALLBACK_SEASON)
            stats = agg_full_season(df)

        if stats["g"] == 0:
            continue

        stats["player"] = pn
        stats["team"] = team
        stats["opponent"] = opp
        stats["mult"] = mult

        rows.append(stats)

    proj = pd.DataFrame(rows)
    proj["player_norm"] = proj["player"].apply(normalize_name)
    st.session_state["proj"] = proj

    st.success("✅ Projections built successfully.")
    st.dataframe(proj)


# -------------------------------------------------
#  STEP 4 — RUN SIMULATION FOR SELECTED GAME
# -------------------------------------------------
st.markdown("### 4) Run Simulation (10,000 Sims)")

if st.button("🎯 Simulate"):
    proj = st.session_state.get("proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build projections first.")
        st.stop()

    props = odds_json(
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
    )

    proj = proj.set_index("player_norm")

    out = []

    for bk in props.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") not in VALID_MARKETS:
                continue

            mkey = m["key"]

            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm not in proj.index:
                    continue
                if pd.isna(o.get("point")):
                    continue

                row = proj.loc[nm]
                line = float(o["point"])
                side = o["name"]

                # Base means/SDs
                def g(col):
                    return row[f"mu_{col}"], row[f"sd_{col}"]

                if mkey=="player_points": mu,sd = g("PTS")
                elif mkey=="player_rebounds": mu,sd = g("REB")
                elif mkey=="player_assists": mu,sd = g("AST")
                elif mkey=="player_threes": mu,sd = g("FG3M")
                elif mkey=="player_blocks": mu,sd = g("BLK")
                elif mkey=="player_steals": mu,sd = g("STL")
                elif mkey=="player_turnovers": mu,sd = g("TOV")
                elif mkey=="player_field_goals": mu,sd = g("FGM")
                elif mkey=="player_frees_made": mu,sd = g("FTM")
                elif mkey=="player_frees_attempts": mu,sd = g("FTA")
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

                # ✅ Apply defense scaling to *all* markets
                mu = mu * row["mult"]

                win_prob, proj_val, _ = t_sim(mu, sd, line)
                if side == "Under":
                    win_prob = 1 - win_prob

                out.append({
                    "Player": row["player"],
                    "Team": row["team"],
                    "Opponent": row["opponent"],
                    "Defense Multiplier": row["mult"],
                    "Market": mkey,
                    "Side": side,
                    "Line": line,
                    "Model Projection": round(proj_val,2),
                    "Win Prob %": round(win_prob*100,2),
                })

    df_out = pd.DataFrame(out).sort_values(["Market","Win Prob %"], ascending=[True,False])
    st.dataframe(df_out)

    st.download_button(
        "Download Results CSV",
        df_out.to_csv(index=False).encode(),
        "nba_sim_results.csv",
        "text/csv"
    )


# -------------------------------------------------
#  FULL SLATE MODE
# -------------------------------------------------
st.markdown("### 5) 🔥 Full Slate Value Finder")

if st.button("📊 Run Full Slate Report"):
    all_ev = list_events()
    master = []

    for ev in all_ev:
        eid = ev["id"]
        h = ev.get("home_team","")
        a = ev.get("away_team","")

        props = odds_json(
            f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{eid}/odds",
            {"apiKey": api_key, "regions": region, "markets": ",".join(markets)}
        )

        pnames = set()
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") in UNSUPPORTED_MARKETS_HIDE: continue
                for o in m.get("outcomes", []):
                    pnames.add(normalize_name(o.get("description","")))

        rows_proj = []
        for pn in sorted(list(pnames)):
            pid = find_player_id(pn)
            if not pid:
                continue

            info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
            pteam = info["TEAM_NAME"].iloc[0]
            opp = a if pteam == h else h
            mult = DEF_RATINGS.get(opp, 1.0)

            df = fetch_gamelog(pid, SEASON)
            stx = agg_full_season(df)
            if stx["g"]==0:
                df = fetch_gamelog(pid, FALLBACK_SEASON)
                stx = agg_full_season(df)
            if stx["g"]==0:
                continue

            stx["player"]=pn
            stx["team"]=pteam
            stx["opp"]=opp
            stx["mult"]=mult

            rows_proj.append(stx)

        if not rows_proj:
            continue

        proj_df = pd.DataFrame(rows_proj)
        proj_df["player_norm"]=proj_df["player"].apply(normalize_name)
        proj_df = proj_df.set_index("player_norm")

        # simulate
        for bk in props.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") not in VALID_MARKETS:
                    continue
                mkey = m["key"]

                for o in m.get("outcomes", []):
                    nm = normalize_name(o.get("description",""))
                    if nm not in proj_df.index:
                        continue
                    if pd.isna(o.get("point")):
                        continue

                    row = proj_df.loc[nm]
                    line = float(o["point"])
                    side = o["name"]

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
                        sd=np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2 + row["sd_AST"]**2)
                    elif mkey=="player_points_rebounds":
                        mu=row["mu_PTS"]+row["mu_REB"]
                        sd=np.sqrt(row["sd_PTS"]**2 + row["sd_REB"]**2)
                    elif mkey=="player_points_assists":
                        mu=row["mu_PTS"]+row["mu_AST"]
                        sd=np.sqrt(row["sd_PTS"]**2 + row["sd_AST"]**2)
                    elif mkey=="player_rebounds_assists":
                        mu=row["mu_REB"]+row["mu_AST"]
                        sd=np.sqrt(row["sd_REB"]**2 + row["sd_AST"]**2)
                    elif mkey=="player_blocks_steals":
                        mu=row["mu_BLK"]+row["mu_STL"]
                        sd=np.sqrt(row["sd_BLK"]**2 + row["sd_STL"]**2)
                    else:
                        continue

                    mu = mu * row["mult"]

                    win_prob, proj_val, _ = t_sim(mu, sd, line)
                    edge = proj_val - line

                    master.append({
                        "Game": f"{a} @ {h}",
                        "Player": row["player"],
                        "Team": row["team"],
                        "Opponent": row["opp"],
                        "Market": mkey,
                        "Side": side,
                        "Line": line,
                        "Projection": round(proj_val,2),
                        "Edge": round(edge,2),
                    })

    full = pd.DataFrame(master).sort_values("Edge", ascending=False)
    st.dataframe(full.head(50))

    chart = full.head(20)[["Player","Edge"]].set_index("Player")
    st.bar_chart(chart)

    st.download_button(
        "Download Full Slate CSV",
        full.to_csv(index=False).encode(),
        "full_slate_value_report.csv",
        "text/csv"
    )
