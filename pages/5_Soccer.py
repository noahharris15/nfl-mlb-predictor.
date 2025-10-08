# pages/3_Soccer.py
# Soccer props — Odds API + soccerdata/FBref
# Requires: soccerdata>=1.7, beautifulsoup4, lxml, html5lib, requests-cache
# Run whole app: streamlit run App.py

import re, math, unicodedata, sys, subprocess
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import requests
from rapidfuzz import process, fuzz

SIM_TRIALS = 10_000

# ---------------- UI / page ----------------
st.set_page_config(page_title="Soccer Props — Odds API + FBref", layout="wide")
st.title("⚽️ Soccer Player Props — Odds API + FBref (via soccerdata)")

# ---------------- small utils ----------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = n.replace("-", " ")
    n = re.sub(r"[.,'’]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n)

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def poisson_yes(lam: float) -> float:
    lam = max(1e-6, float(lam))
    return 1.0 - math.exp(-lam)

def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ---------------- try import soccerdata (show helpful error if missing) ----------------
soccerdata_ok = True
try:
    from soccerdata import FBref
except Exception as e:
    soccerdata_ok = False
    st.error("soccerdata not installed. Add to requirements: "
             "`soccerdata>=1.7 beautifulsoup4 lxml html5lib requests-cache`")
    st.caption(f"(Import error: {e})")

# ---------------- ESPN-equivalent crawl via soccerdata / FBref ----------------
@st.cache_data(show_spinner=True)
def fetch_fbref_player_game_logs(leagues: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull per-match player stats using the VALID stat_type names in soccerdata>=1.7:
    ['summary','keepers','passing','passing_types','defense','possession','misc']
    We need: minutes, goals, shots, shots_on_target, assists, cards, team, date, player.
    """
    frames = []
    for lg in leagues:
        try:
            yr = datetime.strptime(start_date, "%Y-%m-%d").year
            seasons = [f"{yr-1}-{yr}", f"{yr}-{yr+1}"]
            fb = FBref(leagues=[lg], seasons=seasons, no_cache=True)

            summary = fb.read_player_match_stats(stat_type="summary")
            passing = fb.read_player_match_stats(stat_type="passing")
            misc    = fb.read_player_match_stats(stat_type="misc")  # backup for cards if needed

            # Keep columns we care about; soccerdata sometimes varies names slightly
            keep_sum = ["player", "player_id", "team", "date", "minutes", "goals",
                        "shots", "shots_on_target", "assists"]
            for k in keep_sum:
                if k not in summary.columns:
                    summary[k] = np.nan
            sum_df = summary[keep_sum].copy()

            pas_cols = ["player_id", "date", "team", "assists"]
            for k in pas_cols:
                if k not in passing.columns:
                    passing[k] = np.nan
            pas_df = passing[pas_cols].copy()
            pas_df.rename(columns={"assists": "assists_pass"}, inplace=True)

            m_cols = ["player_id", "date", "team", "cards_yellow", "cards_red"]
            for k in m_cols:
                if k not in misc.columns:
                    misc[k] = np.nan
            misc_df = misc[m_cols].copy()

            df = sum_df.merge(pas_df, on=["player_id", "date", "team"], how="left")
            df = df.merge(misc_df, on=["player_id", "date", "team"], how="left")

            df["league"] = lg
            frames.append(df)
        except Exception as e:
            st.warning(f"FBref fetch warning for {lg}: {e}")

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    mask = (all_df["date"] >= pd.to_datetime(start_date)) & (all_df["date"] <= pd.to_datetime(end_date))
    out = all_df.loc[mask].copy()

    # Normalize names
    out["player_norm"] = out["player"].apply(normalize_name)
    return out

@st.cache_data(show_spinner=True)
def build_per_player_avgs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # If minutes missing, fill 0; protect NaNs in sums
    for c in ["goals", "shots", "shots_on_target", "assists", "assists_pass", "cards_yellow", "cards_red"]:
        if c not in df.columns:
            df[c] = 0.0

    # Per-player totals & games
    gcount = df.groupby(["player_norm", "team"], dropna=False)["date"].nunique().rename("g").to_frame()

    sums = df.groupby(["player_norm", "team"], dropna=False).agg(
        goals=("goals","sum"),
        shots=("shots","sum"),
        sot=("shots_on_target","sum"),
        ast=("assists","sum"),
    ).reset_index()

    # sum of squares for SDs
    df["shots2"] = (df["shots"].astype(float).fillna(0.0))**2
    df["sot2"]   = (df["shots_on_target"].astype(float).fillna(0.0))**2
    df["ast2"]   = (df["assists"].astype(float).fillna(0.0))**2
    df["goals2"] = (df["goals"].astype(float).fillna(0.0))**2

    ss = df.groupby(["player_norm", "team"], dropna=False).agg(
        shots2=("shots2","sum"),
        sot2=("sot2","sum"),
        ast2=("ast2","sum"),
        goals2=("goals2","sum")
    ).reset_index()

    agg = sums.merge(gcount.reset_index(), on=["player_norm","team"], how="left") \
              .merge(ss, on=["player_norm","team"], how="left")
    agg["g"] = agg["g"].clip(lower=1)

    # Per-game averages
    agg["mu_goals"] = agg["goals"] / agg["g"]
    agg["mu_shots"] = agg["shots"] / agg["g"]
    agg["mu_sot"]   = agg["sot"]   / agg["g"]
    agg["mu_ast"]   = agg["ast"]   / agg["g"]

    # Sample SDs (Bessel) — minimum floors to avoid absurdly tiny SD
    def sample_sd(sum_x, sum_x2, g):
        g = int(g)
        if g <= 1: return np.nan
        mean = sum_x / g
        var  = (sum_x2 / g) - mean**2
        var  = var * (g / (g - 1))
        return float(np.sqrt(max(var, 1e-6)))

    agg["sd_goals"] = agg.apply(lambda r: sample_sd(r["goals"], r["goals2"], r["g"]), axis=1)
    agg["sd_shots"] = agg.apply(lambda r: sample_sd(r["shots"], r["shots2"], r["g"]), axis=1)
    agg["sd_sot"]   = agg.apply(lambda r: sample_sd(r["sot"],   r["sot2"],   r["g"]), axis=1)
    agg["sd_ast"]   = agg.apply(lambda r: sample_sd(r["ast"],   r["ast2"],   r["g"]), axis=1)

    # Modest inflation for robustness
    inflate = 1.15
    agg["sd_goals"] = np.maximum(0.35, agg["sd_goals"].fillna(0.5)) * inflate
    agg["sd_shots"] = np.maximum(0.50, agg["sd_shots"].fillna(0.8)) * inflate
    agg["sd_sot"]   = np.maximum(0.30, agg["sd_sot"].fillna(0.6)) * inflate
    agg["sd_ast"]   = np.maximum(0.30, agg["sd_ast"].fillna(0.6)) * inflate

    # Team-level lambdas for first/last scorer approximation
    team_sum = agg.groupby("team", dropna=False)["mu_goals"].sum().rename("team_mu_goals").to_frame()
    agg = agg.merge(team_sum, on="team", how="left")
    agg["team_mu_goals"] = agg["team_mu_goals"].replace(0, np.nan)

    # Clean
    out = agg.rename(columns={"player_norm":"Player"})
    return out

# ---------------- Odds API helpers ----------------
VALID_MARKETS = [
    "player_goal_scorer_anytime",
    "player_first_goal_scorer",
    "player_last_goal_scorer",
    "player_shots_on_target",
    "player_shots",
    "player_assists",
]

SOCCER_SPORTS = {
    "English Premier League": "soccer_epl",
    "UEFA Champions League": "soccer_uefa_champs_league",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "MLS": "soccer_usa_mls",
}

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_soccer_events(api_key: str, sport_key: str, lookahead_days: int, region: str):
    base = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    params = {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region}
    return odds_get(base, params)

def fetch_event_props(api_key: str, sport_key: str, event_id: str, region: str, markets: List[str]):
    base = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"}
    return odds_get(base, params)

# ---------------- 1) Date range + league selection ----------------
st.header("1) Date range & league(s) for player averages")
colA, colB = st.columns(2)
with colA:
    start_date = st.date_input("Start date (YYYY/MM/DD)", value=(datetime.utcnow() - timedelta(days=28))).strftime("%Y-%m-%d")
with colB:
    end_date = st.date_input("End date (inclusive)", value=datetime.utcnow()).strftime("%Y-%m-%d")

leagues_pick = st.multiselect(
    "FBref leagues to include (soccerdata keys)",
    options=["ENG-Premier League", "UEFA-Champions League", "ESP-La Liga", "ITA-Serie A",
             "GER-Bundesliga", "FRA-Ligue 1", "USA-MLS"],
    default=["ENG-Premier League", "ESP-La Liga"]
)

# Map UI leagues to soccerdata keys
lg_map = {
    "ENG-Premier League":"ENG-Premier League",
    "UEFA-Champions League":"UEFA-Champions League",
    "ESP-La Liga":"ESP-La Liga",
    "ITA-Serie A":"ITA-Serie A",
    "GER-Bundesliga":"GER-Bundesliga",
    "FRA-Ligue 1":"FRA-Ligue 1",
    "USA-MLS":"USA-MLS",
}
chosen_leagues = [lg_map[l] for l in leagues_pick]

# ---------------- 2) Build per-player projections ----------------
st.header("2) Build per-player averages from soccerdata / FBref")

if not soccerdata_ok:
    st.stop()

if st.button("📥 Build Soccer projections"):
    logs = fetch_fbref_player_game_logs(chosen_leagues, start_date, end_date)
    if logs.empty:
        st.error("No data returned from soccerdata/FBref for this league/date window.")
        st.stop()

    proj = build_per_player_avgs(logs)
    if proj.empty:
        st.error("Could not build per-player averages from the returned data.")
        st.stop()

    # Show raw per-game averages table
    show_cols = ["Player","team","g","mu_goals","mu_shots","mu_sot","mu_ast","sd_goals","sd_shots","sd_sot","sd_ast","team_mu_goals"]
    st.subheader("Per-game averages (from FBref)")
    st.dataframe(proj[show_cols].sort_values("mu_goals", ascending=False).head(50), use_container_width=True)

    st.session_state["soccer_proj"] = proj

# ---------------- 3) Pick a soccer sport + game + markets ----------------
st.header("3) Pick a soccer league & markets from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
sport_label = st.selectbox("Odds API soccer league", list(SOCCER_SPORTS.keys()), index=0)
sport_key = SOCCER_SPORTS[sport_label]
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=2)
markets = st.multiselect("Markets to fetch", VALID_MARKETS, default=VALID_MARKETS)

events = []
if api_key:
    try:
        events = list_soccer_events(api_key, sport_key, lookahead, region)
    except Exception as e:
        st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list upcoming games.")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ---------------- 4) Simulate props ----------------
st.header("4) Fetch props for this event and simulate")

go = st.button("🎲 Fetch lines & simulate (Soccer)")
if go:
    proj = st.session_state.get("soccer_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build Soccer projections first (Step 2)."); st.stop()

    # name set for matching
    proj_names = set(proj["Player"])

    try:
        data = fetch_event_props(api_key, sport_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}"); st.stop()

    # collect bookmaker outcomes
    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                side = o.get("name")
                point = o.get("point")
                if mkey not in VALID_MARKETS or not name or not side:
                    continue
                rows.append({"market": mkey, "player": name, "side": side, "point": (None if point is None else float(point))})

    if not rows:
        st.warning("No player outcomes returned for selected markets.")
        st.stop()

    raw_props = pd.DataFrame(rows)
    props_df = (raw_props.groupby(["market","player","side"], as_index=False)
                        .agg(line=("point","median"), n_books=("point","size")))

    out = []
    for _, r in props_df.iterrows():
        mk, player, line, side = r["market"], r["player"], r["line"], r["side"]
        if player not in proj_names:
            # fuzzy fallback
            match = process.extractOne(player, list(proj_names), scorer=fuzz.token_sort_ratio)
            if match and match[1] >= 87:
                player = match[0]
            else:
                continue

        row = proj.loc[proj["Player"] == player].iloc[0]

        if mk == "player_goal_scorer_anytime":
            lam = float(row["mu_goals"])
            p_yes = poisson_yes(lam)
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            mu, sd = lam, float("nan"); line = None

        elif mk in ("player_first_goal_scorer", "player_last_goal_scorer"):
            # crude share model: player share of team goal rate
            lam_team = float(row["team_mu_goals"]) if not pd.isna(row["team_mu_goals"]) else None
            lam_p = float(row["mu_goals"])
            if not lam_team or lam_team <= 0:
                continue
            share = min(0.95, lam_p / lam_team)
            # Approx probability of being first/last scorer ~ share (rough proxy)
            p_yes = max(0.0, min(1.0, share))
            p = p_yes if side in ("Yes","Over") else (1.0 - p_yes)
            mu, sd = share, float("nan"); line = None

        elif mk == "player_shots":
            mu, sd = float(row["mu_shots"]), float(row["sd_shots"])
            if pd.isna(line) or np.isnan(mu) or np.isnan(sd): continue
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif mk == "player_shots_on_target":
            mu, sd = float(row["mu_sot"]), float(row["sd_sot"])
            if pd.isna(line) or np.isnan(mu) or np.isnan(sd): continue
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        elif mk == "player_assists":
            mu, sd = float(row["mu_ast"]), float(row["sd_ast"])
            if pd.isna(line) or np.isnan(mu) or np.isnan(sd): continue
            p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
            p = p_over if side == "Over" else 1.0 - p_over

        else:
            continue

        out.append({
            "market": mk, "player": player, "side": side,
            "line": (None if line is None else round(float(line), 2)),
            "μ (per-game)": None if pd.isna(mu) else round(float(mu), 3),
            "σ (per-game)": None if (isinstance(sd, float) and np.isnan(sd)) else (None if pd.isna(sd) else round(float(sd), 3)),
            "Win Prob %": round(100*p, 2),
            "books": int(r["n_books"]),
        })

    if not out:
        st.warning("No props matched projections."); st.stop()

    results = (pd.DataFrame(out)
                 .drop_duplicates(subset=["market","player","side"])
                 .sort_values(["market","Win Prob %"], ascending=[True, False])
                 .reset_index(drop=True))

    st.subheader("Results")
    colcfg = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.3f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.3f"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "books": st.column_config.NumberColumn("#Books", width="small"),
    }
    tabs = st.tabs(["All", "Anytime", "First Goal", "Last Goal", "Shots", "Shots on Target", "Assists"])
    with tabs[0]:
        st.dataframe(results, use_container_width=True, hide_index=True, column_config=colcfg)
    with tabs[1]:
        st.dataframe(results[results["market"]=="player_goal_scorer_anytime"], use_container_width=True, hide_index=True, column_config=colcfg)
    with tabs[2]:
        st.dataframe(results[results["market"]=="player_first_goal_scorer"], use_container_width=True, hide_index=True, column_config=colcfg)
    with tabs[3]:
        st.dataframe(results[results["market"]=="player_last_goal_scorer"], use_container_width=True, hide_index=True, column_config=colcfg)
    with tabs[4]:
        st.dataframe(results[results["market"]=="player_shots"], use_container_width=True, hide_index=True, column_config=colcfg)
    with tabs[5]:
        st.dataframe(results[results["market"]=="player_shots_on_target"], use_container_width=True, hide_index=True, column_config=colcfg)
    with tabs[6]:
        st.dataframe(results[results["market"]=="player_assists"], use_container_width=True, hide_index=True, column_config=colcfg)

    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="soccer_props_results.csv",
        mime="text/csv",
    )
