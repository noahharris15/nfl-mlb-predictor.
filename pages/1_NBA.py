# NBA Player Props — Odds API + NBA Stats (nba_api), per-game averages + 10k sims
# Place this file at: pages/1_NBA.py
# IMPORTANT: Do NOT call st.set_page_config here (it's in app.py / NFL page).

import re, unicodedata, datetime as dt
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- NBA Stats ----------
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

st.title("🏀 NBA Player Props — Odds API + NBA Stats (live)")

SIM_TRIALS = 10_000

# -------------------- VALID MARKETS (simulated) --------------------
VALID_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_blocks_steals",
    "player_turnovers",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_rebounds_assists",
    "player_field_goals",
    "player_frees_made",
    "player_frees_attempts",
]

# We will fetch these if selected, but hide from output until modeled:
UNSUPPORTED_MARKETS_HIDE = {
    "player_first_basket",
    "player_first_team_basket",
    "player_double_double",
    "player_triple_double",
    "player_points_q1",
    "player_rebounds_q1",
    "player_assists_q1",
}

ODDS_SPORT = "basketball_nba"  # Odds API sport key

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

def t_over_prob(mu: float, sd: float, line: float, trials: int = SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)
    return float((draws > line).mean())

def sample_sd(sum_x: float, sum_x2: float, g: int, floor: float = 0.0) -> float:
    g = int(g)
    if g <= 1:
        return float("nan")
    mean = sum_x / g
    var = (sum_x2 / g) - (mean ** 2)
    var = var * (g / (g - 1))
    return float(max(np.sqrt(max(var, 1e-9)), floor))

# ------------------ NBA Stats helpers ------------------
@st.cache_data(show_spinner=False)
def _players_index() -> pd.DataFrame:
    # index of all players: id, full_name, first, last
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    # normalize a few keys for lookup
    df["name_norm"] = df["full_name"].apply(normalize_name)
    df["first_norm"] = df["first_name"].apply(normalize_name)
    df["last_norm"]  = df["last_name"].apply(normalize_name)
    return df[["id","full_name","name_norm","first_norm","last_norm"]]

def find_player_id_by_name(name: str) -> Optional[int]:
    df = _players_index()
    n = normalize_name(name)
    # exact match
    hit = df.loc[df["name_norm"] == n]
    if not hit.empty:
        return int(hit.iloc[0]["id"])
    # fallback: last-name contains + first initial
    parts = n.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        hit = df.loc[(df["last_norm"] == last) & (df["first_norm"].str.startswith(first[:1]))]
        if not hit.empty:
            return int(hit.iloc[0]["id"])
    # final fallback: contains
    hit = df.loc[df["name_norm"].str.contains(last if len(parts) else n)]
    if not hit.empty:
        return int(hit.iloc[0]["id"])
    return None

def fetch_player_gamelog_df(player_id: int, season: str, season_type: str) -> pd.DataFrame:
    # Live fetch every time (no cache), per user request
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star=season_type)
    df = gl.get_data_frames()[0]
    # Normalize columns we need
    # Columns include: GAME_DATE, PTS, REB, AST, STL, BLK, TOV, FG3M, FGM, FGA, FTM, FTA
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

def agg_stats_in_range(df: pd.DataFrame, start: dt.date, end: dt.date) -> Dict[str, float]:
    # Filter by date range inclusive
    if "GAME_DATE" in df.columns:
        mask = (df["GAME_DATE"] >= pd.Timestamp(start)) & (df["GAME_DATE"] <= pd.Timestamp(end))
        df = df.loc[mask].copy()
    g = int(df.shape[0])
    if g == 0:
        return {"g": 0}

    def s(col: str) -> Tuple[float, float]:
        if col not in df.columns: 
            return 0.0, 0.0
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        return float(x.sum()), float((x ** 2).sum())

    sums = {}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        sums[col], sums["sq_"+col] = s(col)

    out = {"g": g}
    for col in ["PTS","REB","AST","STL","BLK","TOV","FG3M","FGM","FTM","FTA"]:
        out["mu_"+col] = sums[col] / g
        out["sd_"+col] = sample_sd(sums[col], sums["sq_"+col], g, floor=0.0)
    return out

# ------------------ UI: date scope ------------------
st.markdown("### 1) Pick a season & date range")
season_year = st.number_input("Season start year (e.g., 2024 for 2024-25)", 2015, 2100, 2025, step=1)
season_str = f"{season_year}-{str(season_year+1)[-2:]}"
season_type = st.selectbox("Season Type", ["Regular Season", "Playoffs"], index=0)

# flexible dates inside the season
default_start = dt.date(season_year, 10, 1)
default_end   = dt.date(season_year+1, 6, 30)
start_date, end_date = st.date_input("Dates (inclusive)", (default_start, default_end), format="YYYY-MM-DD")

# ------------------ Build projections (live from NBA Stats) ------------------
st.markdown("### 2) Build per-player projections from NBA Stats (live)")
st.caption("We’ll only build projections **for players who appear in the selected game’s props** (fast). Choose the game below first, then click build.")

# ------------------ Odds API ------------------
st.markdown("### 3) Pick a game & markets (Odds API)")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key (kept local to your session)", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets_pickable = VALID_MARKETS + sorted(list(UNSUPPORTED_MARKETS_HIDE))
markets = st.multiselect("Markets to fetch", markets_pickable, default=VALID_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()

def list_events(api_key: str, lookahead_days: int, region: str):
    base = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events"
    return odds_get(base, {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region})

def fetch_event_props(api_key: str, event_id: str, region: str, markets: List[str]):
    base = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/events/{event_id}/odds"
    return odds_get(base, {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"})

events = []
if api_key:
    try:
        events = list_events(api_key, lookahead, region)
    except Exception as e:
        st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key and pick a lookahead to list upcoming games.")
    st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]
event_id = event["id"]

# ------------------ Build projections for players in this event ------------------
if st.button("📥 Build NBA projections (from NBA Stats)"):
    try:
        # Pull props once to know which players to build
        data_preview = fetch_event_props(api_key, event_id, region, list(set(markets)))
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    # Collect unique player names appearing in outcomes
    player_names = set()
    for bk in data_preview.get("bookmakers", []):
        for m in bk.get("markets", []):
            for o in m.get("outcomes", []):
                nm = normalize_name(o.get("description"))
                if nm:
                    player_names.add(nm)

    if not player_names:
        st.warning("No player names found in props; try different markets.")
        st.stop()

    # Build projections by fetching game logs for each player
    rows = []
    missing_map = {}
    st.write(f"Found **{len(player_names)}** players in props. Pulling NBA logs live…")
    for pn in sorted(player_names):
        pid = find_player_id_by_name(pn)
        if not pid:
            missing_map[pn] = "player_id_not_found"
            continue
        try:
            gldf = fetch_player_gamelog_df(pid, season_str, season_type)
        except Exception as e:
            missing_map[pn] = f"log_error: {e}"
            continue

        ag = agg_stats_in_range(gldf, start_date, end_date)
        if ag["g"] == 0:
            missing_map[pn] = "no_games_in_range"
            continue

        rows.append({
            "Player": pn,
            "g": ag["g"],
            # means (raw)
            "mu_pts": ag["mu_PTS"], "mu_reb": ag["mu_REB"], "mu_ast": ag["mu_AST"],
            "mu_tpm": ag["mu_FG3M"], "mu_blk": ag["mu_BLK"], "mu_stl": ag["mu_STL"],
            "mu_turn": ag["mu_TOV"], "mu_fgm": ag["mu_FGM"], "mu_ftm": ag["mu_FTM"], "mu_fta": ag["mu_FTA"],
            # sds
            "sd_pts": ag["sd_PTS"], "sd_reb": ag["sd_REB"], "sd_ast": ag["sd_AST"],
            "sd_tpm": ag["sd_FG3M"], "sd_blk": ag["sd_BLK"], "sd_stl": ag["sd_STL"],
            "sd_turn": ag["sd_TOV"], "sd_fgm": ag["sd_FGM"], "sd_ftm": ag["sd_FTM"], "sd_fta": ag["sd_FTA"],
        })

    if missing_map:
        with st.expander("⚠️ Players not built (ID not found / no games):"):
            st.json(missing_map)

    if not rows:
        st.error("No projections built. Try a different game/date range.")
        st.stop()

    st.session_state["nba_proj"] = pd.DataFrame(rows)
    st.success("Built projections from NBA Stats.")
    st.dataframe(st.session_state["nba_proj"].head(25), use_container_width=True)

# ------------------ Simulate ------------------
st.markdown("### 4) Fetch lines & simulate")
go = st.button("Fetch lines & simulate (NBA)")

if go:
    proj = st.session_state.get("nba_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build NBA projections first (Step 2)."); st.stop()

    try:
        data = fetch_event_props(api_key, event_id, region, list(set(markets)))
    except Exception as e:
        st.error(f"Props fetch failed: {e}")
        st.stop()

    # Build quick index for lookup by normalized name
    proj = proj.copy()
    proj["player_norm"] = proj["Player"].apply(normalize_name)
    idx = proj.set_index("player_norm")

    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = normalize_name(o.get("description"))
                side = o.get("name")
                point = o.get("point")
                if not name or mkey in UNSUPPORTED_MARKETS_HIDE:
                    continue
                # only keep simulated markets
                if mkey not in VALID_MARKETS:
                    continue
                rows.append({
                    "market": mkey,
                    "player_norm": name,
                    "side": side,
                    "point": (None if point is None else float(point)),
                })

    if not rows:
        st.warning("No supported player outcomes returned for selected markets.")
        st.stop()

    props_df = (pd.DataFrame(rows)
                .groupby(["market","player_norm","side"], as_index=False)
                .agg(line=("point","median"), n_books=("point","size")))

    out_rows = []
    for _, r in props_df.iterrows():
        mkt, name, side, line = r["market"], r["player_norm"], r["side"], r["line"]
        if pd.isna(line) or name not in idx.index:
            continue
        row = idx.loc[name]

        # Defaults
        mu = sd = raw = None

        # Singles
        if mkt == "player_points":
            raw, mu, sd = row["mu_pts"], row["mu_pts"], row["sd_pts"]
        elif mkt == "player_rebounds":
            raw, mu, sd = row["mu_reb"], row["mu_reb"], row["sd_reb"]
        elif mkt == "player_assists":
            raw, mu, sd = row["mu_ast"], row["mu_ast"], row["sd_ast"]
        elif mkt == "player_threes":
            raw, mu, sd = row["mu_tpm"], row["mu_tpm"], row["sd_tpm"]
        elif mkt == "player_blocks":
            raw, mu, sd = row["mu_blk"], row["mu_blk"], row["sd_blk"]
        elif mkt == "player_steals":
            raw, mu, sd = row["mu_stl"], row["mu_stl"], row["sd_stl"]
        elif mkt == "player_turnovers":
            raw, mu, sd = row["mu_turn"], row["mu_turn"], row["sd_turn"]
        elif mkt == "player_field_goals":
            raw, mu, sd = row["mu_fgm"], row["mu_fgm"], row["sd_fgm"]
        elif mkt == "player_frees_made":
            raw, mu, sd = row["mu_ftm"], row["mu_ftm"], row["sd_ftm"]
        elif mkt == "player_frees_attempts":
            raw, mu, sd = row["mu_fta"], row["mu_fta"], row["sd_fta"]

        # Combos (variance ≈ sum of variances, assuming weak correlation)
        elif mkt == "player_points_rebounds_assists":
            raw = row["mu_pts"] + row["mu_reb"] + row["mu_ast"]
            mu  = raw
            sd  = np.sqrt(row["sd_pts"]**2 + row["sd_reb"]**2 + row["sd_ast"]**2)
        elif mkt == "player_points_rebounds":
            raw = row["mu_pts"] + row["mu_reb"]
            mu  = raw
            sd  = np.sqrt(row["sd_pts"]**2 + row["sd_reb"]**2)
        elif mkt == "player_points_assists":
            raw = row["mu_pts"] + row["mu_ast"]
            mu  = raw
            sd  = np.sqrt(row["sd_pts"]**2 + row["sd_ast"]**2)
        elif mkt == "player_rebounds_assists":
            raw = row["mu_reb"] + row["mu_ast"]
            mu  = raw
            sd  = np.sqrt(row["sd_reb"]**2 + row["sd_ast"]**2)
        elif mkt == "player_blocks_steals":
            raw = row["mu_blk"] + row["mu_stl"]
            mu  = raw
            sd  = np.sqrt(row["sd_blk"]**2 + row["sd_stl"]**2)
        else:
            continue  # not supported

        if mu is None or sd is None or pd.isna(mu) or pd.isna(sd):
            continue

        p_over = t_over_prob(float(mu), float(sd), float(line), SIM_TRIALS)
        p = p_over if side == "Over" else 1.0 - p_over

        out_rows.append({
            "market": mkt,
            "player": row["Player"],
            "side": side,
            "line": round(float(line), 2),
            "Avg (raw)": round(float(raw), 2),
            "μ (scaled)": round(float(mu), 2),          # currently same as raw (no defense scaling)
            "σ (per-game)": round(float(sd), 2),
            "Win Prob %": round(100 * p, 2),
            "books": int(r["n_books"]),
        })

    if not out_rows:
        st.warning("No props matched projections."); st.stop()

    results = (pd.DataFrame(out_rows)
               .drop_duplicates(subset=["market","player","side"])
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    # Display (Market-first order, as requested)
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="nba_props_sim_results.csv",
        mime="text/csv",
    )
