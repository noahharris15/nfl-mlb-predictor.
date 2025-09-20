# app.py — NFL + MLB predictor (2025 only) + NFL QB Player Props (Stathead)
# Run: streamlit run app.py

import io
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple

# ---- NFL (nfl_data_py) -------------------------------------------------------
import nfl_data_py as nfl

# ---- MLB (pybaseball + optional MLB-StatsAPI) --------------------------------
from pybaseball import schedule_and_record
try:
    import statsapi  # pip install MLB-StatsAPI
    HAS_STATSAPI = True
except Exception:
    HAS_STATSAPI = False

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10_000
HOME_EDGE_NFL = 0.6   # ~0.6 pts to home mean
EPS = 1e-9

# Your Stathead tiny URLs (QB passing 2025)
STATHEAD_QB_URLS = [
    "https://stathead.com/tiny/t1A4t",
    "https://stathead.com/tiny/0gq7J",
]

# Hard-coded BR team IDs (stable)
MLB_TEAMS_2025 = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
}

# -------------------------- SHARED: simple Poisson/Normal sims ----------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # tiny home tiebreak (NFL vibe)
    p_home = float(wins_home.mean())
    p_away = 1.0 - p_home
    return p_home, p_away, float(h.mean()), float(a.mean()), float((h + a).mean())

def prob_over_normal(mean: float, sd: float, line: float, trials: int = SIM_TRIALS):
    sd = max(1e-6, float(sd))
    sims = np.random.normal(loc=mean, scale=sd, size=trials)
    sims = np.clip(sims, 0, None)
    p_over = float((sims > line).mean())
    return p_over

def prob_over_poisson(lmbda: float, line: float, trials: int = SIM_TRIALS):
    lmbda = max(1e-6, float(lmbda))
    sims = np.random.poisson(lmbda, size=trials)
    p_over = float((sims > line).mean())
    return p_over

# =============================== NFL ==========================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    sched = nfl.import_schedules([2025])

    date_col = None
    for c in ("gameday", "game_date"):
        if c in sched.columns:
            date_col = c
            break

    played = sched.dropna(subset=["home_score", "away_score"])
    home = played.rename(columns={
        "home_team": "team", "away_team": "opp",
        "home_score": "pf", "away_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    away = played.rename(columns={
        "away_team": "team", "home_team": "opp",
        "away_score": "pf", "home_score": "pa"
    })[["team", "opp", "pf", "pa"]]
    long = pd.concat([home, away], ignore_index=True)

    if long.empty:
        per = 45.0 / 2.0
        teams32 = [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
            "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
            "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
            "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
            "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
            "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
            "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
        ]
        rates = pd.DataFrame({"team": teams32, "PF_pg": per, "PA_pg": per})
    else:
        team = long.groupby("team", as_index=False).agg(
            games=("pf", "size"), PF=("pf", "sum"), PA=("pa", "sum")
        )
        rates = pd.DataFrame({
            "team": team["team"],
            "PF_pg": team["PF"] / team["games"],
            "PA_pg": team["PA"] / team["games"],
        })
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"] / 4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink) * rates["PF_pg"] + shrink * prior
        rates["PA_pg"] = (1 - shrink) * rates["PA_pg"] + shrink * prior

    upcoming = sched[sched["home_score"].isna() & sched["away_score"].isna()][
        ["home_team", "away_team"] + ([date_col] if date_col else [])
    ].copy()
    if date_col:
        upcoming = upcoming.rename(columns={date_col: "date"})
    else:
        upcoming["date"] = ""
    for col in ["home_team", "away_team"]:
        upcoming[col] = upcoming[col].astype(str).str.replace(r"\s+", " ", regex=True)
    return rates, upcoming

def nfl_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> Tuple[float, float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["PF_pg"] + A["PA_pg"]) / 2.0 + HOME_EDGE_NFL)
    mu_away = max(EPS, (A["PF_pg"] + H["PA_pg"]) / 2.0)
    return mu_home, mu_away

# =============================== MLB ==========================================
@st.cache_data(show_spinner=False)
def mlb_team_rates_2025():
    rows = []
    for br, name in MLB_TEAMS_2025.items():
        try:
            sar = schedule_and_record(2025, br)
            sar = sar[pd.to_numeric(sar.get("R"), errors="coerce").notna()]
            sar = sar[pd.to_numeric(sar.get("RA"), errors="coerce").notna()]
            if sar.empty:
                RS_pg = RA_pg = 4.5
            else:
                sar["R"] = sar["R"].astype(float)
                sar["RA"] = sar["RA"].astype(float)
                g = max(1, int(len(sar)))
                RS_pg = float(sar["R"].sum() / g)
                RA_pg = float(sar["RA"].sum() / g)
            rows.append({"team": name, "RS_pg": RS_pg, "RA_pg": RA_pg})
        except Exception:
            rows.append({"team": name, "RS_pg": 4.5, "RA_pg": 4.5})
    df = pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
    if not df.empty:
        league_rs = float(df["RS_pg"].mean())
        league_ra = float(df["RA_pg"].mean())
        df["RS_pg"] = 0.9 * df["RS_pg"] + 0.1 * league_rs
        df["RA_pg"] = 0.9 * df["RA_pg"] + 0.1 * league_ra
    return df

def _mlb_team_to_id_map() -> dict:
    if not HAS_STATSAPI:
        return {}
    teams = statsapi.get("teams", {"sportIds": 1}).get("teams", [])
    name_to_id = {}
    for t in teams:
        name_to_id[t["name"]] = t["id"]
        if "teamName" in t and "locationName" in t:
            alt = f'{t["locationName"]} {t["teamName"]}'
            name_to_id[alt] = t["id"]
    return name_to_id

def get_probable_pitchers_and_era(home_team: str, away_team: str):
    if not HAS_STATSAPI:
        return None, None, None, None
    try:
        sched = statsapi.schedule()
    except Exception:
        return None, None, None, None
    h_name = h_era = a_name = a_era = None
    for g in sched:
        if g.get("home_name") == home_team and g.get("away_name") == away_team:
            hp = g.get("home_probable_pitcher"); ap = g.get("away_probable_pitcher")
            if hp:
                try:
                    pid = statsapi.lookup_player(hp)[0]["id"]
                    ps = statsapi.player_stats(pid, group="pitching", type="season")
                    h_era = float(ps[0]["era"]) if ps and ps[0].get("era") not in (None, "", "--") else None
                except Exception:
                    pass
                h_name = hp
            if ap:
                try:
                    pid = statsapi.lookup_player(ap)[0]["id"]
                    ps = statsapi.player_stats(pid, group="pitching", type="season")
                    a_era = float(ps[0]["era"]) if ps and ps[0].get("era") not in (None, "", "--") else None
                except Exception:
                    pass
                a_name = ap
            break
    return h_name, h_era, a_name, a_era

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str,
                   h_pit_era: Optional[float], a_pit_era: Optional[float]) -> Tuple[float, float]:
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)
    LgERA = 4.30
    scale = 0.05
    if h_pit_era is not None:
        mu_away = max(EPS, mu_away - scale * (LgERA - h_pit_era))
    if a_pit_era is not None:
        mu_home = max(EPS, mu_home - scale * (LgERA - a_pit_era))
    return mu_home, mu_away

# ============================ Stathead helpers ================================
def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in df.columns.values
        ]
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.fullmatch(r"")]
    return df

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "O":
            s = df[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            df[c] = pd.to_numeric(s, errors="ignore")
    return df

def biggest_table_from_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError("No tables found.")
    return max(tables, key=lambda t: t.shape[0] * t.shape[1])

def fetch_stathead_table(url: str, cookie: str = "") -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    if cookie:
        import requests
        r = requests.get(url, headers={"User-Agent": headers["User-Agent"], "Cookie": cookie}, timeout=30)
        r.raise_for_status()
        df = biggest_table_from_html(r.text)
    else:
        # try vanilla first
        tables = pd.read_html(url)
        df = max(tables, key=lambda t: t.shape[0] * t.shape[1])
    df = clean_headers(df)
    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str).str.fullmatch(r"\d+")]
    df = df.dropna(how="all")
    df = coerce_numeric(df).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_qb_stathead_2025(urls: list[str], cookie: str = "") -> pd.DataFrame:
    # Merge all provided tables on "Player" (outer) and keep key columns
    dfs = []
    for u in urls:
        try:
            dfs.append(fetch_stathead_table(u, cookie=cookie))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="Player", how="outer", suffixes=("", "_dup"))
        # Drop any exact dup columns
        dup_cols = [c for c in df.columns if c.endswith("_dup")]
        df = df.drop(columns=dup_cols, errors="ignore")

    # The tables you shared include these columns (based on your paste):
    # Player, Team (maybe), G, Cmp, Att, Yds, TD, Int
    for col in ["G","Cmp","Att","Yds","TD","Int"]:
        if col not in df.columns:
            df[col] = np.nan

    # Per-game rates (avoid div by zero)
    g = df["G"].replace(0, np.nan)
    df["Yds_pg"] = df["Yds"] / g
    df["Cmp_pg"] = df["Cmp"] / g
    df["Att_pg"] = df["Att"] / g
    df["TD_pg"]  = df["TD"]  / g
    df["Int_pg"] = df["Int"] / g

    # Fill tiny gaps with zeros for cleaner UI (only on per-game fields)
    for col in ["Yds_pg","Cmp_pg","Att_pg","TD_pg","Int_pg"]:
        df[col] = df[col].fillna(0.0)

    # Keep tidy columns
    keep = ["Player","Team","G","Cmp","Att","Yds","TD","Int","Yds_pg","Cmp_pg","Att_pg","TD_pg","Int_pg"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values("Yds_pg", ascending=False, na_position="last").reset_index(drop=True)
    return df

# ================================ UI ==========================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Matchup win % from **team scoring rates** (NFL: PF/PA; MLB: RS/RA), "
    "MLB optionally nudged by **probable starters + ERA**. "
    "Player props (QB) use quick sims from Stathead 2025 per-game rates."
)

page = st.sidebar.radio(
    "Choose a page",
    ["NFL (2025)", "MLB (2025)", "NFL Player Props (QB, 2025 Stathead)"],
    index=0
)

# -------------------------- NFL page ------------------------------------------
if page == "NFL (2025)":
    st.subheader("🏈 NFL — pick an upcoming matchup")
    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build NFL team rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming 2025 games yet.")
        st.stop()

    show_dates = "date" in upcoming.columns and upcoming["date"].astype(str).str.len().gt(0).any()
    if show_dates:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"] +
                   " — " + upcoming["date"].astype(str)).tolist()
    else:
        choices = (upcoming["home_team"] + " vs " + upcoming["away_team"]).tolist()

    pick = st.selectbox("Matchup", choices, index=0)
    pair = pick.split(" — ")[0]
    home, away = pair.split(" vs ")

    try:
        mu_h, mu_a = nfl_matchup_mu(nfl_rates, home, away)
        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with col2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        with col3: st.metric(label="Expected total", value=f"{exp_t:.1f}")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
    except Exception as e:
        st.error(str(e))

# -------------------------- MLB page ------------------------------------------
elif page == "MLB (2025)":
    st.subheader("⚾ MLB — pick any matchup")
    try:
        mlb_rates = mlb_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't load MLB team rates: {e}")
        st.stop()

    teams = mlb_rates["team"].sort_values().tolist()
    if not teams:
        st.info("No MLB team data yet.")
        st.stop()

    home = st.selectbox("Home team", teams, index=0, key="mlb_home")
    away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="mlb_away")

    h_name = h_era = a_name = a_era = None
    if HAS_STATSAPI:
        h_name, h_era, a_name, a_era = get_probable_pitchers_and_era(home, away)

    try:
        mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away, h_era, a_era)
        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with col2: st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        with col3: st.metric(label="Expected total", value=f"{exp_t:.1f}")
        if h_name or a_name:
            st.caption(
                f"Probable starters — {home}: **{h_name or 'TBD'}**"
                f"{(' (ERA ' + str(h_era) + ')') if h_era is not None else ''} • "
                f"{away}: **{a_name or 'TBD'}**"
                f"{(' (ERA ' + str(a_era) + ')') if a_era is not None else ''}"
            )
        else:
            st.caption("No probable starters found — using team rates only.")
        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
    except Exception as e:
        st.error(str(e))

# --------------------- NFL Player Props (QB, Stathead 2025) -------------------
else:
    st.subheader("🎯 NFL Player Props (QB) — 2025 Stathead quick sim")

    with st.expander("Stathead access (optional)"):
        st.write("If your tiny links require login, paste your browser **Cookie** below (from DevTools → Network).")
        cookie = st.text_input("Cookie (optional)", type="password", value="")

    # Load QB table(s)
    qbs = load_qb_stathead_2025(STATHEAD_QB_URLS, cookie=cookie)
    if qbs.empty:
        st.error("Couldn't load QB tables from Stathead. Open the tiny links in your browser to verify access.")
        st.stop()

    st.markdown("**2025 QB per-game rates (from your Stathead tables)**")
    st.dataframe(
        qbs[["Player","Team","G","Yds_pg","TD_pg","Int_pg","Cmp_pg","Att_pg","Yds","TD","Int","Cmp","Att"]]
        .rename(columns={"Yds_pg":"Yds/G","TD_pg":"TD/G","Int_pg":"INT/G","Cmp_pg":"Cmp/G","Att_pg":"Att/G"})
        .round(2),
        use_container_width=True, height=380
    )

    # Pick player + market
    left, right = st.columns(2)
    with left:
        player = st.selectbox("Player", qbs["Player"].tolist(), index=0)
        market = st.selectbox("Market", ["Pass Yds", "Pass TD", "Interceptions", "Completions"], index=0)
    with right:
        line = st.number_input("Prop line", min_value=0.0, value=250.0 if market=="Pass Yds" else 1.5, step=0.5)

        # Uncertainty knobs (so you can tune tight/loose sims)
        if market in ("Pass Yds","Completions"):
            default_sd = 65.0 if market=="Pass Yds" else 5.0
            sd = st.slider("Simulation SD (controls volatility)", min_value=5.0, max_value=120.0,
                           value=default_sd, step=1.0)
        else:
            sd = None  # not used for Poisson markets

    row = qbs[qbs["Player"] == player].iloc[0]
    G = max(1.0, float(row.get("G", 1.0)))

    # Build per-game rate from table
    if market == "Pass Yds":
        mu = float(row.get("Yds_pg", np.nan)) if pd.notna(row.get("Yds_pg")) else float(row.get("Yds", 0.0))/G
        p_over = prob_over_normal(mu, sd or 65.0, line, SIM_TRIALS)
    elif market == "Completions":
        mu = float(row.get("Cmp_pg", np.nan)) if pd.notna(row.get("Cmp_pg")) else float(row.get("Cmp", 0.0))/G
        p_over = prob_over_normal(mu, sd or 5.0, line, SIM_TRIALS)
    elif market == "Pass TD":
        lam = float(row.get("TD_pg", np.nan)) if pd.notna(row.get("TD_pg")) else float(row.get("TD", 0.0))/G
        p_over = prob_over_poisson(lam, line, SIM_TRIALS)
    else:  # Interceptions
        lam = float(row.get("Int_pg", np.nan)) if pd.notna(row.get("Int_pg")) else float(row.get("Int", 0.0))/G
        p_over = prob_over_poisson(lam, line, SIM_TRIALS)

    p_under = 1.0 - p_over
    def fair_ml(p):
        if p <= 0: return "∞"
        if p >= 1: return "-∞"
        dec = 1.0/p
        if dec >= 2:  # ≥ +100
            return f"+{int(round((dec-1)*100))}"
        else:
            return f"{int(round(-100/(dec-1)))}"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Over %", f"{p_over*100:.1f}%")
    with col2:
        st.metric("Under %", f"{p_under*100:.1f}%")
    with col3:
        st.metric("Fair ML (Over)", fair_ml(p_over))

    st.caption(
        "Method: simple quick-sim.\n"
        "• **Yards / Completions** → Normal(mean=per-game rate, sd slider), truncated at 0.\n"
        "• **TD / INT** → Poisson(lambda = per-game rate).\n"
        "This is intentionally lightweight; you can tighten/loosen variance with the SD slider."
    )
