# app.py — NFL + MLB predictor (2025 only) + Stathead CSV fetcher
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
    import statsapi  # MLB-StatsAPI (pip install MLB-StatsAPI)
    HAS_STATSAPI = True
except Exception:
    HAS_STATSAPI = False

# ----------------------------- constants --------------------------------------
SIM_TRIALS = 10_000
HOME_EDGE_NFL = 0.6   # ~0.6 pts to home mean
EPS = 1e-9

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

# -------------------------- SHARED: simple Poisson sim ------------------------
def simulate_poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    """Simulate score distributions with Poisson; break ties slightly to home (for NFL)."""
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)

    wins_home = (h > a).astype(np.float64)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # tiny home tiebreak (NFL flavor)

    p_home = float(wins_home.mean())
    p_away = 1.0 - p_home
    return p_home, p_away, float(h.mean()), float(a.mean()), float((h + a).mean())

# =============================== NFL ==========================================
@st.cache_data(show_spinner=False)
def nfl_team_rates_2025():
    """Build 2025 NFL team PF/PA per game from completed schedule + upcoming matchups."""
    sched = nfl.import_schedules([2025])

    # normalize possible date column names
    date_col = None
    for c in ("gameday", "game_date"):
        if c in sched.columns:
            date_col = c
            break

    # Completed games -> PF/PA
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
        # fallback
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
        # gentle prior toward league avg to stabilize small samples (weeks 1-2)
        league_total = float((long["pf"] + long["pa"]).mean())
        prior = league_total / 2.0
        shrink = np.clip(1.0 - team["games"] / 4.0, 0.0, 1.0)
        rates["PF_pg"] = (1 - shrink) * rates["PF_pg"] + shrink * prior
        rates["PA_pg"] = (1 - shrink) * rates["PA_pg"] + shrink * prior

    # Upcoming (not yet played) matchups
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
    """
    Team-level RS/RA per game for 2025 using Baseball-Reference schedule-and-record pages.
    """
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
    """Map full team names to MLB-StatsAPI team IDs (for probable pitchers)."""
    if not HAS_STATSAPI:
        return {}
    teams = statsapi.get("teams", {"sportIds": 1}).get("teams", [])
    name_to_id = {}
    for t in teams:
        name_to_id[t["name"]] = t["id"]
        if "teamName" in t and "locationName" in t:
            alt = f'{t["locationName"]} {t["teamName"]}'
            name_to_id[alt] = t["id"]
    # Patch common naming differences
    patches = {
        "Tampa Bay Rays": "Tampa Bay Rays",
        "Arizona Diamondbacks": "Arizona Diamondbacks",
        "Chicago White Sox": "Chicago White Sox",
        "St. Louis Cardinals": "St. Louis Cardinals",
        "San Francisco Giants": "San Francisco Giants",
        "San Diego Padres": "San Diego Padres",
        "Los Angeles Angels": "Los Angeles Angels",
        "Los Angeles Dodgers": "Los Angeles Dodgers",
        "Washington Nationals": "Washington Nationals",
        "Cleveland Guardians": "Cleveland Guardians",
        "Texas Rangers": "Texas Rangers",
        "New York Yankees": "New York Yankees",
        "New York Mets": "New York Mets",
    }
    for k, v in patches.items():
        if k not in name_to_id:
            # find id by exact name match if present
            for t in teams:
                if t["name"] == v:
                    name_to_id[k] = t["id"]
                    break
    return name_to_id

def get_probable_pitchers_and_era(home_team: str, away_team: str) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    """
    Try MLB-StatsAPI to get today's probable pitchers and their season ERA.
    Returns (home_name, home_era, away_name, away_era). Any can be None if unavailable.
    """
    if not HAS_STATSAPI:
        return None, None, None, None

    name_to_id = _mlb_team_to_id_map()
    if home_team not in name_to_id or away_team not in name_to_id:
        return None, None, None, None

    # Grab today's schedule and look for this matchup
    try:
        sched = statsapi.schedule()
    except Exception:
        return None, None, None, None

    home_name = away_name = None
    home_era = away_era = None

    for g in sched:
        ht = g.get("home_name")
        at = g.get("away_name")
        if not ht or not at:
            continue
        if ht == home_team and at == away_team:
            home_pitcher = g.get("home_probable_pitcher")
            away_pitcher = g.get("away_probable_pitcher")

            if home_pitcher:
                try:
                    pid = statsapi.lookup_player(home_pitcher)[0]["id"]
                    pstats = statsapi.player_stats(pid, group="pitching", type="season")
                    if pstats:
                        era = pstats[0].get("era")
                        home_era = float(era) if era not in (None, "", "--") else None
                    home_name = home_pitcher
                except Exception:
                    home_name = home_pitcher

            if away_pitcher:
                try:
                    pid = statsapi.lookup_player(away_pitcher)[0]["id"]
                    pstats = statsapi.player_stats(pid, group="pitching", type="season")
                    if pstats:
                        era = pstats[0].get("era")
                        away_era = float(era) if era not in (None, "", "--") else None
                    away_name = away_pitcher
                except Exception:
                    away_name = away_pitcher
            break

    return home_name, home_era, away_name, away_era

def mlb_matchup_mu(rates: pd.DataFrame, home: str, away: str,
                   h_pit_era: Optional[float], a_pit_era: Optional[float]) -> Tuple[float, float]:
    """
    Base run means from team RS/RA, then nudge by starters' ERA if available.
    A simple adjustment: subtract 0.05 * (LgERA - PitcherERA) from opponent's mean (per 9 IP).
    """
    rH = rates.loc[rates["team"].str.lower() == home.lower()]
    rA = rates.loc[rates["team"].str.lower() == away.lower()]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown MLB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]

    mu_home = max(EPS, (H["RS_pg"] + A["RA_pg"]) / 2.0)
    mu_away = max(EPS, (A["RS_pg"] + H["RA_pg"]) / 2.0)

    # league average ERA proxy (roughly ~4.20–4.40 typical). Use sample from table:
    LgERA = 4.30
    # If starters’ ERA available, apply small opponent-run adjustment
    # Better (lower) ERA → reduce opponent mean slightly; worse → increase.
    scale = 0.05
    if h_pit_era is not None:
        mu_away = max(EPS, mu_away - scale * (LgERA - h_pit_era))
    if a_pit_era is not None:
        mu_home = max(EPS, mu_home - scale * (LgERA - a_pit_era))

    return mu_home, mu_away

# ============================ Stathead → CSV ==================================
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    }
    if cookie:
        headers["Cookie"] = cookie
    resp = pd.read_html(url) if not cookie else None
    # If cookie is needed, use requests + read_html
    if cookie:
        import requests
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        df = biggest_table_from_html(r.text)
    else:
        if not resp:
            # fallback via requests anyway
            import requests
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            df = biggest_table_from_html(r.text)
        else:
            df = max(resp, key=lambda t: t.shape[0] * t.shape[1])

    df = clean_headers(df)
    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str).str.fullmatch(r"\d+")]
    df = df.dropna(how="all")
    df = coerce_numeric(df).reset_index(drop=True)
    return df

# ================================ UI ==========================================
st.set_page_config(page_title="NFL + MLB Predictor — 2025 (stats only)", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025 (stats only)")
st.caption(
    "Win probabilities use **team scoring rates only** (NFL: PF/PA; MLB: RS/RA), "
    "with optional **probable pitchers + ERA** nudging MLB matchups. No injuries, travel, or betting data."
)

page = st.sidebar.radio("Choose a page", ["NFL (2025)", "MLB (2025)", "📥 Stathead → CSV"], index=0)

# -------------------------- NFL page ------------------------------------------
if page == "NFL (2025)":
    st.subheader("🏈 NFL — pick an upcoming matchup")

    try:
        nfl_rates, upcoming = nfl_team_rates_2025()
    except Exception as e:
        st.error(f"Couldn't build NFL team rates: {e}")
        st.stop()

    if upcoming.empty:
        st.info("No upcoming games found in the 2025 schedule yet.")
        st.stop()

    # Show only matchups (no team table), as requested
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
        with col1:
            st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with col2:
            st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        with col3:
            st.metric(label="Expected total", value=f"{exp_t:.1f}")

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

    # Probable pitchers + ERA (from MLB-StatsAPI if installed)
    h_name = h_era = a_name = a_era = None
    if HAS_STATSAPI:
        h_name, h_era, a_name, a_era = get_probable_pitchers_and_era(home, away)

    try:
        mu_h, mu_a = mlb_matchup_mu(mlb_rates, home, away, h_era, a_era)
        p_home, p_away, exp_h, exp_a, exp_t = simulate_poisson_game(mu_h, mu_a, SIM_TRIALS)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=f"{home} win %", value=f"{p_home*100:.1f}%")
        with col2:
            st.metric(label=f"{away} win %", value=f"{p_away*100:.1f}%")
        with col3:
            st.metric(label="Expected total", value=f"{exp_t:.1f}")

        # Show pitcher info if available
        if h_name or a_name:
            st.caption(
                f"Probable starters: "
                f"{home}: **{h_name or 'TBD'}**{(' (ERA ' + str(h_era) + ')' ) if h_era is not None else ''}  •  "
                f"{away}: **{a_name or 'TBD'}**{(' (ERA ' + str(a_era) + ')' ) if a_era is not None else ''}"
            )
        else:
            st.caption("No probable starters found—using team rates only.")

        st.caption(f"Expected score: **{home} {exp_h:.1f} — {away} {exp_a:.1f}**")
    except Exception as e:
        st.error(str(e))

# -------------------------- Stathead → CSV page -------------------------------
else:
    st.subheader("📥 Stathead → CSV (QB passing 2025)")

    st.write("This pulls **both** of your tiny URLs and gives you clean CSVs. "
             "If a URL needs login, paste your browser cookie below.")

    colA, colB = st.columns(2)
    with colA:
        url1 = st.text_input("Tiny URL 1", value="https://stathead.com/tiny/t1A4t")
    with colB:
        url2 = st.text_input("Tiny URL 2", value="https://stathead.com/tiny/0gq7J")

    cookie = st.text_input("Optional cookie (only if page needs login)", type="password", value="")

    def do_fetch(label: str, url: str):
        try:
            df = fetch_stathead_table(url, cookie=cookie.strip())
            st.success(f"{label}: {len(df)} rows × {len(df.columns)} cols")
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                f"⬇️ Download {label} CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{label.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"{label}: {e}")

    if st.button("Fetch both"):
        do_fetch("QB Passing A (t1A4t)", url1)
        do_fetch("QB Passing B (0gq7J)", url2)

    st.markdown(
        """
        **Tip:** If Stathead blocks the fetch, open the tiny link in your browser while logged in,
        copy the request **Cookie** value from DevTools → Network, and paste it above.
        """
    )
