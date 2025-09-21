# app.py — NFL & MLB Poisson sims + Player Props with embedded 2025 NFL defense table

import math
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple

# ---------------- UI setup ----------------
st.set_page_config(page_title="NFL + MLB Predictor — 2025", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025")
st.caption("Team win models use simple Poisson scoring. Player Props read your pasted CSV rows. "
           "Defense strength is baked into the script (EPA/Play, Pass, Rush).")

# --------------- Small helpers ---------------
EPS = 1e-9
SIM_TRIALS = 10000
HOME_EDGE_NFL = 0.6  # ~0.6 pts

def poisson_game(mu_home: float, mu_away: float, trials: int = SIM_TRIALS):
    mu_home = max(0.1, float(mu_home))
    mu_away = max(0.1, float(mu_away))
    h = np.random.poisson(mu_home, size=trials)
    a = np.random.poisson(mu_away, size=trials)
    wins_home = (h > a).astype(float)
    ties = (h == a)
    if ties.any():
        wins_home[ties] = 0.53  # tiny tiebreak
    return float(wins_home.mean()), float(h.mean()), float(a.mean())

# =============================================================================
# NFL page (unchanged in spirit; you can fill team PF/PA manually if needed)
# =============================================================================

st.subheader("Pages")
page = st.tabs(["🏈 NFL", "⚾ MLB", "📈 Player Props"])

# =========================================================================================
# Embedded NFL defense table (2025): EPA/Play, EPA/Pass, EPA/Rush
# Source: the list you provided (ranked 1–32). Negative = stronger defense.
# We'll normalize to a multiplicative factor around 1.00 for each category.
# =========================================================================================

# Map common team codes -> canonical short code used on Player CSVs
TEAM_CODE = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI",
    "CIN": "CIN", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GNB": "GNB", "GB": "GNB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KAN": "KAN", "KC": "KAN", "LVR": "LVR", "LV": "LVR",
    "LAC": "LAC", "LAR": "LAR", "MIA": "MIA", "MIN": "MIN", "NWE": "NWE", "NE": "NWE",
    "NOR": "NOR", "NO": "NOR", "NYG": "NYG", "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT",
    "SEA": "SEA", "SFO": "SFO", "SF": "SFO", "TAM": "TAM", "TB": "TAM", "TEN": "TEN", "WAS": "WAS",
}

# Name -> code (for convenience)
NAME_TO_CODE = {
    "San Francisco 49ers": "SFO",
    "Atlanta Falcons": "ATL",
    "Los Angeles Rams": "LAR",
    "Jacksonville Jaguars": "JAX",
    "Green Bay Packers": "GNB",
    "Denver Broncos": "DEN",
    "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LVR",
    "Minnesota Vikings": "MIN",
    "Washington Commanders": "WAS",
    "Philadelphia Eagles": "PHI",
    "Seattle Seahawks": "SEA",
    "Indianapolis Colts": "IND",
    "Detroit Lions": "DET",
    "Arizona Cardinals": "ARI",
    "Baltimore Ravens": "BAL",
    "Cleveland Browns": "CLE",
    "New Orleans Saints": "NOR",
    "Tampa Bay Buccaneers": "TAM",
    "Cincinnati Bengals": "CIN",
    "Tennessee Titans": "TEN",
    "Houston Texans": "HOU",
    "Buffalo Bills": "BUF",
    "Kansas City Chiefs": "KAN",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "New England Patriots": "NWE",
    "Pittsburgh Steelers": "PIT",
    "Dallas Cowboys": "DAL",
    "Miami Dolphins": "MIA",
}

# EPA values (from your list). Negative is good defense.
DEF_ROWS = [
    ("San Francisco 49ers", -0.16, -0.27, -0.01),
    ("Atlanta Falcons",     -0.14, -0.18, -0.05),
    ("Los Angeles Rams",    -0.13, -0.20,  0.00),
    ("Jacksonville Jaguars",-0.13, -0.20,  0.03),
    ("Green Bay Packers",   -0.12, -0.08, -0.19),
    ("Denver Broncos",      -0.10, -0.03, -0.19),
    ("Los Angeles Chargers",-0.10, -0.20,  0.18),
    ("Las Vegas Raiders",   -0.08,  0.12, -0.45),
    ("Minnesota Vikings",   -0.06, -0.25,  0.13),
    ("Washington Commanders",-0.06,-0.06, -0.04),
    ("Philadelphia Eagles", -0.05, -0.08,  0.00),
    ("Seattle Seahawks",    -0.04,  0.06, -0.17),
    ("Indianapolis Colts",  -0.03, -0.13,  0.17),
    ("Detroit Lions",       -0.03,  0.16, -0.25),
    ("Arizona Cardinals",   -0.02, -0.01, -0.05),
    ("Baltimore Ravens",     0.00, -0.06,  0.12),
    ("Cleveland Browns",     0.01,  0.15, -0.18),
    ("New Orleans Saints",   0.01,  0.05, -0.04),
    ("Tampa Bay Buccaneers", 0.02,  0.08, -0.09),
    ("Cincinnati Bengals",   0.02,  0.03, -0.01),
    ("Tennessee Titans",     0.03,  0.03,  0.04),
    ("Houston Texans",       0.05,  0.03,  0.08),
    ("Buffalo Bills",        0.05,  0.05,  0.05),
    ("Kansas City Chiefs",   0.12,  0.19,  0.04),
    ("Carolina Panthers",    0.12,  0.12,  0.12),
    ("Chicago Bears",        0.14,  0.27,  0.01),
    ("New York Giants",      0.14,  0.12,  0.15),
    ("New York Jets",        0.14,  0.16,  0.11),
    ("New England Patriots", 0.14,  0.33, -0.22),
    ("Pittsburgh Steelers",  0.16,  0.23,  0.09),
    ("Dallas Cowboys",       0.17,  0.26,  0.06),
    ("Miami Dolphins",       0.28,  0.41,  0.13),
]

def build_def_df() -> pd.DataFrame:
    df = pd.DataFrame(DEF_ROWS, columns=["team_name","epa_play","epa_pass","epa_rush"])
    df["team_code"] = df["team_name"].map(NAME_TO_CODE)
    # Normalize → factor around 1 (lower EPA => <1 => harder)
    def to_factor(col: str) -> pd.Series:
        x = df[col].astype(float)
        m = x.mean()
        p95 = x.quantile(0.95)
        p05 = x.quantile(0.05)
        denom = max(abs(p95-m), abs(p05-m), 1e-6)
        f = 1.0 + (x - m) / (2.5*denom)   # spread to approx 0.8–1.2
        return f.clip(0.75, 1.25)
    df["factor_play"] = to_factor("epa_play")
    df["factor_pass"] = to_factor("epa_pass")
    df["factor_rush"] = to_factor("epa_rush")
    return df

DEF_DF = build_def_df()

def defense_factor(team_code: Optional[str], kind: str) -> float:
    """
    kind: 'pass' for QB/WR receiving, 'rush' for RB rushing.
    If team_code missing/unknown → 1.00
    """
    if not team_code:
        return 1.0
    code = TEAM_CODE.get(team_code.upper(), team_code.upper())
    row = DEF_DF.loc[DEF_DF["team_code"] == code]
    if row.empty:
        return 1.0
    if kind == "rush":
        return float(row["factor_rush"].iloc[0])
    else:
        return float(row["factor_pass"].iloc[0])

# =============================================================================
# MLB page (simple; probables optional)
# =============================================================================

with page[1]:
    st.subheader("⚾ MLB — Poisson matchup (manual inputs)")
    colA, colB = st.columns(2)
    with colA:
        mu_home = st.number_input("Expected runs — Home", value=4.6, step=0.1)
    with colB:
        mu_away = st.number_input("Expected runs — Away", value=4.4, step=0.1)
    if st.button("Simulate MLB Game"):
        p_home, h_mean, a_mean = poisson_game(mu_home, mu_away)
        st.success(f"Home win %: **{p_home*100:.1f}%**  | Home runs: **{h_mean:.2f}**  | Away runs: **{a_mean:.2f}**")

# =============================================================================
# NFL page (manual PF/PA inputs to avoid dependency headaches)
# =============================================================================

with page[0]:
    st.subheader("🏈 NFL — Poisson matchup (manual PF/PA)")
    st.caption("Enter each team's expected points (you can use season PF/PA averages).")
    c1, c2, c3 = st.columns(3)
    with c1:
        home = st.text_input("Home team code (e.g., DAL)", value="DAL")
    with c2:
        mu_home = st.number_input("Expected points — Home", value=24.0, step=0.5)
    with c3:
        away = st.text_input("Away team code (e.g., PHI)", value="PHI")
    mu_away = st.number_input("Expected points — Away", value=22.0, step=0.5)
    if st.button("Simulate NFL Game"):
        # Add small home edge
        p_home, h_mean, a_mean = poisson_game(mu_home + HOME_EDGE_NFL, mu_away)
        st.success(f"{home} win %: **{p_home*100:.1f}%** | {home} pts: **{h_mean:.2f}** | {away} pts: **{a_mean:.2f}**")

# =============================================================================
# Player Props page (CSV paste → select player → defense auto-factor → simulation)
# =============================================================================

def parse_pasted_csv(txt: str) -> pd.DataFrame:
    """
    Accepts your pasted QB/RB/WR csv (as text).
    Returns a clean dataframe with lowercase columns, including: Player, Team, Pos, and stat columns.
    Works with the examples you've been sending.
    """
    # Remove obvious header noise like "Receiving,Receiving..." lines
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    # Find the real header row (the one that contains "Player")
    hdr_idx = None
    for i, ln in enumerate(lines[:10]):
        if "Player" in ln.split(","):
            hdr_idx = i
            break
    if hdr_idx is None:
        # fallback: assume first line is header
        hdr_idx = 0
    clean = "\n".join(lines[hdr_idx:])
    df = pd.read_csv(pd.compat.StringIO(clean))
    # standardize
    df.columns = [c.strip() for c in df.columns]
    # Some versions have duplicate stat blocks; keep the first block by dropping trailing dup names
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def normal_over_prob(mu: float, sd: float, line: float) -> float:
    if sd <= 0:
        return float(mu > line)
    z = (line - mu) / sd
    # phi(z)
    return float(1.0 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

with page[2]:
    st.subheader("📈 Player Props — paste your CSV, pick player & market, run sim")
    with st.expander("Paste CSV text (QB / RB / WR)", expanded=True):
        csv_text = st.text_area("Paste the full CSV text here", height=220, placeholder="Paste your CSV (including header row)")

    # Choose market
    market = st.selectbox("Market", [
        "Passing Yards (QB)",
        "Rushing Yards (RB)",
        "Receiving Yards (WR)",
    ])

    # Yardage line
    prop_line = st.number_input("Yardage line", value=230.0 if "Pass" in market else 60.0, step=0.5)

    # Opponent code (optional)
    opp_code = st.text_input("Opponent (team code, optional — e.g., PHI). Leave blank for league-average defense.", value="")

    # Defaults for volatility per market
    DEFAULT_SD = {
        "Passing Yards (QB)": 55.0,
        "Rushing Yards (RB)": 25.0,
        "Receiving Yards (WR)": 28.0,
    }
    sim_sd = DEFAULT_SD[market]

    if st.button("Run Player Prop Simulation"):
        if not csv_text.strip():
            st.warning("Paste your CSV first.")
            st.stop()

        try:
            df = parse_pasted_csv(csv_text)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        # Guess which columns to use
        # We support the columns from your examples.
        lower = {c.lower(): c for c in df.columns}
        # Normalize names we need
        col_player = lower.get("player", "Player")
        col_team   = lower.get("team", "Team")
        col_pos    = lower.get("pos", "Pos")

        # stat columns
        col_pass = next((c for c in df.columns if c.lower() in ("pass yds","pass yds.","yds","passing yards","yds")), None)
        col_rush = next((c for c in df.columns if c.lower() in ("rush yds","rushing yds","yds","rushing yards")), None)
        col_recv = next((c for c in df.columns if c.lower() in ("rec yds","receive yds","receiving yds","yds","receiving yards")), None)

        # Create a selection list showing Player (Team, Pos)
        show_cols = [col_player, col_team, col_pos]
        avail = df[show_cols].copy()
        avail["label"] = avail[col_player].astype(str) + " — " + avail[col_team].astype(str) + " (" + avail[col_pos].astype(str) + ")"
        pick = st.selectbox("Pick player", avail["label"].tolist())
        row = df.iloc[avail["label"].tolist().index(pick)]

        # Base mean by market
        if market.startswith("Passing"):
            # Prefer a passing yards column; fallback to Yds when Pos == QB
            if col_pass is None:
                col_pass = "Yds"
            base_mu = float(pd.to_numeric(row.get(col_pass, np.nan), errors="coerce"))
            d_factor = defense_factor(opp_code, "pass")
        elif market.startswith("Rushing"):
            if col_rush is None:
                # RB block in your CSV uses "Yds" for rushing
                col_rush = "Yds"
            base_mu = float(pd.to_numeric(row.get(col_rush, np.nan), errors="coerce"))
            d_factor = defense_factor(opp_code, "rush")
        else:  # Receiving
            if col_recv is None:
                col_recv = "Yds"
            base_mu = float(pd.to_numeric(row.get(col_recv, np.nan), errors="coerce"))
            d_factor = defense_factor(opp_code, "pass")

        if not np.isfinite(base_mu):
            st.error("Couldn't find a numeric yardage column for this market in your CSV row.")
            st.stop()

        adj_mu = max(0.0, base_mu * d_factor)
        p_over = normal_over_prob(adj_mu, sim_sd, prop_line)
        p_under = 1.0 - p_over

        st.success(
            f"**{row[col_player]} — {market}**\n\n"
            f"CSV mean: **{base_mu:.1f}** yds · Defense factor: **×{d_factor:.3f}** → "
            f"Adjusted mean: **{adj_mu:.1f}** yds · Line: **{prop_line:.1f}**\n\n"
            f"**P(over) = {p_over*100:.1f}%**, **P(under) = {p_under*100:.1f}%**"
        )

        with st.expander("Show player row used"):
            st.dataframe(row.to_frame().T, use_container_width=True)

        with st.expander("Defense table (normalized factors)"):
            show_cols = ["team_code","team_name","epa_play","epa_pass","epa_rush","factor_play","factor_pass","factor_rush"]
            st.dataframe(DEF_DF[show_cols].sort_values("factor_play"), use_container_width=True)
