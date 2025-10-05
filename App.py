# app.py — NFL Player Props (Receptions only) — clean, position-safe, flexible CSV mapping
# Run: streamlit run app.py

import math
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Optional, Tuple

# ---------------------------- App Settings ----------------------------
st.set_page_config(page_title="NFL Player Props — Receptions Only", layout="wide")
st.title("📈 NFL Player Props — Receptions Only (WR/TE/RB)")

SIM_TRIALS_DEFAULT = 10_000
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ---------------------------- Helpers ----------------------------
def american_to_implied_prob(american_odds: float) -> Optional[float]:
    """Convert American odds to implied probability (without vig removal)."""
    try:
        o = float(american_odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 100 / (o + 100)
    else:
        return -o / (-o + 100)

def kelly_fraction(p: float, american_odds: float) -> float:
    """
    Kelly for American odds (fraction of bankroll).
    For +X: b = X/100; For -X: b = 100/X
    """
    try:
        o = float(american_odds)
    except Exception:
        return 0.0
    if o == 0:
        return 0.0
    if o > 0:
        b = o / 100.0
    else:
        b = 100.0 / abs(o)
    q = 1 - p
    edge = (b * p) - q
    denom = b
    if denom == 0:
        return 0.0
    k = edge / denom
    return max(0.0, k)

def fair_price_from_prob(p: float) -> Tuple[Optional[float], Optional[float]]:
    """Return decimal and American fair odds given a probability p."""
    if p <= 0 or p >= 1:
        return None, None
    dec = 1.0 / p
    if p > 0.5:
        # Negative American
        am = -100 * p / (1 - p)
    else:
        # Positive American
        am = 100 * (1 - p) / p
    return dec, am

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def binomial_sample(n_arr: np.ndarray, p: float) -> np.ndarray:
    """Vectorized binomial for multiple trials with (possibly) varying n per trial."""
    # np.random.binomial handles vector n with scalar p
    return np.random.binomial(n_arr.astype(int), clamp(p, 1e-6, 1-1e-6))

def poisson_sample(lam: float, size: int) -> np.ndarray:
    return np.random.poisson(lam=clamp(lam, 1e-6, 1e6), size=size)

def clean_name(x: str) -> str:
    return str(x).strip()

# ---------------------------- UI - Sidebar ----------------------------
st.sidebar.header("Data Inputs")

st.sidebar.markdown("**1) Players (WR/TE/RB)** — a CSV that has at least:")
st.sidebar.markdown("- Player Name\n- Team (abbrev like PHI, KC, etc.)\n- Position (WR/TE/RB)\n- Targets per game (or similar)\n- Catch rate (0–1 or 0–100%)")

players_file = st.sidebar.file_uploader("Upload players CSV (WR/TE/RB)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("**2) Defense vs Receptions** — team-level adjustment (optional):")
st.sidebar.markdown("Columns: Team, Defense_Receptions_Scaler (e.g., 0.95 for tough defense, 1.05 for permissive)")
defense_file = st.sidebar.file_uploader("Upload defense adjustment CSV (optional)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("**3) Odds CSV** — receptions props for your book(s):")
st.sidebar.markdown("Columns: Player, Line, Over_Odds, Under_Odds, (optional) Book, Team")
odds_file = st.sidebar.file_uploader("Upload odds CSV (receptions props)", type=["csv"])

st.sidebar.markdown("---")
sim_trials = st.sidebar.number_input("Simulation trials", min_value=1_000, max_value=200_000, value=SIM_TRIALS_DEFAULT, step=1_000)
seed_input = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=RNG_SEED, step=1)
np.random.seed(seed_input)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If your columns have different names, map them below in the ‘Column Mapper’.")

# ---------------------------- Load Data ----------------------------
def read_csv(file) -> Optional[pd.DataFrame]:
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        # Prevent mixed dtypes with NaNs in numeric
        return df
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None

players_df = read_csv(players_file)
def_df = read_csv(defense_file) if defense_file else None
odds_df = read_csv(odds_file)

# ---------------------------- Column Mapper ----------------------------
st.header("Column Mapper")

st.markdown("""
Map your CSV columns to what the app expects.  
**Players CSV (WR/TE/RB)** must include:  
- Player Name  
- Team (abbrev)  
- Position (WR/TE/RB)  
- Targets per game  
- Catch rate (0–1 or 0–100%)  

**Defense CSV** (optional) should include:  
- Team (defensive team abbrev)  
- Defense Receptions Scaler (e.g., 0.95 to reduce expected targets, 1.05 to boost)
  
**Odds CSV** must include:  
- Player  
- Line (e.g., 5.5)  
- Over_Odds, Under_Odds (American)  
- (Optional) Team, Book
""")

def column_selector(df: Optional[pd.DataFrame], label_map: Dict[str, str], key_prefix: str) -> Dict[str, Optional[str]]:
    selected = {}
    if df is None:
        for tag in label_map:
            selected[tag] = None
        return selected

    cols = ["<none>"] + list(df.columns)
    for tag, label in label_map.items():
        default = "<none>"
        # try auto-pick common names
        for c in df.columns:
            lc = c.lower()
            if tag == "player" and lc in ("player","name","player_name"):
                default = c
            if tag == "team" and lc in ("team","tm","teams","club"):
                default = c
            if tag == "pos" and lc in ("pos","position"):
                default = c
            if tag == "targets_pg" and lc in ("targets_pg","tgt_pg","targets per game","targets"):
                default = c
            if tag == "catch_rate" and lc in ("catch_rate","catch%","catch pct","reception_pct","rec_pct"):
                default = c
            if tag == "def_team" and lc in ("team","def_team","defense_team","def"):
                default = c
            if tag == "def_adj" and lc in ("defense_receptions_scaler","def_adj","receptions_scaler","receptions_adj","def_scaler"):
                default = c
            if tag == "line" and lc in ("line","prop_line","receptions_line"):
                default = c
            if tag == "over" and lc in ("over_odds","o_odds","over"):
                default = c
            if tag == "under" and lc in ("under_odds","u_odds","under"):
                default = c
            if tag == "book" and lc in ("book","sportsbook"):
                default = c

        sel = st.selectbox(f"{key_prefix}: {label}", cols, index=cols.index(default) if default in cols else 0, key=f"{key_prefix}_{tag}")
        selected[tag] = None if sel == "<none>" else sel
    return selected

st.subheader("Players CSV mapping")
players_map = column_selector(
    players_df,
    {
        "player": "Player Name",
        "team": "Team (abbrev)",
        "pos": "Position (WR/TE/RB)",
        "targets_pg": "Targets per game",
        "catch_rate": "Catch rate (0–1 or 0–100%)",
    },
    key_prefix="players"
)

st.subheader("Defense CSV mapping (optional)")
def_map = column_selector(
    def_df,
    {
        "def_team": "Team (defensive team)",
        "def_adj": "Defense Receptions Scaler",
    },
    key_prefix="defense"
)

st.subheader("Odds CSV mapping")
odds_map = column_selector(
    odds_df,
    {
        "player": "Player",
        "team": "Team (optional but recommended)",
        "line": "Line (e.g., 5.5)",
        "over": "Over Odds (American)",
        "under": "Under Odds (American)",
        "book": "Book (optional)",
    },
    key_prefix="odds"
)

# ---------------------------- Processing ----------------------------
def coerce_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.to_numeric(pd.Series([np.nan]*len(series)), errors="coerce")

def normalize_players(players_df: pd.DataFrame, m: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = players_df.copy()
    for k in ["player","team","pos","targets_pg","catch_rate"]:
        if m.get(k) is None:
            raise ValueError(f"Players CSV missing required column mapping: {k}")

    df = df.rename(columns={
        m["player"]: "Player",
        m["team"]: "Team",
        m["pos"]: "Pos",
        m["targets_pg"]: "Targets_PG",
        m["catch_rate"]: "CatchRate"
    })

    # Clean + numeric coercions
    df["Player"] = df["Player"].astype(str).map(clean_name)
    df["Team"] = df["Team"].astype(str).map(clean_name).str.upper()
    df["Pos"] = df["Pos"].astype(str).str.upper().str.strip()

    df["Targets_PG"] = coerce_numeric(df["Targets_PG"])
    df["CatchRate"] = coerce_numeric(df["CatchRate"])

    # If catch rate is in 0-100, convert to 0-1
    mask_pct = df["CatchRate"] > 1.0
    df.loc[mask_pct, "CatchRate"] = df.loc[mask_pct, "CatchRate"] / 100.0

    # Filter ONLY WR/TE/RB for receptions props
    df = df[df["Pos"].isin(["WR","TE","RB"])].copy()

    # Drop rows missing essentials
    df = df.dropna(subset=["Player","Team","Targets_PG","CatchRate"])

    # Cap catch rate a bit to avoid 0 or 1
    df["CatchRate"] = df["CatchRate"].clip(0.02, 0.98)

    return df[["Player","Team","Pos","Targets_PG","CatchRate"]]

def normalize_defense(def_df: pd.DataFrame, m: Dict[str, Optional[str]]) -> pd.DataFrame:
    if m.get("def_team") is None or m.get("def_adj") is None:
        # Return empty -> default = 1.0 later
        return pd.DataFrame(columns=["Team","Def_Receptions_Scaler"])

    df = def_df.copy()
    df = df.rename(columns={
        m["def_team"]: "Team",
        m["def_adj"]: "Def_Receptions_Scaler"
    })
    df["Team"] = df["Team"].astype(str).map(clean_name).str.upper()
    df["Def_Receptions_Scaler"] = coerce_numeric(df["Def_Receptions_Scaler"]).fillna(1.0)
    return df[["Team","Def_Receptions_Scaler"]]

def normalize_odds(odds_df: pd.DataFrame, m: Dict[str, Optional[str]]) -> pd.DataFrame:
    for k in ["player","line","over","under"]:
        if m.get(k) is None:
            raise ValueError(f"Odds CSV missing required column mapping: {k}")

    df = odds_df.copy()
    rename_map = {
        m["player"]: "Player",
        m["line"]: "Line",
        m["over"]: "Over_Odds",
        m["under"]: "Under_Odds",
    }
    if m.get("team"):
        rename_map[m["team"]] = "Team"
    if m.get("book"):
        rename_map[m["book"]] = "Book"

    df = df.rename(columns=rename_map)

    df["Player"] = df["Player"].astype(str).map(clean_name)
    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str).map(clean_name).str.upper()
    else:
        df["Team"] = ""

    df["Book"] = df["Book"].astype(str) if "Book" in df.columns else "—"

    df["Line"] = coerce_numeric(df["Line"])
    df["Over_Odds"] = coerce_numeric(df["Over_Odds"])
    df["Under_Odds"] = coerce_numeric(df["Under_Odds"])

    df = df.dropna(subset=["Player","Line","Over_Odds","Under_Odds"]).copy()

    return df[["Player","Team","Line","Over_Odds","Under_Odds","Book"]]

def attach_defense_scaler(players: pd.DataFrame, defense: Optional[pd.DataFrame]) -> pd.DataFrame:
    if defense is None or defense.empty:
        players["Def_Receptions_Scaler"] = 1.0
        return players
    # If player Team not found in defense table, default 1.0
    merged = players.merge(defense, on="Team", how="left")
    merged["Def_Receptions_Scaler"] = merged["Def_Receptions_Scaler"].fillna(1.0)
    return merged

def merge_players_odds(players: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    # Merge primarily on Player, optionally use Team when available
    with_team = odds["Team"].str.len() > 0
    left_cols = ["Player","Team"]
    p = players.copy()
    o = odds.copy()

    # First attempt: Player + Team
    m1 = p.merge(o[with_team], on=["Player","Team"], how="left", suffixes=("",""))
    # Second attempt: fill remaining by Player only (for rows where Team was blank in odds)
    m2 = p.merge(o[~with_team][["Player","Line","Over_Odds","Under_Odds","Book"]], on="Player", how="left", suffixes=("",""))

    # Combine m1 & m2 preferring m1 where present
    out = p.copy()
    for col in ["Line","Over_Odds","Under_Odds","Book"]:
        out[col] = m1[col]
        out[col] = out[col].fillna(m2[col])

    return out

def simulate_receptions_row(targets_pg: float, catch_rate: float, defense_scaler: float, trials: int) -> np.ndarray:
    """
    Model: Poisson targets with mean = targets_pg * defense_scaler
           Receptions = Binomial(targets, catch_rate)
    Returns array of simulated receptions length=trials
    """
    lam = float(targets_pg) * float(defense_scaler)
    tgts = poisson_sample(lam, trials)
    recs = binomial_sample(tgts, catch_rate)
    return recs.astype(float)

def evaluate_market(line: float, sims: np.ndarray) -> Dict[str, float]:
    """Compute prob over/under, fair odds, EV & Kelly for both sides for given line."""
    # Over if strictly greater than line (supports hooks like 4.5 naturally)
    p_over = float(np.mean(sims > line))
    p_under = 1.0 - p_over

    fair_dec_over, fair_am_over = fair_price_from_prob(p_over)
    fair_dec_under, fair_am_under = fair_price_from_prob(p_under)

    return {
        "p_over": p_over,
        "p_under": p_under,
        "fair_am_over": fair_am_over if fair_am_over is not None else np.nan,
        "fair_am_under": fair_am_under if fair_am_under is not None else np.nan,
    }

def expected_value(p: float, american_odds: float, stake: float = 1.0) -> float:
    """
    EV of a $stake bet. American odds: +X wins X on 100; -X wins 100 on X.
    Payout includes profit only (not stake).
    """
    try:
        o = float(american_odds)
    except Exception:
        return np.nan
    if o > 0:
        win = stake * (o / 100.0)
        lose = stake
    else:
        win = stake * (100.0 / abs(o))
        lose = stake
    return p * win - (1 - p) * lose

# ---------------------------- Main Logic ----------------------------
st.header("Results — Receptions Props")

if players_df is None or odds_df is None:
    st.info("Upload at least the **Players CSV** and **Odds CSV** to proceed. Defense CSV is optional.")
else:
    try:
        players_norm = normalize_players(players_df, players_map)
    except Exception as e:
        st.error(f"Players CSV error: {e}")
        players_norm = None

    defense_norm = None
    if def_df is not None and not def_df.empty:
        try:
            defense_norm = normalize_defense(def_df, def_map)
        except Exception as e:
            st.warning(f"Defense CSV issue (will default to 1.0): {e}")
            defense_norm = None

    try:
        odds_norm = normalize_odds(odds_df, odds_map)
    except Exception as e:
        st.error(f"Odds CSV error: {e}")
        odds_norm = None

    if players_norm is not None and odds_norm is not None:
        # Attach defense scaler
        players_aug = attach_defense_scaler(players_norm, defense_norm)

        # Merge odds onto players
        base = merge_players_odds(players_aug, odds_norm)

        # Keep only rows with odds available
        base = base.dropna(subset=["Line","Over_Odds","Under_Odds"]).copy()

        # Filters
        left, right = st.columns([3,2])
        with left:
            team_filter = st.multiselect(
                "Filter by Team (player team):",
                sorted(base["Team"].dropna().unique().tolist()),
                default=[]
            )
            pos_filter = st.multiselect(
                "Filter by Position:",
                ["WR","TE","RB"],
                default=["WR","TE","RB"]
            )
        with right:
            min_targets = st.number_input("Min Targets per game", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
            min_catch = st.slider("Min Catch Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        filt = base.copy()
        if team_filter:
            filt = filt[filt["Team"].isin(team_filter)]
        if pos_filter:
            filt = filt[filt["Pos"].isin(pos_filter)]
        if min_targets > 0:
            filt = filt[filt["Targets_PG"] >= min_targets]
        if min_catch > 0:
            filt = filt[filt["CatchRate"] >= min_catch]

        if filt.empty:
            st.warning("No rows after filters. Loosen filters or check mappings.")
        else:
            # Simulate per row
            st.write(f"Simulating **{sim_trials:,}** trials per player (seed={seed_input}).")
            rows = []
            for _, r in filt.iterrows():
                sims = simulate_receptions_row(
                    targets_pg=float(r["Targets_PG"]),
                    catch_rate=float(r["CatchRate"]),
                    defense_scaler=float(r["Def_Receptions_Scaler"]),
                    trials=int(sim_trials)
                )
                metrics = evaluate_market(line=float(r["Line"]), sims=sims)

                p_over = metrics["p_over"]
                p_under = metrics["p_under"]

                imp_over = american_to_implied_prob(r["Over_Odds"])
                imp_under = american_to_implied_prob(r["Under_Odds"])

                ev_over = expected_value(p_over, r["Over_Odds"], stake=1.0)
                ev_under = expected_value(p_under, r["Under_Odds"], stake=1.0)

                k_over = kelly_fraction(p_over, r["Over_Odds"])
                k_under = kelly_fraction(p_under, r["Under_Odds"])

                rows.append({
                    "Player": r["Player"],
                    "Team": r["Team"],
                    "Pos": r["Pos"],
                    "Line": float(r["Line"]),
                    "Targets_PG": float(r["Targets_PG"]),
                    "CatchRate": float(r["CatchRate"]),
                    "Def_Scaler": float(r["Def_Receptions_Scaler"]),
                    "Over_Odds": int(r["Over_Odds"]),
                    "Under_Odds": int(r["Under_Odds"]),
                    "Book": r.get("Book", "—"),
                    "P(Over)": round(p_over, 4),
                    "P(Under)": round(p_under, 4),
                    "Implied Over": round(imp_over, 4) if imp_over is not None else np.nan,
                    "Implied Under": round(imp_under, 4) if imp_under is not None else np.nan,
                    "Edge Over": round(p_over - (imp_over if imp_over is not None else 0), 4) if imp_over is not None else np.nan,
                    "Edge Under": round(p_under - (imp_under if imp_under is not None else 0), 4) if imp_under is not None else np.nan,
                    "Fair Am Over": round(metrics["fair_am_over"], 1) if not pd.isna(metrics["fair_am_over"]) else np.nan,
                    "Fair Am Under": round(metrics["fair_am_under"], 1) if not pd.isna(metrics["fair_am_under"]) else np.nan,
                    "EV Over ($1)": round(ev_over, 4),
                    "EV Under ($1)": round(ev_under, 4),
                    "Kelly Over": round(k_over, 4),
                    "Kelly Under": round(k_under, 4),
                    "Sim Mean Rec": round(float(np.mean(sims)), 3),
                    "Sim Med Rec": round(float(np.median(sims)), 3),
                    "Sim Std Rec": round(float(np.std(sims, ddof=1)), 3),
                })

            out_df = pd.DataFrame(rows)

            # Sorting & thresholds
            c1, c2, c3, c4 = st.columns([1.5,1.5,1.5,1])
            with c1:
                sort_by = st.selectbox("Sort by", ["Edge Over","Edge Under","EV Over ($1)","EV Under ($1)","P(Over)","P(Under)","Kelly Over","Kelly Under"], index=0)
            with c2:
                ascending = st.checkbox("Ascending", value=False)
            with c3:
                min_edge = st.number_input("Min absolute edge (either side)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            with c4:
                min_kelly = st.number_input("Min Kelly (either side)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

            # Threshold filter (keep if either side meets thresholds)
            keep = (
                (out_df["Edge Over"].abs() >= min_edge) |
                (out_df["Edge Under"].abs() >= min_edge) |
                (out_df["Kelly Over"] >= min_kelly) |
                (out_df["Kelly Under"] >= min_kelly)
            )
            filtered = out_df[keep].copy() if min_edge > 0 or min_kelly > 0 else out_df.copy()

            filtered = filtered.sort_values(by=sort_by, ascending=ascending, kind="mergesort").reset_index(drop=True)

            st.dataframe(
                filtered[
                    [
                        "Player","Team","Pos","Book","Line",
                        "Targets_PG","CatchRate","Def_Scaler",
                        "Over_Odds","Under_Odds",
                        "P(Over)","P(Under)",
                        "Implied Over","Implied Under",
                        "Edge Over","Edge Under",
                        "Fair Am Over","Fair Am Under",
                        "EV Over ($1)","EV Under ($1)",
                        "Kelly Over","Kelly Under",
                        "Sim Mean Rec","Sim Med Rec","Sim Std Rec",
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            # CSV Download
            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_bytes,
                file_name="receptions_props_results.csv",
                mime="text/csv"
            )

# ---------------------------- Quick Schema Notes ----------------------------
with st.expander("ℹ️ CSV Schema Examples"):
    st.markdown("""
**Players CSV (WR/TE/RB):**
