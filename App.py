# app.py
# -------------------------------------------------------
# Streamlit app with 3 pages:
#   1) NFL            (your existing page)
#   2) MLB            (your existing page)
#   3) NFL Props      (new: upload CSVs, simulate props)
#
# Dependencies: streamlit, pandas, numpy
# -------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL + MLB + Props", layout="centered")

# ------------------------------------------------------------------
# Try to use your existing pages exactly as-is.
# If your project already defines these functions elsewhere, import them.
# Otherwise, the placeholders below will show a friendly message.
# ------------------------------------------------------------------
try:
    # If you keep your old functions in another file, e.g. pages_core.py
    # from pages_core import render_nfl_page, render_mlb_page
    # pass  # <-- uncomment the import above and remove this 'pass'
    raise ImportError  # remove this if you really import from elsewhere
except Exception:
    def render_nfl_page():
        st.header("NFL (existing page)")
        st.info("This is a placeholder. Paste your original NFL page code here.")

    def render_mlb_page():
        st.header("MLB (existing page)")
        st.info("This is a placeholder. Paste your original MLB page code here.")


# ========================= NEW PAGE ========================= #
def render_nfl_props_page():
    st.header("🏈 NFL Player Props — 2025 REG (beta)")
    st.caption("Upload your QB / RB / WR CSVs. Optional: Defense CSV for matchup scaling.")

    # ---- Uploaders ----
    c1, c2 = st.columns(2)
    with c1:
        qbs_file = st.file_uploader("QBs CSV", type=["csv"], key="qbs_up")
        rbs_file = st.file_uploader("RBs CSV", type=["csv"], key="rbs_up")
    with c2:
        wrs_file = st.file_uploader("WRs CSV", type=["csv"], key="wrs_up")
        def_file = st.file_uploader("Defense CSV (optional)", type=["csv"], key="def_up")

    # ---- Helpers ----
    def load_csv(file):
        if not file:
            return None
        return pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")), engine="python")

    def normalize_headers(df):
        """Trim/standardize headers and map common variants."""
        if df is None:
            return None
        df = df.copy()
        df.columns = [c.strip().replace(" ", "").replace("%", "Pct") for c in df.columns]

        # Map common variants to a common set
        rename_map = {
            "Player": "Player",
            "Team": "Team",
            "Pos": "Pos",
            "Season": "Season",
            "Y/G": "Y_per_G",
            "YPG": "Y_per_G",
            "Yds/G": "Y_per_G",
            "YdsperG": "Y_per_G",
            "Yds": "Yds",
            "G": "G"
        }
        for old, new in rename_map.items():
            for cand in {old, old.replace("/", ""), old.replace(" ", "")}:
                if cand in df.columns and new not in df.columns:
                    df = df.rename(columns={cand: new})
        return df

    def ensure_pos(df, pos_tag):
        if df is None:
            return None
        if "Pos" not in df.columns or df["Pos"].isna().all():
            df["Pos"] = pos_tag
        return df

    def ensure_y_per_g(df):
        if df is None:
            return None
        if "Y_per_G" not in df.columns:
            if {"Yds", "G"}.issubset(df.columns):
                with np.errstate(divide="ignore", invalid="ignore"):
                    df["Y_per_G"] = df["Yds"] / df["G"].replace({0: np.nan})
            else:
                df["Y_per_G"] = 0.0
        return df

    # ---- Load & tidy player tables ----
    qbs = ensure_y_per_g(ensure_pos(normalize_headers(load_csv(qbs_file)), "QB"))
    rbs = ensure_y_per_g(ensure_pos(normalize_headers(load_csv(rbs_file)), "RB"))
    wrs = ensure_y_per_g(ensure_pos(normalize_headers(load_csv(wrs_file)), "WR"))

    # Concatenate only frames that have Player/Team/Pos
    need_cols = {"Player", "Team", "Pos"}
    frames = [d for d in [qbs, rbs, wrs] if d is not None and need_cols.issubset(set(d.columns))]
    if not frames:
        st.info("Upload at least one of the three player CSVs to begin.")
        return

    players = pd.concat(frames, ignore_index=True, sort=False)

    # Keep useful columns without breaking your data
    keep_cols = [c for c in ["Player", "Team", "Pos", "Yds", "G", "Season", "Y_per_G"] if c in players.columns]
    players = players[keep_cols].copy()

    with st.expander("What the model uses"):
        st.markdown(
            "- **Y_per_G** from your CSV as the baseline.\n"
            "- Optional **Defense CSV** scales baseline by opponent difficulty.\n"
            "- Then a Normal Monte-Carlo uses your SD (volatility) to estimate Over/Under %.\n"
            "- No Excel writing, so no `xlsxwriter` dependency."
        )

    # ---- Defense CSV ----
    def_df = load_csv(def_file)
    defense_note = ""
    if def_df is not None:
        def_df = def_df.copy()
        def_df.columns = [c.strip().replace(" ", "").lower() for c in def_df.columns]
        # Try to auto-detect typical names
        name_map = {}
        auto_names = {
            "team": ["team", "defteam", "opponent", "def", "tm"],
            "pass_yds_pg": ["passydspg", "passydsallowedpg", "passyards_pg", "passyardsallowedpg", "passydsperg"],
            "rush_yds_pg": ["rushydspg", "rushydsallowedpg", "rushyards_pg", "rushyardsallowedpg", "rushydsperg"],
            "rec_yds_pg":  ["recydspg", "recyards_pg", "recyardsallowedpg", "receivingyds_pg", "receivingydsallowedpg"]
        }
        for want, alts in auto_names.items():
            for a in alts:
                if a in def_df.columns:
                    name_map[a] = want
                    break
        def_df = def_df.rename(columns=name_map)
        if "team" not in def_df.columns:
            defense_note = "Could not detect a defense team column. Defense file will be ignored."
            def_df = None

    # ---- Market / Player selection ----
    market = st.selectbox("Market", ["Pass Yds (QB)", "Rush Yds (RB)", "Rec Yds (WR)"])
    if market == "Pass Yds (QB)":
        pool = players[players["Pos"] == "QB"].copy()
        want_col = "pass_yds_pg"
    elif market == "Rush Yds (RB)":
        pool = players[players["Pos"] == "RB"].copy()
        want_col = "rush_yds_pg"
    else:
        pool = players[players["Pos"] == "WR"].copy()
        want_col = "rec_yds_pg"

    if pool.empty:
        st.warning("No players found for this market in your uploads.")
        return

    player_name = st.selectbox("Player", sorted(pool["Player"].dropna().unique().tolist()))
    row = pool.loc[pool["Player"] == player_name].iloc[0]
    team = row.get("Team", "")
    base_y_pg = float(row.get("Y_per_G", 0.0)) if pd.notna(row.get("Y_per_G", np.nan)) else 0.0

    # ---- Defense factor ----
    st.subheader("Matchup / Defense adjustment")
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        opp_team = st.text_input("Opponent team code (e.g., NYJ, DAL)", value="")
    with cc2:
        manual_factor = st.number_input("Manual defense factor (1.00 = neutral)",
                                        value=1.00, step=0.05, format="%.2f")

    def_factor = 1.0
    if def_df is not None and opp_team.strip():
        sub = def_df[def_df["team"].str.upper() == opp_team.strip().upper()]
        if not sub.empty:
            stat_col = {"pass_yds_pg": "pass_yds_pg",
                        "rush_yds_pg": "rush_yds_pg",
                        "rec_yds_pg": "rec_yds_pg"}[want_col]
            if stat_col in def_df.columns:
                opp_val = float(sub.iloc[0][stat_col])
                league_avg = float(def_df[stat_col].astype(float).mean())
                if league_avg > 0:
                    def_factor = opp_val / league_avg

    if def_df is None and defense_note:
        st.info(defense_note)

    blend = st.slider("Weight: Defense file vs Manual factor", 0, 100, 100,
                      help="0 = use manual only, 100 = use defense file only")
    use_factor = (def_factor * (blend / 100.0)) + (manual_factor * (1 - blend / 100.0))

    # ---- Prop inputs ----
    st.subheader("Prop inputs")
    prop_line = st.number_input("Prop line", value=float(np.round(base_y_pg, 1)), step=0.5, format="%.2f")
    sd = st.slider("Simulation SD (volatility)", min_value=5.0, max_value=80.0, value=25.0, step=0.5)

    # ---- Simulation ----
    if st.button("Compute Over%"):
        mean = max(0.0, base_y_pg * use_factor)
        sims = np.random.normal(loc=mean, scale=sd, size=10000)
        sims = np.clip(sims, 0, None)

        over_prob = float((sims > prop_line).mean())
        under_prob = 1.0 - over_prob

        def prob_to_ml(p):
            if p <= 0.0:
                return "∞"
            if p >= 0.5:
                return str(int(round(-100 * p / (1 - p))))   # favorite
            return f"+{int(round(100 * (1 - p) / p))}"       # underdog

        st.write("---")
        st.markdown(f"**{player_name}** — Team: **{team}** — Market: **{market}**")
        st.caption(f"Baseline Y/G: {base_y_pg:.2f} | Defense factor used: {use_factor:.2f} | Adjusted mean: {mean:.2f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Over %", f"{over_prob*100:.1f}%")
        c2.metric("Under %", f"{under_prob*100:.1f}%")
        c3.metric("Fair ML (Over)", prob_to_ml(over_prob))


# ========================= APP NAV ========================= #
st.title("Sports Model")

page = st.radio("Pick a page", ["NFL", "MLB", "NFL Props (beta)"], horizontal=True)

if page == "NFL":
    render_nfl_page()       # your original NFL page
elif page == "MLB":
    render_mlb_page()       # your original MLB page
else:
    render_nfl_props_page() # NEW props page
