import io
import importlib
from io import BytesIO
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Small helpers
# ---------------------------

def pick_excel_engine() -> Optional[str]:
    for eng, mod in (("xlsxwriter", "xlsxwriter"), ("openpyxl", "openpyxl")):
        if importlib.util.find_spec(mod) is not None:
            return eng
    return None

def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    engine = pick_excel_engine()
    if engine is None:
        # graceful fallback (never crash Cloud)
        buff = BytesIO()
        buff.write(b"Excel writer not available. Add 'xlsxwriter' or 'openpyxl' to requirements.txt.\n")
        for name, df in sheets.items():
            buff.write(f"\n## {name}\n".encode())
            buff.write(df.to_csv(index=False).encode())
        return buff.getvalue()

    output = BytesIO()
    with pd.ExcelWriter(output, engine=engine) as writer:
        for name, df in sheets.items():
            sheet = name[:31].replace("/", "_")
            df.to_excel(writer, sheet_name=sheet, index=False)
    output.seek(0)
    return output.read()

def read_csv_resilient(uploaded, pasted: str) -> Optional[pd.DataFrame]:
    """
    Accepts either an uploaded file or pasted CSV text.
    Tries to skip weird header rows and '-additional' trailer.
    """
    try:
        if uploaded is not None:
            raw = uploaded.read().decode("utf-8", errors="ignore")
        elif pasted.strip():
            raw = pasted
        else:
            return None

        # Some of your CSVs have a first line like "Passing,Passing,...,-additional"
        # and sometimes a final trailing column "-9999". We clean both.
        # Find the real header line that starts with "Rk," or a column we recognize.
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        header_idx = 0
        for i, ln in enumerate(lines):
            if ln.startswith("Rk,") or ln.startswith("Rushing") or ln.startswith("Receiving") or ln.startswith("Passing"):
                # likely not the real header (category), keep scanning
                continue
            if ln.startswith("Rk,") or ln.startswith("Player,"):
                header_idx = i
                break
        candidate = "\n".join(lines[header_idx:])

        df = pd.read_csv(io.StringIO(candidate), engine="python")
        # Drop unnamed or bogus columns
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).startswith("-additional")]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        # Also drop the "-9999" column if present
        if "-9999" in df.columns:
            df = df.drop(columns=["-9999"], errors="ignore")

        # Strip whitespace from headers
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return None

def implied_odds(p: float) -> str:
    # returns American odds string for probability p (e.g., 0.55)
    if p <= 0 or p >= 1:
        return "–"
    if p > 0.5:
        # favorite (negative)
        return f"{int(round(-p/(1-p)*100)):+d}"
    else:
        # underdog (positive)
        return f"{int(round((1-p)/p*100)):+d}"

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

# ---------------------------
# Built-in defense starter (you can replace with a richer table)
# Columns expected: Team, pass_yds_pg, rush_yds_pg, rec_yds_pg, pass_td_pg, rush_td_pg
# ---------------------------
DEF_STARTER = pd.DataFrame({
    "Team": ["BUF","MIA","CIN","WAS","SEA","NYJ","CLE","DET","LAC","PHI","BAL","IND","JAX","GNB","KAN","NWE","TAM","MIN",
             "ARI","DEN","ATL","DAL","NOR","PIT","LVR","LAR","HOU","TEN","CHI","NYG","CAR","SFO"],
    # Placeholder league-average-ish values; replace with your real 2025 table when you have it
    "pass_yds_pg": [215]*32,
    "rush_yds_pg": [110]*32,
    "rec_yds_pg":  [215]*32,
    "pass_td_pg":  [1.4]*32,
    "rush_td_pg":  [0.8]*32
})

# ---------------------------
# App layout
# ---------------------------

st.set_page_config(page_title="NFL + MLB Predictor — 2025", layout="wide")
st.title("🏈⚾ NFL + MLB Predictor — 2025")

tabs = st.tabs(["NFL (teams)", "MLB (teams)", "NFL Props (CSV, beta)"])

# ---------------------------
# Tab 1: NFL team win model (PF/PA)
# ---------------------------
with tabs[0]:
    st.subheader("NFL — quick win probability (PF/PA only)")
    colA, colB = st.columns(2)
    with colA:
        pf_a = st.number_input("Team A: Points For per game", value=24.0, step=0.1)
        pa_a = st.number_input("Team A: Points Against per game", value=21.0, step=0.1)
    with colB:
        pf_b = st.number_input("Team B: Points For per game", value=23.0, step=0.1)
        pa_b = st.number_input("Team B: Points Against per game", value=22.0, step=0.1)
    # Simple Elo-ish net rate
    net_a = pf_a - pa_a
    net_b = pf_b - pa_b
    edge = net_a - net_b
    # map to probability using logistic
    p_a = 1/(1+np.exp(-edge/6.0))
    st.metric("Win % (Team A)", f"{p_a*100:,.1f}%")
    st.caption("Toy model using only PF - PA. No injuries, travel, or market inputs.")

# ---------------------------
# Tab 2: MLB team win model (RS/RA)
# ---------------------------
with tabs[1]:
    st.subheader("MLB — quick win probability (RS/RA only)")
    colA, colB = st.columns(2)
    with colA:
        rs_a = st.number_input("Team A: Runs Scored per game", value=4.6, step=0.01, key="rs_a")
        ra_a = st.number_input("Team A: Runs Allowed per game", value=4.3, step=0.01, key="ra_a")
    with colB:
        rs_b = st.number_input("Team B: Runs Scored per game", value=4.5, step=0.01, key="rs_b")
        ra_b = st.number_input("Team B: Runs Allowed per game", value=4.4, step=0.01, key="ra_b")
    net_a = rs_a - ra_a
    net_b = rs_b - ra_b
    edge = net_a - net_b
    p_a = 1/(1+np.exp(-edge/0.6))
    st.metric("Win % (Team A)", f"{p_a*100:,.1f}%")
    st.caption("Pythagorean-ish quick model. Keep your original MLB page if you already have one.")

# ---------------------------
# Tab 3: NFL Player Props (CSV)
# ---------------------------
with tabs[2]:
    st.header("🏈 NFL Player Props — 2025 (CSV)")

    st.markdown("Upload or paste your 2025 CSVs (same ones you shared). "
                "I’ll parse them, standardize stats per game, adjust by opponent defense, "
                "simulate, and let you download an Excel with all sheets.")

    st.markdown("**1) Load Player CSVs**")
    col1, col2 = st.columns(2)
    with col1:
        qb_file = st.file_uploader("QBs CSV (upload)", type=["csv"])
        qb_paste = st.text_area("...or paste QBs CSV here", height=120)
        rb_file = st.file_uploader("RBs CSV (upload)", type=["csv"])
        rb_paste = st.text_area("...or paste RBs CSV here", height=120)
        wr_file = st.file_uploader("WRs CSV (upload)", type=["csv"])
        wr_paste = st.text_area("...or paste WRs CSV here", height=120)
    with col2:
        def_file = st.file_uploader("Defense Strength CSV (optional)", type=["csv"])
        def_paste = st.text_area("...or paste Defense CSV here (optional)", height=120)
        st.caption("Expected columns: Team, pass_yds_pg, rush_yds_pg, rec_yds_pg, pass_td_pg, rush_td_pg. "
                   "If omitted, a starter table is used.")

    df_qb = read_csv_resilient(qb_file, qb_paste) or pd.DataFrame()
    df_rb = read_csv_resilient(rb_file, rb_paste) or pd.DataFrame()
    df_wr = read_csv_resilient(wr_file, wr_paste) or pd.DataFrame()

    df_def = read_csv_resilient(def_file, def_paste)
    if df_def is None or df_def.empty:
        df_def = DEF_STARTER.copy()

    # Show brief previews
    st.markdown("**2) Parsed previews**")
    pv1, pv2, pv3, pv4 = st.columns(4)
    with pv1: st.dataframe(df_qb.head(8), use_container_width=True)
    with pv2: st.dataframe(df_rb.head(8), use_container_width=True)
    with pv3: st.dataframe(df_wr.head(8), use_container_width=True)
    with pv4: st.dataframe(df_def.head(8), use_container_width=True)

    # ---------------------------
    # Standardize stat columns (per-game baselines)
    # ---------------------------
    def std_qb(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Try to detect columns from your CSV
        # Passing yards per game
        if "Y/G" in out.columns:
            out["pass_yds_pg"] = pd.to_numeric(out["Y/G"], errors="coerce")
        elif "Yds" in out.columns and "G" in out.columns:
            out["pass_yds_pg"] = pd.to_numeric(out["Yds"], errors="coerce") / pd.to_numeric(out["G"], errors="coerce")
        else:
            out["pass_yds_pg"] = np.nan

        # Passing TDs: per game if possible
        if "TD" in out.columns and "G" in out.columns:
            out["pass_td_pg"] = pd.to_numeric(out["TD"], errors="coerce") / pd.to_numeric(out["G"], errors="coerce")
        elif "TD" in out.columns:
            out["pass_td_pg"] = pd.to_numeric(out["TD"], errors="coerce")  # assume per game already
        else:
            out["pass_td_pg"] = np.nan

        # Team column
        team_col = "Team" if "Team" in out.columns else ("Tm" if "Tm" in out.columns else None)
        if team_col:
            out["Team"] = out[team_col]
        out["Player"] = out["Player"]
        return out[["Player","Team","pass_yds_pg","pass_td_pg"]].dropna(how="all")

    def std_rb(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "Y/G" in out.columns:
            out["rush_yds_pg"] = pd.to_numeric(out["Y/G"], errors="coerce")
        elif "Yds" in out.columns and "G" in out.columns:
            out["rush_yds_pg"] = pd.to_numeric(out["Yds"], errors="coerce") / pd.to_numeric(out["G"], errors="coerce")
        else:
            out["rush_yds_pg"] = np.nan
        if "TD" in out.columns and "G" in out.columns:
            out["rush_td_pg"] = pd.to_numeric(out["TD"], errors="coerce") / pd.to_numeric(out["G"], errors="coerce")
        elif "TD" in out.columns:
            out["rush_td_pg"] = pd.to_numeric(out["TD"], errors="coerce")
        else:
            out["rush_td_pg"] = np.nan
        team_col = "Team" if "Team" in out.columns else ("Tm" if "Tm" in out.columns else None)
        if team_col:
            out["Team"] = out[team_col]
        out["Player"] = out["Player"]
        return out[["Player","Team","rush_yds_pg","rush_td_pg"]].dropna(how="all")

    def std_wr(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Receiving Y/G
        if "Y/G" in out.columns:
            out["rec_yds_pg"] = pd.to_numeric(out["Y/G"], errors="coerce")
        elif "Yds" in out.columns and "G" in out.columns:
            out["rec_yds_pg"] = pd.to_numeric(out["Yds"], errors="coerce") / pd.to_numeric(out["G"], errors="coerce")
        else:
            out["rec_yds_pg"] = np.nan
        # Receptions per game (Rec / G if available)
        if "Rec" in out.columns and "G" in out.columns:
            out["rec_pg"] = pd.to_numeric(out["Rec"], errors="coerce") / pd.to_numeric(out["G"], errors="coerce")
        elif "Rec" in out.columns:
            out["rec_pg"] = pd.to_numeric(out["Rec"], errors="coerce")
        else:
            out["rec_pg"] = np.nan
        # TD per game
        if "TD" in out.columns and "G" in out.columns:
            out["rec_td_pg"] = pd.to_numeric(out["TD"], errors="coerce") / pd.to_numeric(out["G"], errors="coerce")
        elif "TD" in out.columns:
            out["rec_td_pg"] = pd.to_numeric(out["TD"], errors="coerce")
        else:
            out["rec_td_pg"] = np.nan
        team_col = "Team" if "Team" in out.columns else ("Tm" if "Tm" in out.columns else None)
        if team_col:
            out["Team"] = out[team_col]
        out["Player"] = out["Player"]
        return out[["Player","Team","rec_pg","rec_yds_pg","rec_td_pg"]].dropna(how="all")

    qb_std = std_qb(df_qb) if not df_qb.empty else pd.DataFrame(columns=["Player","Team","pass_yds_pg","pass_td_pg"])
    rb_std = std_rb(df_rb) if not df_rb.empty else pd.DataFrame(columns=["Player","Team","rush_yds_pg","rush_td_pg"])
    wr_std = std_wr(df_wr) if not df_wr.empty else pd.DataFrame(columns=["Player","Team","rec_pg","rec_yds_pg","rec_td_pg"])

    # ---------------------------
    # Market + Player selection
    # ---------------------------
    st.markdown("**3) Pick a market**")
    market = st.selectbox(
        "Market",
        ["Pass Yds", "Pass TDs", "Rush Yds", "Rush TDs", "Rec Yds", "Receptions", "Rec TDs"],
        index=0
    )

    # Pool players based on market
    if market in ["Pass Yds", "Pass TDs"]:
        pool = qb_std.copy()
    elif market in ["Rush Yds", "Rush TDs"]:
        pool = rb_std.copy()
    else:  # receiving
        pool = wr_std.copy()

    player_name = st.selectbox("Player (must exist in parsed table)", sorted(pool["Player"].unique()) if not pool.empty else [])
    opp_team = st.selectbox("Opponent Defense (Team code)", sorted(df_def["Team"].unique()))
    prop_line = st.number_input("Prop line", value=250.0 if market=="Pass Yds" else (65.0 if market in ["Rush Yds","Rec Yds"] else 1.5), step=0.5)
    sd = st.slider("Simulation SD (volatility)", min_value=5.0, max_value=60.0, value=25.0, step=0.5)

    # ---------------------------
    # Get player baseline + defense adjustment
    # ---------------------------
    def league_avg(col):
        if col in df_def.columns:
            return pd.to_numeric(df_def[col], errors="coerce").mean()
        return np.nan

    def_def_cols = {
        "Pass Yds": "pass_yds_pg",
        "Pass TDs": "pass_td_pg",
        "Rush Yds": "rush_yds_pg",
        "Rush TDs": "rush_td_pg",
        "Rec Yds":  "rec_yds_pg",
        "Receptions":"rec_yds_pg",   # no direct receptions defense; proxy with rec yards defense
        "Rec TDs":  "pass_td_pg"     # proxy: receiving TDs correlate with pass TDs allowed
    }

    # pull defense metric
    def_col = def_def_cols.get(market)
    opp_row = df_def.loc[df_def["Team"] == opp_team]
    opp_metric = safe_float(opp_row[def_col].iloc[0]) if not opp_row.empty and def_col in df_def.columns else None
    lg_metric = league_avg(def_col)

    # player baseline
    base = np.nan
    if not pool.empty and player_name:
        row = pool.loc[pool["Player"] == player_name].head(1)
        if not row.empty:
            if market == "Pass Yds":
                base = safe_float(row["pass_yds_pg"].iloc[0])
            elif market == "Pass TDs":
                base = safe_float(row["pass_td_pg"].iloc[0])
            elif market == "Rush Yds":
                base = safe_float(row["rush_yds_pg"].iloc[0])
            elif market == "Rush TDs":
                base = safe_float(row["rush_td_pg"].iloc[0])
            elif market == "Rec Yds":
                base = safe_float(row["rec_yds_pg"].iloc[0])
            elif market == "Receptions":
                base = safe_float(row["rec_pg"].iloc[0])
            elif market == "Rec TDs":
                base = safe_float(row["rec_td_pg"].iloc[0])

    # Adjust baseline by defense factor (simple ratio: easier D -> increase, tougher -> decrease)
    adj_base = base
    if base is not None and opp_metric and lg_metric and np.isfinite(base) and opp_metric > 0 and lg_metric > 0:
        factor = float(opp_metric) / float(lg_metric)
        # yards/receptions scale linearly, TDs scale more conservatively
        if "TD" in market:
            adj_base = base * (0.5 + 0.5 * factor)   # dampen effect
        else:
            adj_base = base * factor

    # ---------------------------
    # Simulation
    # ---------------------------
    sim_btn = st.button("Compute Over %")
    over_pct = None
    sims_df = pd.DataFrame()
    if sim_btn and pd.notna(adj_base):
        n = 50000
        draws = np.random.normal(loc=adj_base, scale=sd, size=n)
        # Truncate at zero for counts/yards
        draws = np.clip(draws, a_min=0, a_max=None)
        over_pct = (draws > prop_line).mean()
        under_pct = 1 - over_pct

        st.subheader("Results")
        st.metric("Over %", f"{over_pct*100:,.1f}%")
        st.metric("Under %", f"{under_pct*100:,.1f}%")
        st.metric("Fair ML (Over)", implied_odds(over_pct))

        sims_df = pd.DataFrame({"sim_value": draws})
        st.caption(f"Mean (adj.): {adj_base:,.2f} | SD: {sd:,.2f} | n={len(draws):,}")

    # ---------------------------
    # Download Excel (all sheets)
    # ---------------------------
    st.markdown("---")
    st.subheader("Download all sheets (Excel)")
    sheets = {}
    if not df_qb.empty: sheets["QBs"] = df_qb
    if not df_rb.empty: sheets["RBs"] = df_rb
    if not df_wr.empty: sheets["WRs"] = df_wr
    if not df_def.empty: sheets["Defense"] = df_def

    if not pool.empty and player_name:
        meta = pd.DataFrame([{
            "Market": market,
            "Player": player_name,
            "Opponent": opp_team,
            "Prop": prop_line,
            "Base": base,
            "AdjBase": adj_base,
            "SD": sd,
            "Over%": over_pct if over_pct is not None else np.nan,
            "FairOdds(Over)": implied_odds(over_pct) if over_pct is not None else ""
        }])
        sheets["Player Prop (meta)"] = meta
    if not sims_df.empty:
        # Keep sims smaller when downloading (optional downsample)
        sheets["Player Sims"] = sims_df.sample(min(5000, len(sims_df)), random_state=1)

    excel_bytes = to_excel_bytes(sheets) if sheets else b""
    st.download_button(
        "Download results (Excel)",
        data=excel_bytes,
        file_name="nfl_player_props.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=not bool(sheets)
    )

    with st.expander("What this uses"):
        st.write("""
- Your uploaded/pasted **QBs / RBs / WRs** CSVs (like the 2025 tables you sent).
- A defense table (uploaded or a built-in starter). We scale a player's baseline by the ratio:
  `Opponent metric / League average metric` (yards and receptions scale fully; TDs are damped).
- Monte-Carlo (Normal) with your **SD slider** to get Over/Under % and implied fair odds.
- A robust Excel writer (uses `xlsxwriter` or `openpyxl` if available; otherwise a safe text fallback).
        """)
