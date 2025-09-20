# app.py
# -------------------------------------------------------
# Tabs:
#   1) NFL (legacy) – simple CSV viewer (paste OR upload)
#   2) MLB (legacy) – simple CSV viewer (paste OR upload)
#   3) NFL Props    – paste/upload QB, RB, WR (+ optional DEF),
#                     pick player & market, enter a prop line,
#                     auto-adjust for defense, simulate (no extra sliders)
#
# Requirements: streamlit, pandas, numpy
# -------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sports Model", layout="centered")

# ---------- Helpers ----------

def _clean_pasted_csv(text: str) -> str:
    """Drop banner rows like 'Receiving,Receiving,...,-additional' and a trailing sentinel column."""
    if not text:
        return text
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # nix leading banner line
    if lines:
        toks = [t.strip() for t in lines[0].split(",") if t.strip()]
        if len(set(toks)) == 1 or (len(toks) > 2 and toks[-1].lower() == "-additional"):
            lines = lines[1:]

    # if header ends with a sentinel (e.g., -9999), drop last col across file
    if lines:
        hdr = [t.strip() for t in lines[0].split(",")]
        if hdr and (hdr[-1].startswith("-") or hdr[-1].isdigit()):
            keep = len(hdr) - 1
            lines = [",".join(row.split(",")[:keep]) for row in lines]

    return "\n".join(lines)


def _read_from_uploader_or_paste(label: str, key_prefix: str, help_text: str = "") -> pd.DataFrame | None:
    c1, c2 = st.columns([1, 1])
    with c1:
        f = st.file_uploader(f"{label} – Upload CSV", type=["csv"], key=f"{key_prefix}_file", help=help_text)
    with c2:
        txt = st.text_area(f"{label} – Or paste CSV text", height=180, key=f"{key_prefix}_paste", help=help_text)

    if f is not None:
        try:
            return pd.read_csv(io.StringIO(f.getvalue().decode("utf-8")), engine="python")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            return None

    if txt and txt.strip():
        try:
            cleaned = _clean_pasted_csv(txt)
            return pd.read_csv(io.StringIO(cleaned), engine="python")
        except Exception as e:
            st.error(f"Could not parse pasted CSV: {e}")
            return None

    return None


def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    drop = [c for c in df.columns if c == "" or c.lower().startswith("unnamed")]
    if drop:
        df = df.drop(columns=drop)

    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    ren = {}
    mapping = {
        "Player": pick("player", "name"),
        "Team": pick("team", "tm"),
        "Pos": pick("pos", "position"),
        "G": pick("g", "games"),
        "Season": pick("season", "yr", "year"),
        "Yds": pick("yds", "yards", "pass yds", "rush yds", "rec yds"),
        "Y_per_G": pick("y/g", "yds/g", "ypg", "yards/g"),
    }
    for canon, src in mapping.items():
        if src and src != canon:
            ren[src] = canon
    df = df.rename(columns=ren)

    if "Y_per_G" not in df.columns:
        if {"Yds", "G"}.issubset(df.columns):
            y = pd.to_numeric(df["Yds"], errors="coerce")
            g = pd.to_numeric(df["G"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                df["Y_per_G"] = y / g
        else:
            df["Y_per_G"] = np.nan
    return df


def _ensure_pos(df: pd.DataFrame, fallback_pos: str) -> pd.DataFrame:
    df = df.copy()
    if "Pos" not in df.columns or df["Pos"].isna().all():
        df["Pos"] = fallback_pos
    return df


def _fair_ml_from_prob(p: float) -> str:
    if p <= 0: return "∞"
    if p >= 0.5:
        return str(int(round(-100 * p / (1 - p))))
    return f"+{int(round(100 * (1 - p) / p))}"


# ---------- Pages ----------

def page_nfl_legacy():
    st.header("NFL (legacy)")
    st.caption("Paste or upload your CSV. This page is unchanged.")
    df = _read_from_uploader_or_paste("NFL CSV", "nfl_legacy")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Upload or paste your NFL CSV to view it here.")


def page_mlb_legacy():
    st.header("MLB (legacy)")
    st.caption("Paste or upload your CSV. This page is unchanged.")
    df = _read_from_uploader_or_paste("MLB CSV", "mlb_legacy")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Upload or paste your MLB CSV to view it here.")


def page_nfl_props_simple():
    st.header("🏈 NFL Player Props — Simple")

    st.markdown("#### 1) Load your player CSVs")
    qbs = _read_from_uploader_or_paste("QBs CSV", "qbs", "Paste the big list you have, or upload a CSV.")
    rbs = _read_from_uploader_or_paste("RBs CSV", "rbs", "Paste or upload.")
    wrs = _read_from_uploader_or_paste("WRs CSV", "wrs", "Paste or upload.")

    st.markdown("#### 2) (Optional) Defense CSV")
    st.caption("Columns: Team, pass_yds_pg, rush_yds_pg, rec_yds_pg  (yards **allowed** per game).")
    def_df = _read_from_uploader_or_paste("Defense CSV (optional)", "def")

    frames = []
    if qbs is not None: frames.append(_ensure_pos(_standardize_cols(qbs), "QB"))
    if rbs is not None: frames.append(_ensure_pos(_standardize_cols(rbs), "RB"))
    if wrs is not None: frames.append(_ensure_pos(_standardize_cols(wrs), "WR"))

    if not frames:
        st.info("Upload/paste at least one of QBs/RBs/WRs to continue.")
        return

    players = pd.concat(frames, ignore_index=True)
    keep = [c for c in ["Player", "Team", "Pos", "Season", "G", "Yds", "Y_per_G"] if c in players.columns]
    players = players[keep].copy()

    # Normalize defense columns if provided
    defense = None
    if def_df is not None:
        d = def_df.copy()
        d.columns = [c.strip().lower().replace(" ", "") for c in d.columns]
        ren = {}
        # accepted aliases
        ali = {
            "team": ["team", "tm", "def", "opp", "opponent"],
            "pass_yds_pg": ["passydspg", "passyardsallowedpg", "passyardspergame", "passyards_pg"],
            "rush_yds_pg": ["rushydspg", "rushyardsallowedpg", "rushyardspergame", "rushyards_pg"],
            "rec_yds_pg":  ["recydspg", "recyardsallowedpg", "receivingyardspergame", "recyards_pg"],
        }
        for want, options in ali.items():
            for o in options:
                if o in d.columns:
                    ren[o] = want
                    break
        d = d.rename(columns=ren)
        if "team" in d.columns:
            defense = d

    # ----- Pick market & player -----
    st.markdown("#### 3) Pick market & player")
    market = st.selectbox("Market", ["Pass Yds (QB)", "Rush Yds (RB)", "Rec Yds (WR)"])

    if market == "Pass Yds (QB)":
        pool = players[players["Pos"] == "QB"].copy()
        def_col = "pass_yds_pg"
        auto_sd = 40.0
    elif market == "Rush Yds (RB)":
        pool = players[players["Pos"] == "RB"].copy()
        def_col = "rush_yds_pg"
        auto_sd = 25.0
    else:
        pool = players[players["Pos"] == "WR"].copy()
        def_col = "rec_yds_pg"
        auto_sd = 30.0

    if pool.empty:
        st.warning("No players for that market in your uploads. Switch market or upload that position.")
        return

    player_name = st.selectbox("Player", sorted(pool["Player"].dropna().astype(str).unique().tolist()))
    row = pool.loc[pool["Player"] == player_name].iloc[0]
    base_mean = float(pd.to_numeric(row.get("Y_per_G", 0.0), errors="coerce") or 0.0)
    team = str(row.get("Team", ""))

    # ----- Opponent & defense factor -----
    st.markdown("#### 4) Opponent")
    if defense is not None:
        opp = st.selectbox("Opponent team (from Defense CSV)",
                           [""] + sorted(defense["team"].dropna().astype(str).str.upper().unique().tolist()))
    else:
        opp = st.text_input("Opponent team code (e.g., NYJ, DAL)", "")

    # compute defense factor automatically (no slider). If not available => 1.00
    def_factor = 1.0
    if defense is not None and opp:
        sub = defense[defense["team"].str.upper() == opp.upper()]
        if not sub.empty and def_col in sub.columns:
            try:
                opp_val = float(sub.iloc[0][def_col])
                league_avg = float(pd.to_numeric(defense[def_col], errors="coerce").mean())
                if league_avg > 0:
                    def_factor = opp_val / league_avg
            except Exception:
                pass

    # ----- Prop input, then simulate with fixed SD -----
    st.markdown("#### 5) Prop input")
    prop_line = st.number_input("Prop line", value=float(np.round(base_mean, 1)), step=0.5, format="%.2f")

    if st.button("Compute Over%"):
        adjusted_mean = max(0.0, base_mean * def_factor)

        # Fixed, market-specific SD (no user control)
        sd = auto_sd

        sims = np.random.normal(loc=adjusted_mean, scale=sd, size=10000)
        sims = np.clip(sims, 0, None)

        over_p = float((sims > prop_line).mean())
        under_p = 1.0 - over_p

        st.write("---")
        st.subheader(f"{player_name} — {market}")
        st.caption(
            f"Team: {team} | Baseline Y/G: {base_mean:.2f} | "
            f"Defense factor: {def_factor:.2f} | Adjusted mean: {adjusted_mean:.2f} | SD used: {sd:.1f}"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Over %", f"{over_p*100:.1f}%")
        c2.metric("Under %", f"{under_p*100:.1f}%")
        c3.metric("Fair ML (Over)", _fair_ml_from_prob(over_p))


# ---------- Navigation ----------
st.title("Sports Model")

page = st.radio("Pick a page", ["NFL (legacy)", "MLB (legacy)", "NFL Props"], horizontal=True)

if page == "NFL (legacy)":
    page_nfl_legacy()
elif page == "MLB (legacy)":
    page_mlb_legacy()
else:
    page_nfl_props_simple()
