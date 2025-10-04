import streamlit as st
import pandas as pd
import numpy as np
import requests

# Load secrets
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", None)
CFBD_API_KEY = st.secrets.get("CFBD_API_KEY", None)

st.set_page_config(page_title="Sports Predictor", layout="wide")

# -----------------------------
# College Football Data Function
# -----------------------------
@st.cache_data(show_spinner=False)
def cfb_team_stats_2025() -> pd.DataFrame:
    """
    Robust pull from CollegeFootballData for 2025.
    Normalized to: team, off_ppg, def_ppg
    """
    if not CFBD_API_KEY:
        return pd.DataFrame({"team": [], "off_ppg": [], "def_ppg": []})

    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}

    def fetch_stat(year: int, wanted: str, extra_qs: dict | None = None) -> pd.DataFrame:
        qs = {"year": str(year)}
        if extra_qs:
            qs.update(extra_qs)
        url = "https://api.collegefootballdata.com/stats/season"
        r = requests.get(url, headers=headers, params=qs, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "statName" not in df.columns or "team" not in df.columns or "statValue" not in df.columns:
            return pd.DataFrame()
        mask = df["statName"].astype(str).str.lower().str.contains(wanted.lower())
        df = df.loc[mask, ["team", "statValue"]].rename(columns={"statValue": wanted})
        return df

    # Try to get Offense PPG
    off_df = fetch_stat(2025, "Points Per Game")
    if not off_df.empty:
        off_df = off_df.rename(columns={"Points Per Game": "off_ppg"})

    # Try to get Opponent PPG
    def_df = fetch_stat(2025, "Opponent Points Per Game")
    if not def_df.empty:
        def_df = def_df.rename(columns={"Opponent Points Per Game": "def_ppg"})

    # Handle missing cases
    if off_df.empty and def_df.empty:
        return pd.DataFrame({"team": [], "off_ppg": [], "def_ppg": []})

    if def_df.empty:
        off_df["def_ppg"] = off_df["off_ppg"].median()
        return off_df.rename(columns={"team": "team"})[["team", "off_ppg", "def_ppg"]]

    if off_df.empty:
        def_df["off_ppg"] = def_df["def_ppg"].median()
        return def_df.rename(columns={"team": "team"})[["team", "off_ppg", "def_ppg"]]

    # Merge both
    df = pd.merge(off_df, def_df, on="team", how="inner")
    df["off_ppg"] = pd.to_numeric(df["off_ppg"], errors="coerce")
    df["def_ppg"] = pd.to_numeric(df["def_ppg"], errors="coerce")
    return df.dropna().sort_values("team").reset_index(drop=True)


# -----------------------------
# College Football Page
# -----------------------------
def college_football_page():
    st.subheader("🏈🎓 College Football — 2025 (auto from CFBD)")

    if st.button("Clear CFB cache"):
        cfb_team_stats_2025.clear()
        st.success("Cleared CFB cache. Re-run the app.")

    df = cfb_team_stats_2025()
    if df.empty:
        st.error("No CFB stats available. Check your CFBD_API_KEY or API limits.")
        return

    home = st.selectbox("Home team", df["team"].unique())
    away = st.selectbox("Away team", df["team"].unique())

    if home and away:
        home_row = df.loc[df["team"] == home].iloc[0]
        away_row = df.loc[df["team"] == away].iloc[0]

        home_pts = (home_row["off_ppg"] + away_row["def_ppg"]) / 2
        away_pts = (away_row["off_ppg"] + home_row["def_ppg"]) / 2

        total = home_pts + away_pts
        p_home = home_pts / total
        p_away = away_pts / total

        st.markdown(
            f"**{home} vs {away}** — Expected points: {home_pts:.1f}–{away_pts:.1f} "
            f"· P({home} win) = {p_home:.1%}, P({away} win) = {p_away:.1%}"
        )

    with st.expander("Show team table"):
        st.dataframe(df)


# -----------------------------
# Main Navigation
# -----------------------------
page = st.radio("Pick a page", ["NFL", "MLB", "College Football", "Player Props"])

if page == "College Football":
    college_football_page()
elif page == "NFL":
    st.write("NFL page placeholder...")
elif page == "MLB":
    st.write("MLB page placeholder...")
elif page == "Player Props":
    st.write("Player Props page placeholder...")
