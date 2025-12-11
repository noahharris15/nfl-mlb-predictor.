import streamlit as st
from pass_defense_big5 import get_big5_pass_defense

st.title("Defense vs Passes — Big 5 Leagues")

df = get_big5_pass_defense()
league = st.selectbox("League", sorted(df["league"].unique()))
sub = df[df["league"] == league].sort_values("rank_def_vs_passes")

st.dataframe(sub)

team_name = st.text_input("Opponent team name (exact match):")
if team_name:
    row = sub[sub["team"].str.lower() == team_name.lower()]
    if not row.empty:
        r = row.iloc[0]
        st.write(f"**{r['team']}** — passes allowed per match: {r['passes_allowed_per_match']:.1f}")
        st.write(f"Rank vs passes in {league}: **{int(r['rank_def_vs_passes'])}**")
        st.write(f"Percentile (100 = toughest): **{r['def_vs_passes_pct']:.1f}**")
    else:
        st.write("Team not found in this league.")
        """
pass_defense_big5.py

Pulls Big 5 European leagues' "defense vs passes" (passes allowed)
from FBref and ranks teams within each league.

You can:
- import get_big5_pass_defense() in your backend / model, OR
- run `python pass_defense_big5.py` to print a preview.

Designed to be dropped into a website backend (FastAPI/Flask/Streamlit).
"""

import requests
import pandas as pd


BIG5_PASS_TYPES_URL = (
    "https://fbref.com/en/comps/Big5/passing_types/squads/Big-5-European-Leagues-Stats"
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class FBrefScrapeError(Exception):
    pass


def _fetch_big5_pass_types_html() -> str:
    """
    Fetch raw HTML for the Big 5 'Squad Pass Types' page.
    """
    resp = requests.get(BIG5_PASS_TYPES_URL, headers={"User-Agent": USER_AGENT})
    if not resp.ok:
        raise FBrefScrapeError(
            f"FBref request failed ({resp.status_code}) for {BIG5_PASS_TYPES_URL}"
        )
    return resp.text


def _find_opponent_pass_types_table(html: str) -> pd.DataFrame:
    """
    Use pandas.read_html to pull all tables, then heuristically find the
    'Squad Pass Types - Against' (opponent) table.

    FBref usually includes multiple tables (for + against).
    We pick a table that:
    - Has 'Squad' and 'Att' columns
    - Likely is the second one (first = squad, second = opponent)

    If this ever breaks, print the tables and manually inspect.
    """
    tables = pd.read_html(html)
    candidate_indices = []

    for i, df in enumerate(tables):
        cols = [str(c) for c in df.columns]
        if "Squad" in cols and "Att" in cols:
            candidate_indices.append(i)

    if not candidate_indices:
        raise FBrefScrapeError("Could not find any table with columns 'Squad' and 'Att'.")

    # Heuristic: second candidate is usually the opponent (vs) table.
    # If there is only one, use that one.
    if len(candidate_indices) >= 2:
        idx = candidate_indices[1]
    else:
        idx = candidate_indices[0]

    opponent_df = tables[idx]
    return opponent_df


def _clean_opponent_pass_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the opponent pass types table.

    Expected columns (may vary slightly, adjust if needed):
      - 'Squad'  : team name
      - 'Comp'   : competition / league name
      - 'Att'    : opponent passes attempted per squad match (per 90)
      - 'MP' or 'Matches' might exist but for pass types FBref usually
                     states "Stats are per squad match" so 'Att' is per match.

    This returns:
      league, team, passes_allowed_per_match
    """
    # Some tables have multi-index columns; flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in col]).strip() for col in df.columns]

    cols = list(df.columns)

    # Try to detect needed columns robustly
    # Column names may look like 'Squad', 'Comp', 'Att', etc.
    def _find_col(possible_names):
        for name in possible_names:
            for col in cols:
                if str(col).strip().lower() == name.lower():
                    return col
        raise FBrefScrapeError(f"Could not find any of columns: {possible_names}")

    squad_col = _find_col(["Squad"])
    comp_col = _find_col(["Comp", "League"])
    att_col = _find_col(["Att", "Att Passes", "AttPasses"])

    # Filter out non-team rows (like 'League Average' or summary lines)
    df = df.copy()
    df = df[~df[squad_col].isin(["Squad", "Opponent", "League Average"])]

    # Some rows might be totals/NaN; drop rows where Att is NaN
    df = df[df[att_col].notna()]

    # Convert Att to numeric
    df[att_col] = pd.to_numeric(df[att_col], errors="coerce")
    df = df[df[att_col].notna()]

    cleaned = df[[comp_col, squad_col, att_col]].rename(
        columns={
            comp_col: "league",
            squad_col: "team",
            att_col: "passes_allowed_per_match",
        }
    )

    # Strip whitespace
    cleaned["league"] = cleaned["league"].astype(str).str.strip()
    cleaned["team"] = cleaned["team"].astype(str).str.strip()

    return cleaned


def _rank_within_league(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - rank_def_vs_passes: 1 = fewest passes allowed (toughest vs passes)
      - def_vs_passes_pct : percentile (100 = toughest, 0 = easiest)
    """
    df = df.copy()

    def rank_group(group):
        group = group.sort_values("passes_allowed_per_match", ascending=True)
        group["rank_def_vs_passes"] = range(1, len(group) + 1)
        if len(group) > 1:
            group["def_vs_passes_pct"] = 100 * (
                1 - (group["rank_def_vs_passes"] - 1) / (len(group) - 1)
            )
        else:
            group["def_vs_passes_pct"] = 100.0
        return group

    ranked = df.groupby("league", group_keys=False).apply(rank_group)
    return ranked


def get_big5_pass_defense() -> pd.DataFrame:
    """
    Public function: returns a DataFrame with Big 5 leagues'
    defense vs passes.

    Columns:
      - league
      - team
      - passes_allowed_per_match
      - rank_def_vs_passes (1 = toughest vs passes)
      - def_vs_passes_pct  (100 = elite, 0 = very weak)
    """
    html = _fetch_big5_pass_types_html()
    opp_table = _find_opponent_pass_types_table(html)
    cleaned = _clean_opponent_pass_df(opp_table)
    ranked = _rank_within_league(cleaned)
    return ranked


# --------- JSON helper for your website / API --------- #

def get_big5_pass_defense_json():
    """
    Return the ranked table as a list of dicts (JSON-serializable).
    """
    df = get_big5_pass_defense()
    # You can filter to specific leagues here if you only care about EPL, etc.
    return df.to_dict(orient="records")


# --------- CLI preview (optional) --------- #

if __name__ == "__main__":
    df = get_big5_pass_defense()
    # Example: show top 10 toughest vs passes in each league
    for league in df["league"].unique():
        print(f"\n=== {league} — Toughest vs passes ===")
        sub = df[df["league"] == league].sort_values("rank_def_vs_passes").head(10)
        print(sub[["team", "passes_allowed_per_match", "rank_def_vs_passes"]])
