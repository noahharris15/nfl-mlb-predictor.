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
    resp = requests.get(BIG5_PASS_TYPES_URL, headers={"User-Agent": USER_AGENT})
    if not resp.ok:
        raise FBrefScrapeError(
            f"FBref request failed ({resp.status_code}) for {BIG5_PASS_TYPES_URL}"
        )
    return resp.text

def _find_opponent_pass_types_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(html)
    candidate_indices = []

    for i, df in enumerate(tables):
        cols = [str(c) for c in df.columns]
        if "Squad" in cols and "Att" in cols:
            candidate_indices.append(i)

    if not candidate_indices:
        raise FBrefScrapeError("Could not find any table with columns 'Squad' and 'Att'.")

    idx = candidate_indices[1] if len(candidate_indices) >= 2 else candidate_indices[0]
    return tables[idx]

def _clean_opponent_pass_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(map(str, col)).strip() for col in df.columns]

    cols = list(df.columns)

    def _find_col(possible_names):
        for name in possible_names:
            for col in cols:
                if str(col).strip().lower() == name.lower():
                    return col
        raise FBrefScrapeError(f"Could not find any of columns: {possible_names}")

    squad_col = _find_col(["Squad"])
    comp_col  = _find_col(["Comp", "League"])
    att_col   = _find_col(["Att", "Att Passes", "AttPasses"])

    df = df.copy()
    df = df[~df[squad_col].isin(["Squad", "Opponent", "League Average"])]
    df = df[df[att_col].notna()]

    df[att_col] = pd.to_numeric(df[att_col], errors="coerce")
    df = df[df[att_col].notna()]

    cleaned = df[[comp_col, squad_col, att_col]].rename(
        columns={
            comp_col: "league",
            squad_col: "team",
            att_col: "passes_allowed_per_match",
        }
    )

    cleaned["league"] = cleaned["league"].astype(str).str.strip()
    cleaned["team"] = cleaned["team"].astype(str).str.strip()
    return cleaned

def _rank_within_league(df: pd.DataFrame) -> pd.DataFrame:
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

    return df.groupby("league", group_keys=False).apply(rank_group)

def get_big5_pass_defense() -> pd.DataFrame:
    html     = _fetch_big5_pass_types_html()
    table    = _find_opponent_pass_types_table(html)
    cleaned  = _clean_opponent_pass_df(table)
    ranked   = _rank_within_league(cleaned)
    return ranked

def get_big5_pass_defense_json():
    df = get_big5_pass_defense()
    return df.to_dict(orient="records")

if __name__ == "__main__":
    df = get_big5_pass_defense()
    for league in df["league"].unique():
        print(f"\n=== {league} — Toughest vs passes ===")
        sub = df[df["league"] == league].sort_values("rank_def_vs_passes").head(10)
        print(sub[["team", "passes_allowed_per_match", "rank_def_vs_passes"]])
