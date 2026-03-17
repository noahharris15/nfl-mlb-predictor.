"""
NBA Over/Under Model (DATA ONLY)
- Pulls completed current-season games from nba_api
- Builds rolling team features using only prior games
- Projects team scores and game totals using:
    * points for / against
    * offensive/defensive efficiency proxies
    * pace
    * recent form
    * home/away splits
- Pulls live totals from The Odds API
- Compares model projected total vs market total
- Outputs edge table to CSV

INSTALL
pip install pandas numpy requests nba_api

OPTIONAL ENV VAR
Windows PowerShell:
$env:ODDS_API_KEY="YOUR_KEY"

Mac/Linux:
export ODDS_API_KEY="YOUR_KEY"
"""

from __future__ import annotations

import os
import time
import math
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# CONFIG
# -----------------------------
SEASON = "2025-26"          # Change if needed
SEASON_TYPE = "Regular Season"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()

SPORT_KEY = "basketball_nba"
REGIONS = "us"
MARKETS = "totals"
ODDS_FORMAT = "american"

# Rolling windows
RECENT_GAMES = 10           # recent form window
MIN_GAMES_REQUIRED = 8      # minimum completed games before using team in projections

# Blending weights for the projection
W_SEASON = 0.55
W_RECENT = 0.25
W_HOME_AWAY = 0.20

# Regression / shrinkage toward league average
REGRESS_FACTOR = 0.18

# Pace + efficiency weights
PACE_WEIGHT = 0.60
OFFENSE_WEIGHT = 0.55
DEFENSE_WEIGHT = 0.45

# Output
OUTPUT_CSV = "nba_ou_model_output.csv"

# nba_api request throttling
API_SLEEP_SECONDS = 0.7


# -----------------------------
# HELPERS
# -----------------------------
def american_to_implied_prob(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def safe_mean(values: List[float], default: float = np.nan) -> float:
    vals = [v for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else default


def extract_matchup_teams(matchup: str) -> Tuple[str, bool]:
    """
    Returns opponent abbreviation and whether team is home.
    Examples:
      'LAL vs. BOS' -> ('BOS', True)
      'LAL @ BOS'   -> ('BOS', False)
    """
    if " vs. " in matchup:
        left, right = matchup.split(" vs. ")
        return right.strip(), True
    if " @ " in matchup:
        left, right = matchup.split(" @ ")
        return right.strip(), False
    return "", False


def get_team_meta() -> pd.DataFrame:
    t = pd.DataFrame(teams.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "abbreviation": "TEAM_ABBREVIATION", "full_name": "TEAM_NAME"})
    return t[["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]].copy()


def fetch_league_games(season: str, season_type: str) -> pd.DataFrame:
    """
    Pull all games for the season, then dedupe to team-game rows.
    leaguegamefinder returns one row per team per game.
    """
    lgf = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00"
    )
    time.sleep(API_SLEEP_SECONDS)

    df = lgf.get_data_frames()[0].copy()
    if df.empty:
        raise RuntimeError("No NBA games returned from leaguegamefinder.")

    team_meta = get_team_meta()
    df = df.merge(team_meta, on=["TEAM_ID", "TEAM_ABBREVIATION"], how="left", suffixes=("", "_META"))

    # Parse dates and keep key cols
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
    df["FGA"] = pd.to_numeric(df["FGA"], errors="coerce")
    df["FGM"] = pd.to_numeric(df["FGM"], errors="coerce")
    df["FG3A"] = pd.to_numeric(df["FG3A"], errors="coerce")
    df["FG3M"] = pd.to_numeric(df["FG3M"], errors="coerce")
    df["FTA"] = pd.to_numeric(df["FTA"], errors="coerce")
    df["FTM"] = pd.to_numeric(df["FTM"], errors="coerce")
    df["OREB"] = pd.to_numeric(df["OREB"], errors="coerce")
    df["DREB"] = pd.to_numeric(df["DREB"], errors="coerce")
    df["REB"] = pd.to_numeric(df["REB"], errors="coerce")
    df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
    df["STL"] = pd.to_numeric(df["STL"], errors="coerce")
    df["BLK"] = pd.to_numeric(df["BLK"], errors="coerce")
    df["TOV"] = pd.to_numeric(df["TOV"], errors="coerce")
    df["PLUS_MINUS"] = pd.to_numeric(df["PLUS_MINUS"], errors="coerce")

    # Derive home/away + opponent
    parsed = df["MATCHUP"].apply(extract_matchup_teams)
    df["OPP_ABBREVIATION"] = parsed.apply(lambda x: x[0])
    df["IS_HOME"] = parsed.apply(lambda x: x[1]).astype(int)

    # Possessions estimate
    # Poss = FGA - OREB + TOV + 0.44*FTA
    df["POSS_EST"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]

    # Sort
    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return df


def attach_opponent_stats(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Join each team-game row to the opponent's row from the same GAME_ID.
    """
    opp = team_games[[
        "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "PTS", "POSS_EST", "IS_HOME", "PLUS_MINUS"
    ]].copy()

    opp = opp.rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "TEAM_ABBREVIATION": "OPP_ABBR_ROW",
        "TEAM_NAME": "OPP_TEAM_NAME",
        "PTS": "OPP_PTS_ROW",
        "POSS_EST": "OPP_POSS_ROW",
        "IS_HOME": "OPP_IS_HOME",
        "PLUS_MINUS": "OPP_PLUS_MINUS"
    })

    merged = team_games.merge(opp, on="GAME_ID", how="left")

    # Keep only opponent row, not same team row
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()

    # Team-level efficiencies for that game
    merged["OFF_RTG_G"] = (100.0 * merged["PTS"] / merged["POSS_EST"]).replace([np.inf, -np.inf], np.nan)
    merged["DEF_RTG_G"] = (100.0 * merged["OPP_PTS_ROW"] / merged["OPP_POSS_ROW"]).replace([np.inf, -np.inf], np.nan)

    # Approx game pace = average possessions by both teams
    merged["PACE_G"] = ((merged["POSS_EST"] + merged["OPP_POSS_ROW"]) / 2.0).replace([np.inf, -np.inf], np.nan)

    return merged.reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team game row, compute rolling prior-only features.
    """
    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()

    group = df.groupby("TEAM_ID", group_keys=False)

    # Prior-only expanding means
    for col in ["PTS", "OPP_PTS_ROW", "OFF_RTG_G", "DEF_RTG_G", "PACE_G"]:
        df[f"{col}_SEASON_PRIOR"] = group[col].apply(lambda s: s.shift(1).expanding().mean())

    # Prior-only recent window means
    for col in ["PTS", "OPP_PTS_ROW", "OFF_RTG_G", "DEF_RTG_G", "PACE_G"]:
        df[f"{col}_RECENT_PRIOR"] = group[col].apply(
            lambda s: s.shift(1).rolling(RECENT_GAMES, min_periods=3).mean()
        )

    # Home split prior-only
    for col in ["PTS", "OPP_PTS_ROW", "OFF_RTG_G", "DEF_RTG_G", "PACE_G"]:
        df[f"{col}_HOME_PRIOR"] = group.apply(
            lambda g: g[col].where(g["IS_HOME"] == 1).shift(1).expanding().mean()
        ).reset_index(level=0, drop=True)

    # Away split prior-only
    for col in ["PTS", "OPP_PTS_ROW", "OFF_RTG_G", "DEF_RTG_G", "PACE_G"]:
        df[f"{col}_AWAY_PRIOR"] = group.apply(
            lambda g: g[col].where(g["IS_HOME"] == 0).shift(1).expanding().mean()
        ).reset_index(level=0, drop=True)

    # Games played prior
    df["GAMES_PLAYED_PRIOR"] = group.cumcount()

    return df


def get_latest_team_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one latest prior-feature row per team.
    """
    latest = team_df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).groupby("TEAM_ID", as_index=False).tail(1).copy()

    # League averages from latest team priors
    league_avg = {
        "PTS": safe_mean(latest["PTS_SEASON_PRIOR"].tolist(), 114.0),
        "OPP_PTS": safe_mean(latest["OPP_PTS_ROW_SEASON_PRIOR"].tolist(), 114.0),
        "OFF_RTG": safe_mean(latest["OFF_RTG_G_SEASON_PRIOR"].tolist(), 114.0),
        "DEF_RTG": safe_mean(latest["DEF_RTG_G_SEASON_PRIOR"].tolist(), 114.0),
        "PACE": safe_mean(latest["PACE_G_SEASON_PRIOR"].tolist(), 99.0),
    }

    def blend_with_regression(row: pd.Series, metric_prefix: str, home_side: Optional[bool] = None) -> float:
        season_val = row.get(f"{metric_prefix}_SEASON_PRIOR", np.nan)
        recent_val = row.get(f"{metric_prefix}_RECENT_PRIOR", np.nan)

        if home_side is True:
            split_val = row.get(f"{metric_prefix}_HOME_PRIOR", np.nan)
        elif home_side is False:
            split_val = row.get(f"{metric_prefix}_AWAY_PRIOR", np.nan)
        else:
            split_val = np.nan

        vals = []
        weights = []

        if pd.notna(season_val):
            vals.append(season_val)
            weights.append(W_SEASON)
        if pd.notna(recent_val):
            vals.append(recent_val)
            weights.append(W_RECENT)
        if pd.notna(split_val):
            vals.append(split_val)
            weights.append(W_HOME_AWAY)

        if not vals:
            if "OFF_RTG" in metric_prefix:
                return league_avg["OFF_RTG"]
            if "DEF_RTG" in metric_prefix:
                return league_avg["DEF_RTG"]
            if "PACE" in metric_prefix:
                return league_avg["PACE"]
            if "OPP_PTS" in metric_prefix:
                return league_avg["OPP_PTS"]
            return league_avg["PTS"]

        raw = float(np.average(vals, weights=weights))
        gp = row.get("GAMES_PLAYED_PRIOR", 0)

        # More shrink for smaller samples
        sample_shrink = REGRESS_FACTOR if gp >= 20 else min(0.35, REGRESS_FACTOR + (20 - gp) * 0.01)

        if "OFF_RTG" in metric_prefix:
            anchor = league_avg["OFF_RTG"]
        elif "DEF_RTG" in metric_prefix:
            anchor = league_avg["DEF_RTG"]
        elif "PACE" in metric_prefix:
            anchor = league_avg["PACE"]
        elif "OPP_PTS" in metric_prefix:
            anchor = league_avg["OPP_PTS"]
        else:
            anchor = league_avg["PTS"]

        return (1.0 - sample_shrink) * raw + sample_shrink * anchor

    latest["PTS_BLEND_HOME"] = latest.apply(lambda r: blend_with_regression(r, "PTS", True), axis=1)
    latest["PTS_BLEND_AWAY"] = latest.apply(lambda r: blend_with_regression(r, "PTS", False), axis=1)
    latest["OPP_PTS_BLEND_HOME"] = latest.apply(lambda r: blend_with_regression(r, "OPP_PTS_ROW", True), axis=1)
    latest["OPP_PTS_BLEND_AWAY"] = latest.apply(lambda r: blend_with_regression(r, "OPP_PTS_ROW", False), axis=1)
    latest["OFF_RTG_BLEND_HOME"] = latest.apply(lambda r: blend_with_regression(r, "OFF_RTG_G", True), axis=1)
    latest["OFF_RTG_BLEND_AWAY"] = latest.apply(lambda r: blend_with_regression(r, "OFF_RTG_G", False), axis=1)
    latest["DEF_RTG_BLEND_HOME"] = latest.apply(lambda r: blend_with_regression(r, "DEF_RTG_G", True), axis=1)
    latest["DEF_RTG_BLEND_AWAY"] = latest.apply(lambda r: blend_with_regression(r, "DEF_RTG_G", False), axis=1)
    latest["PACE_BLEND_HOME"] = latest.apply(lambda r: blend_with_regression(r, "PACE_G", True), axis=1)
    latest["PACE_BLEND_AWAY"] = latest.apply(lambda r: blend_with_regression(r, "PACE_G", False), axis=1)

    return latest


def fetch_live_totals_odds(api_key: str) -> pd.DataFrame:
    """
    Pull upcoming NBA totals lines from The Odds API.
    """
    if not api_key:
        raise RuntimeError("ODDS_API_KEY environment variable is missing.")

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    games = resp.json()

    rows = []
    for game in games:
        commence = game.get("commence_time")
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        bookmakers = game.get("bookmakers", [])

        totals_points = []
        over_prices = []
        under_prices = []

        for book in bookmakers:
            for market in book.get("markets", []):
                if market.get("key") != "totals":
                    continue

                over_point = np.nan
                under_point = np.nan
                over_price = np.nan
                under_price = np.nan

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if name == "Over":
                        over_point = point
                        over_price = price
                    elif name == "Under":
                        under_point = point
                        under_price = price

                if pd.notna(over_point):
                    totals_points.append(float(over_point))
                if pd.notna(under_point):
                    totals_points.append(float(under_point))
                if pd.notna(over_price):
                    over_prices.append(float(over_price))
                if pd.notna(under_price):
                    under_prices.append(float(under_price))

        if totals_points:
            avg_total = float(np.mean(totals_points))
            rows.append({
                "commence_time": commence,
                "home_team": home_team,
                "away_team": away_team,
                "TOTAL_avg_total": avg_total,
                "OVER_avg_price": safe_mean(over_prices, np.nan),
                "UNDER_avg_price": safe_mean(under_prices, np.nan),
            })

    odds_df = pd.DataFrame(rows)
    if odds_df.empty:
        raise RuntimeError("No totals odds returned from The Odds API.")

    odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], utc=True, errors="coerce")
    return odds_df


def build_team_name_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Maps full team names to abbreviations and vice versa.
    """
    tm = get_team_meta()
    full_to_abbr = dict(zip(tm["TEAM_NAME"], tm["TEAM_ABBREVIATION"]))
    abbr_to_full = dict(zip(tm["TEAM_ABBREVIATION"], tm["TEAM_NAME"]))

    # Odds API usually uses full official names, but keep a few aliases for safety
    aliases = {
        "LA Clippers": "LAC",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "New York Knicks": "NYK",
        "Brooklyn Nets": "BKN",
        "Golden State Warriors": "GSW",
        "Phoenix Suns": "PHX",
        "Oklahoma City Thunder": "OKC",
        "San Antonio Spurs": "SAS",
        "New Orleans Pelicans": "NOP",
    }
    full_to_abbr.update(aliases)
    return full_to_abbr, abbr_to_full


def project_game_total(
    home_row: pd.Series,
    away_row: pd.Series,
    league_avg_total: float = 228.0,
) -> Dict[str, float]:
    """
    Build a pure-stat total projection.
    """

    # Pace projection
    pace_home = home_row["PACE_BLEND_HOME"]
    pace_away = away_row["PACE_BLEND_AWAY"]
    pace_proj = (pace_home + pace_away) / 2.0

    # Efficiency projection
    home_off = home_row["OFF_RTG_BLEND_HOME"]
    away_def = away_row["DEF_RTG_BLEND_AWAY"]

    away_off = away_row["OFF_RTG_BLEND_AWAY"]
    home_def = home_row["DEF_RTG_BLEND_HOME"]

    # Blend offense with opponent defense
    home_eff = OFFENSE_WEIGHT * home_off + DEFENSE_WEIGHT * away_def
    away_eff = OFFENSE_WEIGHT * away_off + DEFENSE_WEIGHT * home_def

    # Score from possession-based efficiency
    home_pts_eff = pace_proj * home_eff / 100.0
    away_pts_eff = pace_proj * away_eff / 100.0

    # Score from traditional points for / allowed
    home_pts_trad = (home_row["PTS_BLEND_HOME"] + away_row["OPP_PTS_BLEND_AWAY"]) / 2.0
    away_pts_trad = (away_row["PTS_BLEND_AWAY"] + home_row["OPP_PTS_BLEND_HOME"]) / 2.0

    # Final team point projections
    home_proj = 0.55 * home_pts_eff + 0.45 * home_pts_trad
    away_proj = 0.55 * away_pts_eff + 0.45 * away_pts_trad

    # Mild home-court scoring bump already partly embedded in splits; keep small
    home_proj += 0.6

    total_proj = home_proj + away_proj

    # Gentle shrink toward league average total
    gp_home = max(int(home_row.get("GAMES_PLAYED_PRIOR", 0)), 0)
    gp_away = max(int(away_row.get("GAMES_PLAYED_PRIOR", 0)), 0)
    gp = min(gp_home, gp_away)

    shrink = 0.10 if gp >= 25 else min(0.22, 0.10 + (25 - gp) * 0.005)
    total_proj = (1.0 - shrink) * total_proj + shrink * league_avg_total

    return {
        "MODEL_HOME_PTS": float(home_proj),
        "MODEL_AWAY_PTS": float(away_proj),
        "MODEL_GAME_PACE": float(pace_proj),
        "MODEL_HOME_EFF": float(home_eff),
        "MODEL_AWAY_EFF": float(away_eff),
        "MODEL_PROJECTED_TOTAL": float(total_proj),
    }


def build_output() -> pd.DataFrame:
    # 1) Historical games and rolling team features
    games = fetch_league_games(SEASON, SEASON_TYPE)
    games = attach_opponent_stats(games)
    games = add_rolling_features(games)

    latest = get_latest_team_features(games)
    latest = latest[latest["GAMES_PLAYED_PRIOR"] >= MIN_GAMES_REQUIRED].copy()

    full_to_abbr, abbr_to_full = build_team_name_maps()

    # league avg total from completed historical games
    hist_totals = games.groupby("GAME_ID", as_index=False).agg(
        GAME_DATE=("GAME_DATE", "max"),
        TOTAL_POINTS=("PTS", "sum")
    )
    league_avg_total = float(hist_totals["TOTAL_POINTS"].mean()) if not hist_totals.empty else 228.0

    # 2) Upcoming odds
    odds = fetch_live_totals_odds(ODDS_API_KEY)

    # 3) Map names to abbreviations
    odds["HOME_ABBR"] = odds["home_team"].map(full_to_abbr)
    odds["AWAY_ABBR"] = odds["away_team"].map(full_to_abbr)

    missing_name_rows = odds[odds["HOME_ABBR"].isna() | odds["AWAY_ABBR"].isna()]
    if not missing_name_rows.empty:
        print("Warning: could not map some team names from odds feed:")
        print(missing_name_rows[["away_team", "home_team"]].drop_duplicates())

    team_feature_map = latest.set_index("TEAM_ABBREVIATION").to_dict("index")

    output_rows = []
    for _, row in odds.iterrows():
        home_abbr = row["HOME_ABBR"]
        away_abbr = row["AWAY_ABBR"]

        if pd.isna(home_abbr) or pd.isna(away_abbr):
            continue
        if home_abbr not in team_feature_map or away_abbr not in team_feature_map:
            continue

        home_row = pd.Series(team_feature_map[home_abbr])
        away_row = pd.Series(team_feature_map[away_abbr])

        proj = project_game_total(home_row, away_row, league_avg_total=league_avg_total)

        vegas_total = float(row["TOTAL_avg_total"])
        model_total = float(proj["MODEL_PROJECTED_TOTAL"])
        edge_total = model_total - vegas_total

        # Side
        best_ou_side = "OVER" if edge_total > 0 else "UNDER"

        # Confidence buckets
        abs_edge = abs(edge_total)
        if abs_edge >= 8:
            confidence = "ELITE"
        elif abs_edge >= 5:
            confidence = "STRONG"
        elif abs_edge >= 3:
            confidence = "PLAYABLE"
        else:
            confidence = "LOW"

        # EV proxy from market vig prices
        over_price = row["OVER_avg_price"]
        under_price = row["UNDER_avg_price"]
        over_imp = american_to_implied_prob(over_price) if pd.notna(over_price) else np.nan
        under_imp = american_to_implied_prob(under_price) if pd.notna(under_price) else np.nan

        output_rows.append({
            "commence_time": row["commence_time"],
            "AWAY_TEAM": row["away_team"],
            "HOME_TEAM": row["home_team"],
            "AWAY_ABBR": away_abbr,
            "HOME_ABBR": home_abbr,

            "MODEL_AWAY_PTS": round(proj["MODEL_AWAY_PTS"], 2),
            "MODEL_HOME_PTS": round(proj["MODEL_HOME_PTS"], 2),
            "MODEL_GAME_PACE": round(proj["MODEL_GAME_PACE"], 2),
            "MODEL_AWAY_EFF": round(proj["MODEL_AWAY_EFF"], 2),
            "MODEL_HOME_EFF": round(proj["MODEL_HOME_EFF"], 2),
            "MODEL_PROJECTED_TOTAL": round(model_total, 2),

            "TOTAL_avg_total": round(vegas_total, 2),
            "OVER_avg_price": over_price,
            "UNDER_avg_price": under_price,
            "OVER_implied_prob": round(over_imp, 4) if pd.notna(over_imp) else np.nan,
            "UNDER_implied_prob": round(under_imp, 4) if pd.notna(under_imp) else np.nan,

            "EDGE_TOTAL": round(edge_total, 2),
            "ABS_EDGE_TOTAL": round(abs_edge, 2),
            "BEST_OU_SIDE": best_ou_side,
            "CONFIDENCE": confidence,

            # extra context fields
            "HOME_GAMES_PLAYED_PRIOR": int(home_row.get("GAMES_PLAYED_PRIOR", 0)),
            "AWAY_GAMES_PLAYED_PRIOR": int(away_row.get("GAMES_PLAYED_PRIOR", 0)),
        })

    out = pd.DataFrame(output_rows)
    if out.empty:
        raise RuntimeError("No output rows were generated. Check odds feed and team name mapping.")

    out = out.sort_values(
        ["ABS_EDGE_TOTAL", "commence_time"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return out


def build_ou_output() -> pd.DataFrame:
    return build_output()


def save_ou_output() -> pd.DataFrame:
    out = build_ou_output()
    out.to_csv(OUTPUT_CSV, index=False)
    return out


def main() -> None:
    out = save_ou_output()

    print(f"\nSaved: {OUTPUT_CSV}\n")
    show_cols = [
        "AWAY_TEAM", "HOME_TEAM", "MODEL_PROJECTED_TOTAL",
        "TOTAL_avg_total", "EDGE_TOTAL", "BEST_OU_SIDE", "CONFIDENCE"
    ]
    print(out[show_cols].to_string(index=False))


# Leave this only if you want to run this file directly from terminal.
# Streamlit will NOT trigger this when importing the file.
if __name__ == "__main__":
    main()
