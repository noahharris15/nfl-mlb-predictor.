"""
NBA Moneyline + Spread Model
- Pulls current season completed games from nba_api
- Builds rolling team features using only prior games
- Trains:
    1) Moneyline model (home win probability)
    2) Spread model (predicted home margin)
- Pulls live odds from The Odds API
- Compares model projections vs market consensus
- Outputs edge tables to CSV

Install:
pip install pandas numpy requests scikit-learn nba_api

Optional:
Set ODDS_API_KEY as an environment variable
Windows PowerShell:
$env:ODDS_API_KEY="YOUR_KEY"

Mac/Linux:
export ODDS_API_KEY="YOUR_KEY"
"""

import os
import time
import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from nba_api.stats.endpoints import LeagueGameFinder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
SEASON = "2025-26"   # this season
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "PASTE_YOUR_ODDS_API_KEY_HERE")

ODDS_SPORT = "basketball_nba"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads"
ODDS_BOOKMAKERS = "draftkings,fanduel,betmgm,caesars,espnbet,betrivers"
ODDS_FORMAT = "american"

OUTPUT_DIR = "."
SLEEP_SECONDS = 0.8  # gentle delay for NBA stats calls


# =========================
# HELPERS
# =========================
def american_to_prob(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def prob_to_american(prob):
    """Fair odds from probability."""
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    if prob >= 0.5:
        return -round((prob / (1 - prob)) * 100)
    return round(((1 - prob) / prob) * 100)


def kelly_fraction(model_prob, american_odds):
    """Raw Kelly fraction for a moneyline bet."""
    if pd.isna(model_prob) or pd.isna(american_odds):
        return 0.0

    odds = float(american_odds)
    p = float(model_prob)

    if odds > 0:
        b = odds / 100.0
    else:
        b = 100.0 / abs(odds)

    q = 1.0 - p
    k = (b * p - q) / b
    return max(0.0, k)


def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=False)


def parse_matchup(matchup_str):
    """
    Example:
    'LAL vs. BOS' => home game
    'LAL @ BOS'   => away game
    """
    matchup_str = str(matchup_str)
    if " vs. " in matchup_str:
        team = matchup_str.split(" vs. ")[0].strip()
        opp = matchup_str.split(" vs. ")[1].strip()
        is_home = 1
    elif " @ " in matchup_str:
        team = matchup_str.split(" @ ")[0].strip()
        opp = matchup_str.split(" @ ")[1].strip()
        is_home = 0
    else:
        team, opp, is_home = None, None, np.nan
    return team, opp, is_home


def clean_team_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()

    aliases = {
        "LA Clippers": "Los Angeles Clippers",
        "Los Angeles Clippers": "Los Angeles Clippers",
        "LA Lakers": "Los Angeles Lakers",
        "Los Angeles Lakers": "Los Angeles Lakers",
        "NY Knicks": "New York Knicks",
        "New York Knicks": "New York Knicks",
        "GS Warriors": "Golden State Warriors",
        "Golden State Warriors": "Golden State Warriors",
        "NO Pelicans": "New Orleans Pelicans",
        "New Orleans Pelicans": "New Orleans Pelicans",
        "SA Spurs": "San Antonio Spurs",
        "San Antonio Spurs": "San Antonio Spurs",
        "PHX Suns": "Phoenix Suns",
        "Phoenix Suns": "Phoenix Suns",
        "OKC Thunder": "Oklahoma City Thunder",
        "Oklahoma City Thunder": "Oklahoma City Thunder",
        "NOP": "New Orleans Pelicans",
        "NYK": "New York Knicks",
        "GSW": "Golden State Warriors",
        "LAC": "Los Angeles Clippers",
        "LAL": "Los Angeles Lakers",
        "SAS": "San Antonio Spurs",
        "PHX": "Phoenix Suns",
        "OKC": "Oklahoma City Thunder",
        "BKN": "Brooklyn Nets",
        "CHA": "Charlotte Hornets",
        "CHI": "Chicago Bulls",
        "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks",
        "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons",
        "HOU": "Houston Rockets",
        "IND": "Indiana Pacers",
        "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks",
        "MIN": "Minnesota Timberwolves",
        "ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers",
        "POR": "Portland Trail Blazers",
        "SAC": "Sacramento Kings",
        "TOR": "Toronto Raptors",
        "UTA": "Utah Jazz",
        "WAS": "Washington Wizards",
        "ATL": "Atlanta Hawks",
        "BOS": "Boston Celtics",
    }
    return aliases.get(name, name)


# =========================
# DATA PULL
# =========================
def fetch_season_games(season=SEASON):
    print(f"Pulling NBA games for season {season}...")
    lgf = LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season,
        league_id_nullable="00"
    )
    time.sleep(SLEEP_SECONDS)
    df = lgf.get_data_frames()[0].copy()

    # Standardize columns we need
    needed = [
        "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "MATCHUP", "WL", "MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A",
        "FTM", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK",
        "PLUS_MINUS"
    ]
    df = df[needed].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["TEAM_NAME"] = df["TEAM_NAME"].map(clean_team_name)
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str)

    parsed = df["MATCHUP"].apply(parse_matchup)
    df["TEAM_ABBR_PARSED"] = parsed.apply(lambda x: x[0])
    df["OPP_ABBR"] = parsed.apply(lambda x: x[1])
    df["IS_HOME"] = parsed.apply(lambda x: x[2]).astype(int)

    # Keep only one season worth of actual completed game rows
    df = df.dropna(subset=["GAME_ID", "GAME_DATE", "TEAM_NAME", "OPP_ABBR"])
    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)

    # Possessions and efficiency features
    # Possessions estimate:
    # FGA - OREB + TOV + 0.44 * FTA
    df["POSS"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    df["EFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)
    df["TOV_PCT"] = df["TOV"] / df["POSS"].replace(0, np.nan)
    df["ORB_PCT_PROXY"] = df["OREB"] / (df["OREB"] + df["DREB"]).replace(0, np.nan)
    df["FTR"] = df["FTA"] / df["FGA"].replace(0, np.nan)
    df["OFF_RTG"] = 100 * df["PTS"] / df["POSS"].replace(0, np.nan)

    return df


def build_game_level_dataset(team_games):
    """
    Merge the two team rows per game into one row:
    home team stats + away team stats + result.
    """
    home = team_games[team_games["IS_HOME"] == 1].copy()
    away = team_games[team_games["IS_HOME"] == 0].copy()

    home_cols = {
        c: f"HOME_{c}" for c in [
            "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "WL", "PTS", "FGM", "FGA", "FG3M",
            "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK",
            "PLUS_MINUS", "POSS", "EFG", "TOV_PCT", "ORB_PCT_PROXY", "FTR", "OFF_RTG"
        ]
    }
    away_cols = {
        c: f"AWAY_{c}" for c in [
            "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "WL", "PTS", "FGM", "FGA", "FG3M",
            "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK",
            "PLUS_MINUS", "POSS", "EFG", "TOV_PCT", "ORB_PCT_PROXY", "FTR", "OFF_RTG"
        ]
    }

    home = home.rename(columns=home_cols)
    away = away.rename(columns=away_cols)

    merged = pd.merge(
        home[["GAME_ID", "GAME_DATE"] + list(home_cols.values())],
        away[["GAME_ID", "GAME_DATE"] + list(away_cols.values())],
        on=["GAME_ID", "GAME_DATE"],
        how="inner"
    ).copy()

    # Targets
    merged["HOME_MARGIN"] = merged["HOME_PTS"] - merged["AWAY_PTS"]
    merged["HOME_WIN"] = (merged["HOME_MARGIN"] > 0).astype(int)

    # Opponent allowed ratings based on same game possessions
    merged["HOME_DEF_RTG"] = 100 * merged["AWAY_PTS"] / merged["AWAY_POSS"].replace(0, np.nan)
    merged["AWAY_DEF_RTG"] = 100 * merged["HOME_PTS"] / merged["HOME_POSS"].replace(0, np.nan)
    merged["HOME_NET_RTG"] = merged["HOME_OFF_RTG"] - merged["HOME_DEF_RTG"]
    merged["AWAY_NET_RTG"] = merged["AWAY_OFF_RTG"] - merged["AWAY_DEF_RTG"]

    return merged.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# =========================
# FEATURE ENGINEERING
# =========================
def add_team_rolling_features(team_games):
    """
    Rolling features per team using only PRIOR games.
    """
    df = team_games.copy().sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    df["WIN"] = (df["WL"] == "W").astype(int)
    df["MARGIN"] = df["PLUS_MINUS"]
    df["REST_DAYS"] = df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
    df["REST_DAYS"] = df["REST_DAYS"].fillna(5).clip(lower=0)
    df["B2B"] = (df["REST_DAYS"] <= 1).astype(int)

    base_cols = ["PTS", "POSS", "EFG", "TOV_PCT", "ORB_PCT_PROXY", "FTR", "OFF_RTG", "WIN", "MARGIN"]

    for window in [5, 10, 15]:
        for col in base_cols:
            df[f"{col}_ROLL_{window}"] = (
                df.groupby("TEAM_ID")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    # Season-to-date expanding means using only prior games
    for col in base_cols:
        df[f"{col}_SEASON_AVG"] = (
            df.groupby("TEAM_ID")[col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

    # Last game features
    for col in ["PTS", "MARGIN", "OFF_RTG", "EFG"]:
        df[f"{col}_LAST"] = df.groupby("TEAM_ID")[col].shift(1)

    return df


def build_model_table(team_games_with_roll):
    """
    Build one row per game with home/away PRE-GAME features.
    """
    feat_cols = [
        "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "IS_HOME",
        "REST_DAYS", "B2B",
        "PTS_ROLL_5", "POSS_ROLL_5", "EFG_ROLL_5", "TOV_PCT_ROLL_5", "ORB_PCT_PROXY_ROLL_5", "FTR_ROLL_5", "OFF_RTG_ROLL_5", "WIN_ROLL_5", "MARGIN_ROLL_5",
        "PTS_ROLL_10", "POSS_ROLL_10", "EFG_ROLL_10", "TOV_PCT_ROLL_10", "ORB_PCT_PROXY_ROLL_10", "FTR_ROLL_10", "OFF_RTG_ROLL_10", "WIN_ROLL_10", "MARGIN_ROLL_10",
        "PTS_ROLL_15", "POSS_ROLL_15", "EFG_ROLL_15", "TOV_PCT_ROLL_15", "ORB_PCT_PROXY_ROLL_15", "FTR_ROLL_15", "OFF_RTG_ROLL_15", "WIN_ROLL_15", "MARGIN_ROLL_15",
        "PTS_SEASON_AVG", "POSS_SEASON_AVG", "EFG_SEASON_AVG", "TOV_PCT_SEASON_AVG", "ORB_PCT_PROXY_SEASON_AVG", "FTR_SEASON_AVG", "OFF_RTG_SEASON_AVG", "WIN_SEASON_AVG", "MARGIN_SEASON_AVG",
        "PTS_LAST", "MARGIN_LAST", "OFF_RTG_LAST", "EFG_LAST",
        "PTS", "PLUS_MINUS", "WL"
    ]

    use = team_games_with_roll[feat_cols].copy()
    home = use[use["IS_HOME"] == 1].copy()
    away = use[use["IS_HOME"] == 0].copy()

    home = home.rename(columns={c: f"HOME_{c}" for c in home.columns if c not in ["GAME_ID", "GAME_DATE"]})
    away = away.rename(columns={c: f"AWAY_{c}" for c in away.columns if c not in ["GAME_ID", "GAME_DATE"]})

    df = pd.merge(home, away, on=["GAME_ID", "GAME_DATE"], how="inner")

    # Targets from actual final score
    df["HOME_MARGIN"] = df["HOME_PTS"] - df["AWAY_PTS"]
    df["HOME_WIN"] = (df["HOME_MARGIN"] > 0).astype(int)

    # Matchup differential features
    diff_pairs = [
        "REST_DAYS",
        "B2B",
        "PTS_ROLL_5", "POSS_ROLL_5", "EFG_ROLL_5", "TOV_PCT_ROLL_5", "ORB_PCT_PROXY_ROLL_5", "FTR_ROLL_5", "OFF_RTG_ROLL_5", "WIN_ROLL_5", "MARGIN_ROLL_5",
        "PTS_ROLL_10", "POSS_ROLL_10", "EFG_ROLL_10", "TOV_PCT_ROLL_10", "ORB_PCT_PROXY_ROLL_10", "FTR_ROLL_10", "OFF_RTG_ROLL_10", "WIN_ROLL_10", "MARGIN_ROLL_10",
        "PTS_ROLL_15", "POSS_ROLL_15", "EFG_ROLL_15", "TOV_PCT_ROLL_15", "ORB_PCT_PROXY_ROLL_15", "FTR_ROLL_15", "OFF_RTG_ROLL_15", "WIN_ROLL_15", "MARGIN_ROLL_15",
        "PTS_SEASON_AVG", "POSS_SEASON_AVG", "EFG_SEASON_AVG", "TOV_PCT_SEASON_AVG", "ORB_PCT_PROXY_SEASON_AVG", "FTR_SEASON_AVG", "OFF_RTG_SEASON_AVG", "WIN_SEASON_AVG", "MARGIN_SEASON_AVG",
        "PTS_LAST", "MARGIN_LAST", "OFF_RTG_LAST", "EFG_LAST"
    ]

    for col in diff_pairs:
        df[f"DIFF_{col}"] = df[f"HOME_{col}"] - df[f"AWAY_{col}"]

    df["HOME_COURT"] = 1.0

    return df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def get_feature_columns(df):
    return [c for c in df.columns if c.startswith("DIFF_")] + ["HOME_COURT"]


# =========================
# MODEL TRAINING
# =========================
def time_split(df, holdout_games=150):
    """
    Keep latest games as validation.
    """
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    split_idx = max(0, len(df) - holdout_games)
    train = df.iloc[:split_idx].copy()
    valid = df.iloc[split_idx:].copy()
    return train, valid


def train_models(model_df):
    features = get_feature_columns(model_df)

    # Remove rows with no prior info at all
    enough_data = model_df[features].notna().sum(axis=1) >= 8
    model_df = model_df[enough_data].copy().reset_index(drop=True)

    train_df, valid_df = time_split(model_df, holdout_games=min(150, max(30, len(model_df)//5)))

    X_train = train_df[features]
    y_train_ml = train_df["HOME_WIN"]
    y_train_spread = train_df["HOME_MARGIN"]

    X_valid = valid_df[features]
    y_valid_ml = valid_df["HOME_WIN"]
    y_valid_spread = valid_df["HOME_MARGIN"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), features)
        ]
    )

    moneyline_model = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000, C=0.7))
    ])

    spread_model = Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(alpha=8.0))
    ])

    moneyline_model.fit(X_train, y_train_ml)
    spread_model.fit(X_train, y_train_spread)

    # Validation metrics
    if len(valid_df) > 0:
        ml_probs = moneyline_model.predict_proba(X_valid)[:, 1]
        sp_preds = spread_model.predict(X_valid)

        try:
            auc = roc_auc_score(y_valid_ml, ml_probs)
        except Exception:
            auc = np.nan

        try:
            ll = log_loss(y_valid_ml, np.clip(ml_probs, 1e-6, 1-1e-6))
        except Exception:
            ll = np.nan

        mae = mean_absolute_error(y_valid_spread, sp_preds)

        print("\nValidation:")
        print(f"Moneyline AUC:      {auc:.4f}" if pd.notna(auc) else "Moneyline AUC:      n/a")
        print(f"Moneyline Log Loss: {ll:.4f}" if pd.notna(ll) else "Moneyline Log Loss: n/a")
        print(f"Spread MAE:         {mae:.4f}")
    else:
        print("\nValidation skipped: not enough held-out games.")

    return moneyline_model, spread_model, model_df


# =========================
# ODDS PULL
# =========================
def fetch_odds_api_games():
    if not ODDS_API_KEY or ODDS_API_KEY == "PASTE_YOUR_ODDS_API_KEY_HERE":
        raise ValueError("Set your Odds API key in ODDS_API_KEY or environment variable.")

    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "bookmakers": ODDS_BOOKMAKERS,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    # Usage headers can help monitor quota
    remaining = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    print(f"Odds API requests used: {used}, remaining: {remaining}")

    data = r.json()
    return data


def flatten_odds(odds_json):
    rows = []

    for game in odds_json:
        game_id = game.get("id")
        commence_time = game.get("commence_time")
        home_team = clean_team_name(game.get("home_team"))
        away_team = clean_team_name(game.get("away_team"))

        for bm in game.get("bookmakers", []):
            bookmaker = bm.get("key")
            for market in bm.get("markets", []):
                mkey = market.get("key")
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "odds_game_id": game_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker,
                        "market": mkey,
                        "team": clean_team_name(outcome.get("name")),
                        "price": outcome.get("price"),
                        "point": outcome.get("point")
                    })

    odds_df = pd.DataFrame(rows)
    if odds_df.empty:
        return odds_df

    odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], utc=True, errors="coerce")
    return odds_df


def build_market_consensus(odds_df):
    """
    Create consensus moneyline and spread by team/game.
    """
    if odds_df.empty:
        return pd.DataFrame()

    # Moneylines
    h2h = odds_df[odds_df["market"] == "h2h"].copy()
    if not h2h.empty:
        h2h["implied_prob"] = h2h["price"].apply(american_to_prob)
        h2h_cons = (
            h2h.groupby(["odds_game_id", "home_team", "away_team", "team"], as_index=False)
            .agg(
                avg_moneyline=("price", "mean"),
                median_moneyline=("price", "median"),
                avg_implied_prob=("implied_prob", "mean"),
                books=("bookmaker", "nunique")
            )
        )
    else:
        h2h_cons = pd.DataFrame()

    # Spreads
    spr = odds_df[odds_df["market"] == "spreads"].copy()
    if not spr.empty:
        spr_cons = (
            spr.groupby(["odds_game_id", "home_team", "away_team", "team"], as_index=False)
            .agg(
                avg_spread=("point", "mean"),
                median_spread=("point", "median"),
                avg_spread_price=("price", "mean"),
                books_spread=("bookmaker", "nunique")
            )
        )
    else:
        spr_cons = pd.DataFrame()

    # Pivot so one row = game
    def pivot_side(df_in, value_cols, side_name):
        if df_in.empty:
            return pd.DataFrame()
        out = df_in.copy()
        out["side"] = np.where(out["team"] == out["home_team"], "HOME", "AWAY")
        pivot_cols = ["odds_game_id", "home_team", "away_team"]

        wide = out[pivot_cols + ["side"] + value_cols].pivot_table(
            index=pivot_cols,
            columns="side",
            values=value_cols,
            aggfunc="first"
        )
        wide.columns = [f"{side}_{col}" for col, side in wide.columns]
        return wide.reset_index()

    h2h_wide = pivot_side(h2h_cons, ["avg_moneyline", "median_moneyline", "avg_implied_prob", "books"], "h2h")
    spr_wide = pivot_side(spr_cons, ["avg_spread", "median_spread", "avg_spread_price", "books_spread"], "spreads")

    if not h2h_wide.empty and not spr_wide.empty:
        consensus = pd.merge(h2h_wide, spr_wide, on=["odds_game_id", "home_team", "away_team"], how="outer")
    elif not h2h_wide.empty:
        consensus = h2h_wide.copy()
    else:
        consensus = spr_wide.copy()

    return consensus


# =========================
# UPCOMING GAME FEATURE BUILD
# =========================
def get_latest_team_feature_snapshot(team_games_with_roll):
    """
    Most recent pregame-derived team state for each team.
    """
    df = team_games_with_roll.copy().sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])
    latest = df.groupby("TEAM_ID").tail(1).copy()

    cols = [
        "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "GAME_DATE",
        "REST_DAYS", "B2B",
        "PTS_ROLL_5", "POSS_ROLL_5", "EFG_ROLL_5", "TOV_PCT_ROLL_5", "ORB_PCT_PROXY_ROLL_5", "FTR_ROLL_5", "OFF_RTG_ROLL_5", "WIN_ROLL_5", "MARGIN_ROLL_5",
        "PTS_ROLL_10", "POSS_ROLL_10", "EFG_ROLL_10", "TOV_PCT_ROLL_10", "ORB_PCT_PROXY_ROLL_10", "FTR_ROLL_10", "OFF_RTG_ROLL_10", "WIN_ROLL_10", "MARGIN_ROLL_10",
        "PTS_ROLL_15", "POSS_ROLL_15", "EFG_ROLL_15", "TOV_PCT_ROLL_15", "ORB_PCT_PROXY_ROLL_15", "FTR_ROLL_15", "OFF_RTG_ROLL_15", "WIN_ROLL_15", "MARGIN_ROLL_15",
        "PTS_SEASON_AVG", "POSS_SEASON_AVG", "EFG_SEASON_AVG", "TOV_PCT_SEASON_AVG", "ORB_PCT_PROXY_SEASON_AVG", "FTR_SEASON_AVG", "OFF_RTG_SEASON_AVG", "WIN_SEASON_AVG", "MARGIN_SEASON_AVG",
        "PTS_LAST", "MARGIN_LAST", "OFF_RTG_LAST", "EFG_LAST"
    ]
    return latest[cols].copy()


def estimate_rest_days(last_game_date, commence_time_utc):
    if pd.isna(last_game_date) or pd.isna(commence_time_utc):
        return np.nan
    if commence_time_utc.tzinfo is None:
        commence_time_utc = commence_time_utc.tz_localize("UTC")
    last_ts = pd.Timestamp(last_game_date)
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    return max(0, (commence_time_utc.normalize() - last_ts.normalize()).days)


def build_upcoming_features(consensus_df, latest_team_state):
    """
    Convert upcoming games from odds feed into model-ready feature rows.
    """
    rows = []
    if consensus_df.empty:
        return pd.DataFrame()

    team_lookup = latest_team_state.copy()
    team_lookup["TEAM_NAME"] = team_lookup["TEAM_NAME"].map(clean_team_name)

    for _, r in consensus_df.iterrows():
        home_team = clean_team_name(r["home_team"])
        away_team = clean_team_name(r["away_team"])

        home = team_lookup[team_lookup["TEAM_NAME"] == home_team]
        away = team_lookup[team_lookup["TEAM_NAME"] == away_team]

        if home.empty or away.empty:
            continue

        home = home.iloc[0]
        away = away.iloc[0]

        commence = pd.to_datetime(pd.NaT)
        if "odds_game_id" in r.index:
            pass

        # Try to reconstruct commence_time from original later if needed
        home_rest_est = home["REST_DAYS"]
        away_rest_est = away["REST_DAYS"]

        row = {
            "odds_game_id": r["odds_game_id"],
            "home_team": home_team,
            "away_team": away_team,
            "HOME_COURT": 1.0,

            # Market fields
            "HOME_avg_moneyline": r.get("HOME_avg_moneyline", np.nan),
            "AWAY_avg_moneyline": r.get("AWAY_avg_moneyline", np.nan),
            "HOME_avg_implied_prob": r.get("HOME_avg_implied_prob", np.nan),
            "AWAY_avg_implied_prob": r.get("AWAY_avg_implied_prob", np.nan),
            "HOME_avg_spread": r.get("HOME_avg_spread", np.nan),
            "AWAY_avg_spread": r.get("AWAY_avg_spread", np.nan),
            "HOME_books": r.get("HOME_books", np.nan),
            "AWAY_books": r.get("AWAY_books", np.nan),
            "HOME_books_spread": r.get("HOME_books_spread", np.nan),
            "AWAY_books_spread": r.get("AWAY_books_spread", np.nan),
        }

        feat_base = [
            "REST_DAYS", "B2B",
            "PTS_ROLL_5", "POSS_ROLL_5", "EFG_ROLL_5", "TOV_PCT_ROLL_5", "ORB_PCT_PROXY_ROLL_5", "FTR_ROLL_5", "OFF_RTG_ROLL_5", "WIN_ROLL_5", "MARGIN_ROLL_5",
            "PTS_ROLL_10", "POSS_ROLL_10", "EFG_ROLL_10", "TOV_PCT_ROLL_10", "ORB_PCT_PROXY_ROLL_10", "FTR_ROLL_10", "OFF_RTG_ROLL_10", "WIN_ROLL_10", "MARGIN_ROLL_10",
            "PTS_ROLL_15", "POSS_ROLL_15", "EFG_ROLL_15", "TOV_PCT_ROLL_15", "ORB_PCT_PROXY_ROLL_15", "FTR_ROLL_15", "OFF_RTG_ROLL_15", "WIN_ROLL_15", "MARGIN_ROLL_15",
            "PTS_SEASON_AVG", "POSS_SEASON_AVG", "EFG_SEASON_AVG", "TOV_PCT_SEASON_AVG", "ORB_PCT_PROXY_SEASON_AVG", "FTR_SEASON_AVG", "OFF_RTG_SEASON_AVG", "WIN_SEASON_AVG", "MARGIN_SEASON_AVG",
            "PTS_LAST", "MARGIN_LAST", "OFF_RTG_LAST", "EFG_LAST"
        ]

        # Override rest with latest known if you want to keep simple/stable
        home_map = home.to_dict()
        away_map = away.to_dict()
        home_map["REST_DAYS"] = home_rest_est
        away_map["REST_DAYS"] = away_rest_est
        home_map["B2B"] = int(pd.notna(home_rest_est) and home_rest_est <= 1)
        away_map["B2B"] = int(pd.notna(away_rest_est) and away_rest_est <= 1)

        for col in feat_base:
            row[f"DIFF_{col}"] = home_map.get(col, np.nan) - away_map.get(col, np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# SCORING UPCOMING GAMES
# =========================
def score_upcoming_games(upcoming_df, moneyline_model, spread_model, feature_cols):
    if upcoming_df.empty:
        return pd.DataFrame()

    X = upcoming_df[feature_cols].copy()
    upcoming_df = upcoming_df.copy()

    upcoming_df["MODEL_HOME_WIN_PROB"] = moneyline_model.predict_proba(X)[:, 1]
    upcoming_df["MODEL_AWAY_WIN_PROB"] = 1 - upcoming_df["MODEL_HOME_WIN_PROB"]
    upcoming_df["MODEL_HOME_FAIR_ML"] = upcoming_df["MODEL_HOME_WIN_PROB"].apply(prob_to_american)
    upcoming_df["MODEL_AWAY_FAIR_ML"] = upcoming_df["MODEL_AWAY_WIN_PROB"].apply(prob_to_american)
    upcoming_df["MODEL_HOME_MARGIN"] = spread_model.predict(X)
    upcoming_df["MODEL_AWAY_MARGIN"] = -upcoming_df["MODEL_HOME_MARGIN"]

    # Moneyline edge
    upcoming_df["EDGE_HOME_ML"] = upcoming_df["MODEL_HOME_WIN_PROB"] - upcoming_df["HOME_avg_implied_prob"]
    upcoming_df["EDGE_AWAY_ML"] = upcoming_df["MODEL_AWAY_WIN_PROB"] - upcoming_df["AWAY_avg_implied_prob"]

    # Spread edge
    # Home spread line is usually negative if favored
    upcoming_df["EDGE_HOME_SPREAD"] = upcoming_df["MODEL_HOME_MARGIN"] + upcoming_df["HOME_avg_spread"]
    upcoming_df["EDGE_AWAY_SPREAD"] = upcoming_df["MODEL_AWAY_MARGIN"] + upcoming_df["AWAY_avg_spread"]

    # Kelly suggestions (quarter Kelly)
    upcoming_df["HOME_KELLY_FULL"] = upcoming_df.apply(
        lambda r: kelly_fraction(r["MODEL_HOME_WIN_PROB"], r["HOME_avg_moneyline"]), axis=1
    )
    upcoming_df["AWAY_KELLY_FULL"] = upcoming_df.apply(
        lambda r: kelly_fraction(r["MODEL_AWAY_WIN_PROB"], r["AWAY_avg_moneyline"]), axis=1
    )
    upcoming_df["HOME_KELLY_QTR"] = upcoming_df["HOME_KELLY_FULL"] * 0.25
    upcoming_df["AWAY_KELLY_QTR"] = upcoming_df["AWAY_KELLY_FULL"] * 0.25

    # Best side labels
    def best_ml_side(r):
        if pd.isna(r["EDGE_HOME_ML"]) and pd.isna(r["EDGE_AWAY_ML"]):
            return "NO_ODDS"
        return "HOME" if r["EDGE_HOME_ML"] >= r["EDGE_AWAY_ML"] else "AWAY"

    def best_spread_side(r):
        if pd.isna(r["EDGE_HOME_SPREAD"]) and pd.isna(r["EDGE_AWAY_SPREAD"]):
            return "NO_ODDS"
        return "HOME" if r["EDGE_HOME_SPREAD"] >= r["EDGE_AWAY_SPREAD"] else "AWAY"

    upcoming_df["BEST_ML_SIDE"] = upcoming_df.apply(best_ml_side, axis=1)
    upcoming_df["BEST_SPREAD_SIDE"] = upcoming_df.apply(best_spread_side, axis=1)

    upcoming_df["BEST_ML_EDGE"] = upcoming_df.apply(
        lambda r: max(r["EDGE_HOME_ML"], r["EDGE_AWAY_ML"]), axis=1
    )
    upcoming_df["BEST_SPREAD_EDGE"] = upcoming_df.apply(
        lambda r: max(r["EDGE_HOME_SPREAD"], r["EDGE_AWAY_SPREAD"]), axis=1
    )

    return upcoming_df


# =========================
# OUTPUTS
# =========================
def save_outputs(scored_df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_path = os.path.join(OUTPUT_DIR, f"nba_model_edges_full_{timestamp}.csv")
    ml_path = os.path.join(OUTPUT_DIR, f"nba_best_moneyline_edges_{timestamp}.csv")
    spread_path = os.path.join(OUTPUT_DIR, f"nba_best_spread_edges_{timestamp}.csv")

    scored_df.to_csv(full_path, index=False)

    ml_cols = [
        "home_team", "away_team",
        "MODEL_HOME_WIN_PROB", "MODEL_AWAY_WIN_PROB",
        "HOME_avg_moneyline", "AWAY_avg_moneyline",
        "MODEL_HOME_FAIR_ML", "MODEL_AWAY_FAIR_ML",
        "EDGE_HOME_ML", "EDGE_AWAY_ML",
        "HOME_KELLY_QTR", "AWAY_KELLY_QTR",
        "BEST_ML_SIDE", "BEST_ML_EDGE"
    ]
    spread_cols = [
        "home_team", "away_team",
        "MODEL_HOME_MARGIN", "MODEL_AWAY_MARGIN",
        "HOME_avg_spread", "AWAY_avg_spread",
        "EDGE_HOME_SPREAD", "EDGE_AWAY_SPREAD",
        "BEST_SPREAD_SIDE", "BEST_SPREAD_EDGE"
    ]

    scored_df[ml_cols].sort_values("BEST_ML_EDGE", ascending=False).to_csv(ml_path, index=False)
    scored_df[spread_cols].sort_values("BEST_SPREAD_EDGE", ascending=False).to_csv(spread_path, index=False)

    print("\nSaved files:")
    print(full_path)
    print(ml_path)
    print(spread_path)


# =========================
# MAIN
# =========================
def main():
    # 1) Pull season team game logs
    team_games = fetch_season_games(SEASON)

    # 2) Add rolling priors
    team_games_roll = add_team_rolling_features(team_games)

    # 3) Build game-level model table
    model_df = build_model_table(team_games_roll)

    # 4) Train models
    moneyline_model, spread_model, trained_table = train_models(model_df)
    feature_cols = get_feature_columns(trained_table)

    # 5) Pull live odds
    odds_json = fetch_odds_api_games()
    odds_df = flatten_odds(odds_json)

    if odds_df.empty:
        print("No odds returned.")
        return

    consensus = build_market_consensus(odds_df)

    if consensus.empty:
        print("Could not build market consensus.")
        return

    # 6) Latest team state
    latest_team_state = get_latest_team_feature_snapshot(team_games_roll)

    # 7) Build upcoming feature rows
    upcoming_df = build_upcoming_features(consensus, latest_team_state)

    if upcoming_df.empty:
        print("Could not map upcoming odds games to team feature rows.")
        return

    # 8) Score games
    scored_df = score_upcoming_games(upcoming_df, moneyline_model, spread_model, feature_cols)

    if scored_df.empty:
        print("No scored games.")
        return

    # 9) Display top edges
    print("\nTop moneyline edges:")
    ml_view = scored_df[
        [
            "home_team", "away_team", "BEST_ML_SIDE", "BEST_ML_EDGE",
            "MODEL_HOME_WIN_PROB", "MODEL_AWAY_WIN_PROB",
            "HOME_avg_moneyline", "AWAY_avg_moneyline",
            "HOME_KELLY_QTR", "AWAY_KELLY_QTR"
        ]
    ].sort_values("BEST_ML_EDGE", ascending=False).head(10)
    print(ml_view.to_string(index=False))

    print("\nTop spread edges:")
    spread_view = scored_df[
        [
            "home_team", "away_team", "BEST_SPREAD_SIDE", "BEST_SPREAD_EDGE",
            "MODEL_HOME_MARGIN", "MODEL_AWAY_MARGIN",
            "HOME_avg_spread", "AWAY_avg_spread"
        ]
    ].sort_values("BEST_SPREAD_EDGE", ascending=False).head(10)
    print(spread_view.to_string(index=False))

    # 10) Save
    save_outputs(scored_df)


if __name__ == "__main__":
    main()
