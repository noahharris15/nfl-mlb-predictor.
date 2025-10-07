# NCAA Player Props — Odds API + ESPN (per-game means; 10k sims)
# Place this file at: pages/3_NCAA_Football.py
# Run whole app: streamlit run App.py

import math, re, unicodedata
from typing import List, Optional, Dict
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- UI / constants ----------------
st.set_page_config(page_title="NCAA Football Props — Odds API + ESPN", layout="wide")
st.title("🏈 NCAA Football Player Props — Odds API + ESPN")

SIM_TRIALS = 10_000
VALID_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    "player_pass_tds",
    "player_rush_reception_yds",
    "player_rush_attempts",
    "player_reception_yds",
    "player_pass_completions",
    "player_pass_attempts",
    "player_field_goals",
]

# ---------------- ESPN endpoints (college FB) ----------------
SB_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
SUM_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"

def http_get(url, params=None, timeout=25) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
def norm_name(n: str) -> str:
    n = str(n or "")
    n = n.split("(")[0]
    n = re.sub(r"[.,'–-]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return strip_accents(n)

@st.cache_data(show_spinner=False)
def list_week_events(year: int, week: int, seasontype: int=2) -> List[str]:
    js = http_get(SB_URL, {"year": year, "week": week, "seasontype": seasontype})
    if not js: return []
    return [str(e.get("id")) for e in js.get("events", []) if e.get("id")]

@st.cache_data(show_spinner=False)
def fetch_box(event_id: str) -> Optional[dict]:
    return http_get(SUM_URL, {"event": event_id})

def _pluck_val(stats: list, idx: int) -> float:
    try:
        return float(stats[idx])
    except Exception:
        return float("nan")

def parse_box_players(box: dict) -> pd.DataFrame:
    """
    Reads ESPN summary -> boxscore players sections.
    Returns ONE row per player with available stats for pass/rush/rec/kicking.
    """
    rows = []
    try:
        teams = box.get("boxscore", {}).get("players", [])
        for t in teams:
            for block in t.get("statistics", []):
                label = (block.get("name") or "").lower()
                for a in block.get("athletes", []):
                    nm = norm_name(a.get("athlete", {}).get("displayName"))
                    stats = a.get("stats") or []

                    row = {"Player": nm,
                           "pass_yds": 0.0, "pass_tds": 0.0, "pass_cmp": 0.0, "pass_att": 0.0,
                           "rush_yds": 0.0, "rush_att": 0.0,
                           "rec": 0.0, "rec_yds": 0.0,
                           "fgm": 0.0}

                    # Passing block typical order: C/ATT, YDS, TD, INT, ...
                    if "passing" in label and len(stats) >= 3:
                        # C/ATT usually like "17-29"
                        ca = stats[0] if len(stats) > 0 else ""
                        try:
                            cmp_, att_ = ca.split("-")
                            row["pass_cmp"] = float(cmp_)
                            row["pass_att"] = float(att_)
                        except Exception:
                            pass
                        row["pass_yds"] = _pluck_val(stats, 1)
                        row["pass_tds"] = _pluck_val(stats, 2)

                    # Rushing block: CAR, YDS, TD, LONG, ...
                    if "rushing" in label and len(stats) >= 2:
                        row["rush_att"] = _pluck_val(stats, 0)
                        row["rush_yds"] = _pluck_val(stats, 1)

                    # Receiving block: REC, YDS, TD, LONG...
                    if "receiving" in label and len(stats) >= 2:
                        row["rec"] = _pluck_val(stats, 0)
                        row["rec_yds"] = _pluck_val(stats, 1)

                    # Kicking block often has "FGM-FGA", "XPM-XPA"
                    if "kicking" in label and len(stats) >= 1:
                        fg = stats[0]
                        try:
                            fgm, fga = fg.split("-")
                            row["fgm"] = float(fgm)
                        except Exception:
                            pass

                    # Only keep if any signal
                    if any(v > 0 for k,v in row.items() if k != "Player"):
                        rows.append(row)
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=["Player","pass_yds","pass_tds","pass_cmp","pass_att","rush_yds","rush_att","rec","rec_yds","fgm"])
    # A player can appear in multiple blocks → sum within game
    return pd.DataFrame(rows).groupby("Player", as_index=False).sum(numeric_only=True)

@st.cache_data(show_spinner=True)
def build_season_agg(year: int, week_lo: int, week_hi: int, seasontype: int) -> pd.DataFrame:
    totals, sumsqs, games = {}, {}, {}
    def init_p(p):
        if p not in totals:
            totals[p] = {"pass_yds":0.0,"pass_tds":0.0,"pass_cmp":0.0,"pass_att":0.0,
                         "rush_yds":0.0,"rush_att":0.0,"rec":0.0,"rec_yds":0.0,"fgm":0.0}
            sumsqs[p] = {"pass_yds":0.0,"rush_yds":0.0,"rec":0.0,"rec_yds":0.0}
            games[p]  = 0

    events = []
    for wk in range(week_lo, week_hi+1):
        events += list_week_events(year, wk, seasontype)

    if not events:
        return pd.DataFrame()

    prog = st.progress(0.0, text=f"Crawling {len(events)} games…")
    for j, ev in enumerate(events, 1):
        box = fetch_box(ev)
        if box:
            df = parse_box_players(box)
            for _, r in df.iterrows():
                p = r["Player"]; init_p(p)
                # mark game played if any stat > 0
                if any(float(r[k]) > 0 for k in ["pass_yds","rush_yds","rec","rec_yds","fgm"]):
                    games[p] += 1
                for k in totals[p]:
                    v = float(r.get(k, 0.0)) if not pd.isna(r.get(k, np.nan)) else 0.0
                    totals[p][k] += v
                for k in ["pass_yds","rush_yds","rec","rec_yds"]:
                    v = float(r.get(k, 0.0)) if not pd.isna(r.get(k, np.nan)) else 0.0
                    sumsqs[p][k] += v*v
        prog.progress(j/len(events))

    rows = []
    for p, stat in totals.items():
        g = max(1, games.get(p, 0))
        rows.append({"Player": p, "g": g, **stat,
                     "sq_pass_yds": sumsqs[p]["pass_yds"],
                     "sq_rush_yds": sumsqs[p]["rush_yds"],
                     "sq_rec": sumsqs[p]["rec"],
                     "sq_rec_yds": sumsqs[p]["rec_yds"]})
    return pd.DataFrame(rows)

def sample_sd(sum_x, sum_x2, g_val):
    g_val = int(g_val)
    if g_val <= 1: return np.nan
    mean = sum_x / g_val
    var  = (sum_x2 / g_val) - (mean**2)
    var  = var * (g_val / (g_val - 1))
    return float(np.sqrt(max(var, 1e-6)))

def t_over_prob(mu: float, sd: float, line: float, trials=SIM_TRIALS) -> float:
    sd = max(1e-6, float(sd))
    draws = mu + sd * np.random.standard_t(df=5, size=trials)  # heavy tails
    return float((draws > line).mean())

# ---------------- UI: scope ----------------
st.markdown("### 1) Season scope")
c1, c2, c3 = st.columns(3)
with c1:
    season = st.number_input("Season", 2015, 2100, 2025, 1)
with c2:
    seasontype = st.selectbox("Season type", options=[("Regular",2),("Postseason",3),("Preseason",1)], index=0, format_func=lambda x: x[0])[1]
with c3:
    week_lo, week_hi = st.slider("Weeks range", 1, 20, (1, 6))

# ---------------- Build projections ----------------
st.markdown("### 2) Build per-player projections from ESPN")
if st.button("📥 Build NCAA projections"):
    season_df = build_season_agg(season, week_lo, week_hi, seasontype)
    if season_df.empty:
        st.error("No data returned from ESPN for this selection."); st.stop()

    g = season_df["g"].clip(lower=1)

    # Per-game means
    season_df["mu_pass_yds"]        = season_df["pass_yds"]/g
    season_df["mu_pass_tds"]        = season_df["pass_tds"]/g
    season_df["mu_pass_cmp"]        = season_df["pass_cmp"]/g
    season_df["mu_pass_att"]        = season_df["pass_att"]/g
    season_df["mu_rush_yds"]        = season_df["rush_yds"]/g
    season_df["mu_rush_att"]        = season_df["rush_att"]/g
    season_df["mu_receptions"]      = season_df["rec"]/g
    season_df["mu_rec_yds"]         = season_df["rec_yds"]/g
    season_df["mu_rush_rec_yds"]    = (season_df["rush_yds"] + season_df["rec_yds"]) / g
    season_df["mu_fg_made"]         = season_df["fgm"]/g

    # Sample SDs (inflated slightly)
    sd_pass = season_df.apply(lambda r: sample_sd(r["pass_yds"], r["sq_pass_yds"], r["g"]), axis=1)
    sd_rush = season_df.apply(lambda r: sample_sd(r["rush_yds"], r["sq_rush_yds"], r["g"]), axis=1)
    sd_rec  = season_df.apply(lambda r: sample_sd(r["rec"],      r["sq_rec"],      r["g"]), axis=1)
    sd_recyd= season_df.apply(lambda r: sample_sd(r["rec_yds"],  r["sq_rec_yds"],  r["g"]), axis=1)

    season_df["sd_pass_yds"]        = np.maximum(35.0, np.nan_to_num(sd_pass)) * 1.15
    season_df["sd_rush_yds"]        = np.maximum(12.0, np.nan_to_num(sd_rush)) * 1.15
    season_df["sd_receptions"]      = np.maximum(1.2,  np.nan_to_num(sd_rec))  * 1.10
    season_df["sd_rec_yds"]         = np.maximum(15.0, np.nan_to_num(sd_recyd))* 1.15
    season_df["sd_rush_rec_yds"]    = np.sqrt(season_df["sd_rush_yds"]**2 + season_df["sd_rec_yds"]**2)

    # Slims saved into session
    st.session_state["ncaa_proj"] = season_df.copy()
    st.success(f"Built projections for {len(season_df)} players.")
    st.dataframe(season_df[["Player","g","mu_pass_yds","mu_rush_yds","mu_receptions","mu_rec_yds"]].sort_values("Player").head(20), use_container_width=True)
# ---------------- Odds API ----------------
st.markdown("### 3) Pick a game & markets from The Odds API")
api_key = (st.secrets.get("odds_api_key") if hasattr(st, "secrets") else None) or st.text_input(
    "Odds API Key", value="", type="password"
)
region = st.selectbox("Region", ["us","us2","eu","uk"], 0)
lookahead = st.slider("Lookahead days", 0, 7, value=1)
markets = st.multiselect("Markets", VALID_MARKETS, default=VALID_MARKETS)

def odds_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200: raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()
def list_events(api_key: str, lookahead_days: int, region: str):
    return odds_get("https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/events",
                    {"apiKey": api_key, "daysFrom": 0, "daysTo": lookahead_days, "regions": region})
def fetch_props(api_key: str, event_id: str, region: str, markets: List[str]):
    return odds_get(f"https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/events/{event_id}/odds",
                    {"apiKey": api_key, "regions": region, "markets": ",".join(markets), "oddsFormat": "american"})

events = []
if api_key:
    try: events = list_events(api_key, lookahead, region)
    except Exception as e: st.error(f"Events fetch error: {e}")

if not events:
    st.info("Enter your Odds API key to list NCAA games."); st.stop()

event_labels = [f'{e["away_team"]} @ {e["home_team"]} — {e.get("commence_time","")}' for e in events]
pick = st.selectbox("Game", event_labels)
event = events[event_labels.index(pick)]; event_id = event["id"]

# ---------------- Simulate ----------------
st.markdown("### 4) Fetch lines & simulate (per-game means only)")
go = st.button("🎲 Fetch lines & simulate (NCAA)")

if go:
    proj = st.session_state.get("ncaa_proj", pd.DataFrame())
    if proj.empty:
        st.warning("Build NCAA projections first (Step 2)."); st.stop()

    try:
        data = fetch_props(api_key, event_id, region, markets)
    except Exception as e:
        st.error(f"Props fetch failed: {e}"); st.stop()

    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mkey = m.get("key")
            for o in m.get("outcomes", []):
                name = norm_name(o.get("description"))
                side = o.get("name")
                line = o.get("point")
                if mkey not in VALID_MARKETS or not name or not side:
                    continue
                rows.append({"market": mkey, "player": name, "side": side, "point": (None if line is None else float(line))})
    if not rows:
        st.warning("No player outcomes returned for these markets."); st.stop()

    props = (pd.DataFrame(rows).groupby(["market","player","side"], as_index=False)
             .agg(line=("point","median"), books=("point","size")))

    # normalize player names in projections too
    proj = proj.copy(); proj["PN"] = proj["Player"].apply(norm_name)
    proj.set_index("PN", inplace=True)

    out = []
    for _, r in props.iterrows():
        player = r["player"]
        if player not in proj.index:  # skip unmatched
            continue
        pr = proj.loc[player]

        m = r["market"]; line = r["line"]; side = r["side"]

        if m == "player_pass_yds":
            mu, sd = float(pr["mu_pass_yds"]), float(pr["sd_pass_yds"])
        elif m == "player_rush_yds":
            mu, sd = float(pr["mu_rush_yds"]), float(pr["sd_rush_yds"])
        elif m == "player_receptions":
            mu, sd = float(pr["mu_receptions"]), float(pr["sd_receptions"])
        elif m == "player_pass_tds":
            # Poisson for TDs from per-game rate
            lam = float(pr["mu_pass_tds"])
            p_over = float((np.random.poisson(lam=lam, size=SIM_TRIALS) > float(line)).mean()) if line is not None else np.nan
            p = p_over if side == "Over" else (1.0 - p_over)
            out.append({"market": m, "player": player, "side": side, "line": line,
                        "μ (per-game)": round(lam,2), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue
        elif m == "player_rush_reception_yds":
            mu, sd = float(pr["mu_rush_rec_yds"]), float(pr["sd_rush_rec_yds"])
        elif m == "player_rush_attempts":
            mu, sd = float(pr["mu_rush_att"]),  max(1.0, float(pr["mu_rush_att"])**0.5)
        elif m == "player_reception_yds":
            mu, sd = float(pr["mu_rec_yds"]), float(pr["sd_rec_yds"])
        elif m == "player_pass_completions":
            mu, sd = float(pr["mu_pass_cmp"]), max(1.0, float(pr["mu_pass_cmp"])**0.5)
        elif m == "player_pass_attempts":
            mu, sd = float(pr["mu_pass_att"]), max(1.0, float(pr["mu_pass_att"])**0.5)
        elif m == "player_field_goals":
            lam = float(pr["mu_fg_made"])
            p_over = float((np.random.poisson(lam=lam, size=SIM_TRIALS) > float(line)).mean()) if line is not None else np.nan
            p = p_over if side == "Over" else (1.0 - p_over)
            out.append({"market": m, "player": player, "side": side, "line": line,
                        "μ (per-game)": round(lam,2), "σ (per-game)": None,
                        "Win Prob %": round(100*p,2), "books": int(r["books"])})
            continue
        else:
            continue

        if line is None or np.isnan(mu) or np.isnan(sd):
            continue
        p_over = t_over_prob(mu, sd, float(line), SIM_TRIALS)
        p = p_over if side == "Over" else (1.0 - p_over)
        out.append({"market": m, "player": player, "side": side, "line": float(line),
                    "μ (per-game)": round(mu,2), "σ (per-game)": round(sd,2),
                    "Win Prob %": round(100*p,2), "books": int(r["books"])})

    if not out:
        st.warning("No matched props to simulate."); st.stop()

    results = (pd.DataFrame(out)
               .sort_values(["market","Win Prob %"], ascending=[True, False])
               .reset_index(drop=True))

    colcfg = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "side": st.column_config.TextColumn("Side", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.2f"),
        "μ (per-game)": st.column_config.NumberColumn("μ (per-game)", format="%.2f"),
        "σ (per-game)": st.column_config.NumberColumn("σ (per-game)", format="%.2f"),
        "Win Prob %": st.column_config.ProgressColumn("Win Prob %", format="%.2f%%", min_value=0, max_value=100),
        "books": st.column_config.NumberColumn("#Books", width="small"),
    }
    st.dataframe(results, hide_index=True, use_container_width=True, column_config=colcfg)
    st.download_button("⬇️ Download CSV", results.to_csv(index=False).encode("utf-8"), "ncaa_props_results.csv", "text/csv")
