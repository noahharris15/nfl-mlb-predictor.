# odds_props_probe.py
import os, sys, time, json, datetime as dt
import requests
import pandas as pd

# ---- YOUR KEY ----
API_KEY = "7401399bd14e8778312da073b621094f"  # <-- replace if desired

BASE = "https://api.the-odds-api.com/v4"
REGION = "us"
ODDS_FORMAT = "decimal"
BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "caesars"]  # tweak as you like

# Markets to probe by league (names follow The Odds API docs)
MARKETS = {
    "americanfootball_nfl": [
        "player_pass_yds", "player_rush_yds",
        "player_reception_yds", "player_receptions",
        # Add more if needed:
        # "player_pass_tds", "player_rush_tds", "player_anytime_td"
    ],
    "basketball_nba": [
        "player_points", "player_rebounds", "player_assists",
        "player_three_points_made"
    ],
    "baseball_mlb": [
        "player_hits", "player_home_runs", "player_rbis", "player_strikeouts"
    ],
    # Soccer player props are often not available; uncomment to test
    # "soccer_epl": ["player_goals", "player_assists", "player_shots_on_target"]
}

# ------------- helpers -------------
def _get(url, params, max_retries=3, backoff=1.0):
    last = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r
            # 429 or transient: brief backoff
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (2 ** i))
                last = r
                continue
            # hard error: stop now
            r.raise_for_status()
        except requests.HTTPError as e:
            # bubble up with body snippet
            msg = f"HTTPError {getattr(last or r, 'status_code', '??')}: {(last or r).text[:300]}"
            raise RuntimeError(msg) from e
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    if isinstance(last, requests.Response):
        raise RuntimeError(f"HTTP {last.status_code}: {last.text[:300]}")
    raise RuntimeError(str(last))

def _now_iso():
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def pick_upcoming_event(sport_key):
    """Pick one upcoming event (needed because player props are returned per-event)."""
    url = f"{BASE}/sports/{sport_key}/events"
    params = {"apiKey": API_KEY, "regions": REGION, "dateFormat": "iso"}
    r = _get(url, params)
    events = r.json()
    if not events:
        return None
    # choose the next event in the future if possible
    now = dt.datetime.utcnow()
    def _parse(t):
        try:
            return dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            return now
    future = sorted([e for e in events if _parse(e.get("commence_time", "")) > now],
                    key=lambda e: _parse(e.get("commence_time", "9999-12-31T00:00:00Z")))
    return (future or [events[0]])[0]

def test_event_markets(sport_key, markets):
    """Hit the /events/{id}/odds endpoint (primary method for props)."""
    ev = pick_upcoming_event(sport_key)
    if not ev:
        return None, [{"sport": sport_key, "market": m, "endpoint": "event_odds",
                       "status": "no_events", "offers": 0, "notes": "No events returned"} for m in markets]
    event_id = ev["id"]
    joined = ",".join(markets)
    url = f"{BASE}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY, "regions": REGION, "oddsFormat": ODDS_FORMAT,
        "markets": joined, "bookmakers": ",".join(BOOKMAKERS),
        "includeLinks": "false"
    }
    try:
        r = _get(url, params)
        js = r.json()
        out = []
        # Count how many bookmakers actually returned each market
        avail = {m: 0 for m in markets}
        # structure: js["bookmakers"][..]["markets"][..]["key"]
        for bk in js.get("bookmakers", []):
            for mk in bk.get("markets", []):
                k = mk.get("key")
                if k in avail:
                    # if there are any outcomes, we call it "has offers"
                    outcomes = mk.get("outcomes", []) or mk.get("outcomesAmerican", [])
                    if outcomes:
                        avail[k] += 1
        for m in markets:
            status = "supported" if avail[m] > 0 else "no_offers"
            out.append({"sport": sport_key, "market": m, "endpoint": "event_odds",
                        "status": status, "offers": avail[m],
                        "notes": "" if avail[m] else "No offers from selected bookmakers"})
        return ev, out
    except RuntimeError as e:
        msg = str(e)
        status = "error_422_invalid_market" if "422" in msg and "invalid" in msg.lower() else "http_error"
        return ev, [{"sport": sport_key, "market": m, "endpoint": "event_odds",
                     "status": status, "offers": 0, "notes": msg[:240]} for m in markets]

def test_regular_odds(sport_key, markets):
    """Try the generic /odds endpoint too (some markets may work there)."""
    url = f"{BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY, "regions": REGION, "oddsFormat": ODDS_FORMAT,
        "markets": ",".join(markets), "bookmakers": ",".join(BOOKMAKERS),
        "includeLinks": "false"
    }
    try:
        r = _get(url, params)
        js = r.json()
        # structure: list of events, each with bookmakers/markets
        avail = {m: 0 for m in markets}
        for ev in js:
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    k = mk.get("key")
                    if k in avail and (mk.get("outcomes") or mk.get("outcomesAmerican")):
                        avail[k] += 1
        out = []
        for m in markets:
            status = "supported" if avail[m] > 0 else "no_offers"
            out.append({"sport": sport_key, "market": m, "endpoint": "odds",
                        "status": status, "offers": avail[m],
                        "notes": "" if avail[m] else "No offers across returned events"})
        return out
    except RuntimeError as e:
        msg = str(e)
        status = "error_422_invalid_market" if "422" in msg and "invalid" in msg.lower() else "http_error"
        return [{"sport": sport_key, "market": m, "endpoint": "odds",
                 "status": status, "offers": 0, "notes": msg[:240]} for m in markets]

def main():
    rows = []
    print(f"Probing player props with key ending …{API_KEY[-6:]}  (region={REGION}, books={','.join(BOOKMAKERS)})")
    for sport_key, markets in MARKETS.items():
        print(f"\n=== {sport_key} ===")
        event, event_results = test_event_markets(sport_key, markets)
        rows.extend(event_results)
        if event:
            et = event.get("commence_time")
            hn = event.get("home_team")
            an = event.get("away_team")
            print(f"Testing event: {hn} vs {an} at {et}")
        # Also try generic odds endpoint (sometimes returns something the event endpoint doesn’t)
        reg_results = test_regular_odds(sport_key, markets)
        rows.extend(reg_results)
        # quick console view
        for r in event_results + reg_results:
            print(f"  [{r['endpoint']}] {r['market']}: {r['status']} (offers={r['offers']})")

    df = pd.DataFrame(rows)
    df = df[["sport", "market", "endpoint", "status", "offers", "notes"]]
    df.sort_values(["sport", "market", "endpoint"], inplace=True)
    out = "odds_api_player_prop_probe.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved detailed report to: {out}")
    # brief summary pivot
    pivot = df.pivot_table(index=["sport","market"],
                           columns="endpoint",
                           values="offers",
                           aggfunc="max",
                           fill_value=0).reset_index()
    print("\nSummary (max offers per endpoint):")
    print(pivot.to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
