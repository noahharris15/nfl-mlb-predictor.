# ========================= CFB (College Football) =============================
# Auto-pull 2025 team scoring from CollegeFootballData (CFBD)
# Shows diagnostics (key presence, response status) and falls back to CSV if needed.

import json
import requests

CFB_SEASON = 2025

def _cfbd_key_info():
    try:
        key = st.secrets.get("CFBD_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        return "", False, "No key in st.secrets"
    # non-sensitive preview (first/last 3 chars + length)
    preview = f"{key[:3]}…{key[-3:]} (len={len(key)})"
    return key, True, preview

@st.cache_data(ttl=1800, show_spinner=False)  # cache 30 min
def _cfbd_request(path: str, params: dict | None = None) -> tuple[int, dict | list | str]:
    """Return (status_code, json_or_text)."""
    base = "https://api.collegefootballdata.com"
    key = st.secrets.get("CFBD_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": "streamlit-cfb-ppg/1.0",
    }
    url = f"{base}{path}"
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=15)
        ct = r.headers.get("Content-Type","")
        if "application/json" in ct:
            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, {"parse_error": True, "text": r.text}
        else:
            return r.status_code, r.text
    except requests.RequestException as e:
        return 0, {"error": str(e)}

@st.cache_data(ttl=1800, show_spinner=False)
def cfb_team_stats_2025() -> pd.DataFrame:
    """
    Build a simple team table with off_ppg and def_ppg for the season.
    Uses CFBD's /stats/season endpoint with teamType=fbs+fcs for completeness.
    """
    # offense
    code_off, off = _cfbd_request(
        "/stats/season",
        {"year": CFB_SEASON, "teamType": "both", "side": "offense"}
    )
    # defense
    code_def, deff = _cfbd_request(
        "/stats/season",
        {"year": CFB_SEASON, "teamType": "both", "side": "defense"}
    )

    if code_off != 200 or code_def != 200:
        # bubble up details so UI can show them
        raise RuntimeError(json.dumps({
            "off_status": code_off, "def_status": code_def,
            "off_sample": (off[:1] if isinstance(off, list) else off),
            "def_sample": (deff[:1] if isinstance(deff, list) else deff),
        }, default=str))

    # CFBD returns a list of dicts; pointsPerGame can be nested under 'stat' or flat
    def _ppg_from_rows(rows):
        out = {}
        for r in rows or []:
            team = r.get("team")
            ppg = (r.get("pointsPerGame")
                   or r.get("stat",{}).get("pointsPerGame")
                   or r.get("ppa",{}).get("pointsPerGame"))
            # also check common alt names that appear in some responses
            if ppg is None:
                ppg = (r.get("scoring",{}).get("pointsPerGame")
                       or r.get("scoring",{}).get("ptsPerGame"))
            if team and ppg is not None:
                try:
                    out[team] = float(ppg)
                except Exception:
                    continue
        return out

    off_map = _ppg_from_rows(off if isinstance(off, list) else [])
    def_map = _ppg_from_rows(deff if isinstance(deff, list) else [])

    # Build DF
    teams = sorted(set(off_map) | set(def_map))
    if not teams:
        raise RuntimeError("CFBD returned empty team list (after parse).")

    df = pd.DataFrame({
        "team": teams,
        "off_ppg": [off_map.get(t, 28.0) for t in teams],
        "def_ppg": [def_map.get(t, 28.0) for t in teams],
    })
    # Gentle shrink toward 28 to stabilize tiny samples early season
    df["off_ppg"] = 0.9*df["off_ppg"] + 0.1*28.0
    df["def_ppg"] = 0.9*df["def_ppg"] + 0.1*28.0
    return df

def _cfb_matchup_mu(rates: pd.DataFrame, home: str, away: str) -> tuple[float, float]:
    rH = rates.loc[rates["team"] == home]
    rA = rates.loc[rates["team"] == away]
    if rH.empty or rA.empty:
        raise ValueError(f"Unknown CFB team(s): {home}, {away}")
    H, A = rH.iloc[0], rA.iloc[0]
    mu_home = max(EPS, (H["off_ppg"] + A["def_ppg"]) / 2.0)
    mu_away = max(EPS, (A["off_ppg"] + H["def_ppg"]) / 2.0)
    return mu_home, mu_away

# ---------------------------- CFB UI block ------------------------------------
elif page == "College Football":
    st.subheader("🏈🎓 College Football — 2025 (auto from CFBD)")

    key, has_key, key_preview = _cfbd_key_info()
    with st.expander("Diagnostics", expanded=False):
        st.write(f"Key present: **{has_key}**")
        if has_key:
            st.write(f"Key preview: `{key_preview}`")
        if st.button("Clear CFB cache"):
            st.cache_data.clear()
            st.success("Cleared CFB cache. Re-run the app.")

    if not has_key:
        st.error("No `CFBD_API_KEY` found in Secrets. Add it in Streamlit Secrets.")
        st.stop()

    # Try to load CFBD
    try:
        rates = cfb_team_stats_2025()
    except Exception as e:
        st.error("CFBD request failed or returned no data. Details below.")
        with st.expander("Error details (for debugging)"):
            st.code(str(e))
        st.info("You can still use a CSV fallback (team, off_ppg, def_ppg).")
        up = st.file_uploader("Upload CFB CSV fallback", type=["csv","xlsx"])
        if up is None:
            st.stop()
        df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        need = {"team","off_ppg","def_ppg"}
        if not need.issubset({c.lower() for c in df.columns}):
            st.error("CSV must contain columns: team, off_ppg, def_ppg.")
            st.stop()
        # normalize col names
        cols = {c.lower(): c for c in df.columns}
        rates = df.rename(columns={
            cols["team"]: "team", cols["off_ppg"]: "off_ppg", cols["def_ppg"]: "def_ppg"
        })[["team","off_ppg","def_ppg"]]

    # Normal CFB UI
    home = st.selectbox("Home team", sorted(rates["team"].unique().tolist()))
    away = st.selectbox("Away team", sorted([t for t in rates["team"].unique().tolist() if t != home]))

    mu_h, mu_a = _cfb_matchup_mu(rates, home, away)
    pH, pA, mH, mA = _poisson_sim(mu_h, mu_a)
    st.markdown(
        f"**{home}** vs **{away}** — Expected points: {mH:.1f}–{mA:.1f} · "
        f"P({home} win) = **{100*pH:.1f}%**, P({away} win) = **{100*pA:.1f}%**"
    )

    with st.expander("Show team table"):
        st.dataframe(rates.sort_values("team").reset_index(drop=True))
