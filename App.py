import streamlit as st
import requests
import pandas as pd

# Load API key from secrets
CFBD_API_KEY = st.secrets["CFBD_API_KEY"]

# Base URL for CollegeFootballData API
BASE_URL = "https://api.collegefootballdata.com"

# Function to fetch team stats
@st.cache_data
def get_cfb_team_stats(year=2025, season_type="regular"):
    url = f"{BASE_URL}/stats/season"
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    params = {"year": year, "seasonType": season_type}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        st.error(f"CFBD API Error: {response.status_code} - {response.text}")
        return pd.DataFrame()
    
    data = response.json()
    
    # Parse into DataFrame
    df = pd.json_normalize(data)
    
    if df.empty:
        st.warning("No data returned. Check year, season type, or API limits.")
    
    return df

# Streamlit app
st.title("🏈 College Football — 2025 (auto from CFBD)")

# Fetch stats
df = get_cfb_team_stats()

if not df.empty:
    # Show raw data
    st.dataframe(df.head(20))
    
    teams = sorted(df["team"].unique())
    home = st.selectbox("Home team", teams)
    away = st.selectbox("Away team", teams)
    
    if home and away and home != away:
        st.write(f"Comparing: **{home} vs {away}**")
else:
    st.error("No CFB stats available. Double-check API key, year, or request limits.")
