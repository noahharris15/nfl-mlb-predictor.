import os, requests, streamlit as st

st.set_page_config(page_title="CFBD Sanity Check", layout="centered")
st.title("🔎 CFBD Sanity Check")

# 1) Read key from secrets or env
key = st.secrets.get("CFBD_API_KEY") if "CFBD_API_KEY" in st.secrets else os.getenv("CFBD_API_KEY")
key = (key or "").strip()
st.write("Key present:", bool(key))
if key:
    st.write("Key preview:", key[:3] + "…" + key[-3:], f"(len={len(key)})")
else:
    st.error("No CFBD_API_KEY found. In Streamlit **Secrets** add:\n\n`CFBD_API_KEY = \"your-key\"`")
    st.stop()

# 2) Hit a stable endpoint with known data (2024 regular season)
url = "https://api.collegefootballdata.com/games"
params = {"year": 2024, "seasonType": "regular"}
headers = {"Authorization": f"Bearer {key}"}

try:
    r = requests.get(url, headers=headers, params=params, timeout=25)
    st.write("HTTP status:", r.status_code)
    # If response isn't JSON, this will throw — exactly what we want to see
    js = r.json()
    st.success(f"OK! Parsed JSON. Game count: {len(js)}")
    # Show a tiny sample
    st.json(js[:3])
except Exception as e:
    st.error(f"Request failed or was not JSON: {e}")
    # Show raw text to see CFBD’s message (429 = rate limit, 401 = auth, 5xx = server)
    with st.expander("Raw response text"):
        st.code(r.text if 'r' in locals() else "(no response)")
