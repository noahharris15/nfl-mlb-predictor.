# --- smoke_test.py (TEMP MAIN) ---
import streamlit as st, sys, os, time, platform
st.set_page_config(page_title="SMOKE TEST", layout="wide")
st.title("✅ NEW BUILD RUNNING")

st.write("**__file__**:", __file__)
st.write("**CWD**:", os.getcwd())
st.write("**Files here**:", sorted(os.listdir(".")))
st.write("**Pages/**:", sorted(os.listdir("pages")) if os.path.isdir("pages") else "No pages/ dir")
st.write("**Python**:", sys.version)
st.write("**Timestamp**:", time.strftime("%Y-%m-%d %H:%M:%S"))
st.write("**Host**:", platform.node())

st.success("If you can read this, the deployment is using THIS file as the main entry point.")
