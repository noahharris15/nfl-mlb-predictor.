import streamlit as st
st.sidebar.button("🧹 Force clear cache", on_click=lambda: (st.cache_data.clear(), st.cache_resource.clear()))
