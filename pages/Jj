import streamlit as st
import pandas as pd
import importlib.util
from pathlib import Path

st.set_page_config(page_title="NBA O/U Model", layout="wide")
st.title("NBA Over / Under Model")

def load_module(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if st.button("Run O/U Model"):
    with st.spinner("Running O/U model..."):
        try:
            backend = load_module("pages/nba_ou_model_backend.py", "nba_ou_model_backend")
            output_df = backend.build_ou_output()

            st.success("O/U model finished.")
            st.dataframe(output_df, width="stretch")

            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="nba_ou_model_output.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error: {e}")
