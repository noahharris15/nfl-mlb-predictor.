import glob
import importlib.util
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="NBA Model", layout="wide")
st.title("NBA Model")
st.caption("Moneyline + Spread model")

ROOT = Path(__file__).resolve().parent.parent


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_latest_csv() -> Path | None:
    patterns = [
        str(ROOT / "nba_model_edges_full_*.csv"),
        str(ROOT / "nba_best_moneyline_edges_*.csv"),
        str(ROOT / "nba_best_spread_edges_*.csv"),
    ]

    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    if not matches:
        return None

    latest = max(matches, key=lambda p: Path(p).stat().st_mtime)
    return Path(latest)


def run_runner_module():
    runner_path = ROOT / "nba_model_runner.py"
    if not runner_path.exists():
        raise FileNotFoundError(
            "Could not find nba_model_runner.py in the repo root."
        )

    runner = load_module(runner_path, "nba_model_runner")

    # Best case: your runner exposes a clean dataframe function
    if hasattr(runner, "build_main_output"):
        result = runner.build_main_output()
        if isinstance(result, pd.DataFrame):
            return result

    if hasattr(runner, "build_output"):
        result = runner.build_output()
        if isinstance(result, pd.DataFrame):
            return result

    # Fallback: run main() and then load newest CSV
    if hasattr(runner, "main"):
        result = runner.main()
        if isinstance(result, pd.DataFrame):
            return result

        latest_csv = find_latest_csv()
        if latest_csv and latest_csv.exists():
            return pd.read_csv(latest_csv)

        return None

    raise RuntimeError(
        "nba_model_runner.py must contain build_main_output(), build_output(), or main()."
    )


def show_http_error(err: requests.HTTPError):
    status = None
    url = None

    if getattr(err, "response", None) is not None:
        status = err.response.status_code
        url = err.response.url

    if status == 401:
        st.error("Odds API returned 401 Unauthorized.")
        st.info("Check your Odds API key in Streamlit secrets or environment variables.")
        if url:
            st.code(url)
    else:
        st.error(f"HTTP error: {err}")
        if url:
            st.code(url)


col1, col2 = st.columns([1, 1])

with col1:
    run_model = st.button("Run NBA Model", type="primary")

with col2:
    show_saved = st.button("Open Latest Saved CSV")

if show_saved:
    latest_csv = find_latest_csv()
    if latest_csv and latest_csv.exists():
        df = pd.read_csv(latest_csv)
        st.success(f"Loaded: {latest_csv.name}")
        st.dataframe(df, width="stretch")
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=latest_csv.name,
            mime="text/csv",
        )
    else:
        st.warning("No saved NBA CSV files were found yet.")

if run_model:
    with st.spinner("Running NBA model..."):
        try:
            output_df = run_runner_module()

            if isinstance(output_df, pd.DataFrame) and not output_df.empty:
                st.success("NBA model finished.")
                st.dataframe(output_df, width="stretch")

                st.download_button(
                    label="Download CSV",
                    data=output_df.to_csv(index=False).encode("utf-8"),
                    file_name="nba_model_output.csv",
                    mime="text/csv",
                )
            elif isinstance(output_df, pd.DataFrame) and output_df.empty:
                st.warning("The model returned an empty DataFrame.")
            else:
                latest_csv = find_latest_csv()
                if latest_csv and latest_csv.exists():
                    df = pd.read_csv(latest_csv)
                    st.success(f"Model finished. Loaded saved file: {latest_csv.name}")
                    st.dataframe(df, width="stretch")
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=latest_csv.name,
                        mime="text/csv",
                    )
                else:
                    st.warning("The model ran, but no DataFrame or saved CSV was found.")

        except requests.HTTPError as e:
            show_http_error(e)

        except FileNotFoundError as e:
            st.error(str(e))

        except Exception as e:
            st.error(f"NBA page error: {e}")
