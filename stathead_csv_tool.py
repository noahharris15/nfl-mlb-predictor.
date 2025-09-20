# stathead_csv_tool.py
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stathead → CSV", layout="wide")

def parse_stathead_text(text: str, columns: list[str]) -> pd.DataFrame:
    """Parse Stathead 'copy text' dumps into a DataFrame based on a column header list."""
    if not text.strip():
        return pd.DataFrame(columns=columns)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hdr = columns

    # Find all places the header repeats (Stathead prints it every 25 rows)
    starts, i = [], 0
    while i <= len(lines) - len(hdr):
        if lines[i:i+len(hdr)] == hdr:
            starts.append(i + len(hdr))
            i += len(hdr)
        else:
            i += 1

    rows = []
    for start in starts:
        i = start
        while i + len(hdr) <= len(lines):
            # stop at the next header
            if lines[i:i+len(hdr)] == hdr:
                break
            chunk = lines[i:i+len(hdr)]
            if len(chunk) < len(hdr):  # partial tail
                break
            # first col should be rank
            if not re.fullmatch(r"\d+", chunk[0]):
                break
            rows.append(chunk)
            i += len(hdr)

    return pd.DataFrame(rows, columns=columns)

def render_stathead_csv_page():
    st.title("📥 Stathead → CSV (paste text, download files)")
    st.caption("Paste the **text** version of the Stathead tables (RB Rushing, WR/TE Receiving).")

    tab1, tab2 = st.tabs(["RB — Rushing", "WR/TE — Receiving"])

    with tab1:
        st.subheader("RB Rushing (2025)")
        rb_text = st.text_area(
            "Paste the RB **Rushing** block here",
            placeholder="Paste the entire Rushing table text…",
            height=220,
            key="rb_text",
        )
        if st.button("Build RB CSV", key="rb_btn"):
            rb_cols = ["Rk","Player","Age","Team","Pos","G","GS","Att","Yds","TD","1D","Succ%","Lng","Y/A","Y/G","A/G","Fmb","Awards"]
            rb_df = parse_stathead_text(rb_text, rb_cols)
            st.dataframe(rb_df, use_container_width=True)
            st.download_button(
                "Download RB CSV",
                rb_df.to_csv(index=False).encode(),
                file_name="nfl_2025_rb_rushing.csv",
                mime="text/csv",
            )

    with tab2:
        st.subheader("WR/TE Receiving (2025)")
        wr_text = st.text_area(
            "Paste the WR/TE **Receiving** block here",
            placeholder="Paste the entire Receiving table text…",
            height=220,
            key="wr_text",
        )
        if st.button("Build WR/TE CSV", key="wr_btn"):
            wr_cols = ["Rk","Player","Age","Team","Pos","G","GS","Tgt","Rec","Yds","Y/R","TD","1D","Succ%","Lng","R/G","Y/G","Ctch%","Y/Tgt","Fmb","Awards"]
            wr_df = parse_stathead_text(wr_text, wr_cols)
            st.dataframe(wr_df, use_container_width=True)
            st.download_button(
                "Download WR/TE CSV",
                wr_df.to_csv(index=False).encode(),
                file_name="nfl_2025_receiving.csv",
                mime="text/csv",
            )

# If this is the entry file on Streamlit, render now:
if __name__ == "__main__":
    render_stathead_csv_page()
