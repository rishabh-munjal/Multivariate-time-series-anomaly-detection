# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# ✅ Import the class from anomaly_pipeline.py (NOT from anomaly_detection.py)
from anomaly_pipeline import AnomalyPipeline

st.set_page_config(page_title="IoT Multivariate Anomaly Detector", page_icon="🛰️", layout="wide")

@st.cache_resource
def get_pipeline():
    # One pipeline instance reused across reruns
    return AnomalyPipeline()

def run_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Use your pipeline as-is: write a temp CSV -> pipeline.run() -> read output.
    Keeps your core code untouched and reproducible.
    """
    pipe = get_pipeline()
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input.csv"
        out = Path(td) / "output.csv"
        df_in.to_csv(inp, index=False)
        pipe.run(str(inp), str(out))
        return pd.read_csv(out)

st.title("IoT Multivariate Anomaly Detector")
st.caption("Upload a CSV (must include a time/date/timestamp column) or run the bundled sample.")

with st.sidebar:
    st.header("Settings")
    use_sample = st.toggle("Use bundled input.csv", value=True)

tab_run, tab_explore = st.tabs(["Run Detection", "Explore Output"])

with tab_run:
    uploaded = None if use_sample else st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Run", use_container_width=True):
        if use_sample:
            p = Path("input.csv")
            if not p.exists():
                st.error("input.csv not found in repo.")
                st.stop()
            df_in = pd.read_csv(p)
        else:
            if not uploaded:
                st.warning("Upload a CSV or enable bundled sample.")
                st.stop()
            df_in = pd.read_csv(uploaded)

        with st.spinner("Running pipeline…"):
            df_out = run_pipeline(df_in)

        st.success("Done.")
        st.subheader("Preview")
        st.dataframe(df_out.head(200), use_container_width=True)

        # Try to identify the timestamp column chosen by your pipeline
        ts_col = next((c for c in df_out.columns if any(k in c.lower() for k in ["time","date","timestamp"])), None)
        if ts_col and "abnormality_score" in df_out.columns:
            # Best-effort datetime conversion for nicer axis
            try:
                df_out[ts_col] = pd.to_datetime(df_out[ts_col])
            except Exception:
                pass
            st.subheader("Abnormality Score over Time")
            st.line_chart(df_out.set_index(ts_col)["abnormality_score"])

        st.download_button("Download Output CSV", df_out.to_csv(index=False).encode(), "output.csv")
        st.session_state["last_output"] = df_out

        # Top-features glance for highest anomalies
        feat_cols = [f"top_feature_{i}" for i in range(1, 8) if f"top_feature_{i}" in df_out.columns]
        if feat_cols and "abnormality_score" in df_out.columns:
            st.subheader("Top Contributing Features (peak anomalies)")
            peaks = df_out.nlargest(10, "abnormality_score")
            idx = st.slider("Pick a row within Top-10", 1, len(peaks), 1)
            row = peaks.iloc[idx-1]
            tags = [t for t in row[feat_cols].tolist() if isinstance(t, str) and t]
            if tags:
                counts = pd.Series(tags).value_counts()
                st.bar_chart(counts)

with tab_explore:
    df = st.session_state.get("last_output")
    if df is None:
        st.info("Run detection first, or upload a ready output.csv below.")
        f = st.file_uploader("Upload output.csv", type=["csv"], key="out_csv")
        if f:
            df = pd.read_csv(f)
            st.session_state["last_output"] = df
    if df is not None:
        st.dataframe(df, use_container_width=True)
        # Free-form plotting
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric:
            pick = st.selectbox("Plot any numeric column", numeric, index=numeric.index("abnormality_score") if "abnormality_score" in numeric else 0)
            st.line_chart(df[pick])
