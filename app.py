import streamlit as st
import pandas as pd
from google import genai
from data_pipeline import run_pipeline_universal

st.set_page_config(page_title="Clean Data, Explained", page_icon="🧹", layout="centered")
st.title("🧹 Clean Data, Explained")
st.caption("Upload any CSV. Get a cleaned file and a plain-English report of every issue found.")

client = genai.Client(api_key=st.secrets["Gemini_Key"])
MODEL = "gemini-3.6-flash"

uploaded = st.file_uploader("Upload your CSV", type="csv")
use_sample = st.button("...or use the sample dataset")

raw_df = None
if uploaded:
    raw_df = pd.read_csv(uploaded)
elif use_sample:
    raw_df = pd.read_csv("orders_dirty.csv")

if raw_df is not None:
    with st.spinner("Analyzing your data and generating a report..."):
        try:
            clean_df, issues_df, stats, schema_info = run_pipeline_universal(
                raw_df, client=client, model=MODEL
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows before", stats["rows_before"])
            col2.metric("Rows after", stats["rows_after"])
            col3.metric("Rows removed", stats["rows_removed"])
            col4.metric("Unique rows affected", stats["unique_rows_affected"])

            if stats["validation_passed"]:
                st.success("Post-clean validation passed — no remaining rule violations found in the output.")
            else:
                st.warning("Post-clean validation found remaining issues — see details below. The exported file may still need review.")
                for v in stats["remaining_violations"]:
                    st.write(f"- {v}")

            with st.expander("How this file was understood"):
                st.write(f"**Deduplication method:** {stats['dedupe_method']}")
                st.write("**Detected column types:**")
                st.json(stats["detected_column_types"])
                st.write("**Inferred semantic roles:**")
                st.json(stats["semantic_roles"])
                if stats["ai_suggested_checks"]:
                    st.write("**AI-suggested business rules (validated and applied):**")
                    for c in stats["ai_suggested_checks"]:
                        st.write(f"- `{c['column_a']} {c['operator']} {c['column_b']}` — {c['reason']}")
                else:
                    st.write("No cross-column business rules were suggested for this file.")

            st.subheader("Data Quality Report")

            if len(issues_df) == 0:
                st.success("No issues found — data looks clean!")
            else:
                errors = issues_df[issues_df["severity"] == "error"]
                anomalies = issues_df[issues_df["severity"] == "warning"]
                changes = issues_df[issues_df["severity"] == "info"]

                if len(errors) > 0:
                    st.markdown("#### 🔴 Errors (removed from the cleaned file)")
                    for category, group in errors.groupby("category"):
                        with st.expander(f"{category} — {len(group)} rows affected"):
                            st.dataframe(group.head(10))

                if len(anomalies) > 0:
                    st.markdown("#### 🟡 Statistical Anomalies (flagged, not removed)")
                    for category, group in anomalies.groupby("category"):
                        with st.expander(f"{category} — {len(group)} rows affected"):
                            st.dataframe(group.head(10))

                if len(changes) > 0:
                    st.markdown("#### 🔵 Standardization Changes (info only)")
                    for category, group in changes.groupby("category"):
                        with st.expander(f"{category} — {len(group)} rows affected"):
                            st.dataframe(group.head(10))

            st.subheader("Download")
            st.download_button("Download cleaned CSV", clean_df.to_csv(index=False), "cleaned_data.csv")
            st.download_button("Download issues detail CSV", issues_df.to_csv(index=False), "issues_detail.csv")

        except Exception as e:
            st.error(f"Something went wrong processing this file: {e}")
            st.info("Try a different file, or check that it's a standard CSV with a header row.")

st.divider()
st.caption("This tool auto-detects column types, semantic roles, and business rules — errors are removed, anomalies are flagged for review.")
