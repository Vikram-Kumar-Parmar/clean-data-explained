"""
data_pipeline.py
Universal, schema-agnostic data cleaning pipeline.
No Colab-specific code here — this file must work identically whether it's
imported from a Colab notebook or from a Streamlit app on Streamlit Cloud.
The Gemini client is created by whoever CALLS this module (app.py or your
notebook) and passed in as a parameter — never created inside this file.
"""

import pandas as pd
import numpy as np
import re
import json
from rapidfuzz import fuzz
from google.genai import types


# ---------- Part A: column type detection ----------

def try_convert_to_numeric(series: pd.Series) -> tuple[pd.Series, bool]:
    """
    Attempts to convert a text column to numbers by stripping common
    currency symbols and thousands separators. Returns the converted
    series and whether it succeeded on most non-null values.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[$£€,]", "", regex=True)
        .str.strip()
    )
    converted = pd.to_numeric(cleaned, errors="coerce")
    non_null_original = series.notna().sum()
    non_null_converted = converted.notna().sum()
    if non_null_original == 0:
        return series, False
    success_rate = non_null_converted / non_null_original
    return converted, success_rate > 0.85  # 85%+ converted cleanly = treat as numeric


def try_convert_to_datetime(series: pd.Series) -> tuple[pd.Series, bool]:
    """Attempts to parse a text column as dates, mixed formats allowed."""
    converted = pd.to_datetime(series, errors="coerce", format="mixed")
    non_null_original = series.notna().sum()
    non_null_converted = converted.notna().sum()
    if non_null_original == 0:
        return series, False
    success_rate = non_null_converted / non_null_original
    return converted, success_rate > 0.85


def infer_and_clean_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Goes column by column. If a column is already numeric/datetime, leave it.
    If it's text, try converting it to numeric, then to datetime, then give up
    and leave it as a text column. Returns the cleaned df plus a dict recording
    what each column was detected as.
    """
    df = df.copy()
    column_types = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "datetime"
            continue

        numeric_attempt, is_numeric = try_convert_to_numeric(df[col])
        if is_numeric:
            df[col] = numeric_attempt
            column_types[col] = "numeric"
            continue

        datetime_attempt, is_datetime = try_convert_to_datetime(df[col])
        if is_datetime:
            df[col] = datetime_attempt
            column_types[col] = "datetime"
            continue

        column_types[col] = "text"

    return df, column_types


# ---------- Part B: universal duplicate detection ----------

def detect_id_column(df: pd.DataFrame) -> str | None:
    """
    Looks for a column that's a good candidate for a unique row identifier:
    mostly unique values, and ideally named something like an ID.
    """
    candidates = []
    for col in df.columns:
        uniqueness = df[col].nunique() / max(len(df), 1)
        name_hint = bool(re.search(r"(^id$|_id$|^id_|order|transaction|record)", col.lower()))
        if uniqueness > 0.95:
            candidates.append((col, uniqueness, name_hint))

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], c[1]), reverse=True)
    return candidates[0][0]


def deduplicate_universal(df: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    df = df.copy()
    before = len(df)
    id_col = detect_id_column(df)

    if id_col:
        df = df.drop_duplicates(subset=[id_col], keep="first")
        method = f"deduplicated on detected ID column: '{id_col}'"
    else:
        df = df.drop_duplicates(keep="first")
        method = "deduplicated on full-row exact matches (no clear ID column found)"

    return df, before - len(df), method


# ---------- Part C: universal numeric outlier detection ----------

def detect_outliers_universal(df: pd.DataFrame, column_types: dict) -> list[dict]:
    issues = []
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper_bound = q3 + 3 * iqr
        lower_bound = q1 - 3 * iqr

        outlier_rows = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
        for idx in outlier_rows.index:
            issues.append({
                "row_index": idx,
                "category": f"Outlier in '{col}'",
                "detail": f"{col} = {df.at[idx, col]} (expected roughly {lower_bound:.1f} to {upper_bound:.1f})"
            })
    return issues


# ---------- Part D: universal near-duplicate category detection ----------

def find_near_duplicate_groups(values: list[str], threshold: int = 85) -> dict:
    """
    Groups similar text values (e.g. 'USA' and 'US') and maps each to the
    longest/fullest-looking value in its group. Returns {original: canonical}.
    """
    unique_vals = list(set(str(v) for v in values if pd.notna(v)))
    groups = []
    used = set()

    for val in unique_vals:
        if val in used:
            continue
        group = [val]
        used.add(val)
        for other in unique_vals:
            if other in used:
                continue
            if fuzz.ratio(val.lower(), other.lower()) >= threshold:
                group.append(other)
                used.add(other)
        groups.append(group)

    mapping = {}
    for group in groups:
        if len(group) > 1:
            canonical = max(group, key=len)
            for v in group:
                mapping[v] = canonical
    return mapping


def standardize_categoricals_universal(df: pd.DataFrame, column_types: dict) -> tuple[pd.DataFrame, list[dict]]:
    df = df.copy()
    changes = []
    text_cols = [c for c, t in column_types.items() if t == "text"]

    for col in text_cols:
        unique_count = df[col].nunique()
        # Skip free-text columns (like names) — real categories repeat a lot,
        # free text rarely does
        if unique_count == 0 or unique_count / max(len(df), 1) > 0.5:
            continue

        mapping = find_near_duplicate_groups(df[col].dropna().unique().tolist())
        if mapping:
            before_values = df[col].copy()
            df[col] = df[col].replace(mapping)
            changed_idx = df.index[df[col] != before_values]
            for idx in changed_idx:
                changes.append({
                    "row_index": idx,
                    "category": f"Standardized value in '{col}'",
                    "detail": f"'{before_values.at[idx]}' -> '{df.at[idx, col]}'"
                })
    return df, changes


# ---------- Part E: AI schema inspection ----------
# NOTE: this module takes an already-created Gemini `client` and `model` as
# parameters. It never creates its own client and never imports anything
# Colab-specific — that keeps this file usable from both Colab and Streamlit.

SCHEMA_SYSTEM_PROMPT = """You are a data analyst inspecting an unfamiliar dataset.
You will be given column names, their detected data types, and a few sample rows.
Respond with ONLY valid JSON (no markdown, no explanation) in this exact structure:

{
  "column_roles": {"column_name": "likely_meaning", ...},
  "suggested_checks": [
    {"column_a": "...", "operator": "<=", "column_b": "...", "reason": "..."}
  ]
}

Rules:
- Only suggest a check in "suggested_checks" if both columns are numeric and the
  relationship is a genuine, common-sense business rule (e.g. a refund shouldn't
  exceed an order total, a start date shouldn't be after an end date).
- valid "operator" values: "<=", ">=", "<", ">"
- If nothing sensible applies, return an empty list for "suggested_checks".
- Keep "reason" to one short sentence.
"""


def inspect_schema(df: pd.DataFrame, column_types: dict, client, model: str) -> dict:
    sample_rows = df.head(3).to_dict(orient="records")
    user_prompt = f"""Columns and detected types:
{json.dumps(column_types, indent=2)}

Sample rows:
{json.dumps(sample_rows, indent=2, default=str)}
"""
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SCHEMA_SYSTEM_PROMPT,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception:
        # Safe fallback — universal cleaning (Parts A-D) still works fully
        # even if the AI schema-inspection call fails for any reason.
        return {"column_roles": {}, "suggested_checks": []}


# ---------- Part F: apply AI-suggested business-rule checks ----------

def apply_suggested_checks(df: pd.DataFrame, suggested_checks: list[dict]) -> list[dict]:
    issues = []
    ops = {
        "<=": lambda a, b: a > b,
        ">=": lambda a, b: a < b,
        "<":  lambda a, b: a >= b,
        ">":  lambda a, b: a <= b,
    }
    for check in suggested_checks:
        col_a, col_b, op = check.get("column_a"), check.get("column_b"), check.get("operator")
        if col_a not in df.columns or col_b not in df.columns or op not in ops:
            continue  # AI hallucinated a column name or bad operator — skip safely
        if not pd.api.types.is_numeric_dtype(df[col_a]) or not pd.api.types.is_numeric_dtype(df[col_b]):
            continue

        violated = df[ops[op](df[col_a], df[col_b])]
        for idx in violated.index:
            issues.append({
                "row_index": idx,
                "category": f"Business rule violation: {col_a} {op} {col_b}",
                "detail": f"{check.get('reason', '')} ({col_a}={df.at[idx, col_a]}, {col_b}={df.at[idx, col_b]})"
            })
    return issues


# ---------- Part G: the full pipeline ----------

def run_pipeline_universal(raw_df: pd.DataFrame, client=None, model: str = "gemini-3.6-flash"):
    """
    Runs the full universal cleaning pipeline on any DataFrame.

    Parameters:
      raw_df : the uploaded/raw data
      client : an already-created genai.Client, e.g.
                 genai.Client(api_key=st.secrets["Gemini_Key"])   in Streamlit
                 genai.Client(api_key=userdata.get("Gemini_Key")) in Colab
               If None, the AI schema-inspection step is skipped entirely and
               only the pure-pandas universal cleaning (Parts A-D) runs.
      model  : Gemini model name to use for schema inspection.

    Returns: clean_df, issues_df, stats, schema_info
    """
    df, column_types = infer_and_clean_types(raw_df)
    df, dupes_removed, dedupe_method = deduplicate_universal(df)
    df, category_changes = standardize_categoricals_universal(df, column_types)

    all_issues = []
    all_issues.extend(detect_outliers_universal(df, column_types))
    all_issues.extend(category_changes)

    if client is not None:
        schema_info = inspect_schema(df, column_types, client, model)
    else:
        schema_info = {"column_roles": {}, "suggested_checks": []}

    all_issues.extend(apply_suggested_checks(df, schema_info.get("suggested_checks", [])))

    issues_df = pd.DataFrame(all_issues)

    stats = {
        "rows_before": len(raw_df),
        "rows_after": len(df),
        "duplicates_removed": dupes_removed,
        "dedupe_method": dedupe_method,
        "total_issues_flagged": len(issues_df),
        "detected_column_types": column_types,
        "ai_suggested_checks": schema_info.get("suggested_checks", []),
    }
    return df, issues_df, stats, schema_info
