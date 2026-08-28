# pipeline.py
# Pure data-cleaning logic — no web/UI code lives here.

import pandas as pd
import numpy as np

COUNTRY_MAP = {
    "USA": "United States", "U.S.": "United States", "US": "United States",
    "UK": "United Kingdom", "U.K.": "United Kingdom", "Britain": "United Kingdom",
}

def clean_price_column(df):
    df = df.copy()
    df["unit_price"] = (
        df["unit_price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.strip()
        .astype(float)
    )
    return df

def parse_messy_dates(df):
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")
    return df

def standardize_countries(df):
    df = df.copy()
    df["country_original"] = df["country"]
    df["country"] = df["country"].replace(COUNTRY_MAP)
    return df

def deduplicate(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    return df, before - len(df)

def detect_issues(df):
    issues = []
    def flag(idx, category, detail):
        issues.append({"order_id": df.at[idx, "order_id"], "category": category, "detail": detail})

    for idx in df[df["customer_name"].isna()].index:
        flag(idx, "Missing Customer Name", "customer_name is blank")

    for idx in df[df["quantity"] < 0].index:
        flag(idx, "Negative Quantity", f"quantity = {df.at[idx,'quantity']}")

    q1, q3 = df["quantity"].quantile([0.25, 0.75])
    upper_bound = q3 + 3 * (q3 - q1)
    for idx in df[df["quantity"] > upper_bound].index:
        flag(idx, "Outlier Quantity", f"quantity = {df.at[idx,'quantity']} (upper bound ~{upper_bound:.0f})")

    if "order_total" not in df.columns:
        df["order_total"] = (df["quantity"] * df["unit_price"]).round(2)
    for idx in df[df["refund_amount"] > df["order_total"]].index:
        flag(idx, "Refund Exceeds Order Total",
             f"refund={df.at[idx,'refund_amount']:.2f} > order_total={df.at[idx,'order_total']:.2f}")

    return pd.DataFrame(issues)

def run_pipeline(raw_df: pd.DataFrame):
    """
    Takes a raw uploaded DataFrame and returns:
      - clean_df: cleaned, deduplicated, correctly-typed data
      - issues_df: every flagged row, with category and detail
      - stats: a small dict of summary numbers for the UI
    """
    df = raw_df.copy()
    df = clean_price_column(df)
    df = parse_messy_dates(df)
    df = standardize_countries(df)
    df, dupes_removed = deduplicate(df)
    issues_df = detect_issues(df)
    stats = {
        "rows_before": len(raw_df),
        "rows_after": len(df),
        "duplicates_removed": dupes_removed,
        "total_issues_flagged": len(issues_df),
    }
    return df, issues_df, stats
