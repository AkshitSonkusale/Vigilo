"""
CSV parser and validator for Vigilo.

Google Ads exports vary depending on locale, selected columns, and
export settings. This module normalizes whatever comes in into a
clean pandas DataFrame with a fixed, predictable schema, or raises
a ParseError with a human-readable message the frontend can display.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd

# Canonical column names we standardize everything to internally.
REQUIRED_COLUMNS = {
    "campaign_name",
    "impressions",
    "clicks",
    "cost",
    "conversions",
}

# Optional but expected — we can derive them if missing.
DERIVABLE_COLUMNS = {"ctr", "cpc", "conversion_rate"}

# Maps many possible header variants (Google Ads changes these across
# locales/export versions) to our canonical snake_case names.
HEADER_ALIASES = {
    "campaign": "campaign_name",
    "campaign name": "campaign_name",
    "campaigns": "campaign_name",
    "impr.": "impressions",
    "impressions": "impressions",
    "clicks": "clicks",
    "cost": "cost",
    "spend": "cost",
    "amount spent": "cost",
    "conversions": "conversions",
    "conv.": "conversions",
    "ctr": "ctr",
    "ctr (%)": "ctr",
    "click-through rate": "ctr",
    "avg. cpc": "cpc",
    "cpc": "cpc",
    "cost per click": "cpc",
    "conv. rate": "conversion_rate",
    "conversion rate": "conversion_rate",
    "conv. rate (%)": "conversion_rate",
}


class ParseError(Exception):
    """Raised when the uploaded CSV cannot be safely parsed."""


@dataclass
class ParseResult:
    df: pd.DataFrame
    warnings: list[str]


def _clean_header(col: str) -> str:
    col = col.strip().lower()
    col = col.replace("\ufeff", "")  # strip BOM if it leaked into a header
    return col


def _to_numeric(series: pd.Series, column_name: str) -> pd.Series:
    """
    Handles messy numeric formatting seen in real exports:
    - currency symbols (₹, $)
    - thousands separators (1,234.56)
    - percentage signs (4.5%)
    - locales using comma as decimal separator (4,5 -> 4.5)
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[₹$,%]", "", regex=True)
        .str.strip()
    )

    # Heuristic: if a value looks like "4,5" with no other comma and no
    # dot, treat comma as a decimal separator (European-style locale).
    def fix_decimal_comma(val: str) -> str:
        if re.match(r"^\d+,\d{1,2}$", val):
            return val.replace(",", ".")
        return val

    cleaned = cleaned.apply(fix_decimal_comma)

    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.isna().any():
        bad_rows = series[numeric.isna()].tolist()
        raise ParseError(
            f"Column '{column_name}' contains values that couldn't be "
            f"read as numbers: {bad_rows[:3]}"
        )
    return numeric


def parse_csv(file_bytes: bytes) -> ParseResult:
    """
    Main entry point. Takes raw uploaded file bytes, returns a clean
    DataFrame with canonical columns: campaign_name, impressions,
    clicks, cost, conversions, ctr, cpc, conversion_rate, roas.
    """
    warnings: list[str] = []

    try:
        text = file_bytes.decode("utf-8-sig")  # handles BOM automatically
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
            warnings.append("File was not UTF-8 encoded; decoded as Latin-1.")
        except Exception as e:
            raise ParseError(f"Could not decode file: {e}")

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise ParseError(f"Could not read file as CSV: {e}")

    if df.empty:
        raise ParseError("The uploaded file has no data rows.")

    # Normalize headers
    df.columns = [_clean_header(c) for c in df.columns]
    rename_map = {c: HEADER_ALIASES[c] for c in df.columns if c in HEADER_ALIASES}
    df = df.rename(columns=rename_map)

    # Drop fully-empty rows (common at the end of Google Ads exports,
    # e.g. a "Total" summary row)
    if "campaign_name" in df.columns:
        total_row_mask = df["campaign_name"].astype(str).str.lower().isin(
            ["total", "total:", "totals", ""]
        )
        if total_row_mask.any():
            warnings.append(
                f"Dropped {total_row_mask.sum()} summary/total row(s) from export."
            )
        df = df[~total_row_mask]
    df = df.dropna(how="all")

    missing_required = REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        raise ParseError(
            "Missing required column(s): "
            f"{', '.join(sorted(missing_required))}. "
            "Make sure your Google Ads export includes Campaign, "
            "Impressions, Clicks, Cost, and Conversions."
        )

    # Clean numeric columns that are present
    numeric_cols = ["impressions", "clicks", "cost", "conversions", "ctr", "cpc", "conversion_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_numeric(df[col], col)

    # Derive any missing optional metrics rather than failing
    if "ctr" not in df.columns:
        df["ctr"] = round((df["clicks"] / df["impressions"].replace(0, pd.NA)) * 100, 2)
        warnings.append("CTR was not in the export; derived from clicks/impressions.")

    if "cpc" not in df.columns:
        df["cpc"] = round(df["cost"] / df["clicks"].replace(0, pd.NA), 2)
        warnings.append("CPC was not in the export; derived from cost/clicks.")

    if "conversion_rate" not in df.columns:
        df["conversion_rate"] = round(
            (df["conversions"] / df["clicks"].replace(0, pd.NA)) * 100, 2
        )
        warnings.append("Conversion rate was not in the export; derived from conversions/clicks.")

    # Fill any NaNs created by divide-by-zero (e.g. 0 clicks) with 0
    df[["ctr", "cpc", "conversion_rate"]] = df[["ctr", "cpc", "conversion_rate"]].fillna(0)

    # Derived metric used throughout the ML pipeline
    df["roas"] = round(df["conversions"] / df["cost"].replace(0, pd.NA), 4).fillna(0)

    # Final column order, drop anything extra Google Ads included
    final_cols = [
        "campaign_name", "impressions", "clicks", "cost",
        "conversions", "ctr", "cpc", "conversion_rate", "roas",
    ]
    df = df[final_cols].reset_index(drop=True)

    return ParseResult(df=df, warnings=warnings)


if __name__ == "__main__":
    # Quick manual test against the bundled sample data
    with open("sample_campaigns.csv", "rb") as f:
        result = parse_csv(f.read())
    print(result.df)
    print("\nWarnings:", result.warnings)
