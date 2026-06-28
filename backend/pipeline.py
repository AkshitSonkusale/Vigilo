"""
Vigilo pipeline orchestrator — updated to wire in the SQL analytics layer.

Flow WITH database (production):
  CSV → Parser → PostgreSQL → SQL queries → ML Pipeline → PostgreSQL

Flow WITHOUT database (demo/local fallback):
  CSV → Parser → ML Pipeline

The SQL layer is activated automatically when DATABASE_URL is set
in the environment. If it's not set, the pipeline falls back to
running ML directly on the parsed DataFrame — so the demo always
works even without a database connection.
"""

from __future__ import annotations

import os

import pandas as pd

from utils.csv_parser import parse_csv, ParseError
from ml.clustering import run_clustering
from ml.anomaly import detect_anomalies
from ml.health_score import compute_health_scores
from ml.recommendations import generate_recommendations
from models import AccountSummary, CampaignResult, VigiloResponse


def _has_database() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _build_account_summary(df: pd.DataFrame, campaigns: list) -> AccountSummary:
    return AccountSummary(
        total_campaigns=len(df),
        total_spend=round(float(df["cost"].sum()), 2),
        total_conversions=int(df["conversions"].sum()),
        account_avg_ctr=round(float(df["ctr"].mean()), 2),
        account_avg_cpc=round(float(df["cpc"].mean()), 2),
        account_avg_conversion_rate=round(float(df["conversion_rate"].mean()), 2),
        account_total_roas=round(
            float(df["conversions"].sum() / df["cost"].sum())
            if df["cost"].sum() > 0 else 0, 4
        ),
    )


def _build_campaigns(final: pd.DataFrame) -> list[CampaignResult]:
    campaigns = []
    for _, row in final.iterrows():
        campaigns.append(CampaignResult(
            campaign_name=str(row["campaign_name"]),
            impressions=int(row["impressions"]),
            clicks=int(row["clicks"]),
            cost=round(float(row["cost"]), 2),
            conversions=int(row["conversions"]),
            ctr=round(float(row["ctr"]), 2),
            cpc=round(float(row["cpc"]), 2),
            conversion_rate=round(float(row["conversion_rate"]), 2),
            roas=round(float(row["roas"]), 4),
            cluster_label=str(row["cluster_label"]),
            is_anomaly=bool(row["is_anomaly"]),
            is_standout=bool(row["is_standout"]),
            health_score=int(row["health_score"]),
            health_category=str(row["health_category"]),
            severity=str(row["severity"]),
            recommendation_text=str(row["recommendation_text"]),
            recommendation_source=str(row["recommendation_source"]),
        ))

    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    campaigns.sort(key=lambda c: (severity_order.get(c.severity, 3), c.health_score))
    return campaigns


def _run_with_sql(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Pipeline with SQL analytics layer active.
    Inserts raw data into PostgreSQL, runs analytical queries,
    feeds SQL output to ML models, writes results back.
    """
    from db.queries import (
        insert_campaigns, get_campaign_rankings,
        get_campaign_flags, insert_results
    )

    warnings = []

    # Insert raw data
    session_id = insert_campaigns(df)
    warnings.append(f"Data stored in PostgreSQL (session: {session_id[:8]}...)")

    # SQL Query 2: campaign rankings with window functions
    ranked_df = get_campaign_rankings(session_id)

    # SQL Query 3: CTE-based pre-flags
    flags_df = get_campaign_flags(session_id)

    # Merge SQL-enriched data back onto original df for ML
    # ML needs the base columns; SQL adds ranking/flag context
    base_cols = ["campaign_name", "impressions", "clicks", "cost",
                 "conversions", "ctr", "cpc", "conversion_rate", "roas"]
    ml_input = df[base_cols].copy()

    # Attach SQL-derived flags as extra context for recommendations
    flag_cols = ["campaign_name", "zero_conversion_flag",
                 "high_cpc_flag", "low_ctr_flag", "total_flags"]
    available_flags = [c for c in flag_cols if c in flags_df.columns]
    if available_flags:
        ml_input = ml_input.merge(
            flags_df[available_flags], on="campaign_name", how="left"
        )

    # Run ML pipeline on SQL-enriched data
    clustered = run_clustering(ml_input, k=4)
    flagged = detect_anomalies(ml_input)
    merged = clustered.merge(
        flagged[["campaign_name", "statistical_outlier",
                 "is_anomaly", "is_standout", "anomaly_score"]],
        on="campaign_name",
    )
    scored = compute_health_scores(merged)
    final = generate_recommendations(scored)

    # Write results back to PostgreSQL
    try:
        insert_results(session_id, final)
        warnings.append("ML results written back to PostgreSQL.")
    except Exception as e:
        warnings.append(f"Could not write results to DB: {e}")

    return final, warnings


def _run_without_sql(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Fallback pipeline with no database.
    Runs entirely in-memory — same ML logic, same output shape.
    """
    warnings = ["Running in demo mode (no DATABASE_URL set)."]

    clustered = run_clustering(df, k=4)
    flagged = detect_anomalies(df)
    merged = clustered.merge(
        flagged[["campaign_name", "statistical_outlier",
                 "is_anomaly", "is_standout", "anomaly_score"]],
        on="campaign_name",
    )
    scored = compute_health_scores(merged)
    final = generate_recommendations(scored)

    return final, warnings


def run_pipeline(file_bytes: bytes) -> VigiloResponse:
    """
    Main entry point. Automatically routes to SQL or fallback
    pipeline based on whether DATABASE_URL is configured.
    """
    parsed = parse_csv(file_bytes)
    df = parsed.df
    base_warnings = list(parsed.warnings)

    if _has_database():
        try:
            final, sql_warnings = _run_with_sql(df)
            warnings = base_warnings + sql_warnings
        except Exception as e:
            # DB failed mid-pipeline — fall back gracefully
            print(f"SQL pipeline failed ({e}), falling back to in-memory.")
            final, fallback_warnings = _run_without_sql(df)
            warnings = base_warnings + fallback_warnings + [f"DB error: {e}"]
    else:
        final, fallback_warnings = _run_without_sql(df)
        warnings = base_warnings + fallback_warnings

    account_summary = _build_account_summary(df, [])
    campaigns = _build_campaigns(final)

    return VigiloResponse(
        account_summary=account_summary,
        campaigns=campaigns,
        warnings=warnings,
    )
