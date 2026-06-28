"""
SQL analytics layer for Vigilo.

This module sits between the CSV parser and the ML pipeline.
Raw campaign data is inserted into PostgreSQL first, then these
analytical queries run against it — aggregations, window functions,
and CTEs — and the output is what gets passed to K-Means and
Isolation Forest instead of raw CSV values.

Why SQL before ML:
  - Aggregations and rankings computed in SQL are more efficient
    than doing them in pandas for large datasets
  - Window functions give each campaign context about the whole
    account (percentile rank, comparison to average) that pure
    in-memory computation would need extra code to replicate
  - The SQL layer is independently testable and queryable —
    a data analyst can inspect the intermediate output without
    touching the ML code at all
  - It makes the resume story true: "SQL analytics layer using
    window functions and CTEs feeds the ML pipeline"

Each query below is production-quality and directly explainable
in a data analyst or ML intern interview.
"""

from __future__ import annotations

import os
import uuid

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_engine():
    """
    Returns a SQLAlchemy engine using the DATABASE_URL environment variable.
    Format: postgresql://user:password@host:port/dbname
    Set this in your .env file locally and as an env var on Render.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Add it to your .env file.\n"
            "Format: postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres"
        )
    return create_engine(url)


# ---------------------------------------------------------------------------
# Insert raw campaign data
# ---------------------------------------------------------------------------

def insert_campaigns(df: pd.DataFrame) -> str:
    """
    Inserts parsed campaign rows into the campaigns table.
    Returns the session_id UUID that groups this upload together.
    All subsequent queries filter by this session_id.
    """
    session_id = str(uuid.uuid4())
    engine = get_engine()

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "session_id": session_id,
            "campaign_name": str(row["campaign_name"]),
            "impressions": int(row["impressions"]),
            "clicks": int(row["clicks"]),
            "cost": float(row["cost"]),
            "conversions": int(row["conversions"]),
            "ctr": float(row["ctr"]),
            "cpc": float(row["cpc"]),
            "conversion_rate": float(row["conversion_rate"]),
            "roas": float(row["roas"]),
        })

    insert_sql = text("""
        INSERT INTO campaigns
          (session_id, campaign_name, impressions, clicks, cost,
           conversions, ctr, cpc, conversion_rate, roas)
        VALUES
          (:session_id, :campaign_name, :impressions, :clicks, :cost,
           :conversions, :ctr, :cpc, :conversion_rate, :roas)
    """)

    with engine.begin() as conn:
        conn.execute(insert_sql, rows)

    return session_id


# ---------------------------------------------------------------------------
# Query 1 — Account-level aggregations
# ---------------------------------------------------------------------------

ACCOUNT_AGGREGATIONS_SQL = """
SELECT
  COUNT(*)                                              AS total_campaigns,
  SUM(cost)                                             AS total_spend,
  SUM(conversions)                                      AS total_conversions,
  ROUND(AVG(ctr)::NUMERIC, 4)                           AS account_avg_ctr,
  ROUND(AVG(cpc)::NUMERIC, 2)                           AS account_avg_cpc,
  ROUND(AVG(conversion_rate)::NUMERIC, 4)               AS account_avg_cvr,
  ROUND(
    SUM(conversions)::NUMERIC / NULLIF(SUM(cost), 0), 4
  )                                                     AS account_roas
FROM campaigns
WHERE session_id = :session_id
"""


def get_account_aggregations(session_id: str) -> dict:
    """
    Account-level summary statistics computed in SQL.
    Used for the dashboard summary cards and as context
    for the Claude API recommendation prompts.
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text(ACCOUNT_AGGREGATIONS_SQL),
            {"session_id": session_id}
        ).fetchone()
    return dict(result._mapping)


# ---------------------------------------------------------------------------
# Query 2 — Campaign rankings with window functions
# ---------------------------------------------------------------------------

CAMPAIGN_RANKINGS_SQL = """
SELECT
  id                                                    AS campaign_id,
  campaign_name,
  cost,
  conversions,
  ctr,
  cpc,
  conversion_rate,
  roas,

  -- Cost per conversion (NULL-safe)
  ROUND(
    cost / NULLIF(conversions, 0), 2
  )                                                     AS cost_per_conversion,

  -- ROAS rank: 1 = best performing campaign by return
  RANK() OVER (
    ORDER BY conversions::FLOAT / NULLIF(cost, 0) DESC
  )                                                     AS roas_rank,

  -- CTR percentile: where does this campaign sit in the account?
  ROUND(
    PERCENT_RANK() OVER (ORDER BY ctr)::NUMERIC, 4
  )                                                     AS ctr_percentile,

  -- Conversion rate percentile
  ROUND(
    PERCENT_RANK() OVER (ORDER BY conversion_rate)::NUMERIC, 4
  )                                                     AS cvr_percentile,

  -- CPC comparison to account average (ratio > 1 = expensive)
  ROUND(
    cpc / NULLIF(AVG(cpc) OVER (), 0)::NUMERIC, 4
  )                                                     AS cpc_vs_avg_ratio,

  -- Account averages as window columns (available per row)
  ROUND(AVG(ctr) OVER ()::NUMERIC, 4)                  AS account_avg_ctr,
  ROUND(AVG(cpc) OVER ()::NUMERIC, 2)                  AS account_avg_cpc,
  ROUND(AVG(conversion_rate) OVER ()::NUMERIC, 4)      AS account_avg_cvr

FROM campaigns
WHERE session_id = :session_id
ORDER BY roas_rank
"""


def get_campaign_rankings(session_id: str) -> pd.DataFrame:
    """
    Per-campaign metrics enriched with window function results.
    This is the DataFrame that feeds into the ML pipeline instead
    of the raw parsed CSV — so every campaign already knows its
    ROAS rank, CTR percentile, and how its CPC compares to
    the account average before K-Means even runs.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text(CAMPAIGN_RANKINGS_SQL),
            conn,
            params={"session_id": session_id}
        )
    return df


# ---------------------------------------------------------------------------
# Query 3 — CTE-based pre-flagging
# ---------------------------------------------------------------------------

PREFLAG_SQL = """
WITH account_stats AS (
  -- Step 1: compute account-level thresholds once
  SELECT
    AVG(cpc)               AS avg_cpc,
    AVG(ctr)               AS avg_ctr,
    AVG(conversion_rate)   AS avg_cvr,
    AVG(cost)              AS avg_cost
  FROM campaigns
  WHERE session_id = :session_id
),

campaign_flags AS (
  -- Step 2: flag each campaign against those thresholds
  SELECT
    c.id                   AS campaign_id,
    c.campaign_name,
    c.cost,
    c.conversions,
    c.cpc,
    c.ctr,
    c.conversion_rate,

    -- Zero conversion flag: meaningful spend, no results
    CASE
      WHEN c.cost > 1000 AND c.conversions = 0
      THEN TRUE ELSE FALSE
    END                    AS zero_conversion_flag,

    -- High CPC flag: costs 1.5x more per click than account average
    CASE
      WHEN c.cpc > a.avg_cpc * 1.5
      THEN TRUE ELSE FALSE
    END                    AS high_cpc_flag,

    -- Low CTR flag: getting less than half the account's average CTR
    CASE
      WHEN c.ctr < a.avg_ctr * 0.5
      THEN TRUE ELSE FALSE
    END                    AS low_ctr_flag,

    -- High spender flag: spending more than 1.5x the account average
    CASE
      WHEN c.cost > a.avg_cost * 1.5
      THEN TRUE ELSE FALSE
    END                    AS high_spend_flag,

    -- Total flags count (useful for severity sorting)
    (
      CASE WHEN c.cost > 1000 AND c.conversions = 0 THEN 1 ELSE 0 END +
      CASE WHEN c.cpc > a.avg_cpc * 1.5 THEN 1 ELSE 0 END +
      CASE WHEN c.ctr < a.avg_ctr * 0.5 THEN 1 ELSE 0 END
    )                      AS total_flags

  FROM campaigns c, account_stats a
  WHERE c.session_id = :session_id
)

-- Step 3: return flagged campaigns, worst first
SELECT * FROM campaign_flags
ORDER BY total_flags DESC, cost DESC
"""


def get_campaign_flags(session_id: str) -> pd.DataFrame:
    """
    CTE-based pre-flagging query. Identifies campaigns that are
    clearly problematic before the ML models even run, based on
    simple threshold rules computed in SQL. These flags are passed
    to the recommendation engine as additional context.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text(PREFLAG_SQL),
            conn,
            params={"session_id": session_id}
        )
    return df


# ---------------------------------------------------------------------------
# Write ML results back to PostgreSQL
# ---------------------------------------------------------------------------

def insert_results(session_id: str, results_df: pd.DataFrame) -> None:
    """
    Writes ML pipeline output (cluster labels, anomaly flags,
    health scores, recommendations) back to PostgreSQL.
    This makes the results queryable independently of the app —
    useful for analytics, debugging, and showing interviewers
    the data sitting in real tables.
    """
    engine = get_engine()

    # Get campaign IDs for this session
    with engine.connect() as conn:
        id_map = pd.read_sql(
            text("SELECT id, campaign_name FROM campaigns WHERE session_id = :sid"),
            conn,
            params={"sid": session_id}
        )

    id_lookup = dict(zip(id_map["campaign_name"], id_map["id"]))

    result_rows = []
    rec_rows = []

    for _, row in results_df.iterrows():
        cid = id_lookup.get(row["campaign_name"])
        if not cid:
            continue

        result_rows.append({
            "campaign_id": str(cid),
            "cluster_label": str(row["cluster_label"]),
            "is_anomaly": bool(row["is_anomaly"]),
            "is_standout": bool(row["is_standout"]),
            "anomaly_score": float(row.get("anomaly_score", 0)),
            "health_score": int(row["health_score"]),
            "health_category": str(row["health_category"]),
            "severity": str(row["severity"]),
        })

        rec_rows.append({
            "campaign_id": str(cid),
            "recommendation_text": str(row["recommendation_text"]),
            "recommendation_source": str(row["recommendation_source"]),
        })

    with engine.begin() as conn:
        if result_rows:
            conn.execute(text("""
                INSERT INTO campaign_results
                  (campaign_id, cluster_label, is_anomaly, is_standout,
                   anomaly_score, health_score, health_category, severity)
                VALUES
                  (:campaign_id, :cluster_label, :is_anomaly, :is_standout,
                   :anomaly_score, :health_score, :health_category, :severity)
            """), result_rows)

        if rec_rows:
            conn.execute(text("""
                INSERT INTO recommendations
                  (campaign_id, recommendation_text, recommendation_source)
                VALUES
                  (:campaign_id, :recommendation_text, :recommendation_source)
            """), rec_rows)
