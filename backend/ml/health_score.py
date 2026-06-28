"""
Health score for Vigilo.

Deliberately NOT a model — this is a transparent, deterministic
formula that combines signals from K-Means (cluster_label) and
Isolation Forest (is_anomaly/is_standout) with raw metric performance.
Stakeholders and interviewers should be able to read this function
top to bottom and understand exactly why a campaign got the score it did.
The Claude API later explains the score in natural language — it
never invents or overrides it.
"""

from __future__ import annotations

import pandas as pd

# Base score awarded purely from cluster membership, before any
# adjustment for individual metric performance or anomaly status.
# These reflect the *typical* quality of each cluster profile.
CLUSTER_BASE_SCORES = {
    "Underinvested Winner": 75,
    "Steady Performer": 60,
    "Keyword Waste": 35,
    "Bleeding Budget": 25,
}
DEFAULT_BASE_SCORE = 50  # fallback for any unrecognized cluster label

ANOMALY_PENALTY = 15
STANDOUT_BONUS = 10


def _metric_adjustment(row: pd.Series, account_avg: dict[str, float]) -> int:
    """
    Small adjustment (-10 to +10) based on how this campaign's raw
    metrics compare to the account average — lets two campaigns in
    the same cluster still be differentiated rather than scoring
    identically just because K-Means grouped them together.
    """
    adjustment = 0

    if account_avg["ctr"] > 0:
        ctr_ratio = row["ctr"] / account_avg["ctr"]
        if ctr_ratio >= 1.3:
            adjustment += 4
        elif ctr_ratio <= 0.7:
            adjustment -= 4

    if account_avg["conversion_rate"] > 0:
        cvr_ratio = row["conversion_rate"] / account_avg["conversion_rate"]
        if cvr_ratio >= 1.3:
            adjustment += 4
        elif cvr_ratio <= 0.7:
            adjustment -= 4

    if account_avg["cpc"] > 0:
        cpc_ratio = row["cpc"] / account_avg["cpc"]
        if cpc_ratio <= 0.7:
            adjustment += 2
        elif cpc_ratio >= 1.5:
            adjustment -= 2

    return adjustment


def _categorize(score: int) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Poor"


def compute_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df to already have cluster_label, is_anomaly, and
    is_standout columns (i.e. run AFTER clustering.run_clustering
    and anomaly.detect_anomalies, joined on campaign_name).

    Returns df with two new columns: health_score (0-100, clipped)
    and health_category (Excellent / Good / Average / Poor).
    """
    required = {"cluster_label", "is_anomaly", "is_standout", "ctr", "cpc", "conversion_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for health scoring: {missing}")

    account_avg = {
        "ctr": df["ctr"].mean(),
        "cpc": df["cpc"].mean(),
        "conversion_rate": df["conversion_rate"].mean(),
    }

    scores = []
    for _, row in df.iterrows():
        base = CLUSTER_BASE_SCORES.get(row["cluster_label"], DEFAULT_BASE_SCORE)
        adjustment = _metric_adjustment(row, account_avg)
        score = base + adjustment

        if row["is_anomaly"]:
            score -= ANOMALY_PENALTY
        if row["is_standout"]:
            score += STANDOUT_BONUS

        score = int(max(0, min(100, score)))
        scores.append(score)

    result = df.copy()
    result["health_score"] = scores
    result["health_category"] = result["health_score"].apply(_categorize)
    return result


if __name__ == "__main__":
    import sys
    sys.path.append("../utils")
    from csv_parser import parse_csv
    from clustering import run_clustering
    from anomaly import detect_anomalies

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    with open("../data/sample_campaigns.csv", "rb") as f:
        parsed = parse_csv(f.read())

    clustered = run_clustering(parsed.df, k=4)
    flagged = detect_anomalies(parsed.df)

    merged = clustered.merge(
        flagged[["campaign_name", "statistical_outlier", "is_anomaly", "is_standout", "anomaly_score"]],
        on="campaign_name",
    )

    scored = compute_health_scores(merged)
    cols = ["campaign_name", "cluster_label", "is_anomaly", "is_standout",
            "health_score", "health_category"]
    print(scored[cols].sort_values("health_score", ascending=False))
