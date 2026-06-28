"""
Isolation Forest anomaly detection for Vigilo.

Runs independently of K-Means clustering — a campaign can be a
totally normal "Bleeding Budget" cluster member, or it can be a
genuine outlier even within its own cluster (e.g. a campaign that
suddenly spent 10x its usual budget overnight). We want to catch
both, so anomaly detection uses the same feature matrix but isn't
conditioned on cluster assignment.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.feature_engineering import build_feature_matrix


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.35) -> pd.DataFrame:
    """
    Takes the cleaned campaign DataFrame, returns it with columns:
      - statistical_outlier (bool): raw Isolation Forest flag, direction-agnostic
      - anomaly_score (float): lower = more isolated/rare
      - is_anomaly (bool): the actionable flag surfaced to users —
        statistical_outlier AND wasteful (see _is_wasteful below)
      - is_standout (bool): statistical_outlier but performing well —
        worth highlighting as a positive callout, not an alert

    Why the split: Isolation Forest finds statistically rare points
    in feature space with no concept of "good" or "bad" — a campaign
    that wildly outperforms everything else is just as "anomalous" to
    the model as one that's wasting budget. Surfacing pure statistical
    outliers as alerts would flag your best campaigns as problems.
    We fix this by combining the model's anomaly score with a
    directional business rule: only isolated points that are ALSO
    underperforming (low ROAS, low conversion rate relative to cost)
    get surfaced as actionable "wasteful" anomalies.

    contamination: expected proportion of statistical outliers in the
    data (before the directional filter is applied). This needs to be
    set wider than you'd normally use, since the directional filter
    splits whatever Isolation Forest finds into "standout" (good) and
    "anomaly" (wasteful) groups — too narrow a contamination budget
    and it gets entirely consumed by one direction, leaving nothing
    for the other. 0.35 is tuned for small accounts (5-15 campaigns)
    so both directions have room; tune down for larger accounts.
    """
    scaled_df, _ = build_feature_matrix(df)

    n_samples = len(scaled_df)
    if n_samples < 5:
        # Isolation Forest needs a reasonable sample size to be meaningful;
        # below this, fall back to flagging nothing rather than guessing.
        result = df.copy()
        result["statistical_outlier"] = False
        result["anomaly_score"] = 0.0
        result["is_anomaly"] = False
        result["is_standout"] = False
        return result

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    predictions = iso.fit_predict(scaled_df)  # -1 = outlier, 1 = normal
    scores = iso.decision_function(scaled_df)  # lower = more isolated

    result = df.copy()
    result["statistical_outlier"] = predictions == -1
    result["anomaly_score"] = scores.round(4)
    result["is_anomaly"] = result["statistical_outlier"] & result.apply(_is_wasteful, axis=1)
    result["is_standout"] = result["statistical_outlier"] & ~result["is_anomaly"]
    return result


def _is_wasteful(row: pd.Series) -> bool:
    """
    Directional business rule: a campaign counts as "wasteful" if it's
    spending meaningfully while converting poorly. Thresholds are
    intentionally simple/explainable rather than another learned model —
    this is the layer a stakeholder can read and trust.
    """
    spent_meaningfully = row["cost"] > 1000
    converting_poorly = row["roas"] < 0.005 or row["conversion_rate"] < 1.0
    return bool(spent_meaningfully and converting_poorly)


if __name__ == "__main__":
    import sys
    sys.path.append("../utils")
    from csv_parser import parse_csv

    with open("../data/sample_campaigns.csv", "rb") as f:
        parsed = parse_csv(f.read())

    flagged = detect_anomalies(parsed.df)
    cols = ["campaign_name", "cost", "conversions", "roas", "conversion_rate",
            "statistical_outlier", "is_anomaly", "is_standout", "anomaly_score"]
    print(flagged[cols].sort_values("anomaly_score"))
