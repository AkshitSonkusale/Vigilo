"""
Feature engineering for Vigilo's ML pipeline.

Takes the cleaned campaign DataFrame (output of csv_parser.parse_csv)
and produces a normalized feature matrix ready for K-Means and
Isolation Forest. Kept separate from the parser because in production
this will eventually take its input from the SQL analytics layer
instead of directly from the parser.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

# The features that actually define a campaign's "performance profile".
# Deliberately excluding raw impressions/clicks — they're scale-dependent
# and would let big campaigns dominate distance calculations regardless
# of whether they're actually performing well.
FEATURE_COLUMNS = ["ctr", "cpc", "conversion_rate", "roas", "cost"]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Returns:
        scaled_df: DataFrame of standardized features, same row order
                   and index as input df, indexed by campaign_name.
        scaler: the fitted StandardScaler, returned so cluster centroids
                can later be inverse-transformed for interpretability.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = df[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(
        scaled, columns=FEATURE_COLUMNS, index=df["campaign_name"]
    )
    return scaled_df, scaler


if __name__ == "__main__":
    import sys
    sys.path.append("../utils")
    from csv_parser import parse_csv

    with open("../data/sample_campaigns.csv", "rb") as f:
        result = parse_csv(f.read())

    scaled_df, scaler = build_feature_matrix(result.df)
    print(scaled_df.round(2))
