"""
Generates a realistic synthetic Google Ads campaign dataset for Vigilo.

Deliberately constructed to contain all four target cluster profiles
plus a couple of clear anomalies, so the ML pipeline has something
meaningful to find. Numbers are randomized within realistic ranges
for an Indian small-business Google Ads account.
"""

import csv
import random

random.seed(42)

# Each tuple: (name, profile_type)
# profile_type drives the metric ranges we sample from below.
CAMPAIGNS = [
    ("Brand Keywords - Search", "underinvested_winner"),
    ("Summer Sale - Search", "bleeding_budget"),
    ("Retargeting - Display", "steady_performer"),
    ("Generic Apparel - Search", "keyword_waste"),
    ("New Arrivals - Search", "steady_performer"),
    ("Festive Offer - Shopping", "underinvested_winner"),
    ("Competitor Terms - Search", "keyword_waste"),
    ("Lookalike Audience - Display", "steady_performer"),
    ("Clearance Sale - Search", "bleeding_budget"),
    ("Local Store - Search", "anomaly_zero_conv"),
    ("Influencer Push - Display", "anomaly_ctr_drop"),
]

# Ranges are (min, max) per metric, per profile type.
# Designed so K-Means clusters separate cleanly and Isolation Forest
# has obvious outliers to catch.
PROFILE_RANGES = {
    "underinvested_winner": {
        "spend": (800, 2500),
        "ctr": (4.5, 7.5),
        "cvr": (6.0, 10.0),
        "cpc": (4, 9),
    },
    "bleeding_budget": {
        "spend": (6000, 9500),
        "ctr": (1.0, 2.0),
        "cvr": (0.3, 1.0),
        "cpc": (18, 30),
    },
    "steady_performer": {
        "spend": (2500, 4500),
        "ctr": (2.5, 4.0),
        "cvr": (2.5, 4.5),
        "cpc": (8, 14),
    },
    "keyword_waste": {
        "spend": (3000, 5500),
        "ctr": (0.8, 1.5),
        "cvr": (0.1, 0.6),
        "cpc": (22, 38),
    },
    # Hand-crafted extreme outliers for Isolation Forest to flag
    "anomaly_zero_conv": {
        "spend": (8400, 8400),
        "ctr": (1.3, 1.3),
        "cvr": (0.0, 0.0),
        "cpc": (28, 28),
    },
    "anomaly_ctr_drop": {
        "spend": (3200, 3200),
        "ctr": (0.4, 0.4),  # sudden CTR crash vs account norm
        "cvr": (1.8, 1.8),
        "cpc": (15, 15),
    },
}


def generate_row(name: str, profile: str) -> dict:
    r = PROFILE_RANGES[profile]
    spend = round(random.uniform(*r["spend"]), 2)
    ctr = round(random.uniform(*r["ctr"]), 2)
    cvr = round(random.uniform(*r["cvr"]), 2)
    cpc = round(random.uniform(*r["cpc"]), 2)

    clicks = max(1, round(spend / cpc))
    impressions = max(clicks, round(clicks / (ctr / 100)))
    conversions = round(clicks * (cvr / 100))

    return {
        "Campaign Name": name,
        "Impressions": impressions,
        "Clicks": clicks,
        "Cost": spend,
        "Conversions": conversions,
        "CTR": ctr,
        "CPC": cpc,
        "Conversion Rate": cvr,
    }


def main():
    rows = [generate_row(name, profile) for name, profile in CAMPAIGNS]

    out_path = "sample_campaigns.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} campaigns to {out_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
