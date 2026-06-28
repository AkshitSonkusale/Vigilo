"""
Recommendation engine for Vigilo.

Takes the fully scored campaign DataFrame (cluster_label, is_anomaly,
is_standout, health_score, health_category already computed) and
generates natural-language, cluster-aware recommendations via the
Groq API (free tier — llama-3.3-70b-versatile).

Design decisions:
  - Severity is computed deterministically in Python, NOT by the LLM.
  - All campaigns sent in one batched request so the LLM can reason
    across campaigns (e.g. "reallocate budget FROM X TO Y").
  - If the API call fails or no key is configured, a deterministic
    rule-based fallback generates serviceable recommendation text so
    the product never breaks.

Groq API:
  - Free tier at console.groq.com — no credit card needed
  - Model: llama-3.3-70b-versatile (free, reliable JSON output)
  - OpenAI-compatible SDK interface
"""

from __future__ import annotations

import json
import os

import pandas as pd

try:
    from groq import Groq
except ImportError:
    Groq = None

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 3000

REQUIRED_COLUMNS = {
    "campaign_name", "cost", "conversions", "ctr", "cpc",
    "conversion_rate", "roas", "cluster_label", "is_anomaly",
    "is_standout", "health_score", "health_category",
}


def compute_severity(row: pd.Series) -> str:
    """
    Deterministic severity — never delegated to the LLM.
    """
    if row["is_anomaly"] or row["health_category"] == "Poor":
        return "High"
    if row["health_category"] == "Average":
        return "Medium"
    return "Low"


def _build_account_context(df: pd.DataFrame) -> dict:
    return {
        "total_campaigns": len(df),
        "total_spend": round(df["cost"].sum(), 2),
        "total_conversions": int(df["conversions"].sum()),
        "account_avg_ctr": round(df["ctr"].mean(), 2),
        "account_avg_cpc": round(df["cpc"].mean(), 2),
        "account_avg_conversion_rate": round(df["conversion_rate"].mean(), 2),
    }


def _build_campaign_payload(df: pd.DataFrame) -> list[dict]:
    payload = []
    for _, row in df.iterrows():
        payload.append({
            "campaign_name": row["campaign_name"],
            "spend": round(row["cost"], 2),
            "conversions": int(row["conversions"]),
            "ctr": row["ctr"],
            "cpc": row["cpc"],
            "conversion_rate": row["conversion_rate"],
            "roas": row["roas"],
            "cluster_label": row["cluster_label"],
            "is_anomaly": bool(row["is_anomaly"]),
            "is_standout": bool(row["is_standout"]),
            "health_score": int(row["health_score"]),
            "health_category": row["health_category"],
            "severity": row["severity"],
        })
    return payload


SYSTEM_PROMPT = """You are Vigilo's recommendation engine for Google Ads campaign optimization.

You will be given account-level context and a list of campaigns. Each campaign already has \
a cluster label, anomaly status, health score, and severity — these were computed by a \
separate deterministic ML pipeline and are FACTS, not suggestions. Do not change them.

Your only job is to write a short, specific, actionable recommendation for each campaign, \
grounded strictly in the numbers and labels provided. Reference exact figures (spend, \
conversions, CTR, etc.) rather than vague language. Where it makes sense, compare a \
struggling campaign to a well-performing one BY NAME to suggest budget reallocation.

Rules:
- 2-3 sentences per campaign, no more.
- Never invent metrics that weren't provided.
- Never change the severity, health score, or cluster label.
- Be direct and specific — "pause this campaign" not "consider reviewing performance".
- All monetary values are in Indian Rupees. Always use the ₹ symbol, never $ or dollars.
- Output ONLY valid JSON, no markdown code fences, no preamble.

Output format (JSON array, one object per campaign, same order as input):
[{"campaign_name": "...", "recommendation": "..."}]"""


def _call_groq(account_context: dict, campaigns: list[dict]) -> str:
    if Groq is None:
        raise RuntimeError("groq package not installed — run: pip install groq")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    user_message = (
        f"Account context:\n{json.dumps(account_context, indent=2)}\n\n"
        f"Campaigns:\n{json.dumps(campaigns, indent=2)}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


def _parse_response(raw_text: str) -> dict[str, str]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    parsed = json.loads(cleaned.strip())
    return {item["campaign_name"]: item["recommendation"] for item in parsed}


# --- Deterministic fallback -------------------------------------------------

FALLBACK_TEMPLATES = {
    "Bleeding Budget": (
        "{name} has spent ₹{spend:,.0f} but converted only {conversions} times "
        "(ROAS {roas:.4f}), well below the account average. Pause or significantly "
        "reduce budget here and redirect spend toward better-performing campaigns."
    ),
    "Keyword Waste": (
        "{name} is converting poorly (CTR {ctr:.1f}%, conversion rate {cvr:.1f}%) "
        "relative to the account average. Review keyword targeting and ad creative "
        "before continuing to invest here."
    ),
    "Steady Performer": (
        "{name} is performing in line with account averages (CTR {ctr:.1f}%, "
        "{conversions} conversions). No urgent action needed — monitor for changes."
    ),
    "Underinvested Winner": (
        "{name} is outperforming the account on CTR ({ctr:.1f}%) and conversion rate "
        "({cvr:.1f}%) while spending only ₹{spend:,.0f}. Increase budget here to "
        "capture more of this demand."
    ),
}

DEFAULT_FALLBACK = (
    "{name} has spent ₹{spend:,.0f} with {conversions} conversions. "
    "Review performance against account averages before making changes."
)


def _fallback_recommendation(row: pd.Series) -> str:
    template = FALLBACK_TEMPLATES.get(row["cluster_label"], DEFAULT_FALLBACK)
    text = template.format(
        name=row["campaign_name"],
        spend=row["cost"],
        conversions=row["conversions"],
        roas=row["roas"],
        ctr=row["ctr"],
        cvr=row["conversion_rate"],
    )
    if row["is_anomaly"]:
        text += " This campaign has been flagged as a statistical anomaly — its behavior is unusual even compared to other underperforming campaigns and warrants immediate review."
    elif row["is_standout"]:
        text += " This campaign also stands out as a statistical outlier in a positive way — its performance profile is unusually strong for this account."
    return text


# --- Main entry point --------------------------------------------------------

def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Falls back to deterministic templates if the
    Groq API call fails for any reason — the product never breaks
    because of an LLM outage or missing key.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()
    result["severity"] = result.apply(compute_severity, axis=1)

    account_context = _build_account_context(result)
    campaign_payload = _build_campaign_payload(result)

    try:
        raw_response = _call_groq(account_context, campaign_payload)
        recommendations = _parse_response(raw_response)
        result["recommendation_text"] = result["campaign_name"].map(recommendations)
        result["recommendation_source"] = "groq_api"

        missing_recs = result["recommendation_text"].isna()
        if missing_recs.any():
            result.loc[missing_recs, "recommendation_text"] = result.loc[missing_recs].apply(
                _fallback_recommendation, axis=1
            )
            result.loc[missing_recs, "recommendation_source"] = "fallback_partial"

    except Exception as e:
        print(f"Groq API call failed, using fallback recommendations: {e}")
        result["recommendation_text"] = result.apply(_fallback_recommendation, axis=1)
        result["recommendation_source"] = "fallback"

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from utils.csv_parser import parse_csv
    from ml.clustering import run_clustering
    from ml.anomaly import detect_anomalies
    from ml.health_score import compute_health_scores

    with open("../data/sample_campaigns.csv", "rb") as f:
        parsed = parse_csv(f.read())

    clustered = run_clustering(parsed.df)
    flagged = detect_anomalies(parsed.df)
    merged = clustered.merge(
        flagged[["campaign_name", "statistical_outlier",
                 "is_anomaly", "is_standout", "anomaly_score"]],
        on="campaign_name",
    )
    scored = compute_health_scores(merged)
    recommended = generate_recommendations(scored)

    print(f"Source: {recommended['recommendation_source'].iloc[0]}\n")
    for _, row in recommended.sort_values("health_score").iterrows():
        print(f"[{row['severity']}] {row['campaign_name']} (score: {row['health_score']})")
        print(f"  {row['recommendation_text']}\n")