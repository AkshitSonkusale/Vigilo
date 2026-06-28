"""
Pydantic models defining the shape of API responses.

Keeping these in a separate file means the frontend team (or you,
when you build the React side) has a single source of truth for
what the API returns — no guessing at field names.
"""

from __future__ import annotations
from pydantic import BaseModel


class AccountSummary(BaseModel):
    total_campaigns: int
    total_spend: float
    total_conversions: int
    account_avg_ctr: float
    account_avg_cpc: float
    account_avg_conversion_rate: float
    account_total_roas: float


class CampaignResult(BaseModel):
    campaign_name: str
    impressions: int
    clicks: int
    cost: float
    conversions: int
    ctr: float
    cpc: float
    conversion_rate: float
    roas: float
    cluster_label: str
    is_anomaly: bool
    is_standout: bool
    health_score: int
    health_category: str
    severity: str
    recommendation_text: str
    recommendation_source: str


class VigiloResponse(BaseModel):
    account_summary: AccountSummary
    campaigns: list[CampaignResult]
    warnings: list[str]
