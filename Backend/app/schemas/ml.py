from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class FeatureContribution(BaseModel):
    feature: str
    importance: float
    description: str


class FeatureContributionDetail(BaseModel):
    feature: str
    importance: float          # absolute SHAP value
    direction: str             # "positive" (increases risk) | "negative" (decreases risk)
    description: str           # plain-English label
    explanation_text: str      # analyst-readable sentence
    pct_of_total: Optional[float] = None


class ExplanationResponse(BaseModel):
    top_features: List[FeatureContributionDetail]
    summary: str               # short 1-liner (legacy compat)
    overall_summary: str       # long regulatory narrative
    reason_codes: List[str]
    confidence: Optional[float] = None


class RiskScoreResponse(BaseModel):
    risk_score: float
    fraud_probability: float
    anomaly_score: float
    rules_score: float
    score_band: str
    risk_tags: List[str]
    decision_recommendation: str


class PredictionRequest(BaseModel):
    account_id: Optional[str] = None
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    prediction_label: int
    fraud_probability: float
    risk_score: float
    alert_recommendation: str
    explanation_summary: str


class DashboardBin(BaseModel):
    bin: str
    count: int


class TrendPoint(BaseModel):
    date: str
    flag_count: int


class DashboardSummaryResponse(BaseModel):
    total_transactions: int
    suspicious_accounts: int
    high_risk_alerts: int
    fraud_probability_distribution: List[DashboardBin]
    trend_data: List[TrendPoint]
    recent_flagged_cases: List[Dict[str, Any]]
