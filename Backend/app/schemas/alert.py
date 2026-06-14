from typing import Optional, List, Any
from pydantic import BaseModel
from datetime import datetime


class ContributingFactor(BaseModel):
    factor: str
    value: str
    weight: float


class AlertGenerateRequest(BaseModel):
    account_id: str
    fraud_probability: float   # 0–1
    anomaly_score: float       # 0–100
    rules_score: float         # 0–100
    risk_score: float          # 0–100
    top_features: Optional[List[Any]] = None
    transaction_id: Optional[str] = None


class AlertBase(BaseModel):
    account_id: str
    severity: str
    status: str
    description: str


class AlertResponse(AlertBase):
    alert_id: str
    alert_type: str
    date: datetime
    reason_codes: Optional[List[str]] = []
    contributing_factors: Optional[List[ContributingFactor]] = []
    recommended_action: Optional[str] = None
    transaction_id: Optional[str] = None

    class Config:
        from_attributes = True
