from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from app.api.dependencies import get_db
from app.db.models import Account, Alert
from app.schemas.ml import DashboardSummaryResponse, DashboardBin, TrendPoint

router = APIRouter()

@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(db: Session = Depends(get_db)):
    # Total accounts/transactions mock (in reality, distinct from accounts)
    total_tx = db.query(Account).count() * 15 # mock multiplier
    
    suspicious_count = db.query(Account).filter(Account.risk_score >= 60).count()
    high_risk_alerts = db.query(Alert).filter(Alert.severity.in_(["CRITICAL", "HIGH"]), Alert.status == "OPEN").count()
    
    # Bins
    bins = [
        DashboardBin(bin="0-10%", count=db.query(Account).filter(Account.fraud_probability < 0.1).count()),
        DashboardBin(bin="10-30%", count=db.query(Account).filter(Account.fraud_probability >= 0.1, Account.fraud_probability < 0.3).count()),
        DashboardBin(bin="30-60%", count=db.query(Account).filter(Account.fraud_probability >= 0.3, Account.fraud_probability < 0.6).count()),
        DashboardBin(bin="60-80%", count=db.query(Account).filter(Account.fraud_probability >= 0.6, Account.fraud_probability < 0.8).count()),
        DashboardBin(bin="80-100%", count=db.query(Account).filter(Account.fraud_probability >= 0.8).count())
    ]
    
    # Recent priority
    recent_cases = db.query(Account).filter(Account.risk_score >= 60).order_by(Account.risk_score.desc()).limit(5).all()
    
    recent_cases_list = []
    for c in recent_cases:
        recent_cases_list.append({
            "account_id": c.account_id,
            "risk_score": c.risk_score,
            "category": c.category,
            "status": c.status,
            "alert_count": c.alert_count,
            "fraud_probability": c.fraud_probability,
            "last_activity": c.last_activity.strftime("%Y-%m-%d %H:%M") if c.last_activity else None
        })
        
    return DashboardSummaryResponse(
        total_transactions=total_tx,
        suspicious_accounts=suspicious_count,
        high_risk_alerts=high_risk_alerts,
        fraud_probability_distribution=bins,
        trend_data=[
            TrendPoint(date="06/01", flag_count=12),
            TrendPoint(date="06/02", flag_count=15),
            TrendPoint(date="06/03", flag_count=9),
            TrendPoint(date="06/04", flag_count=22),
            TrendPoint(date="06/05", flag_count=31),
            TrendPoint(date="06/06", flag_count=28)
        ],
        recent_flagged_cases=recent_cases_list
    )
