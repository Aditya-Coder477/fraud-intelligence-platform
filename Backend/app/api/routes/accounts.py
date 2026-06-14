from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Any, Dict

from app.api.dependencies import get_db
from app.db.models import Account, Transaction, PredictionLog
from app.schemas.account import AccountResponse, AccountDetailResponse, TransactionResponse
from app.schemas.ml import RiskScoreResponse, ExplanationResponse, FeatureContribution, FeatureContributionDetail
from app.services.explanation_service import explanation_service

router = APIRouter()


@router.get("/accounts")
def get_accounts(
    search: str = None,
    risk_min: float = Query(0.0),
    risk_max: float = Query(100.0),
    status: str = None,
    sort_by: str = "risk_score",
    sort_order: str = "desc",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:

    query = db.query(Account)

    if search:
        query = query.filter(Account.account_id.ilike(f"%{search}%"))
    if status:
        query = query.filter(Account.status == status)

    query = query.filter(Account.risk_score >= risk_min, Account.risk_score <= risk_max)

    if sort_order == "desc":
        query = query.order_by(getattr(Account, sort_by).desc())
    else:
        query = query.order_by(getattr(Account, sort_by).asc())

    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "data": [AccountResponse.model_validate(item) for item in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/accounts/{account_id}", response_model=AccountDetailResponse)
def get_account(account_id: str, db: Session = Depends(get_db)):
    account = db.query(Account).filter(Account.account_id == account_id).first()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    pred = (
        db.query(PredictionLog)
        .filter(PredictionLog.account_id == account_id)
        .order_by(PredictionLog.timestamp.desc())
        .first()
    )
    summary = pred.explanation_summary if pred else "No explanation available."

    acc_dict = AccountDetailResponse.model_validate(account).model_dump()
    acc_dict["explanation_summary"] = summary
    return acc_dict


@router.get("/accounts/{account_id}/transactions")
def get_transactions(
    account_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    if account_id == "ALL":
        query = db.query(Transaction).order_by(Transaction.timestamp.desc())
    else:
        query = (
            db.query(Transaction)
            .filter(Transaction.account_id == account_id)
            .order_by(Transaction.timestamp.desc())
        )

    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "data": [TransactionResponse.model_validate(item) for item in items],
        "total": total,
    }


@router.get("/accounts/{account_id}/explanations", response_model=ExplanationResponse)
def get_account_explanations(account_id: str, db: Session = Depends(get_db)):
    """
    Returns real SHAP-based explanation enriched with direction, plain-English text,
    and a regulatory-grade narrative summary.
    """
    account = db.query(Account).filter(Account.account_id == account_id).first()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    # Use real risk values from DB to personalise the narrative
    exp_data = explanation_service.get_explanation(
        account_id=account_id,
        fraud_probability=account.fraud_probability,
        risk_score=account.risk_score,
        top_n=8,
    )

    features = [FeatureContributionDetail(**f) for f in exp_data["top_features"]]

    return ExplanationResponse(
        top_features=features,
        summary=exp_data["summary"],
        overall_summary=exp_data["overall_summary"],
        reason_codes=exp_data["reason_codes"],
        confidence=exp_data["confidence"],
    )


@router.get("/risk/{account_id}", response_model=RiskScoreResponse)
def get_account_risk(account_id: str, db: Session = Depends(get_db)):
    account = db.query(Account).filter(Account.account_id == account_id).first()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    return RiskScoreResponse(
        risk_score=account.risk_score,
        fraud_probability=account.fraud_probability,
        anomaly_score=account.anomaly_score,
        rules_score=account.rules_score,
        score_band=account.category,
        risk_tags=["pass-through", "velocity", "new-account"] if account.risk_score > 60 else ["safe"],
        decision_recommendation=(
            "Immediate block recommended"
            if account.risk_score >= 81
            else "Review required"
            if account.risk_score >= 61
            else "No action"
        ),
    )
