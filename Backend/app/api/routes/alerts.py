import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Any, Dict, Optional

from app.api.dependencies import get_db
from app.db.models import Alert
from app.schemas.alert import AlertResponse, AlertGenerateRequest
from app.services.alert_service import alert_generator

log = logging.getLogger("AlertsRoute")
router = APIRouter()


@router.get("/alerts")
def get_alerts(
    severity: Optional[str] = None,
    status: Optional[str] = None,
    alert_type: Optional[str] = None,
    account_id: Optional[str] = None,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """List alerts with rich filtering support."""
    query = db.query(Alert).order_by(Alert.timestamp.desc())

    if severity:
        query = query.filter(Alert.severity == severity.upper())
    if status:
        query = query.filter(Alert.status == status.upper())
    if alert_type:
        query = query.filter(Alert.alert_type == alert_type.upper())
    if account_id:
        query = query.filter(Alert.account_id == account_id)
    if search:
        query = query.filter(Alert.account_id.ilike(f"%{search}%"))

    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    results = []
    for a in items:
        results.append({
            "alert_id": a.alert_id,
            "account_id": a.account_id,
            "transaction_id": a.transaction_id,
            "alert_type": a.alert_type or "MODEL_SCORE",
            "severity": a.severity,
            "status": a.status,
            "description": a.description,
            "date": a.timestamp,
            "reason_codes": a.reason_codes or [],
            "contributing_factors": a.contributing_factors or [],
            "recommended_action": a.recommended_action or "",
        })

    return {"data": results, "total": total, "page": page, "page_size": page_size}


@router.post("/alerts/generate")
def generate_alert(
    payload: AlertGenerateRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    On-demand alert generation from model risk outputs.
    Call this after running inference to create a structured analyst alert.
    """
    alert_data = alert_generator.generate(
        account_id=payload.account_id,
        fraud_probability=payload.fraud_probability,
        anomaly_score=payload.anomaly_score,
        rules_score=payload.rules_score,
        risk_score=payload.risk_score,
        top_features=payload.top_features,
        transaction_id=payload.transaction_id,
        db=db,
    )
    log.info(f"POST /alerts/generate → {alert_data['alert_id']} [{alert_data['severity']}] for {payload.account_id}")
    return alert_data


def _update_status(alert_id: str, new_status: str, db: Session) -> Dict[str, Any]:
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = new_status
    now = datetime.now(timezone.utc)

    # Audit timestamps
    if new_status == "ACKNOWLEDGED":
        alert.acknowledged_at = now
    elif new_status == "ESCALATED":
        alert.escalated_at = now
    elif new_status == "RESOLVED":
        alert.resolved_at = now

    db.commit()
    log.info(f"Alert {alert_id} status updated → {new_status}")
    return {
        "status": "success",
        "alert_id": alert_id,
        "new_status": new_status,
        "updated_at": now.isoformat(),
    }


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, db: Session = Depends(get_db)):
    return _update_status(alert_id, "ACKNOWLEDGED", db)


@router.post("/alerts/{alert_id}/escalate")
def escalate_alert(alert_id: str, db: Session = Depends(get_db)):
    return _update_status(alert_id, "ESCALATED", db)


@router.post("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str, db: Session = Depends(get_db)):
    return _update_status(alert_id, "RESOLVED", db)
