from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.dependencies import get_db
from app.schemas.ml import PredictionRequest, PredictionResponse
from app.services.ml_service import ml_service
from app.db.models import PredictionLog, Account
import uuid

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_fraud(payload: PredictionRequest, db: Session = Depends(get_db)):
    # Run prediction through the ML service
    result = ml_service.predict(payload.features)
    
    # Generate account_id if not provided
    acc_id = payload.account_id or f"ACC-NEW-{str(uuid.uuid4())[:8].upper()}"
    
    # Build explanation summary based on top features
    top_feats = [f["feature"] for f in result["top_features"]]
    summary = f"Account flagged due to elevated signals in: {', '.join(top_feats)}" if top_feats else "No specific features highlighted."
    
    # Store prediction log
    log = PredictionLog(
        account_id=acc_id,
        input_features=payload.features,
        fraud_probability=result["fraud_probability"],
        anomaly_score=payload.features.get("anomaly_score", 50.0),
        final_score=result["risk_score"],
        decision_recommendation=result["alert_recommendation"],
        explanation_summary=summary
    )
    db.add(log)
    
    # Update or create account record logic would go here in a full system
    # (Checking if Account exists, if so update risk_score, else create)
    
    db.commit()
    
    return PredictionResponse(
        prediction_label=result["prediction_label"],
        fraud_probability=result["fraud_probability"],
        risk_score=result["risk_score"],
        alert_recommendation=result["alert_recommendation"],
        explanation_summary=summary
    )

@router.get("/model/info")
def get_model_info():
    return {
        "status": "loaded" if ml_service.is_loaded else "mock_mode",
        "model_version": "1.0.0",
        "algorithm": "Ensemble (LGBM, XGB, CatBoost)",
        "framework": "scikit-learn"
    }

@router.post("/model/reload")
def reload_models():
    ml_service.load_models()
    return {"status": "success", "message": "Models reloaded."}
