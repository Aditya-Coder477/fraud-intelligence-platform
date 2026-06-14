from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base


class Account(Base):
    __tablename__ = "accounts"

    account_id = Column(String, primary_key=True, index=True)
    profile_data = Column(JSON, nullable=True)
    risk_score = Column(Float, nullable=False, default=0.0)
    fraud_probability = Column(Float, nullable=False, default=0.0)
    anomaly_score = Column(Float, nullable=False, default=0.0)
    rules_score = Column(Float, nullable=False, default=0.0)
    score_band = Column(String, nullable=False, default="SAFE")
    category = Column(String, nullable=False, default="SAFE")
    status = Column(String, nullable=False, default="active")
    alert_count = Column(Integer, default=0)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())

    transactions = relationship("Transaction", back_populates="account")
    alerts = relationship("Alert", back_populates="account")
    predictions = relationship("PredictionLog", back_populates="account")
    explanations = relationship("ExplanationLog", back_populates="account")


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True, index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"))
    amount = Column(Float, nullable=False)
    type = Column(String, nullable=False)
    counterparty = Column(String, nullable=True)
    status = Column(String, nullable=False, default="COMPLETED")
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    account = relationship("Account", back_populates="transactions")


class Alert(Base):
    __tablename__ = "alerts"

    alert_id = Column(String, primary_key=True, index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"))
    transaction_id = Column(String, nullable=True)
    alert_type = Column(String, nullable=False, default="MODEL_SCORE")  # MODEL_SCORE, ANOMALY, RULE, CONVERGENT
    severity = Column(String, nullable=False)                            # CRITICAL, HIGH, MEDIUM, LOW
    status = Column(String, nullable=False, default="OPEN")             # OPEN, ACKNOWLEDGED, ESCALATED, RESOLVED
    description = Column(String, nullable=False)
    # Rich fields for analyst consumption
    reason_codes = Column(JSON, nullable=True)          # ["R01: High Velocity", "R04: Pass-through"]
    contributing_factors = Column(JSON, nullable=True)  # [{"factor": "...", "value": "...", "weight": 0.4}]
    recommended_action = Column(String, nullable=True)
    # Audit trail
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    escalated_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    action_by = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    account = relationship("Account", back_populates="alerts")


class ExplanationLog(Base):
    """Stores SHAP-based explanations per account for audit and replay."""
    __tablename__ = "explanation_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    account_id = Column(String, ForeignKey("accounts.account_id"))
    top_features = Column(JSON, nullable=False)    # [{feature, importance, direction, description}, ...]
    overall_summary = Column(Text, nullable=True)
    reason_codes = Column(JSON, nullable=True)
    model_version = Column(String, nullable=True, default="1.0.0")
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    account = relationship("Account", back_populates="explanations")


class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    account_id = Column(String, ForeignKey("accounts.account_id"))
    input_features = Column(JSON, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    final_score = Column(Float, nullable=False)
    decision_recommendation = Column(String, nullable=True)
    explanation_summary = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    account = relationship("Account", back_populates="predictions")
