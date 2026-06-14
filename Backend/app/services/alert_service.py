"""
Alert Generator Service
=======================
Converts raw model/risk scores into analyst-friendly alerts with:
  - Severity classification (CRITICAL / HIGH / MEDIUM / LOW)
  - Reason codes aligned with regulatory AML codes
  - Contributing factors with weights
  - Plain-English recommended next action
  - Convergent-evidence detection (≥2 independent signals agree)
"""
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.db.models import Alert, Account

log = logging.getLogger("AlertService")

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "CRITICAL": {"risk_score": 81, "fraud_prob": 0.85},
    "HIGH":     {"risk_score": 61, "fraud_prob": 0.65},
    "MEDIUM":   {"risk_score": 40, "fraud_prob": 0.40},
}

# ── Reason code library ───────────────────────────────────────────────────────
REASON_CODES = {
    "high_fraud_prob":   "R01: ML Model — High Fraud Probability",
    "high_anomaly":      "R02: Anomaly Detection — Statistical Outlier",
    "high_rules":        "R03: Rule Engine — Policy Threshold Breach",
    "convergent":        "R04: Convergent Evidence — Multiple Signals Agree",
    "new_account":       "R05: New Account — High Activity Profile",
    "pass_through":      "R06: Flow Pattern — Near-Zero Net Balance (Pass-Through)",
    "velocity":          "R07: Transaction Velocity — Abnormal Frequency",
}

# ── Recommended actions per severity ─────────────────────────────────────────
ACTIONS = {
    "CRITICAL": "Immediately block account, freeze pending transactions, file Suspicious Activity Report (SAR), and escalate to senior AML officer.",
    "HIGH":     "Place account on enhanced monitoring hold. Analyst review required within 4 hours. Consider temporary transaction limits.",
    "MEDIUM":   "Add to enhanced monitoring queue. Schedule analyst review within 24 hours. Request additional KYC documentation.",
    "LOW":      "Log for periodic review. Monitor for additional suspicious signals over the next 7 days.",
}


class AlertGeneratorService:
    """
    Generates structured, analyst-friendly alerts from model risk outputs.
    """

    def determine_severity(self, risk_score: float, fraud_prob: float, anomaly_score: float) -> str:
        if risk_score >= THRESHOLDS["CRITICAL"]["risk_score"] or fraud_prob >= THRESHOLDS["CRITICAL"]["fraud_prob"]:
            return "CRITICAL"
        if risk_score >= THRESHOLDS["HIGH"]["risk_score"] or fraud_prob >= THRESHOLDS["HIGH"]["fraud_prob"]:
            return "HIGH"
        if risk_score >= THRESHOLDS["MEDIUM"]["risk_score"] or fraud_prob >= THRESHOLDS["MEDIUM"]["fraud_prob"]:
            return "MEDIUM"
        return "LOW"

    def determine_alert_type(self, fraud_prob: float, anomaly_score: float, rules_score: float) -> str:
        """Returns primary alert trigger type."""
        elevated = sum([
            fraud_prob >= 0.60,
            anomaly_score >= 60,
            rules_score >= 60,
        ])
        if elevated >= 2:
            return "CONVERGENT"
        if fraud_prob >= 0.60:
            return "MODEL_SCORE"
        if anomaly_score >= 60:
            return "ANOMALY"
        return "RULE"

    def build_reason_codes(self, fraud_prob: float, anomaly_score: float, rules_score: float,
                           top_features: Optional[List[Dict]] = None) -> List[str]:
        codes = []
        if fraud_prob >= 0.60:
            codes.append(REASON_CODES["high_fraud_prob"])
        if anomaly_score >= 60:
            codes.append(REASON_CODES["high_anomaly"])
        if rules_score >= 60:
            codes.append(REASON_CODES["high_rules"])
        # Convergent
        if len(codes) >= 2:
            codes.insert(0, REASON_CODES["convergent"])

        # Feature-derived codes
        if top_features:
            feature_names = [f.get("feature", "").lower() for f in top_features[:5]]
            if any("velocity" in fn or "freq" in fn for fn in feature_names):
                codes.append(REASON_CODES["velocity"])
            if any("flow" in fn or "net" in fn or "in_out" in fn for fn in feature_names):
                codes.append(REASON_CODES["pass_through"])
            if any("age" in fn or "date" in fn for fn in feature_names):
                codes.append(REASON_CODES["new_account"])

        return list(dict.fromkeys(codes))  # deduplicate, preserve order

    def build_contributing_factors(self, fraud_prob: float, anomaly_score: float,
                                   rules_score: float, risk_score: float) -> List[Dict]:
        factors = []
        if fraud_prob > 0:
            factors.append({
                "factor": "ML Fraud Probability",
                "value": f"{fraud_prob * 100:.1f}%",
                "weight": round(fraud_prob, 3)
            })
        if anomaly_score > 0:
            factors.append({
                "factor": "Anomaly Detection Score",
                "value": f"{anomaly_score:.1f}/100",
                "weight": round(anomaly_score / 100, 3)
            })
        if rules_score > 0:
            factors.append({
                "factor": "Rule-Based Alert Score",
                "value": f"{rules_score:.1f}/100",
                "weight": round(rules_score / 100, 3)
            })
        factors.append({
            "factor": "Composite Risk Score",
            "value": f"{risk_score:.1f}/100",
            "weight": round(risk_score / 100, 3)
        })
        return sorted(factors, key=lambda x: x["weight"], reverse=True)

    def build_description(self, account_id: str, severity: str, alert_type: str,
                          fraud_prob: float, anomaly_score: float, rules_score: float,
                          top_features: Optional[List[Dict]] = None) -> str:
        prob_pct = f"{fraud_prob * 100:.0f}%"

        if alert_type == "CONVERGENT":
            signals = []
            if fraud_prob >= 0.60:
                signals.append(f"ML fraud probability ({prob_pct})")
            if anomaly_score >= 60:
                signals.append(f"anomaly score ({anomaly_score:.0f}/100)")
            if rules_score >= 60:
                signals.append(f"rule engine score ({rules_score:.0f}/100)")
            signal_str = ", ".join(signals)
            desc = (
                f"Convergent evidence from {len(signals)} independent detection systems: "
                f"{signal_str}. "
            )
        elif alert_type == "MODEL_SCORE":
            desc = f"ML ensemble model assigned a fraud probability of {prob_pct} to this account. "
        elif alert_type == "ANOMALY":
            desc = f"Anomaly detection flagged this account with a score of {anomaly_score:.0f}/100, indicating significant deviation from normal behaviour. "
        else:
            desc = f"Rule-based policy engine triggered on this account (score: {rules_score:.0f}/100). "

        # Feature narrative
        if top_features and len(top_features) >= 2:
            feat_names = [f.get("description", f.get("feature", "")) for f in top_features[:3]]
            desc += f"Primary drivers: {'; '.join(feat_names)}."

        return desc

    def generate(
        self,
        account_id: str,
        fraud_probability: float,
        anomaly_score: float,
        rules_score: float,
        risk_score: float,
        top_features: Optional[List[Dict]] = None,
        transaction_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured alert dict. Persists to DB if session is provided.
        """
        severity = self.determine_severity(risk_score, fraud_probability, anomaly_score)
        alert_type = self.determine_alert_type(fraud_probability, anomaly_score, rules_score)
        reason_codes = self.build_reason_codes(fraud_probability, anomaly_score, rules_score, top_features)
        contributing_factors = self.build_contributing_factors(fraud_probability, anomaly_score, rules_score, risk_score)
        description = self.build_description(account_id, severity, alert_type,
                                             fraud_probability, anomaly_score, rules_score, top_features)
        recommended_action = ACTIONS[severity]
        alert_id = f"ALT-{uuid.uuid4().hex[:6].upper()}"

        alert_data = {
            "alert_id": alert_id,
            "account_id": account_id,
            "transaction_id": transaction_id,
            "alert_type": alert_type,
            "severity": severity,
            "status": "OPEN",
            "description": description,
            "reason_codes": reason_codes,
            "contributing_factors": contributing_factors,
            "recommended_action": recommended_action,
            "timestamp": datetime.now(timezone.utc),
        }

        if db is not None:
            try:
                alert = Alert(**alert_data)
                db.add(alert)
                # Increment account alert_count
                account = db.query(Account).filter(Account.account_id == account_id).first()
                if account:
                    account.alert_count = (account.alert_count or 0) + 1
                db.commit()
                log.info(f"Alert {alert_id} generated for account {account_id} [{severity}]")
            except Exception as e:
                db.rollback()
                log.error(f"Failed to persist alert: {e}")

        return alert_data


alert_generator = AlertGeneratorService()
