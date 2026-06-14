import os
import sys
import uuid
from datetime import datetime, timedelta
import random

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.db.database import engine, Base, SessionLocal
from app.db.models import Account, Transaction, Alert

def init_db():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

def seed():
    db = SessionLocal()
    try:
        # Check if already seeded
        if db.query(Account).count() > 0:
            print("Database already seeded. Skipping.")
            return

        print("Seeding accounts...")
        accounts_data = [
            {"id": "ACC-8921", "risk": 94, "prob": 0.95, "cat": "BLOCK", "status": "suspended"},
            {"id": "ACC-3342", "risk": 88, "prob": 0.89, "cat": "BLOCK", "status": "active"},
            {"id": "ACC-1092", "risk": 76, "prob": 0.75, "cat": "REVIEW", "status": "investigating"},
            {"id": "ACC-5521", "risk": 65, "prob": 0.68, "cat": "REVIEW", "status": "active"},
            {"id": "ACC-9982", "risk": 52, "prob": 0.45, "cat": "MONITOR", "status": "active"},
            {"id": "ACC-4411", "risk": 41, "prob": 0.38, "cat": "MONITOR", "status": "active"},
            {"id": "ACC-2210", "risk": 15, "prob": 0.05, "cat": "SAFE", "status": "active"},
        ]

        for a in accounts_data:
            acc = Account(
                account_id=a["id"],
                risk_score=a["risk"],
                fraud_probability=a["prob"],
                anomaly_score=random.uniform(50, 95) if a["risk"] > 50 else random.uniform(10, 40),
                rules_score=random.uniform(50, 95) if a["risk"] > 50 else random.uniform(10, 40),
                score_band=a["cat"],
                category=a["cat"],
                status=a["status"],
                alert_count=random.randint(0, 4) if a["risk"] > 60 else 0,
                profile_data={"type": "Retail", "tenure_days": random.randint(1, 365)}
            )
            db.add(acc)

        db.commit()

        print("Seeding alerts & transactions...")
        for a in accounts_data:
            if a["risk"] > 60:
                alert = Alert(
                    alert_id=f"ALT-{random.randint(1000, 9999)}",
                    account_id=a["id"],
                    severity="CRITICAL" if a["risk"] > 80 else "HIGH",
                    description=f"High risk score detected for {a['id']}.",
                    timestamp=datetime.now() - timedelta(minutes=random.randint(5, 60))
                )
                db.add(alert)
                
            # Add some transactions
            for _ in range(3):
                txn = Transaction(
                    transaction_id=f"TXN-{random.randint(100000, 999999)}",
                    account_id=a["id"],
                    amount=random.uniform(100, 15000),
                    type=random.choice(["Wire Transfer", "ACH", "Card Payment"]),
                    counterparty="Entity_" + str(random.randint(1, 50)),
                    status="COMPLETED" if a["risk"] < 80 else "BLOCKED",
                    timestamp=datetime.now() - timedelta(days=random.randint(0, 5))
                )
                db.add(txn)

        db.commit()
        print("Seeding complete!")

    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    seed()
