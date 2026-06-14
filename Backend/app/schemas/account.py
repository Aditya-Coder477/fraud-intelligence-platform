from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class AccountBase(BaseModel):
    account_id: str
    risk_score: float
    fraud_probability: float
    alert_count: int
    status: str
    category: str

class AccountResponse(AccountBase):
    last_activity: Optional[datetime] = None

class AccountDetailResponse(AccountResponse):
    profile: Optional[Dict[str, Any]] = None
    explanation_summary: Optional[str] = None
    
    class Config:
        from_attributes = True

class TransactionBase(BaseModel):
    transaction_id: str
    account_id: str
    amount: float
    type: str
    counterparty: Optional[str] = None
    status: str

class TransactionResponse(TransactionBase):
    timestamp: datetime
    
    class Config:
        from_attributes = True
