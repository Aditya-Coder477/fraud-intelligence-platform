from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "message": "Fraud Intelligence Platform Backend is running."}

@router.get("/version")
def get_version():
    return {"version": "1.0.0", "api": "v1"}
