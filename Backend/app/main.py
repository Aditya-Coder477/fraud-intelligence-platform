from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.api.routes import health, dashboard, accounts, alerts, model
from app.services.ml_service import ml_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Backend API for Banking Fraud Intelligence Platform"
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing ML models...")
    ml_service.load_models()
    logger.info("Application startup complete.")

# Include routers
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["system"])
app.include_router(dashboard.router, prefix=f"{settings.API_V1_STR}", tags=["dashboard"])
app.include_router(accounts.router, prefix=f"{settings.API_V1_STR}", tags=["accounts"])
app.include_router(alerts.router, prefix=f"{settings.API_V1_STR}", tags=["alerts"])
app.include_router(model.router, prefix=f"{settings.API_V1_STR}", tags=["ml"])
from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
