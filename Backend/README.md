# Fraud Intelligence Platform - Backend

This is the FastAPI backend for the Banking Fraud and Mule-Account Detection Platform. It serves REST APIs to the Next.js frontend, manages PostgreSQL records for accounts/alerts/transactions, and runs the trained Machine Learning inference pipeline.

## Features

- **FastAPI**: High performance asynchronous API.
- **PostgreSQL & SQLAlchemy**: Production-ready data layer with Alembic migrations.
- **ML Integration**: Seamlessly imports the `risk_scoring_engine` and `fraud_detection_model` from the parent directory.
- **Pydantic**: Strict request and response schemas matching the frontend.

## Setup Instructions

### 1. Start PostgreSQL
If you don't have a local Postgres instance running, you can use the provided Docker Compose file:
```bash
docker-compose up -d
```

### 2. Python Environment
Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Database Initialization
Before running the server, run the seed script to create the tables and inject some mock data so the dashboard has content:
```bash
python scripts/seed_db.py
```

### 4. Start the API Server
```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
You can view the auto-generated Swagger documentation at `http://localhost:8000/docs`.
