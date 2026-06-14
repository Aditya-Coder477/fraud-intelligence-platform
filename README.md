# FraudIntel: Suspicious Mule Accounts Classification

![Dashboard Preview](./frontend/public/fraudintel-preview.png)

An enterprise-grade Banking Fraud Intelligence Platform built to detect, explain, and alert on suspicious pass-through mule accounts. 

By combining advanced Machine Learning (LightGBM/XGBoost/CatBoost) with an Isolation Forest anomaly layer and a deterministic Rule-Based Scoring Engine, FraudIntel provides analysts with highly accurate risk detection and regulatory-grade narrative explanations.

## 🚀 Key Features

* **Multi-Layered Detection Engine:**
  * **Ensemble ML Model:** Evaluates complex behavioral patterns (e.g., velocity, inflow/outflow balance, profile missingness) using LightGBM, XGBoost, and CatBoost.
  * **Anomaly Detection:** Utilizes an Isolation Forest to flag statistical outliers in transaction volumes and frequencies that the ML model might miss.
  * **Rule-Based Engine:** Deterministic checks for known money laundering typologies (e.g., high-velocity pass-through, immediate fund depletion).
* **Convergent Evidence Alerts:** The Alert Generator Service triggers CRITICAL alerts only when multiple independent systems (ML + Anomaly + Rules) agree, drastically reducing false positives.
* **Explainable AI (XAI):** Real-time, directional SHAP values power a regulatory-grade narrative engine, translating complex mathematical feature importance into plain-English sentences for analysts.
* **Operational Dashboard:** A modern Next.js frontend featuring real-time risk scores, interactive SHAP-bar charts, and a comprehensive Alerts Center for analyst triage.

---

## 🏗️ Architecture

The platform is designed with a clear separation of concerns, ensuring scalability from local development to production deployment.

### 1. The Data & ML Pipeline (Python)
- **Feature Engineering:** `feature_engineering_pipeline.py` standardizes categorical features, scales numerical inputs, and derives complex behavioral features (e.g., `beh_flow_imbalance`, `freq_rarity_score`).
- **Model Training:** `fraud_detection_model.py` trains the ensemble classifier, applying SMOTE to handle class imbalance, and calculates global SHAP values.
- **Risk Scoring Engine:** `risk_scoring_engine.py` unifies the ML probability, anomaly score, and rule triggers into a composite 0-100 `risk_score`.

### 2. The Backend API (FastAPI / PostgreSQL)
- **Database (`app/db`):** Tracks Accounts, Transactions, Prediction Logs, and Alerts using SQLAlchemy.
- **Services (`app/services`):** 
  - `ExplanationService`: Processes SHAP data to build analyst-ready narratives.
  - `AlertGeneratorService`: Evaluates risk thresholds and generates actionable alerts with AML reason codes.
- **REST APIs (`app/api/routes`):** Serves real-time data to the frontend (e.g., `/accounts/{id}/explanations`, `/alerts`, `/dashboard/summary`).

### 3. The Frontend (Next.js / Tailwind CSS / Recharts)
- Responsive, dark-themed UI optimized for fast analyst workflow.
- **Executive Overview:** Top-level metrics and 7-day trend analysis.
- **Alerts Center:** Filterable triage queue (Critical/High/Medium/Low).
- **Account Intelligence Page:** Deep-dive view of an account's composite score, transaction history, and an interactive SHAP explanation panel.

---

## 🛠️ Tech Stack

* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `imbalanced-learn`
* **Backend:** Python 3.10+, FastAPI, Uvicorn, SQLAlchemy, PostgreSQL, Pydantic
* **Frontend:** React 18, Next.js 14, TypeScript, Tailwind CSS, Recharts, Lucide React, shadcn/ui

---

## ⚙️ Getting Started (Local Development)

### Prerequisites
* Python 3.10 or higher
* Node.js 18 or higher
* Docker Desktop (for the PostgreSQL database)

### 1. Start the Database
The backend relies on PostgreSQL. Start the database using Docker:
```bash
cd Backend
docker-compose up -d
```

### 2. Set up the Backend
Create a virtual environment, install dependencies, and sync the database tables:
```bash
cd Backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt

# Create the database tables
python sync_tables.py

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```
*The API will be available at `http://localhost:8000`. You can view the interactive Swagger docs at `http://localhost:8000/docs`.*

### 3. Set up the Frontend
Open a new terminal window, navigate to the frontend directory, install packages, and start the development server:
```bash
cd frontend
npm install
npm run dev
```
*The Dashboard will be available at `http://localhost:3000`.*

---

## 📂 Project Structure

```text
├── .gitignore                      # Git exclusion rules
├── Backend/                        # FastAPI Backend Application
│   ├── app/
│   │   ├── api/routes/             # REST Endpoints (alerts, accounts, ML)
│   │   ├── core/                   # Config & Settings
│   │   ├── db/                     # SQLAlchemy Models & DB Session
│   │   ├── schemas/                # Pydantic validation schemas
│   │   └── services/               # Core business logic (Alerts, Explainability)
│   ├── docker-compose.yml          # PostgreSQL setup
│   └── sync_tables.py              # DB Migration script
├── frontend/                       # Next.js UI
│   ├── src/
│   │   ├── app/                    # Next.js App Router pages
│   │   ├── components/             # Reusable UI components & Layouts
│   │   └── lib/                    # API client and utility functions
│   ├── public/                     # Static assets
│   └── tailwind.config.ts          # Styling configuration
├── models/                         # Serialized ML models (*.pkl)
├── *_pipeline.py                   # Data processing and training scripts
├── shap_feature_importance.csv     # Pre-calculated SHAP explanations
└── README.md                       # You are here
```

---

## 🔒 Security & Disclaimer

This project was built for analytical and demonstration purposes to highlight modern ML operationalization and Explainable AI (XAI) techniques in anti-money laundering (AML) scenarios. 
**Do not deploy this system into a production banking environment without comprehensive security audits, RBAC implementation, and rigorous compliance checks.**
