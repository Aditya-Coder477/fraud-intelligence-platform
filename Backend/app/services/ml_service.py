import os
import sys
import pandas as pd
import joblib

# Add parent directory to sys.path to import existing ML modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from risk_scoring_engine import BankingRiskScoringEngine, RiskInput
    from fraud_detection_model import MuleAccountInferencePipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("WARNING: ML Modules not found. Running in mock/fallback mode.")

class MLService:
    def __init__(self):
        self.inference_pipeline = None
        self.risk_engine = None
        self.is_loaded = False

    def load_models(self):
        if not ML_AVAILABLE:
            return
            
        try:
            # Assuming models are in the parent 'models' folder and engine in 'risk_engine'
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            
            pipeline_path = os.path.join(base_dir, 'models', 'inference_pipeline.pkl')
            self.inference_pipeline = MuleAccountInferencePipeline.load(pipeline_path)
            
            engine_path = os.path.join(base_dir, 'risk_engine', 'engine.pkl')
            if os.path.exists(engine_path):
                self.risk_engine = joblib.load(engine_path)
            else:
                self.risk_engine = BankingRiskScoringEngine()
                
            self.is_loaded = True
            print("Successfully loaded ML models and Risk Engine.")
        except Exception as e:
            print(f"Error loading models: {e}. Fallback mode active.")
            self.is_loaded = False

    def predict(self, features: dict):
        if not self.is_loaded or not self.inference_pipeline:
            # Mock response for API testing if models aren't present
            return self._mock_prediction()
            
        try:
            df = pd.DataFrame([features])
            
            # Predict
            preds = self.inference_pipeline.predict(df)
            mule_prob = float(preds["mule_prob"].iloc[0])
            threshold = self.inference_pipeline.thresholds.get("Ensemble", 0.5)
            pred_class = int(mule_prob >= threshold)
            
            # Anomaly and Alert scores would ideally come from the DB or separate calls
            anomaly_score = features.get("anomaly_score", 50.0)
            alert_score = features.get("alert_score", 50.0)
            
            risk_input = RiskInput(
                ml_prob=mule_prob * 100,
                anomaly_score=anomaly_score,
                behavior_score=50.0, # default
                alert_score=alert_score
            )
            
            risk_result = self.risk_engine.score(risk_input)
            
            # Extract top 3 SHAP features
            explanation = []
            try:
                shap_df = self.inference_pipeline.explain(df, top_n=3)
                top_feats = shap_df["top_features"].iloc[0]
                top_contribs = shap_df["shap_contributions"].iloc[0]
                
                for feat, imp in zip(top_feats, top_contribs):
                    explanation.append({"feature": feat, "importance": float(imp), "description": f"Model feature {feat}"})
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
                    
            return {
                "prediction_label": pred_class,
                "fraud_probability": mule_prob,
                "risk_score": risk_result.final_score,
                "score_band": risk_result.category,
                "alert_recommendation": risk_result.action,
                "top_features": explanation
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._mock_prediction()

    def _mock_prediction(self):
        return {
            "prediction_label": 1,
            "fraud_probability": 0.89,
            "risk_score": 85.0,
            "score_band": "BLOCK",
            "alert_recommendation": "Immediate block recommended.",
            "top_features": [
                {"feature": "Velocity_7d", "importance": 0.45, "description": "High velocity"},
                {"feature": "Age_Days", "importance": 0.30, "description": "New account"}
            ]
        }

ml_service = MLService()
