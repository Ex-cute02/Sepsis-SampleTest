# Step 5: Create a simple FastAPI application (our recommended backend framework)
fastapi_code = '''
"""
Sepsis Prediction API using FastAPI
This is a sample implementation for the MVP model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List
import shap

# Load trained model and preprocessor
model = joblib.load("xgboost_sepsis_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
shap_explainer = joblib.load("shap_explainer.pkl")

app = FastAPI(
    title="Sepsis Prediction API",
    description="ML-powered sepsis survival prediction with SHAP explanations",
    version="1.0.0"
)

class PatientData(BaseModel):
    age: float
    gender: int  # 0=Female, 1=Male
    heart_rate: float
    systolic_bp: float
    temperature: float
    respiratory_rate: float
    wbc_count: float
    lactate: float
    sofa_score: int

class PredictionResponse(BaseModel):
    survival_probability: float
    mortality_probability: float
    risk_level: str
    prediction: str
    shap_explanations: dict

@app.get("/")
async def root():
    return {
        "message": "Sepsis Prediction API",
        "status": "active",
        "model": "XGBoost with SHAP interpretability"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(patient: PatientData):
    try:
        # Convert input to array
        features = np.array([[
            patient.age, patient.gender, patient.heart_rate,
            patient.systolic_bp, patient.temperature, 
            patient.respiratory_rate, patient.wbc_count,
            patient.lactate, patient.sofa_score
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction_proba = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        
        # Calculate SHAP values for explanation
        shap_values = shap_explainer.shap_values(features_scaled)[0]
        
        # Create explanations
        explanations = {}
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
            explanations[feature] = {
                "value": float(features[0, i]),
                "shap_contribution": float(shap_val),
                "impact": "increases_survival" if shap_val > 0 else "decreases_survival"
            }
        
        # Determine risk level
        mortality_prob = prediction_proba[0]
        if mortality_prob < 0.1:
            risk_level = "low"
        elif mortality_prob < 0.3:
            risk_level = "moderate" 
        elif mortality_prob < 0.6:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        return PredictionResponse(
            survival_probability=float(prediction_proba[1]),
            mortality_probability=float(prediction_proba[0]),
            risk_level=risk_level,
            prediction="survived" if prediction == 1 else "deceased",
            shap_explanations=explanations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/feature_importance")
async def get_feature_importance():
    """Get global feature importance from the model"""
    feature_importance = model.feature_importances_
    importance_dict = {}
    for feature, importance in zip(feature_names, feature_importance):
        importance_dict[feature] = float(importance)
    
    return {"feature_importance": importance_dict}

@app.post("/batch_predict")
async def batch_predict(patients: List[PatientData]):
    """Predict for multiple patients"""
    predictions = []
    
    for patient in patients:
        try:
            # Process each patient
            features = np.array([[
                patient.age, patient.gender, patient.heart_rate,
                patient.systolic_bp, patient.temperature,
                patient.respiratory_rate, patient.wbc_count,
                patient.lactate, patient.sofa_score
            ]])
            
            features_scaled = scaler.transform(features)
            prediction_proba = model.predict_proba(features_scaled)[0]
            
            predictions.append({
                "survival_probability": float(prediction_proba[1]),
                "mortality_probability": float(prediction_proba[0]),
                "risk_level": "high" if prediction_proba[0] > 0.3 else "moderate" if prediction_proba[0] > 0.1 else "low"
            })
            
        except Exception as e:
            predictions.append({"error": str(e)})
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

# Save FastAPI application
with open('sepsis_api.py', 'w') as f:
    f.write(fastapi_code)

print("=== STEP 5: FASTAPI APPLICATION CREATED ===\n")
print("FastAPI application created: sepsis_api.py")
print("\nFeatures:")
print("✓ RESTful API endpoints")
print("✓ Real-time sepsis prediction")
print("✓ SHAP-based explanations")
print("✓ Batch prediction support")
print("✓ Health checks and monitoring")
print("✓ Pydantic data validation")

print("\nAPI Endpoints:")
print("- GET  /               : Root endpoint")
print("- GET  /health         : Health check")
print("- POST /predict        : Single patient prediction")
print("- GET  /feature_importance : Global feature importance")
print("- POST /batch_predict  : Multiple patient predictions")

print("\nTo run the API:")
print("pip install fastapi uvicorn")
print("python sepsis_api.py")
print("# API will be available at http://localhost:8000")
print("# Interactive docs at http://localhost:8000/docs")