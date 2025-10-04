#!/usr/bin/env python3
"""
Production-Ready Sepsis Prediction API
Enhanced FastAPI with optimized model, comprehensive validation, and clinical features
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Optional, Union
import os
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Sepsis Prediction API",
    description="Production-ready sepsis prediction with optimized XGBoost model and SHAP explanations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
model = None
scaler = None
feature_names = None
shap_explainer = None
model_metrics = None

class PatientData(BaseModel):
    """Enhanced patient data model with comprehensive validation"""
    
    # Vital Signs
    HR: Optional[float] = Field(None, ge=30, le=200, description="Heart Rate (bpm)")
    O2Sat: Optional[float] = Field(None, ge=70, le=100, description="Oxygen Saturation (%)")
    Temp: Optional[float] = Field(None, ge=32, le=45, description="Temperature (°C)")
    SBP: Optional[float] = Field(None, ge=60, le=250, description="Systolic Blood Pressure (mmHg)")
    MAP: Optional[float] = Field(None, ge=40, le=180, description="Mean Arterial Pressure (mmHg)")
    DBP: Optional[float] = Field(None, ge=30, le=150, description="Diastolic Blood Pressure (mmHg)")
    Resp: Optional[float] = Field(None, ge=5, le=50, description="Respiratory Rate (/min)")
    
    # Lab Values
    Glucose: Optional[float] = Field(None, ge=20, le=800, description="Glucose (mg/dL)")
    
    # Demographics
    Age: float = Field(..., ge=0, le=120, description="Age (years)")
    Gender: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    
    # Temporal
    ICULOS: Optional[float] = Field(None, ge=0, le=1000, description="ICU Length of Stay (hours)")
    
    @validator('HR')
    def validate_heart_rate(cls, v):
        if v is not None and (v < 30 or v > 200):
            raise ValueError('Heart rate must be between 30-200 bpm')
        return v
    
    @validator('Temp')
    def validate_temperature(cls, v):
        if v is not None and (v < 32 or v > 45):
            raise ValueError('Temperature must be between 32-45°C')
        return v

class PredictionResponse(BaseModel):
    """Enhanced prediction response model"""
    patient_id: str
    timestamp: str
    survival_probability: float
    mortality_probability: float
    risk_level: str
    risk_score: float
    prediction: str
    confidence: float
    shap_explanations: Optional[Dict[str, Dict[str, Union[float, str]]]]
    clinical_alerts: List[str]
    recommendations: List[str]
    model_version: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    """API health response model"""
    status: str
    timestamp: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int

# Global counters
prediction_counter = 0
start_time = datetime.now()

def load_model_components():
    """Load all model components with fallback to original model"""
    global model, scaler, feature_names, shap_explainer, model_metrics
    
    try:
        # Try to load optimized model first
        if os.path.exists('optimized_xgboost_sepsis_model.pkl'):
            logger.info("Loading optimized model...")
            model = joblib.load('optimized_xgboost_sepsis_model.pkl')
            scaler = joblib.load('optimized_scaler.pkl')
            feature_names = joblib.load('optimized_feature_names.pkl')
            model_metrics = joblib.load('optimized_model_metrics.pkl')
            logger.info("Optimized model loaded successfully")
        else:
            # Fallback to original model
            logger.info("Loading original model...")
            model = joblib.load('xgboost_sepsis_model.pkl')
            scaler = joblib.load('scaler.pkl')
            feature_names = joblib.load('feature_names.pkl')
            model_metrics = joblib.load('model_metrics.pkl')
            logger.info("Original model loaded successfully")
        
        # Try to load SHAP explainer
        try:
            shap_explainer = joblib.load('shap_explainer.pkl')
            logger.info("SHAP explainer loaded")
        except:
            logger.warning("SHAP explainer not available")
            shap_explainer = None
            
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

def get_risk_level(probability: float) -> tuple:
    """Determine risk level and score based on mortality probability"""
    if probability <= 0.1:
        return "low", 1
    elif probability <= 0.3:
        return "moderate", 2
    elif probability <= 0.6:
        return "high", 3
    else:
        return "critical", 4

def generate_clinical_alerts(patient_data: dict, risk_level: str) -> List[str]:
    """Generate clinical alerts based on patient data and risk level"""
    alerts = []
    
    # Vital sign alerts
    if patient_data.get('HR', 0) > 120:
        alerts.append("Tachycardia detected (HR > 120 bpm)")
    elif patient_data.get('HR', 0) < 50:
        alerts.append("Bradycardia detected (HR < 50 bpm)")
    
    if patient_data.get('Temp', 0) > 38.5:
        alerts.append("High fever detected (Temp > 38.5°C)")
    elif patient_data.get('Temp', 0) < 36:
        alerts.append("Hypothermia detected (Temp < 36°C)")
    
    if patient_data.get('SBP', 0) < 90:
        alerts.append("Hypotension detected (SBP < 90 mmHg)")
    
    if patient_data.get('Resp', 0) > 24:
        alerts.append("Tachypnea detected (RR > 24/min)")
    
    # Risk-based alerts
    if risk_level == "critical":
        alerts.append("CRITICAL: Immediate medical attention required")
    elif risk_level == "high":
        alerts.append("HIGH RISK: Enhanced monitoring recommended")
    
    return alerts

def generate_recommendations(risk_level: str, alerts: List[str]) -> List[str]:
    """Generate clinical recommendations based on risk level and alerts"""
    recommendations = []
    
    if risk_level == "critical":
        recommendations.extend([
            "Consider immediate ICU admission",
            "Initiate sepsis protocol immediately",
            "Obtain blood cultures and start empirical antibiotics",
            "Consider vasopressor support if hypotensive",
            "Monitor lactate levels closely"
        ])
    elif risk_level == "high":
        recommendations.extend([
            "Enhanced monitoring protocols",
            "Consider early intervention",
            "Regular reassessment every 2-4 hours",
            "Prepare for potential escalation of care"
        ])
    elif risk_level == "moderate":
        recommendations.extend([
            "Continue standard monitoring",
            "Regular vital sign checks",
            "Monitor for clinical deterioration",
            "Follow institutional sepsis protocols"
        ])
    else:
        recommendations.extend([
            "Routine monitoring sufficient",
            "Standard care protocols",
            "Continue current treatment plan"
        ])
    
    # Add specific recommendations based on alerts
    if any("fever" in alert.lower() for alert in alerts):
        recommendations.append("Consider antipyretic therapy and infection workup")
    
    if any("hypotension" in alert.lower() for alert in alerts):
        recommendations.append("Fluid resuscitation and hemodynamic support")
    
    return recommendations

@app.on_event("startup")
async def startup_event():
    """Load model components on startup"""
    logger.info("Starting Sepsis Prediction API...")
    load_model_components()
    logger.info("API startup complete")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced Sepsis Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "feature_importance": "/feature_importance",
            "model_info": "/model_info"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        model_version="optimized" if os.path.exists('optimized_xgboost_sepsis_model.pkl') else "original",
        uptime_seconds=uptime,
        total_predictions=prediction_counter
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(patient: PatientData):
    """Enhanced sepsis prediction with comprehensive analysis"""
    global prediction_counter
    start_time_pred = datetime.now()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert patient data to feature vector
        patient_dict = patient.dict()
        
        # Create feature vector matching training data
        feature_vector = []
        for feature_name in feature_names:
            if feature_name in patient_dict:
                value = patient_dict[feature_name]
                feature_vector.append(value if value is not None else 0)
            else:
                # Handle engineered features
                if feature_name == 'HR_SBP_ratio':
                    hr = patient_dict.get('HR', 70)
                    sbp = patient_dict.get('SBP', 120)
                    feature_vector.append(hr / (sbp + 1e-6) if hr and sbp else 0)
                elif feature_name == 'Temp_abnormal':
                    temp = patient_dict.get('Temp', 37)
                    feature_vector.append(1 if temp and (temp < 36 or temp > 38) else 0)
                elif feature_name == 'HR_abnormal':
                    hr = patient_dict.get('HR', 70)
                    feature_vector.append(1 if hr and (hr < 60 or hr > 100) else 0)
                elif feature_name == 'Age_elderly':
                    age = patient_dict.get('Age', 50)
                    feature_vector.append(1 if age >= 65 else 0)
                elif feature_name == 'ICULOS_long':
                    iculos = patient_dict.get('ICULOS', 24)
                    feature_vector.append(1 if iculos and iculos > 72 else 0)
                else:
                    feature_vector.append(0)  # Default value for missing features
        
        # Scale features
        feature_array = np.array(feature_vector).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction_proba = model.predict_proba(feature_scaled)[0]
        survival_prob = float(prediction_proba[1])
        mortality_prob = float(prediction_proba[0])
        
        # Determine risk level
        risk_level, risk_score = get_risk_level(mortality_prob)
        
        # Generate clinical insights
        alerts = generate_clinical_alerts(patient_dict, risk_level)
        recommendations = generate_recommendations(risk_level, alerts)
        
        # Calculate confidence (distance from decision boundary)
        confidence = float(abs(survival_prob - 0.5) * 2)
        
        # SHAP explanations (if available)
        shap_explanations = None
        if shap_explainer is not None:
            try:
                shap_values = shap_explainer.shap_values(feature_scaled)[0]
                shap_explanations = {}
                for i, feature_name in enumerate(feature_names):
                    if i < len(shap_values):
                        shap_explanations[feature_name] = {
                            "value": float(feature_vector[i]),
                            "shap_contribution": float(shap_values[i]),
                            "impact": "increases_survival" if shap_values[i] > 0 else "increases_mortality"
                        }
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
        
        # Processing time
        processing_time = (datetime.now() - start_time_pred).total_seconds() * 1000
        
        # Increment counter
        prediction_counter += 1
        
        # Generate patient ID
        patient_id = f"patient_{prediction_counter}_{int(datetime.now().timestamp())}"
        
        return PredictionResponse(
            patient_id=patient_id,
            timestamp=datetime.now().isoformat(),
            survival_probability=survival_prob,
            mortality_probability=mortality_prob,
            risk_level=risk_level,
            risk_score=risk_score,
            prediction=f"{'High' if mortality_prob > 0.5 else 'Low'} mortality risk",
            confidence=confidence,
            shap_explanations=shap_explanations,
            clinical_alerts=alerts,
            recommendations=recommendations,
            model_version="optimized" if os.path.exists('optimized_xgboost_sepsis_model.pkl') else "original",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/feature_importance")
async def get_feature_importance():
    """Get model feature importance"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if hasattr(model, 'feature_importances_'):
            importance_data = []
            for i, importance in enumerate(model.feature_importances_):
                feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                importance_data.append({
                    "feature": feature_name,
                    "importance": float(importance),
                    "description": get_feature_description(feature_name)
                })
            
            # Sort by importance
            importance_data.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                "feature_importance": importance_data,
                "model_type": type(model).__name__,
                "total_features": len(feature_names)
            }
        else:
            raise HTTPException(status_code=400, detail="Model does not support feature importance")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

def get_feature_description(feature_name: str) -> str:
    """Get clinical description for features"""
    descriptions = {
        'HR': 'Heart Rate - beats per minute',
        'O2Sat': 'Oxygen Saturation - percentage',
        'Temp': 'Body Temperature - degrees Celsius',
        'SBP': 'Systolic Blood Pressure - mmHg',
        'MAP': 'Mean Arterial Pressure - mmHg',
        'DBP': 'Diastolic Blood Pressure - mmHg',
        'Resp': 'Respiratory Rate - breaths per minute',
        'Glucose': 'Blood Glucose - mg/dL',
        'Age': 'Patient Age - years',
        'Gender': 'Patient Gender - 0=Female, 1=Male',
        'ICULOS': 'ICU Length of Stay - hours',
        'HR_SBP_ratio': 'Heart Rate to Systolic BP ratio',
        'Temp_abnormal': 'Temperature abnormality indicator',
        'HR_abnormal': 'Heart rate abnormality indicator',
        'Age_elderly': 'Elderly patient indicator (>=65 years)',
        'ICULOS_long': 'Long ICU stay indicator (>72 hours)'
    }
    return descriptions.get(feature_name, f"Clinical parameter: {feature_name}")

@app.get("/model_info")
async def get_model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_version = "optimized" if os.path.exists('optimized_xgboost_sepsis_model.pkl') else "original"
    
    info = {
        "model_type": type(model).__name__,
        "model_version": model_version,
        "feature_count": len(feature_names),
        "features": feature_names,
        "has_shap": shap_explainer is not None,
        "performance_metrics": model_metrics if model_metrics else {},
        "api_version": "2.0.0",
        "last_updated": datetime.now().isoformat()
    }
    
    return info

if __name__ == "__main__":
    uvicorn.run(
        "Production_Sepsis_API:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )