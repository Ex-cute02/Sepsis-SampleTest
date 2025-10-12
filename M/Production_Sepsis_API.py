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
    """Enhanced patient data model with comprehensive validation for all clinical features"""
    
    # Core Vital Signs
    HR: Optional[float] = Field(None, ge=20, le=250, description="Heart Rate (bpm)")
    O2Sat: Optional[float] = Field(None, ge=50, le=100, description="Oxygen Saturation (%)")
    Temp: Optional[float] = Field(None, ge=30, le=45, description="Temperature (°C)")
    SBP: Optional[float] = Field(None, ge=50, le=300, description="Systolic Blood Pressure (mmHg)")
    MAP: Optional[float] = Field(None, ge=30, le=200, description="Mean Arterial Pressure (mmHg)")
    DBP: Optional[float] = Field(None, ge=20, le=150, description="Diastolic Blood Pressure (mmHg)")
    Resp: Optional[float] = Field(None, ge=5, le=60, description="Respiratory Rate (/min)")
    
    # Laboratory Values
    Glucose: Optional[float] = Field(None, ge=20, le=1000, description="Glucose (mg/dL)")
    BUN: Optional[float] = Field(None, ge=1, le=200, description="Blood Urea Nitrogen (mg/dL)")
    Creatinine: Optional[float] = Field(None, ge=0.1, le=20, description="Creatinine (mg/dL)")
    WBC: Optional[float] = Field(None, ge=0.1, le=100, description="White Blood Cell count (K/μL)")
    Hct: Optional[float] = Field(None, ge=10, le=70, description="Hematocrit (%)")
    Hgb: Optional[float] = Field(None, ge=3, le=25, description="Hemoglobin (g/dL)")
    Platelets: Optional[float] = Field(None, ge=10, le=2000, description="Platelet count (K/μL)")
    Lactate: Optional[float] = Field(None, ge=0.1, le=30, description="Lactate (mmol/L)")
    
    # Additional Lab Values
    Bilirubin_total: Optional[float] = Field(None, ge=0.1, le=50, description="Total Bilirubin (mg/dL)")
    AST: Optional[float] = Field(None, ge=5, le=5000, description="AST (U/L)")
    Alkalinephos: Optional[float] = Field(None, ge=10, le=1000, description="Alkaline Phosphatase (U/L)")
    Calcium: Optional[float] = Field(None, ge=5, le=15, description="Calcium (mg/dL)")
    Chloride: Optional[float] = Field(None, ge=80, le=120, description="Chloride (mEq/L)")
    Magnesium: Optional[float] = Field(None, ge=0.5, le=5, description="Magnesium (mg/dL)")
    Phosphate: Optional[float] = Field(None, ge=1, le=10, description="Phosphate (mg/dL)")
    Potassium: Optional[float] = Field(None, ge=2, le=8, description="Potassium (mEq/L)")
    
    # Blood Gas and Respiratory
    pH: Optional[float] = Field(None, ge=6.8, le=7.8, description="Blood pH")
    PaCO2: Optional[float] = Field(None, ge=15, le=100, description="PaCO2 (mmHg)")
    BaseExcess: Optional[float] = Field(None, ge=-30, le=30, description="Base Excess (mEq/L)")
    HCO3: Optional[float] = Field(None, ge=5, le=50, description="Bicarbonate (mEq/L)")
    FiO2: Optional[float] = Field(None, ge=21, le=100, description="FiO2 (%)")
    
    # Demographics (Required)
    Age: float = Field(..., ge=0, le=120, description="Age (years)")
    Gender: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    
    # Temporal
    ICULOS: Optional[float] = Field(None, ge=0, le=2000, description="ICU Length of Stay (hours)")
    
    # Clinical Context (Optional)
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    unit: Optional[str] = Field(None, description="Hospital unit")
    
    @validator('HR')
    def validate_heart_rate(cls, v):
        if v is not None and (v < 20 or v > 250):
            raise ValueError('Heart rate must be between 20-250 bpm')
        return v
    
    @validator('Temp')
    def validate_temperature(cls, v):
        if v is not None and (v < 30 or v > 45):
            raise ValueError('Temperature must be between 30-45°C')
        return v
    
    @validator('O2Sat')
    def validate_oxygen_saturation(cls, v):
        if v is not None and (v < 50 or v > 100):
            raise ValueError('Oxygen saturation must be between 50-100%')
        return v
    
    @validator('pH')
    def validate_ph(cls, v):
        if v is not None and (v < 6.8 or v > 7.8):
            raise ValueError('pH must be between 6.8-7.8')
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
    """Load all model components with priority for enhanced preprocessing artifacts"""
    global model, scaler, feature_names, shap_explainer, model_metrics
    
    try:
        # Priority 1: Enhanced preprocessing artifacts
        if os.path.exists('enhanced_processed_dataset.csv') and os.path.exists('enhanced_scalers.pkl'):
            logger.info("Loading enhanced preprocessing artifacts...")
            
            # Load enhanced preprocessing components
            enhanced_scalers = joblib.load('enhanced_scalers.pkl')
            enhanced_feature_stats = joblib.load('enhanced_feature_stats.pkl')
            
            # Get the robust scaler from enhanced preprocessing
            scaler = enhanced_scalers.get('robust', enhanced_scalers.get('standard'))
            
            # Load enhanced feature names
            if 'selected_features' in enhanced_feature_stats:
                feature_names = enhanced_feature_stats['selected_features']
            else:
                feature_names = joblib.load('enhanced_feature_names.pkl') if os.path.exists('enhanced_feature_names.pkl') else None
            
            logger.info("Enhanced preprocessing components loaded")
            
        # Priority 2: Optimized model
        elif os.path.exists('optimized_xgboost_sepsis_model.pkl'):
            logger.info("Loading optimized model...")
            scaler = joblib.load('optimized_scaler.pkl')
            feature_names = joblib.load('optimized_feature_names.pkl')
            logger.info("Optimized model components loaded")
            
        # Priority 3: Original model (fallback)
        else:
            logger.info("Loading original model...")
            scaler = joblib.load('scaler.pkl')
            feature_names = joblib.load('feature_names.pkl')
            logger.info("Original model components loaded")
        
        # Load the best available model
        model_files = [
            ('enhanced_best_model.pkl', 'enhanced'),
            ('optimized_xgboost_sepsis_model.pkl', 'optimized'),
            ('xgboost_sepsis_model.pkl', 'original')
        ]
        
        model_loaded = False
        for model_file, model_type in model_files:
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                logger.info(f"{model_type.capitalize()} model loaded successfully")
                model_loaded = True
                break
        
        if not model_loaded:
            raise FileNotFoundError("No model file found")
        
        # Load metrics
        metrics_files = [
            'enhanced_model_metrics.pkl',
            'optimized_model_metrics.pkl', 
            'model_metrics.pkl'
        ]
        
        for metrics_file in metrics_files:
            if os.path.exists(metrics_file):
                model_metrics = joblib.load(metrics_file)
                break
        
        # Try to load SHAP explainer
        shap_files = [
            'enhanced_shap_explainer.pkl',
            'optimized_shap_explainer.pkl',
            'shap_explainer.pkl'
        ]
        
        for shap_file in shap_files:
            if os.path.exists(shap_file):
                try:
                    shap_explainer = joblib.load(shap_file)
                    logger.info(f"SHAP explainer loaded from {shap_file}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load SHAP from {shap_file}: {e}")
                    continue
        
        if shap_explainer is None:
            logger.warning("No SHAP explainer available")
            
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
    """Generate enhanced clinical alerts based on patient data and risk level"""
    alerts = []
    
    # Vital sign alerts with enhanced thresholds
    hr = patient_data.get('HR')
    if hr:
        if hr > 130:
            alerts.append("🔴 SEVERE Tachycardia detected (HR > 130 bpm)")
        elif hr > 100:
            alerts.append("🟡 Tachycardia detected (HR > 100 bpm)")
        elif hr < 50:
            alerts.append("🔴 Bradycardia detected (HR < 50 bpm)")
    
    temp = patient_data.get('Temp')
    if temp:
        if temp > 39:
            alerts.append("🔴 HIGH Fever detected (Temp > 39°C)")
        elif temp > 38.3:
            alerts.append("🟡 Fever detected (Temp > 38.3°C)")
        elif temp < 36:
            alerts.append("🔴 Hypothermia detected (Temp < 36°C)")
    
    sbp = patient_data.get('SBP')
    if sbp:
        if sbp < 80:
            alerts.append("🔴 SEVERE Hypotension detected (SBP < 80 mmHg)")
        elif sbp < 90:
            alerts.append("🟡 Hypotension detected (SBP < 90 mmHg)")
        elif sbp > 180:
            alerts.append("🟡 Hypertension detected (SBP > 180 mmHg)")
    
    resp = patient_data.get('Resp')
    if resp:
        if resp > 30:
            alerts.append("🔴 SEVERE Tachypnea detected (RR > 30/min)")
        elif resp > 24:
            alerts.append("🟡 Tachypnea detected (RR > 24/min)")
        elif resp < 10:
            alerts.append("🔴 Bradypnea detected (RR < 10/min)")
    
    o2sat = patient_data.get('O2Sat')
    if o2sat:
        if o2sat < 85:
            alerts.append("🔴 SEVERE Hypoxemia detected (O2Sat < 85%)")
        elif o2sat < 90:
            alerts.append("🟡 Hypoxemia detected (O2Sat < 90%)")
    
    # Laboratory alerts
    lactate = patient_data.get('Lactate')
    if lactate:
        if lactate > 4:
            alerts.append("🔴 SEVERE Hyperlactatemia detected (Lactate > 4 mmol/L)")
        elif lactate > 2:
            alerts.append("🟡 Hyperlactatemia detected (Lactate > 2 mmol/L)")
    
    wbc = patient_data.get('WBC')
    if wbc:
        if wbc > 20:
            alerts.append("🔴 SEVERE Leukocytosis detected (WBC > 20 K/μL)")
        elif wbc > 12:
            alerts.append("🟡 Leukocytosis detected (WBC > 12 K/μL)")
        elif wbc < 4:
            alerts.append("🟡 Leukopenia detected (WBC < 4 K/μL)")
    
    creatinine = patient_data.get('Creatinine')
    if creatinine:
        if creatinine > 3:
            alerts.append("🔴 SEVERE Renal dysfunction (Creatinine > 3 mg/dL)")
        elif creatinine > 1.5:
            alerts.append("🟡 Renal dysfunction (Creatinine > 1.5 mg/dL)")
    
    platelets = patient_data.get('Platelets')
    if platelets:
        if platelets < 50:
            alerts.append("🔴 SEVERE Thrombocytopenia (Platelets < 50 K/μL)")
        elif platelets < 100:
            alerts.append("🟡 Thrombocytopenia (Platelets < 100 K/μL)")
    
    # Clinical composite score alerts
    sirs_score = calculate_engineered_feature('SIRS_Score', patient_data)
    if sirs_score >= 3:
        alerts.append("🔴 HIGH SIRS Score detected (≥3 criteria)")
    elif sirs_score >= 2:
        alerts.append("🟡 Moderate SIRS Score detected (≥2 criteria)")
    
    qsofa_score = calculate_engineered_feature('qSOFA_Score', patient_data)
    if qsofa_score >= 2:
        alerts.append("🔴 HIGH qSOFA Score detected (≥2 criteria)")
    
    sofa_score = calculate_engineered_feature('SOFA_Total', patient_data)
    if sofa_score >= 6:
        alerts.append("🔴 HIGH SOFA Score detected (≥6 points)")
    elif sofa_score >= 2:
        alerts.append("🟡 Elevated SOFA Score detected (≥2 points)")
    
    # Shock index alert
    shock_index = calculate_engineered_feature('Shock_Index', patient_data)
    if shock_index > 1.0:
        alerts.append("🔴 Elevated Shock Index detected (>1.0)")
    elif shock_index > 0.9:
        alerts.append("🟡 Borderline Shock Index detected (>0.9)")
    
    # Risk-based alerts
    if risk_level == "critical":
        alerts.append("🚨 CRITICAL: Immediate medical attention required")
    elif risk_level == "high":
        alerts.append("⚠️ HIGH RISK: Enhanced monitoring recommended")
    elif risk_level == "moderate":
        alerts.append("🟡 MODERATE RISK: Close monitoring advised")
    
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

@app.post("/predict_batch")
async def predict_batch(patients: List[PatientData]):
    """Batch prediction endpoint for multiple patients"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        for i, patient in enumerate(patients):
            try:
                # Process each patient (reuse single prediction logic)
                patient_dict = patient.dict()
                feature_vector = []
                
                for feature_name in feature_names:
                    if feature_name in patient_dict:
                        value = patient_dict[feature_name]
                        feature_vector.append(value if value is not None else 0)
                    else:
                        value = calculate_engineered_feature(feature_name, patient_dict)
                        feature_vector.append(value)
                
                # Scale and predict
                feature_array = np.array(feature_vector).reshape(1, -1)
                feature_scaled = scaler.transform(feature_array)
                prediction_proba = model.predict_proba(feature_scaled)[0]
                
                mortality_prob = float(prediction_proba[0])
                risk_level, risk_score = get_risk_level(mortality_prob)
                
                predictions.append({
                    "patient_index": i,
                    "patient_id": patient_dict.get('patient_id', f'batch_patient_{i}'),
                    "survival_probability": float(prediction_proba[1]),
                    "mortality_probability": mortality_prob,
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "prediction": f"{'High' if mortality_prob > 0.5 else 'Low'} mortality risk"
                })
                
            except Exception as e:
                predictions.append({
                    "patient_index": i,
                    "error": str(e)
                })
        
        return {
            "batch_size": len(patients),
            "successful_predictions": len([p for p in predictions if "error" not in p]),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(patient: PatientData):
    """Enhanced sepsis prediction with comprehensive analysis"""
    global prediction_counter
    start_time_pred = datetime.now()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert patient data to feature vector with enhanced feature engineering
        patient_dict = patient.dict()
        
        # Create feature vector matching training data with enhanced features
        feature_vector = []
        for feature_name in feature_names:
            if feature_name in patient_dict:
                value = patient_dict[feature_name]
                feature_vector.append(value if value is not None else 0)
            else:
                # Handle enhanced engineered features
                value = calculate_engineered_feature(feature_name, patient_dict)
                feature_vector.append(value)
        
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

def calculate_engineered_feature(feature_name: str, patient_dict: dict) -> float:
    """Calculate enhanced engineered features based on patient data"""
    
    # Clinical Composite Scores
    if feature_name == 'SIRS_Score':
        sirs = 0
        temp = patient_dict.get('Temp')
        hr = patient_dict.get('HR')
        resp = patient_dict.get('Resp')
        wbc = patient_dict.get('WBC')
        
        if temp and (temp > 38 or temp < 36):
            sirs += 1
        if hr and hr > 90:
            sirs += 1
        if resp and resp > 20:
            sirs += 1
        if wbc and (wbc > 12 or wbc < 4):
            sirs += 1
        return sirs
    
    elif feature_name == 'qSOFA_Score':
        qsofa = 0
        sbp = patient_dict.get('SBP')
        resp = patient_dict.get('Resp')
        
        if sbp and sbp <= 100:
            qsofa += 1
        if resp and resp >= 22:
            qsofa += 1
        return qsofa
    
    elif feature_name == 'SOFA_Total':
        sofa = 0
        
        # Respiratory SOFA
        o2sat = patient_dict.get('O2Sat')
        if o2sat:
            if o2sat < 85:
                sofa += 4
            elif o2sat < 88:
                sofa += 3
            elif o2sat < 92:
                sofa += 2
            elif o2sat < 96:
                sofa += 1
        
        # Cardiovascular SOFA
        map_val = patient_dict.get('MAP')
        if map_val:
            if map_val < 55:
                sofa += 4
            elif map_val < 60:
                sofa += 3
            elif map_val < 65:
                sofa += 2
            elif map_val < 70:
                sofa += 1
        
        # Renal SOFA
        creatinine = patient_dict.get('Creatinine')
        if creatinine:
            if creatinine > 5.0:
                sofa += 4
            elif creatinine > 3.5:
                sofa += 3
            elif creatinine > 2.0:
                sofa += 2
            elif creatinine > 1.2:
                sofa += 1
        
        # Coagulation SOFA
        platelets = patient_dict.get('Platelets')
        if platelets:
            if platelets < 20:
                sofa += 4
            elif platelets < 50:
                sofa += 3
            elif platelets < 100:
                sofa += 2
            elif platelets < 150:
                sofa += 1
        
        return sofa
    
    # Physiological Ratios and Indices
    elif feature_name == 'Shock_Index':
        hr = patient_dict.get('HR', 70)
        sbp = patient_dict.get('SBP', 120)
        return hr / (sbp + 1e-6)
    
    elif feature_name == 'Pulse_Pressure':
        sbp = patient_dict.get('SBP', 120)
        dbp = patient_dict.get('DBP', 80)
        return sbp - dbp
    
    elif feature_name == 'BUN_Creatinine_Ratio':
        bun = patient_dict.get('BUN', 15)
        creatinine = patient_dict.get('Creatinine', 1.0)
        return bun / (creatinine + 1e-6)
    
    # Clinical Indicators
    elif feature_name == 'Hyperlactatemia':
        lactate = patient_dict.get('Lactate', 1.5)
        return 1 if lactate > 2.0 else 0
    
    elif feature_name == 'MAP_Critical':
        map_val = patient_dict.get('MAP', 70)
        return 1 if map_val < 65 else 0
    
    elif feature_name == 'Septic_Shock_Risk':
        map_val = patient_dict.get('MAP', 70)
        lactate = patient_dict.get('Lactate', 1.5)
        risk = 0
        if map_val < 65:
            risk += 1
        if lactate > 2.0:
            risk += 1
        return risk
    
    # Age Categories
    elif feature_name == 'Age_Elderly':
        age = patient_dict.get('Age', 50)
        return 1 if age >= 65 else 0
    
    elif feature_name == 'Age_Very_Elderly':
        age = patient_dict.get('Age', 50)
        return 1 if age >= 80 else 0
    
    # Temporal Features
    elif feature_name == 'ICU_Late':
        iculos = patient_dict.get('ICULOS', 24)
        return 1 if iculos > 72 else 0
    
    elif feature_name == 'ICU_Early':
        iculos = patient_dict.get('ICULOS', 24)
        return 1 if iculos <= 6 else 0
    
    # Organ Dysfunction Indicators
    elif feature_name == 'Organ_Dysfunction_Count':
        count = 0
        sbp = patient_dict.get('SBP')
        o2sat = patient_dict.get('O2Sat')
        creatinine = patient_dict.get('Creatinine')
        platelets = patient_dict.get('Platelets')
        
        if sbp and sbp < 90:
            count += 1
        if o2sat and o2sat < 90:
            count += 1
        if creatinine and creatinine > 2.0:
            count += 1
        if platelets and platelets < 100:
            count += 1
        
        return count
    
    # Abnormality Indicators
    elif feature_name in ['Temp_Fever', 'HR_Tachycardia', 'Resp_Tachypnea', 'SBP_Hypotension']:
        if 'Temp_Fever' in feature_name:
            temp = patient_dict.get('Temp', 37)
            return 1 if temp > 38.3 else 0
        elif 'HR_Tachycardia' in feature_name:
            hr = patient_dict.get('HR', 70)
            return 1 if hr > 100 else 0
        elif 'Resp_Tachypnea' in feature_name:
            resp = patient_dict.get('Resp', 16)
            return 1 if resp > 24 else 0
        elif 'SBP_Hypotension' in feature_name:
            sbp = patient_dict.get('SBP', 120)
            return 1 if sbp < 90 else 0
    
    # Default for unknown features
    return 0

def get_feature_description(feature_name: str) -> str:
    """Get clinical description for enhanced features"""
    descriptions = {
        # Basic Vitals
        'HR': 'Heart Rate - beats per minute',
        'O2Sat': 'Oxygen Saturation - percentage',
        'Temp': 'Body Temperature - degrees Celsius',
        'SBP': 'Systolic Blood Pressure - mmHg',
        'MAP': 'Mean Arterial Pressure - mmHg',
        'DBP': 'Diastolic Blood Pressure - mmHg',
        'Resp': 'Respiratory Rate - breaths per minute',
        
        # Laboratory Values
        'Glucose': 'Blood Glucose - mg/dL',
        'BUN': 'Blood Urea Nitrogen - mg/dL',
        'Creatinine': 'Serum Creatinine - mg/dL',
        'WBC': 'White Blood Cell Count - K/μL',
        'Hct': 'Hematocrit - percentage',
        'Hgb': 'Hemoglobin - g/dL',
        'Platelets': 'Platelet Count - K/μL',
        'Lactate': 'Serum Lactate - mmol/L',
        
        # Demographics
        'Age': 'Patient Age - years',
        'Gender': 'Patient Gender - 0=Female, 1=Male',
        'ICULOS': 'ICU Length of Stay - hours',
        
        # Clinical Scores
        'SIRS_Score': 'Systemic Inflammatory Response Syndrome Score (0-4)',
        'qSOFA_Score': 'Quick Sequential Organ Failure Assessment Score (0-3)',
        'SOFA_Total': 'Sequential Organ Failure Assessment Total Score',
        
        # Physiological Indices
        'Shock_Index': 'Shock Index (HR/SBP) - circulatory assessment',
        'Pulse_Pressure': 'Pulse Pressure (SBP-DBP) - cardiac function',
        'BUN_Creatinine_Ratio': 'BUN/Creatinine Ratio - renal function assessment',
        
        # Clinical Indicators
        'Hyperlactatemia': 'Elevated lactate indicator (>2.0 mmol/L)',
        'MAP_Critical': 'Critical MAP indicator (<65 mmHg)',
        'Septic_Shock_Risk': 'Septic shock risk indicator',
        'Organ_Dysfunction_Count': 'Number of organ systems with dysfunction',
        
        # Age Categories
        'Age_Elderly': 'Elderly patient indicator (≥65 years)',
        'Age_Very_Elderly': 'Very elderly patient indicator (≥80 years)',
        
        # Temporal Categories
        'ICU_Early': 'Early ICU stay indicator (≤6 hours)',
        'ICU_Late': 'Late ICU stay indicator (>72 hours)',
        
        # Abnormality Indicators
        'Temp_Fever': 'Fever indicator (>38.3°C)',
        'HR_Tachycardia': 'Tachycardia indicator (>100 bpm)',
        'Resp_Tachypnea': 'Tachypnea indicator (>24/min)',
        'SBP_Hypotension': 'Hypotension indicator (<90 mmHg)'
    }
    return descriptions.get(feature_name, f"Clinical parameter: {feature_name}")

@app.get("/model_info")
async def get_model_info():
    """Get detailed model information with enhanced preprocessing details"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Determine model version based on available files
    if os.path.exists('enhanced_processed_dataset.csv'):
        model_version = "enhanced_preprocessing"
    elif os.path.exists('optimized_xgboost_sepsis_model.pkl'):
        model_version = "optimized"
    else:
        model_version = "original"
    
    # Get preprocessing info
    preprocessing_info = {}
    if os.path.exists('enhanced_feature_stats.pkl'):
        try:
            enhanced_stats = joblib.load('enhanced_feature_stats.pkl')
            preprocessing_info = {
                "enhanced_preprocessing": True,
                "selected_features_count": len(enhanced_stats.get('selected_features', [])),
                "feature_selection_method": "multi_method_composite_scoring",
                "clinical_priority_features": True,
                "temporal_features": True,
                "physiological_features": True
            }
        except:
            preprocessing_info = {"enhanced_preprocessing": False}
    
    info = {
        "model_type": type(model).__name__,
        "model_version": model_version,
        "feature_count": len(feature_names),
        "features": feature_names[:20] if len(feature_names) > 20 else feature_names,  # Limit for readability
        "total_features": len(feature_names),
        "has_shap": shap_explainer is not None,
        "performance_metrics": model_metrics if model_metrics else {},
        "preprocessing_info": preprocessing_info,
        "supported_features": {
            "clinical_scores": ["SIRS_Score", "qSOFA_Score", "SOFA_Total"],
            "physiological_indices": ["Shock_Index", "Pulse_Pressure", "BUN_Creatinine_Ratio"],
            "clinical_alerts": True,
            "batch_prediction": True,
            "shap_explanations": shap_explainer is not None
        },
        "api_version": "2.1.0",
        "last_updated": datetime.now().isoformat()
    }
    
    return info

@app.get("/validate_model")
async def validate_model():
    """Validate model performance and readiness"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    validation_results = {
        "model_loaded": True,
        "scaler_loaded": scaler is not None,
        "feature_names_loaded": feature_names is not None,
        "shap_available": shap_explainer is not None,
        "feature_count": len(feature_names) if feature_names else 0,
        "model_type": type(model).__name__,
        "preprocessing_version": "enhanced" if os.path.exists('enhanced_processed_dataset.csv') else "standard"
    }
    
    # Test prediction with sample data
    try:
        sample_patient = {
            'Age': 65, 'Gender': 1, 'HR': 95, 'SBP': 110, 'Temp': 37.5,
            'Resp': 18, 'O2Sat': 95, 'WBC': 8.5, 'Lactate': 1.8
        }
        
        feature_vector = []
        for feature_name in feature_names:
            if feature_name in sample_patient:
                feature_vector.append(sample_patient[feature_name])
            else:
                value = calculate_engineered_feature(feature_name, sample_patient)
                feature_vector.append(value)
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)
        prediction_proba = model.predict_proba(feature_scaled)[0]
        
        validation_results.update({
            "prediction_test": "passed",
            "sample_prediction": {
                "survival_probability": float(prediction_proba[1]),
                "mortality_probability": float(prediction_proba[0])
            }
        })
        
    except Exception as e:
        validation_results.update({
            "prediction_test": "failed",
            "error": str(e)
        })
    
    # Overall status
    validation_results["status"] = "ready" if all([
        validation_results["model_loaded"],
        validation_results["scaler_loaded"],
        validation_results["feature_names_loaded"],
        validation_results.get("prediction_test") == "passed"
    ]) else "not_ready"
    
    return validation_results

if __name__ == "__main__":
    uvicorn.run(
        "Production_Sepsis_API:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )