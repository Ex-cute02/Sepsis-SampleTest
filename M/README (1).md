# Sepsis Prediction System - Sample MVP

This is a complete sample implementation of the sepsis prediction system using our recommended technology stack.

## Features

- **Machine Learning**: XGBoost model with SHAP interpretability
- **Backend**: FastAPI with RESTful endpoints
- **Frontend**: React with TypeScript
- **Containerization**: Docker and Docker Compose
- **Data Processing**: Pandas, Scikit-learn preprocessing
- **Model Serving**: Real-time predictions with explanations

## Quick Start

### 1. Backend API
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python sepsis_api.py

# API available at: http://localhost:8000
# Interactive docs at: http://localhost:8000/docs
```

### 2. With Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# API: http://localhost:8000
# Frontend: http://localhost:3000
```

### 3. Manual Testing
```bash
# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 65,
       "gender": 1,
       "heart_rate": 95,
       "systolic_bp": 110,
       "temperature": 38.2,
       "respiratory_rate": 22,
       "wbc_count": 12.5,
       "lactate": 3.2,
       "sofa_score": 4
     }'
```

## Project Structure

```
sepsis-prediction-mvp/
├── sepsis_api.py              # FastAPI backend
├── SepsisPredictor.tsx        # React frontend component
├── sepsis_sample_dataset.csv  # Sample dataset
├── xgboost_sepsis_model.pkl   # Trained model
├── scaler.pkl                 # Feature scaler
├── shap_explainer.pkl         # SHAP explainer
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-service setup
└── README.md                  # This file
```

## Model Performance

- **Algorithm**: XGBoost with SHAP interpretability
- **Dataset**: 1,000 synthetic sepsis patients
- **Features**: 9 clinical variables (age, vitals, lab values)
- **Performance**: AUC-ROC: 0.57, F1-Score: 0.85
- **Interpretability**: SHAP feature importance and individual explanations

## API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check
- `POST /predict`: Single patient prediction
- `GET /feature_importance`: Global feature importance
- `POST /batch_predict`: Batch predictions

## Key Clinical Features

1. **Age**: Patient age in years
2. **Gender**: 0=Female, 1=Male
3. **Heart Rate**: Beats per minute
4. **Systolic BP**: Blood pressure (mmHg)
5. **Temperature**: Body temperature (°C)
6. **Respiratory Rate**: Breaths per minute
7. **WBC Count**: White blood cell count (×10³/μL)
8. **Lactate**: Lactate level (mmol/L)
9. **SOFA Score**: Sequential organ failure assessment

## Technology Stack

- **Backend**: FastAPI, Python 3.11
- **ML**: XGBoost, SHAP, Scikit-learn
- **Frontend**: React, TypeScript, Tailwind CSS
- **Data**: Pandas, NumPy
- **Deployment**: Docker, Docker Compose
- **API**: RESTful with automatic OpenAPI docs

## Next Steps for Full MVP

1. **Database Integration**: PostgreSQL for patient records, InfluxDB for time-series
2. **Authentication**: OAuth 2.0 with role-based access
3. **Real-time Monitoring**: Kafka for streaming vital signs
4. **Model Serving**: MLflow for model versioning
5. **Frontend Enhancement**: Complete React application with dashboard
6. **Testing**: Unit tests, integration tests, clinical validation
7. **Deployment**: Kubernetes, cloud deployment (AWS/GCP/Azure)
8. **Monitoring**: Prometheus, Grafana for system health

## Disclaimer

This is a demonstration model for educational purposes. Always consult medical professionals for clinical decisions.
