@echo off
echo Starting Enhanced Sepsis Prediction API Server...
echo.

echo Checking for required model files...
if not exist "M\enhanced_processed_dataset.csv" (
    if not exist "M\optimized_xgboost_sepsis_model.pkl" (
        if not exist "M\xgboost_sepsis_model.pkl" (
            echo ERROR: No model files found!
            echo Please run the preprocessing and training pipeline first.
            pause
            exit /b 1
        )
    )
)

cd M
echo Enhanced Backend API starting at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo Model Info: http://localhost:8000/model_info
echo.

python Production_Sepsis_API.py