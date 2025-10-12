@echo off
echo Starting Enhanced Sepsis Prediction System...
echo.

echo Checking system requirements...
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

echo Installing enhanced React frontend dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo Failed to install frontend dependencies
    pause
    exit /b 1
)

echo.
echo Starting Enhanced API server...
cd ..\M
start "Enhanced Sepsis API" cmd /k "python Production_Sepsis_API.py"

echo.
echo Waiting for Enhanced API server to start...
timeout /t 8 /nobreak > nul

echo.
echo Starting Enhanced React Frontend...
cd ..\frontend
start "Enhanced React Frontend" cmd /k "npm start"

echo.
echo Enhanced System is starting up...
echo ================================
echo Enhanced Backend API: http://localhost:8000
echo Enhanced Dashboard: http://localhost:3000
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo Model Info: http://localhost:8000/model_info
echo.
echo Enhanced Features:
echo - 40+ clinical parameters support
echo - Real-time clinical alerts
echo - Advanced SHAP explanations  
echo - Batch patient analysis
echo - Enhanced preprocessing integration
echo - Clinical recommendations
echo.
echo Press any key to exit...
pause > nul