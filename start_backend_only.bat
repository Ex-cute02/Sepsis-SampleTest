@echo off
echo Starting Sepsis Prediction API Server...
echo.

cd M
echo Backend API starting at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.

uvicorn sepsis_api_FastAPI:app --host 0.0.0.0 --port 8000 --reload