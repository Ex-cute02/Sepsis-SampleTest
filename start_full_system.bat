@echo off
echo Starting Sepsis Prediction System...
echo.

echo Installing frontend dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo Failed to install frontend dependencies
    pause
    exit /b 1
)

echo.
echo Starting backend API server...
cd ..\M
start "Sepsis API Server" cmd /k "uvicorn sepsis_api_FastAPI:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo Waiting for API server to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting React frontend...
cd ..\frontend
start "React Frontend" cmd /k "npm start"

echo.
echo System is starting up...
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit...
pause > nul