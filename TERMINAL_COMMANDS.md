# 🚀 Enhanced Sepsis System - Terminal Commands

## Quick Start Commands

### **Option 1: Complete System (Recommended)**
```bash
# Windows
start_full_system.bat

# Manual (2 separate terminals)
# Terminal 1 - Backend:
cd M
python Production_Sepsis_API.py

# Terminal 2 - Frontend:
cd frontend
npm start
```

### **Option 2: Individual Components**

#### **Backend Only:**
```bash
# Windows
start_backend_only.bat

# Manual:
cd M
python Production_Sepsis_API.py
```

#### **Frontend Only:**
```bash
# Windows  
start_frontend_only.bat

# Manual:
cd frontend
npm install  # First time only
npm start
```

## 🔧 **Manual Setup Commands**

### **First Time Setup:**
```bash
# Install frontend dependencies
cd frontend
npm install

# Verify Python dependencies
pip install fastapi uvicorn pandas numpy scikit-learn joblib pydantic

# Check if model files exist
dir M\*.pkl  # Windows
ls M/*.pkl   # Linux/Mac
```

### **Development Commands:**
```bash
# Start backend with auto-reload
cd M
uvicorn Production_Sepsis_API:app --reload --host 0.0.0.0 --port 8000

# Start frontend with hot reload
cd frontend
npm start

# Build frontend for production
cd frontend
npm run build
```

## 🌐 **Access URLs**

After starting the system:

- **Enhanced Frontend**: http://localhost:3000
- **Enhanced API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **Model Information**: http://localhost:8000/model_info

## 🧪 **Testing Commands**

### **Test API Endpoints:**
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model_info

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 65, "Gender": 1, "HR": 95, "SBP": 110, "Temp": 38.2}'
```

### **Frontend Testing:**
```bash
# Run tests
cd frontend
npm test

# Check for linting issues
cd frontend
npm run lint  # If available
```

## 🔍 **Troubleshooting Commands**

### **Check System Status:**
```bash
# Check if ports are in use
netstat -an | findstr :8000  # Backend port
netstat -an | findstr :3000  # Frontend port

# Check Python version
python --version

# Check Node.js version
node --version
npm --version
```

### **Debug Backend:**
```bash
cd M
python -c "import fastapi, uvicorn, pandas, numpy, sklearn, joblib; print('All dependencies OK')"

# Run with verbose logging
cd M
python Production_Sepsis_API.py --log-level debug
```

### **Debug Frontend:**
```bash
cd frontend
npm run build  # Check for build errors
npm audit      # Check for security issues
```

## 🚀 **Production Commands**

### **Build for Production:**
```bash
# Build frontend
cd frontend
npm run build

# Serve built frontend
npx serve -s build -l 3000

# Run backend in production mode
cd M
uvicorn Production_Sepsis_API:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📊 **System Verification**

### **Verify Everything is Working:**
```bash
# 1. Check backend health
curl http://localhost:8000/health

# 2. Check model validation
curl http://localhost:8000/validate_model

# 3. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 68, "Gender": 1, "HR": 105, "SBP": 88, "Temp": 38.7, "WBC": 16.2, "Lactate": 3.8}'

# 4. Open frontend in browser
start http://localhost:3000  # Windows
open http://localhost:3000   # Mac
xdg-open http://localhost:3000  # Linux
```

## ⚡ **Quick Commands Summary**

```bash
# Complete system (easiest)
start_full_system.bat

# Individual components
start_backend_only.bat
start_frontend_only.bat

# Manual (2 terminals)
cd M && python Production_Sepsis_API.py
cd frontend && npm start

# Test everything is working
curl http://localhost:8000/health && start http://localhost:3000
```

## 🎯 **Success Indicators**

**Backend Ready:**
```
Enhanced Backend API starting at: http://localhost:8000
INFO: Started server process
Enhanced preprocessing components loaded
INFO: Application startup complete.
```

**Frontend Ready:**
```
webpack compiled successfully
Local: http://localhost:3000
On Your Network: http://192.168.x.x:3000
```

**System Working:**
- ✅ Backend API docs accessible at http://localhost:8000/docs
- ✅ Frontend dashboard accessible at http://localhost:3000
- ✅ Health check returns "healthy" status
- ✅ Predictions work with enhanced clinical alerts

The **simplest approach** is to just run `start_full_system.bat` which handles everything automatically! 🚀