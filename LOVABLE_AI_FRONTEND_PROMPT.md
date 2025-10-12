# 🏥 Sepsis Prediction System - Complete Frontend Design Prompt for Lovable AI

## 🎯 Project Overview

Create a **professional, modern, and intuitive medical AI dashboard** for a sepsis prediction system that helps healthcare professionals assess patient risk using machine learning and explainable AI (SHAP). The system should feel like a premium clinical decision support tool used in top-tier hospitals.

---

## 🎨 Design Requirements

### **Visual Style & Branding**
- **Theme:** Clean, medical-grade interface with a modern healthcare aesthetic
- **Color Palette:** 
  - Primary: Medical blue (#2563eb, #3b82f6)
  - Success/Low Risk: Green (#22c55e, #16a34a)
  - Warning/Moderate Risk: Amber (#f59e0b, #d97706)
  - Danger/High Risk: Red (#ef4444, #dc2626)
  - Critical: Deep red (#b91c1c, #991b1b)
  - Background: Light gray (#f8fafc, #f1f5f9)
  - Cards: White with subtle shadows
- **Typography:** Clean, readable fonts (Inter, Roboto, or similar)
- **Icons:** Medical and data visualization icons (Lucide React style)
- **Layout:** Responsive, mobile-first design with proper spacing

### **User Experience Principles**
- **Clinical Workflow Integration:** Intuitive for busy healthcare professionals
- **Data Clarity:** Clear presentation of complex medical data
- **Trust & Transparency:** Visible AI explanations and confidence levels
- **Accessibility:** WCAG 2.1 AA compliant
- **Performance:** Fast loading and responsive interactions

---

## 🏗️ Application Structure

### **Main Navigation Tabs**
1. **🩺 Patient Prediction** - Individual patient risk assessment
2. **📊 Feature Importance** - Global model insights
3. **📈 Batch Analysis** - Multiple patient processing (future)
4. **⚙️ Settings** - System configuration (future)

### **Header Component**
- **Logo/Title:** "Sepsis Prediction System" with medical cross icon
- **Status Indicators:** API connection, model status, last update
- **User Info:** Current user/session info (placeholder)
- **Quick Actions:** Help, settings, logout buttons

### **Footer Component**
- **Disclaimer:** Medical decision support tool notice
- **Credits:** "Built with XGBoost ML model and SHAP interpretability"
- **Version Info:** Model version, last updated
- **Links:** Documentation, support, privacy policy

---

## 📋 Core Features & Components

### **1. Patient Prediction Tab (Main Feature)**

#### **Patient Information Form**
```
Layout: 3-column responsive grid (mobile: 1 column)

Demographics Section:
- Age* (number input, 0-120, with validation)
- Gender* (dropdown: Female/Male)
- ICU Hours (number input, optional)
- Patient ID (text input, optional)
- Unit (text input, e.g., "ICU", optional)

Vital Signs Section (with medical icons):
- Heart Rate (bpm) - Heart icon
- Systolic BP (mmHg) - Activity icon  
- Diastolic BP (mmHg)
- MAP (mmHg)
- Temperature (°C) - Thermometer icon
- Respiratory Rate (/min) - Lungs icon
- O2 Saturation (%) - Droplet icon

Laboratory Values Section:
- WBC (K/μL) - Test tube icon
- Hemoglobin (g/dL)
- Hematocrit (%)
- Platelets (K/μL)
- Lactate (mmol/L)
- Creatinine (mg/dL)
- BUN (mg/dL)
- Glucose (mg/dL)

Expandable Advanced Section:
- Additional Labs: Bilirubin, AST, Alkaline Phos
- Electrolytes: Calcium, Chloride, Magnesium, Phosphate, Potassium
- Blood Gas: pH, PaCO2, Base Excess, HCO3
- Respiratory: FiO2

Form Features:
- Real-time validation with helpful error messages
- "Load Sample Data" button for testing
- Clear visual indicators for required fields
- Tooltips with normal ranges
- Auto-save draft functionality
```

#### **Prediction Results Display**
```
Risk Level Card (Large, Prominent):
- Color-coded risk level (Low/Moderate/High/Critical)
- Large percentage display (survival probability)
- Risk level icon and description
- Confidence score indicator

Probability Visualization:
- Interactive pie chart (survival vs mortality)
- Animated progress bars
- Color-coded percentages

Key Metrics Grid:
- Prediction outcome
- Risk score (1-4)
- Survival probability
- Mortality risk
- Processing time
- Model confidence

Clinical Alerts Section:
- Color-coded alert cards
- Icons for severity levels
- Expandable alert details
- Priority sorting

Treatment Recommendations:
- Evidence-based suggestions
- Risk-level specific protocols
- Actionable clinical steps
- Reference links (future)
```

#### **SHAP Explanations (AI Transparency)**
```
Visualization Toggle Bar:
- 📊 Bar Chart (default)
- 🌊 Waterfall Plot
- ⚖️ Force Plot (Tug-of-War)
- 📋 Summary View

Bar Chart View:
- Interactive bar chart showing feature impacts
- Color coding: Green (increases survival), Red (increases risk)
- Hover tooltips with patient values
- Sortable by impact magnitude
- Feature importance percentages

Waterfall Plot:
- Step-by-step prediction breakdown
- Base rate → individual contributions → final prediction
- Cumulative impact visualization
- Interactive tooltips
- Summary statistics

Force Plot (Tug-of-War):
- Visual battle between survival vs risk factors
- Dynamic gauge showing final prediction
- Separate panels for positive/negative forces
- Feature strength indicators
- Animated interactions

Summary View:
- Risk factors categorization
- Top survival factors list
- Top risk factors list
- Educational explanations
- Clinical interpretation guide
```

### **2. Feature Importance Tab**

#### **Global Model Insights**
```
Feature Importance Chart:
- Interactive bar chart of all 19 features
- Ranked by global importance to the model
- Hover tooltips with descriptions
- Responsive design for mobile

Summary Statistics Cards:
- Total features analyzed
- Top 5 combined importance
- Highest single feature impact
- Model type and version

Feature Details Grid:
- Individual feature cards
- Importance percentages
- Progress bars
- Clinical descriptions
- Normal ranges

Educational Content:
- "Understanding Feature Importance" guide
- Model transparency explanations
- Clinical relevance notes
- Comparison with SHAP values
```

### **3. System Status & Health**

#### **API Connection Monitoring**
```
Status Indicators:
- Backend API connection (green/red dot)
- Model loading status
- SHAP explainer status
- Last prediction timestamp

Performance Metrics:
- Average prediction time
- Total predictions made
- System uptime
- Error rate (if any)

Health Check Results:
- Model version information
- Feature count validation
- Memory usage (if available)
- Response time metrics
```

---

## 🎛️ Interactive Elements

### **Form Interactions**
- **Smart Validation:** Real-time validation with helpful messages
- **Auto-complete:** Common values and ranges
- **Keyboard Navigation:** Full keyboard accessibility
- **Mobile Optimization:** Touch-friendly inputs

### **Chart Interactions**
- **Hover Effects:** Detailed tooltips on all charts
- **Click Actions:** Drill-down capabilities where relevant
- **Zoom/Pan:** For detailed chart exploration
- **Export Options:** Save charts as images (future)

### **Loading States**
- **Skeleton Screens:** For chart loading
- **Progress Indicators:** For prediction processing
- **Animated Spinners:** For API calls
- **Success Animations:** For completed predictions

### **Error Handling**
- **Graceful Degradation:** Fallbacks for failed API calls
- **User-Friendly Messages:** Clear error explanations
- **Retry Mechanisms:** Easy retry buttons
- **Offline Support:** Basic functionality when offline

---

## 📱 Responsive Design

### **Desktop (1200px+)**
- 3-column form layout
- Side-by-side charts and results
- Full navigation visible
- Detailed tooltips and descriptions

### **Tablet (768px - 1199px)**
- 2-column form layout
- Stacked chart sections
- Collapsible navigation
- Optimized touch targets

### **Mobile (< 768px)**
- Single-column layout
- Swipeable chart tabs
- Bottom navigation
- Simplified interactions

---

## 🔧 Technical Implementation

### **Frontend Stack**
```javascript
// Core Framework
React 18+ with hooks
TypeScript for type safety

// Styling
Tailwind CSS for utility-first styling
Custom CSS for medical-specific components

// Charts & Visualizations
Recharts for interactive charts
D3.js for custom SHAP visualizations
Framer Motion for animations

// Icons & UI
Lucide React for medical icons
Headless UI for accessible components
React Hook Form for form management

// State Management
React Context for global state
React Query for API state management

// Utilities
Axios for API calls
Date-fns for date handling
Lodash for data manipulation
```

### **API Integration**
```javascript
// Endpoints to integrate
GET  /health              - System health check
POST /predict             - Single patient prediction
POST /predict_batch       - Multiple patients
GET  /feature_importance  - Global model insights
GET  /model_info         - Model metadata

// Response handling
- Loading states for all API calls
- Error boundaries for failed requests
- Retry logic with exponential backoff
- Response caching where appropriate
```

### **Data Flow**
```
User Input → Form Validation → API Call → Loading State → 
Results Processing → SHAP Calculation → Visualization Rendering → 
Interactive Display → Clinical Recommendations
```

---

## 🎨 Component Specifications

### **Color-Coded Risk Levels**
```css
Low Risk (0-10%):
- Background: bg-green-50
- Border: border-green-200
- Text: text-green-800
- Icon: CheckCircle (green)

Moderate Risk (10-30%):
- Background: bg-yellow-50
- Border: border-yellow-200
- Text: text-yellow-800
- Icon: AlertTriangle (yellow)

High Risk (30-60%):
- Background: bg-orange-50
- Border: border-orange-200
- Text: text-orange-800
- Icon: AlertTriangle (orange)

Critical Risk (60%+):
- Background: bg-red-50
- Border: border-red-200
- Text: text-red-800
- Icon: XCircle (red)
```

### **Medical Icons Mapping**
```javascript
Demographics: User, Calendar, MapPin
Vital Signs: Heart, Activity, Thermometer, Droplets
Laboratory: TestTube, Microscope, Beaker
Respiratory: Wind, Lungs, Gauge
Neurological: Brain, Zap, Eye
Alerts: AlertTriangle, AlertCircle, Bell
Actions: Play, Pause, RefreshCw, Download
Navigation: ChevronLeft, ChevronRight, Menu
Status: CheckCircle, XCircle, Clock, Wifi
```

### **Animation Guidelines**
```css
Micro-interactions:
- Button hover: 150ms ease-in-out
- Form focus: 200ms ease-in-out
- Chart transitions: 300ms ease-in-out

Loading animations:
- Skeleton screens: 1.5s pulse
- Spinners: 1s linear infinite
- Progress bars: smooth transitions

Success states:
- Checkmark animation: 500ms ease-out
- Result reveal: 400ms slide-up
- Chart drawing: 800ms ease-in-out
```

---

## 📊 Sample Data & Testing

### **Sample Patient Data**
```javascript
// High-risk patient example
{
  Age: 68,
  Gender: 1, // Male
  HR: 105,
  O2Sat: 92,
  Temp: 38.7,
  SBP: 88,
  MAP: 62,
  DBP: 55,
  Resp: 26,
  Glucose: 145,
  BUN: 28,
  Creatinine: 1.8,
  WBC: 16.2,
  Hct: 32,
  Hgb: 10.5,
  Platelets: 95,
  Lactate: 3.8,
  pH: 7.32,
  HCO3: 18,
  ICULOS: 18,
  patient_id: "demo_patient_001",
  unit: "ICU"
}

// Low-risk patient example
{
  Age: 45,
  Gender: 0, // Female
  HR: 78,
  O2Sat: 98,
  Temp: 37.1,
  SBP: 120,
  MAP: 85,
  DBP: 75,
  Resp: 16,
  WBC: 7.2,
  Lactate: 1.1,
  pH: 7.42,
  patient_id: "demo_patient_002",
  unit: "General Ward"
}
```

### **Expected API Responses**
```javascript
// Prediction response with SHAP
{
  patient_id: "patient_123",
  timestamp: "2025-10-12T22:30:00Z",
  survival_probability: 0.75,
  mortality_probability: 0.25,
  risk_level: "moderate",
  risk_score: 2,
  prediction: "Low mortality risk",
  confidence: 0.85,
  shap_explanations: {
    HR: {
      value: 105,
      shap_contribution: 0.12,
      impact: "increases_mortality"
    },
    // ... 18 more features
  },
  clinical_alerts: [
    "Elevated heart rate detected",
    "Monitor for signs of sepsis"
  ],
  recommendations: [
    "Continue monitoring vital signs",
    "Consider blood cultures if fever persists"
  ],
  model_version: "optimized",
  processing_time_ms: 4.2
}

// Feature importance response
{
  feature_importance: [
    {
      feature: "Temp_missing",
      importance: 0.165,
      description: "Temperature data availability flag"
    },
    // ... 18 more features
  ],
  model_type: "XGBClassifier",
  total_features: 19
}
```

---

## 🚀 Implementation Priorities

### **Phase 1: Core Functionality**
1. ✅ Patient form with validation
2. ✅ Basic prediction display
3. ✅ SHAP bar chart visualization
4. ✅ Feature importance tab
5. ✅ Responsive layout

### **Phase 2: Enhanced Visualizations**
1. ✅ SHAP waterfall plot
2. ✅ SHAP force plot (tug-of-war)
3. ✅ Interactive tooltips
4. ✅ Animation improvements
5. ✅ Mobile optimization

### **Phase 3: Advanced Features**
1. 🔄 Batch prediction interface
2. 🔄 Historical patient tracking
3. 🔄 Export functionality
4. 🔄 Advanced settings
5. 🔄 User authentication

### **Phase 4: Clinical Integration**
1. 🔄 EHR integration hooks
2. 🔄 Clinical workflow optimization
3. 🔄 Audit logging
4. 🔄 Compliance features
5. 🔄 Advanced reporting

---

## 🎯 Success Metrics

### **User Experience**
- **Intuitive Navigation:** Healthcare professionals can use without training
- **Fast Performance:** < 2 second prediction results
- **Clear Visualizations:** SHAP explanations are easily understood
- **Mobile Friendly:** Full functionality on tablets and phones

### **Clinical Value**
- **Transparent AI:** Clear explanation of model decisions
- **Actionable Insights:** Specific clinical recommendations
- **Risk Stratification:** Clear risk level communication
- **Decision Support:** Enhances clinical judgment

### **Technical Excellence**
- **Reliability:** 99.9% uptime for predictions
- **Accuracy:** Consistent with backend model performance
- **Accessibility:** WCAG 2.1 AA compliance
- **Security:** HIPAA-ready architecture (future)

---

## 📝 Additional Notes

### **Medical Compliance Considerations**
- Include appropriate medical disclaimers
- Emphasize clinical decision support (not replacement)
- Provide clear confidence intervals
- Enable audit trails for predictions

### **Future Enhancements**
- Real-time patient monitoring integration
- Multi-language support for international use
- Advanced analytics dashboard
- Machine learning model comparison tools

### **Development Guidelines**
- Follow React best practices and hooks patterns
- Implement comprehensive error boundaries
- Use TypeScript for type safety
- Include unit tests for critical components
- Document all medical terminology and calculations

---

**🏆 Goal: Create a world-class medical AI interface that healthcare professionals trust and rely on for sepsis prediction and clinical decision support.**

This system should feel like it belongs in the most advanced hospitals and research institutions, providing transparency, reliability, and actionable insights that save lives.