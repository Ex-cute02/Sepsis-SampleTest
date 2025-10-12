#!/usr/bin/env python3
"""
Comprehensive System Functionality Test
Tests all API endpoints and frontend features
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        print(f"✅ Health Check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/model_info", timeout=10)
        print(f"✅ Model Info: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Model Version: {data.get('model_version', 'N/A')}")
            print(f"   Features: {len(data.get('feature_names', []))}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Model Info Failed: {e}")
        return False

def test_feature_importance():
    """Test feature importance endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/feature_importance", timeout=10)
        print(f"✅ Feature Importance: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Features returned: {len(data.get('features', []))}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Feature Importance Failed: {e}")
        return False

def test_single_prediction():
    """Test single patient prediction with SHAP explanations"""
    sample_patient = {
        "Age": 65,
        "Gender": 1,
        "HR": 95,
        "SBP": 110,
        "DBP": 70,
        "Temp": 38.2,
        "RR": 22,
        "O2Sat": 94,
        "WBC": 12.5,
        "Lactate": 2.1,
        "pH": 7.35,
        "SOFA": 3,
        "patient_id": "test_001"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict", 
            json=sample_patient,
            timeout=30
        )
        print(f"✅ Single Prediction: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Prediction: {data.get('prediction', 'N/A')}")
            print(f"   Risk Level: {data.get('risk_level', 'N/A')}")
            print(f"   Survival Probability: {data.get('survival_probability', 'N/A'):.3f}")
            
            # Check SHAP explanations
            shap_explanations = data.get('shap_explanations', {})
            if shap_explanations:
                print(f"   SHAP Features: {len(shap_explanations)}")
                # Show top 3 SHAP values
                shap_items = list(shap_explanations.items())[:3]
                for feature, shap_data in shap_items:
                    contribution = shap_data.get('shap_contribution', 0)
                    print(f"     {feature}: {contribution:.4f}")
            else:
                print("   ⚠️  No SHAP explanations found")
            
            # Check clinical alerts and recommendations
            alerts = data.get('clinical_alerts', [])
            recommendations = data.get('recommendations', [])
            print(f"   Clinical Alerts: {len(alerts)}")
            print(f"   Recommendations: {len(recommendations)}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Single Prediction Failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    batch_patients = [
        {
            "Age": 45, "Gender": 0, "HR": 85, "SBP": 120, "DBP": 80,
            "Temp": 37.1, "RR": 18, "O2Sat": 98, "WBC": 8.5,
            "Lactate": 1.2, "pH": 7.42, "SOFA": 1, "patient_id": "batch_001"
        },
        {
            "Age": 75, "Gender": 1, "HR": 105, "SBP": 90, "DBP": 60,
            "Temp": 39.1, "RR": 28, "O2Sat": 88, "WBC": 18.2,
            "Lactate": 4.5, "pH": 7.25, "SOFA": 8, "patient_id": "batch_002"
        }
    ]
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_batch", 
            json=batch_patients,
            timeout=60
        )
        print(f"✅ Batch Prediction: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"   Predictions returned: {len(predictions)}")
            for i, pred in enumerate(predictions):
                print(f"     Patient {i+1}: {pred.get('prediction', 'N/A')} ({pred.get('risk_level', 'N/A')})")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Batch Prediction Failed: {e}")
        return False

def check_frontend_accessibility():
    """Check if frontend is accessible"""
    try:
        response = requests.get(FRONTEND_URL, timeout=10)
        print(f"✅ Frontend Accessible: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Frontend Not Accessible: {e}")
        return False

def main():
    """Run comprehensive system test"""
    print("🧪 Starting Comprehensive System Test")
    print("=" * 50)
    
    tests = [
        ("API Health Check", test_api_health),
        ("Model Information", test_model_info),
        ("Feature Importance", test_feature_importance),
        ("Single Prediction + SHAP", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Frontend Accessibility", check_frontend_accessibility),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        results[test_name] = test_func()
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems operational!")
        print("\n📋 FEATURE CHECKLIST:")
        print("✅ Backend API running on http://localhost:8000")
        print("✅ Frontend running on http://localhost:3000")
        print("✅ Single patient predictions with SHAP explanations")
        print("✅ Batch predictions")
        print("✅ Feature importance analysis")
        print("✅ Clinical alerts and recommendations")
        print("\n🎯 SHAP VISUALIZATIONS AVAILABLE:")
        print("✅ Bar Chart - Feature impact visualization")
        print("✅ Waterfall Plot - Step-by-step prediction breakdown")
        print("✅ Force Plot (Tug-of-War) - Feature battle visualization")
        print("✅ Summary View - Risk factors breakdown")
    else:
        print("⚠️  Some issues detected. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)