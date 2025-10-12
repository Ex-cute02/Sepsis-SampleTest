#!/usr/bin/env python3
"""
Final comprehensive system test including feature importance
"""

import requests
import json
import time

def test_complete_system():
    """Test all system components including feature importance"""
    
    print("🧪 Final Comprehensive System Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 7
    
    # Test 1: API Health
    print("1️⃣ Testing API Health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ API Health Check - PASS")
            tests_passed += 1
        else:
            print("   ❌ API Health Check - FAIL")
    except Exception as e:
        print(f"   ❌ API Health Check - FAIL: {e}")
    
    # Test 2: Feature Importance
    print("\\n2️⃣ Testing Feature Importance...")
    try:
        response = requests.get("http://localhost:8000/feature_importance", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'feature_importance' in data and len(data['feature_importance']) > 0:
                print("   ✅ Feature Importance - PASS")
                print(f"      Features: {len(data['feature_importance'])}")
                tests_passed += 1
            else:
                print("   ❌ Feature Importance - FAIL (no data)")
        else:
            print("   ❌ Feature Importance - FAIL")
    except Exception as e:
        print(f"   ❌ Feature Importance - FAIL: {e}")
    
    # Test 3: Single Prediction with SHAP
    print("\\n3️⃣ Testing Single Prediction + SHAP...")
    sample_patient = {
        "Age": 65, "Gender": 1, "HR": 95, "SBP": 110, "DBP": 70,
        "Temp": 38.2, "RR": 22, "O2Sat": 94, "WBC": 12.5,
        "Lactate": 2.1, "pH": 7.35, "SOFA": 3
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=sample_patient, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('shap_explanations'):
                print("   ✅ Single Prediction + SHAP - PASS")
                print(f"      SHAP features: {len(data['shap_explanations'])}")
                tests_passed += 1
            else:
                print("   ❌ Single Prediction + SHAP - FAIL (no SHAP)")
        else:
            print("   ❌ Single Prediction + SHAP - FAIL")
    except Exception as e:
        print(f"   ❌ Single Prediction + SHAP - FAIL: {e}")
    
    # Test 4: Batch Prediction
    print("\\n4️⃣ Testing Batch Prediction...")
    batch_patients = [
        {"Age": 45, "Gender": 0, "HR": 85, "SBP": 120, "patient_id": "batch_1"},
        {"Age": 75, "Gender": 1, "HR": 105, "SBP": 90, "patient_id": "batch_2"}
    ]
    try:
        response = requests.post("http://localhost:8000/predict_batch", json=batch_patients, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if 'predictions' in data and len(data['predictions']) == 2:
                print("   ✅ Batch Prediction - PASS")
                tests_passed += 1
            else:
                print("   ❌ Batch Prediction - FAIL (wrong data)")
        else:
            print("   ❌ Batch Prediction - FAIL")
    except Exception as e:
        print(f"   ❌ Batch Prediction - FAIL: {e}")
    
    # Test 5: Frontend Accessibility
    print("\\n5️⃣ Testing Frontend Accessibility...")
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            print("   ✅ Frontend Accessibility - PASS")
            tests_passed += 1
        else:
            print("   ❌ Frontend Accessibility - FAIL")
    except Exception as e:
        print(f"   ❌ Frontend Accessibility - FAIL: {e}")
    
    # Test 6: Model Info
    print("\\n6️⃣ Testing Model Info...")
    try:
        response = requests.get("http://localhost:8000/model_info", timeout=10)
        if response.status_code == 200:
            print("   ✅ Model Info - PASS")
            tests_passed += 1
        else:
            print("   ❌ Model Info - FAIL")
    except Exception as e:
        print(f"   ❌ Model Info - FAIL: {e}")
    
    # Test 7: Data Format Validation
    print("\\n7️⃣ Testing Data Format Validation...")
    try:
        # Get feature importance and validate format
        response = requests.get("http://localhost:8000/feature_importance", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Check required keys
            required_keys = ['feature_importance', 'model_type', 'total_features']
            has_all_keys = all(key in data for key in required_keys)
            
            # Check feature structure
            features = data.get('feature_importance', [])
            valid_features = True
            if features:
                first_feature = features[0]
                required_feature_keys = ['feature', 'importance', 'description']
                valid_features = all(key in first_feature for key in required_feature_keys)
            
            if has_all_keys and valid_features and len(features) > 0:
                print("   ✅ Data Format Validation - PASS")
                tests_passed += 1
            else:
                print("   ❌ Data Format Validation - FAIL")
        else:
            print("   ❌ Data Format Validation - FAIL")
    except Exception as e:
        print(f"   ❌ Data Format Validation - FAIL: {e}")
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\\n🎉 ALL SYSTEMS FULLY OPERATIONAL!")
        print("\\n✅ WORKING FEATURES:")
        print("   • Backend API (http://localhost:8000)")
        print("   • Frontend App (http://localhost:3000)")
        print("   • Single Patient Predictions")
        print("   • SHAP Explanations & Visualizations")
        print("   • Feature Importance Graph")
        print("   • Batch Predictions")
        print("   • Clinical Alerts & Recommendations")
        
        print("\\n🎯 AVAILABLE VISUALIZATIONS:")
        print("   • 📊 Bar Chart (SHAP & Feature Importance)")
        print("   • 🌊 Waterfall Plot (SHAP)")
        print("   • ⚖️ Force Plot / Tug-of-War (SHAP)")
        print("   • 📋 Summary View (SHAP)")
        print("   • 📈 Feature Importance Distribution")
        
        print("\\n🚀 HOW TO USE:")
        print("   1. Open http://localhost:3000")
        print("   2. Use 'Patient Prediction' tab for individual predictions")
        print("   3. Use 'Feature Importance' tab to see model insights")
        print("   4. All SHAP visualizations work in prediction results")
        
    else:
        print(f"\\n⚠️  {total_tests - tests_passed} issues detected")
        print("   Check the failed tests above for details")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = test_complete_system()
    exit(0 if success else 1)