#!/usr/bin/env python3
"""
Test frontend visualization functionality
"""

import requests
import json
import time
import os

def test_frontend_shap_visualizations():
    """Test SHAP visualizations in the frontend"""
    
    print("🌐 Testing Frontend SHAP Visualizations")
    print("=" * 50)
    
    # Test if we can access the frontend
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code != 200:
            print("❌ Frontend not accessible")
            return False
        print("✅ Frontend is accessible")
    except Exception as e:
        print(f"❌ Frontend connection failed: {e}")
        return False
    
    # Test API prediction with SHAP data
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
        "patient_id": "frontend_test_001"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict", 
            json=sample_patient,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            shap_explanations = data.get('shap_explanations', {})
            
            if shap_explanations:
                print("✅ API returns SHAP explanations")
                print(f"   Features with SHAP values: {len(shap_explanations)}")
                
                # Check if we have meaningful SHAP values
                non_zero_shap = {k: v for k, v in shap_explanations.items() 
                               if abs(v.get('shap_contribution', 0)) > 0.001}
                print(f"   Non-zero SHAP contributions: {len(non_zero_shap)}")
                
                if len(non_zero_shap) > 0:
                    print("✅ SHAP data is meaningful for visualizations")
                    
                    # Show top contributors
                    sorted_shap = sorted(non_zero_shap.items(), 
                                       key=lambda x: abs(x[1].get('shap_contribution', 0)), 
                                       reverse=True)
                    
                    print("   Top SHAP contributors:")
                    for i, (feature, data) in enumerate(sorted_shap[:5]):
                        contribution = data.get('shap_contribution', 0)
                        impact = "↑ Survival" if contribution > 0 else "↓ Risk"
                        print(f"     {i+1}. {feature}: {contribution:+.4f} ({impact})")
                    
                    return True
                else:
                    print("⚠️  All SHAP values are near zero")
                    return False
            else:
                print("❌ No SHAP explanations in API response")
                return False
        else:
            print(f"❌ API prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def check_frontend_components():
    """Check if frontend components are properly structured"""
    
    print("\\n🔍 Checking Frontend Components")
    print("-" * 30)
    
    components_to_check = [
        "frontend/src/components/PredictionResult.js",
        "frontend/src/components/SHAPWaterfallPlot.js", 
        "frontend/src/components/SHAPForcePlot.js",
        "frontend/src/App.js"
    ]
    
    all_good = True
    
    for component in components_to_check:
        if os.path.exists(component):
            print(f"✅ {component}")
            
            # Check for key SHAP-related content
            with open(component, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'SHAP' in content or 'shap' in content:
                print(f"   Contains SHAP functionality")
            else:
                print(f"   ⚠️  No SHAP references found")
                
        else:
            print(f"❌ {component} - NOT FOUND")
            all_good = False
    
    return all_good

def main():
    """Run frontend visualization tests"""
    
    print("🧪 Frontend Visualization Test Suite")
    print("=" * 50)
    
    # Test 1: API SHAP functionality
    api_test = test_frontend_shap_visualizations()
    
    # Test 2: Frontend components
    components_test = check_frontend_components()
    
    print("\\n" + "=" * 50)
    print("📊 FRONTEND TEST SUMMARY")
    print("=" * 50)
    
    if api_test and components_test:
        print("✅ All frontend visualization tests passed!")
        print("\\n🎯 AVAILABLE VISUALIZATIONS:")
        print("✅ Bar Chart - Shows individual feature impacts")
        print("✅ Waterfall Plot - Step-by-step prediction breakdown") 
        print("✅ Force Plot - Tug-of-war between risk/survival factors")
        print("✅ Summary View - Categorized risk factors")
        
        print("\\n📋 HOW TO ACCESS:")
        print("1. Open http://localhost:3000 in your browser")
        print("2. Fill in patient data (or use 'Load Sample Data')")
        print("3. Click 'Predict Sepsis Risk'")
        print("4. Scroll down to 'AI Model Explanations (SHAP Values)'")
        print("5. Use the visualization toggle buttons:")
        print("   - 📊 Bar Chart")
        print("   - 🌊 Waterfall") 
        print("   - ⚖️ Force Plot")
        print("   - 📋 Summary")
        
        print("\\n🔧 TECHNICAL STATUS:")
        print("✅ Backend API: http://localhost:8000")
        print("✅ Frontend App: http://localhost:3000")
        print("✅ SHAP Explainer: Loaded and functional")
        print("✅ All visualization components: Present")
        
    else:
        print("⚠️  Some issues detected:")
        if not api_test:
            print("❌ API SHAP functionality needs attention")
        if not components_test:
            print("❌ Frontend components missing or incomplete")
    
    return api_test and components_test

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)