#!/usr/bin/env python3
"""
Test SHAP functionality specifically
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_shap_detailed():
    """Test SHAP explanations in detail"""
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
        "patient_id": "shap_test_001"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict", 
            json=sample_patient,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("🔍 Full API Response:")
            print(json.dumps(data, indent=2))
            
            print("\n📊 SHAP Analysis:")
            shap_explanations = data.get('shap_explanations')
            if shap_explanations:
                print(f"✅ SHAP explanations found: {len(shap_explanations)} features")
                
                # Sort by absolute SHAP contribution
                shap_items = [(k, v) for k, v in shap_explanations.items()]
                shap_items.sort(key=lambda x: abs(x[1].get('shap_contribution', 0)), reverse=True)
                
                print("\n🎯 Top SHAP Contributors:")
                for i, (feature, data) in enumerate(shap_items[:10]):
                    contribution = data.get('shap_contribution', 0)
                    value = data.get('value', 0)
                    impact = data.get('impact', 'unknown')
                    print(f"  {i+1:2d}. {feature:15s}: {contribution:+8.4f} (value: {value:6.2f}) - {impact}")
                
                return True
            else:
                print("❌ No SHAP explanations found")
                print("🔍 Available keys in response:")
                for key in data.keys():
                    print(f"  - {key}: {type(data[key])}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing SHAP Functionality")
    print("=" * 50)
    success = test_shap_detailed()
    print("\n" + "=" * 50)
    if success:
        print("✅ SHAP functionality working!")
    else:
        print("❌ SHAP functionality needs attention")