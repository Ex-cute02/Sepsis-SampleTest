#!/usr/bin/env python3
"""
Test feature importance functionality
"""

import requests
import json

def test_feature_importance():
    """Test the feature importance endpoint"""
    
    print("🔍 Testing Feature Importance Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8000/feature_importance", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Feature importance endpoint working")
            print(f"   Model type: {data.get('model_type', 'Unknown')}")
            print(f"   Total features: {data.get('total_features', 0)}")
            
            feature_importance = data.get('feature_importance', [])
            print(f"   Features returned: {len(feature_importance)}")
            
            if feature_importance:
                print("\n🎯 Top 10 Most Important Features:")
                for i, item in enumerate(feature_importance[:10]):
                    feature = item.get('feature', 'Unknown')
                    importance = item.get('importance', 0)
                    percentage = importance * 100
                    print(f"  {i+1:2d}. {feature:20s}: {importance:.6f} ({percentage:5.2f}%)")
                
                print(f"\n📊 Feature Importance Statistics:")
                importances = [item.get('importance', 0) for item in feature_importance]
                print(f"   Max importance: {max(importances):.6f}")
                print(f"   Min importance: {min(importances):.6f}")
                print(f"   Sum of all: {sum(importances):.6f}")
                
                # Check for meaningful data
                non_zero = [imp for imp in importances if imp > 0.001]
                print(f"   Features > 0.1%: {len(non_zero)}")
                
                return True
            else:
                print("❌ No feature importance data returned")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_feature_importance()
    print("\n" + "=" * 50)
    if success:
        print("✅ Feature importance functionality working!")
        print("💡 The frontend should now display the feature importance graph.")
    else:
        print("❌ Feature importance needs attention.")