#!/usr/bin/env python3
"""
Test feature importance frontend integration
"""

import requests
import json
import time

def test_feature_importance_integration():
    """Test the complete feature importance flow"""
    
    print("🔍 Testing Feature Importance Integration")
    print("=" * 60)
    
    # Test 1: API Endpoint
    print("1️⃣ Testing API Endpoint...")
    try:
        response = requests.get("http://localhost:8000/feature_importance", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ API endpoint working")
            
            # Validate structure
            required_keys = ['feature_importance', 'model_type', 'total_features']
            for key in required_keys:
                if key not in data:
                    print(f"   ❌ Missing key: {key}")
                    return False
                else:
                    print(f"   ✅ Has key: {key}")
            
            feature_importance = data['feature_importance']
            if not isinstance(feature_importance, list):
                print("   ❌ feature_importance is not a list")
                return False
            
            if len(feature_importance) == 0:
                print("   ❌ feature_importance is empty")
                return False
            
            print(f"   ✅ {len(feature_importance)} features returned")
            
            # Check feature structure
            first_feature = feature_importance[0]
            required_feature_keys = ['feature', 'importance', 'description']
            for key in required_feature_keys:
                if key not in first_feature:
                    print(f"   ❌ Feature missing key: {key}")
                    return False
            
            print("   ✅ Feature structure is correct")
            
        else:
            print(f"   ❌ API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ API Test failed: {e}")
        return False
    
    # Test 2: Frontend Accessibility
    print("\\n2️⃣ Testing Frontend Accessibility...")
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            print("   ✅ Frontend accessible")
        else:
            print(f"   ❌ Frontend error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Frontend test failed: {e}")
        return False
    
    # Test 3: Data Format Compatibility
    print("\\n3️⃣ Testing Data Format Compatibility...")
    
    # Simulate frontend data transformation
    try:
        api_data = data['feature_importance']
        
        # This is what the frontend does
        chart_data = []
        for item in api_data:
            transformed_item = {
                'feature': item['feature'].replace('_', ' ').title(),
                'originalFeature': item['feature'],
                'importance': item['importance'],
                'percentage': f"{item['importance'] * 100:.1f}",
                'description': item['description']
            }
            chart_data.append(transformed_item)
        
        # Sort by importance (what frontend does)
        chart_data.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"   ✅ Successfully transformed {len(chart_data)} features")
        print("   ✅ Data sorting works")
        
        # Check for reasonable values
        max_importance = max(item['importance'] for item in chart_data)
        min_importance = min(item['importance'] for item in chart_data)
        
        if max_importance > 1.0:
            print(f"   ⚠️  Warning: Max importance > 1.0: {max_importance}")
        
        if min_importance < 0:
            print(f"   ⚠️  Warning: Negative importance: {min_importance}")
        
        print(f"   ✅ Importance range: {min_importance:.6f} to {max_importance:.6f}")
        
    except Exception as e:
        print(f"   ❌ Data transformation failed: {e}")
        return False
    
    # Test 4: Show Sample Data
    print("\\n4️⃣ Sample Feature Importance Data:")
    print("   " + "-" * 50)
    for i, item in enumerate(chart_data[:5]):
        feature = item['feature']
        importance = item['importance']
        percentage = item['percentage']
        print(f"   {i+1}. {feature:20s}: {importance:.6f} ({percentage}%)")
    
    print("\\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("\\n📋 Integration Status:")
    print("✅ API endpoint working correctly")
    print("✅ Frontend accessible")
    print("✅ Data format compatible")
    print("✅ Feature transformation working")
    print("\\n💡 The feature importance graph should now work in the frontend.")
    print("   Navigate to: http://localhost:3000")
    print("   Click on: 'Feature Importance' tab")
    
    return True

if __name__ == "__main__":
    success = test_feature_importance_integration()
    if not success:
        print("\\n❌ Some tests failed. Check the errors above.")
        exit(1)