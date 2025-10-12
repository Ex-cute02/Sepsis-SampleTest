#!/usr/bin/env python3
"""
Fix SHAP explainer for the current model
"""

import joblib
import numpy as np
import pandas as pd
import shap
import os

def regenerate_shap_explainer():
    """Regenerate SHAP explainer for the current model"""
    
    # Change to M directory where model files are located
    os.chdir('M')
    
    print("🔧 Regenerating SHAP explainer...")
    
    # Load the current model
    model_files = [
        'optimized_xgboost_sepsis_model.pkl',
        'enhanced_xgboost_sepsis_model.pkl',
        'xgboost_sepsis_model.pkl'
    ]
    
    model = None
    model_type = None
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                model_type = model_file.replace('_xgboost_sepsis_model.pkl', '')
                print(f"✅ Loaded model: {model_file}")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
                continue
    
    if model is None:
        print("❌ No model found!")
        return False
    
    # Load feature names
    feature_files = [
        f'{model_type}_feature_names.pkl',
        'feature_names.pkl'
    ]
    
    feature_names = None
    for feature_file in feature_files:
        if os.path.exists(feature_file):
            try:
                feature_names = joblib.load(feature_file)
                print(f"✅ Loaded feature names: {feature_file} ({len(feature_names)} features)")
                break
            except Exception as e:
                print(f"❌ Failed to load {feature_file}: {e}")
                continue
    
    if feature_names is None:
        print("❌ No feature names found!")
        return False
    
    # Load scaler
    scaler_files = [
        f'{model_type}_scaler.pkl',
        'scaler.pkl'
    ]
    
    scaler = None
    for scaler_file in scaler_files:
        if os.path.exists(scaler_file):
            try:
                scaler = joblib.load(scaler_file)
                print(f"✅ Loaded scaler: {scaler_file}")
                break
            except Exception as e:
                print(f"❌ Failed to load {scaler_file}: {e}")
                continue
    
    if scaler is None:
        print("❌ No scaler found!")
        return False
    
    # Load training data for SHAP background
    try:
        X_train_files = [
            'X_train_scaled.npy',
            'X_test_scaled.npy'  # Fallback
        ]
        
        X_background = None
        for X_file in X_train_files:
            if os.path.exists(X_file):
                X_background = np.load(X_file)
                print(f"✅ Loaded background data: {X_file} {X_background.shape}")
                break
        
        if X_background is None:
            print("⚠️  No training data found, creating synthetic background...")
            # Create synthetic background data
            n_samples = 100
            n_features = len(feature_names)
            X_background = np.random.normal(0, 1, (n_samples, n_features))
            print(f"✅ Created synthetic background: {X_background.shape}")
        
        # Use a subset for efficiency
        if X_background.shape[0] > 100:
            indices = np.random.choice(X_background.shape[0], 100, replace=False)
            X_background = X_background[indices]
            print(f"✅ Using background subset: {X_background.shape}")
        
    except Exception as e:
        print(f"❌ Error loading background data: {e}")
        return False
    
    # Create SHAP explainer
    try:
        print("🔄 Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model, X_background)
        print("✅ SHAP TreeExplainer created successfully")
        
        # Test the explainer with a sample
        print("🧪 Testing SHAP explainer...")
        test_sample = X_background[:1]  # Use first sample as test
        shap_values = explainer.shap_values(test_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, use positive class
        
        print(f"✅ SHAP test successful: {shap_values.shape}")
        print(f"   Sample SHAP values: {shap_values[0][:5]}...")
        
        # Save the explainer
        explainer_filename = f'{model_type}_shap_explainer.pkl'
        joblib.dump(explainer, explainer_filename)
        print(f"✅ SHAP explainer saved as: {explainer_filename}")
        
        # Also save as default name
        joblib.dump(explainer, 'shap_explainer.pkl')
        print("✅ SHAP explainer saved as: shap_explainer.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating SHAP explainer: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 SHAP Explainer Fix Tool")
    print("=" * 40)
    
    success = regenerate_shap_explainer()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ SHAP explainer regenerated successfully!")
        print("🔄 Please restart the API server to load the new explainer.")
    else:
        print("❌ Failed to regenerate SHAP explainer.")
        print("💡 Check the error messages above for details.")