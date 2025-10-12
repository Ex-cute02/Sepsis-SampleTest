#!/usr/bin/env python3
"""
Fix SHAP explainer with proper feature alignment
"""

import joblib
import numpy as np
import pandas as pd
import shap
import os

def regenerate_shap_explainer_v2():
    """Regenerate SHAP explainer with proper feature alignment"""
    
    # Change to M directory where model files are located
    os.chdir('M')
    
    print("🔧 Regenerating SHAP explainer (v2)...")
    
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
                print(f"   Model expects {model.n_features_in_} features")
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
    
    # Create synthetic background data with correct dimensions
    try:
        print("🔄 Creating synthetic background data...")
        n_samples = 100
        n_features = model.n_features_in_
        
        # Create realistic synthetic data based on typical patient values
        np.random.seed(42)  # For reproducibility
        
        # Generate synthetic patient data
        synthetic_data = []
        for _ in range(n_samples):
            patient = {
                'Age': np.random.normal(65, 15),
                'Gender': np.random.choice([0, 1]),
                'HR': np.random.normal(85, 20),
                'SBP': np.random.normal(120, 25),
                'DBP': np.random.normal(75, 15),
                'Temp': np.random.normal(37.5, 1.5),
                'RR': np.random.normal(18, 5),
                'O2Sat': np.random.normal(96, 4),
                'WBC': np.random.normal(10, 5),
                'Lactate': np.random.normal(1.5, 1),
                'pH': np.random.normal(7.4, 0.1),
                'SOFA': np.random.randint(0, 10),
            }
            
            # Create feature vector matching the model's expectations
            feature_vector = []
            for feature_name in feature_names:
                if feature_name in patient:
                    value = patient[feature_name]
                else:
                    # Handle engineered features with reasonable defaults
                    if 'MAP' in feature_name:
                        value = patient.get('DBP', 75) + (patient.get('SBP', 120) - patient.get('DBP', 75)) / 3
                    elif 'BMI' in feature_name:
                        value = 25.0  # Average BMI
                    elif 'Pulse_Pressure' in feature_name:
                        value = patient.get('SBP', 120) - patient.get('DBP', 75)
                    elif 'Age_Risk' in feature_name:
                        value = 1 if patient.get('Age', 65) > 65 else 0
                    elif 'Temp_Risk' in feature_name:
                        value = 1 if patient.get('Temp', 37) > 38.3 else 0
                    elif 'SIRS' in feature_name:
                        value = np.random.randint(0, 4)
                    else:
                        value = 0  # Default for unknown features
                
                feature_vector.append(value)
            
            synthetic_data.append(feature_vector)
        
        X_background = np.array(synthetic_data)
        print(f"✅ Created synthetic background: {X_background.shape}")
        
        # Scale the background data
        X_background_scaled = scaler.transform(X_background)
        print(f"✅ Scaled background data: {X_background_scaled.shape}")
        
    except Exception as e:
        print(f"❌ Error creating background data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create SHAP explainer with additivity check disabled
    try:
        print("🔄 Creating SHAP TreeExplainer (with additivity check disabled)...")
        explainer = shap.TreeExplainer(model, X_background_scaled, check_additivity=False)
        print("✅ SHAP TreeExplainer created successfully")
        
        # Test the explainer with a sample
        print("🧪 Testing SHAP explainer...")
        test_sample = X_background_scaled[:1]  # Use first sample as test
        shap_values = explainer.shap_values(test_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, use positive class
        
        print(f"✅ SHAP test successful: {shap_values.shape}")
        print(f"   Sample SHAP values: {shap_values[0][:5]}...")
        print(f"   SHAP values range: [{shap_values.min():.4f}, {shap_values.max():.4f}]")
        
        # Save the explainer
        explainer_filename = f'{model_type}_shap_explainer.pkl'
        joblib.dump(explainer, explainer_filename)
        print(f"✅ SHAP explainer saved as: {explainer_filename}")
        
        # Also save as default name
        joblib.dump(explainer, 'shap_explainer.pkl')
        print("✅ SHAP explainer saved as: shap_explainer.pkl")
        
        # Save feature names for reference
        with open('shap_feature_info.txt', 'w') as f:
            f.write(f"Model features: {model.n_features_in_}\\n")
            f.write(f"Feature names: {len(feature_names)}\\n")
            f.write("\\nFeature list:\\n")
            for i, name in enumerate(feature_names):
                f.write(f"{i:2d}: {name}\\n")
        
        print("✅ Feature info saved to: shap_feature_info.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating SHAP explainer: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 SHAP Explainer Fix Tool v2")
    print("=" * 40)
    
    success = regenerate_shap_explainer_v2()
    
    print("\\n" + "=" * 40)
    if success:
        print("✅ SHAP explainer regenerated successfully!")
        print("🔄 Please restart the API server to load the new explainer.")
    else:
        print("❌ Failed to regenerate SHAP explainer.")
        print("💡 Check the error messages above for details.")