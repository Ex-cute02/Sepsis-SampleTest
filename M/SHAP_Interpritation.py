# Step 4: Model Interpretability with SHAP
import shap
import matplotlib.pyplot as plt

print("=== STEP 4: MODEL INTERPRETABILITY WITH SHAP ===\n")

# Load the trained model
xgb_model = joblib.load('xgboost_sepsis_model.pkl')
X_test_scaled = np.load('X_test_scaled.npy')
feature_names = joblib.load('feature_names.pkl')

print("Adding SHAP interpretability to our XGBoost model...")

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

print(f"SHAP values computed for {X_test_scaled.shape[0]} test samples")
print(f"SHAP values shape: {shap_values.shape}")

# Calculate mean absolute SHAP values for feature importance
mean_shap_values = np.abs(shap_values).mean(0)
shap_importance_df = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': mean_shap_values
}).sort_values('shap_importance', ascending=False)

print("\n=== SHAP FEATURE IMPORTANCE ===")
print(shap_importance_df)

# Save SHAP values and explainer
np.save('shap_values.npy', shap_values)
joblib.dump(explainer, 'shap_explainer.pkl')

print("\nSHAP analysis completed!")
print("Files saved:")
print("- shap_values.npy")
print("- shap_explainer.pkl")

# Example: Explain a single prediction
print("\n=== EXAMPLE: INDIVIDUAL PREDICTION EXPLANATION ===")
sample_idx = 0
sample_features = X_test_scaled[sample_idx:sample_idx+1]
sample_shap = shap_values[sample_idx]

prediction_proba = xgb_model.predict_proba(sample_features)[0]
prediction = xgb_model.predict(sample_features)[0]

print(f"Sample {sample_idx} prediction:")
print(f"- Predicted class: {'Survived' if prediction == 1 else 'Deceased'}")
print(f"- Survival probability: {prediction_proba[1]:.3f}")
print(f"- Mortality probability: {prediction_proba[0]:.3f}")

print(f"\nFeature contributions (SHAP values):")
for i, (feature, shap_val) in enumerate(zip(feature_names, sample_shap)):
    feature_value = sample_features[0, i]
    direction = "↑ Survival" if shap_val > 0 else "↓ Survival"
    print(f"- {feature:15}: {feature_value:6.2f} → {shap_val:+6.3f} ({direction})")

# Summary of model interpretability
print(f"\n=== MODEL INTERPRETABILITY SUMMARY ===")
print("✓ XGBoost model trained with clinical sepsis data")
print("✓ SHAP explainer created for feature importance analysis")
print("✓ Individual prediction explanations available")
print("✓ Feature contributions quantified for clinical decision support")
print("\nThis interpretable model can help clinicians understand:")
print("- Which features most influence sepsis survival predictions")
print("- How each patient's specific values contribute to their risk assessment")
print("- Clinical reasoning behind AI-generated alerts and recommendations")