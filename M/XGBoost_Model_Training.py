# Step 3: Model Training with XGBoost (our recommended algorithm)
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

print("=== STEP 3: MODEL TRAINING WITH XGBOOST ===\n")

# Load preprocessed data
X_train_scaled = np.load('X_train_scaled.npy')
X_test_scaled = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
feature_names = joblib.load('feature_names.pkl')

print("Loaded preprocessed data:")
print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Train XGBoost model (our recommended algorithm from the tech stack)
print("\nTraining XGBoost model...")

# XGBoost parameters optimized for medical/clinical data
xgb_model = xgb.XGBClassifier(
    n_estimators=100,           # Number of trees
    max_depth=6,                # Maximum tree depth
    learning_rate=0.1,          # Learning rate
    subsample=0.8,              # Sample ratio
    colsample_bytree=0.8,       # Feature ratio
    random_state=42,            # Reproducibility
    eval_metric='logloss',      # Evaluation metric
    objective='binary:logistic' # Binary classification
)

# Fit the model
xgb_model.fit(X_train_scaled, y_train)

print("XGBoost model training completed!")

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n=== MODEL PERFORMANCE METRICS ===")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"AUC-ROC:   {auc:.3f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Deceased', 'Survived']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Format: [[TN, FP], [FN, TP]]")

# Feature importance
feature_importance = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance_df)

# Save the trained model
joblib.dump(xgb_model, 'xgboost_sepsis_model.pkl')
print("\nModel saved as: xgboost_sepsis_model.pkl")

# Save performance metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision, 
    'recall': recall,
    'f1_score': f1,
    'auc_roc': auc
}
joblib.dump(metrics, 'model_metrics.pkl')
print("Metrics saved as: model_metrics.pkl")