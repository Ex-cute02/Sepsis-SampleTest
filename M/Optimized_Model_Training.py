#!/usr/bin/env python3
"""
Optimized Model Training for Real Sepsis Dataset
Advanced pipeline with feature engineering, class balancing, and hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== OPTIMIZED SEPSIS PREDICTION MODEL TRAINING ===\n")

# Configuration
SAMPLE_SIZE = 100000  # Use subset for faster development, increase for final model
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

print("Loading and preprocessing real clinical dataset...")
df = pd.read_csv('../Dataset.csv')
print(f"Original dataset: {df.shape}")

# Sample for faster processing (remove this for final model)
if len(df) > SAMPLE_SIZE:
    print(f"Sampling {SAMPLE_SIZE:,} records for faster processing...")
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled dataset: {df.shape}")

# Feature Engineering
print("\n=== FEATURE ENGINEERING ===")

# Define feature categories
vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
lab_values = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
              'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
              'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
              'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
              'Fibrinogen', 'Platelets']
demographics = ['Age', 'Gender']
temporal = ['ICULOS']  # Focus on ICU length of stay

# Select features with reasonable data availability (< 90% missing)
feature_availability = df.isnull().mean()
available_features = feature_availability[feature_availability < 0.9].index.tolist()
available_features = [f for f in available_features if f not in ['Unnamed: 0', 'Hour', 'Unit1', 'Unit2', 'HospAdmTime', 'Patient_ID', 'SepsisLabel']]

print(f"Available features (< 90% missing): {len(available_features)}")
print(f"Features: {available_features}")

# Create feature matrix
X = df[available_features].copy()
y = df['SepsisLabel'].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Advanced Feature Engineering
print("\n=== ADVANCED FEATURE ENGINEERING ===")

# 1. Vital signs ratios and combinations
if 'HR' in X.columns and 'SBP' in X.columns:
    X['HR_SBP_ratio'] = X['HR'] / (X['SBP'] + 1e-6)
    
if 'Temp' in X.columns:
    X['Temp_abnormal'] = ((X['Temp'] < 36) | (X['Temp'] > 38)).astype(int)
    
if 'HR' in X.columns:
    X['HR_abnormal'] = ((X['HR'] < 60) | (X['HR'] > 100)).astype(int)

# 2. Lab value ratios
if 'WBC' in X.columns:
    X['WBC_abnormal'] = ((X['WBC'] < 4) | (X['WBC'] > 12)).astype(int)

# 3. Age categories
if 'Age' in X.columns:
    X['Age_elderly'] = (X['Age'] >= 65).astype(int)
    X['Age_category'] = pd.cut(X['Age'], bins=[0, 40, 65, 80, 100], labels=[0, 1, 2, 3]).astype(float)

# 4. ICULOS categories
if 'ICULOS' in X.columns:
    X['ICULOS_long'] = (X['ICULOS'] > 72).astype(int)  # > 3 days

print(f"Enhanced feature matrix shape: {X.shape}")

# Handle missing values with advanced imputation
print("\n=== ADVANCED MISSING VALUE HANDLING ===")

# Separate numeric and categorical features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# For features with < 50% missing: KNN imputation
# For features with > 50% missing: Simple imputation + missing indicator
high_missing_features = []
low_missing_features = []

for col in numeric_features:
    missing_rate = X[col].isnull().mean()
    if missing_rate > 0.5:
        high_missing_features.append(col)
    else:
        low_missing_features.append(col)

print(f"High missing features (>50%): {len(high_missing_features)}")
print(f"Low missing features (<=50%): {len(low_missing_features)}")

# Create missing indicators for high missing features
for col in high_missing_features:
    X[f'{col}_missing'] = X[col].isnull().astype(int)

# Impute high missing features with median
high_missing_imputer = SimpleImputer(strategy='median')
if high_missing_features:
    X[high_missing_features] = high_missing_imputer.fit_transform(X[high_missing_features])

# Impute low missing features with KNN (limited to prevent memory issues)
if low_missing_features and len(low_missing_features) <= 20:  # Limit for memory
    knn_imputer = KNNImputer(n_neighbors=5)
    X[low_missing_features] = knn_imputer.fit_transform(X[low_missing_features])
else:
    # Fallback to median imputation
    low_missing_imputer = SimpleImputer(strategy='median')
    if low_missing_features:
        X[low_missing_features] = low_missing_imputer.fit_transform(X[low_missing_features])

print("Missing value imputation completed")
print(f"Remaining missing values: {X.isnull().sum().sum()}")

# Feature Selection
print("\n=== FEATURE SELECTION ===")

# Remove features with zero variance
from sklearn.feature_selection import VarianceThreshold
variance_selector = VarianceThreshold(threshold=0.01)
X_variance = variance_selector.fit_transform(X)
selected_features = X.columns[variance_selector.get_support()].tolist()
X = X[selected_features]

print(f"After variance threshold: {X.shape}")

# Statistical feature selection
if X.shape[1] > 50:  # Only if we have many features
    print("Applying statistical feature selection...")
    selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_feature_names = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(X_selected, columns=selected_feature_names)
    print(f"After statistical selection: {X.shape}")

# Split the data
print("\n=== DATA SPLITTING ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training target distribution: {y_train.value_counts().to_dict()}")

# Feature Scaling
print("\n=== FEATURE SCALING ===")
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")

# Handle Class Imbalance
print("\n=== HANDLING CLASS IMBALANCE ===")
print(f"Original class distribution: {y_train.value_counts().to_dict()}")

# Try different sampling strategies
sampling_strategies = {
    'SMOTE': SMOTE(random_state=RANDOM_STATE),
    'ADASYN': ADASYN(random_state=RANDOM_STATE),
    'SMOTETomek': SMOTETomek(random_state=RANDOM_STATE)
}

best_strategy = None
best_score = 0

for strategy_name, sampler in sampling_strategies.items():
    try:
        print(f"\nTesting {strategy_name}...")
        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
        print(f"Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        # Quick validation with simple model
        rf_quick = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(rf_quick, X_resampled, y_resampled, cv=3, scoring='roc_auc')
        avg_score = cv_scores.mean()
        print(f"{strategy_name} CV AUC: {avg_score:.3f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_strategy = (strategy_name, sampler)
            
    except Exception as e:
        print(f"Error with {strategy_name}: {e}")

if best_strategy:
    print(f"\nBest sampling strategy: {best_strategy[0]} (AUC: {best_score:.3f})")
    X_train_final, y_train_final = best_strategy[1].fit_resample(X_train_scaled, y_train)
else:
    print("Using original data without resampling")
    X_train_final, y_train_final = X_train_scaled, y_train

print(f"Final training set: {X_train_final.shape}")
print(f"Final target distribution: {pd.Series(y_train_final).value_counts().to_dict()}")

# Model Training with Hyperparameter Optimization
print("\n=== OPTIMIZED MODEL TRAINING ===")

# XGBoost with optimized parameters for imbalanced data
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'scale_pos_weight': [1, 5, 10],  # Handle class imbalance
    'random_state': [RANDOM_STATE]
}

print("Performing hyperparameter optimization...")
xgb_model = xgb.XGBClassifier()

# Use smaller parameter grid for faster execution
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 200],
    'scale_pos_weight': [1, 5]
}

grid_search = GridSearchCV(
    xgb_model, 
    param_grid, 
    cv=3,  # Reduced for speed
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_final, y_train_final)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Train final model
best_model = grid_search.best_estimator_
print("\nTraining final optimized model...")

# Evaluation
print("\n=== MODEL EVALUATION ===")
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Calculate comprehensive metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print("=== FINAL MODEL PERFORMANCE ===")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"AUC-ROC:   {auc:.3f}")
print(f"Avg Precision: {avg_precision:.3f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Sepsis', 'Sepsis']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Format: [[TN, FP], [FN, TP]]")

# Feature importance
feature_importance = best_model.feature_importances_
feature_names = X.columns.tolist()
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n=== TOP 15 FEATURE IMPORTANCE ===")
print(feature_importance_df.head(15))

# Save the optimized model and components
print("\n=== SAVING OPTIMIZED MODEL ===")
joblib.dump(best_model, 'optimized_xgboost_sepsis_model.pkl')
joblib.dump(scaler, 'optimized_scaler.pkl')
joblib.dump(feature_names, 'optimized_feature_names.pkl')

# Save performance metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision, 
    'recall': recall,
    'f1_score': f1,
    'auc_roc': auc,
    'avg_precision': avg_precision,
    'best_params': grid_search.best_params_
}
joblib.dump(metrics, 'optimized_model_metrics.pkl')

# Save feature importance
feature_importance_df.to_csv('optimized_feature_importance.csv', index=False)

print("Optimized model training completed!")
print("\nFiles saved:")
print("- optimized_xgboost_sepsis_model.pkl")
print("- optimized_scaler.pkl") 
print("- optimized_feature_names.pkl")
print("- optimized_model_metrics.pkl")
print("- optimized_feature_importance.csv")

print(f"\n=== PERFORMANCE IMPROVEMENT ===")
print(f"Previous model AUC: 0.571")
print(f"Optimized model AUC: {auc:.3f}")
print(f"Improvement: {((auc - 0.571) / 0.571 * 100):+.1f}%")

if auc > 0.8:
    print("🎉 EXCELLENT: AUC > 0.8 - Model ready for clinical validation!")
elif auc > 0.7:
    print("✅ GOOD: AUC > 0.7 - Model shows strong predictive performance")
elif auc > 0.6:
    print("⚠️ MODERATE: AUC > 0.6 - Model needs further optimization")
else:
    print("❌ POOR: AUC < 0.6 - Model requires significant improvement")