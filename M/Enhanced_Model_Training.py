#!/usr/bin/env python3
"""
Enhanced Model Training for Real Sepsis Dataset
Addressing overfitting, class imbalance, and feature selection issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== ENHANCED SEPSIS PREDICTION MODEL TRAINING ===\n")

# Enhanced Configuration
SAMPLE_SIZE = 200000  # Increased sample size
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

print("Loading and preprocessing real clinical dataset...")
df = pd.read_csv('../Dataset.csv')
print(f"Original dataset: {df.shape}")

# Use larger sample for better representation
if len(df) > SAMPLE_SIZE:
    print(f"Sampling {SAMPLE_SIZE:,} records for enhanced training...")
    # Stratified sampling to maintain class distribution
    df_majority = df[df['SepsisLabel'] == 0].sample(n=int(SAMPLE_SIZE * 0.982), random_state=RANDOM_STATE)
    df_minority = df[df['SepsisLabel'] == 1].sample(n=int(SAMPLE_SIZE * 0.018), random_state=RANDOM_STATE)
    df = pd.concat([df_majority, df_minority]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Enhanced sampled dataset: {df.shape}")
    print(f"Sepsis rate maintained: {df['SepsisLabel'].mean():.3f}")

# Enhanced Feature Engineering
print("\n=== ENHANCED FEATURE ENGINEERING ===")

# More comprehensive feature selection - include more lab values with reasonable availability
vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
lab_values = ['Glucose', 'BUN', 'Creatinine', 'Hct', 'Hgb', 'WBC', 'Platelets']  # More available labs
demographics = ['Age', 'Gender']
temporal = ['ICULOS']

# Select features with < 80% missing (more lenient)
feature_availability = df.isnull().mean()
available_features = feature_availability[feature_availability < 0.8].index.tolist()
available_features = [f for f in available_features if f not in ['Unnamed: 0', 'Hour', 'Unit1', 'Unit2', 'HospAdmTime', 'Patient_ID', 'SepsisLabel']]

print(f"Available features (< 80% missing): {len(available_features)}")
print(f"Features: {available_features}")

# Create enhanced feature matrix
X = df[available_features].copy()
y = df['SepsisLabel'].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Enhanced Feature Engineering with clinical knowledge
print("\n=== CLINICAL FEATURE ENGINEERING ===")

# 1. Vital signs combinations and ratios
if 'HR' in X.columns and 'SBP' in X.columns:
    X['Shock_Index'] = X['HR'] / (X['SBP'] + 1e-6)  # Clinical shock index
    
if 'SBP' in X.columns and 'DBP' in X.columns:
    X['Pulse_Pressure'] = X['SBP'] - X['DBP']
    
if 'MAP' in X.columns:
    X['MAP_low'] = (X['MAP'] < 65).astype(int)  # Clinical threshold

# 2. Temperature categories (clinical significance)
if 'Temp' in X.columns:
    X['Fever'] = (X['Temp'] > 38.3).astype(int)
    X['Hypothermia'] = (X['Temp'] < 36).astype(int)
    X['Temp_normal'] = ((X['Temp'] >= 36) & (X['Temp'] <= 38.3)).astype(int)

# 3. Heart rate categories
if 'HR' in X.columns:
    X['Tachycardia'] = (X['HR'] > 100).astype(int)
    X['Bradycardia'] = (X['HR'] < 60).astype(int)

# 4. Respiratory categories
if 'Resp' in X.columns:
    X['Tachypnea'] = (X['Resp'] > 20).astype(int)

# 5. Lab value categories
if 'WBC' in X.columns:
    X['Leukocytosis'] = (X['WBC'] > 12).astype(int)
    X['Leukopenia'] = (X['WBC'] < 4).astype(int)

# 6. Age risk categories
if 'Age' in X.columns:
    X['Age_high_risk'] = (X['Age'] >= 65).astype(int)
    X['Age_very_high_risk'] = (X['Age'] >= 75).astype(int)

# 7. ICU stay categories
if 'ICULOS' in X.columns:
    X['ICU_prolonged'] = (X['ICULOS'] > 48).astype(int)  # > 2 days
    X['ICU_very_long'] = (X['ICULOS'] > 168).astype(int)  # > 1 week

print(f"Enhanced feature matrix shape: {X.shape}")

# Improved Missing Value Handling
print("\n=== IMPROVED MISSING VALUE HANDLING ===")

# Strategy: Use domain knowledge for imputation
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Clinical-informed imputation
imputation_values = {
    'HR': 80,      # Normal resting HR
    'SBP': 120,    # Normal systolic BP
    'DBP': 80,     # Normal diastolic BP
    'MAP': 93,     # Normal MAP
    'Temp': 37,    # Normal body temperature
    'Resp': 16,    # Normal respiratory rate
    'O2Sat': 98,   # Normal oxygen saturation
    'Age': 65,     # Median ICU age
    'ICULOS': 24,  # Median ICU stay
    'Glucose': 100, # Normal glucose
    'WBC': 7,      # Normal WBC count
    'Hct': 40,     # Normal hematocrit
    'Hgb': 13,     # Normal hemoglobin
    'Platelets': 250, # Normal platelet count
    'BUN': 15,     # Normal BUN
    'Creatinine': 1.0  # Normal creatinine
}

# Create missing indicators for features with >30% missing
for col in numeric_features:
    missing_rate = X[col].isnull().mean()
    if missing_rate > 0.3:
        X[f'{col}_missing'] = X[col].isnull().astype(int)

# Impute with clinical values
for col in numeric_features:
    if col in imputation_values:
        X[col].fillna(imputation_values[col], inplace=True)
    else:
        X[col].fillna(X[col].median(), inplace=True)

print("Clinical-informed imputation completed")
print(f"Remaining missing values: {X.isnull().sum().sum()}")

# Enhanced Feature Selection
print("\n=== ENHANCED FEATURE SELECTION ===")

# Remove constant features
from sklearn.feature_selection import VarianceThreshold
variance_selector = VarianceThreshold(threshold=0.001)  # More lenient
X_variance = variance_selector.fit_transform(X)
selected_features = X.columns[variance_selector.get_support()].tolist()
X = X[selected_features]

print(f"After variance threshold: {X.shape}")

# Recursive Feature Elimination with Cross-Validation
if X.shape[1] > 30:
    print("Applying RFECV for optimal feature selection...")
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
    rfecv = RFECV(estimator=rf_selector, step=1, cv=3, scoring='roc_auc', n_jobs=-1)
    X_selected = rfecv.fit_transform(X, y)
    selected_feature_names = X.columns[rfecv.support_].tolist()
    X = pd.DataFrame(X_selected, columns=selected_feature_names)
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"After RFECV: {X.shape}")

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
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")

# Enhanced Class Balancing
print("\n=== ENHANCED CLASS BALANCING ===")
print(f"Original class distribution: {y_train.value_counts().to_dict()}")

# More conservative resampling to prevent overfitting
sampling_strategies = {
    'Undersampling': RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE),  # 10:1 ratio
    'SMOTE_Conservative': SMOTE(sampling_strategy=0.2, random_state=RANDOM_STATE),  # 5:1 ratio
    'SMOTEENN': SMOTEENN(sampling_strategy=0.15, random_state=RANDOM_STATE)  # 6.7:1 ratio
}

best_strategy = None
best_score = 0

for strategy_name, sampler in sampling_strategies.items():
    try:
        print(f"\nTesting {strategy_name}...")
        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
        print(f"Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        # More rigorous validation
        xgb_quick = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(xgb_quick, X_resampled, y_resampled, cv=5, scoring='roc_auc')
        avg_score = cv_scores.mean()
        std_score = cv_scores.std()
        print(f"{strategy_name} CV AUC: {avg_score:.3f} ± {std_score:.3f}")
        
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

# Enhanced Model Training with Regularization
print("\n=== ENHANCED MODEL TRAINING ===")

# More conservative XGBoost parameters to prevent overfitting
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,  # Reduced depth
    'learning_rate': 0.05,  # Lower learning rate
    'n_estimators': 500,  # More trees with lower learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,  # L1 regularization
    'reg_lambda': 1,  # L2 regularization
    'min_child_weight': 5,  # Prevent overfitting
    'random_state': RANDOM_STATE
}

print("Training enhanced XGBoost model with regularization...")
enhanced_model = xgb.XGBClassifier(**xgb_params)

# Train with early stopping
enhanced_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print("Enhanced model training completed!")

# Comprehensive Evaluation
print("\n=== COMPREHENSIVE MODEL EVALUATION ===")
y_pred = enhanced_model.predict(X_test_scaled)
y_pred_proba = enhanced_model.predict_proba(X_test_scaled)[:, 1]

# Calculate all metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print("=== ENHANCED MODEL PERFORMANCE ===")
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

# Clinical metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\n=== CLINICAL METRICS ===")
print(f"Sensitivity (Recall): {recall:.3f}")
print(f"Specificity:          {specificity:.3f}")
print(f"PPV (Precision):      {ppv:.3f}")
print(f"NPV:                  {npv:.3f}")

# Feature importance
feature_importance = enhanced_model.feature_importances_
feature_names = X.columns.tolist()
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n=== TOP 15 FEATURE IMPORTANCE ===")
print(feature_importance_df.head(15))

# Save the enhanced model
print("\n=== SAVING ENHANCED MODEL ===")
joblib.dump(enhanced_model, 'enhanced_xgboost_sepsis_model.pkl')
joblib.dump(scaler, 'enhanced_scaler.pkl')
joblib.dump(feature_names, 'enhanced_feature_names.pkl')

# Save performance metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision, 
    'recall': recall,
    'f1_score': f1,
    'auc_roc': auc,
    'avg_precision': avg_precision,
    'specificity': specificity,
    'npv': npv,
    'ppv': ppv
}
joblib.dump(metrics, 'enhanced_model_metrics.pkl')

# Save feature importance
feature_importance_df.to_csv('enhanced_feature_importance.csv', index=False)

print("Enhanced model training completed!")
print("\nFiles saved:")
print("- enhanced_xgboost_sepsis_model.pkl")
print("- enhanced_scaler.pkl") 
print("- enhanced_feature_names.pkl")
print("- enhanced_model_metrics.pkl")
print("- enhanced_feature_importance.csv")

print(f"\n=== PERFORMANCE COMPARISON ===")
print(f"Previous optimized AUC: 0.653")
print(f"Enhanced model AUC: {auc:.3f}")
improvement = ((auc - 0.653) / 0.653 * 100) if auc > 0.653 else ((auc - 0.653) / 0.653 * 100)
print(f"Change: {improvement:+.1f}%")

if auc > 0.8:
    print("🎉 EXCELLENT: AUC > 0.8 - Model ready for clinical validation!")
elif auc > 0.7:
    print("✅ GOOD: AUC > 0.7 - Model shows strong predictive performance")
elif auc > 0.6:
    print("⚠️ MODERATE: AUC > 0.6 - Model performance acceptable")
else:
    print("❌ NEEDS IMPROVEMENT: AUC < 0.6 - Requires further optimization")

# Cross-validation for final validation
print(f"\n=== FINAL CROSS-VALIDATION ===")
cv_scores = cross_val_score(enhanced_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Individual CV scores: {[f'{score:.3f}' for score in cv_scores]}")

if cv_scores.std() < 0.05:
    print("✅ Model shows consistent performance across folds")
else:
    print("⚠️ Model performance varies across folds - may need more regularization")