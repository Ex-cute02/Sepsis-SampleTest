#!/usr/bin/env python3
"""
Real Dataset Analysis for Sepsis Prediction
Analyzing the full clinical dataset for optimal model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== REAL DATASET ANALYSIS FOR SEPSIS PREDICTION ===\n")

# Load the real dataset
print("Loading real clinical dataset...")
df = pd.read_csv('../Dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Total records: {len(df):,}")
print(f"Total features: {df.shape[1]}")
print(f"Unique patients: {df['Patient_ID'].nunique():,}")

# Target analysis
print(f"\n=== TARGET ANALYSIS ===")
target_counts = df['SepsisLabel'].value_counts()
print("Target distribution:")
print(f"No Sepsis (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"Sepsis (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")
sepsis_rate = df['SepsisLabel'].mean()
print(f"Sepsis rate: {sepsis_rate:.3f} ({sepsis_rate*100:.1f}%)")

# Missing values analysis
print(f"\n=== MISSING VALUES ANALYSIS ===")
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_values / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing_Count': missing_values,
    'Missing_Percent': missing_percent
})
print("Top 15 columns with missing values:")
print(missing_df.head(15))

# Feature categories
vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
lab_values = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
              'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
              'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
              'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 
              'Fibrinogen', 'Platelets']
demographics = ['Age', 'Gender']
temporal = ['Hour', 'ICULOS', 'HospAdmTime']

print(f"\n=== FEATURE CATEGORIES ===")
print(f"Vital Signs ({len(vital_signs)}): {vital_signs}")
print(f"Lab Values ({len(lab_values)}): {lab_values[:5]}... (and {len(lab_values)-5} more)")
print(f"Demographics ({len(demographics)}): {demographics}")
print(f"Temporal ({len(temporal)}): {temporal}")

# Patient-level analysis
print(f"\n=== PATIENT-LEVEL ANALYSIS ===")
patient_stats = df.groupby('Patient_ID').agg({
    'SepsisLabel': ['max', 'sum', 'count'],
    'ICULOS': 'max',
    'Age': 'first',
    'Gender': 'first'
}).round(2)

patient_stats.columns = ['Ever_Sepsis', 'Sepsis_Hours', 'Total_Hours', 'Max_ICULOS', 'Age', 'Gender']
print(f"Patients with sepsis: {patient_stats['Ever_Sepsis'].sum():,} / {len(patient_stats):,} ({patient_stats['Ever_Sepsis'].mean()*100:.1f}%)")
print(f"Average ICU stay: {patient_stats['Max_ICULOS'].mean():.1f} hours")
print(f"Average age: {patient_stats['Age'].mean():.1f} years")
print(f"Gender distribution: {patient_stats['Gender'].value_counts().to_dict()}")

# Save analysis results
print(f"\n=== SAVING ANALYSIS RESULTS ===")
missing_df.to_csv('missing_values_analysis.csv')
patient_stats.to_csv('patient_level_analysis.csv')

print("Analysis complete!")
print("Files saved:")
print("- missing_values_analysis.csv")
print("- patient_level_analysis.csv")

# Quick correlation analysis for key features
print(f"\n=== CORRELATION ANALYSIS (Key Features) ===")
key_features = ['HR', 'Temp', 'SBP', 'Resp', 'Lactate', 'WBC', 'Age', 'ICULOS', 'SepsisLabel']
available_features = [f for f in key_features if f in df.columns]
corr_matrix = df[available_features].corr()
print("Correlation with SepsisLabel:")
sepsis_corr = corr_matrix['SepsisLabel'].abs().sort_values(ascending=False)
print(sepsis_corr.head(10))