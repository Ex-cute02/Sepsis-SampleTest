#!/usr/bin/env python3
"""
Advanced Data Preprocessing and Cleaning Pipeline
Comprehensive preprocessing for optimal sepsis prediction model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.stats import zscore
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataPreprocessor:
    """Advanced data preprocessing pipeline for sepsis prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.imputers = {}
        self.transformers = {}
        self.outlier_detectors = {}
        self.feature_stats = {}
        
    def load_and_analyze_data(self, file_path='../Dataset.csv', sample_size=None):
        """Load and perform initial data analysis"""
        print("=== LOADING AND ANALYZING DATASET ===\n")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        print(f"Total records: {len(df):,}")
        print(f"Unique patients: {df['Patient_ID'].nunique():,}")
        
        # Sample if needed
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size:,} records...")
            # Stratified sampling maintaining class distribution
            sepsis_rate = df['SepsisLabel'].mean()
            df_majority = df[df['SepsisLabel'] == 0].sample(
                n=int(sample_size * (1 - sepsis_rate)), 
                random_state=self.random_state
            )
            df_minority = df[df['SepsisLabel'] == 1].sample(
                n=int(sample_size * sepsis_rate), 
                random_state=self.random_state
            )
            df = pd.concat([df_majority, df_minority]).sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
            print(f"Sampled dataset shape: {df.shape}")
        
        # Basic statistics
        print(f"\n=== DATASET OVERVIEW ===")
        print(f"Sepsis rate: {df['SepsisLabel'].mean():.3f} ({df['SepsisLabel'].mean()*100:.1f}%)")
        print(f"Average age: {df['Age'].mean():.1f} years")
        print(f"Gender distribution: {df['Gender'].value_counts().to_dict()}")
        print(f"Average ICU stay: {df['ICULOS'].mean():.1f} hours")
        
        return df
    
    def analyze_missing_patterns(self, df):
        """Comprehensive missing data analysis"""
        print("\n=== MISSING DATA ANALYSIS ===\n")
        
        # Missing value statistics
        missing_stats = pd.DataFrame({
            'Missing_Count': df.isnull().sum(),
            'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2),
            'Data_Type': df.dtypes
        }).sort_values('Missing_Percent', ascending=False)
        
        print("Missing value summary:")
        print(missing_stats.head(15))
        
        # Missing patterns by patient
        patient_missing = df.groupby('Patient_ID').apply(
            lambda x: x.isnull().sum().sum()
        ).describe()
        print(f"\nMissing values per patient statistics:")
        print(patient_missing)
        
        # Correlation between missing values and target
        missing_indicators = df.isnull().astype(int)
        missing_target_corr = missing_indicators.corrwith(df['SepsisLabel']).abs().sort_values(ascending=False)
        print(f"\nTop 10 missing patterns correlated with sepsis:")
        print(missing_target_corr.head(10))
        
        self.feature_stats['missing_stats'] = missing_stats
        return missing_stats
    
    def detect_and_handle_outliers(self, df, features):
        """Advanced outlier detection and handling"""
        print("\n=== OUTLIER DETECTION AND HANDLING ===\n")
        
        outlier_summary = {}
        
        for feature in features:
            if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
                # Statistical outlier detection
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Z-score outliers
                z_scores = np.abs(zscore(df[feature].dropna()))
                z_outliers = z_scores > 3
                
                # Clinical bounds (domain knowledge)
                clinical_bounds = self.get_clinical_bounds(feature)
                clinical_outliers = (
                    (df[feature] < clinical_bounds['min']) | 
                    (df[feature] > clinical_bounds['max'])
                )
                
                outlier_count = clinical_outliers.sum()
                outlier_percent = (outlier_count / len(df)) * 100
                
                outlier_summary[feature] = {
                    'outlier_count': outlier_count,
                    'outlier_percent': outlier_percent,
                    'clinical_bounds': clinical_bounds
                }
                
                # Handle outliers based on clinical knowledge
                if outlier_percent > 0.1:  # If >0.1% outliers
                    print(f"{feature}: {outlier_count} outliers ({outlier_percent:.2f}%)")
                    # Cap outliers to clinical bounds
                    df[feature] = df[feature].clip(
                        lower=clinical_bounds['min'], 
                        upper=clinical_bounds['max']
                    )
        
        self.feature_stats['outlier_summary'] = outlier_summary
        return df
    
    def get_clinical_bounds(self, feature):
        """Get clinically reasonable bounds for features"""
        clinical_bounds = {
            'HR': {'min': 20, 'max': 250},
            'O2Sat': {'min': 50, 'max': 100},
            'Temp': {'min': 30, 'max': 45},
            'SBP': {'min': 50, 'max': 300},
            'MAP': {'min': 30, 'max': 200},
            'DBP': {'min': 20, 'max': 150},
            'Resp': {'min': 5, 'max': 60},
            'Age': {'min': 0, 'max': 120},
            'ICULOS': {'min': 0, 'max': 2000},
            'Glucose': {'min': 20, 'max': 1000},
            'BUN': {'min': 1, 'max': 200},
            'Creatinine': {'min': 0.1, 'max': 20},
            'WBC': {'min': 0.1, 'max': 100},
            'Hct': {'min': 10, 'max': 70},
            'Hgb': {'min': 3, 'max': 25},
            'Platelets': {'min': 10, 'max': 2000},
            'Lactate': {'min': 0.1, 'max': 30}
        }
        return clinical_bounds.get(feature, {'min': -np.inf, 'max': np.inf})
    
    def advanced_feature_engineering(self, df):
        """Comprehensive clinical feature engineering"""
        print("\n=== ADVANCED FEATURE ENGINEERING ===\n")
        
        original_features = df.shape[1]
        
        # 1. Clinical Composite Scores
        print("Creating clinical composite scores...")
        
        # SIRS Criteria (Systemic Inflammatory Response Syndrome)
        sirs_criteria = 0
        if 'Temp' in df.columns:
            sirs_criteria += ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(int)
        if 'HR' in df.columns:
            sirs_criteria += (df['HR'] > 90).astype(int)
        if 'Resp' in df.columns:
            sirs_criteria += (df['Resp'] > 20).astype(int)
        if 'WBC' in df.columns:
            sirs_criteria += ((df['WBC'] > 12) | (df['WBC'] < 4)).astype(int)
        df['SIRS_Score'] = sirs_criteria
        
        # qSOFA Score (quick Sequential Organ Failure Assessment)
        qsofa_score = 0
        if 'SBP' in df.columns:
            qsofa_score += (df['SBP'] <= 100).astype(int)
        if 'Resp' in df.columns:
            qsofa_score += (df['Resp'] >= 22).astype(int)
        # GCS would be ideal but not available, use altered mental status proxy
        df['qSOFA_Score'] = qsofa_score
        
        # 2. Vital Sign Ratios and Indices
        print("Creating vital sign ratios...")
        
        if 'HR' in df.columns and 'SBP' in df.columns:
            df['Shock_Index'] = df['HR'] / (df['SBP'] + 1e-6)
            df['Shock_Index_High'] = (df['Shock_Index'] > 0.9).astype(int)
        
        if 'SBP' in df.columns and 'DBP' in df.columns:
            df['Pulse_Pressure'] = df['SBP'] - df['DBP']
            df['Pulse_Pressure_Narrow'] = (df['Pulse_Pressure'] < 25).astype(int)
        
        if 'MAP' in df.columns:
            df['MAP_Critical'] = (df['MAP'] < 65).astype(int)
        
        # 3. Laboratory Ratios
        print("Creating laboratory ratios...")
        
        if 'BUN' in df.columns and 'Creatinine' in df.columns:
            df['BUN_Creatinine_Ratio'] = df['BUN'] / (df['Creatinine'] + 1e-6)
            df['AKI_Risk'] = (df['BUN_Creatinine_Ratio'] > 20).astype(int)
        
        if 'WBC' in df.columns and 'Hct' in df.columns:
            df['WBC_Hct_Ratio'] = df['WBC'] / (df['Hct'] + 1e-6)
        
        # 4. Time-based Features
        print("Creating temporal features...")
        
        if 'ICULOS' in df.columns:
            df['ICU_Day'] = (df['ICULOS'] / 24).astype(int)
            df['ICU_Hour_of_Day'] = df['ICULOS'] % 24
            df['ICU_Early'] = (df['ICULOS'] <= 6).astype(int)
            df['ICU_Late'] = (df['ICULOS'] > 72).astype(int)
        
        # 5. Age-based Risk Categories
        print("Creating age-based features...")
        
        if 'Age' in df.columns:
            df['Age_Pediatric'] = (df['Age'] < 18).astype(int)
            df['Age_Adult'] = ((df['Age'] >= 18) & (df['Age'] < 65)).astype(int)
            df['Age_Elderly'] = ((df['Age'] >= 65) & (df['Age'] < 80)).astype(int)
            df['Age_Very_Elderly'] = (df['Age'] >= 80).astype(int)
            df['Age_Squared'] = df['Age'] ** 2
        
        # 6. Clinical Severity Indicators
        print("Creating severity indicators...")
        
        # Multi-organ dysfunction indicators
        organ_dysfunction = 0
        if 'SBP' in df.columns:
            organ_dysfunction += (df['SBP'] < 90).astype(int)  # Cardiovascular
        if 'O2Sat' in df.columns:
            organ_dysfunction += (df['O2Sat'] < 90).astype(int)  # Respiratory
        if 'Creatinine' in df.columns:
            organ_dysfunction += (df['Creatinine'] > 2.0).astype(int)  # Renal
        if 'Platelets' in df.columns:
            organ_dysfunction += (df['Platelets'] < 100).astype(int)  # Hematologic
        df['Organ_Dysfunction_Count'] = organ_dysfunction
        
        # 7. Interaction Features
        print("Creating interaction features...")
        
        if 'Age' in df.columns and 'SIRS_Score' in df.columns:
            df['Age_SIRS_Interaction'] = df['Age'] * df['SIRS_Score']
        
        if 'Gender' in df.columns and 'Age' in df.columns:
            df['Gender_Age_Interaction'] = df['Gender'] * df['Age']
        
        new_features = df.shape[1] - original_features
        print(f"Created {new_features} new features")
        print(f"Total features: {df.shape[1]}")
        
        return df
    
    def advanced_imputation(self, df, target_col='SepsisLabel'):
        """Advanced missing value imputation strategies"""
        print("\n=== ADVANCED MISSING VALUE IMPUTATION ===\n")
        
        # Separate features by missing percentage
        missing_stats = df.isnull().mean()
        
        low_missing = missing_stats[missing_stats <= 0.3].index.tolist()
        medium_missing = missing_stats[(missing_stats > 0.3) & (missing_stats <= 0.7)].index.tolist()
        high_missing = missing_stats[missing_stats > 0.7].index.tolist()
        
        # Remove non-feature columns
        exclude_cols = ['Patient_ID', 'Hour', 'Unit1', 'Unit2', 'HospAdmTime', 'Unnamed: 0', target_col]
        low_missing = [col for col in low_missing if col not in exclude_cols]
        medium_missing = [col for col in medium_missing if col not in exclude_cols]
        high_missing = [col for col in high_missing if col not in exclude_cols]
        
        print(f"Low missing (<30%): {len(low_missing)} features")
        print(f"Medium missing (30-70%): {len(medium_missing)} features")
        print(f"High missing (>70%): {len(high_missing)} features")
        
        # Strategy 1: KNN Imputation for low missing features
        if low_missing:
            print("Applying KNN imputation for low missing features...")
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            df[low_missing] = knn_imputer.fit_transform(df[low_missing])
            self.imputers['knn'] = knn_imputer
        
        # Strategy 2: Iterative imputation for medium missing features
        if medium_missing:
            print("Applying iterative imputation for medium missing features...")
            iterative_imputer = IterativeImputer(
                estimator=None,  # Uses BayesianRidge by default
                max_iter=10,
                random_state=self.random_state
            )
            df[medium_missing] = iterative_imputer.fit_transform(df[medium_missing])
            self.imputers['iterative'] = iterative_imputer
        
        # Strategy 3: Clinical-informed imputation for high missing features
        if high_missing:
            print("Applying clinical-informed imputation for high missing features...")
            for col in high_missing:
                if col in df.columns:
                    # Create missing indicator
                    df[f'{col}_Missing'] = df[col].isnull().astype(int)
                    
                    # Impute with clinical normal values
                    clinical_normals = {
                        'Lactate': 1.5, 'AST': 25, 'BUN': 15, 'Alkalinephos': 100,
                        'Calcium': 9.5, 'Chloride': 100, 'Creatinine': 1.0,
                        'Bilirubin_direct': 0.2, 'Glucose': 100, 'Magnesium': 2.0,
                        'Phosphate': 3.5, 'Potassium': 4.0, 'Bilirubin_total': 1.0,
                        'TroponinI': 0.01, 'PTT': 30, 'Fibrinogen': 300,
                        'BaseExcess': 0, 'HCO3': 24, 'PaCO2': 40, 'pH': 7.4,
                        'SaO2': 98, 'EtCO2': 35, 'FiO2': 21
                    }
                    
                    normal_value = clinical_normals.get(col, df[col].median())
                    df[col].fillna(normal_value, inplace=True)
        
        print("Imputation completed")
        print(f"Remaining missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def feature_transformation(self, df, target_col='SepsisLabel'):
        """Advanced feature transformations"""
        print("\n=== FEATURE TRANSFORMATIONS ===\n")
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col != target_col]
        
        # 1. Skewness correction
        print("Analyzing and correcting skewness...")
        skewed_features = []
        
        for feature in numeric_features:
            if feature in df.columns:
                skewness = df[feature].skew()
                if abs(skewness) > 1:  # Highly skewed
                    skewed_features.append(feature)
                    print(f"{feature}: skewness = {skewness:.2f}")
        
        # Apply power transformation to highly skewed features
        if skewed_features:
            print(f"Applying power transformation to {len(skewed_features)} skewed features...")
            power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            df[skewed_features] = power_transformer.fit_transform(df[skewed_features])
            self.transformers['power'] = power_transformer
        
        # 2. Normalization/Standardization
        print("Applying robust scaling...")
        scaler = RobustScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        self.scalers['robust'] = scaler
        
        return df
    
    def intelligent_feature_selection(self, df, target_col='SepsisLabel'):
        """Intelligent feature selection based on clinical relevance and statistical significance"""
        print("\n=== INTELLIGENT FEATURE SELECTION ===\n")
        
        X = df.drop(columns=[target_col, 'Patient_ID'], errors='ignore')
        y = df[target_col]
        
        original_features = X.shape[1]
        print(f"Starting with {original_features} features")
        
        # 1. Remove constant and quasi-constant features
        print("Removing constant and quasi-constant features...")
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X)
        selected_features = X.columns[variance_selector.get_support()].tolist()
        X = X[selected_features]
        print(f"After variance threshold: {X.shape[1]} features")
        
        # 2. Statistical significance test
        print("Selecting statistically significant features...")
        # Use all features if reasonable number, otherwise select top 50
        k = min(50, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_feature_names = X.columns[selector.get_support()].tolist()
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        print(f"After statistical selection: {len(selected_feature_names)} features")
        print("\nTop 15 most significant features:")
        print(feature_scores.head(15))
        
        # 3. Clinical relevance filtering
        print("Applying clinical relevance filtering...")
        clinical_priority_features = [
            'SIRS_Score', 'qSOFA_Score', 'Shock_Index', 'MAP_Critical',
            'Organ_Dysfunction_Count', 'Age_Elderly', 'ICU_Late',
            'HR', 'Temp', 'SBP', 'Resp', 'Age', 'ICULOS'
        ]
        
        # Ensure clinical priority features are included if available
        final_features = selected_feature_names.copy()
        for feature in clinical_priority_features:
            if feature in X.columns and feature not in final_features:
                final_features.append(feature)
        
        print(f"Final feature count: {len(final_features)}")
        
        # Create final dataset
        final_df = df[final_features + [target_col, 'Patient_ID']].copy()
        
        self.feature_stats['selected_features'] = final_features
        self.feature_stats['feature_scores'] = feature_scores
        
        return final_df
    
    def generate_preprocessing_report(self, original_df, processed_df):
        """Generate comprehensive preprocessing report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE PREPROCESSING REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET TRANSFORMATION SUMMARY")
        print(f"Original shape: {original_df.shape}")
        print(f"Processed shape: {processed_df.shape}")
        print(f"Features added: {processed_df.shape[1] - original_df.shape[1]}")
        
        print(f"\n🔧 PREPROCESSING STEPS APPLIED")
        print("✅ Advanced outlier detection and clinical bounds enforcement")
        print("✅ Comprehensive feature engineering (clinical scores, ratios, interactions)")
        print("✅ Multi-strategy missing value imputation")
        print("✅ Skewness correction with power transformations")
        print("✅ Robust scaling for outlier resistance")
        print("✅ Intelligent feature selection with clinical prioritization")
        
        print(f"\n📈 DATA QUALITY IMPROVEMENTS")
        original_missing = original_df.isnull().sum().sum()
        processed_missing = processed_df.isnull().sum().sum()
        print(f"Missing values: {original_missing:,} → {processed_missing:,}")
        print(f"Missing value reduction: {((original_missing - processed_missing) / original_missing * 100):.1f}%")
        
        if 'selected_features' in self.feature_stats:
            print(f"\n🎯 FEATURE SELECTION RESULTS")
            print(f"Selected features: {len(self.feature_stats['selected_features'])}")
            print("Top clinical features included:")
            clinical_features = [f for f in self.feature_stats['selected_features'] 
                               if any(keyword in f.lower() for keyword in 
                                    ['sirs', 'qsofa', 'shock', 'organ', 'age', 'icu'])]
            for feature in clinical_features[:10]:
                print(f"  • {feature}")
        
        return processed_df
    
    def save_preprocessing_artifacts(self, processed_df, prefix='advanced'):
        """Save all preprocessing artifacts"""
        print(f"\n💾 SAVING PREPROCESSING ARTIFACTS")
        
        # Save processed dataset
        processed_df.to_csv(f'{prefix}_processed_dataset.csv', index=False)
        print(f"✅ Saved: {prefix}_processed_dataset.csv")
        
        # Save preprocessing components
        joblib.dump(self.scalers, f'{prefix}_scalers.pkl')
        joblib.dump(self.imputers, f'{prefix}_imputers.pkl')
        joblib.dump(self.transformers, f'{prefix}_transformers.pkl')
        joblib.dump(self.feature_stats, f'{prefix}_feature_stats.pkl')
        
        print(f"✅ Saved: {prefix}_scalers.pkl")
        print(f"✅ Saved: {prefix}_imputers.pkl")
        print(f"✅ Saved: {prefix}_transformers.pkl")
        print(f"✅ Saved: {prefix}_feature_stats.pkl")
        
        return True

def main():
    """Main preprocessing pipeline"""
    print("=== ADVANCED DATA PREPROCESSING PIPELINE ===\n")
    
    # Initialize preprocessor
    preprocessor = AdvancedDataPreprocessor(random_state=42)
    
    # Load and analyze data
    df = preprocessor.load_and_analyze_data(sample_size=300000)  # Larger sample
    original_df = df.copy()
    
    # Comprehensive preprocessing pipeline
    missing_stats = preprocessor.analyze_missing_patterns(df)
    
    # Select features with reasonable availability
    available_features = missing_stats[missing_stats['Missing_Percent'] < 80].index.tolist()
    available_features = [f for f in available_features if f not in 
                         ['Unnamed: 0', 'Hour', 'Unit1', 'Unit2', 'HospAdmTime', 'Patient_ID', 'SepsisLabel']]
    
    df = preprocessor.detect_and_handle_outliers(df, available_features)
    df = preprocessor.advanced_feature_engineering(df)
    df = preprocessor.advanced_imputation(df)
    df = preprocessor.feature_transformation(df)
    df = preprocessor.intelligent_feature_selection(df)
    
    # Generate report and save artifacts
    processed_df = preprocessor.generate_preprocessing_report(original_df, df)
    preprocessor.save_preprocessing_artifacts(processed_df)
    
    print(f"\n🎉 PREPROCESSING COMPLETE!")
    print("Ready for advanced model training with cleaned and optimized dataset.")
    
    return processed_df, preprocessor

if __name__ == "__main__":
    processed_df, preprocessor = main()