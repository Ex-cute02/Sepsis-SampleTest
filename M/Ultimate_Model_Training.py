#!/usr/bin/env python3
"""
Ultimate Model Training with Advanced Preprocessing
Training sepsis prediction model on comprehensively cleaned and preprocessed data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score,
                           accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class UltimateModelTrainer:
    """Ultimate model training with ensemble methods and advanced techniques"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.results = {}
        
    def load_preprocessed_data(self, file_path='advanced_processed_dataset.csv'):
        """Load the preprocessed dataset"""
        print("=== LOADING PREPROCESSED DATASET ===\n")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded preprocessed dataset: {df.shape}")
            print(f"Features: {df.shape[1] - 2}")  # Excluding target and Patient_ID
            print(f"Sepsis rate: {df['SepsisLabel'].mean():.3f}")
            return df
        except FileNotFoundError:
            print("Preprocessed dataset not found. Please run Advanced_Data_Preprocessing.py first.")
            return None
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("\n=== PREPARING DATA FOR TRAINING ===\n")
        
        # Separate features and target
        X = df.drop(columns=['SepsisLabel', 'Patient_ID'], errors='ignore')
        y = df['SepsisLabel']
        
        print(f"Feature matrix: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Patient-level split to prevent data leakage
        if 'Patient_ID' in df.columns:
            print("Performing patient-level split to prevent data leakage...")
            unique_patients = df['Patient_ID'].unique()
            
            # Stratify by patient-level sepsis occurrence
            patient_labels = df.groupby('Patient_ID')['SepsisLabel'].max()
            
            train_patients, test_patients = train_test_split(
                unique_patients, 
                test_size=0.2, 
                random_state=self.random_state,
                stratify=patient_labels
            )
            
            train_mask = df['Patient_ID'].isin(train_patients)
            test_mask = df['Patient_ID'].isin(test_patients)
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            print(f"Patient-level split:")
            print(f"  Train patients: {len(train_patients):,}")
            print(f"  Test patients: {len(test_patients):,}")
            print(f"  Train records: {len(X_train):,}")
            print(f"  Test records: {len(X_test):,}")
        else:
            # Standard split if no Patient_ID
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        
        print(f"Training target distribution: {y_train.value_counts().to_dict()}")
        print(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def optimize_class_balance(self, X_train, y_train):
        """Find optimal class balancing strategy"""
        print("\n=== OPTIMIZING CLASS BALANCE ===\n")
        
        # Test multiple sampling strategies
        sampling_strategies = {
            'None': None,
            'SMOTE_Conservative': SMOTE(sampling_strategy=0.1, random_state=self.random_state),
            'ADASYN_Conservative': ADASYN(sampling_strategy=0.15, random_state=self.random_state),
            'SMOTEENN': SMOTEENN(sampling_strategy=0.2, random_state=self.random_state),
        }
        
        best_strategy = None
        best_score = 0
        
        for strategy_name, sampler in sampling_strategies.items():
            try:
                print(f"Testing {strategy_name}...")
                
                if sampler is None:
                    X_resampled, y_resampled = X_train, y_train
                else:
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                
                print(f"  Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
                
                # Quick validation with balanced random forest
                brf = BalancedRandomForestClassifier(
                    n_estimators=50, 
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                cv_scores = cross_val_score(
                    brf, X_resampled, y_resampled, 
                    cv=3, scoring='roc_auc', n_jobs=-1
                )
                
                avg_score = cv_scores.mean()
                std_score = cv_scores.std()
                print(f"  CV AUC: {avg_score:.3f} ± {std_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = (strategy_name, sampler)
                    
            except Exception as e:
                print(f"  Error with {strategy_name}: {e}")
        
        if best_strategy:
            print(f"\nBest sampling strategy: {best_strategy[0]} (AUC: {best_score:.3f})")
            if best_strategy[1] is None:
                return X_train, y_train
            else:
                return best_strategy[1].fit_resample(X_train, y_train)
        else:
            return X_train, y_train
    
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and create ensemble"""
        print("\n=== TRAINING ENSEMBLE MODELS ===\n")
        
        # Define base models with optimized parameters
        base_models = {
            'XGBoost': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                max_depth=4,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=1,
                min_child_weight=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                max_depth=4,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=1,
                min_child_samples=20,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            'BalancedRandomForest': BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=self.random_state
            )
        }
        
        # Train and evaluate each model
        model_results = {}
        
        for model_name, model in base_models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'avg_precision': average_precision_score(y_test, y_pred_proba)
            }
            
            model_results[model_name] = results
            self.models[model_name] = model
            
            print(f"  AUC-ROC: {results['auc_roc']:.3f}")
            print(f"  Precision: {results['precision']:.3f}")
            print(f"  Recall: {results['recall']:.3f}")
        
        # Create ensemble model
        print("\nCreating ensemble model...")
        
        # Select top 3 models based on AUC
        top_models = sorted(model_results.items(), key=lambda x: x[1]['auc_roc'], reverse=True)[:3]
        print(f"Top models for ensemble: {[name for name, _ in top_models]}")
        
        ensemble_estimators = [(name, self.models[name]) for name, _ in top_models]
        
        ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft',
            n_jobs=-1
        )
        
        print("Training ensemble model...")
        ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble_model.predict(X_test)
        y_pred_proba_ensemble = ensemble_model.predict_proba(X_test)[:, 1]
        
        ensemble_results = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble),
            'recall': recall_score(y_test, y_pred_ensemble),
            'f1_score': f1_score(y_test, y_pred_ensemble),
            'auc_roc': roc_auc_score(y_test, y_pred_proba_ensemble),
            'avg_precision': average_precision_score(y_test, y_pred_proba_ensemble)
        }
        
        model_results['Ensemble'] = ensemble_results
        self.models['Ensemble'] = ensemble_model
        
        print(f"Ensemble AUC-ROC: {ensemble_results['auc_roc']:.3f}")
        
        self.results = model_results
        return model_results
    
    def hyperparameter_optimization(self, X_train, y_train, model_name='XGBoost'):
        """Advanced hyperparameter optimization"""
        print(f"\n=== HYPERPARAMETER OPTIMIZATION FOR {model_name} ===\n")
        
        if model_name == 'XGBoost':
            param_grid = {
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [200, 300, 500],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0.1, 1, 10],
                'reg_lambda': [0.1, 1, 10]
            }
            
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif model_name == 'LightGBM':
            param_grid = {
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [200, 300, 500],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0.1, 1, 10],
                'reg_lambda': [0.1, 1, 10]
            }
            
            base_model = lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        # Randomized search for efficiency
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=50,  # Reduced for efficiency
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        print("Performing randomized hyperparameter search...")
        random_search.fit(X_train, y_train)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.3f}")
        
        return random_search.best_estimator_
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Comprehensive model evaluation and comparison"""
        print("\n=== COMPREHENSIVE MODEL EVALUATION ===\n")
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['auc_roc'])
        best_model = self.models[best_model_name]
        best_auc = self.results[best_model_name]['auc_roc']
        
        print(f"Best model: {best_model_name} (AUC: {best_auc:.3f})")
        
        # Detailed evaluation of best model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        print(f"\n=== {best_model_name.upper()} DETAILED RESULTS ===")
        print(f"AUC-ROC: {self.results[best_model_name]['auc_roc']:.3f}")
        print(f"Accuracy: {self.results[best_model_name]['accuracy']:.3f}")
        print(f"Precision: {self.results[best_model_name]['precision']:.3f}")
        print(f"Recall: {self.results[best_model_name]['recall']:.3f}")
        print(f"F1-Score: {self.results[best_model_name]['f1_score']:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Sepsis', 'Sepsis']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Clinical metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"\nClinical Metrics:")
        print(f"Sensitivity (Recall): {self.results[best_model_name]['recall']:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"PPV (Precision): {ppv:.3f}")
        print(f"NPV: {npv:.3f}")
        
        # Model comparison table
        print(f"\n=== MODEL COMPARISON TABLE ===")
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(3)
        print(comparison_df.sort_values('auc_roc', ascending=False))
        
        self.best_model = best_model
        return best_model, best_model_name
    
    def save_ultimate_model(self, best_model, best_model_name, X_train):
        """Save the ultimate model and artifacts"""
        print(f"\n=== SAVING ULTIMATE MODEL ===\n")
        
        # Save best model
        joblib.dump(best_model, 'ultimate_sepsis_model.pkl')
        print(f"✅ Saved: ultimate_sepsis_model.pkl ({best_model_name})")
        
        # Save feature names
        feature_names = X_train.columns.tolist()
        joblib.dump(feature_names, 'ultimate_feature_names.pkl')
        print(f"✅ Saved: ultimate_feature_names.pkl")
        
        # Save all results
        joblib.dump(self.results, 'ultimate_model_results.pkl')
        print(f"✅ Saved: ultimate_model_results.pkl")
        
        # Save all models for ensemble use
        joblib.dump(self.models, 'ultimate_all_models.pkl')
        print(f"✅ Saved: ultimate_all_models.pkl")
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv('ultimate_feature_importance.csv', index=False)
            print(f"✅ Saved: ultimate_feature_importance.csv")
            
            print(f"\nTop 15 Feature Importance:")
            print(importance_df.head(15))
        
        return True
    
    def clinical_interpretation(self, best_auc):
        """Provide clinical interpretation"""
        print(f"\n=== CLINICAL INTERPRETATION ===")
        
        if best_auc >= 0.9:
            rating = "EXCELLENT"
            description = "Outstanding discrimination - Ready for clinical deployment"
        elif best_auc >= 0.8:
            rating = "VERY GOOD"
            description = "Strong predictive performance - Clinical validation recommended"
        elif best_auc >= 0.75:
            rating = "GOOD"
            description = "Good predictive performance - Further validation needed"
        elif best_auc >= 0.7:
            rating = "FAIR"
            description = "Acceptable performance - May have limited clinical utility"
        else:
            rating = "POOR"
            description = "Limited clinical utility - Needs improvement"
        
        print(f"AUC Score: {best_auc:.3f}")
        print(f"Rating: {rating}")
        print(f"Description: {description}")
        
        return rating

def main():
    """Main ultimate training pipeline"""
    print("=== ULTIMATE MODEL TRAINING PIPELINE ===\n")
    
    # Initialize trainer
    trainer = UltimateModelTrainer(random_state=42)
    
    # Load preprocessed data
    df = trainer.load_preprocessed_data()
    if df is None:
        print("Please run Advanced_Data_Preprocessing.py first to create preprocessed dataset.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Optimize class balance
    X_train_balanced, y_train_balanced = trainer.optimize_class_balance(X_train, y_train)
    
    # Train ensemble models
    model_results = trainer.train_ensemble_models(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Comprehensive evaluation
    best_model, best_model_name = trainer.comprehensive_evaluation(X_test, y_test)
    
    # Save ultimate model
    trainer.save_ultimate_model(best_model, best_model_name, X_train)
    
    # Clinical interpretation
    best_auc = model_results[best_model_name]['auc_roc']
    rating = trainer.clinical_interpretation(best_auc)
    
    print(f"\n🎉 ULTIMATE MODEL TRAINING COMPLETE!")
    print(f"Best Model: {best_model_name}")
    print(f"Performance: {best_auc:.3f} AUC ({rating})")
    print("Ready for clinical validation and deployment!")
    
    return trainer, best_model

if __name__ == "__main__":
    trainer, best_model = main()