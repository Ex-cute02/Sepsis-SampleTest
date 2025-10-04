#!/usr/bin/env python3
"""
Advanced Model Evaluation and Comparison
Comprehensive evaluation of the optimized sepsis prediction model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_comprehensive(model, X_test, y_test, feature_names):
    """Comprehensive model evaluation"""
    print("=== COMPREHENSIVE MODEL EVALUATION ===\n")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    print("=== PERFORMANCE METRICS ===")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score:           {f1:.4f}")
    print(f"AUC-ROC:            {auc:.4f}")
    print(f"Average Precision:  {avg_precision:.4f}")
    
    # Confusion Matrix Analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\n=== CLINICAL METRICS ===")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"PPV (Precision):      {ppv:.4f}")
    print(f"NPV:                  {npv:.4f}")
    
    print(f"\n=== CONFUSION MATRIX ===")
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")
    
    # Clinical interpretation
    print(f"\n=== CLINICAL INTERPRETATION ===")
    if auc >= 0.9:
        interpretation = "EXCELLENT - Outstanding discrimination"
    elif auc >= 0.8:
        interpretation = "GOOD - Strong predictive performance"
    elif auc >= 0.7:
        interpretation = "FAIR - Acceptable performance"
    elif auc >= 0.6:
        interpretation = "POOR - Limited clinical utility"
    else:
        interpretation = "FAIL - No better than random"
    
    print(f"AUC Interpretation: {interpretation}")
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        print(f"\n=== TOP 10 FEATURE IMPORTANCE ===")
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:20}: {row['importance']:.4f}")
    
    # Plot curves
    plot_roc_curve(y_test, y_pred_proba, "Optimized Model ROC Curve")
    plot_precision_recall_curve(y_test, y_pred_proba, "Optimized Model PR Curve")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'avg_precision': avg_precision,
        'specificity': specificity,
        'npv': npv,
        'confusion_matrix': cm
    }

def compare_models():
    """Compare original vs optimized model performance"""
    print("=== MODEL COMPARISON ===\n")
    
    # Load original metrics
    try:
        original_metrics = joblib.load('model_metrics.pkl')
        print("Original Model Performance:")
        for metric, value in original_metrics.items():
            print(f"  {metric}: {value:.4f}")
    except:
        print("Original model metrics not found")
        original_metrics = {'auc_roc': 0.571}  # From previous results
    
    # Load optimized metrics
    try:
        optimized_metrics = joblib.load('optimized_model_metrics.pkl')
        print("\nOptimized Model Performance:")
        for metric, value in optimized_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    except:
        print("Optimized model metrics not found")
        return
    
    # Calculate improvements
    print("\n=== PERFORMANCE IMPROVEMENTS ===")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        if metric in original_metrics and metric in optimized_metrics:
            original = original_metrics[metric]
            optimized = optimized_metrics[metric]
            improvement = ((optimized - original) / original * 100) if original > 0 else 0
            print(f"{metric:15}: {original:.4f} → {optimized:.4f} ({improvement:+.1f}%)")

if __name__ == "__main__":
    print("=== ADVANCED MODEL EVALUATION ===\n")
    
    try:
        # Load optimized model and data
        print("Loading optimized model...")
        model = joblib.load('optimized_xgboost_sepsis_model.pkl')
        scaler = joblib.load('optimized_scaler.pkl')
        feature_names = joblib.load('optimized_feature_names.pkl')
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Number of features: {len(feature_names)}")
        
        # Load test data (you'll need to recreate this or save it from training)
        print("\nNote: Test data needs to be loaded separately")
        print("Run this after the optimized training completes")
        
        # Compare models
        compare_models()
        
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run Optimized_Model_Training.py first")
    except Exception as e:
        print(f"Error: {e}")