#!/usr/bin/env python3
"""
Comprehensive Model Performance Analysis
Analysis of sepsis prediction model results and recommendations for improvement
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

def analyze_model_performance():
    """Analyze and compare model performance"""
    
    print("=== COMPREHENSIVE MODEL PERFORMANCE ANALYSIS ===\n")
    
    # Load model metrics
    try:
        original_metrics = joblib.load('model_metrics.pkl')
        optimized_metrics = joblib.load('optimized_model_metrics.pkl')
        
        print("📊 MODEL COMPARISON SUMMARY")
        print("=" * 50)
        
        models = {
            'Original (Sample Data)': original_metrics,
            'Optimized (Real Data)': optimized_metrics
        }
        
        for model_name, metrics in models.items():
            print(f"\n{model_name}:")
            print(f"  AUC-ROC:   {metrics.get('auc_roc', 0):.3f}")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}")
            print(f"  Precision: {metrics.get('precision', 0):.3f}")
            print(f"  Recall:    {metrics.get('recall', 0):.3f}")
            print(f"  F1-Score:  {metrics.get('f1_score', 0):.3f}")
        
    except FileNotFoundError:
        print("Model metrics files not found")
        return
    
    # Analysis of results
    print(f"\n🔍 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    original_auc = original_metrics.get('auc_roc', 0)
    optimized_auc = optimized_metrics.get('auc_roc', 0)
    
    print(f"\n1. AUC-ROC Improvement:")
    print(f"   Original: {original_auc:.3f} → Optimized: {optimized_auc:.3f}")
    improvement = ((optimized_auc - original_auc) / original_auc * 100) if original_auc > 0 else 0
    print(f"   Change: {improvement:+.1f}%")
    
    # Identify key issues
    print(f"\n2. Key Observations:")
    
    opt_precision = optimized_metrics.get('precision', 0)
    opt_recall = optimized_metrics.get('recall', 0)
    
    if opt_precision < 0.1:
        print(f"   ⚠️  Very low precision ({opt_precision:.3f}) - High false positive rate")
    
    if opt_recall < 0.1:
        print(f"   ⚠️  Very low recall ({opt_recall:.3f}) - Missing many sepsis cases")
    
    if optimized_auc < 0.7:
        print(f"   ⚠️  Moderate AUC ({optimized_auc:.3f}) - Limited clinical utility")
    
    # Root cause analysis
    print(f"\n3. Root Cause Analysis:")
    print(f"   📈 Extreme class imbalance: 98.2% no sepsis, 1.8% sepsis")
    print(f"   🎯 High CV scores (0.999) vs test AUC (0.653) suggests overfitting")
    print(f"   📊 Real clinical data is much more challenging than synthetic data")
    print(f"   🔧 Current resampling may be too aggressive")
    
    return original_metrics, optimized_metrics

def clinical_interpretation(auc_score):
    """Provide clinical interpretation of AUC score"""
    
    print(f"\n🏥 CLINICAL INTERPRETATION")
    print("=" * 50)
    
    if auc_score >= 0.9:
        interpretation = "EXCELLENT"
        description = "Outstanding discrimination - Ready for clinical validation"
        recommendation = "Proceed with clinical trials and regulatory approval"
    elif auc_score >= 0.8:
        interpretation = "GOOD"
        description = "Strong predictive performance - Clinically useful"
        recommendation = "Validate with external datasets and clinical experts"
    elif auc_score >= 0.7:
        interpretation = "FAIR"
        description = "Acceptable performance - May have clinical utility"
        recommendation = "Further optimization needed before clinical deployment"
    elif auc_score >= 0.6:
        interpretation = "POOR"
        description = "Limited clinical utility - Needs significant improvement"
        recommendation = "Major model redesign required"
    else:
        interpretation = "FAIL"
        description = "No better than random - Not suitable for clinical use"
        recommendation = "Complete model overhaul needed"
    
    print(f"AUC Score: {auc_score:.3f}")
    print(f"Rating: {interpretation}")
    print(f"Description: {description}")
    print(f"Recommendation: {recommendation}")
    
    return interpretation

def improvement_recommendations():
    """Provide specific recommendations for model improvement"""
    
    print(f"\n🚀 IMPROVEMENT RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = [
        {
            "category": "1. Data Strategy",
            "items": [
                "Use full dataset (1.5M records) instead of 100K sample",
                "Implement patient-level splitting to prevent data leakage",
                "Consider temporal validation (train on older data, test on newer)",
                "Include more lab values with reasonable availability (<70% missing)"
            ]
        },
        {
            "category": "2. Feature Engineering",
            "items": [
                "Create more clinical composite scores (SIRS, qSOFA)",
                "Add time-series features (trends, slopes, variability)",
                "Include interaction terms between vital signs",
                "Use domain knowledge for feature creation"
            ]
        },
        {
            "category": "3. Class Imbalance Solutions",
            "items": [
                "Try cost-sensitive learning instead of resampling",
                "Use ensemble methods with different sampling strategies",
                "Implement focal loss for extreme imbalance",
                "Consider threshold optimization for clinical metrics"
            ]
        },
        {
            "category": "4. Model Architecture",
            "items": [
                "Try ensemble methods (Random Forest + XGBoost)",
                "Implement deep learning approaches (LSTM for time series)",
                "Use calibrated classifiers for better probability estimates",
                "Consider hierarchical models (patient-level + time-level)"
            ]
        },
        {
            "category": "5. Validation Strategy",
            "items": [
                "Implement nested cross-validation",
                "Use stratified sampling by hospital/unit",
                "Validate on external datasets",
                "Include clinical expert evaluation"
            ]
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['category']}:")
        for item in rec['items']:
            print(f"   • {item}")
    
    return recommendations

def next_steps_plan():
    """Provide actionable next steps"""
    
    print(f"\n📋 IMMEDIATE NEXT STEPS")
    print("=" * 50)
    
    steps = [
        "1. Run Enhanced_Model_Training.py with fixed early stopping",
        "2. Implement cost-sensitive XGBoost with class weights",
        "3. Try ensemble approach combining multiple models",
        "4. Validate on patient-level splits to prevent leakage",
        "5. Optimize threshold for clinical metrics (sensitivity/specificity)",
        "6. Create comprehensive validation report",
        "7. Prepare for clinical expert review"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n🎯 SUCCESS CRITERIA")
    print("=" * 20)
    print("   • AUC-ROC > 0.75 (minimum for clinical utility)")
    print("   • Sensitivity > 0.80 (catch most sepsis cases)")
    print("   • Specificity > 0.70 (limit false alarms)")
    print("   • Consistent performance across validation folds")
    print("   • Clinical expert approval")

def main():
    """Main analysis function"""
    
    # Analyze current performance
    try:
        original_metrics, optimized_metrics = analyze_model_performance()
        
        # Clinical interpretation
        current_auc = optimized_metrics.get('auc_roc', 0)
        clinical_interpretation(current_auc)
        
        # Improvement recommendations
        improvement_recommendations()
        
        # Next steps
        next_steps_plan()
        
        print(f"\n✅ ANALYSIS COMPLETE")
        print("=" * 30)
        print("Review recommendations and implement next steps for improved performance.")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Please ensure model training has completed and metric files exist.")

if __name__ == "__main__":
    main()