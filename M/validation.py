# Recommended additions for better validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Cross-validation
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, 
                           cv=StratifiedKFold(5), scoring='roc_auc')
print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Learning curves
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    xgb_model, X_train_scaled, y_train, cv=5)
