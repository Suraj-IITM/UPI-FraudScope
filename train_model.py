"""
train_model.py
===============
Trains and evaluates fraud-detection models on upi_transactions.csv.

Compares 3 models:
  - Logistic Regression   (simple, interpretable baseline)
  - Random Forest         (non-linear, handles feature interactions)
  - Gradient Boosting     (usually strongest tabular performer)

Evaluation uses:
  - Stratified train/test split (fraud is rare -> stratify to keep ratio)
  - Precision, Recall, F1 (accuracy is misleading on imbalanced data)
  - ROC-AUC and PR-AUC (PR-AUC matters most for rare-event detection)
  - Confusion matrix
  - Feature importance (for the tree models)

The best model (by PR-AUC) is saved to fraud_model.joblib for use in
fraud_detector.py (the hybrid rule + ML detector).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve
)

FEATURES = [
    "amount", "hour", "is_new_recipient", "is_new_device",
    "txns_last_hour", "location_jump", "recipient_age_days",
    "sender_account_age_days", "contains_urgency_words",
    "upi_id_suspicious_keyword", "is_known_fraud_upi_db",
]
TARGET = "label"


def load_data(path="upi_transactions.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y, df


def evaluate(name, model, X_test, y_test, scaled=False, scaler=None):
    X_eval = scaler.transform(X_test) if scaled else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=3))
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}   (more informative than ROC-AUC here, since fraud is rare)")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                 Predicted Legit   Predicted Fraud")
    print(f"  Actual Legit   {cm[0][0]:>14}   {cm[0][1]:>15}")
    print(f"  Actual Fraud   {cm[1][0]:>14}   {cm[1][1]:>15}")
    return pr_auc


def main():
    X, y, df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Fraud in train: {y_train.mean():.2%}  |  Fraud in test: {y_test.mean():.2%}")

    results = {}

    # --- Logistic Regression (needs scaled features) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_s, y_train)
    results["Logistic Regression"] = evaluate(
        "Logistic Regression", logreg, X_test, y_test, scaled=True, scaler=scaler
    )

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate("Random Forest", rf, X_test, y_test)

    # --- Gradient Boosting ---
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    results["Gradient Boosting"] = evaluate("Gradient Boosting", gb, X_test, y_test)

    # --- Pick best model by PR-AUC (best metric for rare-event fraud detection) ---
    best_name = max(results, key=results.get)
    print(f"\n{'#'*60}")
    print(f"  BEST MODEL: {best_name}  (PR-AUC = {results[best_name]:.4f})")
    print(f"{'#'*60}")

    best_model = {"Logistic Regression": logreg, "Random Forest": rf, "Gradient Boosting": gb}[best_name]

    # Feature importance (tree models only)
    if best_name in ("Random Forest", "Gradient Boosting"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\n  Feature Importance:")
        for feat, imp in importances.items():
            bar = "█" * int(imp * 50)
            print(f"    {feat:<28} {imp:.3f}  {bar}")

    # Save artifacts
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler if best_name == "Logistic Regression" else None,
        "features": FEATURES,
    }, "fraud_model.joblib")
    print(f"\nSaved best model -> fraud_model.joblib")


if __name__ == "__main__":
    main()
