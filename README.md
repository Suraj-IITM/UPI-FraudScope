# UPI FraudScope

A **hybrid rule-based + machine learning** fraud-risk scoring system for UPI
transactions. A trained ML model estimates fraud probability from
transaction behavior; a rule engine independently flags transactions
matching known, regulator-documented scam patterns. The two scores are
combined into a single 0–100 risk verdict with a plain-language
explanation of *why*. It can run as a batch pipeline or as a **real-time API**
that scores each payment as it arrives.

Why hybrid, not pure ML:
- **Rules** give instant, explainable, zero-training-needed flags for
  *known* bad actors (e.g. a UPI ID already reported to
  cybercrime.gov.in) — and update the moment a new scam ID is added.
- **ML** generalizes to *new, unseen* fraud patterns that don't match
  any hardcoded rule, and gives a continuous probability instead of a
  brittle yes/no.
- This mirrors how real bank/NPCI fraud systems are actually built: a
  fast rules layer for known bad actors, plus a statistical model layer
  for anomaly detection.

## Topics
✅ cybersecurity ✅ upi ✅ fraud-detection ✅ machine-learning
✅ python ✅ fintech ✅ india ✅ security ✅ real-time ✅ fastapi

## Pipeline

```bash
pip install -r requirements.txt

# 1. Generate a synthetic labeled transaction dataset (~20k rows, ~6% fraud)
python generate_data.py

# 2. Train and compare 3 models, save the best one
python train_model.py

# 3. Run the hybrid detector (loads fraud_model.joblib + rule engine)
python fraud_detector.py
```

`train_model.py` trains **Logistic Regression, Random Forest, and
Gradient Boosting**, evaluates all three with a stratified train/test
split (fraud is rare, so stratification keeps the ratio intact), and
selects the best model by **PR-AUC** — not accuracy, since accuracy is
misleading when ~94% of transactions are legitimate. It also prints a
confusion matrix and feature importances for the winning model.

**Last run: Gradient Boosting won, PR-AUC = 0.9916** (ROC-AUC 0.9995,
fraud-class precision 0.956 / recall 0.943 on a held-out test set).

*Caveat worth stating
