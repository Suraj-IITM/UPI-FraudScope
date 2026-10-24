# UPI FraudScope

A **hybrid rule-based + machine learning** fraud-risk scoring system for UPI
transactions. A trained ML model estimates fraud probability from
transaction behavior; a rule engine independently flags transactions
matching known, regulator-documented scam patterns. The two scores are
combined into a single 0–100 risk verdict with a plain-language
explanation of *why*.

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
✅ python ✅ fintech ✅ india ✅ security

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

*Caveat worth stating up front: `amount` alone drives ~82% of the
model's decisions. That likely reflects the synthetic data generator
making fraud/legit amounts fall in fairly distinct ranges — real
transactions would probably be noisier. The obvious next step, given
access to real or anonymized data, would be to check whether this
separation holds up.*

`fraud_detector.py` then combines the trained model's fraud probability
(65% weight) with the rule engine's flags (35% weight) into a final
score, with one hard override: a UPI ID already confirmed fraudulent
always blocks regardless of the model's score.

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | generates the synthetic labeled training dataset |
| `train_model.py` | trains/compares 3 models, saves the best as `fraud_model.joblib` |
| `fraud_detector.py` | hybrid detector — combines the trained model + rule engine |
| `upi_fraudscope.py` | standalone rule-only detector (no ML dependency, fully interpretable) |
| `fraud_patterns.json` | cited regulatory context + scam categories + sample fraud-ID list |
| `requirements.txt` | dependencies |
| `sample_txn_fraud.json` / `sample_txn_safe.json` / `sample_txn_batch.json` | demo transactions for `upi_fraudscope.py` |

## What's real here

UPI fraud in India is a large, well-documented, rapidly growing problem.
The rules and thresholds are grounded in real public sources, not invented:

- `fraud_patterns.json` cites RBI circulars, RBI discussion-paper
  figures, NPCI advisories, and I4C public alerts for each rule — e.g.
  the "amount just below the UPI limit" rule is a documented evasion
  pattern, and the "₹1 verification payment" rule comes from a specific
  I4C Q2 2026 advisory on fake customer-care pages.
- There's no public, queryable database of scam UPI handles (unlike
  malware threat-intel feeds, banks/I4C don't publish fraud-account
  identifiers). So the "known fraud UPI ID" list is explicitly labeled
  as **illustrative, not a real database** — it demonstrates the rule
  engine, not a usable blocklist.
- Training data is synthetic, since real UPI transaction logs aren't
  public. `generate_data.py` simulates transactions using distributions
  that mirror publicly documented UPI fraud typologies — the standard
  approach in academic fraud-detection work when real data isn't
  accessible.

## Scope and honesty

This demonstrates a full detection *pipeline* — synthetic data
generation, model training/evaluation/selection, hybrid rule+ML
scoring, batch processing — built on real regulatory context. It does
not claim to be a production system: real deployments use far more
signals (device fingerprinting, behavioral biometrics, network-graph
analysis for mule accounts — see RBI's "MuleHunter.AI", cited in
`fraud_patterns.json`) than this pipeline captures.
