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
✅ python ✅ fintech  ✅ security

