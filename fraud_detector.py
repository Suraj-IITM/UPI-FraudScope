"""
fraud_detector.py
==================
HYBRID UPI FRAUD DETECTOR — Rules Engine + Trained ML Model

Why hybrid, not pure ML:
  - Rules give instant, explainable, zero-latency flags for KNOWN patterns
    (e.g. a UPI ID already reported to cybercrime.gov.in) -- no model
    training needed, updates instantly when new scam IDs are reported.
  - ML generalizes to NEW/unseen fraud patterns that don't match any
    hardcoded rule, and gives a continuous risk score instead of a
    brittle yes/no.
  - This mirrors how real bank/NPCI fraud systems work in practice:
    a fast rules layer for known bad actors + a statistical model layer
    for anomaly detection.

FINAL_SCORE = w_rules * rule_score + w_ml * ml_probability * 100
"""

import json
import os
import threading
import warnings

import joblib
import numpy as np

MODEL_PATH = "fraud_model.joblib"
PATTERNS_PATH = "fraud_patterns.json"

_lock = threading.Lock()
ML_MODEL = ML_SCALER = ML_FEATURES = ML_NAME = None


def load_model(path=MODEL_PATH):
    """Load (or hot-reload) the trained model. Safe to call while serving."""
    global ML_MODEL, ML_SCALER, ML_FEATURES, ML_NAME
    artifact = joblib.load(path)
    with _lock:
        ML_MODEL = artifact["model"]
        ML_SCALER = artifact["scaler"]
        ML_FEATURES = artifact["features"]
        ML_NAME = artifact["model_name"]
    return ML_NAME


try:
    load_model()
except FileNotFoundError:
    raise SystemExit("fraud_model.joblib not found. Run train_model.py first.")


def load_known_fraud_ids(path=PATTERNS_PATH):
    """Single source of truth: fraud_patterns.json (no duplicate hardcoded list)."""
    with open(path) as f:
        return set(json.load(f)["known_fraud_upi_ids_sample"]["ids"])

# ── Rule engine (kept from original prototype, still useful for
#    known/reported fraud UPI IDs and scam keyword matching) ────────
SCAM_KEYWORDS_UPI = ["prize", "lottery", "winner", "reward", "free", "lucky",
                      "claim", "customs", "rescue", "helpdesk", "support", "refund", "kbc",
                      "hrdesk", "recruit", "fedex", "parcel"]
SCAM_NOTES = ["kbc", "lottery", "prize", "customs clearance", "digital arrest",
              "fedex", "income tax", "blocking your account", "court order",
              "job registration", "registration fee", "processing fee",
              "clearance fee", "verification payment", "trading profit",
              "investment"]
KNOWN_FRAUD_UPIIDS = load_known_fraud_ids()

RULES = {
    "Just Below Reporting Limit (₹48K-50K / ₹98K-1L)":
        lambda t: (48000 <= t["amount"] <= 49999) or (98000 <= t["amount"] <= 99999),
    "Suspicious UPI ID Keywords":
        lambda t: any(k in t.get("upi", "").lower() for k in SCAM_KEYWORDS_UPI),
    "Scam Payment Note Detected":
        lambda t: any(k in t.get("note", "").lower() for k in SCAM_NOTES),
    "UPI ID in Reported-Fraud Database":
        lambda t: t.get("upi", "") in KNOWN_FRAUD_UPIIDS,
}
RULE_SCORES = {
    "Just Below Reporting Limit (₹48K-50K / ₹98K-1L)": 25,
    "Suspicious UPI ID Keywords": 30,
    "Scam Payment Note Detected": 35,
    "UPI ID in Reported-Fraud Database": 45,
}

# Weighting: ML carries more weight since it's evaluated (PR-AUC ~0.99);
# rules act as a fast override for definitively known bad actors.
W_RULES = 0.35
W_ML = 0.65


def apply_rules(txn: dict):
    hits, score = [], 0
    for rule, check in RULES.items():
        if check(txn):
            hits.append(rule)
            score += RULE_SCORES[rule]
    return hits, min(100, score)


def ml_predict(txn: dict):
    """txn must contain every feature in ML_FEATURES.
    Missing features raise an error instead of silently becoming 0 --
    in a live system a silent 0 would quietly LOWER the fraud score."""
    with _lock:
        model, scaler, features = ML_MODEL, ML_SCALER, ML_FEATURES
    missing = [f for f in features if f not in txn]
    if missing:
        raise ValueError(f"Missing ML features: {missing}")
    X = np.array([[float(txn[f]) for f in features]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy input vs model fitted on a DataFrame
        if scaler is not None:
            X = scaler.transform(X)
        return float(model.predict_proba(X)[0, 1])


def verdict(score):
    if score >= 70:
        return "BLOCK", "HIGH FRAUD RISK — transaction blocked"
    if score >= 45:
        return "WARN", "SUSPICIOUS — manual review required"
    if score >= 20:
        return "CAUTION", "VERIFY RECIPIENT — extra OTP step required"
    return "SAFE", "LOW RISK — transaction allowed"


def analyze_transaction(txn: dict):
    """
    txn: dict with keys — amount, amount_vs_usual, hour, is_new_recipient, is_new_device,
    txns_last_hour, location_jump, recipient_age_days,
    sender_account_age_days, contains_urgency_words,
    upi_id_suspicious_keyword, is_known_fraud_upi_db,
    plus optional 'upi' and 'note' strings for the rule engine.
    """
    rule_hits, rule_score = apply_rules(txn)
    ml_proba = ml_predict(txn)
    ml_score = ml_proba * 100

    final_score = round(W_RULES * rule_score + W_ML * ml_score, 1)
    # Corroboration floor: two independent scam rules firing together (e.g. a
    # scam-style ID AND a scam-style note) is strong evidence even when the
    # amount is tiny and the model is unsure -- never let it pass as SAFE.
    if len(rule_hits) >= 2:
        final_score = max(final_score, 45.0)
    # Hard override: a UPI ID already confirmed fraudulent always blocks,
    # regardless of what the model says.
    if txn.get("upi", "") in KNOWN_FRAUD_UPIIDS:
        final_score = 100.0

    # Step-up policy: a high-value payment to someone never paid before is not
    # a scam by itself, but it should never pass without at least an extra check.
    if txn.get("amount", 0) >= 50000 and txn.get("is_new_recipient"):
        final_score = max(final_score, 20.0)

    label, action = verdict(final_score)

    return {
        "rule_hits": rule_hits,
        "rule_score": rule_score,
        "ml_model": ML_NAME,
        "ml_fraud_probability": round(ml_proba, 4),
        "final_score": final_score,
        "verdict": label,
        "action": action,
    }


# ── Demo: original 7 scam scenarios, now scored by the hybrid system ──
DEMO_TXNS = {
    "KBC Lottery Scam": {
        "amount": 49999, "hour": 14, "is_new_recipient": 1, "is_new_device": 0,
        "amount_vs_usual": 8,
        "txns_last_hour": 1, "location_jump": 0, "recipient_age_days": 2,
        "sender_account_age_days": 900, "contains_urgency_words": 1,
        "upi_id_suspicious_keyword": 1, "is_known_fraud_upi_db": 0,
        "upi": "claim.kbc.prize.2026@ybl", "note": "KBC Lottery Prize Processing Fee",
    },
    "FedEx / Customs Scam": {
        "amount": 85000, "hour": 15, "is_new_recipient": 1, "is_new_device": 1,
        "amount_vs_usual": 20,
        "txns_last_hour": 2, "location_jump": 0, "recipient_age_days": 1,
        "sender_account_age_days": 500, "contains_urgency_words": 1,
        "upi_id_suspicious_keyword": 1, "is_known_fraud_upi_db": 1,
        "upi": "customs.helpdesk@upi", "note": "customs clearance fine urgent",
    },
    "Digital Arrest Scam": {
        "amount": 99500, "hour": 3, "is_new_recipient": 1, "is_new_device": 0,
        "amount_vs_usual": 40,
        "txns_last_hour": 5, "location_jump": 0, "recipient_age_days": 1,
        "sender_account_age_days": 1200, "contains_urgency_words": 1,
        "upi_id_suspicious_keyword": 0, "is_known_fraud_upi_db": 1,
        "upi": "digitalrescue@ybl", "note": "digital arrest bail payment urgent",
    },
    "Fake Job Offer Scam": {
        "amount": 2999, "hour": 11, "is_new_recipient": 1, "is_new_device": 0,
        "amount_vs_usual": 15,
        "txns_last_hour": 1, "location_jump": 0, "recipient_age_days": 5,
        "sender_account_age_days": 200, "contains_urgency_words": 1,
        "upi_id_suspicious_keyword": 0, "is_known_fraud_upi_db": 1,
        "upi": "jobs.registration@paytm", "note": "Google job registration fee",
    },
    "Investment / Trading Scam": {
        "amount": 15000, "hour": 13, "is_new_recipient": 1, "is_new_device": 0,
        "amount_vs_usual": 6,
        "txns_last_hour": 1, "location_jump": 0, "recipient_age_days": 10,
        "sender_account_age_days": 700, "contains_urgency_words": 0,
        "upi_id_suspicious_keyword": 0, "is_known_fraud_upi_db": 1,
        "upi": "trading.profit@okaxis", "note": "trading group membership profit",
    },
    "Stolen Phone Account Takeover": {
        "amount": 49800, "hour": 1, "is_new_recipient": 1, "is_new_device": 1,
        "amount_vs_usual": 12,
        "txns_last_hour": 7, "location_jump": 1, "recipient_age_days": 30,
        "sender_account_age_days": 800, "contains_urgency_words": 0,
        "upi_id_suspicious_keyword": 0, "is_known_fraud_upi_db": 0,
        "upi": "unknown.receiver99@ybl", "note": "transfer",
    },
    "Legitimate Rent Payment (control)": {
        "amount": 8000, "hour": 10, "is_new_recipient": 0, "is_new_device": 0,
        "amount_vs_usual": 1,
        "txns_last_hour": 1, "location_jump": 0, "recipient_age_days": 400,
        "sender_account_age_days": 1500, "contains_urgency_words": 0,
        "upi_id_suspicious_keyword": 0, "is_known_fraud_upi_db": 0,
        "upi": "ramesh.landlord@okhdfc", "note": "monthly rent June 2026",
    },
}


def run_demo():
    print(f"\nUsing ML model: {ML_NAME}  (rules weight={W_RULES}, ML weight={W_ML})\n")
    print(f"{'Scenario':<35}{'Rules':<8}{'ML Prob':<10}{'Final':<8}{'Verdict':<10}Action")
    print("-" * 110)
    for name, txn in DEMO_TXNS.items():
        r = analyze_transaction(txn)
        print(f"{name:<35}{r['rule_score']:<8}{r['ml_fraud_probability']:<10}"
              f"{r['final_score']:<8}{r['verdict']:<10}{r['action']}")
        if r["rule_hits"]:
            print(f"    Rule flags: {', '.join(r['rule_hits'])}")
    print()


if __name__ == "__main__":
    run_demo()
