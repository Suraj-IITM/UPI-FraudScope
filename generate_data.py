"""
generate_data.py
=================
Generates a SYNTHETIC UPI transaction dataset for training a fraud
detection model. This is NOT real bank data (real UPI transaction data
is never public, for obvious privacy/regulatory reasons). Instead, we
simulate transactions using distributions and patterns that mirror
publicly documented UPI fraud typologies (RBI/NPCI fraud reports,
news reports on KBC/digital-arrest/customs scams, etc.)

Why synthetic data is an acceptable and standard approach here:
  - Real UPI transaction datasets are confidential (banks/NPCI).
  - Academic fraud-detection papers commonly use synthetic or
    anonymized-and-perturbed data for this exact reason.
  - The goal at this stage is to demonstrate a working ML pipeline
    and evaluation methodology, which a professor can later extend
    with access to real/anonymized data via institutional partnerships.

Output: upi_transactions.csv  (~20,000 rows, ~6% fraud rate,
        which roughly mirrors reported UPI fraud incidence)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_TOTAL = 20000
FRAUD_RATE = 0.06          # ~6% fraud, roughly realistic for a flagged-transaction stream
N_FRAUD = int(N_TOTAL * FRAUD_RATE)
N_LEGIT = N_TOTAL - N_FRAUD

FRAUD_TYPES = [
    "lottery_prize", "customs_fine", "digital_arrest",
    "fake_job", "investment_scam", "account_takeover"
]

CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune",
          "Hyderabad", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]


def gen_legit(n):
    """Legitimate transactions: rent, groceries, friends, bills, shopping.
    Includes some 'hard' cases that resemble fraud signals (new online
    seller, first-time bill payment, occasional late-night order) so the
    classes are NOT trivially separable -- this mirrors real-world overlap
    between genuine and fraudulent behaviour."""
    df = pd.DataFrame({
        "amount": np.round(RNG.lognormal(mean=6.5, sigma=1.1, size=n), 2).clip(10, 90000),
        "hour": RNG.choice(range(24), size=n,
                            p=_hour_profile(peak=(9, 21))),
        "is_new_recipient": RNG.choice([0, 1], size=n, p=[0.78, 0.22]),
        "is_new_device": RNG.choice([0, 1], size=n, p=[0.93, 0.07]),
        "txns_last_hour": RNG.poisson(1.0, size=n).clip(0, 6),
        "location_jump": RNG.choice([0, 1], size=n, p=[0.95, 0.05]),
        "recipient_age_days": RNG.exponential(scale=200, size=n).clip(0.5, 3000),
        "sender_account_age_days": RNG.uniform(30, 3000, size=n),
        "contains_urgency_words": RNG.choice([0, 1], size=n, p=[0.92, 0.08]),
        "upi_id_suspicious_keyword": RNG.choice([0, 1], size=n, p=[0.94, 0.06]),
        "is_known_fraud_upi_db": np.zeros(n, dtype=int),
        "label": np.zeros(n, dtype=int),
        "fraud_type": ["none"] * n,
    })
    return df


def gen_fraud(n):
    """Fraudulent transactions, split across known UPI scam typologies."""
    types = RNG.choice(FRAUD_TYPES, size=n)
    rows = []
    for t in types:
        # NOTE: probabilities below are deliberately imperfect / noisy --
        # real fraudsters increasingly avoid obvious keywords and known IDs,
        # so a meaningful fraction of fraud rows carry NO obvious flag.
        # This keeps the classification task realistic instead of trivial.
        if t == "lottery_prize":
            amount = RNG.uniform(15000, 50000)
            hour = RNG.choice(range(9, 22))
            urgency, kw, fraud_db = RNG.choice([0, 1], p=[0.35, 0.65]), RNG.choice([0, 1], p=[0.55, 0.45]), RNG.choice([0, 1], p=[0.75, 0.25])
            new_dev, loc_jump, freq = 0, 0, RNG.integers(1, 3)
        elif t == "customs_fine":
            amount = RNG.uniform(30000, 100000)
            hour = RNG.choice(range(10, 20))
            urgency, kw, fraud_db = RNG.choice([0, 1], p=[0.3, 0.7]), RNG.choice([0, 1], p=[0.55, 0.45]), RNG.choice([0, 1], p=[0.7, 0.3])
            new_dev, loc_jump, freq = RNG.choice([0, 1], p=[0.7, 0.3]), 0, RNG.integers(1, 4)
        elif t == "digital_arrest":
            amount = RNG.uniform(40000, 150000)
            hour = RNG.choice(range(0, 24))
            urgency, kw, fraud_db = RNG.choice([0, 1], p=[0.35, 0.65]), RNG.choice([0, 1], p=[0.6, 0.4]), RNG.choice([0, 1], p=[0.75, 0.25])
            new_dev, loc_jump, freq = RNG.choice([0, 1], p=[0.6, 0.4]), 0, RNG.integers(2, 7)
        elif t == "fake_job":
            amount = RNG.uniform(1000, 7000)
            hour = RNG.choice(range(9, 22))
            urgency, kw, fraud_db = RNG.choice([0, 1], p=[0.4, 0.6]), RNG.choice([0, 1], p=[0.6, 0.4]), RNG.choice([0, 1], p=[0.8, 0.2])
            new_dev, loc_jump, freq = 0, 0, RNG.integers(1, 3)
        elif t == "investment_scam":
            amount = RNG.uniform(4000, 40000)
            hour = RNG.choice(range(9, 23))
            urgency, kw, fraud_db = RNG.choice([0, 1], p=[0.5, 0.5]), RNG.choice([0, 1], p=[0.6, 0.4]), RNG.choice([0, 1], p=[0.75, 0.25])
            new_dev, loc_jump, freq = 0, 0, RNG.integers(1, 3)
        else:  # account_takeover -- deliberately the hardest class: no keywords/urgency at all
            amount = RNG.uniform(8000, 50000)
            hour = RNG.choice(range(0, 6))
            urgency, kw, fraud_db = 0, 0, RNG.choice([0, 1], p=[0.8, 0.2])
            new_dev, loc_jump, freq = RNG.choice([0, 1], p=[0.2, 0.8]), RNG.choice([0, 1], p=[0.3, 0.7]), RNG.integers(2, 10)

        rows.append({
            "amount": round(amount, 2),
            "hour": hour,
            "is_new_recipient": int(RNG.choice([0, 1], p=[0.08, 0.92])),  # rare case: compromised known contact
            "is_new_device": new_dev,
            "txns_last_hour": freq,
            "location_jump": loc_jump,
            "recipient_age_days": float(np.clip(RNG.exponential(scale=15), 0, 60)),  # scam IDs usually young, but noisier
            "sender_account_age_days": RNG.uniform(30, 3000),
            "contains_urgency_words": urgency,
            "upi_id_suspicious_keyword": kw,
            "is_known_fraud_upi_db": fraud_db,
            "label": 1,
            "fraud_type": t,
        })
    return pd.DataFrame(rows)


def _hour_profile(peak=(9, 21)):
    """Weighted probability so legit transactions cluster in waking hours."""
    p = np.ones(24)
    p[peak[0]:peak[1]] = 4.0
    p[0:5] = 0.2
    return p / p.sum()


def main():
    legit = gen_legit(N_LEGIT)
    fraud = gen_fraud(N_FRAUD)
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df["txn_id"] = [f"TXN{100000+i}" for i in range(len(df))]
    df = df[["txn_id", "amount", "hour", "is_new_recipient", "is_new_device",
              "txns_last_hour", "location_jump", "recipient_age_days",
              "sender_account_age_days", "contains_urgency_words",
              "upi_id_suspicious_keyword", "is_known_fraud_upi_db",
              "fraud_type", "label"]]
    df.to_csv("upi_transactions.csv", index=False)
    print(f"Generated {len(df)} transactions -> upi_transactions.csv")
    print(f"Fraud rate: {df['label'].mean():.2%}")
    print(df["fraud_type"].value_counts())


if __name__ == "__main__":
    main()
