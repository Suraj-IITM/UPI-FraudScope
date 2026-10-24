import argparse
import os
import shutil
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from train_model import FEATURES

MODEL_PATH = "fraud_model.joblib"


def read_jsonl(path):
    return pd.read_json(path, lines=True) if os.path.exists(path) else pd.DataFrame()


def load_live(assume_legit_after_days):
    log = read_jsonl(os.getenv("SCORE_LOG", "scored_log.jsonl"))
    fb = read_jsonl(os.getenv("FEEDBACK_LOG", "feedback_log.jsonl"))
    if log.empty:
        return pd.DataFrame()
    df = log.merge(fb[["txn_id", "is_fraud"]], on="txn_id", how="left") if not fb.empty \
        else log.assign(is_fraud=np.nan)
    if assume_legit_after_days is not None:
        age_days = (pd.Timestamp.now().timestamp() - df["ts"]) / 86400
        df.loc[df["is_fraud"].isna() & (age_days >= assume_legit_after_days), "is_fraud"] = 0
    df = df.dropna(subset=["is_fraud"])
    df["label"] = df["is_fraud"].astype(int)
    return df[FEATURES + ["label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-weight", type=float, default=5.0)
    ap.add_argument("--assume-legit-after-days", type=float, default=None)
    ap.add_argument("--min-real-fraud", type=int, default=20)
    ap.add_argument("--reload-url", default=None)
    a = ap.parse_args()

    base = pd.read_csv("upi_transactions.csv")[FEATURES + ["label"]]
    base["w"] = 1.0
    live = load_live(a.assume_legit_after_days)
    n_fraud = int(live["label"].sum()) if len(live) else 0
    print(f"synthetic rows: {len(base)} | labelled live rows: {len(live)} (fraud: {n_fraud})")
    if n_fraud < a.min_real_fraud:
        print(f"Not enough real fraud labels (<{a.min_real_fraud}). Skipping retrain.")
        return
    live = live.assign(w=a.real_weight)
    data = pd.concat([base, live], ignore_index=True)

    X, y, w = data[FEATURES], data["label"], data["w"]
    Xtr, Xte, ytr, yte, wtr, _ = train_test_split(X, y, w, test_size=0.25, stratify=y, random_state=42)

    new = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr, sample_weight=wtr)
    new_ap = average_precision_score(yte, new.predict_proba(Xte)[:, 1])

    cur = joblib.load(MODEL_PATH)
    Xc = cur["scaler"].transform(Xte) if cur["scaler"] is not None else Xte
    cur_ap = average_precision_score(yte, cur["model"].predict_proba(Xc)[:, 1])
    print(f"PR-AUC  current: {cur_ap:.4f} | new: {new_ap:.4f}")

    if new_ap < cur_ap:
        print("New model is worse. Keeping the current one.")
        return

    shutil.copy(MODEL_PATH, "fraud_model.prev.joblib")
    joblib.dump({"model": new, "model_name": "Gradient Boosting (retrained)",
                 "scaler": None, "features": FEATURES}, MODEL_PATH + ".tmp")
    os.replace(MODEL_PATH + ".tmp", MODEL_PATH)
    print("Saved new model. Old model kept as fraud_model.prev.joblib")

    if a.reload_url:
        print(urllib.request.urlopen(urllib.request.Request(a.reload_url, method="POST")).read().decode())


if __name__ == "__main__":
    main()
