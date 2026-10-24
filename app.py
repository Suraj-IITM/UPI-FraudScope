import json
import os
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import fraud_detector as fd
from feature_store import InMemoryFeatureStore, RedisFeatureStore

LOG_PATH = os.getenv("SCORE_LOG", "scored_log.jsonl")
FEEDBACK_PATH = os.getenv("FEEDBACK_LOG", "feedback_log.jsonl")
_log_lock = threading.Lock()

if os.getenv("REDIS_URL"):
    import redis
    store = RedisFeatureStore(redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True))
    BACKEND = "redis"
else:
    store = InMemoryFeatureStore()
    BACKEND = "memory"

app = FastAPI(title="UPI FraudScope real-time API")


class PaymentRequest(BaseModel):
    sender_id: str
    recipient_upi: str
    amount: float = Field(gt=0)
    note: str = ""
    device_id: Optional[str] = None
    location: Optional[str] = None
    timestamp: Optional[float] = None
    sender_account_age_days: Optional[float] = None


class Feedback(BaseModel):
    txn_id: str
    is_fraud: bool


def _append(path, obj):
    with _log_lock, open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "model": fd.ML_NAME, "feature_store": BACKEND}


@app.post("/score")
def score(p: PaymentRequest):
    t0 = time.perf_counter()
    req = p.model_dump()
    req["timestamp"] = req["timestamp"] or time.time()

    feats = store.compute(req)                           # 1. features from history
    txn = dict(feats, upi=p.recipient_upi, note=p.note)  # 2. rules read upi + note
    try:
        result = fd.analyze_transaction(txn)             # 3. rules + ML
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    store.record(req, allow_learning=result["verdict"] != "BLOCK")  # 4. update history

    txn_id = uuid.uuid4().hex
    _append(LOG_PATH, {"txn_id": txn_id, "ts": req["timestamp"], **feats,
                       "final_score": result["final_score"], "verdict": result["verdict"]})
    return {
        "txn_id": txn_id,
        "verdict": result["verdict"],
        "action": result["action"],
        "final_score": result["final_score"],
        "ml_fraud_probability": result["ml_fraud_probability"],
        "rule_hits": result["rule_hits"],
        "features": feats,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.post("/feedback")
def feedback(fb: Feedback):
    _append(FEEDBACK_PATH, {"txn_id": fb.txn_id, "is_fraud": int(fb.is_fraud), "ts": time.time()})
    return {"saved": True}


@app.post("/reload")
def reload_model():
    return {"reloaded": fd.load_model()}
