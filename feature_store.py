import threading
import time
import uuid
from collections import defaultdict, deque

from fraud_detector import (KNOWN_FRAUD_UPIIDS, SCAM_KEYWORDS_UPI)

SECONDS_PER_DAY = 86400
HOUR = 3600

URGENCY_WORDS = ["urgent", "immediately", "fine", "penalty", "fee", "arrest",
                 "bail", "blocked", "blocking", "court", "last chance",
                 "processing fee", "registration", "release", "clearance"]


def _text_features(req):
    upi = (req.get("recipient_upi") or "").lower()
    note = (req.get("note") or "").lower()
    return {
        "contains_urgency_words": int(any(w in note for w in URGENCY_WORDS)),
        "upi_id_suspicious_keyword": int(any(k in upi for k in SCAM_KEYWORDS_UPI)),
        "is_known_fraud_upi_db": int(upi in KNOWN_FRAUD_UPIIDS),
    }


def _hour_of(ts):
    return time.localtime(ts).tm_hour


class InMemoryFeatureStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.recent = defaultdict(deque)     # sender -> timestamps of recent attempts
        self.recipients = defaultdict(set)   # sender -> recipients already paid OK
        self.devices = defaultdict(set)      # sender -> trusted device ids
        self.home = {}                       # sender -> usual location
        self.sender_first_seen = {}
        self.upi_first_seen = {}             # recipient upi -> first timestamp (system-wide)

    def compute(self, req):
        now = req.get("timestamp") or time.time()
        s, r = req["sender_id"], req["recipient_upi"]
        with self._lock:
            q = self.recent[s]
            txns_last_hour = sum(1 for t in q if now - t <= HOUR) + 1  # +1 = this one
            is_new_recipient = int(r not in self.recipients[s])
            known_devices = self.devices[s]
            is_new_device = int(bool(known_devices) and req.get("device_id") not in known_devices)
            home = self.home.get(s)
            location_jump = int(home is not None and req.get("location") not in (None, home))
            sender_age = req.get("sender_account_age_days")
            if sender_age is None:
                first = self.sender_first_seen.get(s, now)
                sender_age = (now - first) / SECONDS_PER_DAY
            rec_first = self.upi_first_seen.get(r)
            recipient_age = 0.0 if rec_first is None else (now - rec_first) / SECONDS_PER_DAY

        feats = {
            "amount": float(req["amount"]),
            "hour": _hour_of(now),
            "is_new_recipient": is_new_recipient,
            "is_new_device": is_new_device,
            "txns_last_hour": txns_last_hour,
            "location_jump": location_jump,
            "recipient_age_days": recipient_age,
            "sender_account_age_days": float(sender_age),
        }
        feats.update(_text_features(req))
        return feats

    def record(self, req, allow_learning):
        """Always count the attempt (velocity). Only remember recipient/device/
        location as TRUSTED if the payment was not blocked."""
        now = req.get("timestamp") or time.time()
        s, r = req["sender_id"], req["recipient_upi"]
        with self._lock:
            q = self.recent[s]
            q.append(now)
            while q and now - q[0] > HOUR:
                q.popleft()
            self.sender_first_seen.setdefault(s, now)
            self.upi_first_seen.setdefault(r, now)
            if allow_learning:
                self.recipients[s].add(r)
                if req.get("device_id"):
                    self.devices[s].add(req["device_id"])
                if req.get("location"):
                    self.home.setdefault(s, req["location"])


class RedisFeatureStore:
    """Same logic, state in Redis. client = redis.Redis(decode_responses=True)."""

    def __init__(self, client):
        self.r = client

    def compute(self, req):
        now = req.get("timestamp") or time.time()
        s, rcp = req["sender_id"], req["recipient_upi"]
        pipe = self.r.pipeline()
        pipe.zcount(f"recent:{s}", now - HOUR, "+inf")
        pipe.sismember(f"recipients:{s}", rcp)
        pipe.scard(f"devices:{s}")
        pipe.sismember(f"devices:{s}", req.get("device_id") or "")
        pipe.get(f"home:{s}")
        pipe.get(f"sender_first:{s}")
        pipe.get(f"upi_first:{rcp}")
        n_recent, known_rcp, n_dev, dev_known, home, s_first, r_first = pipe.execute()

        sender_age = req.get("sender_account_age_days")
        if sender_age is None:
            sender_age = (now - float(s_first)) / SECONDS_PER_DAY if s_first else 0.0
        feats = {
            "amount": float(req["amount"]),
            "hour": _hour_of(now),
            "is_new_recipient": int(not known_rcp),
            "is_new_device": int(n_dev > 0 and not dev_known),
            "txns_last_hour": int(n_recent) + 1,
            "location_jump": int(home is not None and req.get("location") not in (None, home)),
            "recipient_age_days": (now - float(r_first)) / SECONDS_PER_DAY if r_first else 0.0,
            "sender_account_age_days": float(sender_age),
        }
        feats.update(_text_features(req))
        return feats

    def record(self, req, allow_learning):
        now = req.get("timestamp") or time.time()
        s, rcp = req["sender_id"], req["recipient_upi"]
        pipe = self.r.pipeline()
        pipe.zadd(f"recent:{s}", {f"{now}:{uuid.uuid4().hex}": now})
        pipe.zremrangebyscore(f"recent:{s}", "-inf", now - HOUR)
        pipe.expire(f"recent:{s}", HOUR * 2)
        pipe.setnx(f"sender_first:{s}", now)
        pipe.setnx(f"upi_first:{rcp}", now)
        if allow_learning:
            pipe.sadd(f"recipients:{s}", rcp)
            if req.get("device_id"):
                pipe.sadd(f"devices:{s}", req["device_id"])
            if req.get("location"):
                pipe.setnx(f"home:{s}", req["location"])
        pipe.execute()
