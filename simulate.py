import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
    out["_rtt_ms"] = (time.perf_counter() - t0) * 1000
    return out


def ts(days_ago, hour):
    d = datetime.fromtimestamp(time.time() - days_ago * 86400).replace(hour=hour, minute=0, second=0)
    return d.timestamp()


def show(label, r):
    print(f"{label:<40} {r['verdict']:<8} score={r['final_score']:<6} ml={r['ml_fraud_probability']:<7} "
          f"hits={len(r['rule_hits'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    U = a.url

    # history: ravi has paid his landlord monthly for a year, from his own phone
    for d in (330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30):
        post(U, "/score", dict(sender_id="ravi", recipient_upi="ramesh.landlord@okhdfc", amount=8000,
                               note="monthly rent", device_id="ravi-phone", location="Delhi",
                               timestamp=ts(d, 10), sender_account_age_days=1500))
    for d in (50, 40, 20, 10):
        post(U, "/score", dict(sender_id="priya", recipient_upi="grocer@okaxis", amount=600,
                               note="groceries", device_id="priya-phone", location="Pune",
                               timestamp=ts(d, 18), sender_account_age_days=900))

    print("\n=== SCENARIOS ===")
    show("Ravi pays rent again", post(U, "/score", dict(
        sender_id="ravi", recipient_upi="ramesh.landlord@okhdfc", amount=8000, note="monthly rent",
        device_id="ravi-phone", location="Delhi", timestamp=ts(0, 10), sender_account_age_days=1500)))
    show("Priya -> KBC prize 'fee' (Rs 49,999)", post(U, "/score", dict(
        sender_id="priya", recipient_upi="claim.kbc.prize.2026@ybl", amount=49999,
        note="KBC Lottery Prize Processing Fee", device_id="priya-phone", location="Pune",
        timestamp=ts(0, 14), sender_account_age_days=900)))
    print("Stolen-phone attack on Ravi (new device, other city, 1 AM, rapid payments):")
    base = ts(0, 1)
    for i in range(6):
        show(f"   attempt {i+1} (Rs 49,800)", post(U, "/score", dict(
            sender_id="ravi", recipient_upi="unknown.receiver99@ybl", amount=49800, note="transfer",
            device_id="thief-phone", location="Mumbai", timestamp=base + i * 120,
            sender_account_age_days=1500)))

    print(f"\n=== LATENCY: {a.n} requests, {a.threads} threads ===")

    def one(i):
        return post(U, "/score", dict(sender_id=f"user{i % 50}", recipient_upi=f"shop{i % 20}@okaxis",
                                      amount=100 + (i * 37) % 4000, note="purchase",
                                      device_id=f"dev{i % 50}", location="Delhi",
                                      sender_account_age_days=500))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(one, range(a.n)))
    wall = time.perf_counter() - t0
    rtt = sorted(r["_rtt_ms"] for r in res)
    srv = sorted(r["latency_ms"] for r in res)
    pct = lambda xs, p: xs[min(len(xs) - 1, int(len(xs) * p))]
    print(f"throughput      : {a.n / wall:.0f} requests/sec")
    print(f"server-side ms  : p50={pct(srv,.5):.2f}  p95={pct(srv,.95):.2f}  p99={pct(srv,.99):.2f}")
    print(f"round-trip ms   : p50={pct(rtt,.5):.2f}  p95={pct(rtt,.95):.2f}  p99={pct(rtt,.99):.2f}")


if __name__ == "__main__":
    main()
