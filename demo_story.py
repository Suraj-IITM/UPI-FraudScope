"""
demo_story.py -- realistic story demo for UPI FraudScope

    python demo_story.py

Plays four short, realistic stories (a normal day, a "digital arrest" scam,
a stolen-phone takeover, a fake job offer) through the real-time API, then lets
you type your own payments with time of day, device and city.

It starts its OWN fresh API on a free port each run, so the results are the
same every time (no leftover history from earlier practice runs).

Needs: pip install -r requirements.txt, and python generate_data.py +
python train_model.py already run (so fraud_model.joblib exists).
"""
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime

import requests

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:          # pasted into a notebook cell: stay in the current folder
    pass

if not os.path.exists("fraud_model.joblib"):
    sys.exit("fraud_model.joblib not found. Run: python generate_data.py && python train_model.py")

# ───────────────────────── start a fresh API ─────────────────────────
s = socket.socket(); s.bind(("127.0.0.1", 0)); PORT = s.getsockname()[1]; s.close()
BASE = f"http://127.0.0.1:{PORT}"
if "_story_srv" in globals() and _story_srv.poll() is None:
    _story_srv.terminate()
log = open("server_story.log", "w")
_story_srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT)],
                              stdout=log, stderr=log, start_new_session=True)
for _ in range(60):
    time.sleep(1)
    try:
        requests.get(BASE + "/health", timeout=1); break
    except Exception:
        pass
else:
    print("API did not start. Log:\n", open("server_story.log").read()[-1500:])
    _story_srv.terminate(); sys.exit(1)


# ───────────────────────── helpers ─────────────────────────
def ts(days_ago, hour, minute=0):
    d = datetime.fromtimestamp(time.time() - days_ago * 86400)
    return d.replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp()


def pay(**p):
    r = requests.post(BASE + "/score", json=p)
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:200]}")
    return r.json()


ICON = {"SAFE": "✅", "CAUTION": "🟡", "WARN": "🟠", "BLOCK": "⛔"}
MEANING = {"SAFE": "payment allowed", "CAUTION": "extra OTP asked",
           "WARN": "held for manual review", "BLOCK": "payment blocked"}
RULE_TEXT = {
    "Just Below Reporting Limit (₹48K-50K / ₹98K-1L)": "amount sits just below a reporting limit",
    "Suspicious UPI ID Keywords": "ID contains scam-style keywords",
    "Scam Payment Note Detected": "note matches a known scam phrase",
    "UPI ID in Reported-Fraud Database": "ID is in the reported-fraud list",
}


def why(r):
    f, out = r["features"], []
    if f["is_new_recipient"]:
        out.append("first payment to this recipient")
    if f["is_new_device"]:
        out.append("unrecognised device")
    if f["location_jump"]:
        out.append("different city than usual")
    if f["txns_last_hour"] >= 2:
        out.append(f"{f['txns_last_hour']} payments in the last hour")
    if f["hour"] < 6:
        out.append("late-night hour")
    if f.get("amount_vs_usual", 1) >= 5:
        out.append(f"{f['amount_vs_usual']:.0f}x larger than this customer's usual payment")
    elif f["amount"] >= 40000:
        out.append("large amount")
    out += [RULE_TEXT.get(h, h) for h in r["rule_hits"]]
    return out or ["no warning signs"]


def show(r):
    v = r["verdict"]
    print(f"      {ICON.get(v, '')} {v}  -  {MEANING.get(v, '')}   (risk {r['final_score']}/100, "
          f"model {r['ml_fraud_probability']:.1%}, {r['latency_ms']} ms)")
    print("      signals: " + " · ".join(why(r)))


PERSONAS = {  # sender_id: (device, usual city, account age in days)
    "meera.iyer": ("meera-samsung", "Bengaluru", 2500),
    "karan.mehta": ("karan-iphone", "Pune", 1100),
    "ananya.rao": ("ananya-redmi", "Vellore", 400),
}


def send(sender, upi, amount, note, when, device=None, city=None):
    dev, home, age = PERSONAS.get(sender, (sender + "-phone", "Delhi", 365))
    r = pay(sender_id=sender, recipient_upi=upi, amount=amount, note=note,
            device_id=device or dev, location=city or home, timestamp=when,
            sender_account_age_days=age)
    return r


def pause(msg="Press Enter for the next step..."):
    input(f"\n   ▶ {msg}")


def step(title, text):
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}\n{text}")


# ───────────────────────── build past history ─────────────────────────
print("Preparing customer history (a year of ordinary payments)...")
for d in range(330, 0, -30):
    send("meera.iyer", "lakshmi.k1974@okicici", 3000, "maid salary", ts(d, 10))
    send("meera.iyer", "bescom.bill@okaxis", 1450, "electricity", ts(d - 1, 11))
    send("meera.iyer", "freshmart.indiranagar@ybl", 900, "groceries", ts(d - 2, 17))
for d in range(330, 0, -15):
    send("karan.mehta", "swiggy.payments@okicici", 450, "food", ts(d, 20))
    send("karan.mehta", "pg.owner.sunil@oksbi", 14000, "rent", ts(d - 1, 9))
for d in (60, 30, 10):
    send("ananya.rao", "canteen.vit@ybl", 120, "lunch", ts(d, 13))
print("Ready.\n")

try:
    # ───────────────────────── STORY 1 ─────────────────────────
    step("STORY 1 - An ordinary day",
         "Meera Iyer (61, retired teacher, Bengaluru) and Karan Mehta (28, engineer, Pune)\n"
         "make everyday payments from their own phones.")
    print("\n   Meera pays her maid Rs 3,000 (usual recipient, usual phone, 10:05 AM)")
    show(send("meera.iyer", "lakshmi.k1974@okicici", 3000, "maid salary", ts(1, 10, 5)))
    print("\n   Karan splits a dinner bill, Rs 1,200, with a friend he has not paid before (9 PM)")
    show(send("karan.mehta", "neha.s92@okaxis", 1200, "dinner split", ts(1, 21)))
    pause()

    # ───────────────────────── STORY 2 ─────────────────────────
    step("STORY 2 - 'Digital arrest' scam",
         "A caller poses as a CBI officer, says Meera is under 'digital arrest' and must\n"
         "transfer money 'for verification'. Nothing is hacked: Meera uses her own phone\n"
         "and her own PIN. The recipient ID looks like an ordinary person.")
    print("\n   2:20 PM - Meera sends Rs 98,500 to rohit.verma7741@okicici")
    show(send("meera.iyer", "rohit.verma7741@okicici", 98500, "documents", ts(0, 14, 20)))
    print("\n   2:41 PM - the caller asks for a second transfer, Rs 49,500, to the same account")
    show(send("meera.iyer", "rohit.verma7741@okicici", 49500, "documents", ts(0, 14, 41)))
    pause()

    # ───────────────────────── STORY 3 ─────────────────────────
    step("STORY 3 - Stolen phone / SIM swap",
         "At 2 AM a thief, on a different device and in Delhi, tries to drain Karan's\n"
         "account (Karan lives in Pune). The recipient is just a phone-number UPI ID.")
    for m in (7, 9, 11):
        print(f"\n   2:{m:02d} AM - Rs 49,000 to 9876501234@ybl from an unknown device in Delhi")
        show(send("karan.mehta", "9876501234@ybl", 49000, "", ts(0, 2, m),
                  device="unknown-device-7f3a", city="Delhi"))
    pause()

    # ───────────────────────── STORY 4 ─────────────────────────
    step("STORY 4 - Fake job offer",
         "Ananya (22, student) is told to pay a small 'registration fee' for a\n"
         "work-from-home job she found on Telegram.")
    print("\n   4:00 PM - Rs 2,999 to hrdesk.talentbridge@ybl, note: 'Job registration fee'")
    show(send("ananya.rao", "hrdesk.talentbridge@ybl", 2999, "Job registration fee", ts(0, 16)))
    pause()

    # ───────────────────────── STORY 5 ─────────────────────────
    step("STORY 5 - A large payment that is not clearly a scam",
         "Back home, Karan buys a used bike and pays the seller Rs 75,000 (first time paying\n"
         "this person). The system does not block it, but it does not wave it through either.")
    print("\n   5:00 PM - Karan sends Rs 75,000 to bike.seller.amit@okaxis from his own phone in Pune")
    show(send("karan.mehta", "bike.seller.amit@okaxis", 75000, "bike", ts(0, 17)))
    print("\n   (A real bank would ask for an extra check, such as a call or OTP, before releasing it.)")

    # ───────────────────────── FREE MODE ─────────────────────────
    print(f"\n{'═' * 78}\nTRY YOUR OWN PAYMENT\n{'═' * 78}")
    print("Customers with history: meera.iyer, karan.mehta, ananya.rao   (any other name = new customer)")
    print("Press Enter on 'Who is paying' to finish.\n")
    last_ts = {}
    while True:
        sender = input("Who is paying: ").strip().lower()
        if not sender:
            break
        upi = input("Recipient UPI ID (e.g. name@okaxis): ").strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{2,64}@[A-Za-z][A-Za-z0-9]{1,31}", upi):
            print("   That is not a valid UPI ID (expected name@bank). Try again.\n"); continue
        try:
            amount = float(input("Amount in Rs (1 to 1,00,000): "))
        except ValueError:
            print("   Amount must be a number.\n"); continue
        if not 1 <= amount <= 100000:
            print("   UPI allows Rs 1 to Rs 1,00,000 per payment. Try again.\n"); continue
        note = input("Note (optional): ").strip()[:100]
        t = input("Time of day, 24h HH:MM (Enter = 11:00): ").strip() or "11:00"
        m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", t)
        if not m:
            print("   Use HH:MM, for example 02:30 or 14:45.\n"); continue
        new_dev = input("From an unrecognised device? (y/N): ").strip().lower().startswith("y")
        city = input("City (Enter = usual city): ").strip() or None
        # free-mode payments happen "tomorrow"; if a customer's previous payment is later
        # than the time typed, move this one to the following day so time never runs backwards
        when = ts(-1, int(m.group(1)), int(m.group(2)))
        while when <= last_ts.get(sender, 0):
            when += 86400
        last_ts[sender] = when
        r = send(sender, upi, amount, note, when,
                 device="unrecognised-device" if new_dev else None, city=city)
        print()
        show(r)
        print()
finally:
    _story_srv.terminate()
    print("\nDemo finished. API stopped.")
