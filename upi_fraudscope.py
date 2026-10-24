"""
UPI FraudScope
===============
A rule-based tool that checks UPI transactions for signs of fraud and
gives a risk score from 0-100. It can check one transaction at a time
or a batch of several, and explains which scam pattern a flagged
transaction looks like.

USAGE
------
    python upi_fraudscope.py <transactions.json> [options]

    Options:
      --batch             treat the input file as a list of transactions
      --export json       also save the report as fraud_report.json
      --export csv        also save results as fraud_alerts.csv
      --patterns PATH     use a different fraud_patterns.json file
      --quiet             skip the rule-by-rule output while scanning

    Examples:
      python upi_fraudscope.py sample_txn_fraud.json
      python upi_fraudscope.py sample_txn_batch.json --batch --export csv

CHANGELOG
----------
  v1.0 - basic rule engine, checks one transaction at a time
  v1.1 - added batch mode for checking many transactions together
  v2.0 - moved the fraud rules/thresholds into fraud_patterns.json
         instead of hardcoding them, and added sources for each one
  v2.1 - added command line arguments
  v2.2 - added JSON/CSV export
  v2.3 - added scam category matching (tells you which scam type it
         looks like, not just that it's risky)
  v2.4 - added basic error handling for bad input files
"""

import argparse
import csv
import datetime
import json
import os
import sys


def load_fraud_patterns(path):
    """Load fraud_patterns.json. Exits with a clear message on failure
    rather than crashing deep inside a rule function."""
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Fraud patterns file not found: {path}")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] Fraud patterns file is not valid JSON: {path}\n        {e}")
    return data


class UPIFraudScope:
    def __init__(self, patterns):
        self.patterns = patterns
        self.known_fraud_upi_ids = set(patterns['known_fraud_upi_ids_sample']['ids'])
        self.scam_categories = patterns['scam_categories']

        # build keyword -> category dict once so we don't have to loop
        # through all categories again for every single transaction
        self.keyword_to_category = {}
        for cat in self.scam_categories:
            for kw in cat['typical_keywords']:
                self.keyword_to_category[kw.lower()] = cat['name']

        self.rules = [
            ("Unusual Transaction Time",      self._check_time,           15),
            ("High Value Transaction",        self._check_amount,         20),
            ("New Recipient",                 self._check_new_recipient,  20),
            ("Amount Just Below UPI Limit",   self._check_round_limit,    25),
            ("High Transaction Frequency",    self._check_frequency,      20),
            ("Unrecognized Device",           self._check_device,         25),
            ("Location Mismatch",             self._check_location,      20),
            ("Suspicious UPI ID Keyword",     self._check_upi_id,         35),
            ("Known Scam Payment Note",       self._check_note,           40),
            ("Known Fraud UPI ID (sample db)", self._check_fraud_db,      50),
        ]

    # ── SINGLE TRANSACTION ──────────────────────
    def analyze(self, transaction, quiet=False):
        if not quiet:
            self._print_transaction(transaction)
            print("[*] Running fraud detection rules...\n")

        alerts = []
        risk_score = 0
        for rule_name, rule_func, max_score in self.rules:
            result = rule_func(transaction)
            if not quiet:
                status = "HIT " if result else "OK  "
                print(f"  [{status}] {rule_name}")
            if result:
                result['rule'] = rule_name
                result['max_score'] = max_score
                alerts.append(result)
                risk_score += result['score']

        risk_score = min(100, risk_score)
        category_match = self._match_scam_category(transaction, alerts)
        return self._build_result(transaction, alerts, risk_score, category_match)

    # ── BATCH ────────────────────────────────────
    def analyze_batch(self, transactions, quiet=False):
        results = []
        for txn in transactions:
            results.append(self.analyze(txn, quiet=True))

        if not quiet:
            print(f"[*] Batch analysis: {len(transactions)} transactions\n")
            print(f"  {'#':<4} {'Amount':>10}  {'Recipient UPI':<32}  {'Risk':>5}  Verdict")
            print(f"  {'-'*4} {'-'*10}  {'-'*32}  {'-'*5}  {'-'*22}")
            for i, r in enumerate(results, 1):
                amt = r['transaction'].get('amount', 0)
                upi = r['transaction'].get('recipient_upi', '?')
                print(f"  {i:<4} Rs.{amt:>9,}  {upi:<32}  {r['risk_score']:>4}  {r['verdict']}")
            print()
        return results

    # ── FRAUD RULES ─────────────────────────────
    # Most of these thresholds are just heuristics (a guess based on what
    # seemed reasonable). The ones based on a real advisory say so in the
    # comment, and the full source is listed in fraud_patterns.json.
    def _check_time(self, t):
        try:
            hour = datetime.datetime.strptime(t['timestamp'], '%Y-%m-%d %H:%M:%S').hour
            if hour < 6 or hour >= 23:
                return {'detail': f"Transaction at {hour:02d}:00 — unusual hour",
                        'score': 15, 'severity': 'MEDIUM'}
        except Exception:
            pass

    def _check_amount(self, t):
        amt = t.get('amount', 0)
        if amt > 50000:
            return {'detail': f"Very high amount: Rs.{amt:,}", 'score': 20, 'severity': 'HIGH'}
        elif amt > 25000:
            return {'detail': f"High amount: Rs.{amt:,}", 'score': 10, 'severity': 'MEDIUM'}

    def _check_new_recipient(self, t):
        if t.get('is_new_recipient', False):
            return {'detail': "First-ever payment to this UPI ID — no prior transaction history",
                     'score': 20, 'severity': 'HIGH'}

    def _check_round_limit(self, t):
        # staying just under a transaction limit is a known trick to
        # avoid the extra verification that kicks in above the limit
        amt = t.get('amount', 0)
        if 48000 <= amt <= 49999:
            return {'detail': f"Rs.{amt:,} — just below Rs.50,000 (a documented limit-evasion pattern)",
                     'score': 25, 'severity': 'HIGH'}
        if 98000 <= amt <= 99999:
            return {'detail': f"Rs.{amt:,} — just below Rs.1,00,000 (a documented limit-evasion pattern)",
                     'score': 25, 'severity': 'HIGH'}

    def _check_frequency(self, t):
        count = t.get('transactions_last_hour', 0)
        if count > 3:
            return {'detail': f"{count} transactions in the last 60 minutes — unusually high frequency",
                     'score': 20, 'severity': 'HIGH'}

    def _check_device(self, t):
        if t.get('new_device', False):
            device = t.get('device_name', 'Unknown device')
            return {'detail': f"Payment from a new, unrecognized device: {device}",
                     'score': 25, 'severity': 'CRITICAL'}

    def _check_location(self, t):
        if t.get('location_mismatch', False):
            home, current = t.get('home_city', '?'), t.get('current_city', '?')
            return {'detail': f"Location mismatch — home: {home}, current: {current}",
                     'score': 20, 'severity': 'HIGH'}

    def _check_upi_id(self, t):
        upi = t.get('recipient_upi', '').lower()
        for kw, category in self.keyword_to_category.items():
            if kw in upi:
                return {'detail': f"UPI ID contains a keyword associated with '{category}': {t['recipient_upi']}",
                         'score': 35, 'severity': 'CRITICAL'}

    def _check_note(self, t):
        note = t.get('payment_note', '').lower()
        for kw, category in self.keyword_to_category.items():
            if kw in note:
                return {'detail': f"Payment note matches a known '{category}' pattern: \"{t.get('payment_note')}\"",
                         'score': 40, 'severity': 'CRITICAL'}

    def _check_fraud_db(self, t):
        upi = t.get('recipient_upi', '')
        if upi in self.known_fraud_upi_ids:
            return {'detail': f"UPI ID matches the sample fraud-pattern list: {upi}",
                     'score': 50, 'severity': 'CRITICAL'}

    # ── SCAM CATEGORY MATCHING ───────────────────
    def _match_scam_category(self, t, alerts):
        """checks which scam category (from fraud_patterns.json) this
        transaction looks closest to, so we can tell the user *why* it
        seems risky, not just that it does"""
        text = f"{t.get('recipient_upi','')} {t.get('payment_note','')}".lower()
        for cat in self.scam_categories:
            if any(kw in text for kw in cat['typical_keywords']):
                return cat
        return None

    # ── RESULT / VERDICT ─────────────────────────
    def _verdict(self, score):
        if score >= 70:
            return "BLOCK — high fraud risk", "Transaction blocked"
        elif score >= 45:
            return "WARN — suspicious", "Manual review required"
        elif score >= 20:
            return "CAUTION — verify recipient", "Extra OTP / step-up auth required"
        return "SAFE — low fraud risk", "Transaction allowed"

    def _build_result(self, transaction, alerts, risk_score, category_match):
        verdict, action = self._verdict(risk_score)
        return {
            'transaction': transaction,
            'alerts': sorted(alerts, key=lambda a: -a['score']),
            'risk_score': risk_score,
            'verdict': verdict,
            'recommended_action': action,
            'matched_scam_category': category_match,
        }

    # ── PRINTING ──────────────────────────────────
    def _print_transaction(self, t):
        print(f"[*] Analyzing transaction:")
        print(f"    Amount    : Rs.{t.get('amount',0):,}")
        print(f"    To        : {t.get('recipient_upi','?')}")
        print(f"    Time      : {t.get('timestamp','?')}")
        print(f"    Note      : {t.get('payment_note','(none)')}\n")


def print_text_report(result):
    line = "=" * 62
    t = result['transaction']
    print(f"\n{line}")
    print("           UPI FRAUDSCOPE — FRAUD ANALYSIS REPORT")
    print(line)
    print(f"\n  FRAUD RISK SCORE : {result['risk_score']}/100")
    print(f"  VERDICT          : {result['verdict']}")
    print(f"  ACTION           : {result['recommended_action']}")
    print(f"\n  FLAGS TRIGGERED  : {len(result['alerts'])}")

    if result['alerts']:
        print(f"\n{'-'*62}")
        print("  FRAUD INDICATORS DETECTED:")
        print('-'*62)
        for a in result['alerts']:
            print(f"\n  [{a['severity']}] {a['rule']}")
            print(f"     Detail    : {a['detail']}")
            print(f"     Risk pts  : +{a['score']}")

    print(f"\n{'-'*62}")
    print("  TRANSACTION DETAILS:")
    print('-'*62)
    print(f"  Amount     : Rs.{t.get('amount',0):,}")
    print(f"  Recipient  : {t.get('recipient_upi','?')}")
    print(f"  Time       : {t.get('timestamp','?')}")
    print(f"  Note       : {t.get('payment_note','(none)')}")
    print(f"  Device     : {t.get('device_name','Known device')}")

    cat = result['matched_scam_category']
    if cat and result['risk_score'] >= 45:
        print(f"\n{'-'*62}")
        print("  LIKELY SCAM PATTERN (for context/education):")
        print('-'*62)
        print(f"  Category : {cat['name']}")
        print(f"  {cat['description']}")
        print(f"  Source   : {cat['source']}")

    print(f"\n{line}")
    print(f"  FINAL STATUS: {result['recommended_action']}")
    print(f"{line}\n")


def save_json_report(results, path='fraud_report.json'):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    return path


def save_csv_report(results, path='fraud_alerts.csv'):
    if not isinstance(results, list):
        results = [results]
    fieldnames = ['amount', 'recipient_upi', 'timestamp', 'risk_score',
                  'verdict', 'recommended_action', 'matched_category', 'num_flags']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            t = r['transaction']
            cat = r['matched_scam_category']
            writer.writerow({
                'amount': t.get('amount', 0),
                'recipient_upi': t.get('recipient_upi', '?'),
                'timestamp': t.get('timestamp', '?'),
                'risk_score': r['risk_score'],
                'verdict': r['verdict'],
                'recommended_action': r['recommended_action'],
                'matched_category': cat['name'] if cat else '',
                'num_flags': len(r['alerts']),
            })
    return path


def load_transactions(path, batch):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Transaction file not found: {path}")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] Transaction file is not valid JSON: {path}\n        {e}")

    if batch:
        if not isinstance(data, list):
            sys.exit(f"[ERROR] --batch expects a JSON list of transactions, got {type(data).__name__}")
        return data
    else:
        if not isinstance(data, dict):
            sys.exit(f"[ERROR] Expected a single JSON transaction object (use --batch for a list), "
                      f"got {type(data).__name__}")
        return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score UPI transactions for fraud risk using rules grounded in public RBI/NPCI/I4C advisories."
    )
    parser.add_argument('transactions_file', help="path to a JSON transaction (or list, with --batch)")
    parser.add_argument('--batch', action='store_true',
                         help="treat the input as a list of transactions instead of one")
    parser.add_argument('--patterns', default='fraud_patterns.json',
                         help="path to the fraud patterns file (default: fraud_patterns.json)")
    parser.add_argument('--export', action='append', choices=['json', 'csv'], default=[],
                         help="also write 'json' and/or 'csv' output (repeatable)")
    parser.add_argument('--quiet', action='store_true', help="skip the rule-by-rule scan output")
    return parser.parse_args()


def main():
    args = parse_args()
    patterns = load_fraud_patterns(args.patterns)
    scope = UPIFraudScope(patterns)
    data = load_transactions(args.transactions_file, args.batch)

    if args.batch:
        results = scope.analyze_batch(data, quiet=args.quiet)
        if not args.quiet:
            for r in results:
                print_text_report(r)
        to_export = results
    else:
        result = scope.analyze(data, quiet=args.quiet)
        print_text_report(result)
        to_export = result

    saved = []
    if 'json' in args.export:
        saved.append(save_json_report(to_export))
    if 'csv' in args.export:
        saved.append(save_csv_report(to_export))
    if saved:
        print("Saved: " + ", ".join(saved))


if __name__ == "__main__":
    main()
