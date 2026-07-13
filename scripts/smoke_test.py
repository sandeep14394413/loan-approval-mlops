#!/usr/bin/env python3
"""
Smoke test script — called by CI after Kind deploy.
Reads JSON response from stdin and asserts required fields.
"""
import sys
import json

raw = sys.stdin.read().strip()
if not raw:
    print("ERROR: Empty response from /predict")
    sys.exit(1)

try:
    d = json.loads(raw)
except json.JSONDecodeError as e:
    print(f"ERROR: Could not parse JSON response: {e}")
    print(f"Raw response was: {raw}")
    sys.exit(1)

print(f"Parsed response: {d}")

required_fields = ["loan_safe", "prediction"]
for field in required_fields:
    if field not in d:
        print(f"ERROR: Missing '{field}' in response")
        sys.exit(1)

print("Smoke test PASSED")
sys.exit(0)
