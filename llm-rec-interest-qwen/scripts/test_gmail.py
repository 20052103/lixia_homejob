# -*- coding: utf-8 -*-
"""
Quick smoke test for the Gmail tool.
Run from the project root:
    python scripts/test_gmail.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.gmail_tool import fetch_emails_as_text

print("=" * 60)
print("Gmail Tool - Smoke Test")
print("=" * 60)
print()
print("Fetching up to 5 emails from today (newer_than:1d)...")
print("A browser window will open for OAuth consent on first run.")
print()

try:
    result = fetch_emails_as_text(
        query="newer_than:1d",
        max_results=5,
        include_body=False,   # subject/sender/snippet only, no full body
    )
    print(result)
    print()
    print("✅ Test passed!")
except FileNotFoundError as e:
    print(f"❌ credentials.json not found:\n  {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
