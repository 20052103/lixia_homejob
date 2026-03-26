# -*- coding: utf-8 -*-
"""
Diagnose Gmail send issues.
Run: python scripts/test_gmail_send.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TOKEN_PATH = os.path.join(os.path.dirname(__file__), "..", "agent", "gmail_token.json")
CREDS_PATH = os.path.join(os.path.dirname(__file__), "..", "agent", "credentials.json")

# ── Step 1: Check token scopes ────────────────────────────────
print("=" * 60)
print("Step 1: Checking token scopes")
print("=" * 60)
import json
if os.path.exists(TOKEN_PATH):
    with open(TOKEN_PATH) as f:
        token_data = json.load(f)
    scopes = token_data.get("scopes", token_data.get("scope", "NOT FOUND"))
    print(f"Token scopes: {scopes}")
    has_send = "gmail.send" in str(scopes)
    print(f"Has send scope: {'✅ YES' if has_send else '❌ NO — delete gmail_token.json and re-run!'}")
    if not has_send:
        print("\n👉 Fix: delete agent/gmail_token.json and restart the agent to re-authorize.")
        sys.exit(1)
else:
    print("❌ No token found — run the agent first to authorize.")
    sys.exit(1)

# ── Step 2: Try sending a test email to yourself ──────────────
print()
print("=" * 60)
print("Step 2: Sending test email to yourself")
print("=" * 60)

from agent.gmail_tool import send_email, _get_credentials
from googleapiclient.discovery import build

creds = _get_credentials(CREDS_PATH, TOKEN_PATH)
service = build("gmail", "v1", credentials=creds)
profile = service.users().getProfile(userId="me").execute()
my_email = profile.get("emailAddress")
print(f"Your Gmail: {my_email}")

try:
    result = send_email(
        to=my_email,
        subject="[Agent Test] Send test",
        body="This is a test email from your local agent.\nIf you see this, send is working!",
        credentials_path=CREDS_PATH,
        token_path=TOKEN_PATH,
    )
    print(result)
    print()
    print("✅ Send API call succeeded!")
    print("👉 Check your inbox (and Sent folder) to confirm delivery.")
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()
