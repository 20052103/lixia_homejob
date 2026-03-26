# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# ============================================================
# LM Studio / OpenAI-compatible local endpoint
# ============================================================
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL_NAME = os.getenv("LM_STUDIO_MODEL_NAME", "Qwen_Qwen3.5-27B-Q4_K_M.gguf")

# ============================================================
# Default generation params
# ============================================================
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "768"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.95"))
DEFAULT_MAX_STEPS = int(os.getenv("DEFAULT_MAX_STEPS", "6"))

# ============================================================
# Web search config
# provider:
#   - "serpapi"    -> Google-backed via SerpAPI
#   - "duckduckgo" -> free fallback
#   - "auto"       -> use SerpAPI if key exists, else DuckDuckGo
# ============================================================
WEB_SEARCH_PROVIDER = "serpapi"
# os.getenv("WEB_SEARCH_PROVIDER", "auto").strip().lower()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()
WEB_SEARCH_TIMEOUT_SEC = int(os.getenv("WEB_SEARCH_TIMEOUT_SEC", "15"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_USER_AGENT = os.getenv(
    "WEB_SEARCH_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
)

# ============================================================
# Gmail API config
# credentials.json: download from Google Cloud Console
# gmail_token.json: auto-created after first OAuth consent
# ============================================================
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
GMAIL_CREDENTIALS_PATH = os.getenv(
    "GMAIL_CREDENTIALS_PATH",
    os.path.join(_AGENT_DIR, "credentials.json"),
)
GMAIL_TOKEN_PATH = os.getenv(
    "GMAIL_TOKEN_PATH",
    os.path.join(_AGENT_DIR, "gmail_token.json"),
)

# ============================================================
# Google Calendar API config
# Reuses credentials.json from Gmail setup.
# gcal_token.json is separate so Gmail auth is unaffected.
# Enable "Google Calendar API" in Cloud Console if not already.
# ============================================================
GCAL_CREDENTIALS_PATH = os.getenv(
    "GCAL_CREDENTIALS_PATH",
    os.path.join(_AGENT_DIR, "credentials.json"),
)
GCAL_TOKEN_PATH = os.getenv(
    "GCAL_TOKEN_PATH",
    os.path.join(_AGENT_DIR, "gcal_token.json"),
)

# ============================================================
# Google Drive API config
# Reuses credentials.json from Gmail setup.
# gdrive_token.json is separate so other tokens are unaffected.
# Enable "Google Drive API" in Cloud Console if not already.
# ============================================================
GDRIVE_CREDENTIALS_PATH = os.getenv(
    "GDRIVE_CREDENTIALS_PATH",
    os.path.join(_AGENT_DIR, "credentials.json"),
)
GDRIVE_TOKEN_PATH = os.getenv(
    "GDRIVE_TOKEN_PATH",
    os.path.join(_AGENT_DIR, "gdrive_token.json"),
)