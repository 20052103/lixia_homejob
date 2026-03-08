# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# ============================================================
# LM Studio / OpenAI-compatible local endpoint
# ============================================================
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:8080/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL_NAME = os.getenv("LM_STUDIO_MODEL_NAME", "local-model")

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