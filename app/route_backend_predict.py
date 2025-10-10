# route_backend_predict.py
# Purpose: Lightweight endpoint that returns ONE "next word" suggestion for the chat input.
# Works with Azure OpenAI in Azure Gov (.azure.us). Uses your existing env vars.
#
# Changes (kept minimal, no env var name changes):
# - Added endpoint sanitation (strip & rstrip('/')) and one-time logging of the effective endpoint/version.
# - Clearer error classification in logs (dns_resolution_failed / network_timeout / auth_or_firewall / unknown).
# - No changes to your env var names or external API shape; failures still return {"suggestion": ""}.

import os
import re
from flask import Blueprint, request, jsonify, current_app

# Requires: openai >= 1.0.0
# pip install --upgrade openai
from openai import AzureOpenAI

bp_predict = Blueprint("predict", __name__)  # Register in app.py: app.register_blueprint(bp_predict)


# -------- Env helpers --------
def _first_env(names, default=None):
    """Return the first non-empty environment variable from a list of names."""
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


def _get_azure_client():
    """
    Build an AzureOpenAI client using your existing env variables.
    Priorities:
      Key:       AZURE_OPENAI_KEY (then AZURE_OPENAI_API_KEY)
      Endpoint:  AZURE_OPENAI_ENDPOINT (then azure_openai_endpoint, then azure_openai_embedding_endpoint)
      Version:   AZURE_OPENAI_API_VERSION (then azure_openai_api_version, then azure_openai_embedding_api_version)
    """
    api_key = _first_env(["AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY"])
    endpoint = _first_env(["AZURE_OPENAI_ENDPOINT", "azure_openai_endpoint", "azure_openai_embedding_endpoint"])
    api_version = _first_env(
        ["AZURE_OPENAI_API_VERSION", "azure_openai_api_version", "azure_openai_embedding_api_version"],
        default="2024-06-01"
    )

    if endpoint:
        endpoint = endpoint.strip().rstrip("/")  # guard against whitespace/trailing slash

    if not api_key or not endpoint:
        raise RuntimeError("Missing Azure OpenAI credentials (key/endpoint).")

    # Helpful one-time log for diagnostics
    try:
        current_app.logger.info(f"AOAI endpoint in use: {endpoint}, api_version={api_version}")
    except Exception:
        pass

    return AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)


# -------- Tiny cleaner for single-word suggestion --------
_WORD_RE = re.compile(r"^[A-Za-z0-9_\-\u00C0-\u024F\u1E00-\u1EFF']+$")  # keep it readable; allow latin accents


def _sanitize_word(text: str) -> str:
    """Return a clean single 'word-like' token from the LLM output, else empty string."""
    if not text:
        return ""
    # Take first token-like chunk (split on whitespace and punctuation)
    first = text.strip().split()[0]
    # If it ends with punctuation, drop it
    first = first.strip(".,;:!?—–…\"'()[]{}")
    # Keep only word-like strings
    return first if _WORD_RE.match(first) else ""


# -------- Route --------
@bp_predict.route("/api/predict-next-word", methods=["POST"])
def predict_next_word():
    try:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"suggestion": ""}), 200

        client = _get_azure_client()

        # Build a super-light prompt for next-token hinting
        system_msg = {"role": "system", "content": "You predict the next single word only."}
        user_msg = {"role": "user", "content": f"Continue the last word or suggest the next single word:\n\n{text}"}

        # Use your configured deployment name
        deployment = _first_env(["AZURE_OPENAI_DEPLOYMENT", "embedding_model"], default="gpt-4o")

        # Keep a tiny budget; we only need one short token
        resp = client.chat.completions.create(
            model=deployment,
            messages=[system_msg, user_msg],
            temperature=0.2,
            max_tokens=2,
            stop=[" ", "\n", ".", ",", ";", ":", "!", "?", "—", "–", "…", "\t"],
        )

        raw = (resp.choices[0].message.content or "").strip()
        suggestion = _sanitize_word(raw)
        return jsonify({"suggestion": suggestion}), 200

    except Exception as exc:
        # Fail safe: log and return empty suggestion (no UI disruption)
        try:
            msg = str(exc)
            # Classify for quicker ops triage
            if ("Name or service not known" in msg) or ("getaddrinfo" in msg) or ("NXDOMAIN" in msg):
                hint = "dns_resolution_failed"
            elif "timed out" in msg:
                hint = "network_timeout"
            elif ("403" in msg) or ("401" in msg):
                hint = "auth_or_firewall"
            else:
                hint = "unknown"
            current_app.logger.error(f"predict-next-word error ({hint}): {msg}")
        except Exception:
            pass
        return jsonify({"suggestion": ""}), 200
