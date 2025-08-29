# route_backend_predict.py
# Purpose: Lightweight endpoint that returns ONE "next word" suggestion for the chat input.
# Works with Azure OpenAI in Azure Gov (.azure.us). Uses your existing env vars.

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
    endpoint = _first_env(
        ["AZURE_OPENAI_ENDPOINT", "azure_openai_endpoint", "azure_openai_embedding_endpoint"]
    )
    api_version = _first_env(
        ["AZURE_OPENAI_API_VERSION", "azure_openai_api_version", "azure_openai_embedding_api_version"],
        default="2024-06-01",
    )

    if not api_key or not endpoint:
        raise RuntimeError(
            "Missing Azure OpenAI credentials. Ensure AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT are set."
        )

    # Azure Gov endpoints look like: https://<resource>.openai.azure.us
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )


def _sanitize_word(w: str) -> str:
    """
    Keep a single, safe token (letters, numbers, hyphen). Trim quotes/punct.
    Cap length to avoid UI oddities.
    """
    w = (w or "").strip()
    w = re.sub(r'^[\"\'`«»“”]+|[\"\'`«»“”]+$', "", w)  # strip surrounding quotes
    w = re.split(r"\s+", w)[0]                         # keep first token only
    w = re.sub(r"[^A-Za-z0-9\-]", "", w)               # alnum + hyphen only
    return w[:24]


@bp_predict.route("/api/predict-next-word", methods=["POST"])
def predict_next_word():
    """
    Body:    { "prefix": "<user is typing text>" }
    Returns: { "suggestion": "<one next word or empty>" }
    """
    # Feature flag (default ON if not set in config)
    if not current_app.config.get("ENABLE_TYPING_PREDICTION", True):
        return jsonify({"suggestion": ""}), 200

    data = request.get_json(silent=True) or {}
    prefix = (data.get("prefix") or "")[-500:]  # Trim overly long inputs

    # Only try on reasonable prefixes (caret-at-end logic is enforced client-side)
    if len(prefix) < 3 or not re.search(r"\w$", prefix):
        return jsonify({"suggestion": ""}), 200

    # Deployment to use for chat completions
    deployment = _first_env(
        ["AZURE_OPENAI_DEPLOYMENT", "DEFAULT_GPT_DEPLOYMENT", "azure_openai_deployment"],
        default="gpt-4o-mini",
    )

    try:
        client = _get_azure_client()

        system_prompt = (
            "You are a next-word suggester. "
            "Given partial user input, output ONLY the single most likely next WORD. "
            "No spaces, no punctuation, no quotes. If uncertain, output nothing."
        )
        user_prompt = f"User is typing this prefix:\n{prefix}\nNext word only:"

        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
            current_app.logger.exception("predict-next-word error: %s", exc)
        except Exception:
            pass
        return jsonify({"suggestion": ""}), 200