# route_backend_predict.py
# Purpose: finish-the-current-word suggestion for the chat input.
# Works with Azure OpenAI in Azure Gov (.azure.us). Uses your existing env vars.
#
# Behavior:
# - Accept BOTH {"prefix": "..."} and {"text": "..."}.
# - If the user ended with whitespace -> return "" (nothing to finish).
# - If the last token is very short -> return "" (too early to be useful).
# - Ask the model to return ONLY the missing tail of that token (the suffix).
# - Sanitize to a single, no-space, no-punctuation suffix.
# - If the model tries to return the same fragment or an empty string -> "".

import os
import re
from flask import Blueprint, request, jsonify, current_app
from openai import AzureOpenAI

bp_predict = Blueprint("predict", __name__)


def _first_env(names, default=None):
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


def _get_azure_client():
    api_key = _first_env(["AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY"])
    endpoint = _first_env([
        "AZURE_OPENAI_ENDPOINT",
        "azure_openai_endpoint",
        "azure_openai_embedding_endpoint",
    ])
    api_version = _first_env(
        ["AZURE_OPENAI_API_VERSION", "azure_openai_api_version", "azure_openai_embedding_api_version"],
        default="2024-06-01",
    )

    if endpoint:
        endpoint = endpoint.strip().rstrip("/")

    if not api_key or not endpoint:
        raise RuntimeError("Missing Azure OpenAI credentials (key/endpoint).")

    try:
        current_app.logger.info(f"[predict] using endpoint={endpoint}, api_version={api_version}")
    except Exception:
        pass

    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )


# simple "is this a reasonable word part?"
_WORD_CHARS_RE = re.compile(r"[A-Za-z0-9_\-\u00C0-\u024F\u1E00-\u1EFF']+")


def _extract_last_token(raw: str):
    """
    Return (head, fragment) where:
      head     = everything before the current word fragment
      fragment = the incomplete word at the end (no trailing space in raw)
    Example:
      "how am i suppo" -> ("how am i ", "suppo")
      "document "      -> ("document ", "")
    """
    if not raw:
        return "", ""
    # raw here is the ORIGINAL (not rstrip'ed) so we can see if it ends with space
    if raw.endswith((" ", "\t", "\n")):
        # cursor is at word boundary -> nothing to finish
        return raw, ""
    # split on whitespace to get the last "word-like" chunk
    parts = re.split(r"(\s+)", raw)
    last = parts[-1]
    head = "".join(parts[:-1])
    return head, last


def _sanitize_suffix(s: str):
    """
    Keep only the wordy part, no spaces/punct, short.
    This is a suffix, so we don't split on whitespace here — we just strip it.
    """
    s = (s or "").strip()
    # drop any whitespace in the middle (we only want a tail, not a new word)
    s = re.sub(r"\s+", "", s)
    # drop punctuation around
    s = s.strip(".,;:!?—–…\"'()[]{}")
    return s[:24]


@bp_predict.route("/api/predict-next-word", methods=["POST"])
def predict_next_word():
    # feature flag (default on)
    if not current_app.config.get("ENABLE_TYPING_PREDICTION", True):
        return jsonify({"suggestion": ""}), 200

    try:
        payload = request.get_json(silent=True) or {}
        original = (payload.get("prefix") or payload.get("text") or "")
        # trim super-long input for safety
        original = original[-500:]

        # 1) figure out if there's actually a word to finish
        head, fragment = _extract_last_token(original)

        # if there's no fragment (user ended with space) -> nothing to finish
        if not fragment:
            return jsonify({"suggestion": ""}), 200

        # if fragment is too short, don't bother
        if len(fragment) < 2:
            return jsonify({"suggestion": ""}), 200

        # make sure fragment is word-ish
        if not _WORD_CHARS_RE.fullmatch(fragment):
            return jsonify({"suggestion": ""}), 200

        client = _get_azure_client()

        # we tell the model explicitly: give me ONLY the missing tail AFTER this fragment
        system_msg = {
            "role": "system",
            "content": (
                "You are an autocomplete engine for a chat box. "
                "The user has typed the beginning of ONE word. "
                "You MUST respond with ONLY the remaining characters that would complete that SAME word. "
                "Do NOT repeat the part the user already typed. "
                "Do NOT add spaces. "
                "Do NOT start a new word. "
                "If you cannot complete it, reply with an empty string."
            ),
        }
        user_msg = {
            "role": "user",
            "content": f"Complete this partial word by returning ONLY the remaining characters (the suffix): '{fragment}'",
        }

        deployment = _first_env(
            ["AZURE_OPENAI_DEPLOYMENT", "DEFAULT_GPT_DEPLOYMENT", "azure_openai_deployment", "embedding_model"],
            default="gpt-4o-mini",
        )

        resp = client.chat.completions.create(
            model=deployment,
            messages=[system_msg, user_msg],
            temperature=0.0,
            max_tokens=4,
            stop=[" ", "\n", ".", ","],  # Azure-safe (max 4)
        )

        raw_suffix = (resp.choices[0].message.content or "")
        suffix = _sanitize_suffix(raw_suffix)

        # If the model echoed the fragment or returned nothing, give nothing
        if not suffix:
            return jsonify({"suggestion": ""}), 200

        return jsonify({"suggestion": suffix}), 200

    except Exception as exc:
        try:
            current_app.logger.error(f"predict-next-word error: {exc}")
        except Exception:
            pass
        return jsonify({"suggestion": ""}), 200
