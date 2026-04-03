"""
Shared LLM utilities used across all modules.

Provides:
  - get_openai_client()  — cached singleton OpenAI client
  - get_gemini_client()  — cached singleton Gemini client (OpenAI-compatible)
  - get_llm_client()     — returns the active client based on llm_provider setting
  - parse_llm_json()     — strip markdown fences and parse JSON
  - llm_call()           — tenacity-wrapped chat completion with retry
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@lru_cache(maxsize=1)
def get_openai_client() -> openai.OpenAI:
    """Return the process-wide OpenAI client (created once)."""
    settings = get_settings()
    return openai.OpenAI(api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def get_gemini_client() -> openai.OpenAI:
    """Return the process-wide Gemini client via OpenAI-compatible endpoint."""
    settings = get_settings()
    return openai.OpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_BASE_URL)


def get_llm_client() -> openai.OpenAI:
    """Return the active LLM client based on the llm_provider setting."""
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return get_gemini_client()
    return get_openai_client()


def parse_llm_json(text: str) -> dict | list:
    """
    Strip optional markdown code fences and parse JSON from an LLM response.

    Handles:
        ```json
        {...}
        ```
    and bare JSON strings alike.
    """
    cleaned = _clean_llm_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _repair_json_text(cleaned)
        return json.loads(repaired)


def coerce_text_response(response) -> str:
    """Normalize SDK- or test-style responses into plain text."""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, list) and content:
        text = getattr(content[0], "text", None)
        if isinstance(text, str):
            return text
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return str(response or "")


def _clean_llm_json_text(text: str) -> str:
    text = coerce_text_response(text).strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    text = text.strip()
    extracted = _extract_json_region(text)
    return extracted.strip()


def _extract_json_region(text: str) -> str:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        return text
    start = min(starts)
    return text[start:]


def _repair_json_text(text: str) -> str:
    text = text.strip()
    text = _close_unterminated_string(text)
    text = _balance_json_delimiters(text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _close_unterminated_string(text: str) -> str:
    escaped = False
    in_string = False
    out: list[str] = []
    for ch in text:
        out.append(ch)
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
        elif in_string and ch == "\n":
            out[-1] = "\\n"
    if in_string:
        out.append('"')
    return "".join(out)


def _balance_json_delimiters(text: str) -> str:
    stack: list[str] = []
    escaped = False
    in_string = False
    out: list[str] = []

    for ch in text:
        out.append(ch)
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()
            else:
                out.pop()

    while stack:
        out.append(stack.pop())
    return "".join(out)


@retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def llm_call(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    system: str | None = None,
) -> str:
    """
    Single wrapper around client.chat.completions.create with automatic retry.
    Retries on rate-limit, server errors, and timeouts (up to 3 attempts).
    Returns the text content of the first choice.
    """
    client = get_llm_client()
    all_messages: list[dict] = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=all_messages,
    )
    return response.choices[0].message.content
