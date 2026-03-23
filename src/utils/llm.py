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
    text = text.strip()
    if text.startswith("```"):
        # drop opening fence line
        text = text.split("```", maxsplit=2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


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
