"""
Shared LLM utilities used across all modules.

Provides:
  - get_anthropic_client()  — cached singleton Anthropic client
  - parse_llm_json()        — strip markdown fences and parse JSON
  - llm_call()              — tenacity-wrapped message create with retry
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    """Return the process-wide Anthropic client (created once)."""
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


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
        (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APITimeoutError)
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
) -> anthropic.types.Message:
    """
    Single wrapper around client.messages.create with automatic retry.
    Retries on rate-limit, server errors, and timeouts (up to 3 attempts).
    """
    client = get_anthropic_client()
    kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
    if system:
        kwargs["system"] = system
    return client.messages.create(**kwargs)
