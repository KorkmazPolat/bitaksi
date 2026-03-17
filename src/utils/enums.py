"""Shared enumerations used across the codebase."""
from enum import Enum


class RetrievalStrategy(str, Enum):
    DIRECT = "direct"
    EXPANSION = "expansion"
    HYDE = "hyde"
    DECOMPOSITION = "decomposition"
    NONE = "none"


class ContentType(str, Enum):
    TEXT = "text"
    VISUAL = "visual"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
