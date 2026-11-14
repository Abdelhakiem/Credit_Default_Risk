"""JSON logging shared by command-line jobs and the future API."""

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime

from .config import settings

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(value: str) -> Token:
    """Bind a request ID to the current execution context."""
    return _correlation_id.set(value)


def reset_correlation_id(token: Token) -> None:
    """Restore the previous context after a request completes."""
    _correlation_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record, including the active request ID."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": _correlation_id.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None) -> None:
    """Install a single JSON handler on the root logger."""
    chosen_level = (level or settings.log_level).upper()
    if chosen_level not in logging.getLevelNamesMapping():
        raise ValueError(f"Invalid log level: {chosen_level}")

    root = logging.getLogger()
    root.setLevel(chosen_level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
