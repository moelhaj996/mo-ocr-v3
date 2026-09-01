"""Structured logging with a per-document correlation id."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

_doc_id: ContextVar[str] = ContextVar("doc_id", default="-")


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.doc_id = _doc_id.get()
        return True


def set_doc_id(doc_id: str) -> None:
    _doc_id.set(doc_id)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(doc_id)s] %(name)s: %(message)s"
            )
        )
        handler.addFilter(_CorrelationFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
