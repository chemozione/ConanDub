"""Logging helpers with verbosity controls."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional


def _resolve_level(verbosity: int) -> int:
    """Map verbosity counter to logging levels."""

    if verbosity <= -1:
        return logging.ERROR if verbosity <= -2 else logging.WARNING
    if verbosity == 0:
        return logging.INFO
    if verbosity == 1:
        return logging.DEBUG
    return logging.NOTSET


@lru_cache(maxsize=1)
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger configured with our formatter."""

    logger = logging.getLogger(name)
    return logger


def configure_logging(verbosity: int = 0) -> None:
    """Configure root logging handler once."""

    level = _resolve_level(verbosity)
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return

    handler: logging.Handler
    try:
        from rich.logging import RichHandler  # type: ignore

        handler = RichHandler(rich_tracebacks=False, markup=False)
        formatter: logging.Formatter | None = None
    except ModuleNotFoundError:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[handler])

