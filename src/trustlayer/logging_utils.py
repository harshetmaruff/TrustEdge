"""Logging helpers.

AGENTS.md requires ``logging`` rather than ``print`` for execution traces, so
every module routes its traces through this single configured logger.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, stream: Optional[object] = None) -> None:
    """Configure the root logger once, writing to stderr.

    Args:
        level: Minimum level to emit.
        stream: Optional stream override, mainly used by tests.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root logger on first use."""
    configure_logging()
    return logging.getLogger(name)
