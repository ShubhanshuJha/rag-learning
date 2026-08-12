"""
Application-wide logger setup.

Import `get_logger(__name__)` in any module that needs to log, so every
log line is tagged with its originating module — useful once you're
debugging why a specific /ask call returned a bad answer.
"""

import logging
import sys

from app.config import settings

_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)
