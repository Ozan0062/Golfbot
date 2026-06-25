"""
Logging setup for the project.
Prints colors to terminal and saves a log file.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ── Default configuration ─────────────────────────────────────────────────────

DEFAULT_LEVEL   = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_DIR         = Path(__file__).parent / "logs"
LOG_FILENAME    = "golfbot.log"
MAX_BYTES       = 5 * 1024 * 1024   # 5 MB per file
BACKUP_COUNT    = 3                 # keep golfbot.log + 3 rotated files

# ANSI colour codes for the console handler
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}

_SETUP_DONE = False


# ── Formatters ────────────────────────────────────────────────────────────────

class ColourFormatter(logging.Formatter):
    """Adds ANSI colour to the level name in console output."""

    FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    DATEFMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset  = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


_PLAIN_FMT = logging.Formatter(
    fmt     = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)


# ── Public API ────────────────────────────────────────────────────────────────

def setup_logging(
    level:       str  = DEFAULT_LEVEL,
    log_to_file: bool = True,
    log_dir:     Path = LOG_DIR,
    colour:      bool = True,
) -> None:
    """Call once at program start to setup log file and console output."""
    global _SETUP_DONE
    if _SETUP_DONE:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(ColourFormatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ) if colour else _PLAIN_FMT)
    root.addHandler(ch)

    # Rotating file handler
    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / LOG_FILENAME
        fh = RotatingFileHandler(
            log_path,
            maxBytes    = MAX_BYTES,
            backupCount = BACKUP_COUNT,
            encoding    = "utf-8",
        )
        fh.setLevel(logging.DEBUG)   # always write everything to file
        fh.setFormatter(_PLAIN_FMT)
        root.addHandler(fh)
        # Log the path so the user knows where to look
        logging.getLogger(__name__).info("Log file → %s", log_path.resolve())

    # Silence noisy third-party libraries
    for noisy in ("ultralytics", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _SETUP_DONE = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the current file."""
    return logging.getLogger(name)