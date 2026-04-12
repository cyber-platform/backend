"""Logging configuration for backend service with dual-channel output.

Provides both stdout (for Docker logs) and file-based logging (for persistent storage).
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    name: str = "backend",
    level: str | None = None,
    log_dir: str | None = None,
) -> logging.Logger:
    """Setup dual-channel logger with stdout and file handlers.

    Args:
        name: Logger name (default: backend)
        level: Log level (default: from LOG_LEVEL env var or INFO)
        log_dir: Directory for log files (default: from LOG_DIR env var or ./logs)

    Returns:
        Configured logger instance
    """
    # Determine log level
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Determine log directory
    if log_dir is None:
        log_dir = os.environ.get("LOG_DIR", "./logs")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stdout handler (for Docker logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (for persistent logs)
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path / "backend.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        # If file logging fails, at least we have console logging
        logger.warning(f"Failed to setup file logging: {e}")

    return logger


def get_logger(name: str = "backend") -> logging.Logger:
    """Get or create logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger
