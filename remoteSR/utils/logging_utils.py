from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(
    log_file: str | Path | None, level: int = logging.INFO
) -> logging.Logger:
    """
    Configure a logger that writes to both stdout and an optional file.
    """
    logger = logging.getLogger("remoteSR")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
