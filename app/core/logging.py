from __future__ import annotations

import sys
from loguru import logger

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )