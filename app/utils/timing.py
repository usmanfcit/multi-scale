from __future__ import annotations

import time
from contextlib import contextmanager
from loguru import logger


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        logger.info("{} took {:.2f}ms", label, dt)