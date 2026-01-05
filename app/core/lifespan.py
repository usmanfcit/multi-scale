from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from app.core.config import Settings
from app.dependencies.container import Container


def build_lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = Container.from_settings(settings)
        await container.start()
        app.state.container = container
        logger.info("App started")

        yield

        await container.stop()
        logger.info("App stopped")

    return lifespan