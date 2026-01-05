from __future__ import annotations


class AppError(RuntimeError):
    """Base application error."""


class BadRequest(AppError):
    """Invalid request data."""


class DependencyError(AppError):
    """External dependency failed (e.g., model, Pinecone)."""