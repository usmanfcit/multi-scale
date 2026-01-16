from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from loguru import logger

from app.dependencies.container import Container, get_container
from app.models.schemas import BBox, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(...),
    assigned_category: str | None = Form(default=None),
    top_k: int = Form(default=20),
    container: Container = Depends(get_container),
) -> SearchResponse:
    from app.core.errors import BadRequest
    
    # Validate image file
    if not image.filename:
        raise BadRequest("Image file is required")
    
    # Validate top_k
    if top_k <= 0:
        raise BadRequest(f"top_k must be greater than 0, got {top_k}")
    if top_k > 100:
        raise BadRequest(f"top_k cannot exceed 100, got {top_k}")

    # Normalize empty category string to None
    if assigned_category is not None and assigned_category.strip() == "":
        assigned_category = None
    
    logger.info(
        f"Search request - category: {assigned_category}, top_k: {top_k}, "
        f"image_filename: {image.filename}"
    )
    
    svc = container.search_service
    result = await svc.search_room_image(
        room_image_file=image,
        assigned_category=assigned_category,
        top_k=top_k,
    )
    return result