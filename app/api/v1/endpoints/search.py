from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from loguru import logger

from app.dependencies.container import Container, get_container
from app.models.schemas import BBox, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(...),
    category: str | None = Form(default=None),
    bbox_x1: int | None = Form(default=None),
    bbox_y1: int | None = Form(default=None),
    bbox_x2: int | None = Form(default=None),
    bbox_y2: int | None = Form(default=None),
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
    
    # Validate bounding box: either all coordinates are provided or none
    bbox_coords = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
    bbox_provided = [c is not None for c in bbox_coords]
    
    if any(bbox_provided) and not all(bbox_provided):
        raise BadRequest("Invalid bounding box: all coordinates (x1, y1, x2, y2) must be provided together")
    
    bbox = None
    if all(bbox_provided):
        # Validate bounding box coordinates
        if bbox_x1 >= bbox_x2 or bbox_y1 >= bbox_y2:
            raise BadRequest(f"Invalid bounding box: x1 ({bbox_x1}) must be < x2 ({bbox_x2}) and y1 ({bbox_y1}) must be < y2 ({bbox_y2})")
        if bbox_x1 < 0 or bbox_y1 < 0 or bbox_x2 < 0 or bbox_y2 < 0:
            raise BadRequest("Invalid bounding box: coordinates must be non-negative")
        bbox = BBox(x1=bbox_x1, y1=bbox_y1, x2=bbox_x2, y2=bbox_y2)
        logger.debug(f"Search request with bbox: {bbox}")

    # Normalize empty category string to None
    if category is not None and category.strip() == "":
        category = None
    
    logger.info(f"Search request - category: {category}, top_k: {top_k}, has_bbox: {bbox is not None}, image_filename: {image.filename}")
    
    svc = container.search_service
    result = await svc.search_room_image(
        room_image_file=image,
        category=category,
        bbox=bbox,
        top_k=top_k,
    )
    return result