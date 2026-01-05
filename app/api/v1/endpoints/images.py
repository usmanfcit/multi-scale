from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette import status

from app.dependencies.container import Container, get_container

router = APIRouter()


@router.get("/images/{image_id}")
async def get_image(
    image_id: str,
    container: Container = Depends(get_container),
):
    """Retrieve an image by image_id"""
    image_path = container.search_service.image_io.get_image_path(image_id)
    
    if image_path is None or not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with id '{image_id}' not found"
        )
    
    # Determine media type based on file extension
    ext = image_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    
    return FileResponse(
        path=str(image_path),
        media_type=media_type
    )

