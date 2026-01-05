from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.dependencies.container import Container, get_container
from app.models.schemas import CatalogUpsertResponse

router = APIRouter()


@router.post("/catalog/upsert", response_model=CatalogUpsertResponse)
async def upsert_catalog_item(
    sku_id: str = Form(...),
    category: str = Form(...),
    image: UploadFile = File(...),
    # optional JSON-ish strings for simplicity; in production accept proper JSON body
    attributes_json: str | None = Form(default=None),
    container: Container = Depends(get_container),
) -> CatalogUpsertResponse:
    # Normalize category to lowercase for consistency
    normalized_category = category.lower().strip()
    svc = container.search_service
    return await svc.upsert_catalog_image(
        sku_id=sku_id,
        category=normalized_category,
        image_file=image,
        attributes_json=attributes_json,
    )