from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SearchHit(BaseModel):
    sku_id: str
    score: float
    image_id: str | None = None
    category: str | None = None
    attributes: dict | None = None


class SearchResponse(BaseModel):
    query_category: str | None
    hits: list[SearchHit] = Field(default_factory=list)
    message: str | None = None


class CatalogUpsertResponse(BaseModel):
    sku_id: str
    image_id: str
    upserted: bool


class DetectedObject(BaseModel):
    """A detected object with bounding box and category"""
    category: str
    bbox: BBox
    score: float
    object_id: int  # Index in the detection list


class DetectionResponse(BaseModel):
    """Response containing all detected objects in an image"""
    objects: list[DetectedObject] = Field(default_factory=list)
    image_width: int
    image_height: int


class SegmentedObject(BaseModel):
    """A detected object with segmentation mask"""
    category: str
    bbox: BBox
    score: float
    object_id: int
    mask_base64: str  # Base64 encoded mask image (PNG)


class DetectionSegmentationResponse(BaseModel):
    """Response containing all detected and segmented objects"""
    objects: list[SegmentedObject] = Field(default_factory=list)
    image_width: int
    image_height: int