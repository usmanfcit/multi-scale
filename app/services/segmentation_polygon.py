from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image, ImageDraw
from loguru import logger

from app.models.domain import Segment
from app.models.schemas import BBox


@dataclass
class PolygonSegmentationService:
    """
    Segmentation using polygon masks from RF-DETR API.
    
    No model loading required - uses pre-computed polygon masks
    from detection API response.
    """
    
    async def segment(
        self, 
        img: Image.Image, 
        bbox: BBox,
        mask_polygon: list | None = None
    ) -> Segment:
        """
        Create segmentation mask from polygon points.
        
        Args:
            img: Input image
            bbox: Bounding box (for compatibility and fallback)
            mask_polygon: List of [x, y] polygon points from RF-DETR API
        
        Returns:
            Segment with boolean mask array
        """
        if not mask_polygon or len(mask_polygon) < 3:
            logger.warning(
                "No valid polygon provided (need at least 3 points), "
                "falling back to rectangle mask"
            )
            return self._rectangle_fallback(img, bbox)
        
        try:
            w, h = img.size
            
            logger.debug(f"Creating polygon mask from {len(mask_polygon)} points")
            
            # Create empty mask
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)
            
            # Convert polygon points to tuples for PIL
            polygon_tuples = [(p[0], p[1]) for p in mask_polygon]
            
            # Draw filled polygon
            draw.polygon(polygon_tuples, fill=255)
            
            # Convert to numpy boolean array
            mask_np = np.array(mask)
            mask_np = mask_np > 128  # Threshold to boolean
            
            # Fill holes in the mask (remove any interior gaps) using opencv
            # Convert to uint8 for opencv
            mask_uint8 = mask_np.astype(np.uint8) * 255
            # Find contours and fill
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_uint8, contours, -1, 255, -1)  # Fill all contours
            mask_np = (mask_uint8 > 0).astype(bool)
            
            # Calculate coverage statistics
            coverage = mask_np.sum() / mask_np.size * 100
            
            logger.debug(
                f"Polygon segmentation complete: mask coverage {coverage:.1f}%, "
                f"shape: {mask_np.shape}"
            )
            
            return Segment(
                bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
                mask=mask_np
            )
            
        except Exception as e:
            logger.error(f"Polygon segmentation failed: {e}, using rectangle fallback")
            return self._rectangle_fallback(img, bbox)
    
    def _rectangle_fallback(self, img: Image.Image, bbox: BBox) -> Segment:
        """
        Fallback segmentation using rectangle mask.
        Used when polygon is not available or invalid.
        """
        w, h = img.size
        mask = np.zeros((h, w), dtype=bool)
        
        # Fill rectangle region
        mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = True
        
        coverage = mask.sum() / mask.size * 100
        logger.debug(
            f"Rectangle fallback mask: coverage {coverage:.1f}%, "
            f"bbox: ({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})"
        )
        
        return Segment(
            bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
            mask=mask
        )
