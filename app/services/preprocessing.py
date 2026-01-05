from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from loguru import logger

from app.models.schemas import BBox


@dataclass
class PreprocessingService:
    """
    Image preprocessing service with rotation augmentation.
    
    Features:
    - Multi-crop generation (tight, medium, full)
    - Rotation augmentation for angle invariance (90°, 180°, 270°)
    - Mask application with smart background handling
    - Bbox validation and clamping
    
    Best Practices:
    - Always clamp bboxes before cropping
    - Use rotation augmentation for products with varying angles
    - Apply masks to remove background noise
    """
    enable_rotation_aug: bool = True  # Enable rotation augmentation for angle invariance
    
    def clamp_bbox(self, bbox: BBox, w: int, h: int) -> BBox:
        x1 = max(0, min(w - 1, bbox.x1))
        y1 = max(0, min(h - 1, bbox.y1))
        x2 = max(0, min(w, bbox.x2))
        y2 = max(0, min(h, bbox.y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bbox after clamping")
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def crop_multi(self, img: Image.Image, bbox: BBox) -> dict[str, Image.Image]:
        """
        Generate multiple crops for robust feature extraction.
        
        Crops:
        - tight: Exact bbox crop (focused on object)
        - medium: Bbox with 15% padding (includes context)
        - full: Full image (global context)
        - rotated variants: 90°, 180°, 270° (if rotation_aug enabled)
        
        Args:
            img: Input image
            bbox: Bounding box for object
            
        Returns:
            Dictionary of crop names to crop images
        """
        w, h = img.size
        tight = img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))

        # Medium crop with 15% padding for context
        pad = int(0.15 * max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1))
        mx1 = max(0, bbox.x1 - pad)
        my1 = max(0, bbox.y1 - pad)
        mx2 = min(w, bbox.x2 + pad)
        my2 = min(h, bbox.y2 + pad)
        medium = img.crop((mx1, my1, mx2, my2))

        crops = {"tight": tight, "medium": medium, "full": img}
        
        # Add rotation-augmented crops for angle invariance
        # Handles products photographed at different angles (0°, 90°, 180°, 270°)
        if self.enable_rotation_aug:
            logger.debug("Adding rotation-augmented crops for angle invariance")
            try:
                for angle in [90, 180, 270]:
                    # Rotate crops (expand=True maintains full content)
                    crops[f"tight_rot{angle}"] = tight.rotate(
                        angle, expand=True, resample=Image.Resampling.BILINEAR
                    )
                    crops[f"medium_rot{angle}"] = medium.rotate(
                        angle, expand=True, resample=Image.Resampling.BILINEAR
                    )
                logger.debug(f"Added {len([k for k in crops if 'rot' in k])} rotated crops")
            except Exception as e:
                logger.warning(f"Failed to generate rotated crops (non-critical): {e}")
        
        return crops

    def apply_mask_on_crop(
        self, 
        crop: Image.Image, 
        mask: np.ndarray | None,
        bbox: BBox | None = None
    ) -> Image.Image:
        """
        Apply mask to crop image, removing background for cleaner embeddings.
        
        Uses SAM2.1 mask to isolate the product from background clutter.
        Background is replaced with mean foreground color (more natural than black).
        
        Args:
            crop: The cropped image
            mask: Full image mask (H, W) as boolean array from SAM2.1
            bbox: Bounding box used for crop (to extract mask region)
            
        Returns:
            Masked crop image with background removed
        """
        if mask is None:
            logger.debug("No mask provided, returning original crop")
            return crop
        
        try:
            # Convert crop to numpy
            crop_array = np.array(crop)
            
            if bbox is not None:
                # Extract mask region corresponding to the crop
                crop_mask = mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                
                # Resize mask if needed to match crop size
                if crop_mask.shape[:2] != crop_array.shape[:2]:
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray(crop_mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize(
                        (crop_array.shape[1], crop_array.shape[0]),
                        Image.Resampling.NEAREST
                    )
                    crop_mask = np.array(mask_img) > 128
            else:
                # If no bbox, assume mask matches crop size
                crop_mask = mask
                if crop_mask.shape[:2] != crop_array.shape[:2]:
                    logger.warning(
                        f"Mask size mismatch: {crop_mask.shape} vs crop {crop_array.shape}, "
                        f"returning original crop"
                    )
                    return crop
            
            # Apply mask
            masked_array = crop_array.copy()
            
            # Use mean foreground color for background (more natural than black)
            if crop_mask.sum() > 0:  # Check if mask is not empty
                # Calculate mean color of foreground
                mean_color = crop_array[crop_mask].mean(axis=0).astype(np.uint8)
                # Replace background with mean color
                masked_array[~crop_mask] = mean_color
                
                coverage = crop_mask.sum() / crop_mask.size * 100
                logger.debug(f"Applied mask with {coverage:.1f}% foreground coverage")
            else:
                # If mask is empty, return original (segmentation failed)
                logger.warning("Empty mask, returning original crop")
                return crop
            
            # Option 2: Zero out background (alternative approach)
            # masked_array[~crop_mask] = 0
            
            return Image.fromarray(masked_array)
            
        except Exception as e:
            logger.error(f"Error applying mask to crop: {e}")
            # Fallback: return original crop
            return crop