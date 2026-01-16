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

    def crop_base(self, img: Image.Image, bbox: BBox) -> tuple[dict[str, Image.Image], dict[str, BBox]]:
        """
        Generate base crops (tight, medium, full) without rotations.
        Returns crops and their corresponding bboxes for mask extraction.
        
        This method is used to separate base crop generation from rotation augmentation,
        allowing masks to be applied before rotations are generated.
        
        Features:
        - Minimum crop size enforcement (300px) to prevent quality loss from upscaling
        - Adaptive padding based on object size (smaller objects get more context)
        - Smart boundary handling when expanding crops
        
        Args:
            img: Input image
            bbox: Bounding box for object
            
        Returns:
            Tuple of (crops_dict, bboxes_dict) where:
            - crops_dict: {"tight": crop, "medium": crop, "full": crop}
            - bboxes_dict: {"tight": bbox, "medium": bbox, "full": bbox}
        """
        w, h = img.size
        
        # Calculate object dimensions
        obj_w = bbox.x2 - bbox.x1
        obj_h = bbox.y2 - bbox.y1
        obj_size = max(obj_w, obj_h)
        
        # MINIMUM CROP SIZE: Ensure crops are large enough for quality embeddings
        # Prevents extreme upscaling (e.g., 80px → 512px) which causes blurry features
        min_crop_dim = 300  # Minimum dimension in pixels
        
        # Tight crop: exact bbox OR expanded to minimum size
        if obj_size < min_crop_dim:
            # Small object - expand bbox to meet minimum while centering object
            expansion_needed = min_crop_dim - obj_size
            pad_x = expansion_needed // 2
            pad_y = expansion_needed // 2
            
            tx1 = max(0, bbox.x1 - pad_x)
            ty1 = max(0, bbox.y1 - pad_y)
            tx2 = min(w, bbox.x2 + pad_x)
            ty2 = min(h, bbox.y2 + pad_y)
            
            # If we hit image boundaries, expand the other side to maintain minimum size
            if tx2 - tx1 < min_crop_dim and tx2 < w:
                tx2 = min(w, tx1 + min_crop_dim)
            if tx2 - tx1 < min_crop_dim and tx1 > 0:
                tx1 = max(0, tx2 - min_crop_dim)
            if ty2 - ty1 < min_crop_dim and ty2 < h:
                ty2 = min(h, ty1 + min_crop_dim)
            if ty2 - ty1 < min_crop_dim and ty1 > 0:
                ty1 = max(0, ty2 - min_crop_dim)
            
            tight_bbox = BBox(x1=tx1, y1=ty1, x2=tx2, y2=ty2)
            tight_crop = img.crop((tx1, ty1, tx2, ty2))
            logger.info(
                f"Small object detected ({obj_size:.0f}px), expanded tight crop to "
                f"{tx2-tx1:.0f}×{ty2-ty1:.0f}px (prevents upscaling quality loss)"
            )
        else:
            # Normal sized object - use exact bbox
            tight_crop = img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            tight_bbox = bbox
        
        # Medium crop: ADAPTIVE PADDING based on object size
        # Smaller objects benefit from more surrounding context
        tight_w = tight_bbox.x2 - tight_bbox.x1
        tight_h = tight_bbox.y2 - tight_bbox.y1
        tight_size = max(tight_w, tight_h)
        
        if tight_size < 150:
            padding_ratio = 0.40  # 40% padding for very small objects
        elif tight_size < 300:
            padding_ratio = 0.20  # 20% padding for small-medium objects
        else:
            padding_ratio = 0.08  # 8% padding for larger objects
        
        pad = int(padding_ratio * tight_size)
        mx1 = max(0, tight_bbox.x1 - pad)
        my1 = max(0, tight_bbox.y1 - pad)
        mx2 = min(w, tight_bbox.x2 + pad)
        my2 = min(h, tight_bbox.y2 + pad)
        medium_crop = img.crop((mx1, my1, mx2, my2))
        medium_bbox = BBox(x1=mx1, y1=my1, x2=mx2, y2=my2)
        
        logger.debug(
            f"Crop dimensions: tight={tight_w:.0f}×{tight_h:.0f}px, "
            f"medium={mx2-mx1:.0f}×{my2-my1:.0f}px (padding={padding_ratio*100:.0f}%)"
        )
        
        # Full crop: entire image
        full_crop = img
        full_bbox = BBox(x1=0, y1=0, x2=w, y2=h)
        
        crops = {
            "tight": tight_crop,
            "medium": medium_crop,
            "full": full_crop
        }
        
        bboxes = {
            "tight": tight_bbox,
            "medium": medium_bbox,
            "full": full_bbox
        }
        
        return crops, bboxes
    
    def add_rotated_crops(self, base_crops: dict[str, Image.Image]) -> dict[str, Image.Image]:
        """
        Add rotation-augmented crops from base crops.
        
        This generates 90°, 180°, 270° rotations from the provided base crops.
        Use this AFTER masking to ensure rotations have the same masking as base crops.
        
        Args:
            base_crops: Dictionary with "tight", "medium", "full" crops
            
        Returns:
            Dictionary with base crops + rotated variants
        """
        crops = base_crops.copy()
        
        if self.enable_rotation_aug:
            logger.debug("Adding rotation-augmented crops for angle invariance")
            try:
                for angle in [90, 180, 270]:
                    # Rotate tight and medium crops (full stays as-is)
                    if "tight" in base_crops:
                        crops[f"tight_rot{angle}"] = base_crops["tight"].rotate(
                            angle, expand=True, resample=Image.Resampling.BILINEAR
                        )
                    if "medium" in base_crops:
                        crops[f"medium_rot{angle}"] = base_crops["medium"].rotate(
                            angle, expand=True, resample=Image.Resampling.BILINEAR
                        )
                logger.debug(f"Added {len([k for k in crops if 'rot' in k])} rotated crops")
            except Exception as e:
                logger.warning(f"Failed to generate rotated crops (non-critical): {e}")
        
        return crops

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

        # Medium crop with 8% padding for context
        pad = int(0.08 * max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1))
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