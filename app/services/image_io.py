from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import UploadFile


@dataclass
class ImageIOService:
    """
    Image I/O service with advanced validation and quality checks.
    
    Features:
    - Comprehensive file validation (size, format, dimensions)
    - Automatic resizing for large images (prevents OOM)
    - Quality checks (detects blank/corrupt images)
    - Support for JPEG, PNG, WebP formats
    - Separate directories for catalog and search images
    - Graceful error handling with detailed messages
    
    Best Practices:
    - Always validate images before processing
    - Resize large images to prevent memory issues
    - Provide clear error messages for debugging
    """
    catalog_images_dir: Path = Path("images/catalog")
    search_images_dir: Path = Path("images/search")
    min_dimension: int = 100
    max_dimension: int = 4096
    max_file_size_mb: int = 10
    
    def __post_init__(self):
        """Ensure image directories exist"""
        self.catalog_images_dir.mkdir(parents=True, exist_ok=True)
        self.search_images_dir.mkdir(parents=True, exist_ok=True)
        from loguru import logger
        logger.info(f"ImageIOService initialized:")
        logger.info(f"  Catalog images: {self.catalog_images_dir.absolute()}")
        logger.info(f"  Search images: {self.search_images_dir.absolute()}")
    
    async def read_upload_as_rgb(self, file: UploadFile) -> Image.Image:
        """
        Read and validate uploaded image with comprehensive checks.
        
        Validation includes:
        - File size (prevents uploads that are too large)
        - Image format (JPEG, PNG, WebP, etc.)
        - Dimensions (min/max pixel size)
        - Image quality (detects blank/corrupt images)
        - Color mode conversion (ensures RGB)
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            PIL Image in RGB mode, resized if necessary
            
        Raises:
            ValueError: If image is invalid, corrupted, or fails validation
        """
        from loguru import logger
        
        # Read file data
        data = await file.read()
        
        # Check file size (prevent OOM from huge files)
        size_mb = len(data) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(
                f"Image file too large ({size_mb:.1f}MB). "
                f"Maximum allowed: {self.max_file_size_mb}MB. "
                f"Please compress or resize your image."
            )
        
        logger.debug(f"Reading image file: {size_mb:.2f}MB")
        
        # Try to open image
        try:
            img = Image.open(BytesIO(data))
        except Exception as e:
            raise ValueError(
                f"Invalid or corrupted image file: {str(e)}. "
                f"Please ensure the file is a valid image (JPEG, PNG, WebP)."
            )
        
        # Log original format and mode
        logger.debug(f"Original image: format={img.format}, mode={img.mode}, size={img.size}")
        
        # Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
        try:
            if img.mode != "RGB":
                logger.debug(f"Converting from {img.mode} to RGB")
                img = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot convert image to RGB: {str(e)}")
        
        # Validate dimensions
        w, h = img.size
        
        if w < self.min_dimension or h < self.min_dimension:
            raise ValueError(
                f"Image too small ({w}x{h} pixels). "
                f"Minimum dimension: {self.min_dimension}px. "
                f"Please use a higher resolution image."
            )
        
        # Resize if too large (prevents OOM and speeds up processing)
        if w > self.max_dimension or h > self.max_dimension:
            original_size = (w, h)
            logger.info(
                f"Image exceeds maximum dimension ({w}x{h}), "
                f"resizing to max {self.max_dimension}px"
            )
            img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
            logger.info(f"Resized from {original_size} to {img.size}")
        
        # Quality checks: Detect potentially problematic images
        try:
            img_array = np.array(img)
            
            # Check for uniform/blank images (very low variance)
            std_dev = img_array.std()
            if std_dev < 1.0:
                logger.warning(
                    f"Image appears to be uniform/blank (std={std_dev:.2f}). "
                    f"This may affect search quality."
                )
                # Don't reject, but warn user
            
            # Check for extremely dark or bright images
            mean_brightness = img_array.mean()
            if mean_brightness < 10:
                logger.warning("Image is extremely dark, may affect detection quality")
            elif mean_brightness > 245:
                logger.warning("Image is extremely bright, may affect detection quality")
            
            logger.debug(
                f"Image quality metrics: std={std_dev:.1f}, "
                f"brightness={mean_brightness:.1f}, "
                f"size={img.size}"
            )
            
        except Exception as e:
            logger.debug(f"Quality check skipped: {e}")
        
        return img

    def pil_to_numpy_rgb(self, img: Image.Image) -> np.ndarray:
        return np.array(img, dtype=np.uint8)
    
    async def save_image(
        self, 
        image_id: str, 
        image_file: UploadFile, 
        image_type: str = "catalog"
    ) -> Path:
        """
        Save uploaded image file to disk.
        
        Args:
            image_id: Unique identifier for the image
            image_file: Uploaded file from FastAPI
            image_type: Type of image - "catalog" or "search" (default: "catalog")
            
        Returns:
            Path to the saved image file
        """
        from loguru import logger
        
        # Select appropriate directory based on image type
        if image_type == "search":
            target_dir = self.search_images_dir
        else:
            target_dir = self.catalog_images_dir
        
        # Reset file pointer to beginning
        await image_file.seek(0)
        data = await image_file.read()
        
        # Determine file extension from content type or filename
        content_type = image_file.content_type or ""
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        else:
            # Try to infer from filename
            filename = image_file.filename or ""
            if filename.lower().endswith((".jpg", ".jpeg")):
                ext = ".jpg"
            elif filename.lower().endswith(".png"):
                ext = ".png"
            elif filename.lower().endswith(".webp"):
                ext = ".webp"
            else:
                ext = ".jpg"  # Default to jpg
        
        image_path = target_dir / f"{image_id}{ext}"
        image_path.write_bytes(data)
        logger.debug(f"Saved {image_type} image: {image_path}")
        return image_path
    
    def get_image_path(self, image_id: str, image_type: str | None = None) -> Path | None:
        """
        Get path to image file by image_id.
        
        Args:
            image_id: Unique identifier for the image
            image_type: Type of image - "catalog", "search", or None to search both
            
        Returns:
            Path to the image file if found, None otherwise
        """
        # Determine which directories to search
        if image_type == "search":
            directories = [self.search_images_dir]
        elif image_type == "catalog":
            directories = [self.catalog_images_dir]
        else:
            # Search both directories (catalog first, then search)
            directories = [self.catalog_images_dir, self.search_images_dir]
        
        # Try common extensions in each directory
        for directory in directories:
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                image_path = directory / f"{image_id}{ext}"
                if image_path.exists():
                    return image_path
        return None