from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4
import json

from fastapi import UploadFile

from app.core.errors import BadRequest
from app.models.schemas import BBox, CatalogUpsertResponse, SearchHit, SearchResponse
from app.services.image_io import ImageIOService
from app.services.preprocessing import PreprocessingService
from app.services.detection import DetectionService
from app.services.segmentation import SegmentationService
from app.services.embedding import EmbeddingService
from app.services.attributes import AttributeService
from app.services.rerank import RerankService
from app.repositories.pinecone_repo import PineconeVectorRepository


@dataclass
class SearchService:
    image_io: ImageIOService
    preprocessing: PreprocessingService
    detection: DetectionService
    segmentation: SegmentationService
    embedding: EmbeddingService
    attributes: AttributeService
    vectors: PineconeVectorRepository
    rerank: RerankService

    async def upsert_catalog_image(
        self,
        *,
         pinecone_id: str,
        assigned_category: str,
        image_file: UploadFile,
        metadata: dict[str, Any],
    ) -> CatalogUpsertResponse:
        from loguru import logger
        from app.utils.timing import timed
        
        logger.info(f"Upserting catalog item: ID={pinecone_id}, category={assigned_category}")
        
        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(image_file)
        
        w, h = img.size
        logger.debug(f"Catalog image size: {w}x{h}")

        # Run detection using assigned_category as hint
        mask_polygon = None
        
        with timed("Detection"):
            detections = await self.detection.detect(img, category_hint=assigned_category)
        
        logger.info(f"Detected {len(detections)} objects")
        
        # STRICT: Reject if no detection (no fallback, no manual bbox)
        if not detections:
            raise BadRequest(
                f"No objects detected in image for category '{assigned_category}'. "
                f"Cannot ingest product without detection. "
                f"Please ensure the image contains a clear view of the product with good lighting."
            )
        
        # Use detected bbox and polygon (no category validation, no manual bbox)
        det = detections[0]  # Best detection
        logger.debug(f"Top detection: {det.category} (score: {det.score:.3f})")
        
        bbox = BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3])
        mask_polygon = getattr(det, 'mask_polygon', None)
        
        logger.info(
            f"Using detected bbox: {bbox} "
            f"(category: {det.category}, score: {det.score:.3f})"
        )
        
        if mask_polygon:
            logger.info(f"Polygon mask available with {len(mask_polygon)} points")
        
        # Segment the product to remove background
        # If polygon is available (from RF-DETR), use it; otherwise use SAM2.1/GrabCut
        with timed("Segmentation"):
            # Check if segmentation service supports polygon
            if hasattr(self.segmentation, 'segment') and mask_polygon:
                # PolygonSegmentationService supports mask_polygon parameter
                try:
                    segment = await self.segmentation.segment(img, bbox, mask_polygon=mask_polygon)
                except TypeError:
                    # Fallback if segmentation doesn't support polygon parameter
                    segment = await self.segmentation.segment(img, bbox)
            else:
                segment = await self.segmentation.segment(img, bbox)
        
        with timed("Crop generation and Masking"):
            # Generate base crops (tight, medium, full) and their bboxes
            base_crops, base_bboxes = self.preprocessing.crop_base(img, bbox)

            # Apply mask ONLY to tight crop
            base_crops["tight"] = self.preprocessing.apply_mask_on_crop(
                base_crops["tight"], segment.mask, bbox=base_bboxes["tight"]
            )
            # Medium crop remains unmasked with natural background
            # Full crop remains unmasked with natural background

            # Now generate rotations from the (masked) tight and (unmasked) medium base crops
            crops = self.preprocessing.add_rotated_crops(base_crops)
            
            # Log masking strategy
            masked_count = 1 + len([k for k in crops.keys() if k.startswith("tight_rot")])
            natural_count = 1 + len([k for k in crops.keys() if k.startswith("medium_rot")]) + 1  # +1 for full
            logger.debug(
                f"Masking strategy: {masked_count} tight crops (polygon masked), "
                f"{natural_count} medium+full crops (natural background)"
            )

        # DEBUG: Save all crops to debug directory for first 3 items
        try:
            from pathlib import Path
            debug_dir = Path("debug_crop_image")
            debug_dir.mkdir(exist_ok=True)
            
            # Count existing items to limit to first 3
            existing_files = list(debug_dir.glob("*_tight.jpg"))
            if len(existing_files) < 3:
                import time
                timestamp = int(time.time() * 1000)
                
                for crop_name, crop_img in crops.items():
                    debug_path = debug_dir / f"{pinecone_id}_{timestamp}_{crop_name}.jpg"
                    crop_img.save(debug_path, quality=95)
                
                logger.info(f"💾 Saved {len(crops)} debug crops to {debug_dir}/ for product {pinecone_id}")
        except Exception as e:
            logger.warning(f"Failed to save debug crops (non-critical): {e}")

        with timed("Embedding"):
            vector = self.embedding.embed_crops(crops)
        
        # Use pinecone_id directly as vector_id (no UUID generation)
        vector_id = pinecone_id

        # Note: Images are NOT saved to local disk - frontend uses external CDN URLs from metadata
        # All processing happens in-memory for performance
        
        # Metadata is already prepared with correct types from ingestion script
        # It includes: image_url, product_url, name_english, name_arabic, 
        #              category (from assigned_category), price_amount, price_unit, is_active,
        #              store_id, countries, store
        
        # Upsert to Pinecone using assigned_category as namespace
        with timed("Vector upsert"):
            self.vectors.upsert(
                vector_id=vector_id,
                vector=vector,
                metadata=metadata,
                namespace=assigned_category
            )
        
        logger.info(
            f"Successfully upserted: ID={pinecone_id}, namespace={assigned_category}"
        )

        return CatalogUpsertResponse(
            pinecone_id=pinecone_id,
            upserted=True,
            message=f"Successfully upserted product {pinecone_id} to namespace {assigned_category}"
        )

    async def search_room_image(
        self,
        *,
        room_image_file: UploadFile,
        assigned_category: str | None,
        top_k: int,
    ) -> SearchResponse:
        from loguru import logger
        from app.utils.timing import timed
        import numpy as np
        
        logger.info(f"Search request: category={assigned_category}, top_k={top_k}")

        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(room_image_file)
        
        w, h = img.size
        logger.debug(f"Image size: {w}x{h}")

        # Run detection (REQUIRED - no manual bbox, no fallback)
        mask_polygon = None
        
        with timed("Detection"):
            detections = await self.detection.detect(
                img, 
                category_hint=assigned_category.lower().strip() if assigned_category else None
            )
        
        logger.info(f"Detected {len(detections)} objects")
        
        # STRICT: Reject if no detection
        if not detections:
            raise BadRequest(
                "Could not detect any objects in the image. "
                "Please ensure the image shows a clear view of the product "
                "with good lighting and contrast."
            )
        
        # Use detected bbox and polygon (no manual override)
        det = detections[0]
        logger.debug(f"Top detection: {det.category} (score: {det.score:.3f})")
        
        bbox = BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3])
        mask_polygon = getattr(det, 'mask_polygon', None)
        query_category = det.category.lower().strip() if det.category else None
        
        logger.info(f"Using detected bbox and polygon mask from RF-DETR")
        if mask_polygon:
            logger.info(f"Polygon mask available with {len(mask_polygon)} points")
        
        try:
            bbox = self.preprocessing.clamp_bbox(bbox, w, h)
        except ValueError as e:
            raise BadRequest(f"Invalid bounding box: {str(e)}")
        
        # Segment with polygon mask if available (same as ingestion)
        with timed("Segmentation"):
            # Check if segmentation service supports polygon
            if hasattr(self.segmentation, 'segment') and mask_polygon:
                try:
                    segment = await self.segmentation.segment(img, bbox, mask_polygon=mask_polygon)
                    logger.debug("Used polygon-based segmentation")
                except TypeError:
                    # Fallback if segmentation doesn't support polygon parameter
                    segment = await self.segmentation.segment(img, bbox)
                    logger.debug("Fallback to SAM2/GrabCut segmentation")
            else:
                segment = await self.segmentation.segment(img, bbox)
                logger.debug("Used SAM2/GrabCut segmentation")

        # Generate crops with same pattern as ingestion: base crops first, then rotations
        with timed("Crop generation and Masking"):
            # Generate base crops (tight, medium, full) and their bboxes
            base_crops, base_bboxes = self.preprocessing.crop_base(img, bbox)

            # Apply mask ONLY to tight crop (same as ingestion)
            base_crops["tight"] = self.preprocessing.apply_mask_on_crop(
                base_crops["tight"], segment.mask, bbox=base_bboxes["tight"]
            )
            
            # Medium and full crops keep natural background (not masked)

            # Now generate rotations from the (masked) tight and (unmasked) medium base crops
            crops = self.preprocessing.add_rotated_crops(base_crops)
        
        # Validate crops are not empty
        for crop_name, crop_img in crops.items():
            if crop_img.size[0] == 0 or crop_img.size[1] == 0:
                raise BadRequest(
                    f"Invalid crop '{crop_name}': crop has zero size. "
                    f"Please check your bounding box coordinates."
                )

        # DEBUG: Save all crops to debug directory for visualization
        try:
            from pathlib import Path
            debug_dir = Path("debug_crops_search")
            debug_dir.mkdir(exist_ok=True)
            
            import time
            timestamp = int(time.time() * 1000)
            
            for crop_name, crop_img in crops.items():
                debug_path = debug_dir / f"{timestamp}_{crop_name}.jpg"
                crop_img.save(debug_path, quality=95)
            
            logger.info(f"💾 Saved {len(crops)} debug crops to {debug_dir}/ with timestamp {timestamp}")
        except Exception as e:
            logger.warning(f"Failed to save debug crops (non-critical): {e}")

        with timed("Embedding"):
            query_vector = self.embedding.embed_crops(crops)
        
        logger.debug(f"Query vector norm: {np.linalg.norm(query_vector):.3f}")

        # Adaptive candidate count based on catalog size for optimal recall
        # For large catalogs (50K+), retrieve more candidates to ensure relevant items aren't missed
        catalog_size = await self._estimate_catalog_size(query_category)
        
        if catalog_size > 50000:
            # Very large catalog (50K+): Retrieve more candidates
            candidate_multiplier = 40
            candidate_k = min(max(top_k * candidate_multiplier, 1000), 5000)
        elif catalog_size > 10000:
            # Large catalog (10K-50K): Moderate increase
            candidate_multiplier = 30
            candidate_k = min(max(top_k * candidate_multiplier, 500), 2000)
        else:
            # Small/Medium catalog (<10K): Standard retrieval
            candidate_multiplier = 15
            candidate_k = min(max(top_k * candidate_multiplier, 100), 1000)
        
        logger.info(
            f"Catalog size estimate: {catalog_size:,} products, "
            f"retrieving {candidate_k} candidates (multiplier: {candidate_multiplier}x)"
        )

        with timed("Vector search"):
            candidates = self.vectors.query(
                vector=query_vector,
                top_k=candidate_k,
                category=query_category,  # Use category-based namespace for fast search
            )
        
        logger.info(f"Retrieved {len(candidates)} candidates from Pinecone (namespace: {query_category or 'default'})")

        # Check if no candidates were found
        if not candidates:
            message = "No products found in the catalog"
            if query_category:
                message += f" matching the category '{query_category}'"
            message += ". Please add products to the catalog first."
            logger.warning(message)
            return SearchResponse(
                query_category=query_category,
                hits=[],
                message=message
            )

        with timed("Reranking"):
            # Get extra results for deduplication
            rerank_k = top_k * 3
            reranked = self.rerank.rerank(
                query_vector=query_vector,
                candidates=candidates,
                top_k=rerank_k,
                exact_first=True,
            )
        
        logger.info(f"Reranked to {len(reranked)} results")
        if reranked:
            logger.debug(
                f"Score range: {reranked[0]['final_score']:.3f} (best) to "
                f"{reranked[-1]['final_score']:.3f} (worst)"
            )

        # Check if reranking resulted in no hits
        if not reranked:
            message = "No relevant products found"
            if query_category:
                message += f" for category '{query_category}'"
            message += ". Try adjusting your search criteria or use a different image."
            logger.warning(message)
            return SearchResponse(
                query_category=query_category,
                hits=[],
                message=message
            )

        # Deduplicate by pinecone_id (since each product has unique pinecone_id, 
        # this effectively removes duplicate vector entries if any exist)
        seen_ids = {}
        for c in reranked:
            # Get pinecone_id from the vector id (stored in 'id' field)
            product_id = c.get("id")
            if not product_id:
                logger.warning("Skipping result without ID")
                continue
                
            if product_id not in seen_ids:
                seen_ids[product_id] = c
            elif c["final_score"] > seen_ids[product_id]["final_score"]:
                # Replace with higher scoring instance
                seen_ids[product_id] = c

        # Take top K unique products
        unique_reranked = sorted(
            seen_ids.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]
        
        logger.info(f"After deduplication: {len(unique_reranked)} unique products")

        # Build response hits with new metadata structure
        hits = []
        for c in unique_reranked:
            metadata = c.get("metadata", {})
            
            hits.append(
                SearchHit(
                    pinecone_id=c.get("id", "unknown"),
                    score=float(c["final_score"]),
                    image_url=metadata.get("image_url"),
                    product_url=metadata.get("product_url"),
                    name_english=metadata.get("name_english"),
                    name_arabic=metadata.get("name_arabic"),
                    category=metadata.get("category"),
                    price_amount=metadata.get("price_amount"),
                    price_unit=metadata.get("price_unit"),
                    is_active=metadata.get("is_active"),
                    store_id=metadata.get("store_id"),
                    countries=metadata.get("countries"),
                    store=metadata.get("store"),
                )
            )

        return SearchResponse(query_category=query_category, hits=hits)
    
    async def _estimate_catalog_size(self, category: str | None) -> int:
        """
        Estimate catalog size for the given category to optimize candidate retrieval.
        
        For large catalogs (50K+), we need to retrieve more candidates to ensure
        relevant items aren't missed during ANN search.
        
        Args:
            category: Category to estimate size for (None = all categories)
            
        Returns:
            Approximate number of products in the catalog
        """
        from loguru import logger
        
        try:
            # Get index stats from Pinecone
            stats = self.vectors.get_stats(category=category)
            vector_count = stats.get('total_vector_count', 1000)
            
            logger.debug(
                f"Catalog size for category '{category or 'all'}': {vector_count:,} vectors"
            )
            
            return vector_count
            
        except Exception as e:
            logger.warning(f"Failed to get catalog size (using default): {e}")
            # Fallback: assume medium catalog
            return 5000