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
        sku_id: str,
        category: str,
        image_file: UploadFile,
        attributes_json: str | None,
    ) -> CatalogUpsertResponse:
        from loguru import logger
        from app.utils.timing import timed
        
        logger.info(f"Upserting catalog item: SKU={sku_id}, category={category}")
        
        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(image_file)
        
        w, h = img.size
        logger.debug(f"Catalog image size: {w}x{h}")

        # Use full image bbox for catalog products
        bbox = BBox(x1=0, y1=0, x2=w, y2=h)
        
        # Segment the product to remove background (consistent with search pipeline)
        with timed("Segmentation"):
            segment = await self.segmentation.segment(img, bbox)
        
        with timed("Crop generation"):
            crops = self.preprocessing.crop_multi(img, bbox)
            # Apply mask to tight crop for clean product features
            crops["tight"] = self.preprocessing.apply_mask_on_crop(
                crops["tight"], segment.mask, bbox=bbox
            )

        with timed("Embedding"):
            vector = self.embedding.embed_crops(crops)
        
        attrs = self.attributes.parse_attributes_json(attributes_json)

        image_id = str(uuid4())
        vector_id = f"{sku_id}:{image_id}"

        # Save the image file to disk (reset file pointer first)
        await image_file.seek(0)
        await self.image_io.save_image(image_id, image_file, image_type="catalog")

        # Pinecone metadata only accepts primitive types (string, number, boolean, list of strings)
        # So we need to serialize the attributes dict as a JSON string
        attributes_str = json.dumps(attrs) if attrs else None
        
        metadata = {
            "sku_id": sku_id,
            "image_id": image_id,
            "category": category,  # Category is already normalized at the endpoint level
        }
        # Only include attributes if it's not empty
        if attributes_str:
            metadata["attributes"] = attributes_str
        
        # Upsert to Pinecone - will use category as namespace automatically
        with timed("Vector upsert"):
            self.vectors.upsert(vector_id=vector_id, vector=vector, metadata=metadata)
        
        logger.info(f"Successfully upserted: SKU={sku_id}, image_id={image_id}, namespace={category}")

        return CatalogUpsertResponse(sku_id=sku_id, image_id=image_id, upserted=True)

    async def search_room_image(
        self,
        *,
        room_image_file: UploadFile,
        category: str | None,
        bbox: BBox | None,
        top_k: int,
    ) -> SearchResponse:
        from loguru import logger
        from app.utils.timing import timed
        import numpy as np
        
        logger.info(f"Search request: category={category}, has_bbox={bbox is not None}, top_k={top_k}")

        with timed("Image load"):
            img = await self.image_io.read_upload_as_rgb(room_image_file)
        
        w, h = img.size
        logger.debug(f"Image size: {w}x{h}")

        if bbox is None:
            with timed("Detection"):
                detections = await self.detection.detect(
                    img, 
                    category_hint=category.lower().strip() if category else None
                )
            
            logger.info(f"Detected {len(detections)} objects")
            if detections:
                logger.debug(f"Top detection: {detections[0].category} (score: {detections[0].score:.3f})")
            
            if not detections:
                # Fallback: use full image as bbox when detection fails
                logger.warning("No detections, using full image as fallback")
                bbox = BBox(x1=0, y1=0, x2=w, y2=h)
                query_category = category.lower().strip() if category else None
            else:
                det = detections[0]
                bbox = BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3])
                query_category = det.category.lower().strip() if det.category else (category.lower().strip() if category else None)
        else:
            query_category = category.lower().strip() if category else None
            logger.debug(f"Using provided bbox: {bbox}")

        try:
            bbox = self.preprocessing.clamp_bbox(bbox, w, h)
        except ValueError as e:
            raise BadRequest(f"Invalid bounding box: {str(e)}")
        
        with timed("Segmentation"):
            segment = await self.segmentation.segment(img, bbox)

        with timed("Crop generation"):
            crops = self.preprocessing.crop_multi(img, bbox)
            # Apply mask to tight crop with bbox parameter
            crops["tight"] = self.preprocessing.apply_mask_on_crop(
                crops["tight"], segment.mask, bbox=bbox
            )
        
        # Validate crops are not empty
        for crop_name, crop_img in crops.items():
            if crop_img.size[0] == 0 or crop_img.size[1] == 0:
                raise BadRequest(
                    f"Invalid crop '{crop_name}': crop has zero size. "
                    f"Please check your bounding box coordinates."
                )

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

        # Deduplicate by SKU, keeping highest score for each
        seen_skus = {}
        for c in reranked:
            sku = c["metadata"]["sku_id"]
            if sku not in seen_skus:
                seen_skus[sku] = c
            elif c["final_score"] > seen_skus[sku]["final_score"]:
                # Replace with higher scoring instance
                seen_skus[sku] = c

        # Take top K unique SKUs
        unique_reranked = sorted(
            seen_skus.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]
        
        logger.info(f"After deduplication: {len(unique_reranked)} unique SKUs")

        # Build response hits
        hits = []
        for c in unique_reranked:
            metadata = c["metadata"]
            # Parse attributes from JSON string back to dict
            attributes = None
            if "attributes" in metadata and metadata["attributes"]:
                try:
                    attributes = json.loads(metadata["attributes"])
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, set to None
                    attributes = None
            
            hits.append(
                SearchHit(
                    sku_id=metadata["sku_id"],
                    image_id=metadata.get("image_id"),
                    category=metadata.get("category"),
                    attributes=attributes,
                    score=float(c["final_score"]),
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