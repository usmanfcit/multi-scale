from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from loguru import logger

from app.services.visual_matcher import VisualCrossEncoder


@dataclass
class MultiStageRerankService:
    """
    Advanced multi-stage reranking for large catalogs (50K+ products).
    
    Strategy:
    - Stage 1: Fast semantic filtering (top 30% candidates)
    - Stage 2: Fine-grained instance+semantic matching (final top_k)
    
    Features:
    - Handles 50K-1M+ products efficiently
    - Two-stage cascade reduces computation
    - Adaptive stage 1 cutoff based on catalog size
    - Non-linear scoring for better discrimination
    
    Best Practices:
    - Fast coarse filtering before detailed matching
    - Normalize all vectors before comparison
    - Use non-linear scaling for score distribution
    - Log performance metrics at each stage
    """
    matcher: VisualCrossEncoder

    def rerank(
        self,
        *,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        exact_first: bool,
    ) -> list[dict[str, Any]]:
        """
        Two-stage reranking for efficient and accurate results.
        
        Args:
            query_vector: Query embedding (fused instance+semantic)
            candidates: Candidate vectors from ANN search
            top_k: Number of final results to return
            exact_first: If True, prioritize exact matches
            
        Returns:
            Reranked candidates with final_score
        """
        if not query_vector or not candidates:
            logger.debug("Empty query or candidates, returning empty list")
            return []
        
        q = np.array(query_vector, dtype=np.float32)
        if q.shape[0] == 0:
            logger.error("Query vector has zero length")
            return []

        logger.debug(f"Starting two-stage reranking: {len(candidates)} candidates -> top {top_k}")

        # Stage 1: Fast semantic filtering
        # Only use semantic component for speed
        logger.debug(f"Stage 1: Semantic filtering {len(candidates)} candidates")
        
        stage1_results = []
        mid = len(q) // 2
        q_sem = q[mid:]
        q_sem_normalized = q_sem / (np.linalg.norm(q_sem) + 1e-12)
        
        for c in candidates:
            # Validate candidate
            if "values" not in c or not c["values"]:
                logger.debug(f"Skipping candidate without values: {c.get('id', 'unknown')}")
                continue
            
            vec = np.array(c["values"], dtype=np.float32)
            if vec.shape[0] == 0 or vec.shape[0] != q.shape[0]:
                logger.debug(
                    f"Skipping candidate with invalid vector: "
                    f"shape {vec.shape[0]} vs expected {q.shape[0]}"
                )
                continue
            
            # Quick semantic similarity (no instance features)
            cand_sem = vec[mid:]
            cand_sem_normalized = cand_sem / (np.linalg.norm(cand_sem) + 1e-12)
            sem_score = float(np.dot(q_sem_normalized, cand_sem_normalized))
            
            c["stage1_score"] = sem_score
            stage1_results.append(c)
        
        if not stage1_results:
            logger.warning("No valid candidates after stage 1 filtering")
            return []
        
        # Adaptive stage 1 cutoff based on catalog size
        # For large catalogs, keep more candidates; for small, keep fewer
        if len(stage1_results) > 10000:
            stage1_ratio = 0.2  # Keep top 20% for very large catalogs
        elif len(stage1_results) > 1000:
            stage1_ratio = 0.3  # Keep top 30% for large catalogs
        else:
            stage1_ratio = 0.5  # Keep top 50% for small/medium catalogs
        
        stage1_k = max(int(len(stage1_results) * stage1_ratio), top_k * 3, 50)
        stage1_results.sort(key=lambda x: x["stage1_score"], reverse=True)
        stage1_filtered = stage1_results[:stage1_k]
        
        logger.debug(
            f"Stage 1 complete: filtered {len(stage1_results)} -> {len(stage1_filtered)} "
            f"(ratio: {stage1_ratio:.1%}, min score: {stage1_filtered[-1]['stage1_score']:.3f})"
        )
        
        # Stage 2: Fine-grained instance+semantic matching
        logger.debug(f"Stage 2: Fine-grained matching on {len(stage1_filtered)} candidates")
        
        q_inst = q[:mid]
        q_inst_normalized = q_inst / (np.linalg.norm(q_inst) + 1e-12)
        
        reranked = []
        for c in stage1_filtered:
            vec = np.array(c["values"], dtype=np.float32)
            
            # Split into instance and semantic components
            cand_inst = vec[:mid]
            cand_inst_normalized = cand_inst / (np.linalg.norm(cand_inst) + 1e-12)
            cand_sem = vec[mid:]
            cand_sem_normalized = cand_sem / (np.linalg.norm(cand_sem) + 1e-12)
            
            # Compute separate similarities
            inst_sim = float(np.dot(q_inst_normalized, cand_inst_normalized))
            sem_sim = float(np.dot(q_sem_normalized, cand_sem_normalized))
            
            # Weighted combination based on use case
            if exact_first:
                # Prioritize exact matches (fine-grained features)
                exact_weight = 0.7
                semantic_weight = 0.3
            else:
                # Balanced approach
                exact_weight = 0.5
                semantic_weight = 0.5
            
            final_score = (exact_weight * inst_sim) + (semantic_weight * sem_sim)
            
            # Apply non-linear scaling for better discrimination
            final_score = self._apply_scaling(final_score)
            
            # Store component scores for debugging
            c["inst_score"] = inst_sim
            c["sem_score"] = sem_sim
            c["final_score"] = final_score
            reranked.append(c)

        # Sort by final score
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        
        logger.debug(
            f"Stage 2 complete: {len(reranked)} results, "
            f"score range: {reranked[0]['final_score']:.3f} (best) to "
            f"{reranked[-1]['final_score']:.3f} (worst)"
        )
        
        # Log top result details for debugging
        if reranked:
            top = reranked[0]
            logger.debug(
                f"Top result: final={top['final_score']:.3f}, "
                f"inst={top.get('inst_score', 0):.3f}, "
                f"sem={top.get('sem_score', 0):.3f}, "
                f"sku={top.get('metadata', {}).get('sku_id', 'unknown')}"
            )
        
        return reranked[:top_k]
    
    def _apply_scaling(self, score: float) -> float:
        """
        Apply non-linear scaling to spread out score distribution.
        Makes ranking more discriminative by emphasizing differences.
        
        Uses power scaling: score^1.5
        - High scores (0.9) -> very high (0.95)
        - Medium scores (0.7) -> lower (0.58)
        - Low scores (0.5) -> very low (0.35)
        
        Args:
            score: Raw similarity score [0, 1]
            
        Returns:
            Scaled score [0, 1]
        """
        if score < 0:
            return 0.0
        
        # Power scaling (makes high scores higher, low scores lower)
        # This creates more separation between good and mediocre results
        scaled = score ** 1.5
        
        return min(1.0, scaled)


@dataclass
class HybridRerankService:
    """
    Hybrid reranking with automatic fallback.
    
    Uses MultiStageRerankService for large catalogs (>1000 candidates),
    falls back to simple reranking for small catalogs.
    
    Best for: Production environments with varying catalog sizes
    """
    matcher: VisualCrossEncoder
    
    def __post_init__(self):
        self._multistage = MultiStageRerankService(matcher=self.matcher)
    
    def rerank(
        self,
        *,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        exact_first: bool,
    ) -> list[dict[str, Any]]:
        """
        Adaptive reranking based on catalog size.
        
        Args:
            query_vector: Query embedding
            candidates: Candidate vectors
            top_k: Number of results
            exact_first: Prioritize exact matches
            
        Returns:
            Reranked candidates
        """
        # Use multi-stage for large candidate sets
        if len(candidates) > 1000:
            logger.debug("Using multi-stage reranking for large catalog")
            return self._multistage.rerank(
                query_vector=query_vector,
                candidates=candidates,
                top_k=top_k,
                exact_first=exact_first,
            )
        else:
            # Simple reranking for small catalogs
            logger.debug("Using simple reranking for small catalog")
            return self._simple_rerank(
                query_vector=query_vector,
                candidates=candidates,
                top_k=top_k,
                exact_first=exact_first,
            )
    
    def _simple_rerank(
        self,
        *,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        exact_first: bool,
    ) -> list[dict[str, Any]]:
        """Simple single-stage reranking for small catalogs."""
        if not query_vector or not candidates:
            return []
        
        q = np.array(query_vector, dtype=np.float32)
        if q.shape[0] == 0:
            return []

        reranked = []
        for c in candidates:
            if "values" not in c or not c["values"]:
                continue
            
            vec = np.array(c["values"], dtype=np.float32)
            if vec.shape[0] == 0 or vec.shape[0] != q.shape[0]:
                continue

            # Use the visual matcher for scoring
            score = self.matcher.score(
                q,
                vec,
                exact_weight=0.7 if exact_first else 0.5,
                semantic_weight=0.3 if exact_first else 0.5,
            )
            c["final_score"] = score
            reranked.append(c)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k]

