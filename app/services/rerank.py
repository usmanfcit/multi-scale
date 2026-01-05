from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.visual_matcher import VisualCrossEncoder


class RerankService(ABC):
    @abstractmethod
    def rerank(
        self,
        *,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        exact_first: bool,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass
class VisualCrossEncoderRerankService(RerankService):
    """
    Final human-like ranking using visual cross-encoder logic.
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
        if not query_vector:
            return []
        
        q = np.array(query_vector, dtype=np.float32)
        if q.shape[0] == 0:
            return []

        reranked = []
        for c in candidates:
            # Validate candidate has values and they're not empty
            if "values" not in c or not c["values"]:
                continue
            
            vec = np.array(c["values"], dtype=np.float32)
            
            # Skip if vector is empty or shape mismatch
            if vec.shape[0] == 0 or vec.shape[0] != q.shape[0]:
                continue

            score = self.matcher.score(
                q,
                vec,
                exact_weight=1.0 if exact_first else 0.6,
                semantic_weight=0.4 if exact_first else 1.0,
            )
            c["final_score"] = score
            reranked.append(c)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k]