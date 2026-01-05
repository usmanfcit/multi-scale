from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VisualCrossEncoder:
    """
    Improved visual similarity scorer with separate instance and semantic components.
    Provides human-like ranking by weighing fine-grained and coarse features separately.
    """

    def score(
        self,
        query_vec: np.ndarray,
        candidate_vec: np.ndarray,
        *,
        exact_weight: float,
        semantic_weight: float,
    ) -> float:
        """
        Dual-objective scoring with separate instance/semantic handling.
        
        Args:
            query_vec: Fused embedding [instance_features | semantic_features]
            candidate_vec: Fused embedding [instance_features | semantic_features]
            exact_weight: Weight for instance matching (fine-grained details)
            semantic_weight: Weight for semantic matching (coarse style/category)
            
        Returns:
            Combined score with non-linear scaling for better discrimination
        """
        # Split vector into instance and semantic components
        # Assuming first half is instance (ViT), second half is semantic (CLIP)
        mid = len(query_vec) // 2
        
        query_inst = query_vec[:mid]
        query_sem = query_vec[mid:]
        cand_inst = candidate_vec[:mid]
        cand_sem = candidate_vec[mid:]
        
        # Normalize components (should already be normalized, but ensure it)
        query_inst = query_inst / (np.linalg.norm(query_inst) + 1e-12)
        query_sem = query_sem / (np.linalg.norm(query_sem) + 1e-12)
        cand_inst = cand_inst / (np.linalg.norm(cand_inst) + 1e-12)
        cand_sem = cand_sem / (np.linalg.norm(cand_sem) + 1e-12)
        
        # Compute separate similarities
        inst_sim = float(np.dot(query_inst, cand_inst))
        sem_sim = float(np.dot(query_sem, cand_sem))
        
        # Weighted combination
        final_score = (exact_weight * inst_sim) + (semantic_weight * sem_sim)
        
        # Apply non-linear scaling to spread out scores
        # This helps distinguish between good and great matches
        final_score = self._apply_scaling(final_score)
        
        return final_score
    
    def _apply_scaling(self, score: float) -> float:
        """
        Apply non-linear scaling to spread out the score distribution.
        Makes the ranking more discriminative.
        
        Maps similarity scores to emphasize differences in the high-quality range.
        """
        if score < 0:
            return 0.0
        
        # Power scaling (makes high scores higher, low scores lower)
        # This creates more separation between results
        scaled = score ** 1.5
        
        return min(1.0, scaled)