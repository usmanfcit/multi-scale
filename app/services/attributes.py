from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass
class AttributeService:
    def parse_attributes_json(self, s: str | None) -> dict:
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}

    def build_filter(self, category: str | None) -> dict | None:
        if not category:
            return None
        # Normalize category to lowercase for consistent filtering
        # Pinecone metadata filter example: {"category": {"$eq": "bed"}}
        normalized_category = category.lower().strip()
        return {"category": {"$eq": normalized_category}}