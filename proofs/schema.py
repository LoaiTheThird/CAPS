from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class ProofNode(BaseModel):
    """
    A single node in a proof tree.

    id:        unique within the proof
    text:      natural-language content (fact, intermediate claim, or hypothesis)
    premises:  list of node IDs this node depends on
    rule:      (optional) label for the inference rule (e.g. 'lexical', 'AND', 'world-knowledge')
    """
    id: str
    text: str
    premises: List[str] = []
    rule: Optional[str] = None


class ProofTree(BaseModel):
    """
    A full proof tree for a single example (e.g. one EntailmentBank item).
    """
    id: str                  # e.g. 'toy-001' or an EntailmentBank id
    hypothesis_id: str       # node id of the root hypothesis
    nodes: List[ProofNode]

    def to_dict(self) -> dict:
        """Convenience helper for JSON serialization."""
        return self.model_dump()
