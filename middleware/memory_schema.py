"""
middleware/memory_schema.py
===========================
Defines the MemoryObject dataclass — the core unit of storage for the
Living Memory write path.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MemoryObject:
    """
    Represents a single memory unit stored in ChromaDB.

    Attributes:
        text:             The raw memory content.
        role:             Who said it — "user" or "assistant".
        session_id:       Which conversation session this belongs to.
        importance_score: Normalised importance (0.0-1.0), from LLM score / 10.
        memory_type:      Category e.g. "episodic", "semantic", "procedural".
        status:           "active" or "archived".
        is_contradiction: True if this memory contradicted an older one.
        contradicts_id:   Full ID of the archived memory this contradicted, or None.
        llm_reasoning:    One-sentence reasoning string returned by Job 3 LLM.
        access_count:     How many times this memory has been retrieved.
        last_accessed_at: Timestamp of most recent retrieval — used for decay.
        id:               Auto-generated UUID (set at creation time).
        created_at:       ISO-format timestamp (set at creation time).
    """

    # ── Required fields (must be passed explicitly) ───────────────────────────
    text:             str
    role:             str
    session_id:       str
    importance_score: float

    # ── Optional fields with sensible defaults ────────────────────────────────
    memory_type:      str           = "episodic"
    status:           str           = "active"
    is_contradiction: bool          = False
    contradicts_id:   Optional[str] = None
    llm_reasoning:    str           = ""

    # ── Access tracking — used by the decay formula in the read path ──────────
    # Every time a memory is retrieved, access_count increments and
    # last_accessed_at refreshes. Frequently used memories decay more slowly.
    # Untouched memories fade faster. This is the spacing effect from
    # human memory research applied to AI agents.
    access_count:     int = 0
    last_accessed_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    # ── Auto-generated — do not pass manually ─────────────────────────────────
    id:         str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_metadata(self) -> dict:
        """
        Serialises all fields (except text) into a flat dict suitable for
        ChromaDB metadata. ChromaDB metadata values must be str, int, float,
        or bool — no None allowed, so we convert None to empty string.
        """
        return {
            "role":             self.role,
            "session_id":       self.session_id,
            "importance_score": self.importance_score,
            "memory_type":      self.memory_type,
            "status":           self.status,
            "is_contradiction": self.is_contradiction,
            "contradicts_id":   self.contradicts_id or "",
            "llm_reasoning":    self.llm_reasoning,
            "access_count":     self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "created_at":       self.created_at,
        }

    @classmethod
    def from_chromadb(cls, doc: str, meta: dict, memory_id: str) -> "MemoryObject":
        """
        Reconstructs a MemoryObject from a ChromaDB query result.
        Used in the read path when fetching stored memories.
        """
        return cls(
            id               = memory_id,
            text             = doc,
            role             = meta.get("role", "user"),
            session_id       = meta.get("session_id", ""),
            importance_score = float(meta.get("importance_score", 0.5)),
            memory_type      = meta.get("memory_type", "episodic"),
            status           = meta.get("status", "active"),
            is_contradiction = bool(meta.get("is_contradiction", False)),
            contradicts_id   = meta.get("contradicts_id") or None,
            llm_reasoning    = meta.get("llm_reasoning", ""),
            access_count     = int(meta.get("access_count", 0)),
            last_accessed_at = meta.get("last_accessed_at", datetime.now().isoformat()),
            created_at       = meta.get("created_at", datetime.now().isoformat()),
        )