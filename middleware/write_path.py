"""
middleware/write_path.py
========================
The write path — decides what gets saved to memory and how.

Flow:
  1. Embed the new memory
  2. Run Job 2: math filter + collision override
  3. Run Job 3: single LLM call (importance + contradiction)
  4. Save or discard based on results

This is your core thesis contribution for Phase 3.
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import chromadb
from chromadb.utils import embedding_functions

from middleware.memory_schema import MemoryObject

load_dotenv()

# ── Groq cloud client — qwen-3.6-27b via Groq API ────────────────────────────
GROQ_MODEL   = os.getenv("GROQ_MODEL", "qwen-3.6-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)


def _groq_call(fn, *args, retries: int = 4, **kwargs):
    """
    Retry wrapper with exponential back-off for Groq rate-limit (429) errors.
    Normal users on the free tier won't see a crash — just a short pause.
    """
    delay = 2.0
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            if attempt == retries - 1:
                raise
            print(f"  [Groq] Rate limit hit — retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{retries})")
            time.sleep(delay)
            delay *= 2   # 2 → 4 → 8 → 16 seconds

# ── Local embedding function (no API cost, runs on your machine) ──────────────
embed_fn = embedding_functions.DefaultEmbeddingFunction()

# ── ChromaDB setup ────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./data/middleware_db")
collection = chroma_client.get_or_create_collection(
    name="living_memory",
    embedding_function=embed_fn,       # Chroma handles vectors automatically
    metadata={"hnsw:space": "cosine"}
)

# ── Tunable thresholds (these become your experimental variables) ─────────────
MATH_FILTER_THRESHOLD   = 0.35   # minimum math score to pass to LLM
COLLISION_HIGH          = 0.75   # above this → send to LLM regardless of math
EXACT_DUPLICATE_CUTOFF  = 0.98   # above this → obvious duplicate, discard immediately
LLM_IMPORTANCE_CUTOFF   = 5     # LLM score below this → discard
ALPHA                   = 0.4    # weight for semantic density
BETA                    = 0.6    # weight for novelty  (alpha + beta must = 1.0)
DENSITY_WINDOW          = 5      # how many recent memories to use for density calc


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def embed(text: str) -> list[float]:
    """
    Convert text to a vector using ChromaDB's built-in sentence-transformers.
    Runs locally — no API key needed, no cost, no internet required.
    embed_fn returns a list of lists, so we grab index [0] for the single text.
    """
    return embed_fn([text])[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns a value between 0 (totally different) and 1 (identical).
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_recent_embeddings(session_id: str, n: int = DENSITY_WINDOW) -> list[list[float]]:
    """
    Fetch the embeddings of the N most recent active memories in this session.
    Used to calculate semantic density — how on-topic is the new memory?
    """
    if collection.count() == 0:
        return []

    results = collection.get(
        where={"$and": [
            {"session_id": {"$eq": session_id}},
            {"status": {"$eq": "active"}}
        ]},
        include=["embeddings", "metadatas"]
    )

    if len(results["embeddings"]) == 0:
        return []

    # Sort by created_at descending and take the most recent N
    paired = list(zip(
        results["embeddings"],
        [m["created_at"] for m in results["metadatas"]]
    ))
    paired.sort(key=lambda x: x[1], reverse=True)

    return [emb for emb, _ in paired[:n]]


def get_top_similar(
    new_embedding: list[float],
    session_id: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Find the top-K most similar active memories to the new memory.
    Used for both collision detection and contradiction checking.
    Returns list of dicts with id, text, similarity.
    """
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[new_embedding],
        n_results=min(top_k, collection.count()),
        where={"$and": [
            {"status": {"$eq": "active"}},
            {"session_id": {"$eq": session_id}},
        ]},
        include=["documents", "metadatas", "distances"]
    )

    similar = []
    for doc, meta, dist, mem_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0]
    ):
        similar.append({
            "id":         mem_id,
            "text":       doc,
            "metadata":   meta,
            "similarity": round(1 - dist, 4)
        })

    return similar


# ══════════════════════════════════════════════════════════════════════════════
# JOB 2 — MATH FILTER + COLLISION OVERRIDE
# ══════════════════════════════════════════════════════════════════════════════

def job2_math_filter(
    new_embedding: list[float],
    session_id: str,
    top_similar: list[dict]
) -> tuple[str, float]:
    """
    Applies the math filter with collision override logic.

    Returns:
        ("discard", score)       → memory is an obvious duplicate or too low
        ("llm", score)           → memory should go to LLM for evaluation
        ("collision_llm", score) → memory triggered collision override, send to LLM
    """

    # ── Fix 3: Cold start — database is empty, first memory auto-passes ─────────
    if not top_similar:
        print("  [Job 2] Cold start — empty database. First memory gets automatic pass.")
        return "llm", 0.60

    # ── Get max similarity against existing memories ──────────────────────────
    max_similarity = max(
        [m["similarity"] for m in top_similar], default=0.0
    )

    print(f"  [Job 2] Max similarity to existing memories: {max_similarity:.3f}")

    # ── Exact duplicate check — discard immediately ───────────────────────────
    if max_similarity >= EXACT_DUPLICATE_CUTOFF:
        print(f"  [Job 2] Exact duplicate detected ({max_similarity:.3f} ≥ {EXACT_DUPLICATE_CUTOFF}). Discarding.")
        return "discard", 0.0

    # ── Collision override — bypass math, go straight to LLM ─────────────────
    if max_similarity >= COLLISION_HIGH:
        print(f"  [Job 2] Collision override triggered ({max_similarity:.3f} >= {COLLISION_HIGH}).")
        print(f"  [Job 2] High similarity — could be contradiction or duplicate. Sending to LLM.")
        return "collision_llm", max_similarity

    # ── Standard math path ────────────────────────────────────────────────────
    # Semantic Density: how on-topic is this relative to recent conversation?
    recent_embeddings = get_recent_embeddings(session_id)
    if recent_embeddings:
        avg_recent = np.mean(recent_embeddings, axis=0).tolist()
        semantic_density = cosine_similarity(new_embedding, avg_recent)
    else:
        semantic_density = 0.5   # no history yet, assume neutral

    # Novelty: how new is this information?
    novelty = 1.0 - max_similarity

    # Composite score
    math_score = (ALPHA * semantic_density) + (BETA * novelty)

    print(f"  [Job 2] Semantic Density: {semantic_density:.3f} | Novelty: {novelty:.3f}")
    print(f"  [Job 2] Math Score: {math_score:.3f} (threshold: {MATH_FILTER_THRESHOLD})")

    if math_score < MATH_FILTER_THRESHOLD:
        print(f"  [Job 2] Score too low. Discarding.")
        return "discard", math_score

    print(f"  [Job 2] Passed math filter. Sending to LLM.")
    return "llm", math_score


# ══════════════════════════════════════════════════════════════════════════════
# JOB 3 — SINGLE LLM CALL (IMPORTANCE + CONTRADICTION)
# ══════════════════════════════════════════════════════════════════════════════

def job3_llm_evaluate(
    new_text: str,
    top_similar: list[dict],
    is_collision: bool = False
) -> dict:
    """
    Single LLM call that evaluates importance AND contradiction at once.

    Uses chain-of-thought: forces a "reasoning" field before the boolean
    so the LLM thinks before committing to true/false.

    Args:
        new_text: the new memory text
        top_similar: top-3 most similar existing memories
        is_collision: True if this was triggered by collision override
                      (tells LLM to focus more on contradiction detection)

    Returns:
        dict with keys: reasoning, importance, contradiction, contradicts_id
    """

    # Format existing memories for the prompt
    existing_block = "\n".join([
        f"[ID: {m['id'][:8]}...] {m['text']}"
        for m in top_similar
    ]) if top_similar else "None"

    collision_note = (
        "\nIMPORTANT: This memory was flagged as highly similar to an existing one. "
        "Pay extra attention to whether it is a contradiction or just a rephrasing."
        if is_collision else ""
    )

    prompt = f"""You are a memory evaluation engine for an AI assistant.
You will be given a NEW memory and the TOP 3 most similar existing memories.
Your job is to evaluate the new memory on two dimensions.{collision_note}

NEW MEMORY:
\"{new_text}\"

TOP 3 SIMILAR EXISTING MEMORIES:
{existing_block}

Respond ONLY with valid JSON in exactly this format — no extra text:
{{
  "reasoning": "<one sentence: explain your evaluation logic>",
  "importance": <integer 0-10>,
  "contradiction": <true or false>,
  "contradicts_id": "<full ID of contradicted memory or null>"
}}

RULES:
- importance: How valuable is this for a long-term personal assistant to remember?
  * 8-10: Critical facts — names, decisions, strong preferences, deadlines
  * 5-7:  Useful context — hobbies, work topics, recurring themes  
  * 2-4:  Minor details — one-off mentions, filler, small talk
  * 0-1:  Worthless — pleasantries, confirmations ("ok", "thanks", "yes")
- contradiction: true ONLY if the new memory directly conflicts with an existing one
  (not just similar — they must be logically incompatible)
- contradicts_id: the FULL id of the memory being contradicted, or null
"""

    response = _groq_call(
        llm_client.chat.completions.create,
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,          # low = consistent output
        max_tokens=300,           # JSON response is short; cap keeps it fast
    )

    raw = response.choices[0].message.content.strip()

    # response_format guarantees valid JSON so this rarely fails,
    # but we keep the fallback for safety
    try:
        result = json.loads(raw)
        assert "importance"    in result
        assert "contradiction" in result
        assert "reasoning"     in result
        if (
            isinstance(result["importance"], bool)
            or not isinstance(result["importance"], (int, float))
            or not 0 <= result["importance"] <= 10
            or not isinstance(result["contradiction"], bool)
            or not isinstance(result["reasoning"], str)
        ):
            raise ValueError("invalid field types or ranges")
        result.setdefault("contradicts_id", None)
        if result["contradicts_id"] is not None and not isinstance(
            result["contradicts_id"], str
        ):
            raise ValueError("invalid contradicts_id")
        return result
    except (json.JSONDecodeError, AssertionError, ValueError, TypeError):
        print(f"  [Job 3] Warning: unexpected response. Raw: {raw[:100]}")
        return {
            "reasoning":      "fallback — unexpected response format",
            "importance":     5,
            "contradiction":  False,
            "contradicts_id": None
        }


# ══════════════════════════════════════════════════════════════════════════════
# ARCHIVE — Mark old memory as outdated
# ══════════════════════════════════════════════════════════════════════════════

def archive_memory(memory_id: str, reason: str = "contradicted") -> bool:
    """
    Archives an existing memory by updating its status to 'archived'.
    Does NOT delete it — keeps it for thesis audit trail and analysis.
    """
    try:
        existing = collection.get(ids=[memory_id], include=["metadatas"])
        if not existing["metadatas"]:
            print(f"  [Archive] Memory {memory_id[:8]}... not found.")
            return False

        meta = existing["metadatas"][0]
        meta["status"] = "archived"
        meta["archived_reason"] = reason
        meta["archived_at"] = datetime.now().isoformat()

        collection.update(
            ids=[memory_id],
            metadatas=[meta]
        )
        print(f"  [Archive] Memory {memory_id[:8]}... archived (reason: {reason})")
        return True
    except Exception as e:
        print(f"  [Archive] Error archiving {memory_id[:8]}...: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SAVE — Commit memory to ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

def save_memory(memory: MemoryObject, embedding: list[float]):
    """Saves a MemoryObject to ChromaDB with its embedding and metadata."""
    collection.add(
        ids=[memory.id],
        embeddings=[embedding],
        documents=[memory.text],
        metadatas=[memory.to_metadata()]
    )
    print(f"  [Save] Memory saved: '{memory.text[:60]}...' | Score: {memory.importance_score:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WRITE PATH — entry point
# ══════════════════════════════════════════════════════════════════════════════

def process_memory(
    text: str,
    role: str,
    session_id: str,
    created_at: str | None = None,
    last_accessed_at: str | None = None,
) -> str:
    """
    Full write path pipeline for one conversation turn.

    Args:
        text: the message content
        role: "user" or "assistant"
        session_id: current session identifier
        created_at: optional timestamp override for evaluation
        last_accessed_at: optional timestamp override for evaluation

    Returns:
        "saved"      → memory was saved normally
        "updated"    → memory was saved, an old one was archived
        "discarded"  → memory was not worth saving
        "duplicate"  → exact duplicate detected
        "error"      → memory update could not be completed safely
    """

    print(f"\n[Write Path] Processing: '{text[:60]}...'")

    # ── Step 1: Embed the new memory ──────────────────────────────────────────
    new_embedding = embed(text)

    # ── Step 2: Find top-3 similar existing memories ──────────────────────────
    top_similar = get_top_similar(new_embedding, session_id=session_id, top_k=3)

    # ── Step 3: Job 2 — Math filter + collision override ─────────────────────
    decision, math_score = job2_math_filter(new_embedding, session_id, top_similar)

    if decision == "discard":
        return "discarded"

    if decision == "duplicate":
        return "duplicate"

    is_collision = (decision == "collision_llm")

    # ── Step 4: Job 3 — Single LLM call ──────────────────────────────────────
    print(f"  [Job 3] Calling LLM for importance + contradiction check...")
    llm_result = job3_llm_evaluate(text, top_similar, is_collision=is_collision)

    print(f"  [Job 3] Reasoning: {llm_result['reasoning']}")
    print(f"  [Job 3] Importance: {llm_result['importance']}/10 | Contradiction: {llm_result['contradiction']}")

    # ── Step 5: Act on LLM result ─────────────────────────────────────────────

    # Importance too low — discard
    if llm_result["importance"] < LLM_IMPORTANCE_CUTOFF:
        print(f"  [Job 3] Importance too low ({llm_result['importance']} < {LLM_IMPORTANCE_CUTOFF}). Discarding.")
        return "discarded"

    # Normalise importance score to 0.0–1.0
    final_score = round(llm_result["importance"] / 10.0, 2)

    # Handle contradiction — archive old memory first
    full_id = None
    if llm_result["contradiction"] and not llm_result.get("contradicts_id"):
        print("  [Write Path] Contradiction was reported without a target. "
              "Rejecting replacement.")
        return "error"

    if llm_result["contradiction"] and llm_result["contradicts_id"]:
        full_id = None
        # Match short ID from LLM to full ID in top_similar
        for m in top_similar:
            if m["id"].startswith(llm_result["contradicts_id"][:8]):
                full_id = m["id"]
                break

        if full_id:
            if not archive_memory(full_id, reason="contradicted by new memory"):
                print("  [Write Path] Could not archive contradicted memory. "
                      "Rejecting replacement.")
                return "error"
        else:
            print("  [Write Path] Contradiction target did not match an active "
                  "memory. Rejecting replacement.")
            return "error"

    # Build and save the memory object
    memory = MemoryObject(
        text=text,
        role=role,
        session_id=session_id,
        importance_score=final_score,
        memory_type="episodic",
        status="active",
        is_contradiction=bool(llm_result["contradiction"]),
        contradicts_id=full_id if llm_result["contradiction"] else None,
        llm_reasoning=llm_result["reasoning"],
        last_accessed_at=last_accessed_at or datetime.now().isoformat(),
    )
    if created_at is not None:
        memory.created_at = created_at

    save_memory(memory, new_embedding)

    if llm_result["contradiction"]:
        return "updated"
    return "saved"


# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════

def get_write_path_stats() -> dict:
    """Returns memory database statistics for evaluation."""
    if collection.count() == 0:
        return {"total": 0, "active": 0, "archived": 0, "contradictions": 0}

    all_meta = collection.get(include=["metadatas"])["metadatas"]

    active        = sum(1 for m in all_meta if m.get("status") == "active")
    archived      = sum(1 for m in all_meta if m.get("status") == "archived")
    contradictions = sum(1 for m in all_meta if m.get("is_contradiction"))

    return {
        "total":          collection.count(),
        "active":         active,
        "archived":       archived,
        "contradictions": contradictions,
    }