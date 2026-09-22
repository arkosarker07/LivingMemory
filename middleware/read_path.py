"""
middleware/read_path.py
=======================
The read path — decides what gets retrieved and injected into the LLM prompt.

Formula:
    U = S * [ I(m) * max(floor, e^(-λ * Δt)) * (1 + log(1 + access_count)) ]

Where:
    S            = Semantic Similarity (Cosine match to current query)
    I(m)         = Importance score (0.0 – 1.0)
    e^(-λ * Δt)  = Exponential decay over time (Δt in hours)
    floor        = Minimum decay floor (memories never fully drop to 0)
    log bonus    = Spacing effect (frequently accessed memories decay slower)
"""

import os
import math
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import chromadb
from chromadb.utils import embedding_functions

from middleware.memory_schema import MemoryObject

load_dotenv()

GROQ_MODEL  = os.getenv("GROQ_MODEL", "qwen-3.6-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)


def _groq_call(fn, *args, retries: int = 4, **kwargs):
    """
    Thin retry wrapper around any llm_client.chat.completions.create call.
    Handles Groq rate-limit (429) with exponential back-off so a normal
    user never sees a crash — just a brief pause.
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

# ── Local embeddings (must match write path) ──────────────────────────────────
embed_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./data/middleware_db")
collection = chroma_client.get_or_create_collection(
    name="living_memory",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

# ── Tunable parameters (Thesis Experimental Variables) ────────────────────────
CANDIDATE_POOL_K  = 20       # fetch wide pool before filtering
TOP_N_INJECT      = 5        # max memories to inject into prompt
UTILITY_THRESHOLD = 0.10     # below this = dormant (forgotten for now)

DECAY_LAMBDA      = 0.001    # λ: 0.001 = ~30 day half-life (slower decay)
DECAY_FLOOR       = 0.20     # Memories never decay below 20% of their importance


def embed(text: str) -> list[float]:
    """Convert text to vector."""
    return embed_fn([text])[0]


def compute_utility(memory: MemoryObject, now: datetime, similarity: float) -> float:
    """
    Computes the dynamic utility score for a memory.
    """
    try:
        last_accessed = datetime.fromisoformat(memory.last_accessed_at)
        if last_accessed.tzinfo is not None:
            now_aware = now.replace(tzinfo=timezone.utc)
            delta_hours = (now_aware - last_accessed).total_seconds() / 3600
        else:
            delta_hours = (now - last_accessed).total_seconds() / 3600
    except Exception:
        delta_hours = 0.0

    # Ensure delta_hours is never negative due to microsecond time-drift
    delta_hours = max(0.0, delta_hours)

    importance = memory.importance_score
    
    # Exponential decay with a biological floor
    decay_factor = max(DECAY_FLOOR, math.exp(-DECAY_LAMBDA * delta_hours))
    
    # Frequency bonus (Spacing effect)
    frequency_bonus = 1 + math.log(1 + memory.access_count)

    # FINAL UTILITY: Similarity * (Importance * Decay * Bonus)
    utility = similarity * (importance * decay_factor * frequency_bonus)

    return round(utility, 4)


def retrieve_memories(
    query: str,
    session_id: str,
    simulate_hours_passed: float = 0.0,
) -> list[tuple[MemoryObject, float]]:
    """
    Retrieves, scores, and filters memories for the LLM.
    simulate_hours_passed allows time-travel testing for the thesis.
    """
    if collection.count() == 0:
        return []

    # Fast-forward clock for testing decay
    now = datetime.now()
    if simulate_hours_passed > 0:
        now += timedelta(hours=simulate_hours_passed)

    # 1. Fetch Candidates
    query_embedding = embed(query)
    n_candidates = min(CANDIDATE_POOL_K, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
        where={"$and": [
            {"status": {"$eq": "active"}},
            {"session_id": {"$eq": session_id}},
        ]},
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        return []

    # 2. Score Candidates
    scored = []
    for doc, meta, dist, mem_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0]
    ):
        memory = MemoryObject.from_chromadb(doc, meta, mem_id)
        
        # Distance 0.0 is perfect match. S = 1.0 - distance.
        similarity = max(0.0, 1.0 - dist)
        
        utility = compute_utility(memory, now, similarity)
        scored.append((memory, utility, similarity))

    # 3. Filter by Utility Threshold (Active Forgetting)
    eligible = [(m, u, s) for m, u, s in scored if u >= UTILITY_THRESHOLD]
    
    # 4. Sort and take Top-N
    eligible.sort(key=lambda x: x[1], reverse=True)
    top = eligible[:TOP_N_INJECT]
    
    return [(m, u) for m, u, _ in top]


def update_access_metadata(memories: list[MemoryObject], simulate_hours_passed: float = 0.0):
    """
    Increments access count for retrieved memories to reinforce them.
    """
    now = datetime.now()
    if simulate_hours_passed > 0:
        now += timedelta(hours=simulate_hours_passed)
        
    now_str = now.isoformat()

    for memory in memories:
        try:
            existing = collection.get(ids=[memory.id], include=["metadatas"])
            if not existing["metadatas"]:
                continue
            meta = existing["metadatas"][0]
            meta["access_count"] = int(meta.get("access_count", 0)) + 1
            meta["last_accessed_at"] = now_str
            collection.update(ids=[memory.id], metadatas=[meta])
        except Exception:
            pass


def chat(user_message: str, session_id: str, session_history: list[dict], simulate_hours_passed: float = 0.0) -> tuple[str, list[MemoryObject]]:
    """Full LLM injection pipeline."""
    scored_memories = retrieve_memories(
        user_message,
        session_id=session_id,
        simulate_hours_passed=simulate_hours_passed,
    )
    retrieved = [m for m, _ in scored_memories]

    if retrieved:
        update_access_metadata(retrieved, simulate_hours_passed)

    # Build prompt
    if retrieved:
        memory_lines = [f"[UTILITY {u:.2f}] {m.text}" for m, u in scored_memories]
        memory_block = "\n".join(memory_lines)
        system_content = f"Use these retrieved memories to answer the user:\n---\n{memory_block}\n---"
    else:
        system_content = "You are a helpful AI assistant. No relevant memories were found."

    messages = [{"role": "system", "content": system_content}]
    messages.extend(session_history)
    messages.append({"role": "user", "content": user_message})

    response = _groq_call(
        llm_client.chat.completions.create,
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=512,        # keeps responses snappy and within free-tier TPM
    )

    return response.choices[0].message.content, retrieved