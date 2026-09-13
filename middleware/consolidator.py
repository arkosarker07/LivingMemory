"""
middleware/consolidator.py
==========================
Phase 5 — The Memory Consolidator.

Runs as a triggered background process (every N writes) that:

  1. Fetches all active episodic memories from ChromaDB
  2. Builds a cosine similarity matrix across all of them
  3. Applies agglomerative clustering (no K needed — bottom-up)
  4. For each cluster >= 3 members: calls LLM to merge into one summary
  5. Saves the new semantic memory with inherited metadata (max rule)
  6. Hard-deletes the original fragments

Why hard-delete here (not archive like Phase 3)?
  Phase 3 archives because the AI is making a judgment call about user
  intent — it could be wrong, so we keep an undo button.
  Phase 5 hard-deletes because consolidation is lossless compression —
  no information is lost, the summary IS the originals. Keeping the
  fragments would negate the token-efficiency gains of the thesis.

Why agglomerative clustering (not K-Means)?
  K-Means requires you to specify K (number of clusters) in advance.
  You cannot know how many topics a user has discussed. Agglomerative
  clustering is bottom-up — it merges the most similar pairs first and
  stops when no pair exceeds the similarity threshold. No K needed.

access_count inheritance rule — MAX not SUM:
  Summing access counts compounds across consolidation runs and makes
  memories mathematically immortal (frequency bonus inflates without
  bound). Max preserves the ceiling of proven usefulness without
  inflating the decay formula.
"""

import os
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from middleware.memory_schema import MemoryObject

load_dotenv()

# ── OpenAI client (pointing to Ollama) ─────────────────────────────────────────
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ── ChromaDB — same collection as write and read paths ────────────────────────
embed_fn     = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./data/middleware_db")
collection   = chroma_client.get_or_create_collection(
    name="living_memory",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

# ── Tunable parameters ────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.75   # cosine similarity cutoff for clustering
                                # 0.75 = aggressive, 0.90 = conservative
                                # Aligned with write-path collision override (0.75)
                                # — if similar enough to flag as duplicate, merge it
MIN_CLUSTER_SIZE      = 3      # minimum members to trigger a merge
                                # below this, redundancy is acceptable
MIN_MEMORIES_TO_RUN   = 6      # cold start guard — skip if DB too small
CONSOLIDATION_TRIGGER = 50     # run every N write operations (set in chatbot)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH ALL ACTIVE EPISODIC MEMORIES
# ══════════════════════════════════════════════════════════════════════════════

def fetch_episodic_memories() -> tuple[list[MemoryObject], list[list[float]]]:
    """
    Fetches all active episodic memories and their embeddings from ChromaDB.

    Only episodic memories are consolidation candidates:
      - Semantic memories are already consolidated summaries → skip
      - Archived memories are contradicted facts → skip

    Returns:
        (memories, embeddings) — parallel lists
    """
    if collection.count() == 0:
        return [], []

    results = collection.get(
        where={"$and": [
            {"status":      {"$eq": "active"}},
            {"memory_type": {"$eq": "episodic"}}
        ]},
        include=["documents", "metadatas"]
    )

    if not results["ids"]:
        return [], []

    memories = []
    embeddings = []

    for doc, meta, mem_id in zip(
        results["documents"],
        results["metadatas"],
        results["ids"]
    ):
        mem = MemoryObject.from_chromadb(doc, meta, mem_id)
        memories.append(mem)
        embeddings.append(embed_fn([doc])[0])

    return memories, embeddings


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD SIMILARITY MATRIX AND CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def cluster_memories(
    memories: list[MemoryObject],
    embeddings: list[list[float]]
) -> dict[int, list[MemoryObject]]:
    """
    Groups memories into clusters of semantically similar content using
    agglomerative clustering on their embedding vectors.

    Why agglomerative (not K-Means):
      K-Means requires specifying K upfront. We don't know how many
      topics the user has discussed. Agglomerative clustering is
      bottom-up — it merges the most similar pairs iteratively and
      stops when no remaining pair exceeds the threshold. K emerges
      from the data, not from our assumption.

    Args:
        memories:   list of MemoryObject instances
        embeddings: parallel list of embedding vectors

    Returns:
        dict mapping cluster_id → list of MemoryObject in that cluster
        Only clusters with >= MIN_CLUSTER_SIZE members are returned.
    """
    if len(memories) < 2:
        return {}

    emb_matrix = np.array(embeddings)

    # Agglomerative clustering with cosine distance
    # distance_threshold = 1 - similarity_threshold
    # (sklearn uses distance, not similarity)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=round(1.0 - SIMILARITY_THRESHOLD, 4)
    )

    labels = clustering.fit_predict(emb_matrix)

    # Group memories by cluster label
    clusters: dict[int, list[MemoryObject]] = {}
    for memory, label in zip(memories, labels):
        clusters.setdefault(label, []).append(memory)

    # Filter to only clusters that meet minimum size
    qualifying = {
        label: members
        for label, members in clusters.items()
        if len(members) >= MIN_CLUSTER_SIZE
    }

    print(f"  [Consolidator] {len(clusters)} total clusters found.")
    print(f"  [Consolidator] {len(qualifying)} clusters qualify "
          f"(>= {MIN_CLUSTER_SIZE} members).")

    return qualifying


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MERGE CLUSTER WITH ONE LLM CALL
# ══════════════════════════════════════════════════════════════════════════════

def merge_cluster(members: list[MemoryObject]) -> str | None:
    """
    Sends a cluster of memory fragments to the LLM and asks it to
    synthesise them into one concise semantic memory.

    Uses a lightweight model (qwen/qwen3.8-27b) — this is a summarisation
    task, not a reasoning task. Heavy reasoning models are reserved for
    Phase 3 contradiction detection.

    Args:
        members: list of MemoryObject instances in the cluster

    Returns:
        consolidated text string, or None if the LLM call fails
    """

    # Format fragments for the prompt
    fragments = "\n".join([
        f"[{i+1}] {m.text}"
        for i, m in enumerate(members)
    ])

    prompt = f"""You are a memory consolidation engine for a long-term AI assistant.

You will receive a group of related memory fragments about the same topic.
Your job is to synthesise them into ONE concise, accurate semantic memory.

RULES:
- Preserve every unique fact from all fragments
- Remove all redundancy and repetition
- Write in third person ("User prefers...", not "I prefer...")
- Maximum 2 sentences
- Do not add any information not present in the fragments
- Do not mention that this is a consolidation or summary

MEMORY FRAGMENTS:
{fragments}

Respond with ONLY the consolidated memory text. No preamble, no explanation."""

    try:
        response = llm_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,    # low temperature = factual, consistent output
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()

        # Sanity check — reject empty or suspiciously short outputs
        if len(result) < 20:
            print(f"  [Consolidator] Warning: LLM output too short: '{result}'")
            return None

        return result

    except Exception as e:
        print(f"  [Consolidator] LLM merge call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — INHERIT METADATA (MAX RULE) AND SAVE
# ══════════════════════════════════════════════════════════════════════════════

def inherit_metadata(members: list[MemoryObject]) -> dict:
    """
    Computes the inherited metadata for a consolidated memory.

    The MAX rule for access_count and importance_score:
      Using SUM would compound across consolidation runs, inflating
      the frequency bonus and making memories mathematically immortal.
      MAX preserves the ceiling of proven usefulness — "this concept
      has been as useful as its most-used fragment" — without inflating
      the decay formula beyond meaningful bounds.

    The EARLIEST created_at:
      Preserves the true age of the concept. A topic first mentioned
      in 2024 that gets consolidated in 2026 should still age from 2024
      in the decay formula, not from the consolidation event.

    The MOST RECENT last_accessed_at:
      Uses the freshest retrieval timestamp so decay is calculated from
      the last time any fragment of this concept was actually used.

    Returns:
        dict of metadata values for the new semantic memory
    """

    importance_score  = max(m.importance_score  for m in members)
    access_count      = max(m.access_count      for m in members)  # MAX not SUM
    last_accessed_at  = max(m.last_accessed_at  for m in members)  # most recent
    created_at        = min(m.created_at        for m in members)  # earliest
    session_id        = members[0].session_id                       # from first member
    source_ids        = [m.id for m in members]                    # audit trail

    return {
        "importance_score": importance_score,
        "access_count":     access_count,
        "last_accessed_at": last_accessed_at,
        "created_at":       created_at,
        "session_id":       session_id,
        "source_ids":       source_ids,
    }


def save_consolidated_memory(
    text: str,
    metadata: dict
) -> str:
    """
    Saves the new consolidated semantic memory to ChromaDB.

    Returns the new memory's ID.
    """
    memory = MemoryObject(
        text             = text,
        role             = "assistant",          # consolidation is system-generated
        session_id       = metadata["session_id"],
        importance_score = metadata["importance_score"],
        memory_type      = "semantic",            # type cast — not episodic anymore
        status           = "active",
        is_contradiction = False,
        access_count     = metadata["access_count"],
        last_accessed_at = metadata["last_accessed_at"],
    )

    # Override auto-generated created_at with inherited earliest timestamp
    memory.created_at = metadata["created_at"]

    collection.add(
        ids        = [memory.id],
        documents  = [memory.text],
        metadatas  = [{
            **memory.to_metadata(),
            "is_consolidated": True,
            "source_ids":      json.dumps(metadata["source_ids"]),  # store as JSON string
        }]
    )

    print(f"  [Consolidator] Saved semantic memory: '{text[:60]}...'")
    return memory.id


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — HARD DELETE ORIGINALS
# ══════════════════════════════════════════════════════════════════════════════

def delete_originals(members: list[MemoryObject]) -> bool:
    """
    Permanently deletes the original episodic fragments from ChromaDB.

    Hard-delete (not archive) because:
      - No information is lost — the summary IS the originals compressed
      - Archiving them would negate the token efficiency gains
      - Unlike Phase 3 contradictions, there is no judgment call here
        that could be wrong and need an undo button

    The source_ids field on the consolidated memory serves as the
    audit trail — it records which originals were merged.
    """
    ids_to_delete = [m.id for m in members]

    try:
        collection.delete(ids=ids_to_delete)
        print(f"  [Consolidator] Hard-deleted {len(ids_to_delete)} "
              f"original fragments.")
        return True
    except Exception as e:
        print(f"  [Consolidator] Warning: delete failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONSOLIDATOR — entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_consolidator() -> dict:
    """
    Runs the full consolidation pipeline.

    Called by the main chatbot every CONSOLIDATION_TRIGGER writes.
    Designed to be safe to call at any time — has guards against
    cold start and empty clusters.

    Returns:
        stats dict — useful for thesis evaluation logging
    """

    print(f"\n{'='*50}")
    print(f"[Consolidator] Starting consolidation run...")
    print(f"[Consolidator] Model: {OLLAMA_MODEL}")
    print(f"[Consolidator] Threshold: {SIMILARITY_THRESHOLD} | "
          f"Min cluster: {MIN_CLUSTER_SIZE}")

    stats = {
        "run_at":           datetime.now().isoformat(),
        "clusters_found":   0,
        "clusters_merged":  0,
        "memories_deleted": 0,
        "memories_saved":   0,
        "skipped_reason":   None,
    }

    # ── Step 1: Fetch episodic memories ──────────────────────────────────────
    memories, embeddings = fetch_episodic_memories()
    print(f"[Consolidator] Found {len(memories)} active episodic memories.")

    # ── Cold start guard ──────────────────────────────────────────────────────
    if len(memories) < MIN_MEMORIES_TO_RUN:
        msg = (f"Only {len(memories)} episodic memories — "
               f"need at least {MIN_MEMORIES_TO_RUN}. Skipping.")
        print(f"[Consolidator] {msg}")
        stats["skipped_reason"] = msg
        return stats

    # ── Step 2: Cluster independently per session ────────────────────────────
    # A semantic summary must never combine memories belonging to different
    # users or browser sessions.
    memories_by_session: dict[str, list[MemoryObject]] = {}
    embeddings_by_session: dict[str, list[list[float]]] = {}
    for memory, embedding in zip(memories, embeddings):
        memories_by_session.setdefault(memory.session_id, []).append(memory)
        embeddings_by_session.setdefault(memory.session_id, []).append(embedding)

    qualifying_clusters: list[tuple[str, int, list[MemoryObject]]] = []
    for session_id, session_memories in memories_by_session.items():
        session_clusters = cluster_memories(
            session_memories,
            embeddings_by_session[session_id],
        )
        qualifying_clusters.extend(
            (session_id, cluster_id, members)
            for cluster_id, members in session_clusters.items()
        )

    stats["clusters_found"] = len(qualifying_clusters)

    if not qualifying_clusters:
        print("[Consolidator] No qualifying clusters. Nothing to merge.")
        stats["skipped_reason"] = "No clusters met minimum size threshold."
        return stats

    # ── Steps 3–5: Merge each cluster ─────────────────────────────────────────
    for session_id, cluster_id, members in qualifying_clusters:
        print(f"\n  [Consolidator] Processing cluster {cluster_id} "
              f"({len(members)} members):")
        for m in members:
            print(f"    - '{m.text[:60]}...'")

        # Step 3: LLM merge call
        consolidated_text = merge_cluster(members)

        if consolidated_text is None:
            print(f"  [Consolidator] Skipping cluster {cluster_id} "
                  f"— LLM merge failed.")
            continue

        print(f"  [Consolidator] Merged into: '{consolidated_text[:80]}...'")

        # Step 4: Inherit metadata (MAX rule) and save
        metadata   = inherit_metadata(members)
        new_mem_id = save_consolidated_memory(consolidated_text, metadata)

        # Step 5: Hard-delete originals
        if not delete_originals(members):
            # Avoid leaving both the summary and its fragments active when
            # cleanup fails. The fragments remain available for retry.
            try:
                collection.delete(ids=[new_mem_id])
            except Exception as rollback_error:
                print(f"  [Consolidator] Warning: summary rollback failed: "
                      f"{rollback_error}")
            continue

        stats["clusters_merged"]  += 1
        stats["memories_deleted"] += len(members)
        stats["memories_saved"]   += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[Consolidator] Run complete.")
    print(f"  Clusters merged:   {stats['clusters_merged']}")
    print(f"  Memories deleted:  {stats['memories_deleted']}")
    print(f"  Memories saved:    {stats['memories_saved']}")
    net = stats["memories_deleted"] - stats["memories_saved"]
    print(f"  Net DB reduction:  -{net} memories")
    print(f"{'='*50}")

    return stats