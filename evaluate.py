"""
evaluate.py
===========
LoCoMo Evaluation Harness — Living Memory Thesis

Runs the 4-condition ablation study on 2 LoCoMo conversations and
produces thesis-ready F1, token-efficiency, and memory-count results.

Conditions:
  1. Naive RAG         — baseline/naive_rag.py only
  2. Scoring Only      — write path ON,  consolidator OFF
  3. Consolidation Only— write path OFF, consolidator ON  (raw storage + consolidator)
  4. Full Living Memory— write path ON,  consolidator ON

Usage:
  python evaluate.py            # full run (~4-5 hours on free Groq tier)
  python evaluate.py --dry-run  # 5 QA pairs per condition (~5 minutes)

Output:
  evaluation_results.json   — raw numbers
  evaluation_report.md      — thesis-ready table
"""

import os
import sys
import json
import time
import math
import shutil
import string
import argparse
import re
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# ── OpenAI SDK (pointing to local Ollama) ────────────────────────────────────
from dotenv import load_dotenv
from openai import OpenAI

# ── NLTK for Porter stemming (F1 scorer) ─────────────────────────────────────
try:
    import nltk
    from nltk.stem import PorterStemmer
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    _stemmer = PorterStemmer()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARN] NLTK not installed. F1 scoring will use unstemmed tokens.")
    print("       Install with: pip install nltk")

# ── Force UTF-8 stdout on Windows (needed for checkmarks/emoji in print) ──────
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── ChromaDB (for DB reset between conditions) ────────────────────────────────
import os
os.environ["ALLOW_RESET"] = "TRUE"
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these to change evaluation parameters
# ══════════════════════════════════════════════════════════════════════════════

CONVERSATION_INDICES  = [0]        # which LoCoMo conversations to evaluate
HOURS_PER_SESSION     = 72            # simulated hours between sessions (3 days)
QUERY_DELAY_SECONDS   = 2.5           # pause between Groq API calls (free tier)
MAX_RETRY_WAIT        = 600.0         # allow up to 10 mins wait for extreme rate limits
DRY_RUN_QA_LIMIT      = 5            # QA pairs per condition in --dry-run mode

BASELINE_DB_PATH      = "./data/baseline_db"
MIDDLEWARE_DB_PATH     = "./data/middleware_db"

RESULTS_JSON_PATH     = "./evaluation_results.json"
RESULTS_MD_PATH       = "./evaluation_report.md"

# Skip adversarial questions — LoCoMo encodes types as integers:
#   1 = single_hop, 2 = multi_hop, 3 = temporal, 4 = open_domain, 5 = adversarial
# String fallback also included for datasets that use string labels.
SKIP_TYPES  = {"adversarial", "5", 5}

# Maps LoCoMo integer type codes to readable names for the results table
TYPE_MAP = {
    "1": "single_hop",
    "2": "multi_hop",
    "3": "temporal",
    "4": "open_domain",
    "5": "adversarial",
    # String labels (some versions)
    "single_hop":   "single_hop",
    "multi_hop":    "multi_hop",
    "temporal":     "temporal",
    "open_domain":  "open_domain",
    "adversarial":  "adversarial",
    "unknown":      "unknown",
}

# LLM model — reads from .env (LLM_MODEL), falls back to OLLAMA_MODEL or qwen2.5-coder:7b
LLM_MODEL            = os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"))

# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT — shared across all conditions
# ══════════════════════════════════════════════════════════════════════════════

llm_client    = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def _llm_call(messages: list[dict], max_tokens: int = 200) -> object:
    """
    Calls the local Ollama instance via OpenAI SDK.
    No rate limit backoffs needed for local execution.
    """
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response


# ══════════════════════════════════════════════════════════════════════════════
# F1 SCORER — token-level with Porter stemming (matches LoCoMo paper)
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_tokens(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, tokenise, and apply Porter stemming.
    Matches the token-level F1 calculation used in the LoCoMo benchmark paper.
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    if NLTK_AVAILABLE:
        tokens = [_stemmer.stem(t) for t in tokens]
    # Remove stopwords that add noise (a, an, the, is, are)
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been"}
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Computes token-level F1 between a prediction and ground truth string.
    Returns value between 0.0 (no overlap) and 1.0 (perfect match).
    """
    pred_tokens = _normalise_tokens(prediction)
    gold_tokens = _normalise_tokens(ground_truth)

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = defaultdict(int)
    gold_set = defaultdict(int)
    for t in pred_tokens:
        pred_set[t] += 1
    for t in gold_tokens:
        gold_set[t] += 1

    # Token overlap (intersection of multisets)
    overlap = sum(min(pred_set[t], gold_set[t]) for t in pred_set if t in gold_set)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall    = overlap / len(gold_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE RESET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def reset_baseline_db():
    """Wipes the naive RAG database clean."""
    import chromadb
    c = chromadb.PersistentClient(path=BASELINE_DB_PATH)
    c.reset()
    
    import importlib
    try:
        from baseline import naive_rag
        importlib.reload(naive_rag)
    except Exception:
        pass
    print(f"  [DB] Baseline DB reset.")


def reset_middleware_db():
    """Wipes the Living Memory middleware database clean."""
    import chromadb
    c = chromadb.PersistentClient(path=MIDDLEWARE_DB_PATH)
    c.reset()
    
    import importlib
    try:
        from middleware import write_path
        importlib.reload(write_path)
    except Exception: pass
    
    try:
        from middleware import read_path
        importlib.reload(read_path)
    except Exception: pass
    
    try:
        from middleware import consolidator
        importlib.reload(consolidator)
    except Exception: pass
    
    print(f"  [DB] Middleware DB reset.")


def get_middleware_memory_count() -> int:
    """Returns the number of active memories in the middleware DB."""
    try:
        import importlib
        from middleware import write_path
        importlib.reload(write_path)   # ensure we have the post-ingest client
        all_meta = write_path.collection.get(include=["metadatas"])["metadatas"]
        return sum(1 for m in all_meta if m.get("status") == "active")
    except Exception:
        return 0


def get_baseline_memory_count() -> int:
    """Returns the number of memories in the baseline DB."""
    try:
        import importlib
        from baseline import naive_rag
        importlib.reload(naive_rag)    # ensure we have the post-ingest client
        return naive_rag.collection.count()
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

# LoCoMo is a JSON file at the snap-research GitHub repo.
# HuggingFace listing exists but has no data files — use direct download.
LOCOMO_URL        = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_CACHE_PATH = "./data/locomo10.json"


def load_locomo(conv_indices: list[int]) -> list[dict]:
    """
    Downloads (or loads from cache) the LoCoMo dataset JSON and extracts
    the specified conversation indices.

    The dataset is a JSON file at:
    https://github.com/snap-research/locomo/blob/main/data/locomo10.json

    Returns a list of conversation dicts, each with:
      - "sessions": list of session dicts, each with {session_id, turns: [{speaker, text}]}
      - "qa_pairs": list of QA pair dicts, each with {question, answer, type}
    """
    # ── Download or load from cache ───────────────────────────────────────────
    cache_path = Path(LOCOMO_CACHE_PATH)
    if cache_path.exists():
        print(f"\n[Dataset] Loading LoCoMo from cache: {LOCOMO_CACHE_PATH}")
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        print(f"\n[Dataset] Downloading LoCoMo from GitHub...")
        print(f"  URL: {LOCOMO_URL}")
        print(f"  This may take a moment (the file is ~10MB)...")
        try:
            os.makedirs(os.path.dirname(LOCOMO_CACHE_PATH), exist_ok=True)
            urllib.request.urlretrieve(LOCOMO_URL, LOCOMO_CACHE_PATH)
            with open(cache_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            print(f"  Downloaded and cached to {LOCOMO_CACHE_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to download LoCoMo dataset: {e}")
            print(f"  Please download manually from:")
            print(f"  https://github.com/snap-research/locomo/blob/main/data/locomo10.json")
            print(f"  and save it to: {LOCOMO_CACHE_PATH}")
            sys.exit(1)

    # raw_data is a list of 10 conversation objects
    if not isinstance(raw_data, list):
        # Some versions wrap in a dict
        raw_data = raw_data.get("data", list(raw_data.values()))

    print(f"[Dataset] Found {len(raw_data)} conversations in dataset.")

    conversations = []
    for idx in conv_indices:
        if idx >= len(raw_data):
            print(f"[Dataset] WARNING: conversation index {idx} out of range ({len(raw_data)} total). Skipping.")
            continue
        raw = raw_data[idx]
        conv = _parse_locomo_conversation(raw, idx)
        conversations.append(conv)
        print(f"[Dataset] Conv {idx}: {len(conv['sessions'])} sessions, "
              f"{len(conv['qa_pairs'])} QA pairs (adversarial excluded)")

    return conversations


def _parse_locomo_conversation(raw: dict, idx: int) -> dict:
    """
    Parses one raw LoCoMo JSON object into a clean conversation dict.

    Real LoCoMo schema (from snap-research/locomo):
      {
        "sample_id": "conv_0",
        "conversation": {
          "session_1": [
            {"speaker": "A", "dia_id": 0, "text": "..."},
            {"speaker": "B", "dia_id": 1, "text": "..."},
            ...
          ],
          "session_1_date_time": "2021-09-01 10:00:00",
          "speaker_a": "Alice",
          "speaker_b": "Bob",
          "session_2": [...],
          ...
        },
        "qa": [
          {"question": "...", "answer": "...", "type": "single_hop"},
          ...
        ]
      }
    """
    # ── Sessions ──────────────────────────────────────────────────────────────
    sessions = []
    raw_conv = raw.get("conversation", {})

    if isinstance(raw_conv, dict):
        # Collect all session_N keys (exclude metadata keys)
        session_keys = sorted(
            [k for k in raw_conv if re.match(r"^session_\d+$", k)],
            key=lambda k: int(k.split("_")[1])
        )

        for s_num, skey in enumerate(session_keys):
            turns_raw = raw_conv[skey]
            turns = []

            if isinstance(turns_raw, list):
                for turn in turns_raw:
                    if isinstance(turn, dict):
                        speaker = turn.get("speaker", "A")
                        text    = turn.get("text", "").strip()
                        if text:
                            turns.append({"speaker": str(speaker), "text": text})

            if turns:
                sessions.append({
                    "session_id":  s_num + 1,
                    "turns":       turns,
                })

    elif isinstance(raw_conv, list):
        # Fallback: some versions might store sessions as a list
        for s_idx, session_data in enumerate(raw_conv):
            turns = []
            if isinstance(session_data, list):
                for turn in session_data:
                    if isinstance(turn, dict):
                        text = turn.get("text", "").strip()
                        if text:
                            turns.append({"speaker": turn.get("speaker", "A"), "text": text})
            if turns:
                sessions.append({"session_id": s_idx + 1, "turns": turns})

    # ── QA Pairs ──────────────────────────────────────────────────────────────
    qa_pairs = []
    raw_qa   = raw.get("qa", [])

    if isinstance(raw_qa, list):
        for qa in raw_qa:
            if not isinstance(qa, dict):
                continue

            question = qa.get("question", "").strip()

            # Ground truth: may be "answer", "answers", or a list
            answer_raw = qa.get("answer", qa.get("answers", ""))
            if isinstance(answer_raw, list):
                # Take first non-empty answer, or join them
                answer = next((str(a).strip() for a in answer_raw if str(a).strip()), "")
            else:
                answer = str(answer_raw).strip()

            # Question type (may be int in some LoCoMo versions — cast to str)
            q_type = str(
                qa.get("type") or
                qa.get("category") or
                qa.get("question_type") or
                "unknown"
            ).lower().strip()

            # Skip adversarial and blank entries
            if not question or not answer:
                continue
            if q_type in SKIP_TYPES:
                continue

            # Normalise integer type codes → readable names for the results table
            q_type = TYPE_MAP.get(q_type, q_type)

            qa_pairs.append({
                "question": question,
                "answer":   answer,
                "type":     q_type,
            })

    print(f"  [Parser] Conv {idx}: {len(sessions)} sessions, "
          f"{len(qa_pairs)} QA pairs (adversarial excluded)")

    return {
        "conv_idx":  idx,
        "sessions":  sessions,
        "qa_pairs":  qa_pairs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# QA QUERY RUNNER — shared across conditions
# ══════════════════════════════════════════════════════════════════════════════

def ask_question_naive_rag(question: str, session_id: str) -> tuple[str, int]:
    """
    Asks a question using the naive RAG read path.
    Returns (answer_text, total_tokens_used).
    Note: uses the naive_rag module's retrieve + LLM pipeline.
    """
    # Import fresh each time so module-level state is current
    from baseline import naive_rag

    # Retrieve relevant memories
    memories = naive_rag.retrieve_memories(question, top_k=5)

    # Build prompt
    messages = naive_rag.build_prompt(question, memories, session_id)

    # Call LLM directly so we capture token usage
    response = _llm_call(messages, max_tokens=200)
    answer   = response.choices[0].message.content.strip()
    tokens   = response.usage.total_tokens

    return answer, tokens


def ask_question_living_memory(
    question: str,
    session_id: str,
    simulate_hours: float = 0.0
) -> tuple[str, int]:
    """
    Asks a question using the Living Memory read path.
    Returns (answer_text, total_tokens_used).
    """
    from middleware import read_path

    # Retrieve memories with time simulation
    scored_memories = read_path.retrieve_memories(
        question,
        session_id=session_id,
        simulate_hours_passed=simulate_hours,
    )
    retrieved       = [m for m, _ in scored_memories]

    if retrieved:
        read_path.update_access_metadata(retrieved, simulate_hours_passed=simulate_hours)

    # Build prompt (mirrors read_path.chat() but captures token usage)
    if retrieved:
        memory_lines  = [f"[UTILITY {u:.2f}] {m.text}" for m, u in scored_memories]
        memory_block  = "\n".join(memory_lines)
        system_content = (
            f"Use these retrieved memories to answer the user's question accurately "
            f"and concisely:\n---\n{memory_block}\n---"
        )
    else:
        system_content = (
            "You are a helpful AI assistant. "
            "No relevant memories were found. Answer based on your general knowledge if possible."
        )

    messages = [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": question},
    ]

    response = _llm_call(messages, max_tokens=200)
    answer   = response.choices[0].message.content.strip()
    tokens   = response.usage.total_tokens

    return answer, tokens


# ══════════════════════════════════════════════════════════════════════════════
# INGEST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ingest_naive_rag(sessions: list[dict], conv_idx: int):
    """
    Ingests all sessions into the naive RAG baseline database.
    Saves every turn as a raw memory (no scoring filter).
    """
    # We need a fresh module import after DB reset
    import importlib
    from baseline import naive_rag
    importlib.reload(naive_rag)

    total_turns = 0
    session_id  = f"locomo_conv{conv_idx}"

    for s_idx, session in enumerate(sessions):
        print(f"    Session {s_idx+1}/{len(sessions)}: "
              f"{len(session['turns'])} turns", end="", flush=True)

        for turn in session["turns"]:
            text = turn["text"].strip()
            if not text:
                continue
            naive_rag.save_memory(role=turn["speaker"], content=text, session_id=session_id)
            total_turns += 1

        print(f" ✓")

    print(f"  [Ingest] Saved {total_turns} raw turns to baseline DB.")
    return total_turns


def _process_memory_with_retry(
    write_path_module,
    text: str,
    speaker: str,
    session_id: str,
    created_at: str | None = None,
    last_accessed_at: str | None = None,
) -> str:
    """
    Calls process_memory directly since Ollama has no rate limits.
    """
    try:
        return write_path_module.process_memory(
            text,
            speaker,
            session_id,
            created_at=created_at,
            last_accessed_at=last_accessed_at,
        )
    except Exception as e:
        # Log and skip this turn
        print(f"\n  [Ingest] Warning: process_memory failed: {e}")
        return "discarded"


def ingest_write_path(sessions: list[dict], conv_idx: int, run_consolidator: bool = False):
    """
    Ingests all sessions through the Living Memory write path (importance scoring).
    Optionally fires the consolidator after each session.
    Returns (total_saved, total_discarded).
    """
    import importlib
    from middleware import write_path
    importlib.reload(write_path)

    saved       = 0
    discarded   = 0
    write_count = 0
    session_id  = f"locomo_conv{conv_idx}"

    for s_idx, session in enumerate(sessions):
        sim_hours = s_idx * HOURS_PER_SESSION
        session_time = (datetime.now() + timedelta(hours=sim_hours)).isoformat()
        print(f"    Session {s_idx+1}/{len(sessions)} "
              f"[t+{sim_hours}h]: {len(session['turns'])} turns", end="", flush=True)

        for turn in session["turns"]:
            text = turn["text"].strip()
            if not text:
                continue

            result = _process_memory_with_retry(
                write_path,
                text,
                turn["speaker"],
                session_id,
                created_at=session_time,
                last_accessed_at=session_time,
            )
            if result in ("saved", "updated"):
                saved += 1
                write_count += 1
            else:
                discarded += 1

        print(f" ok (+{saved} saved so far)")

        # Fire consolidator after each session if enabled
        if run_consolidator and write_count >= 6:  # MIN_MEMORIES_TO_RUN
            from middleware.consolidator import run_consolidator as consolidate
            print(f"    [Consolidator] Running after session {s_idx+1}...")
            consolidate()

    print(f"  [Ingest] Write path: {saved} saved, {discarded} discarded.")
    return saved, discarded


def ingest_raw_plus_consolidator(sessions: list[dict], conv_idx: int):
    """
    Condition 3 — Consolidation Only.
    Stores every turn without importance scoring, then runs the consolidator.

    Strategy: use the middleware DB but bypass the write path's LLM filter.
    We directly embed and save every turn as a MemoryObject with importance_score=0.5
    (neutral), then let the consolidator compress them.
    """
    import importlib
    from middleware import write_path
    importlib.reload(write_path)

    from middleware.memory_schema import MemoryObject
    import chromadb
    from chromadb.utils import embedding_functions

    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    client   = chromadb.PersistentClient(path=MIDDLEWARE_DB_PATH)
    col      = client.get_or_create_collection(
        name="living_memory",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    total_saved = 0
    session_id  = f"locomo_conv{conv_idx}"

    for s_idx, session in enumerate(sessions):
        sim_hours = s_idx * HOURS_PER_SESSION
        session_time = (datetime.now() + timedelta(hours=sim_hours)).isoformat()
        print(f"    Session {s_idx+1}/{len(sessions)} "
              f"[t+{sim_hours}h]: {len(session['turns'])} turns", end="", flush=True)

        for turn in session["turns"]:
            text = turn["text"].strip()
            if not text or len(text) < 10:  # skip empty/trivial turns
                continue

            import uuid
            memory = MemoryObject(
                text             = text,
                role             = turn["speaker"],
                session_id       = session_id,
                importance_score = 0.5,   # neutral score — no LLM filter
                memory_type      = "episodic",
                status           = "active",
                created_at       = session_time,
                last_accessed_at = session_time,
            )
            embedding = write_path.embed(text)
            col.add(
                ids        = [memory.id],
                embeddings = [embedding],
                documents  = [memory.text],
                metadatas  = [memory.to_metadata()],
            )
            total_saved += 1

        print(f" ✓")

    print(f"  [Ingest] Raw storage: {total_saved} turns saved.")

    # Now run the consolidator to compress everything
    from middleware.consolidator import run_consolidator as consolidate
    print(f"  [Consolidator] Running post-ingest consolidation...")
    stats = consolidate()
    print(f"  [Consolidator] Done: {stats.get('clusters_merged', 0)} clusters merged, "
          f"{stats.get('memories_deleted', 0)} deleted → "
          f"{stats.get('memories_saved', 0)} semantic memories created.")

    return total_saved


# ══════════════════════════════════════════════════════════════════════════════
# QUERY PHASE — runs QA pairs against a condition
# ══════════════════════════════════════════════════════════════════════════════

def run_query_phase(
    qa_pairs: list[dict],
    conv_idx: int,
    mode: str,                    # "naive_rag" or "living_memory"
    simulate_hours: float = 0.0,  # total simulated hours (last session's offset)
    dry_run: bool = False,
) -> dict:
    """
    Asks every QA pair against the specified system, records answers and tokens.

    Returns a results dict with:
      - f1_scores:       {type: [f1, f1, ...]}
      - total_tokens:    int
      - total_queries:   int
      - answers:         [(question, predicted, ground_truth, f1, type), ...]
    """
    session_id = f"locomo_conv{conv_idx}"

    if dry_run:
        qa_pairs = qa_pairs[:DRY_RUN_QA_LIMIT]
        print(f"  [Dry Run] Limiting to {DRY_RUN_QA_LIMIT} QA pairs.")

    f1_scores      = defaultdict(list)
    total_tokens   = 0
    total_queries  = 0
    answers_log    = []

    print(f"  [Query] Running {len(qa_pairs)} QA pairs (mode: {mode})...")

    for i, qa in enumerate(qa_pairs):
        question     = qa["question"]
        ground_truth = qa["answer"]
        q_type       = qa["type"]

        # Print progress every 10 questions
        if i % 10 == 0:
            print(f"    [{i+1}/{len(qa_pairs)}] type={q_type} | "
                  f"Q: {question[:60]}...")

        try:
            if mode == "naive_rag":
                answer, tokens = ask_question_naive_rag(question, session_id)
            else:
                answer, tokens = ask_question_living_memory(
                    question, session_id, simulate_hours=simulate_hours
                )
        except Exception as e:
            print(f"    [ERROR] Query failed: {e}")
            answer = ""
            tokens = 0

        f1    = compute_f1(answer, ground_truth)
        total_tokens  += tokens
        total_queries += 1
        f1_scores[q_type].append(f1)
        answers_log.append((question, answer, ground_truth, f1, q_type))

    # Summarise F1 per type
    f1_summary = {}
    for q_type, scores in f1_scores.items():
        f1_summary[q_type] = round(sum(scores) / len(scores), 4) if scores else 0.0

    overall_f1 = (
        round(sum(sum(v) for v in f1_scores.values()) /
              sum(len(v) for v in f1_scores.values()), 4)
        if total_queries > 0 else 0.0
    )

    tokens_per_query = round(total_tokens / total_queries, 1) if total_queries > 0 else 0.0

    return {
        "f1_overall":      overall_f1,
        "f1_by_type":      f1_summary,
        "total_tokens":    total_tokens,
        "tokens_per_query": tokens_per_query,
        "total_queries":   total_queries,
        "answers_log":     answers_log,   # kept for manual inspection
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONDITION RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_condition_1_naive_rag(conversation: dict, dry_run: bool) -> dict:
    """Condition 1 — Naive RAG baseline."""
    print("\n" + "="*60)
    print("  CONDITION 1 — Naive RAG")
    print("="*60)

    reset_baseline_db()

    # Ingest all sessions
    print("[Ingest Phase]")
    ingest_naive_rag(conversation["sessions"], conversation["conv_idx"])
    memory_count = get_baseline_memory_count()
    print(f"  Memory count after ingest: {memory_count}")

    # Query phase
    print("[Query Phase]")
    n_sessions    = len(conversation["sessions"])
    sim_hours     = (n_sessions - 1) * HOURS_PER_SESSION

    results = run_query_phase(
        qa_pairs      = conversation["qa_pairs"],
        conv_idx      = conversation["conv_idx"],
        mode          = "naive_rag",
        simulate_hours= sim_hours,
        dry_run       = dry_run,
    )
    results["memory_count_after_ingest"] = memory_count
    results["condition"] = "naive_rag"
    return results


def run_condition_2_scoring_only(conversation: dict, dry_run: bool) -> dict:
    """Condition 2 — Write Path scoring ON, Consolidator OFF."""
    print("\n" + "="*60)
    print("  CONDITION 2 — Scoring Only (Write Path ON, Consolidator OFF)")
    print("="*60)

    reset_middleware_db()

    print("[Ingest Phase]")
    ingest_write_path(
        sessions         = conversation["sessions"],
        conv_idx         = conversation["conv_idx"],
        run_consolidator = False,    # consolidator explicitly off
    )
    memory_count = get_middleware_memory_count()
    print(f"  Active memory count after ingest: {memory_count}")

    print("[Query Phase]")
    n_sessions = len(conversation["sessions"])
    sim_hours  = (n_sessions - 1) * HOURS_PER_SESSION

    results = run_query_phase(
        qa_pairs       = conversation["qa_pairs"],
        conv_idx       = conversation["conv_idx"],
        mode           = "living_memory",
        simulate_hours = sim_hours,
        dry_run        = dry_run,
    )
    results["memory_count_after_ingest"] = memory_count
    results["condition"] = "scoring_only"
    return results


def run_condition_3_consolidation_only(conversation: dict, dry_run: bool) -> dict:
    """Condition 3 — Consolidator ON, Write Path scoring OFF (raw storage)."""
    print("\n" + "="*60)
    print("  CONDITION 3 — Consolidation Only (Write Path OFF, Consolidator ON)")
    print("="*60)

    reset_middleware_db()

    print("[Ingest Phase]")
    ingest_raw_plus_consolidator(
        sessions = conversation["sessions"],
        conv_idx = conversation["conv_idx"],
    )
    memory_count = get_middleware_memory_count()
    print(f"  Active memory count after ingest + consolidation: {memory_count}")

    print("[Query Phase]")
    n_sessions = len(conversation["sessions"])
    sim_hours  = (n_sessions - 1) * HOURS_PER_SESSION

    results = run_query_phase(
        qa_pairs       = conversation["qa_pairs"],
        conv_idx       = conversation["conv_idx"],
        mode           = "living_memory",
        simulate_hours = sim_hours,
        dry_run        = dry_run,
    )
    results["memory_count_after_ingest"] = memory_count
    results["condition"] = "consolidation_only"
    return results


def run_condition_4_full_living_memory(conversation: dict, dry_run: bool) -> dict:
    """Condition 4 — Full Living Memory (Write Path ON + Consolidator ON)."""
    print("\n" + "="*60)
    print("  CONDITION 4 — Full Living Memory (Everything ON)")
    print("="*60)

    reset_middleware_db()

    print("[Ingest Phase]")
    ingest_write_path(
        sessions         = conversation["sessions"],
        conv_idx         = conversation["conv_idx"],
        run_consolidator = True,     # consolidator fires after each session
    )
    memory_count = get_middleware_memory_count()
    print(f"  Active memory count after ingest: {memory_count}")

    print("[Query Phase]")
    n_sessions = len(conversation["sessions"])
    sim_hours  = (n_sessions - 1) * HOURS_PER_SESSION

    results = run_query_phase(
        qa_pairs       = conversation["qa_pairs"],
        conv_idx       = conversation["conv_idx"],
        mode           = "living_memory",
        simulate_hours = sim_hours,
        dry_run        = dry_run,
    )
    results["memory_count_after_ingest"] = memory_count
    results["condition"] = "full_living_memory"
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_results(all_results: list[dict]) -> dict:
    """
    Averages results across conversations for each condition.
    all_results is a flat list of condition result dicts.
    """
    by_condition = defaultdict(list)
    for r in all_results:
        by_condition[r["condition"]].append(r)

    aggregated = {}
    for condition, results in by_condition.items():
        # Average F1 overall
        f1_overall = round(
            sum(r["f1_overall"] for r in results) / len(results), 4
        )

        # Average F1 by type
        all_types = set()
        for r in results:
            all_types.update(r["f1_by_type"].keys())

        f1_by_type = {}
        for t in sorted(all_types):
            scores = [r["f1_by_type"].get(t, 0.0) for r in results if t in r["f1_by_type"]]
            f1_by_type[t] = round(sum(scores) / len(scores), 4) if scores else 0.0

        # Average token stats
        tokens_per_query = round(
            sum(r["tokens_per_query"] for r in results) / len(results), 1
        )

        # Average memory count
        memory_count = round(
            sum(r["memory_count_after_ingest"] for r in results) / len(results), 0
        )

        # Total queries run
        total_queries = sum(r["total_queries"] for r in results)

        aggregated[condition] = {
            "f1_overall":               f1_overall,
            "f1_by_type":               f1_by_type,
            "tokens_per_query":         tokens_per_query,
            "memory_count_after_ingest": int(memory_count),
            "total_queries":            total_queries,
        }

    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
# REPORT WRITER
# ══════════════════════════════════════════════════════════════════════════════

CONDITION_LABELS = {
    "naive_rag":           "Naive RAG (Baseline)",
    "scoring_only":        "Scoring Only",
    "consolidation_only":  "Consolidation Only",
    "full_living_memory":  "Full Living Memory",
}

CONDITION_ORDER = [
    "naive_rag",
    "scoring_only",
    "consolidation_only",
    "full_living_memory",
]


def write_results(all_results: list[dict], metadata: dict):
    """Writes evaluation_results.json and evaluation_report.md."""

    aggregated = aggregate_results(all_results)

    output = {
        "metadata":   metadata,
        "conditions": aggregated,
        "per_conversation": [
            {k: v for k, v in r.items() if k != "answers_log"}
            for r in all_results
        ],
    }

    import os
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            with open(RESULTS_JSON_PATH, "r") as f:
                old_data = json.load(f)
            # Merge conditions
            old_data.setdefault("conditions", {}).update(output["conditions"])
            
            # Replace rerun entries so stale metrics cannot survive.
            merged_rows = {
                (r["condition"], r["conv_idx"]): r
                for r in old_data.setdefault("per_conversation", [])
            }
            for r in output["per_conversation"]:
                merged_rows[(r["condition"], r["conv_idx"])] = r
            old_data["per_conversation"] = list(merged_rows.values())
            
            # Update metadata conditions run
            old_conds = set(old_data["metadata"].get("conditions_run", []))
            old_conds.update(output["metadata"]["conditions_run"])
            old_data["metadata"]["conditions_run"] = sorted(list(old_conds))
            
            output = old_data
            aggregated = output["conditions"]  # update for markdown generator
        except Exception as e:
            print(f"[Warning] Could not merge existing JSON: {e}")

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] Results saved to {RESULTS_JSON_PATH}")

    # ── Markdown report ───────────────────────────────────────────────────────
    lines = [
        "# Living Memory — LoCoMo Evaluation Results",
        "",
        f"**Date:** {metadata['date']}",
        f"**Model:** `{metadata['model']}`",
        f"**Conversations evaluated:** {metadata['conversations']}",
        f"**Total QA pairs:** {sum(r['total_queries'] for r in all_results)}",
        f"**Dry run:** {'Yes' if metadata.get('dry_run') else 'No'}",
        "",
        "---",
        "",
        "## Main Results Table",
        "",
        "| Condition | F1 Overall | Tokens/Query | DB Size (active) |",
        "|---|---|---|---|",
    ]

    for cond in CONDITION_ORDER:
        if cond not in aggregated:
            continue
        r     = aggregated[cond]
        label = CONDITION_LABELS.get(cond, cond)
        lines.append(
            f"| {label} "
            f"| **{r['f1_overall']:.4f}** "
            f"| {r['tokens_per_query']:.0f} "
            f"| {r['memory_count_after_ingest']} |"
        )

    # F1 by question type
    lines += [
        "",
        "---",
        "",
        "## F1 Score by Question Type",
        "",
    ]

    # Collect all types
    all_types = set()
    for r in aggregated.values():
        all_types.update(r["f1_by_type"].keys())
    all_types = sorted(all_types)

    header = "| Condition | " + " | ".join(t.replace("_", " ").title() for t in all_types) + " |"
    sep    = "|---|" + "---|" * len(all_types)
    lines += [header, sep]

    for cond in CONDITION_ORDER:
        if cond not in aggregated:
            continue
        r     = aggregated[cond]
        label = CONDITION_LABELS.get(cond, cond)
        cells = [f"{r['f1_by_type'].get(t, 0.0):.4f}" for t in all_types]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # Token efficiency comparison
    if "naive_rag" in aggregated and "full_living_memory" in aggregated:
        naive_tok = aggregated["naive_rag"]["tokens_per_query"]
        full_tok  = aggregated["full_living_memory"]["tokens_per_query"]
        if naive_tok > 0:
            token_reduction = round((1 - full_tok / naive_tok) * 100, 1)
            lines += [
                "",
                "---",
                "",
                "## Token Efficiency",
                "",
                f"Full Living Memory uses **{token_reduction:+.1f}%** {'fewer' if token_reduction > 0 else 'more'} "
                f"tokens per query compared to Naive RAG.",
                f"- Naive RAG: {naive_tok:.0f} tokens/query",
                f"- Full Living Memory: {full_tok:.0f} tokens/query",
            ]

    # Memory compression
    if "naive_rag" in aggregated and "full_living_memory" in aggregated:
        naive_mem = aggregated["naive_rag"]["memory_count_after_ingest"]
        full_mem  = aggregated["full_living_memory"]["memory_count_after_ingest"]
        if naive_mem > 0:
            mem_reduction = round((1 - full_mem / naive_mem) * 100, 1)
            lines += [
                "",
                "---",
                "",
                "## Memory Compression",
                "",
                f"Full Living Memory stores **{mem_reduction:+.1f}%** {'fewer' if mem_reduction > 0 else 'more'} "
                f"active memories compared to Naive RAG.",
                f"- Naive RAG: {naive_mem} memories",
                f"- Full Living Memory: {full_mem} memories",
            ]

    lines += [
        "",
        "---",
        "",
        "## Methodology Notes",
        "",
        f"- Evaluation conducted on {len(CONVERSATION_INDICES)} configured conversation(s) "
        "from the LoCoMo benchmark. "
        "from the LoCoMo benchmark, consistent with computational constraints.",
        "- Adversarial questions excluded (standard practice; task tests unanswerable-query detection, "
        "not memory reconstruction).",
        f"- Simulated time gap between sessions: {HOURS_PER_SESSION} hours ({HOURS_PER_SESSION//24} days).",
        "- F1 score: token-level with Porter stemming, matches LoCoMo paper metric.",
        f"- All conditions use model: `{LLM_MODEL}`.",
    ]

    with open(RESULTS_MD_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"[Output] Report saved to {RESULTS_MD_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo evaluation harness for Living Memory thesis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Run only {DRY_RUN_QA_LIMIT} QA pairs per condition (fast test)"
    )
    parser.add_argument(
        "--condition",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only a specific condition (1=Naive RAG, 2=Scoring, 3=Consolidation, 4=Full)"
    )
    parser.add_argument(
        "--conv",
        type=int,
        nargs="+",
        default=CONVERSATION_INDICES,
        help="Which conversation indices to evaluate (default: 0 1)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Living Memory — LoCoMo Evaluation Harness")
    print("="*60)
    print(f"  Model:         {LLM_MODEL}")
    print(f"  Conversations: {args.conv}")
    print(f"  Dry run:       {args.dry_run}")
    print(f"  Condition:     {args.condition or 'All (1-4)'}")
    print("="*60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    conversations = load_locomo(args.conv)
    if not conversations:
        print("[ERROR] No conversations loaded. Check your dataset indices.")
        sys.exit(1)

    # ── Run conditions ────────────────────────────────────────────────────────
    all_results = []
    condition_runners = {
        1: run_condition_1_naive_rag,
        2: run_condition_2_scoring_only,
        3: run_condition_3_consolidation_only,
        4: run_condition_4_full_living_memory,
    }

    conditions_to_run = [args.condition] if args.condition else [1, 2, 3, 4]

    for conv in conversations:
        print(f"\n{'='*60}")
        print(f"  Processing Conversation {conv['conv_idx']} "
              f"({len(conv['sessions'])} sessions, {len(conv['qa_pairs'])} QA pairs)")
        print(f"{'='*60}")

        for cond_num in conditions_to_run:
            runner = condition_runners[cond_num]
            result = runner(conv, dry_run=args.dry_run)
            result["conv_idx"] = conv["conv_idx"]
            all_results.append(result)

            # Print quick summary after each condition
            print(f"\n  ✓ Condition {cond_num} complete:")
            print(f"    F1 Overall:     {result['f1_overall']:.4f}")
            print(f"    Tokens/Query:   {result['tokens_per_query']:.0f}")
            print(f"    Memory Count:   {result['memory_count_after_ingest']}")
            print(f"    F1 by type:     {result['f1_by_type']}")

    # ── Write results ─────────────────────────────────────────────────────────
    metadata = {
        "date":          datetime.now().isoformat(),
        "model":         LLM_MODEL,
        "conversations": args.conv,
        "dry_run":       args.dry_run,
        "conditions_run": conditions_to_run,
        "hours_per_session": HOURS_PER_SESSION,
    }
    write_results(all_results, metadata)

    print("\n" + "="*60)
    print("  Evaluation complete!")
    print(f"  Results: {RESULTS_JSON_PATH}")
    print(f"  Report:  {RESULTS_MD_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
