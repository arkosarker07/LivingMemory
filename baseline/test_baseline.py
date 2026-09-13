"""
baseline/test_baseline.py
=========================
Simulates a multi-session conversation to test the naive RAG baseline.

Run this BEFORE you build Living Memory to establish what "bad" looks like.
Then run the same test on your middleware and compare the results.

Key Test Objectives:
  1. In-session context flow
  2. Cross-session long-term recall
  3. Contradiction handling (stale memory over-retrieval)
"""

import sys
import os
import time

# Set standard output encoding to UTF-8 and replace errors to handle emojis/unicode characters on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure the root project directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baseline.naive_rag
from baseline.naive_rag import chat, get_stats, chroma_client


def reset_database():
    """
    Clears the ChromaDB collection before running the test suite.
    Ensures a clean slate so test metrics remain deterministic and reproducible.
    """
    print("[TEST SETUP] Clearing previous test database...")
    try:
        chroma_client.delete_collection("naive_rag_memory")
    except Exception:
        pass  # Collection didn't exist yet

    # Re-create fresh collection
    from chromadb.utils import embedding_functions
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    
    baseline.naive_rag.collection = chroma_client.get_or_create_collection(
        name="naive_rag_memory",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )


def run_session(session_id: str, turns: list[tuple[str, str]]):
    """
    Runs one conversation session.

    Args:
        session_id: unique ID for this session
        turns: list of (user_message, expected_topic) pairs
    """
    print(f"\n{'='*60}")
    print(f"  SESSION: {session_id}")
    print(f"{'='*60}")

    for user_msg, topic in turns:
        print(f"\n[Topic: {topic}]")
        print(f"User: {user_msg}")
        reply = chat(user_msg, session_id)
        print(f"Bot:  {reply}")
        # Space out API calls so the test suite doesn't burn through free-tier RPM.
        time.sleep(2)


def main():
    # ── 0. Ensure fresh state for evaluation ─────────────────────────────────
    reset_database()

    # ── Session 1: Introduce personal facts ──────────────────────────────────
    run_session("session_001", [
        ("Hi! My name is Alex and I am studying computer science.", "introduction"),
        ("I prefer Python over Java for my projects.", "preference"),
        ("I am working on a thesis about AI memory systems.", "thesis topic"),
        ("My supervisor is Professor Rahman at Dhaka University.", "supervisor"),
    ])

    # ── Session 2 (Cross-Session Recall Test) ───────────────────────────────
    # Since session_002 has a different ID, short-term history is empty.
    # The agent MUST rely on ChromaDB long-term retrieval.
    run_session("session_002", [
        ("Hey, what do you remember about me?", "recall test"),
        ("What programming language do I prefer?", "preference recall"),
        ("Can you help me with my thesis research?", "context recall"),
    ])

    # ── Session 3: Introduce a contradiction ─────────────────────────────────
    # We update a key preference. Naive RAG should struggle here because
    # cosine similarity will retrieve BOTH Python and JavaScript memories equally.
    run_session("session_003", [
        ("I changed my mind — I actually prefer JavaScript now, not Python.", "contradiction update"),
        ("What's my favourite programming language?", "contradiction resolution test"),
    ])

    # ── Final Evaluation Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EVALUATION REPORT (BASELINE)")
    print(f"{'='*60}")
    stats = get_stats()
    print(f"System:                  {stats['system']}")
    print(f"Total memories stored:   {stats['total_memories']}")
    print("\nManual Verification Checklist for your Thesis log:")
    print("  [ ] 1. Cross-Session Recall (Session 2): Did it fetch Alex's name & thesis?")
    print("  [ ] 2. Contradiction Resolution (Session 3): Did it say JavaScript, or get confused by Python?")
    print("  [ ] 3. Token Bloat: Note how memory count grew strictly monotonically.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()