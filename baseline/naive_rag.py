"""
baseline/naive_rag_groq.py
==========================
This is your BASELINE system — updated to use Groq for lightning-fast 
LLM generation and a local open-source model for vector embeddings.

What this does:
  - Stores every conversation turn in ChromaDB using local embeddings
  - Maintains a short-term sliding window (so it can handle follow-up pronouns)
  - Fetches the top-5 most similar past messages
  - The Groq API (using Llama 3) answers using that context
"""

import os
import uuid
import time
import re
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ── 1. Set up API Keys and OpenAI ───────────────────────────────────────────────
# Load environment variables from the .env file
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ── 2. Set up ChromaDB & Local Embeddings ─────────────────────────────────────
# Because Groq doesn't do embeddings, we use Chroma's built-in local model.
# This runs "sentence-transformers/all-MiniLM-L6-v2" on your machine.
embed_fn = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="./data/baseline_db")

# By attaching embed_fn here, Chroma automatically turns our text into 
# vectors in the background. We don't have to call an embed() function manually!
collection = chroma_client.get_or_create_collection(
    name="naive_rag_memory",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

# ── Short-Term Memory ─────────────────────────────────────────────────────────
# Store the last few turns in memory for immediate context (sliding window)
session_histories = {}


# ── Save a message to the database ────────────────────────────────────────────
def save_memory(role: str, content: str, session_id: str):
    """
    Saves one conversation turn to ChromaDB.
    Chroma automatically converts the `content` string into a vector.
    """
    memory_id = str(uuid.uuid4())

    collection.add(
        ids=[memory_id],
        documents=[content],
        metadatas=[{
            "role": role,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }]
    )
    return memory_id


# ── Retrieve relevant past memories ───────────────────────────────────────────
def retrieve_memories(query: str, top_k: int = 5) -> list[dict]:
    """
    Fetches the top-K most semantically similar memories to the query.
    """
    if collection.count() == 0:
        return []

    # We pass the raw text query; Chroma embeds it locally and searches
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    memories = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        memories.append({
            "content": doc,
            "role": meta["role"],
            "timestamp": meta["timestamp"],
            "similarity": round(1 - dist, 3)
        })

    return memories


# ── Build the prompt with retrieved context ────────────────────────────────────
def build_prompt(user_message: str, memories: list[dict], session_id: str) -> list[dict]:
    """
    Constructs the message list to send to the Groq LLM.
    """
    # 1. Format retrieved memories
    if memories:
        memory_block = "\n".join([
            f"[{m['role'].upper()} | {m['timestamp'][:10]}]: {m['content'][:300]}"
            for m in memories
        ])
        context = f"""You are a helpful AI assistant with memory of past conversations.

Here are retrieved memories from past conversations:
---
{memory_block}
---
Use these to inform your response if relevant. If memories are contradictory or outdated, use your best judgment."""
    else:
        context = """You are a helpful AI assistant. This is the start of your 
conversation history — no past memories yet."""

    # 2. Get or create the short-term history for this session
    if session_id not in session_histories:
        session_histories[session_id] = []
        
    recent_history = session_histories[session_id]

    # 3. Combine System Prompt + Recent History + Current Message
    messages = [{"role": "system", "content": context}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_message})
    
    return messages


def _call_llm(messages: list[dict]):
    """Call Ollama via OpenAI SDK directly."""
    response = llm_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    return response


# ── Get a response from the LLM ───────────────────────────────────────────────
def chat(user_message: str, session_id: str) -> str:
    """
    Full pipeline: retrieve → build prompt → call LLM → save → return response.
    """
    print(f"\n[RAG] Retrieving memories for: '{user_message[:50]}...'")

    # Step 1: Retrieve relevant past memories
    memories = retrieve_memories(user_message, top_k=3)
    print(f"[RAG] Found {len(memories)} relevant memories")

    if memories:
        for m in memories:
            print(f"      → [{m['similarity']}] {m['content'][:60]}...")

    # Step 2: Build the prompt (now includes sliding window history)
    messages = build_prompt(user_message, memories, session_id)

    # Step 3: Call Ollama using the configured instant model.
    response = _call_llm(messages)
    assistant_reply = response.choices[0].message.content

    # Step 4: Save both the user message AND the reply to long-term memory
    save_memory("user", user_message, session_id)
    save_memory("assistant", assistant_reply, session_id)

    # Step 5: Save to short-term memory (Sliding window of last 4 turns = 8 messages)
    if session_id not in session_histories:
        session_histories[session_id] = []
        
    session_histories[session_id].append({"role": "user", "content": user_message})
    session_histories[session_id].append({"role": "assistant", "content": assistant_reply})
    
    session_histories[session_id] = session_histories[session_id][-8:]

    print(f"[RAG] Saved 2 new memories. Total in DB: {collection.count()}")

    return assistant_reply


# ── Stats: useful for your thesis evaluation ───────────────────────────────────
def get_stats() -> dict:
    return {
        "total_memories": collection.count(),
        "system": "Naive RAG (baseline via Groq)"
    }


# ── Interactive demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Naive RAG Baseline (Groq Edition) — Living Memory Thesis")
    print("=" * 60)
    print("Type 'quit' to exit | Type 'stats' to see memory count\n")

    session = "demo_session_groq"

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print(f"\nFinal stats: {get_stats()}")
            break
        if user_input.lower() == "stats":
            print(f"Stats: {get_stats()}")
            continue

        reply = chat(user_input, session)
        print(f"\nAssistant: {reply}\n")