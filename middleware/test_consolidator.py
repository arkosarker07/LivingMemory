"""
middleware/test_consolidator.py
===============================
Injects redundant memories to prove Phase 5 successfully clusters,
merges, and hard-deletes the originals.
"""

import sys, os, time
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean DB BEFORE importing middleware (ChromaDB locks the file on import)
if os.path.exists("./data/middleware_db"):
    shutil.rmtree("./data/middleware_db")
    print("Cleaned old test database.\n")

from middleware.write_path import process_memory
from middleware.consolidator import run_consolidator, collection

SESSION = "test_consolidation"

print("="*60)
print("PHASE 1: Injecting Raw Memories")
print("="*60)

# Inject 5 highly redundant memories (Should cluster)
python_statements = [
    "I really like using Python.",
    "Python is my preferred programming language for projects.",
    "I said Python is my favourite language to code in.",
    "For my thesis work, I strictly use Python.",
    "I prefer Python over Java."
]

for stmt in python_statements:
    process_memory(stmt, "user", SESSION)
    time.sleep(0.5)

# Inject 2 completely unrelated memories (Should NOT cluster)
process_memory("I live in Dhaka, Bangladesh.", "user", SESSION)
process_memory("I have a MacBook Air M1.", "user", SESSION)

initial_count = collection.count()
print(f"\n[Test] Initial database size: {initial_count} memories.")


print("\n" + "="*60)
print("PHASE 2: Running Consolidator")
print("="*60)

stats = run_consolidator()

print("\n" + "="*60)
print("PHASE 3: Results & Verification")
print("="*60)

final_count = collection.count()
print(f"Final database size : {final_count} memories")
print(f"Clusters merged     : {stats['clusters_merged']}")
print(f"Fragments deleted   : {stats['memories_deleted']}")

# Fetch the newly created semantic memory
semantic_memories = collection.get(
    where={"memory_type": {"$eq": "semantic"}},
    include=["documents", "metadatas"]
)

if semantic_memories and semantic_memories["documents"]:
    print("\n[PASS] SUCCESS: Found consolidated semantic memory:")
    print(f"  Text: {semantic_memories['documents'][0]}")

    meta = semantic_memories["metadatas"][0]
    print(f"  Inherited Importance : {meta['importance_score']}")
    print(f"  Status               : {meta['status']}")
else:
    print("\n[FAIL] No semantic memory was generated.")

# Verify Net Reduction
expected_final = initial_count - stats['memories_deleted'] + stats['memories_saved']
if final_count == expected_final:
    print(f"\n[PASS] SUCCESS: Database size correctly reduced from {initial_count} to {final_count}.")
else:
    print(f"\n[FAIL] Database size mismatch. Expected {expected_final}, got {final_count}.")