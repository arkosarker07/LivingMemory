"""
middleware/test_write_path.py
==============================
Tests the write path with four scenarios:

  Test 1 — Normal save       : important new fact → should save
  Test 2 — Low value discard : small talk → should discard
  Test 3 — Exact duplicate   : same text twice → should discard
  Test 4 — Contradiction     : Python → JavaScript → should archive old, save new

Run with:
    python -m middleware.test_write_path
"""

import sys, os, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean DB BEFORE importing write_path — write_path opens ChromaDB at module
# load time, which immediately locks chroma.sqlite3. Delete first, import second.
if os.path.exists("./data/middleware_db"):
    shutil.rmtree("./data/middleware_db")
    print("Cleaned old test database.\n")

from middleware.write_path import process_memory, get_write_path_stats

SESSION = "test_session_001"


def run_test(name: str, text: str, expected: str):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Input: \"{text}\"")
    print(f"Expected result: {expected}")
    print("-" * 60)
    result = process_memory(text, "user", SESSION)
    status = "PASS" if result == expected else f"FAIL (got '{result}')"
    print(f"\nResult: {result} — {status}")
    return result == expected


passes = 0
total  = 4

# Test 1 — Normal important fact
passes += run_test(
    name="Normal save — important fact",
    text="My name is Alex and I am studying computer science at Dhaka University.",
    expected="saved"
)

# Test 2 — Low value small talk
passes += run_test(
    name="Discard — small talk",
    text="ok",
    expected="discarded"
)

# Test 3 — Exact duplicate
passes += run_test(
    name="Exact duplicate — same sentence",
    text="My name is Alex and I am studying computer science at Dhaka University.",
    expected="discarded"
)

# Test 4 — Contradiction (the critical one)
# First save the Python preference
process_memory("I prefer Python for all my programming projects.", "user", SESSION)

# Now contradict it
passes += run_test(
    name="Contradiction — Python → JavaScript",
    text="Actually I changed my mind. I now prefer JavaScript over Python.",
    expected="updated"
)

# Final stats
print(f"\n{'='*60}")
print(f"RESULTS: {passes}/{total} tests passed")
print(f"\nDatabase stats: {get_write_path_stats()}")
print(f"  Active memories   = retrievable")
print(f"  Archived memories = contradicted, kept for audit")
print(f"{'='*60}")