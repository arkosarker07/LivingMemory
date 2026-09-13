"""
middleware/test_read_path.py
============================
Tests the read path formula combining Similarity (S), Importance (I), 
Decay (D) with a biological floor, and Frequency Bonus (B).
"""

import sys, os, math
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from middleware.memory_schema import MemoryObject
from middleware.read_path import compute_utility, DECAY_LAMBDA, UTILITY_THRESHOLD, DECAY_FLOOR


def simulate_memory(hours_ago: float, access_count: int, importance: float = 0.8) -> MemoryObject:
    past_time = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
    return MemoryObject(
        text=f"Simulated memory from {hours_ago} hrs ago",
        role="user",
        session_id="test_session",
        importance_score=importance,
        access_count=access_count,
        last_accessed_at=past_time,
    )

def run_test(name: str, memory: MemoryObject, similarity: float, condition: str, expected: bool):
    now = datetime.now()
    utility = compute_utility(memory, now, similarity)

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Similarity (S) : {similarity}")
    print(f"  Importance (I) : {memory.importance_score}")
    print(f"  Hours Ago (Δt) : {math.floor((now - datetime.fromisoformat(memory.last_accessed_at)).total_seconds()/3600)}")
    print(f"  Access Count   : {memory.access_count}")
    print(f"  => UTILITY     : {utility:.4f} (Threshold: {UTILITY_THRESHOLD})")
    
    # Evaluate the string condition (e.g. "utility >= 0.10")
    result = eval(condition.replace("utility", str(utility)))
    status = "✓ PASS" if result == expected else "✗ FAIL"
    print(f"  Result         : {status}")
    return result == expected

passes = 0
total = 4

# TEST 1: Fresh Context (Just saved, high relevance)
passes += run_test(
    name="Fresh relevant memory -> Injected",
    memory=simulate_memory(hours_ago=0.1, access_count=0),
    similarity=0.90,
    condition="utility >= 0.10",
    expected=True
)

# TEST 2: Old Noise (30 days ago, low relevance) -> Active Forgetting
passes += run_test(
    name="Old irrelevant memory (Noise) -> Dropped",
    memory=simulate_memory(hours_ago=720, access_count=0),
    similarity=0.20, # Low similarity
    condition="utility < 0.10",
    expected=True
)

# TEST 3: The Argentina Reactivation (90 days ago, High Relevance)
# 90 days = 2160 hours. Decay floor keeps it alive, high similarity saves it.
passes += run_test(
    name="The 3-Month Reactivation (High Sim, Old Memory) -> Injected",
    memory=simulate_memory(hours_ago=2160, access_count=0),
    similarity=0.95, # Direct question asked by user
    condition="utility >= 0.10",
    expected=True
)

# TEST 4: The Expert Reactivation (90 days ago, frequently used)
passes += run_test(
    name="3-Month Frequent Memory (Spacing Effect Bonus)",
    memory=simulate_memory(hours_ago=2160, access_count=10),
    similarity=0.95,
    condition="utility > 0.30", # Should score massive due to access=10
    expected=True
)

print(f"\n{'='*60}")
print(f"RESULTS: {passes}/{total} tests passed")
print(f"Formula: U = S * (I * max({DECAY_FLOOR}, e^(-{DECAY_LAMBDA}*Δt)) * (1 + log(1 + A)))")
print(f"{'='*60}")