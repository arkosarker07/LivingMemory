# Living Memory — LoCoMo Evaluation Results

**Date:** 2026-09-13T22:55:27.052748
**Model:** `qwen2.5:7b`
**Conversations evaluated:** [0, 1]
**Total QA pairs:** 233
**Dry run:** No

---

## Main Results Table

| Condition | F1 Overall | Tokens/Query | DB Size (active) |
|---|---|---|---|
| Naive RAG (Baseline) | **0.0469** | 341 | 394 |
| Scoring Only | **0.0534** | 266 | 218 |
| Consolidation Only | **0.0549** | 239 | 382 |
| Full Living Memory | **0.0529** | 263 | 212 |

---

## F1 Score by Question Type

| Condition | Multi Hop | Open Domain | Single Hop | Temporal |
|---|---|---|---|---|
| Naive RAG (Baseline) | 0.0035 | 0.0748 | 0.0410 | 0.0590 |
| Scoring Only | 0.0084 | 0.0789 | 0.0535 | 0.0791 |
| Consolidation Only | 0.0022 | 0.0860 | 0.0534 | 0.0755 |
| Full Living Memory | 0.0029 | 0.0809 | 0.0603 | 0.0594 |

---

## Token Efficiency

Full Living Memory uses **+22.9%** fewer tokens per query compared to Naive RAG.
- Naive RAG: 341 tokens/query
- Full Living Memory: 263 tokens/query

---

## Memory Compression

Full Living Memory stores **+46.2%** fewer active memories compared to Naive RAG.
- Naive RAG: 394 memories
- Full Living Memory: 212 memories

---

## Methodology Notes

- Evaluation conducted on 2 configured conversation(s) from the LoCoMo benchmark. from the LoCoMo benchmark, consistent with computational constraints.
- Adversarial questions excluded (standard practice; task tests unanswerable-query detection, not memory reconstruction).
- Simulated time gap between sessions: 72 hours (3 days).
- F1 score: token-level with Porter stemming, matches LoCoMo paper metric.
- All conditions use model: `qwen2.5:7b`.