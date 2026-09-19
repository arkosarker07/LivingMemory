# Living Memory — LoCoMo Evaluation Results

**Date:** 2026-09-19T02:40:49.947291
**Model:** `qwen2.5:7b`
**Conversations evaluated:** [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
**Total QA pairs:** 1540
**Dry run:** No

---

## Main Results Table

| Condition | F1 Overall | Tokens/Query | DB Size (active) |
|---|---|---|---|
| Naive RAG (Baseline) | **0.0575** | 341 | 588 |
| Scoring Only | **0.0524** | 281 | 276 |
| Consolidation Only | **0.0527** | 239 | 572 |
| Full Living Memory | **0.0537** | 278 | 271 |

---

## F1 Score by Question Type

| Condition | Multi Hop | Open Domain | Single Hop | Temporal |
|---|---|---|---|---|
| Naive RAG (Baseline) | 0.0148 | 0.0768 | 0.0536 | 0.0504 |
| Scoring Only | 0.0118 | 0.0651 | 0.0581 | 0.0622 |
| Consolidation Only | 0.0087 | 0.0672 | 0.0600 | 0.0605 |
| Full Living Memory | 0.0118 | 0.0677 | 0.0576 | 0.0628 |

---

## Token Efficiency

Full Living Memory uses **+18.5%** fewer tokens per query compared to Naive RAG.
- Naive RAG: 341 tokens/query
- Full Living Memory: 278 tokens/query

---

## Memory Compression

Full Living Memory stores **+53.9%** fewer active memories compared to Naive RAG.
- Naive RAG: 588 memories
- Full Living Memory: 271 memories

---

## Methodology Notes

- Evaluation conducted on 10 configured conversation(s) from the LoCoMo benchmark. from the LoCoMo benchmark, consistent with computational constraints.
- Adversarial questions excluded (standard practice; task tests unanswerable-query detection, not memory reconstruction).
- Simulated time gap between sessions: 72 hours (3 days).
- F1 score: token-level with Porter stemming, matches LoCoMo paper metric.
- All conditions use model: `qwen2.5:7b`.