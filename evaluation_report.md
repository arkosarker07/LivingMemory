# Living Memory — LoCoMo Evaluation Results

**Date:** 2026-09-18T15:26:10.961295
**Model:** `qwen2.5:7b`
**Conversations evaluated:** [0, 1]
**Total QA pairs:** 233
**Dry run:** No

---

## Main Results Table

| Condition | F1 Overall | Tokens/Query | DB Size (active) |
|---|---|---|---|
| Naive RAG (Baseline) | **0.0462** | 338 | 394 |
| Scoring Only | **0.0499** | 266 | 218 |
| Consolidation Only | **0.0549** | 241 | 382 |
| Full Living Memory | **0.0536** | 277 | 215 |

---

## F1 Score by Question Type

| Condition | Multi Hop | Open Domain | Single Hop | Temporal |
|---|---|---|---|---|
| Naive RAG (Baseline) | 0.0019 | 0.0740 | 0.0408 | 0.0631 |
| Scoring Only | 0.0051 | 0.0764 | 0.0472 | 0.0693 |
| Consolidation Only | 0.0037 | 0.0819 | 0.0599 | 0.0878 |
| Full Living Memory | 0.0032 | 0.0864 | 0.0405 | 0.0824 |

---

## Token Efficiency

Full Living Memory uses **+18.2%** fewer tokens per query compared to Naive RAG.
- Naive RAG: 338 tokens/query
- Full Living Memory: 277 tokens/query

---

## Memory Compression

Full Living Memory stores **+45.4%** fewer active memories compared to Naive RAG.
- Naive RAG: 394 memories
- Full Living Memory: 215 memories

---

## Methodology Notes

- Evaluation conducted on 2 configured conversation(s) from the LoCoMo benchmark. from the LoCoMo benchmark, consistent with computational constraints.
- Adversarial questions excluded (standard practice; task tests unanswerable-query detection, not memory reconstruction).
- Simulated time gap between sessions: 72 hours (3 days).
- F1 score: token-level with Porter stemming, matches LoCoMo paper metric.
- All conditions use model: `qwen2.5:7b`.