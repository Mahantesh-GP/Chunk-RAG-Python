# Azure RAG Evaluation — Hybrid Search + Chunk Attribution

## Overview

This project implements your architect's idea of TWO attribution approaches:

| Approach | When | Method | LLM Cost |
|---|---|---|---|
| **Pre-Answer** | BEFORE LLM call | Azure AI Search reranker score | ❌ Free |
| **Post-Answer** | AFTER LLM call | Azure AI Eval Groundedness | ✅ Token cost |

---

## Quick Start

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Setup credentials
```bash
cp .env.example .env
# Fill in Azure OpenAI + Azure AI Search credentials
```

### Step 3 — Index your chunks
```bash
# With your chunks file
python index_chunks.py your_chunks.json

# With sample data
python index_chunks.py
```

### Step 4 — Run evaluation
```bash
# With your 300+ questions
python evaluate.py your_questions.json

# With sample questions
python evaluate.py
```

---

## Input File Formats

### chunks.json
```json
[
  {
    "id": "chunk_001",
    "content": "chunk text here",
    "source": "document.pdf",
    "chunk_strategy": "fixed_1024",
    "chunk_index": 1
  }
]
```

### questions.json
```json
[
  { "id": "q_001", "question": "What is EF Core?" },
  { "id": "q_002", "question": "How does RAG work?" }
]
```

---

## Output — output_results.json

```json
{
  "metadata": {
    "total_questions": 300,
    "successful": 298,
    "elapsed_seconds": 245
  },
  "aggregate": {
    "avg_pre_answer_utilization": 0.6,
    "avg_post_answer_utilization": 0.75,
    "avg_groundedness_score": 4.2
  },
  "results": [
    {
      "id": "q_001",
      "question": "What is EF Core?",
      "answer": "EF Core is a .NET ORM...",
      "retrieved_chunks": [...],
      "pre_answer_attribution": {
        "reranker_scores": [3.8, 3.2, 1.1, 0.5, 0.2],
        "attribution_list": [1, 1, 0, 0, 0],
        "chunk_utilization": 0.4
      },
      "post_answer_attribution": {
        "overall_groundedness": 5,
        "chunk_scores": [5, 4, 1, 1, 1],
        "attribution_list": [1, 1, 0, 0, 0],
        "chunk_utilization": 0.4
      }
    }
  ]
}
```

---

## How Hybrid Search Works

```
User Question
    ↓
┌─────────────────────────────────────┐
│  Azure AI Search Hybrid             │
│  Vector Search (embeddings) ──┐     │
│  Keyword Search (BM25)   ────RRF    │
│  Semantic Reranker        ────↓     │
│  Returns top-K chunks + scores      │
└─────────────────────────────────────┘
    ↓
Pre-Answer Attribution (reranker score)
    ↓
GPT-4o generates answer
    ↓
Post-Answer Attribution (groundedness)
    ↓
output_results.json
```

---

## Score Guide

### Pre-Answer (Reranker Score 0-4)
- **0-1** → Low relevance → NOT attributed
- **2-4** → High relevance → ATTRIBUTED (default threshold = 2)

### Post-Answer (Groundedness 1-5)
- **1-2** → Not grounded
- **3**   → Partially grounded → ATTRIBUTED threshold
- **4-5** → Well grounded → ATTRIBUTED
