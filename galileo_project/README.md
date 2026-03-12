# Galileo RAG Evaluation — Chunk Attribution & Utilization

## Prerequisites
1. Free Galileo account → https://www.rungalileo.io (sign up, get API key)
2. Azure OpenAI resource with GPT-4o deployment

---

## Quick Start (3 steps)

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Configure credentials
```bash
cp .env.example .env
# Fill in:
#   GALILEO_API_KEY         → from rungalileo.io dashboard
#   GALILEO_PROJECT_NAME    → any name you like
#   AZURE_OPENAI_API_KEY    → your Azure key
#   AZURE_OPENAI_ENDPOINT   → https://YOUR-RESOURCE.openai.azure.com/
#   AZURE_OPENAI_DEPLOYMENT → gpt-4o
```

### Step 3 — Run

**Mode 1 — Pre-set answers (fastest, no Azure cost)**
```bash
python evaluate.py
```

**Mode 2 — Live Azure OpenAI answers (real pipeline test)**
```bash
python evaluate.py live
```

---

## What Gets Evaluated

| Galileo Scorer | What It Measures |
|---|---|
| `chunk_attribution_utilization_gpt` | Which chunks contributed + % utilization |
| `context_adherence_luna` | Is the answer grounded in chunks? |
| `completeness_luna` | Did the answer cover all relevant info? |

---

## What You'll See in Galileo Dashboard

After running, you get a URL to your Galileo project showing:
- Per-chunk attribution (which chunks were used vs ignored)
- Utilization score per query
- Side-by-side view of question → chunks → answer → scores
- Trends across multiple runs

---

## Plug In Your Real RAG Data

In `evaluate.py`, replace `RAG_SAMPLES` with your actual pipeline output:

```python
RAG_SAMPLES = [
    {
        "question":  "user query",
        "documents": ["chunk1", "chunk2", "chunk3"],  # from Azure AI Search
        "answer":    "LLM generated answer",
    },
]
```

---

## Ragas vs Galileo — Which to Use?

| | Ragas | Galileo |
|---|---|---|
| Results location | Console + CSV | Dashboard UI |
| Chunk Attribution | ✅ Faithfulness metric | ✅ Dedicated scorer |
| Setup effort | Low | Low (needs free account) |
| Best for | Offline CI/CD testing | Visual debugging & monitoring |
