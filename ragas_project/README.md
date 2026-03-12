# Ragas RAG Evaluation — Chunk Attribution & Utilization

## Quick Start (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Set up your Azure OpenAI credentials
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and fill in your Azure OpenAI values:
# AZURE_OPENAI_API_KEY=your-key
# AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
# AZURE_OPENAI_DEPLOYMENT=gpt-4o
# AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-ada-002
```

### Step 3 — Run
```bash
python evaluate.py
```

---

## What the Metrics Mean

| Metric | Maps To | What It Tells You |
|---|---|---|
| `faithfulness` | Chunk Attribution | Was the answer grounded in retrieved chunks? |
| `context_precision` | Chunk Utilization | Were retrieved chunks actually useful? |
| `context_recall` | Coverage | Were all needed chunks retrieved? |
| `answer_relevancy` | Quality | Was the answer relevant to the question? |

## Score Guide
- **0.0 – 0.4** → Poor — chunks ignored / answer hallucinated
- **0.4 – 0.7** → Fair — partial grounding
- **0.7 – 1.0** → Good — well grounded in context

---

## Plugging In Your Own RAG Data

In `evaluate.py`, replace `RAG_SAMPLES` with your real pipeline output:

```python
RAG_SAMPLES = [
    {
        "question":     "your user query",
        "contexts":     ["chunk1 text", "chunk2 text", "chunk3 text"],  # from vector store
        "answer":       "your LLM generated answer",
        "ground_truth": "the correct expected answer",
    },
    # ... more samples
]
```

## Output
- Console: aggregate scores + per-sample table
- File: `ragas_results.csv` — full breakdown per query
