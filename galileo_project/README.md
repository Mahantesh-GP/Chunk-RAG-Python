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


# Uninstall old
pip uninstall promptquality -y

# Install new
pip install "galileo[openai]" python-dotenv





import os
from dotenv import load_dotenv
load_dotenv()

from galileo import GalileoMetrics
from galileo.datasets import create_dataset
from galileo.experiments import run_experiment
from galileo.prompts import create_prompt
from galileo.resources.models.prompt_run_settings import PromptRunSettings
from galileo import Message, MessageRole

os.environ["GALILEO_API_KEY"]  = os.getenv("GALILEO_API_KEY")
os.environ["GALILEO_PROJECT"]  = "rag_attribution_utilization"

# Your RAG samples
samples = [
    {
        "query": "What is Azure OpenAI?",
        "context": "Azure OpenAI provides GPT-4 via REST API on Microsoft Azure. It supports VNet integration.",
        "answer": "Azure OpenAI is a Microsoft service giving access to GPT-4 via REST API on Azure."
    },
    {
        "query": "How does RAG reduce hallucination?",
        "context": "RAG grounds LLM responses in retrieved documents, reducing hallucination by conditioning on real source material.",
        "answer": "RAG reduces hallucination by grounding answers in retrieved documents instead of model memory."
    },
    {
        "query": "What is EF Core?",
        "context": "EF Core is an open source ORM for .NET. It supports SQL Server and uses DbContext for data access.",
        "answer": "EF Core is a .NET ORM that supports SQL Server using DbContext for querying and saving data."
    }
]

# Create dataset
dataset = create_dataset(
    name="rag-eval-dataset",
    data=[{"input": s["query"], "context": s["context"]} for s in samples]
)

# Create prompt template
prompt = create_prompt(
    name="rag-prompt",
    template=[
        Message(role=MessageRole.system, content="Answer using this context: {context}"),
        Message(role=MessageRole.user, content="{input}")
    ]
)

# Run experiment with metrics
results = run_experiment(
    name="rag-chunk-eval",
    dataset=dataset,
    prompt=prompt,
    settings=PromptRunSettings(
        model="gpt-4o",
    ),
    metrics=[
        GalileoMetrics.context_adherence,
        GalileoMetrics.completeness,
        GalileoMetrics.chunk_attribution,
    ]
)

print("✅ Done! Check Experiments tab in Galileo dashboard")
print(results)