# RAG Evaluation Framework (LlamaIndex-aligned)

This project provides a lightweight framework to evaluate chunking strategies
for Retrieval-Augmented Generation (RAG), following LlamaIndex recommendations.

Features:
- Four chunking strategies: fixed-size, structure-based, semantic, multigranular
- Simple evaluator for Response Time, Relevancy, and Faithfulness
- CLI runner to process `docs/` and `queries.json`

Quick start:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Initialize project structure (creates `docs/`, `queries.json`, `.env.example`):

```bash
python poc_chunking_azure.py --setup
```

3. Add documents to `docs/` and edit `queries.json`.

4. Run the evaluation via the package CLI:

```bash
python -m rag_eval.cli --docs docs --queries queries.json --output results.json
```

Notes:
- For production-quality chunking and embeddings, integrate LlamaIndex loaders
  and tokenizers, and replace the `SemanticChunker`'s `NotImplementedError`
  with a clustering implementation using real embeddings.
