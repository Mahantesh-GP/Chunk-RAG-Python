"""Command-line runner for running evaluations across strategies."""
import argparse
import os
import json
from typing import List

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from rag_eval.strategies import (
    FixedSizeChunker,
    StructureChunker,
    SemanticChunker,
    MultigranularChunker,
    MockEmbedder,
)
from rag_eval.evaluator import RAGEvaluator


def load_documents_from_dir(docs_dir: str) -> List[str]:
    """Load documents from directory, supporting .txt, .md, and .pdf files."""
    texts = []
    for fn in os.listdir(docs_dir):
        path = os.path.join(docs_dir, fn)
        if not os.path.isfile(path):
            continue
            
        # Handle text and markdown files
        if fn.lower().endswith((".txt", ".md")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                print(f"Loaded: {fn}")
            except Exception as e:
                print(f"Error loading {fn}: {e}")
        
        # Handle PDF files
        elif fn.lower().endswith(".pdf"):
            if not PDF_AVAILABLE:
                print(f"Skipping {fn}: pypdf not installed. Run: pip install pypdf")
                continue
            try:
                reader = PdfReader(path)
                text_parts = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                if text_parts:
                    full_text = "\n\n".join(text_parts)
                    texts.append(full_text)
                    print(f"Loaded: {fn} ({len(reader.pages)} pages)")
                else:
                    print(f"Warning: {fn} has no extractable text")
            except Exception as e:
                print(f"Error loading PDF {fn}: {e}")
    
    return texts


def main():
    parser = argparse.ArgumentParser(description="Run RAG chunking strategy evaluation")
    parser.add_argument("--docs", default="docs", help="Directory with docs")
    parser.add_argument("--queries", default="queries.json", help="Queries JSON file")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON")
    args = parser.parse_args()

    docs = load_documents_from_dir(args.docs)
    if not docs:
        print("No documents found in", args.docs)
        return

    if os.path.exists(args.queries):
        with open(args.queries, "r", encoding="utf-8") as f:
            qdata = json.load(f)
            if isinstance(qdata, dict) and "queries" in qdata:
                queries = qdata["queries"]
            elif isinstance(qdata, list):
                queries = qdata
            else:
                raise ValueError("Invalid queries.json format")
    else:
        queries = ["What is the main topic?", "What are the key findings?"]

    # Prepare strategies
    embedder = MockEmbedder(embedding_dim=128)  # Demo embedder
    strategies = {
        "fixed": FixedSizeChunker(chunk_size=256, chunk_overlap=20),
        "structure": StructureChunker(min_chunk_size=128, max_chunk_size=512),
        "semantic": SemanticChunker(embedder=embedder, chunk_size=512, similarity_threshold=0.75),
        "multigranular": MultigranularChunker(),
    }

    evaluator = RAGEvaluator(top_k=5)

    # For each document and strategy, produce chunks and run queries
    for doc in docs:
        for name, strat in strategies.items():
            try:
                chunks = strat.chunk(doc)
            except NotImplementedError:
                print(f"Strategy '{name}' requires extra setup; skipping.")
                continue
            for q in queries:
                evaluator.evaluate(q, name, chunks)

    # Export results
    summary = evaluator.summary()
    results = [
        {
            "query": r.query,
            "strategy": r.strategy,
            "response_time": r.response_time,
            "relevancy": r.relevancy,
            "faithfulness": r.faithfulness,
            "retrieved_chunks": r.chunks,
        }
        for r in evaluator.results
    ]
    out = {"summary": summary, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
