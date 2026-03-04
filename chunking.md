# Chunking in RAG Systems

*Last updated: March 4, 2026*

Chunking is a critical preprocessing step in retrieval-augmented generation (RAG) systems. It involves dividing long documents into smaller, coherent pieces (``chunks``) that can be indexed, embedded, and retrieved efficiently. Good chunking improves retrieval accuracy, reduces context window waste, and enhances downstream language model responses.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Chunking Matters](#why-chunking-matters)
3. [Chunking Methods]
   - [Fixed-Size Chunking](#fixed-size-chunking)
   - [Semantic Similarity Chunking](#semantic-similarity-chunking)
   - [Hybrid Chunking (Docling)](#hybrid-chunking-docling)
4. [Comparison of Methods](#comparison-of-methods)
5. [Implementation Details](#implementation-details)
6. [Evaluation and Findings](#evaluation-and-findings)
7. [Best Practices](#best-practices)
8. [Appendix: Sample Code](#appendix-sample-code)

---

## Introduction

This document describes the chunking techniques we evaluated during our RAG research. The goal is to provide clear guidance and examples so that others can replicate or extend the work.

## Why Chunking Matters

Large documents cannot be directly fed into most retrieval systems or LLMs due to token limits. Chunking:

- Preserves semantic boundaries.
- Enables efficient vector storage and search.
- Improves relevance of retrieved passages.

## Chunking Methods

### Fixed-Size Chunking

Documents are split into equal-sized segments irrespective of content. We experimented with token sizes of **128**, **512**, **1024**, and **2048**. Parsing was done using Azure Digital Infinity (AZ DI) with Markdown output; the `llama-index` library handled the chunk generation and indexing.

### Semantic Similarity Chunking

This method uses embedding models (via AZ DI MD) to identify points where the semantic meaning shifts. Chunks are created so that each contains a cohesive concept. It avoids splitting sentences but requires extra embedding computations.

### Hybrid Chunking (Docling)

Docling’s `HybridChunker` merges structural heuristics (e.g., heading or paragraph boundaries) with semantic checks. It first applies rule-based splits, then fine-tunes boundaries based on semantic similarity to maintain coherence and avoid mid-sentence breaks.

## Comparison of Methods

| Method                   | Pros                                             | Cons                                           | Best Use Case                               |
|--------------------------|--------------------------------------------------|------------------------------------------------|---------------------------------------------|
| Fixed-Size               | Fast, predictable memory usage                   | May cut sentences or ideas mid-chunk          | Small/regular data                          |
| Semantic Similarity      | Natural boundaries, high coherence               | Slower, embedding cost                        | Rich or complex text where semantics matter |
| Hybrid (Docling)         | Balanced speed & quality, fewer awkward splits   | More complex implementation                   | General-purpose, variable-structure docs   |

## Implementation Details

- **Azure Document Intelligence (Azure DI)**: Microsoft cloud service for OCR, layout analysis, and text extraction. We used it to ingest PDFs/Office docs and convert them to clean Markdown before chunking. (Previously mis‑referred to as "Azure Digital Infinity".)
- **LLama-Index**: Generated fixed-size chunks and provided indexing utilities.
- **Docling HybridChunker**: Invoked via Python; configuration shown in appendix.

### Visualizing Chunk Boundaries

To make the differences more tangible, we recommend capturing and embedding screenshots or diagrams from sample documents:

1. **Original document extract** – show a page of text with headings, lists, and paragraphs for context.
2. **Fixed-size chunks** – overlay colored boxes every N tokens (e.g. 512), demonstrating how splits may occur mid-sentence.
3. **Semantic chunks** – overlay based on computed semantic breaks so chunks align with sentences or paragraphs.
4. **Hybrid chunks** – illustrate how document structure (headers, paragraphs) is respected and then refined via semantic similarity.

> _Add image files under a `screenshots/` folder and reference them using Markdown, e.g.:_
> ```md
> ![Fixed chunks](screenshots/fixed.png)
> ```

Visual aids help reviewers quickly grasp why hybrid chunking often appears cleaner and why fixed-size splits can be jarring.

## Evaluation and Findings

We measured the following metrics:
- Retrieval precision@k
- Token efficiency in LLM prompts
- Downstream response quality (human evaluation)

**Key takeaway:** Hybrid chunking consistently delivered the best trade-off between retrieval accuracy and prompt efficiency, especially on documents with mixed formatting (code, tables, prose).

## Best Practices

1. Start with hybrid chunking as a baseline; fallback to fixed-size for performance-sensitive workloads.
2. Cache embeddings if using semantic methods to reduce repeated computation.
3. Experiment with token size ranges for fixed chunking based on document type.

## Appendix: Sample Code

```python
# Example using llama-index for fixed-size chunking
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

reader = SimpleDirectoryReader("./data")
index = GPTVectorStoreIndex.from_documents(reader.load_data())

# Semantic chunking example (pseudo-code)
chunks = semantic_chunker.split(document)

# Docling hybrid example
from docling import HybridChunker
hc = HybridChunker()
chunks = hc.chunk(document)
```

More code snippets and datasets can be added as the project evolves.

---

Feel free to extend this document with diagrams, additional evaluation graphs, or links to related research.