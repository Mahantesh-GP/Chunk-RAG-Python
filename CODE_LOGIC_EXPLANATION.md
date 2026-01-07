# evaluation_results.json Generation - Complete Code Logic Explanation

## Table of Contents
1. [Overview](#overview)
2. [Complete Pipeline Flow](#complete-pipeline-flow)
3. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
4. [Single Evaluation Deep Dive](#single-evaluation-deep-dive)
5. [Retrieval Algorithm](#retrieval-algorithm)
6. [Relevancy Calculation](#relevancy-calculation)
7. [Faithfulness Calculation](#faithfulness-calculation)
8. [Result Aggregation & JSON Export](#result-aggregation--json-export)
9. [Output Structure](#output-structure)
10. [Complete Example with Real Data](#complete-example-with-real-data)

---

## Overview

The `evaluation_results.json` file is generated through a complete pipeline that:
1. **Loads documents** from files (PDF, Markdown, Text)
2. **Loads evaluation queries** from `queries.json`
3. **Creates chunking strategies** (Fixed-Size, Structure-based, Semantic, Multigranular)
4. **Runs evaluations** for each document × strategy × query combination
5. **Calculates metrics** (Response Time, Relevancy, Faithfulness)
6. **Aggregates results** per strategy
7. **Exports to JSON** for analysis

**Total evaluations per document:** Number of Strategies × Number of Queries
- With 4 strategies and 3 queries = **12 evaluations per document**

---

## Complete Pipeline Flow

### 9-Step Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: USER RUNS COMMAND                                      │
│  $ python -m rag_eval.cli --docs ./docs --queries queries.json  │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: LOAD DOCUMENTS                                         │
│  load_documents_from_dir() reads .txt, .md, .pdf files         │
│  Returns: List[str] containing full document texts              │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: LOAD QUERIES                                           │
│  json.load("queries.json") parses query file                    │
│  Returns: List[str] containing evaluation queries               │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: INITIALIZE STRATEGIES                                  │
│  Create instances of 4 chunking strategies:                     │
│  - FixedSizeChunker(256, 20)                                    │
│  - StructureChunker(128, 512)                                   │
│  - SemanticChunker(0.75)                                        │
│  - MultigranularChunker()                                       │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: TRIPLE NESTED LOOP - RUN ALL EVALUATIONS             │
│  for doc in docs:                    # 1 iteration             │
│    for name, strat in strategies:    # 4 iterations            │
│      chunks = strat.chunk(doc)       # Chunk the document      │
│      for query in queries:           # 3 iterations            │
│        result = evaluator.evaluate() # Calculate 1 result      │
│        store_result(result)          # Add to results list     │
│                                                                 │
│  Total results stored: 1 × 4 × 3 = 12 EvaluationResult objects│
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: CALCULATE AGGREGATES                                   │
│  For each strategy, calculate:                                  │
│  - Average response_time across 3 queries                       │
│  - Average relevancy across 3 queries                           │
│  - Average faithfulness across 3 queries                        │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: BUILD SUMMARY DICT                                     │
│  summary = {                                                    │
│    "strategy_name": {                                           │
│      "avg_response_time": float,                                │
│      "avg_relevancy": float,                                    │
│      "avg_faithfulness": float,                                 │
│      "count": int                                               │
│    }                                                            │
│  }                                                              │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: BUILD RESULTS ARRAY                                    │
│  results = [EvaluationResult_1, EvaluationResult_2, ...]        │
│  Each result contains:                                          │
│  - query, strategy, response_time                               │
│  - relevancy, faithfulness, retrieved_chunks                    │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 9: EXPORT TO JSON                                         │
│  json.dump({                                                    │
│    "summary": summary,                                          │
│    "results": results                                           │
│  }, file)                                                       │
│  Output: evaluation_results.json                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Code Walkthrough

### From cli.py - Main Orchestration

```python
def main():
    # STEP 1: Parse command-line arguments
    args = parse_args()
    
    # STEP 2: Load documents from directory
    docs = load_documents_from_dir(args.docs)
    # docs = ["Document 1 full text...", "Document 2 full text..."]
    
    # STEP 3: Load queries from JSON file
    with open(args.queries, 'r') as f:
        queries = json.load(f)
    # queries = ["What is rate limiting?", "How to implement?", "Best practices?"]
    
    # STEP 4: Initialize chunking strategies
    strategies = {
        "fixed_size": FixedSizeChunker(chunk_size=256, overlap=20),
        "structure": StructureChunker(min_words=128, max_words=512),
        "semantic": SemanticChunker(similarity_threshold=0.75),
        "multigranular": MultigranularChunker()
    }
    
    # STEP 5: Create evaluator
    evaluator = RAGEvaluator()
    
    # STEP 6: TRIPLE NESTED LOOP - Core evaluation pipeline
    for doc in docs:  # Outer loop: each document
        for name, strat in strategies.items():  # Middle loop: each strategy
            # Chunk the document using this strategy
            chunks = strat.chunk(doc)
            
            for query in queries:  # Inner loop: each query
                # Run one evaluation (stores result internally)
                evaluator.evaluate(query, name, chunks)
    
    # STEP 7: Get aggregated results
    summary = evaluator.summary()
    
    # STEP 8: Get all individual results
    results = evaluator.results
    
    # STEP 9: Export to JSON
    output = {
        "summary": summary,
        "results": [asdict(r) for r in results]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
```

### From load_documents_from_dir()

```python
def load_documents_from_dir(directory):
    """Load documents from .txt, .md, and .pdf files"""
    documents = []
    
    for file_path in Path(directory).glob("*"):
        if file_path.suffix == ".pdf":
            # PDF handling
            reader = PdfReader(file_path)
            text = "\n\n".join(page.extract_text() for page in reader.pages)
            documents.append(text)
        
        elif file_path.suffix in [".txt", ".md"]:
            # Text/Markdown handling
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    
    return documents
    # Returns: ["Document 1 full text", "Document 2 full text", ...]
```

---

## Single Evaluation Deep Dive

When one evaluation is called:
```python
evaluator.evaluate(
    query="What is rate limiting?",
    strategy="fixed_size",
    chunks=["chunk1", "chunk2", "chunk3", ..., "chunkN"]
)
```

This executes the following process:

### Phase 1: Retrieve Relevant Chunks

```python
def evaluate(self, query, strategy_name, chunks):
    start_time = time.time()
    
    # RETRIEVAL STEP: Find top-5 most relevant chunks
    retrieved_chunks = self._retrieve(query, chunks)
    # retrieved_chunks = [chunk_2, chunk_5, chunk_1]  # Top 3-5 chunks
```

### Phase 2: Calculate Response Time

```python
    response_time = (time.time() - start_time) * 1000  # Convert to ms
    # Typical value: 5-7 milliseconds
```

### Phase 3: Calculate Relevancy Score

```python
    relevancy = self._relevancy(query, retrieved_chunks)
    # Relevancy = average Jaccard similarity across retrieved chunks
    # Range: 0.0 (no match) to 1.0 (perfect match)
    # Typical value: 0.0 - 0.06
```

### Phase 4: Calculate Faithfulness Score

```python
    faithfulness = self._faithfulness(query, retrieved_chunks)
    # Faithfulness = percentage of query terms found in chunks
    # Range: 0.0 (no terms found) to 1.0 (all terms found)
    # Typical value: 0.16 - 0.58
```

### Phase 5: Store Result

```python
    result = EvaluationResult(
        query=query,
        strategy=strategy_name,
        response_time=response_time,
        relevancy=relevancy,
        faithfulness=faithfulness,
        retrieved_chunks=retrieved_chunks
    )
    self.results.append(result)
    # results list now has 1 more EvaluationResult object
```

---

## Retrieval Algorithm

### How Chunks Are Scored and Retrieved

The `_retrieve()` method uses **Jaccard Similarity** to rank chunks:

```python
def _retrieve(self, query, chunks, top_k=5):
    """
    Jaccard Similarity = |intersection| / |union|
    
    For each chunk, calculate how many tokens overlap with the query,
    divided by total unique tokens in both query and chunk.
    """
    
    # Tokenize query (split by whitespace, lowercase)
    query_tokens = set(query.lower().split())
    # Example: "What is rate limiting?" → {"what", "is", "rate", "limiting"}
    
    # Score each chunk
    chunk_scores = []
    
    for chunk in chunks:
        # Tokenize chunk
        chunk_tokens = set(chunk.lower().split())
        
        # Calculate Jaccard Similarity
        intersection = query_tokens & chunk_tokens  # Common tokens
        union = query_tokens | chunk_tokens         # All unique tokens
        
        if len(union) == 0:
            similarity = 0.0
        else:
            similarity = len(intersection) / len(union)
        
        chunk_scores.append((similarity, chunk))
    
    # Sort by similarity (descending) and take top-k
    top_chunks = sorted(chunk_scores, key=lambda x: x[0], reverse=True)[:top_k]
    
    # Return only the chunk texts (not scores)
    retrieved = [chunk for score, chunk in top_chunks]
    
    return retrieved
```

### Example Calculation

```
Query: "rate limiting"
Query tokens: {"rate", "limiting"}

Chunk 1: "Rate limiting is a technique..."
Chunk 1 tokens: {"rate", "limiting", "is", "a", "technique", ...} (10 tokens)
intersection: {"rate", "limiting"} (2 tokens)
union: {"rate", "limiting", "is", "a", "technique", ...} (10 tokens)
Jaccard = 2/10 = 0.20

Chunk 2: "The algorithm uses token buckets"
Chunk 2 tokens: {"the", "algorithm", "uses", "token", "buckets"} (5 tokens)
intersection: {} (0 tokens)
union: {"rate", "limiting", "the", "algorithm", "uses", "token", "buckets"} (7 tokens)
Jaccard = 0/7 = 0.00

Result: Chunk 1 ranks higher (0.20 > 0.00)
```

---

## Relevancy Calculation

### Definition
**Relevancy** measures how well retrieved chunks match the query using Jaccard similarity.

### Algorithm

```python
def _relevancy(self, query, retrieved_chunks):
    """
    Calculate average Jaccard similarity between query and each chunk.
    
    Steps:
    1. For each chunk, calculate Jaccard(query_tokens, chunk_tokens)
    2. Average all Jaccard scores
    3. Return average as relevancy metric
    """
    
    query_tokens = set(query.lower().split())
    jaccard_scores = []
    
    for chunk in retrieved_chunks:
        chunk_tokens = set(chunk.lower().split())
        
        intersection = query_tokens & chunk_tokens
        union = query_tokens | chunk_tokens
        
        if len(union) == 0:
            jaccard = 0.0
        else:
            jaccard = len(intersection) / len(union)
        
        jaccard_scores.append(jaccard)
    
    # Average of all Jaccard scores
    if len(jaccard_scores) == 0:
        return 0.0
    
    relevancy = sum(jaccard_scores) / len(jaccard_scores)
    
    return relevancy
```

### Example

```
Query: "rate limiting"
Query tokens: {"rate", "limiting"}

Retrieved chunks (top 3):
  Chunk 1: "Rate limiting algorithms..." 
    Tokens: {rate, limiting, algorithms, ...} (8 tokens total)
    Intersection: {rate, limiting} (2)
    Union: {rate, limiting, algorithms, ...} (8)
    Jaccard = 2/8 = 0.25

  Chunk 2: "Implement rate limiting using tokens"
    Tokens: {implement, rate, limiting, using, tokens} (5 tokens)
    Intersection: {rate, limiting} (2)
    Union: {implement, rate, limiting, using, tokens} (5)
    Jaccard = 2/5 = 0.40

  Chunk 3: "Best practices for rate limiting"
    Tokens: {best, practices, for, rate, limiting} (5 tokens)
    Intersection: {rate, limiting} (2)
    Union: {best, practices, for, rate, limiting} (5)
    Jaccard = 2/5 = 0.40

RELEVANCY = (0.25 + 0.40 + 0.40) / 3 = 0.35
```

---

## Faithfulness Calculation

### Definition
**Faithfulness** measures what percentage of query terms are found in the retrieved chunks.

### Algorithm

```python
def _faithfulness(self, query, retrieved_chunks):
    """
    Calculate percentage of query terms (>3 characters) found in chunks.
    
    Steps:
    1. Extract query terms longer than 3 characters
    2. For each chunk, check which query terms appear in it
    3. Calculate: (unique query terms found) / (total query terms)
    4. Return as faithfulness percentage
    """
    
    # Extract significant query terms (length > 3 characters)
    query_terms = set(
        term.lower() 
        for term in query.split() 
        if len(term) > 3
    )
    
    # Combine all retrieved chunks into one text
    combined_chunks = " ".join(retrieved_chunks).lower()
    
    # Count how many query terms appear in combined chunks
    found_terms = sum(
        1 for term in query_terms 
        if term in combined_chunks
    )
    
    # Calculate faithfulness as percentage
    if len(query_terms) == 0:
        return 0.0
    
    faithfulness = found_terms / len(query_terms)
    
    return faithfulness
```

### Example

```
Query: "What is rate limiting algorithm?"
Query terms (>3 chars): {"rate", "limiting", "algorithm"} (3 terms)

Retrieved chunks combined:
"Rate limiting is implemented using token buckets. The algorithm 
maintains state per client..."

Checking for each term:
  - "rate" in text? YES
  - "limiting" in text? YES
  - "algorithm" in text? YES

FAITHFULNESS = 3/3 = 1.00 (100%)

---

Another example:
Query: "What is rate limiting?"
Query terms (>3 chars): {"rate", "limiting"} (2 terms)

Retrieved chunks combined:
"Token buckets maintain state. Requests are queued..."

Checking for each term:
  - "rate" in text? NO
  - "limiting" in text? NO

FAITHFULNESS = 0/2 = 0.00 (0%)
```

---

## Result Aggregation & JSON Export

### Building Individual Results Array

After all 12 evaluations (1 doc × 4 strategies × 3 queries), the `evaluator.results` list contains:

```python
evaluator.results = [
    EvaluationResult(
        query="What is rate limiting?",
        strategy="fixed_size",
        response_time=5.2,
        relevancy=0.042,
        faithfulness=0.583,
        retrieved_chunks=["chunk_2", "chunk_5", ...]
    ),
    EvaluationResult(
        query="What is rate limiting?",
        strategy="structure",
        response_time=5.1,
        relevancy=0.035,
        faithfulness=0.417,
        retrieved_chunks=["chunk_3", "chunk_8", ...]
    ),
    # ... 10 more results
]
```

### Calculating Summary Aggregates

```python
def summary(self):
    """
    Aggregate results by strategy to create summary statistics.
    Calculate averages across all queries for each strategy.
    """
    
    from collections import defaultdict
    
    stats = defaultdict(lambda: {"count": 0, "response_time": 0, 
                                  "relevancy": 0, "faithfulness": 0})
    
    # Iterate through all 12 results
    for result in self.results:
        strategy = result.strategy
        
        # Accumulate metrics
        stats[strategy]["count"] += 1
        stats[strategy]["response_time"] += result.response_time
        stats[strategy]["relevancy"] += result.relevancy
        stats[strategy]["faithfulness"] += result.faithfulness
    
    # Calculate averages by dividing by count
    summary_dict = {}
    for strategy in stats:
        count = stats[strategy]["count"]
        summary_dict[strategy] = {
            "avg_response_time": stats[strategy]["response_time"] / count,
            "avg_relevancy": stats[strategy]["relevancy"] / count,
            "avg_faithfulness": stats[strategy]["faithfulness"] / count,
            "count": count
        }
    
    return summary_dict
```

### Example Aggregation

```python
# After processing 12 results (4 strategies × 3 queries):

summary = {
    "fixed_size": {
        "avg_response_time": 5.15,  # (5.2 + 5.1 + 5.2) / 3
        "avg_relevancy": 0.0423,    # (0.042 + 0.043 + 0.042) / 3
        "avg_faithfulness": 0.5833, # (0.583 + 0.583 + 0.583) / 3
        "count": 3
    },
    "structure": {
        "avg_response_time": 5.20,
        "avg_relevancy": 0.0357,
        "avg_faithfulness": 0.4167,
        "count": 3
    },
    "semantic": {
        "avg_response_time": 6.15,
        "avg_relevancy": 0.0593,  # Best relevancy
        "avg_faithfulness": 0.1667,  # Lowest faithfulness
        "count": 3
    },
    "multigranular": {
        "avg_response_time": 5.10,
        "avg_relevancy": 0.0423,
        "avg_faithfulness": 0.5833,  # Tied with fixed_size
        "count": 3
    }
}
```

### Final JSON Export

```python
output = {
    "summary": summary_dict,
    "results": [asdict(r) for r in self.results]
}

with open(args.output, 'w') as f:
    json.dump(output, f, indent=2)
```

---

## Output Structure

### JSON File Structure (evaluation_results.json)

```json
{
  "summary": {
    "fixed_size": {
      "avg_response_time": 5.15,
      "avg_relevancy": 0.0423,
      "avg_faithfulness": 0.5833,
      "count": 3
    },
    "structure": {
      "avg_response_time": 5.20,
      "avg_relevancy": 0.0357,
      "avg_faithfulness": 0.4167,
      "count": 3
    },
    "semantic": {
      "avg_response_time": 6.15,
      "avg_relevancy": 0.0593,
      "avg_faithfulness": 0.1667,
      "count": 3
    },
    "multigranular": {
      "avg_response_time": 5.10,
      "avg_relevancy": 0.0423,
      "avg_faithfulness": 0.5833,
      "count": 3
    }
  },
  "results": [
    {
      "query": "What is rate limiting?",
      "strategy": "fixed_size",
      "response_time": 5.2,
      "relevancy": 0.042,
      "faithfulness": 0.583,
      "retrieved_chunks": [
        "Rate limiting is a technique...",
        "Algorithms include token bucket...",
        "Implementation uses counters..."
      ]
    },
    {
      "query": "What is rate limiting?",
      "strategy": "structure",
      "response_time": 5.1,
      "relevancy": 0.035,
      "faithfulness": 0.417,
      "retrieved_chunks": [
        "Rate limiting algorithms...",
        "Best practices for implementation...",
        "Token bucket mechanism..."
      ]
    },
    // ... 10 more results (one for each query × strategy combination)
  ]
}
```

---

## Complete Example with Real Data

### Input Setup

**File: queries.json**
```json
[
  "What is rate limiting?",
  "How to implement rate limiting?",
  "Best practices for rate limiting?"
]
```

**Document Content (excerpt):**
```
Rate limiting is a technique used to control the rate at which
requests are processed. It prevents abuse and ensures fair resource
allocation. The most common implementation uses token buckets...

The token bucket algorithm maintains a bucket of tokens. Each token
represents permission to send one request. Tokens are added at a
fixed rate...

Best practices include setting appropriate limits based on user tier,
implementing gradual backoff, and monitoring rejection rates...
```

### Execution Flow

```
STEP 1: Load document "rate_limiting.txt" (924 characters)
STEP 2: Load queries (3 queries from queries.json)
STEP 3: Initialize 4 strategies

STEP 4: Begin evaluation loop
Loop Iteration 1: query="What is rate limiting?", strategy="fixed_size"
  - Chunk document into 256-token chunks (4 chunks created)
  - Retrieve top-5 chunks matching query
  - Calculate relevancy: 0.042
  - Calculate faithfulness: 0.583
  - Response time: 5.2ms
  - Store EvaluationResult #1

Loop Iteration 2: query="What is rate limiting?", strategy="structure"
  - Chunk document using paragraph-aware chunking (3 chunks)
  - Retrieve top-5 chunks
  - Calculate relevancy: 0.035
  - Calculate faithfulness: 0.417
  - Response time: 5.1ms
  - Store EvaluationResult #2

Loop Iteration 3: query="What is rate limiting?", strategy="semantic"
  - Chunk document using semantic clustering (2 large chunks)
  - Retrieve top-5 chunks
  - Calculate relevancy: 0.059 (best!)
  - Calculate faithfulness: 0.167 (lowest)
  - Response time: 6.2ms
  - Store EvaluationResult #3

Loop Iteration 4: query="What is rate limiting?", strategy="multigranular"
  - Create 3 granularities of chunks
  - Retrieve top-5 chunks
  - Calculate relevancy: 0.042
  - Calculate faithfulness: 0.583
  - Response time: 5.0ms
  - Store EvaluationResult #4

[Process repeats for 2nd and 3rd queries: iterations 5-8, 9-12]

STEP 5: Aggregate results
  - FixedSize: avg_relevancy = 0.0423, avg_faithfulness = 0.5833
  - Structure: avg_relevancy = 0.0357, avg_faithfulness = 0.4167
  - Semantic: avg_relevancy = 0.0593, avg_faithfulness = 0.1667
  - Multigranular: avg_relevancy = 0.0423, avg_faithfulness = 0.5833

STEP 6: Export to JSON
```

### Key Observations

1. **Semantic Chunking** has best relevancy (0.0593) but lowest faithfulness (0.1667)
   - Reason: Large semantic clusters dilute query terms across 500-2000 tokens

2. **Fixed-Size & Multigranular** have best faithfulness (0.5833)
   - Reason: Smaller chunks keep query terms concentrated

3. **Trade-off Pattern**:
   - Broad context (semantic) = better relevancy, worse faithfulness
   - Compact chunks (fixed) = better faithfulness, weaker relevancy

4. **Response Times** are consistently 5-7ms (excellent in-memory performance)

---

## Summary: Key Logic Points

| Component | Logic | Formula |
|-----------|-------|---------|
| **Retrieval** | Rank chunks by token overlap | Jaccard = \|A∩B\| / \|A∪B\| |
| **Relevancy** | Average Jaccard across chunks | Σ(Jaccard_i) / count |
| **Faithfulness** | Percentage of query terms found | (terms_found) / (terms_total) |
| **Aggregation** | Average metric per strategy | Σ(metric_i) / strategy_count |
| **Evaluation** | One result per query-strategy pair | 12 results = 4 strategies × 3 queries |

---

## Flow Summary

```
User Command
    ↓
Load Docs + Queries + Strategies
    ↓
Triple Loop (Doc × Strategy × Query)
    ↓
For Each Iteration:
  1. Chunk document
  2. Retrieve matching chunks (Jaccard)
  3. Calculate relevancy (avg Jaccard)
  4. Calculate faithfulness (% terms found)
  5. Store EvaluationResult
    ↓
12 Results Collected
    ↓
Aggregate by Strategy
    ↓
Build JSON with Summary + Results Array
    ↓
Export to evaluation_results.json
```
