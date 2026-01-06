# RAG Chunking Strategy Evaluation - Complete Guide

## Table of Contents
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Chunking Strategies Explained](#chunking-strategies-explained)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Run](#how-to-run)
- [Customization](#customization)
- [Understanding Results](#understanding-results)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### Run Evaluation (Already Set Up)
```powershell
# 1. Run the evaluation
.venv\Scripts\python -m rag_eval.cli --docs docs --queries queries.json --output evaluation_results.json

# 2. View the summary
.venv\Scripts\python show_results.py
```

### Fresh Installation
```powershell
# 1. Navigate to project
cd C:\Learnings\Chunk-RAG-Python

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Run tests
pytest tests/test_basic.py -v

# 5. Run evaluation
python -m rag_eval.cli
```

---

## System Architecture

### Overall Flow
```
User Query → Chunker → Chunks → Retriever → Evaluation Metrics → Results
```

### Components

```
rag_eval/
├── strategies.py       # 4 chunking strategies + MockEmbedder
├── evaluator.py        # RAGEvaluator with 3 metrics
├── cli.py             # Command-line orchestration
└── __init__.py        # Package exports

tests/
└── test_basic.py      # Unit tests for chunkers

docs/
└── sample.md          # Sample document for evaluation

queries.json           # Evaluation queries
evaluation_results.json # Output results
show_results.py        # Results visualization
```

### Data Flow

1. **Load Documents** → Read all `.md` and `.txt` files from `docs/`
2. **Load Queries** → Parse `queries.json`
3. **Initialize Strategies** → Create 4 different chunkers
4. **Chunk Documents** → Each strategy processes documents
5. **Retrieve** → For each query, find top-K relevant chunks
6. **Evaluate** → Compute 3 metrics per query
7. **Aggregate** → Calculate per-strategy averages
8. **Export** → Save to JSON

---

## Chunking Strategies Explained

### 1. FixedSizeChunker (Simplest)

**Configuration:**
```python
FixedSizeChunker(chunk_size=256, chunk_overlap=20)
```

**Algorithm:**
1. Split text by whitespace into words
2. Group words into chunks of ~256 tokens
3. Overlap chunks by 20 tokens to preserve context

**Example:**
```
Input: "Machine learning is AI. Neural networks learn patterns. Deep learning uses layers..."

Output:
Chunk 1: [words 0-256]
Chunk 2: [words 236-492]  ← overlaps with Chunk 1 by 20 words
Chunk 3: [words 472-728]
```

**Pros:**
- ✅ Fast and predictable
- ✅ Consistent chunk sizes
- ✅ No dependencies

**Cons:**
- ❌ Ignores semantic boundaries
- ❌ May split mid-sentence
- ❌ No awareness of document structure

**Best For:** Simple documents, consistent processing needs

---

### 2. StructureChunker (Document-Aware)

**Configuration:**
```python
StructureChunker(min_chunk_size=128, max_chunk_size=512)
```

**Algorithm:**
1. Split text by double newlines (paragraphs)
2. Merge small paragraphs until reaching `min_chunk_size`
3. Split large paragraphs if they exceed `max_chunk_size`
4. Respect heading boundaries

**Example:**
```
Input:
## Heading 1
Short paragraph (50 words)
Another short paragraph (100 words)

## Heading 2
Very long paragraph (600 words)

Output:
Chunk 1: [Heading 1 + merged paragraphs = 150 words]
Chunk 2: [Heading 2 + first 300 words]
Chunk 3: [Heading 2 + last 300 words]
```

**Pros:**
- ✅ Respects document structure
- ✅ Preserves paragraph boundaries
- ✅ Maintains heading context

**Cons:**
- ❌ Variable chunk sizes
- ❌ May create very small/large chunks
- ❌ Assumes well-structured documents

**Best For:** Markdown/HTML documents, technical documentation

---

### 3. SemanticChunker (AI-Powered) ⭐

**Configuration:**
```python
SemanticChunker(
    embedder=MockEmbedder(),
    chunk_size=512,
    similarity_threshold=0.75
)
```

**Algorithm:**

**Step 1: Initial Token-Based Splitting**
```python
# Uses LlamaIndex TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
passages = splitter.split_text(text)
```

**Step 2: Compute Embeddings**
```python
embeddings = embedder.embed_documents(passages)
# Each passage → 128-dimensional vector
# Example: [0.23, -0.15, 0.87, ..., 0.45]
```

**Step 3: Greedy Semantic Clustering**
```python
clusters = []
centroids = []  # Average embedding of each cluster

for passage, embedding in zip(passages, embeddings):
    # Calculate cosine similarity to all centroids
    similarities = [cosine(embedding, c) for c in centroids]
    
    # Find best match above threshold
    best_idx = argmax(similarities)
    best_sim = similarities[best_idx]
    
    if best_sim >= similarity_threshold (0.75):
        # Add to existing cluster
        clusters[best_idx].append(passage)
        # Update centroid (running average)
        centroids[best_idx] = update_centroid(...)
    else:
        # Create new cluster
        clusters.append([passage])
        centroids.append(embedding)

# Final chunks = joined cluster texts
return ["\n\n".join(cluster) for cluster in clusters]
```

**Cosine Similarity Formula:**
```python
def cosine(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    return dot_product / (norm_a * norm_b)
    # Returns value between -1 (opposite) and 1 (identical)
```

**Example:**
```
Input Passages:
1. "Machine learning uses algorithms to learn from data..."
2. "Neural networks are a type of ML model with layers..."
3. "Healthcare applications include disease diagnosis..."
4. "Medical AI can predict patient outcomes accurately..."

Embeddings:
Passage 1 → [0.8, 0.6, 0.1, ...] (ML topic)
Passage 2 → [0.7, 0.5, 0.2, ...] (ML topic, similar to 1)
Passage 3 → [0.1, 0.2, 0.9, ...] (Healthcare topic)
Passage 4 → [0.2, 0.1, 0.8, ...] (Healthcare topic, similar to 3)

Clustering:
Cluster 1: [Passage 1, Passage 2]  ← ML-related
Cluster 2: [Passage 3, Passage 4]  ← Healthcare-related

Output:
Chunk 1: "Machine learning uses algorithms... Neural networks are..."
Chunk 2: "Healthcare applications include... Medical AI can..."
```

**Pros:**
- ✅ Groups semantically related content
- ✅ Maintains topical coherence
- ✅ Better retrieval relevancy
- ✅ Language-agnostic (works with any embedder)

**Cons:**
- ❌ Requires embeddings (API costs)
- ❌ Slower processing
- ❌ Variable chunk sizes
- ❌ Quality depends on embedder

**Best For:** Complex documents, multi-topic content, production RAG systems

---

### 4. MultigranularChunker (Hybrid)

**Configuration:**
```python
MultigranularChunker(granularities=[256, 512, 1024])
```

**Algorithm:**
1. Create three FixedSizeChunkers with different sizes
2. Chunk the text three times
3. Deduplicate exact copies
4. Return combined list

**Example:**
```
Input: 2000-word document

Processing:
Fine-grained (256):   [C1, C2, C3, C4, C5, C6, C7, C8]
Medium-grained (512): [C9, C10, C11, C12]
Coarse-grained (1024): [C13, C14]

Output: [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14]
Total: 14 chunks at 3 different granularities
```

**Pros:**
- ✅ Captures multiple levels of detail
- ✅ Flexible retrieval (fine or coarse)
- ✅ No additional dependencies

**Cons:**
- ❌ More chunks = slower retrieval
- ❌ Higher storage requirements
- ❌ Redundant information

**Best For:** Hierarchical documents, when granularity needs vary by query

---

## Evaluation Metrics

### 1. Response Time (Latency)

**What it measures:** How fast the system retrieves chunks

**Implementation:**
```python
start = time.time()
retrieved_chunks = retrieve_top_k(query, chunks, top_k=5)
response_time = time.time() - start
```

**Typical Values:**
- In-memory (current): ~0.0001s
- Vector DB (Pinecone, Weaviate): ~0.010-0.050s
- Azure Cognitive Search: ~0.020-0.100s

**Why it matters:** User experience in production RAG systems

---

### 2. Relevancy (Jaccard Similarity)

**What it measures:** Lexical overlap between query and retrieved chunks

**Formula:**
```python
query_terms = set(query.lower().split())
chunk_terms = set(retrieved_chunks_combined.lower().split())

intersection = query_terms & chunk_terms
union = query_terms | chunk_terms

relevancy = len(intersection) / len(union)
```

**Example:**
```
Query: "How do neural networks work?"
Query Terms: {"how", "do", "neural", "networks", "work"}

Retrieved Chunk: "Neural networks use layers to learn patterns..."
Chunk Terms: {"neural", "networks", "use", "layers", "to", "learn", "patterns"}

Intersection: {"neural", "networks"}  (2 terms)
Union: {"how", "do", "neural", "networks", "work", "use", "layers", "to", "learn", "patterns"}  (10 terms)

Relevancy = 2 / 10 = 0.20 (20%)
```

**Range:** 0.0 (no overlap) to 1.0 (perfect match)

**Why it matters:** Indicates if retrieval finds topically relevant content

---

### 3. Faithfulness (Query Term Coverage)

**What it measures:** Percentage of query terms present in retrieved chunks

**Formula:**
```python
query_terms = set(query.lower().split())
chunk_text = " ".join(retrieved_chunks).lower()

present_terms = {term for term in query_terms if term in chunk_text}
faithfulness = len(present_terms) / len(query_terms)
```

**Example:**
```
Query: "What are the applications of machine learning in healthcare?"
Query Terms: {"what", "are", "the", "applications", "of", "machine", "learning", "in", "healthcare"}

Retrieved Text: "Machine learning applications include healthcare diagnosis..."
Present Terms: {"machine", "learning", "applications", "healthcare"}  (4 terms)
Missing Terms: {"what", "are", "the", "of", "in"}  (5 terms)

Faithfulness = 4 / 9 = 0.44 (44%)
```

**Range:** 0.0 (no query terms found) to 1.0 (all query terms present)

**Why it matters:** Ensures retrieved content actually addresses the query

---

## How to Run

### Basic Usage

```powershell
# Run with default settings (docs/, queries.json, evaluation_results.json)
python -m rag_eval.cli

# View results
python show_results.py
```

### Custom Configuration

```powershell
# Specify custom paths
python -m rag_eval.cli \
  --docs my_documents \
  --queries custom_queries.json \
  --output my_results.json
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--docs` | `docs` | Directory containing `.md` or `.txt` files |
| `--queries` | `queries.json` | JSON file with query list |
| `--output` | `evaluation_results.json` | Output file path |

---

## Customization

### Add More Documents

```powershell
# Create new document
echo "Your content here" > docs/new_doc.md

# Or copy existing files
cp path/to/your/document.txt docs/
```

**Supported formats:** `.md`, `.txt`

---

### Modify Queries

Edit `queries.json`:
```json
{
  "queries": [
    "What is machine learning?",
    "How do neural networks work?",
    "What are the applications of AI?",
    "Your new question here"
  ]
}
```

**Alternative format (also supported):**
```json
[
  "Question 1",
  "Question 2",
  "Question 3"
]
```

---

### Change Strategy Parameters

Edit `rag_eval/cli.py`:

```python
strategies = {
    "fixed": FixedSizeChunker(
        chunk_size=512,      # Change from 256 to 512
        chunk_overlap=50     # Change from 20 to 50
    ),
    "structure": StructureChunker(
        min_chunk_size=256,  # Change from 128
        max_chunk_size=1024  # Change from 512
    ),
    "semantic": SemanticChunker(
        embedder=embedder,
        chunk_size=1024,     # Change from 512
        similarity_threshold=0.80  # Change from 0.75 (stricter)
    ),
    "multigranular": MultigranularChunker()
}
```

---

### Use Real OpenAI Embeddings

Replace MockEmbedder with real embeddings:

```python
# Install OpenAI
pip install openai

# Update cli.py
from openai import OpenAI

class OpenAIEmbedder:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]

# Use in strategies
embedder = OpenAIEmbedder(api_key="sk-...")
strategies = {
    "semantic": SemanticChunker(embedder=embedder, ...)
}
```

---

### Add New Metrics

Edit `rag_eval/evaluator.py`:

```python
def _my_custom_metric(self, query: str, chunks: List[str]) -> float:
    """Your custom metric implementation."""
    # Example: Average chunk length
    total_length = sum(len(chunk) for chunk in chunks)
    return total_length / len(chunks) if chunks else 0.0

def evaluate(self, query: str, strategy_name: str, chunks: List[str]):
    # ... existing code ...
    
    # Add your metric
    custom_score = self._my_custom_metric(query, retrieved_chunks)
    
    result = EvaluationResult(
        query=query,
        strategy=strategy_name,
        response_time=response_time,
        relevancy=relevancy,
        faithfulness=faithfulness,
        custom_metric=custom_score,  # Add here
        chunks=retrieved_chunks
    )
```

---

## Understanding Results

### Output Structure

`evaluation_results.json` contains:

```json
{
  "summary": {
    "strategy_name": {
      "avg_response_time": 0.0001,
      "avg_relevancy": 0.1234,
      "avg_faithfulness": 0.5678,
      "count": 3
    }
  },
  "results": [
    {
      "query": "What is machine learning?",
      "strategy": "fixed",
      "response_time": 0.0,
      "relevancy": 0.0150,
      "faithfulness": 0.6667,
      "retrieved_chunks": ["...", "..."]
    }
  ]
}
```

---

### Interpreting Results

**Example Output:**
```
fixed:           Relevancy 0.0174 | Faithfulness 0.6944
structure:       Relevancy 0.0209 | Faithfulness 0.4630
semantic:        Relevancy 0.1451 ⭐ | Faithfulness 0.3556
multigranular:   Relevancy 0.0174 | Faithfulness 0.6944 ⭐
```

**Analysis:**

1. **Semantic has best Relevancy (0.1451)**
   - Groups semantically similar content
   - Retrieved chunks are topically coherent
   - Better lexical overlap with queries

2. **Fixed/Multigranular have best Faithfulness (0.6944)**
   - Smaller chunks = more focused content
   - Each chunk covers fewer topics
   - Higher density of query terms

3. **Trade-offs:**
   - Semantic: Better for broad questions ("Tell me about AI")
   - Fixed: Better for specific fact-finding ("What is the definition of X?")
   - Structure: Best for navigating well-organized documents
   - Multigranular: Best when query complexity varies

---

### When to Choose Each Strategy

| Strategy | Use When | Avoid When |
|----------|----------|------------|
| **Fixed** | - Simple documents<br>- Predictable processing<br>- Consistent chunk sizes needed | - Documents have important structure<br>- Semantic coherence matters |
| **Structure** | - Markdown/HTML documents<br>- Heading-based navigation<br>- Preserving document hierarchy | - Unstructured text<br>- Paragraphs vary wildly in size |
| **Semantic** | - Multi-topic documents<br>- Production RAG systems<br>- Query relevancy critical | - API costs prohibitive<br>- Processing speed critical<br>- Simple fact retrieval |
| **Multigranular** | - Mixed query types<br>- Hierarchical content<br>- Need flexibility | - Storage limited<br>- Simple use case<br>- Speed is critical |

---

## Advanced Usage

### Run Tests

```powershell
# All tests
pytest tests/test_basic.py -v

# Specific test
pytest tests/test_basic.py::test_semantic_chunker -v

# With coverage
pytest tests/test_basic.py --cov=rag_eval
```

---

### Programmatic Usage

```python
from rag_eval import FixedSizeChunker, RAGEvaluator

# Create chunker
chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=20)

# Chunk document
text = "Your long document here..."
chunks = chunker.chunk(text)

# Create evaluator
evaluator = RAGEvaluator(top_k=5)

# Evaluate queries
queries = ["Question 1", "Question 2"]
for query in queries:
    result = evaluator.evaluate(query, "fixed", chunks)
    print(f"Relevancy: {result.relevancy:.4f}")

# Get summary
summary = evaluator.summary()
print(summary)
```

---

### Integration with Azure Cognitive Search

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Create search client
client = SearchClient(
    endpoint="https://your-service.search.windows.net",
    index_name="documents",
    credential=AzureKeyCredential("your-api-key")
)

# Upload chunks with embeddings
for i, chunk in enumerate(chunks):
    doc = {
        "id": str(i),
        "content": chunk,
        "embedding": embedder.embed_text(chunk)
    }
    client.upload_documents([doc])

# Query with vector search
results = client.search(
    search_text="What is machine learning?",
    vector=embedder.embed_text("What is machine learning?"),
    top=5
)
```

---

### MockEmbedder Explained

**Why use it?**
- ✅ No API keys required
- ✅ Deterministic (same text → same embedding)
- ✅ No costs
- ✅ Fast for testing

**How it works:**
```python
import hashlib

def _hash_to_embedding(text: str) -> List[float]:
    # Create deterministic hash
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()  # 32 bytes
    
    # Convert to 128-dimensional vector
    embedding = []
    for i in range(128):
        byte_idx = i % 32  # Cycle through hash bytes
        val = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Scale to [-1, 1]
        embedding.append(val)
    
    return embedding
```

**Limitations:**
- ❌ NOT semantically meaningful
- ❌ "cat" and "dog" have random similarity
- ❌ Only for testing/demos

**For production:** Use OpenAI, Cohere, or HuggingFace embeddings

---

## Troubleshooting

### Import Errors

```powershell
# Reinstall package
pip install -e .

# Verify installation
python -c "import rag_eval; print(rag_eval.__file__)"
```

---

### No Documents Found

```powershell
# Check directory
ls docs/

# Ensure files are .md or .txt
# Add documents if empty
echo "Sample content" > docs/test.md
```

---

### MockEmbedder Required Error

```python
# Ensure embedder is passed to SemanticChunker
embedder = MockEmbedder(embedding_dim=128)
chunker = SemanticChunker(embedder=embedder)  # Don't forget this!
```

---

### Test Failures

```powershell
# Update LlamaIndex
pip install --upgrade llama-index

# Check Python version
python --version  # Should be 3.12+

# Clear cache
pytest --cache-clear tests/
```

---

## Next Steps

### 1. **Add Real Embeddings**
Replace MockEmbedder with OpenAI or Cohere for production-quality semantic chunking.

### 2. **Integrate Vector Database**
Connect to Pinecone, Weaviate, or Azure Cognitive Search for scalable retrieval.

### 3. **Add More Strategies**
Implement recursive chunking, topic-based chunking, or custom strategies.

### 4. **Enhance Metrics**
Add BLEU score, ROUGE score, or semantic similarity metrics.

### 5. **Build UI**
Create a Streamlit or Gradio interface for interactive evaluation.

### 6. **Scale to Production**
Deploy as REST API with FastAPI, add caching, and optimize for performance.

---

## Resources

- **LlamaIndex Documentation:** https://docs.llamaindex.ai/
- **Azure Cognitive Search:** https://learn.microsoft.com/azure/search/
- **OpenAI Embeddings:** https://platform.openai.com/docs/guides/embeddings
- **RAG Best Practices:** https://www.pinecone.io/learn/retrieval-augmented-generation/

---

## License

MIT License - See LICENSE file for details

---

## Support

For issues or questions:
1. Check this guide first
2. Run tests to verify setup: `pytest tests/test_basic.py -v`
3. Check `evaluation_results.json` for detailed output
4. Review error messages carefully

---

**Happy Evaluating! 🚀**
