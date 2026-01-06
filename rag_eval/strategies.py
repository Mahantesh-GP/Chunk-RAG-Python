"""
Chunking strategies following LlamaIndex recommendations.
Implements four strategies:
 - FixedSizeChunker: token-length fixed chunks
 - StructureChunker: uses simple heading/section boundaries
 - SemanticChunker: groups text by semantic similarity (embedding-based)
 - MultigranularChunker: combines multiple granularities/hybrid

These implementations are minimal and intended to be adapted to
your project's loaders and tokenizer (e.g., use tiktoken or LlamaIndex
tokenizers for production).
"""
from typing import List, Tuple, Callable, Optional
import math
import hashlib

try:
    # LlamaIndex token splitter
    from llama_index.text_splitter import TokenTextSplitter
except Exception:
    TokenTextSplitter = None


class MockEmbedder:
    """Simple mock embedder for testing without API calls.
    
    Uses deterministic hashing to produce consistent pseudo-embeddings.
    For production, replace with OpenAI, Cohere, or other real embedders.
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
    
    def _hash_to_embedding(self, text: str) -> List[float]:
        """Convert text to deterministic embedding via hashing."""
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        # Convert bytes to normalized floats in [-1, 1]
        embedding = []
        for i in range(self.embedding_dim):
            byte_idx = i % len(hash_bytes)
            val = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(val)
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return [self._hash_to_embedding(t) for t in texts]
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        return self._hash_to_embedding(text)


class FixedSizeChunker:
    """Fixed-size token chunker. Splits by approximate token counts.

    Note: This implementation uses whitespace tokenization as a lightweight
    approximation. Replace with a real tokenizer for precise token counts.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            i += self.chunk_size - self.chunk_overlap
        return chunks


class StructureChunker:
    """Structure-based chunker: split on headings and paragraphs.

    This simple heuristic splits on blank lines and common markdown headings.
    For robust behavior, use LlamaIndex's SimpleNodeParser or document loaders.
    """

    def __init__(self, min_chunk_size: int = 128, max_chunk_size: int = 512):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        # Split on two or more newlines (paragraphs)
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        buffer = []
        for p in paras:
            buffer.append(p)
            cur = "\n\n".join(buffer)
            if len(cur.split()) >= self.min_chunk_size:
                chunks.append(cur)
                buffer = []
        if buffer:
            chunks.append("\n\n".join(buffer))
        # Ensure chunks aren't too large; further split if necessary
        final_chunks: List[str] = []
        for c in chunks:
            words = c.split()
            if len(words) > self.max_chunk_size:
                # naive split
                fc = FixedSizeChunker(chunk_size=self.max_chunk_size, chunk_overlap=0)
                final_chunks.extend(fc.chunk(c))
            else:
                final_chunks.append(c)
        return final_chunks


class SemanticChunker:
    """Semantic chunker using LlamaIndex token splitter and embedder-based clustering.

    The embedder should provide one of these methods (checked in order):
      - `embed_documents(list[str]) -> List[List[float]]`
      - `embed_texts(list[str]) -> List[List[float]]`
      - `embed(list[str]) -> List[List[float]]`

    If `TokenTextSplitter` from LlamaIndex is not available, falls back to
    paragraph splitting.
    """

    def __init__(
        self,
        embedder: Optional[object] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.75,
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        if TokenTextSplitter is not None:
            self.splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        else:
            self.splitter = None

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Call the embedder with a common embedding API."""
        if self.embedder is None:
            raise ValueError("SemanticChunker requires an embedder to compute embeddings.")

        # Common method names used across libraries
        for name in ("embed_documents", "embed_texts", "embed"):
            func = getattr(self.embedder, name, None)
            if callable(func):
                return func(texts)

        # If embedder is a callable itself
        if callable(self.embedder):
            return self.embedder(texts)

        raise AttributeError("Provided embedder does not expose a compatible embed API")

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def chunk(self, text: str) -> List[str]:
        # 1) Split into candidate passages using token splitter or paragraphs
        if self.splitter is not None:
            try:
                passages = self.splitter.split_text(text)
            except Exception:
                passages = [p.strip() for p in text.split("\n\n") if p.strip()]
        else:
            passages = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not passages:
            return []

        # 2) Compute embeddings for passages
        embeddings = self._embed_texts(passages)

        # 3) Greedy clustering by similarity to cluster centroids
        clusters: List[List[int]] = []
        centroids: List[List[float]] = []

        for idx, vec in enumerate(embeddings):
            best_sim = -1.0
            best_i = None
            for i, c in enumerate(centroids):
                sim = self._cosine(vec, c)
                if sim > best_sim:
                    best_sim = sim
                    best_i = i

            if best_i is None or best_sim < self.similarity_threshold:
                # start new cluster
                clusters.append([idx])
                centroids.append(vec[:])
            else:
                # add to best cluster and update centroid (simple average)
                clusters[best_i].append(idx)
                # update centroid
                old = centroids[best_i]
                n = len(clusters[best_i])
                centroids[best_i] = [(old_j * (n - 1) + v_j) / n for old_j, v_j in zip(old, vec)]

        # 4) Build chunk texts by joining passages in each cluster
        result_chunks: List[str] = []
        for cluster in clusters:
            texts = [passages[i] for i in cluster]
            result_chunks.append("\n\n".join(texts))

        return result_chunks


class MultigranularChunker:
    """Hybrid chunker: index multiple granularities and return combined chunks.

    Strategy:
      - Produce chunks at multiple sizes (small/medium/large)
      - Optionally deduplicate or prioritize overlapping chunks
    """

    def __init__(self, granularities: List[Tuple[int, int]] = None):
        # granularities: list of (chunk_size, chunk_overlap)
        if granularities is None:
            granularities = [(256, 20), (512, 40), (1024, 80)]
        self.granularities = granularities
        self._chunkers = [FixedSizeChunker(s, o) for s, o in self.granularities]

    def chunk(self, text: str) -> List[str]:
        # Produce all granularities and return as combined list
        results = []
        for c in self._chunkers:
            results.extend(c.chunk(text))
        # Simple deduplication while preserving order
        seen = set()
        deduped = []
        for r in results:
            k = r[:200]
            if k not in seen:
                deduped.append(r)
                seen.add(k)
        return deduped
