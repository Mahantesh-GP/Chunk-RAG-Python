"""
Evaluator module: runs queries against indexes (or local chunks) and
computes Response Time, Relevancy, and Faithfulness metrics.

This is a lightweight evaluator focused on chunk-level evaluation so you
can compare chunking strategies without committing to a specific search
service. It also includes hooks to integrate Azure Cognitive Search or
another vector DB.
"""
import time
from typing import List, Dict


class EvaluationResult:
    def __init__(self, query: str, strategy: str, response_time: float, relevancy: float, faithfulness: float, chunks: List[str]):
        self.query = query
        self.strategy = strategy
        self.response_time = response_time
        self.relevancy = relevancy
        self.faithfulness = faithfulness
        self.chunks = chunks


class RAGEvaluator:
    """Evaluate chunking strategies.

    For quick experiments, the evaluator performs a naive retrieval based on
    token overlap between query and chunks. For production, replace the
    retrieval with a vector search (Azure Cognitive Search, Milvus, etc.).
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.results: List[EvaluationResult] = []

    def _retrieve(self, query: str, chunks: List[str]) -> List[str]:
        # Naive retrieval: rank by token overlap
        qtokens = set(query.lower().split())
        scored = []
        for c in chunks:
            ctokens = set(c.lower().split())
            if not ctokens:
                score = 0.0
            else:
                score = len(qtokens & ctokens) / len(qtokens | ctokens)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[: self.top_k]]

    def _relevancy(self, query: str, retrieved: List[str]) -> float:
        if not retrieved:
            return 0.0
        qtokens = set(query.lower().split())
        scores = []
        for c in retrieved:
            ctokens = set(c.lower().split())
            denom = len(qtokens | ctokens) or 1
            scores.append(len(qtokens & ctokens) / denom)
        return sum(scores) / len(scores)

    def _faithfulness(self, query: str, retrieved: List[str]) -> float:
        if not retrieved:
            return 0.0
        qterms = [t for t in query.lower().split() if len(t) > 3]
        if not qterms:
            return 0.5
        scores = []
        for c in retrieved:
            text = c.lower()
            matched = sum(1 for t in qterms if t in text)
            scores.append(matched / len(qterms))
        return sum(scores) / len(scores)

    def evaluate(self, query: str, strategy_name: str, chunks: List[str]) -> EvaluationResult:
        start = time.time()
        retrieved = self._retrieve(query, chunks)
        elapsed = time.time() - start
        relevancy = self._relevancy(query, retrieved)
        faithfulness = self._faithfulness(query, retrieved)
        result = EvaluationResult(query, strategy_name, elapsed, relevancy, faithfulness, retrieved)
        self.results.append(result)
        return result

    def summary(self) -> Dict[str, Dict[str, float]]:
        from collections import defaultdict

        stats = defaultdict(lambda: {"avg_response_time": 0.0, "avg_relevancy": 0.0, "avg_faithfulness": 0.0, "count": 0})
        for r in self.results:
            s = stats[r.strategy]
            s["avg_response_time"] += r.response_time
            s["avg_relevancy"] += r.relevancy
            s["avg_faithfulness"] += r.faithfulness
            s["count"] += 1
        for k, v in stats.items():
            c = v["count"] or 1
            v["avg_response_time"] /= c
            v["avg_relevancy"] /= c
            v["avg_faithfulness"] /= c
        return dict(stats)
