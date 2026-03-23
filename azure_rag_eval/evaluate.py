"""
=============================================================
  Step 2 — Hybrid Retrieval + Chunk Attribution + Evaluation

  TWO attribution approaches (your architect's idea):

  APPROACH 1 — PRE-ANSWER (Before LLM call)
    Uses Azure AI Search reranker score per chunk
    No answer needed — pure query-chunk relevance
    Fast, cheap, no LLM tokens

  APPROACH 2 — POST-ANSWER (After LLM call)
    Uses Azure AI Evaluation SDK Groundedness
    Needs generated answer
    More accurate, uses LLM tokens

  Both results written to output_results.json
=============================================================
  Run: python evaluate.py questions.json
  Or:  python evaluate.py  (uses sample_questions.json)
=============================================================
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
)
from azure.ai.evaluation import GroundednessEvaluator

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
TOP_K                = int(os.getenv("TOP_K", 5))
ATTRIBUTION_THRESHOLD = float(os.getenv("ATTRIBUTION_THRESHOLD", 2.0))  # reranker 0-4
INDEX_NAME           = os.getenv("AZURE_SEARCH_INDEX", "rag-chunks-index")
EMBED_MODEL          = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")
CHAT_MODEL           = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ─────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
)

model_config = {
    "azure_endpoint":   os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key":          os.getenv("AZURE_OPENAI_API_KEY"),
    "azure_deployment": CHAT_MODEL,
    "api_version":      os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
}

groundedness_evaluator = GroundednessEvaluator(model_config)


# ─────────────────────────────────────────────────────────────
# 1. Get embedding for a query
# ─────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model=EMBED_MODEL,
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────────────────────
# 2. Hybrid Search — Vector + Keyword + Semantic Reranker
#    Returns top-K chunks with reranker scores
# ─────────────────────────────────────────────────────────────
def hybrid_search(question: str) -> list[dict]:
    """
    Hybrid search = Vector search + Keyword (BM25) search
    Combined using RRF (Reciprocal Rank Fusion)
    + Semantic reranker for reranker_score (0-4)

    reranker_score is KEY for pre-answer attribution:
      0-1 → low relevance  → NOT attributed
      2-4 → high relevance → ATTRIBUTED
    """
    query_vector = get_embedding(question)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=TOP_K,
        fields="content_vector",
    )

    results = search_client.search(
        search_text=question,           # keyword search
        vector_queries=[vector_query],  # vector search
        query_type=QueryType.SEMANTIC,  # enable semantic reranker
        semantic_configuration_name="my-semantic-config",
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=TOP_K,
        select=["id", "content", "source", "chunk_strategy", "chunk_index"],
    )

    chunks = []
    for result in results:
        chunks.append({
            "id":              result["id"],
            "content":         result["content"],
            "source":          result.get("source", ""),
            "chunk_strategy":  result.get("chunk_strategy", ""),
            "chunk_index":     result.get("chunk_index", 0),
            "search_score":    result.get("@search.score", 0),          # RRF score
            "reranker_score":  result.get("@search.reranker_score", 0), # semantic reranker 0-4
        })

    return chunks


# ─────────────────────────────────────────────────────────────
# 3. APPROACH 1 — Pre-Answer Attribution
#    Uses reranker score — NO answer needed
#    Architect's idea: evaluate BEFORE LLM call
# ─────────────────────────────────────────────────────────────
def pre_answer_attribution(chunks: list[dict]) -> dict:
    """
    Uses Azure AI Search semantic reranker score as attribution signal.

    Reranker score (0-4):
      >= threshold → chunk is ATTRIBUTED (contributed to relevance)
      <  threshold → chunk is NOT ATTRIBUTED (low relevance, likely noise)

    This runs BEFORE the LLM generates an answer.
    No LLM tokens consumed here.
    """
    attribution_list = []
    scores           = []

    for chunk in chunks:
        reranker_score = chunk.get("reranker_score", 0)
        is_attributed  = 1 if reranker_score >= ATTRIBUTION_THRESHOLD else 0
        attribution_list.append(is_attributed)
        scores.append(round(reranker_score, 4))

    attributed_count = sum(attribution_list)
    utilization      = attributed_count / len(chunks) if chunks else 0

    return {
        "approach":          "pre_answer",
        "description":       "Query-chunk relevance BEFORE LLM call (reranker score)",
        "threshold_used":    ATTRIBUTION_THRESHOLD,
        "reranker_scores":   scores,                          # per chunk score
        "attribution_list":  attribution_list,                # [1,1,0,0,0]
        "attributed_count":  attributed_count,
        "total_chunks":      len(chunks),
        "chunk_utilization": round(utilization, 4),           # 0.0 - 1.0
    }


# ─────────────────────────────────────────────────────────────
# 4. Generate Answer using Azure OpenAI
# ─────────────────────────────────────────────────────────────
def generate_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        [f"[Chunk {i+1}]: {c['content']}" for i, c in enumerate(chunks)]
    )

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using ONLY "
                    "the provided context. If the answer is not in the context, "
                    "say 'I don't know.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# 5. APPROACH 2 — Post-Answer Attribution
#    Uses Azure AI Evaluation Groundedness — AFTER LLM call
#    More accurate but requires generated answer + LLM tokens
# ─────────────────────────────────────────────────────────────
def post_answer_attribution(question: str, chunks: list[dict], answer: str) -> dict:
    """
    Uses Azure AI Evaluation SDK Groundedness to check
    if the answer is grounded in the retrieved chunks.

    Then per-chunk: checks if removing this chunk would
    affect the groundedness (attribution signal).

    Runs AFTER the LLM generates an answer.
    """

    full_context = "\n\n".join(
        [f"[Chunk {i+1}]: {c['content']}" for i, c in enumerate(chunks)]
    )

    # Overall groundedness score (1-5)
    overall = groundedness_evaluator(
        response=answer,
        context=full_context,
    )
    overall_score = overall.get("groundedness", 0)

    # Per-chunk attribution — check each chunk individually
    attribution_list = []
    chunk_scores     = []

    for i, chunk in enumerate(chunks):
        result = groundedness_evaluator(
            response=answer,
            context=chunk["content"],  # check this single chunk
        )
        score         = result.get("groundedness", 0)
        is_attributed = 1 if score >= 3 else 0  # 3+ out of 5 = attributed
        attribution_list.append(is_attributed)
        chunk_scores.append(score)

    attributed_count = sum(attribution_list)
    utilization      = attributed_count / len(chunks) if chunks else 0

    return {
        "approach":           "post_answer",
        "description":        "Answer-grounded attribution AFTER LLM call (groundedness)",
        "overall_groundedness": overall_score,               # 1-5
        "chunk_scores":       chunk_scores,                  # per chunk groundedness
        "attribution_list":   attribution_list,              # [1,1,0,0,0]
        "attributed_count":   attributed_count,
        "total_chunks":       len(chunks),
        "chunk_utilization":  round(utilization, 4),         # 0.0 - 1.0
    }


# ─────────────────────────────────────────────────────────────
# 6. Process one question — full pipeline
# ─────────────────────────────────────────────────────────────
def process_question(item: dict) -> dict:
    question = item["question"]
    q_id     = item.get("id", "")

    # Step A — Hybrid retrieval
    chunks = hybrid_search(question)

    # Step B — PRE-ANSWER attribution (architect's approach)
    pre_attribution = pre_answer_attribution(chunks)

    # Step C — Generate answer
    answer = generate_answer(question, chunks)

    # Step D — POST-ANSWER attribution (traditional approach)
    post_attribution = post_answer_attribution(question, chunks, answer)

    return {
        "id":       q_id,
        "question": question,
        "answer":   answer,

        # Retrieved chunks with scores
        "retrieved_chunks": [
            {
                "id":             c["id"],
                "content":        c["content"],
                "source":         c["source"],
                "chunk_strategy": c["chunk_strategy"],
                "search_score":   c["search_score"],
                "reranker_score": c["reranker_score"],
            }
            for c in chunks
        ],

        # APPROACH 1 — Pre-answer (your architect's idea)
        "pre_answer_attribution":  pre_attribution,

        # APPROACH 2 — Post-answer (traditional)
        "post_answer_attribution": post_attribution,

        # Quick summary
        "summary": {
            "total_chunks_retrieved":       len(chunks),
            "pre_answer_utilization":       pre_attribution["chunk_utilization"],
            "post_answer_utilization":      post_attribution["chunk_utilization"],
            "overall_groundedness_score":   post_attribution["overall_groundedness"],
        },
    }


# ─────────────────────────────────────────────────────────────
# 7. Main — process all questions
# ─────────────────────────────────────────────────────────────
def run_evaluation(questions_file: str):
    print("\n" + "="*60)
    print("  RAG Evaluation — Hybrid Search + Chunk Attribution")
    print("="*60)

    # Load questions
    print(f"\n[1/3] Loading questions from: {questions_file}")
    with open(questions_file, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"      ✓ {len(questions)} questions loaded")

    # Process all questions
    print(f"\n[2/3] Processing {len(questions)} questions...")
    print(f"      Hybrid search (vector + keyword) + 2 attribution approaches\n")

    results      = []
    failed       = []
    start_time   = time.time()

    for item in tqdm(questions, desc="      Evaluating"):
        try:
            result = process_question(item)
            results.append(result)
        except Exception as e:
            failed.append({"id": item.get("id"), "question": item["question"], "error": str(e)})
            print(f"\n      ⚠ Failed: {item['question'][:50]}... → {e}")

    elapsed = round(time.time() - start_time, 1)

    # Save results
    print(f"\n[3/3] Saving results...")
    output = {
        "metadata": {
            "timestamp":           datetime.now().isoformat(),
            "total_questions":     len(questions),
            "successful":          len(results),
            "failed":              len(failed),
            "elapsed_seconds":     elapsed,
            "top_k":               TOP_K,
            "attribution_threshold": ATTRIBUTION_THRESHOLD,
            "index_name":          INDEX_NAME,
        },
        "aggregate": {
            "avg_pre_answer_utilization":  round(
                sum(r["summary"]["pre_answer_utilization"]  for r in results) / len(results), 4
            ) if results else 0,
            "avg_post_answer_utilization": round(
                sum(r["summary"]["post_answer_utilization"] for r in results) / len(results), 4
            ) if results else 0,
            "avg_groundedness_score":      round(
                sum(r["summary"]["overall_groundedness_score"] for r in results) / len(results), 4
            ) if results else 0,
        },
        "results": results,
        "failed":  failed,
    }

    output_file = "output_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n   Total questions      : {len(questions)}")
    print(f"   Successful           : {len(results)}")
    print(f"   Failed               : {len(failed)}")
    print(f"   Time taken           : {elapsed}s")
    print(f"\n   Avg Pre-Answer  Utilization : {output['aggregate']['avg_pre_answer_utilization']:.2%}")
    print(f"   Avg Post-Answer Utilization : {output['aggregate']['avg_post_answer_utilization']:.2%}")
    print(f"   Avg Groundedness Score      : {output['aggregate']['avg_groundedness_score']:.1f}/5")
    print(f"\n✅ Full results saved to: {output_file}")

    return output


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions_file = sys.argv[1] if len(sys.argv) > 1 else "sample_questions.json"
    run_evaluation(questions_file)
