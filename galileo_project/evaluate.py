"""
=============================================================
  Galileo RAG Evaluation — Chunk Attribution & Utilization
  Works with: Azure OpenAI (primary) + Galileo free tier
  Sign up: https://www.rungalileo.io
=============================================================
"""

import os
from dotenv import load_dotenv
import pandas as pd
import openai

import promptquality as pq
from promptquality import Scorers

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 1. Configure Azure OpenAI
# ─────────────────────────────────────────────────────────────
openai.api_type    = "azure"
openai.api_base    = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key     = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
DEPLOYMENT         = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# ─────────────────────────────────────────────────────────────
# 2. Sample RAG data — same 3 queries as Ragas project
#    so you can compare scores side-by-side
# ─────────────────────────────────────────────────────────────
RAG_SAMPLES = [
    {
        "question": "What is Azure OpenAI Service and what models does it support?",
        "documents": [
            "Azure OpenAI Service provides REST API access to OpenAI's powerful language models including GPT-4, GPT-3.5-Turbo, and Embeddings model series.",
            "The service runs on Azure infrastructure and supports private networking through VNet integration and private endpoints.",
            "Azure OpenAI supports fine-tuning on GPT-3.5-Turbo and Babbage models to customize them for specific use cases.",
            "Microsoft Excel was first released in 1985 for the Apple Macintosh.",  # ← noise chunk
        ],
        "answer": "Azure OpenAI Service is a cloud offering by Microsoft that provides REST API access to OpenAI models such as GPT-4 and GPT-3.5-Turbo. It supports private networking via VNet and allows fine-tuning on select models.",
    },
    {
        "question": "How does RAG help reduce hallucination in LLMs?",
        "documents": [
            "Retrieval-Augmented Generation (RAG) grounds LLM responses in external documents retrieved at query time, reducing hallucination.",
            "By conditioning the model on retrieved context, RAG ensures answers are factually tied to source material rather than model memory.",
            "RAG pipelines typically use a vector database like Azure AI Search to retrieve semantically similar chunks.",
            "Python was created by Guido van Rossum and first released in 1991.",  # ← noise chunk
        ],
        "answer": "RAG reduces hallucination by grounding LLM responses in retrieved documents. Instead of relying on model memory, the LLM generates answers based on retrieved chunks from a vector store like Azure AI Search.",
    },
    {
        "question": "What is EF Core and how does it work with SQL Server?",
        "documents": [
            "Entity Framework Core (EF Core) is a lightweight, extensible, open source ORM for .NET.",
            "EF Core supports SQL Server via the Microsoft.EntityFrameworkCore.SqlServer package.",
            "EF Core uses DbContext to manage entity objects during runtime, which includes querying and saving data.",
            "EF Core migrations allow developers to incrementally update the database schema to keep it in sync with the application model.",
        ],
        "answer": "EF Core is a .NET ORM that enables developers to work with SQL Server using .NET objects. It uses DbContext for managing queries and saving data, and supports migrations to keep the database schema in sync with the application model.",
    },
]


# ─────────────────────────────────────────────────────────────
# 3. Helper — call Azure OpenAI to build the prompt template
#    (Galileo needs the prompt template, not just the final answer)
# ─────────────────────────────────────────────────────────────
def build_prompt(question: str, documents: list[str]) -> str:
    context = "\n\n".join([f"[Chunk {i+1}]: {doc}" for i, doc in enumerate(documents)])
    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""


# ─────────────────────────────────────────────────────────────
# 4. Run Galileo Evaluation
# ─────────────────────────────────────────────────────────────
def run_evaluation():
    print("\n" + "="*60)
    print("  Galileo RAG Evaluation — Chunk Attribution & Utilization")
    print("="*60)

    # ── Login to Galileo ──────────────────────────────────────
    print("\n[1/4] Logging in to Galileo...")
    pq.login(api_key=os.getenv("GALILEO_API_KEY"))
    print("      ✓ Galileo connected")

    # ── Define scorers ────────────────────────────────────────
    # chunk_attribution_utilization_gpt → uses GPT under the hood via Galileo
    # context_adherence                 → is the answer grounded in chunks?
    # completeness                      → did the answer cover all relevant info?
    scorers = [
        Scorers.chunk_attribution_utilization_gpt,  # ← THE KEY METRIC
        Scorers.context_adherence_luna,              # lightweight, no extra cost
        Scorers.completeness_luna,
    ]

    # ── Init Galileo project ──────────────────────────────────
    project_name = os.getenv("GALILEO_PROJECT_NAME", "rag-chunk-evaluation")
    print(f"\n[2/4] Initialising Galileo project: '{project_name}'...")
    pq.init(project_name=project_name, scorers=scorers)
    print("      ✓ Project ready")

    # ── Log each RAG sample ───────────────────────────────────
    print(f"\n[3/4] Logging {len(RAG_SAMPLES)} RAG samples to Galileo...")

    for i, sample in enumerate(RAG_SAMPLES):
        prompt_text = build_prompt(sample["question"], sample["documents"])

        pq.log(
            query=sample["question"],
            prompt=prompt_text,
            documents=sample["documents"],   # ← retrieved chunks
            response=sample["answer"],        # ← LLM generated answer
        )
        print(f"      ✓ Sample {i+1}/{len(RAG_SAMPLES)} logged: {sample['question'][:55]}...")

    # ── Finish and send to Galileo ────────────────────────────
    print("\n[4/4] Sending to Galileo for scoring (takes ~30-60 secs)...")
    run_link = pq.finish()

    # ─── Local summary ────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)

    print(f"""
📊 Metrics computed by Galileo:

   chunk_attribution_utilization  → which chunks contributed to the answer
                                    and what % of retrieved chunks were used
   context_adherence              → was the answer grounded in the chunks?
   completeness                   → did the answer cover all relevant info?

🔗 View full results with per-chunk breakdown in Galileo dashboard:
   {run_link}

📖 Score Guide:
   0.0 - 0.4  → Poor   | chunks not being used / answer hallucinated
   0.4 - 0.7  → Fair   | partial grounding / some irrelevant chunks
   0.7 - 1.0  → Good   | answer well grounded in retrieved context
""")

    return run_link


# ─────────────────────────────────────────────────────────────
# 5. Bonus — run the LIVE pipeline (optional)
#    Uncomment if you want Claude to actually call Azure OpenAI
#    and generate answers on the fly instead of using pre-set answers
# ─────────────────────────────────────────────────────────────
def run_live_pipeline():
    """
    Generates answers from Azure OpenAI in real-time,
    then logs them to Galileo for evaluation.
    """
    print("\n[LIVE MODE] Generating answers from Azure OpenAI...")

    pq.login(api_key=os.getenv("GALILEO_API_KEY"))
    project_name = os.getenv("GALILEO_PROJECT_NAME", "rag-chunk-evaluation-live")

    scorers = [
        Scorers.chunk_attribution_utilization_gpt,
        Scorers.context_adherence_luna,
    ]
    pq.init(project_name=project_name, scorers=scorers)

    for i, sample in enumerate(RAG_SAMPLES):
        prompt_text = build_prompt(sample["question"], sample["documents"])

        # Call Azure OpenAI
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
        )
        generated_answer = response.choices[0].message.content.strip()

        print(f"   Q: {sample['question'][:60]}...")
        print(f"   A: {generated_answer[:80]}...\n")

        pq.log(
            query=sample["question"],
            prompt=prompt_text,
            documents=sample["documents"],
            response=generated_answer,
        )

    run_link = pq.finish()
    print(f"\n✅ Live results: {run_link}")
    return run_link


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"

    if mode == "live":
        # python evaluate.py live
        # → calls Azure OpenAI to generate answers, then evaluates
        run_live_pipeline()
    else:
        # python evaluate.py
        # → uses pre-set sample answers, evaluates immediately
        run_evaluation()
