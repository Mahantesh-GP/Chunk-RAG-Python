"""
=============================================================
  Step 1 — Index Chunks into Azure AI Search
  Reads your chunks JSON → creates embeddings → indexes
  Supports vector + keyword hybrid search
=============================================================
  Run: python index_chunks.py chunks.json
  Or:  python index_chunks.py  (uses sample_chunks.json)
=============================================================
"""

import os
import sys
import json
from dotenv import load_dotenv
from tqdm import tqdm

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)

search_index_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
)

INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "rag-chunks-index")
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")
VECTOR_DIM  = 1536  # text-embedding-ada-002 dimension


# ─────────────────────────────────────────────────────────────
# 1. Create or Update Azure AI Search Index
# ─────────────────────────────────────────────────────────────
def create_index():
    print(f"\n[1/3] Creating Azure AI Search index: '{INDEX_NAME}'...")

    fields = [
        # Required ID field
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),

        # Chunk content — searchable via keyword
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft"
        ),

        # Metadata fields
        SimpleField(name="source",         type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_strategy", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index",    type=SearchFieldDataType.Int32,  filterable=True),

        # Vector field — for semantic/vector search
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIM,
            vector_search_profile_name="hnswProfile",
        ),
    ]

    # Vector search config — HNSW algorithm
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnswAlgo")],
        profiles=[VectorSearchProfile(name="hnswProfile", algorithm_configuration_name="hnswAlgo")],
    )

    # Semantic search config — for reranker scores
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        ),
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    search_index_client.create_or_update_index(index)
    print(f"      ✓ Index '{INDEX_NAME}' ready")


# ─────────────────────────────────────────────────────────────
# 2. Generate Embedding for a chunk
# ─────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model=EMBED_MODEL,
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────────────────────
# 3. Index chunks into Azure AI Search
# ─────────────────────────────────────────────────────────────
def index_chunks(chunks: list[dict]):
    print(f"\n[2/3] Generating embeddings and indexing {len(chunks)} chunks...")

    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
    )

    documents = []
    for chunk in tqdm(chunks, desc="      Embedding chunks"):
        doc = {
            "id":              chunk["id"],
            "content":         chunk["content"],
            "source":          chunk.get("source", ""),
            "chunk_strategy":  chunk.get("chunk_strategy", ""),
            "chunk_index":     chunk.get("chunk_index", 0),
            "content_vector":  get_embedding(chunk["content"]),
        }
        documents.append(doc)

    # Upload in batches of 100
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = search_client.upload_documents(documents=batch)
        succeeded = sum(1 for r in result if r.succeeded)
        print(f"      ✓ Batch {i//batch_size + 1}: {succeeded}/{len(batch)} chunks indexed")

    print(f"\n[3/3] Indexing complete!")
    print(f"      ✓ {len(documents)} chunks indexed to '{INDEX_NAME}'")
    print(f"      ✓ Ready for hybrid search")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunks_file = sys.argv[1] if len(sys.argv) > 1 else "sample_chunks.json"

    print("="*60)
    print("  Azure AI Search — Chunk Indexer")
    print("="*60)
    print(f"\nLoading chunks from: {chunks_file}")

    with open(chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"✓ Loaded {len(chunks)} chunks")

    create_index()
    index_chunks(chunks)

    print("\n✅ Done! Now run: python evaluate.py questions.json")
