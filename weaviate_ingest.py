"""
weaviate_ingest.py
─────────────────────────────────────────────────────────────────────────────
Ingest chunks vào Weaviate với local embeddings (BAAI/bge-base-en-v1.5).
Vectors được generate bằng sentence-transformers, push thẳng vào Weaviate.

Prerequisites:
    pip install -r requirements.txt

    # Apple Silicon — MPS acceleration (tự động detect)
    # Không cần setup gì thêm

Setup Weaviate (local Docker):
    docker compose up -d
    docker compose ps     # verify running
    docker compose down   # stop

Usage:
    # Full pipeline: build chunks → embed → ingest
    python weaviate_ingest.py \
        --kb    paula_choice_cleaned.json \
        --url   http://localhost:8080 \
        --reset

    # Chỉ ingest từ chunks đã build sẵn
    python weaviate_ingest.py \
        --chunks chunks.json \
        --url    http://localhost:8080

    # Với test query sau ingest
    python weaviate_ingest.py \
        --kb    paula_choice_cleaned.json \
        --reset --stats \
        --test-query "humectant safe for sensitive skin"
"""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    import weaviate
    import weaviate.classes as wvc
    from weaviate.classes.config import (
        Configure, Property, DataType,
        Tokenization, VectorDistances,
    )
    from weaviate.classes.query import Filter, MetadataQuery, HybridFusion
    from weaviate.util import generate_uuid5
except ImportError:
    raise ImportError("Run: pip install weaviate-client")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")

from chunker import Chunker, Chunk, dedup_retrieved_chunks


# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "Ingredient"
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM    = 768
BATCH_SIZE       = 100
EMBED_BATCH_SIZE = 64
DEFAULT_TOP_K    = 5
DEFAULT_ALPHA    = 0.5


# ── Embedder ──────────────────────────────────────────────────────────────────

class LocalEmbedder:
    """
    Wrapper cho sentence-transformers với Apple Silicon MPS support.

    BAAI/bge-base-en-v1.5:
      Dimension : 768
      Context   : 512 tokens (~400 words)
      Size      : ~440MB (download 1 lần, cache sau)
      Speed     : ~2–3 min trên CPU
                  ~20–30s trên Apple Silicon MPS
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size:  int = EMBED_BATCH_SIZE,
    ):
        self.model_name  = model_name
        self.batch_size  = batch_size
        self._model: Optional[SentenceTransformer] = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"[Embedder] Loading {self.model_name} on {device}...")
        self._model = SentenceTransformer(self.model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        print(f"[Embedder] Ready ✓  dim={dim}, device={device}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed list of texts — dùng cho ingest."""
        self._load()
        return self._model.encode(
            texts,
            batch_size           = self.batch_size,
            show_progress_bar    = len(texts) > 200,
            normalize_embeddings = True,   # cosine similarity
            convert_to_numpy     = True,
        )

    def embed_query(self, query: str) -> list[float]:
        """Embed single query — dùng cho retrieval."""
        self._load()
        vec = self._model.encode(
            query,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )
        return vec.tolist()


# ── Schema ────────────────────────────────────────────────────────────────────

def get_collection_config() -> dict:
    """
    Weaviate schema — vectorizer=none vì mình tự generate vectors.
    BM25 tokenization được set trên text fields để support hybrid search.
    """
    return {
        "name": COLLECTION_NAME,
        "description": "Ingredient knowledge base chunks từ Paula's Choice",

        "vectorizer_config": Configure.Vectorizer.none(),

        "vector_index_config": Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),

        "properties": [
            # ── Embedded field (BM25 tokenized) ──────────────────────────────
            Property(
                name="chunk_text",
                data_type=DataType.TEXT,
                tokenization=Tokenization.LOWERCASE,
                skip_vectorization=True,
            ),
            # ── Identity ─────────────────────────────────────────────────────
            Property(
                name="ingredient_name",
                data_type=DataType.TEXT,
                tokenization=Tokenization.LOWERCASE,
                skip_vectorization=True,
            ),
            Property(
                name="chunk_type",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
            Property(
                name="chunk_index",
                data_type=DataType.INT,
                skip_vectorization=True,
            ),
            Property(
                name="chunk_total",
                data_type=DataType.INT,
                skip_vectorization=True,
            ),
            # ── Filterable metadata ───────────────────────────────────────────
            Property(
                name="concern_group",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
            Property(
                name="safety_label",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
            Property(
                name="safety_score",
                data_type=DataType.INT,
                skip_vectorization=True,
            ),
            Property(
                name="categories",
                data_type=DataType.TEXT_ARRAY,
                tokenization=Tokenization.LOWERCASE,
                skip_vectorization=True,
            ),
            Property(
                name="benefits",
                data_type=DataType.TEXT_ARRAY,
                tokenization=Tokenization.LOWERCASE,
                skip_vectorization=True,
            ),
            Property(
                name="warnings",
                data_type=DataType.TEXT_ARRAY,
                tokenization=Tokenization.LOWERCASE,
                skip_vectorization=True,
            ),
            # ── Source ────────────────────────────────────────────────────────
            Property(
                name="source_url",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
            Property(
                name="date_modified",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
        ],
    }


# ── Client ────────────────────────────────────────────────────────────────────

def get_client(
    url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
) -> weaviate.WeaviateClient:
    if api_key:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url      = url,
            auth_credentials = weaviate.auth.AuthApiKey(api_key),
        )

    host = url.replace("http://", "").replace("https://", "").split(":")[0]
    port = int(url.split(":")[-1]) if ":" in url.split("/")[-1] else 8080
    return weaviate.connect_to_local(host=host, port=port)


# ── Schema management ─────────────────────────────────────────────────────────

def setup_collection(
    client: weaviate.WeaviateClient,
    reset:  bool = False,
) -> None:
    exists = client.collections.exists(COLLECTION_NAME)

    if exists and reset:
        print(f"[Setup] Deleting collection: {COLLECTION_NAME}")
        client.collections.delete(COLLECTION_NAME)
        exists = False

    if not exists:
        cfg = get_collection_config()
        client.collections.create(
            name                = cfg["name"],
            description         = cfg["description"],
            vectorizer_config   = cfg["vectorizer_config"],
            vector_index_config = cfg["vector_index_config"],
            properties          = cfg["properties"],
        )
        print(f"[Setup] Created collection: {COLLECTION_NAME} ✓")
    else:
        print(f"[Setup] Collection exists (--reset to recreate)")


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_chunks(
    client:     weaviate.WeaviateClient,
    chunks:     list[Chunk],
    embedder:   LocalEmbedder,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Step 1 — Generate embeddings cho tất cả chunks (batch, MPS accelerated)
    Step 2 — Insert vào Weaviate với deterministic UUID
    """
    collection = client.collections.get(COLLECTION_NAME)

    # Step 1: Embed
    print(f"\n[Embed] Generating embeddings for {len(chunks)} chunks...")
    t0      = time.time()
    texts   = [c.chunk_text for c in chunks]
    vectors = embedder.embed(texts)
    print(f"[Embed] Done in {time.time() - t0:.1f}s  shape={vectors.shape}")

    # Step 2: Insert
    print(f"\n[Ingest] Inserting into Weaviate (batch={batch_size})...")
    t0 = time.time()

    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for chunk, vector in tqdm(
            zip(chunks, vectors),
            total=len(chunks),
            desc="Inserting",
        ):
            uuid = generate_uuid5(
                f"{chunk.ingredient_name}::{chunk.chunk_type}::{chunk.chunk_index}"
            )
            batch.add_object(
                properties={
                    "chunk_text":      chunk.chunk_text,
                    "ingredient_name": chunk.ingredient_name,
                    "chunk_type":      chunk.chunk_type,
                    "chunk_index":     chunk.chunk_index,
                    "chunk_total":     chunk.chunk_total,
                    "concern_group":   chunk.concern_group,
                    "safety_label":    chunk.safety_label,
                    "safety_score":    chunk.safety_score,
                    "categories":      chunk.categories,
                    "benefits":        chunk.benefits,
                    "warnings":        chunk.warnings,
                    "source_url":      chunk.source_url,
                    "date_modified":   chunk.date_modified,
                },
                uuid   = uuid,
                vector = vector.tolist(),
            )

    failed = collection.batch.failed_objects
    errors = len(failed) if failed else 0
    error_samples = [
        {"msg": str(f.message), "name": f.object_.properties.get("ingredient_name")}
        for f in (failed or [])[:5]
    ]

    print(f"[Ingest] Done in {time.time() - t0:.1f}s")
    print(f"         inserted : {len(chunks) - errors}")
    print(f"         errors   : {errors}")
    if error_samples:
        print(f"         samples  : {error_samples}")

    return {"inserted": len(chunks) - errors, "errors": errors}


# ── Retriever ─────────────────────────────────────────────────────────────────

class IngredientRetriever:
    """
    Query interface cho RAG pipeline.
    Import class này trong RAG pipeline để query Weaviate.
    """

    def __init__(
        self,
        client:   weaviate.WeaviateClient,
        embedder: LocalEmbedder,
    ):
        self.client     = client
        self.embedder   = embedder
        self.collection = client.collections.get(COLLECTION_NAME)

    def hybrid_search(
        self,
        query:            str,
        top_k:            int                 = DEFAULT_TOP_K,
        alpha:            float               = DEFAULT_ALPHA,
        concern_group:    Optional[str]       = None,
        categories:       Optional[list[str]] = None,
        safety_score_min: Optional[int]       = None,
        chunk_type:       Optional[str]       = None,
        dedup:            bool                = True,
    ) -> list[dict]:
        """
        Hybrid search (BM25 + vector) với metadata filters.

        alpha tuning:
          0.3 → keyword-heavy  ("phenoxyethanol safe?")
          0.5 → balanced       (default MVP)
          0.7 → semantic-heavy ("ingredient for dry sensitive skin")
        """
        # Build filters
        filter_parts = []
        if concern_group:
            filter_parts.append(
                Filter.by_property("concern_group").equal(concern_group)
            )
        if categories:
            filter_parts.append(
                Filter.by_property("categories").contains_any(categories)
            )
        if safety_score_min is not None:
            filter_parts.append(
                Filter.by_property("safety_score").greater_or_equal(safety_score_min)
            )
        if chunk_type:
            filter_parts.append(
                Filter.by_property("chunk_type").equal(chunk_type)
            )

        filters = None
        if len(filter_parts) == 1:
            filters = filter_parts[0]
        elif len(filter_parts) > 1:
            filters = filter_parts[0]
            for f in filter_parts[1:]:
                filters = filters & f

        # Query
        query_vector = self.embedder.embed_query(query)
        response = self.collection.query.hybrid(
            query          = query,
            vector         = query_vector,
            alpha          = alpha,
            limit          = top_k,
            filters        = filters,
            fusion_type    = HybridFusion.RANKED,
            return_metadata= MetadataQuery(score=True),
        )

        results = []
        for obj in response.objects:
            r = dict(obj.properties)
            r["_score"] = obj.metadata.score if obj.metadata else None
            results.append(r)

        return dedup_retrieved_chunks(results) if dedup else results

    def get_by_name(self, ingredient_name: str) -> Optional[dict]:
        """Fetch summary chunk cho 1 ingredient — dùng trong RAG post-processing."""
        response = self.collection.query.fetch_objects(
            filters=(
                Filter.by_property("ingredient_name").equal(ingredient_name.lower())
                & Filter.by_property("chunk_type").equal("summary")
            ),
            limit=1,
        )
        return dict(response.objects[0].properties) if response.objects else None

    def collection_stats(self) -> dict:
        total   = self.collection.aggregate.over_all(total_count=True).total_count
        summary = self.collection.aggregate.over_all(
            filters=Filter.by_property("chunk_type").equal("summary"),
            total_count=True,
        ).total_count
        detail  = self.collection.aggregate.over_all(
            filters=Filter.by_property("chunk_type").equal("detail"),
            total_count=True,
        ).total_count
        return {"total": total, "summary": summary, "detail": detail}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk → embed (BAAI/bge-base-en-v1.5) → ingest to Weaviate"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--kb",     help="Path to paula_choice_cleaned.json")
    src.add_argument("--chunks", help="Path to pre-built chunks.json")

    parser.add_argument("--url",         default="http://localhost:8080")
    parser.add_argument("--api-key",     default=None)
    parser.add_argument("--reset",       action="store_true")
    parser.add_argument("--batch",       type=int, default=BATCH_SIZE)
    parser.add_argument("--stats",       action="store_true")
    parser.add_argument("--save-chunks", default=None,
                        help="Export chunks.json before ingesting")
    parser.add_argument("--test-query",  default=None)

    args = parser.parse_args()

    # Load / build chunks
    if args.kb:
        chunker = Chunker(args.kb)
        chunks  = chunker.build_all()
    else:
        with open(args.chunks, encoding="utf-8") as f:
            chunks = [Chunk(**c) for c in json.load(f)]
        print(f"[Main] Loaded {len(chunks)} chunks")

    if args.save_chunks:
        with open(args.save_chunks, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
        print(f"[Main] Chunks saved → {args.save_chunks}")

    # Embedder
    embedder = LocalEmbedder()

    # Connect
    print(f"\n[Main] Connecting to Weaviate @ {args.url}")
    client = get_client(url=args.url, api_key=args.api_key)
    print(f"[Main] Weaviate v{client.get_meta()['version']} ✓")

    # Ingest
    setup_collection(client, reset=args.reset)
    ingest_chunks(client, chunks, embedder, batch_size=args.batch)

    # Stats
    if args.stats:
        r     = IngredientRetriever(client, embedder)
        stats = r.collection_stats()
        print(f"\n[Stats] total={stats['total']}  summary={stats['summary']}  detail={stats['detail']}")

    # Test query
    if args.test_query:
        print(f"\n[Test] '{args.test_query}'")
        r       = IngredientRetriever(client, embedder)
        results = r.hybrid_search(args.test_query, top_k=5)
        for res in results:
            print(
                f"  [{res.get('chunk_type','?'):7s}] "
                f"{res.get('ingredient_name','?'):40s} "
                f"concern={res.get('concern_group','?'):20s} "
                f"score={res.get('_score') or 0:.4f}"
            )

    client.close()
    print("\n✅ Done")
