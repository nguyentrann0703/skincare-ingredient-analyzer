"""
rag_pipeline.py
─────────────────────────────────────────────────────────────────────────────
RAG Pipeline — Luồng 2 của app.

Supports 2 query types:
  - ingredient_specific : "What is Niacinamide?" → lookup by name
  - open_ended          : "What's good for sensitive skin?" → hybrid search

Flow:
    User query
        ↓
    QueryClassifier         ← rule-based, detect query type + extract params
        ↓
    IngredientRetriever     ← Weaviate hybrid search + metadata filters
        ↓
    PromptBuilder           ← build prompt từ retrieved chunks
        ↓
    OllamaLLM               ← Qwen2.5:7b via Ollama
        ↓
    RAGResponse             ← structured output cho Streamlit

Usage:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    response = pipeline.query("What does niacinamide do for skin?")
    print(response.answer)
"""

from __future__ import annotations

import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import weaviate
from weaviate.classes.query import Filter, MetadataQuery, HybridFusion

from weaviate_ingest import LocalEmbedder, IngredientRetriever, get_client
from chunker import dedup_retrieved_chunks


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "qwen2.5:7b"
OLLAMA_TIMEOUT   = 120          # seconds

WEAVIATE_URL     = "http://localhost:8080"
TOP_K_SPECIFIC   = 3            # ingredient-specific: ít chunk hơn, focused
TOP_K_OPEN       = 5            # open-ended: nhiều chunk hơn, broader context
ALPHA            = 0.5


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    query:        str
    query_type:   str                   # "ingredient_specific" | "open_ended"
    answer:       str                   # LLM generated answer
    sources:      list[str]             # ingredient names used as context
    chunks_used:  int                   # number of chunks retrieved
    latency_ms:   int                   # total pipeline latency
    error:        Optional[str] = None  # error message nếu có


# ── Query Classifier ──────────────────────────────────────────────────────────

class QueryClassifier:
    """
    Rule-based classifier — không dùng LLM để classify.
    Detect query type + extract metadata filters từ query text.

    ingredient_specific: query chứa tên ingredient có trong KB index
    open_ended         : còn lại
    """

    def __init__(self, ingredient_names: list[str]):
        # Sort by length descending — longer names match first
        # Prevents "Niacin" from matching before "Niacinamide"
        sorted_names = sorted(ingredient_names, key=len, reverse=True)
        self._names: dict[str, str] = {
            self._norm(n): n for n in sorted_names
        }
        # Keep insertion order (Python 3.7+) — longest first
        self._names_ordered = list(self._names.items())

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def classify(self, query: str) -> dict:
        """
        Returns:
        {
            "type"              : "ingredient_specific" | "open_ended",
            "ingredient_name"   : str | None,   # nếu ingredient_specific
            "filters"           : dict,          # metadata filters cho Weaviate
        }
        """
        query_norm = self._norm(query)

        # Check từng ingredient name — longest first to prevent partial match
        matched_name = None
        for norm_name, orig_name in self._names_ordered:
            # Word boundary check: tên phải xuất hiện như whole word trong query
            # Tránh "Niacin" match trong "Niacinamide"
            pattern = r'(?<![a-z0-9])' + re.escape(norm_name) + r'(?![a-z0-9])'
            if re.search(pattern, query_norm):
                matched_name = orig_name
                break

        if matched_name:
            return {
                "type":             "ingredient_specific",
                "ingredient_name":  matched_name,
                "filters":          {},
            }

        # Open-ended — extract implicit filters từ keywords
        filters = {}

        concern_keywords = {
            "safe":        "no_concern",
            "gentle":      "no_concern",
            "sensitive":   "no_concern",
            "avoid":       "potential_concern",
            "harmful":     "potential_concern",
            "bad":         "potential_concern",
            "irritat":     "potential_concern",
        }
        for kw, group in concern_keywords.items():
            if kw in query_norm:
                filters["concern_group"] = group
                break

        category_keywords = {
            "humectant":    "Humectant",
            "moistur":      "Humectant",
            "antioxidant":  "Antioxidant",
            "preservative": "Preservative",
            "fragrance":    "Fragrance: Synthetic and Natural",
            "exfoliant":    "Exfoliant",
            "emollient":    "Emollient",
            "peptide":      "Peptides",
            "sunscreen":    "Sunscreen Agent",
        }
        matched_cats = []
        for kw, cat in category_keywords.items():
            if kw in query_norm:
                matched_cats.append(cat)
        if matched_cats:
            filters["categories"] = matched_cats

        return {
            "type":            "open_ended",
            "ingredient_name": None,
            "filters":         filters,
        }


# ── Prompt Builder ────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Build prompt từ retrieved chunks + user query.
    2 templates: ingredient_specific và open_ended.
    """

    SYSTEM_PROMPT = """You are a cosmetic ingredient expert with deep knowledge of skincare science.
Your role is to provide accurate, evidence-based information about cosmetic ingredients.
Always be clear, concise, and helpful. Base your answers only on the provided context.
If information is not available in the context, say so — do not invent facts."""

    TEMPLATE_SPECIFIC = """Context information about {ingredient_name}:
{context}

User question: {query}

Provide a structured answer with:
- **Description**: What this ingredient is
- **Key Benefits**: Main benefits for skin
- **Safety**: Safety profile and recommended usage
- **Warnings**: Any concerns or contraindications (if applicable)
- **Source**: Paula's Choice Ingredient Dictionary

Keep your answer factual and concise."""

    TEMPLATE_OPEN = """Context information from the ingredient knowledge base:
{context}

User question: {query}

Based on the provided context, answer the question clearly.
- Reference specific ingredient names when relevant
- Be concise and practical
- If recommending ingredients, mention their safety profile
- Source: Paula's Choice Ingredient Dictionary"""

    def build(
        self,
        query:           str,
        query_type:      str,
        chunks:          list[dict],
        ingredient_name: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Returns: (system_prompt, user_prompt)
        """
        # Format context từ chunks
        context_parts = []
        for chunk in chunks:
            name  = chunk.get("ingredient_name", "")
            ctype = chunk.get("chunk_type", "")
            text  = chunk.get("chunk_text", "")

            if ctype == "summary":
                context_parts.append(f"[{name} — Overview]\n{text}")
            else:
                idx   = chunk.get("chunk_index", "")
                total = chunk.get("chunk_total", "")
                context_parts.append(f"[{name} — Detail {idx}]\n{text}")

        context = "\n\n".join(context_parts)

        if query_type == "ingredient_specific" and ingredient_name:
            user_prompt = self.TEMPLATE_SPECIFIC.format(
                ingredient_name = ingredient_name,
                context         = context,
                query           = query,
            )
        else:
            user_prompt = self.TEMPLATE_OPEN.format(
                context = context,
                query   = query,
            )

        return self.SYSTEM_PROMPT, user_prompt


# ── Ollama LLM ────────────────────────────────────────────────────────────────

class OllamaLLM:
    """
    Wrapper cho Ollama API — Qwen2.5:7b.
    Support cả streaming và non-streaming.
    """

    def __init__(
        self,
        base_url:  str = OLLAMA_BASE_URL,
        model:     str = OLLAMA_MODEL,
        timeout:   int = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout

    def is_available(self) -> bool:
        """Check Ollama server + model availability."""
        try:
            resp = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def generate(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float = 0.3,    # low temp → more factual
        max_tokens:    int   = 1024,
    ) -> str:
        """Non-streaming generation."""
        payload = {
            "model":  self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json    = payload,
            timeout = self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float = 0.3,
        max_tokens:    int   = 1024,
    ):
        """
        Streaming generation — yields text chunks.
        Dùng cho Streamlit st.write_stream().
        """
        payload = {
            "model":  self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json    = payload,
            timeout = self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Main pipeline — wire tất cả components lại.

    Usage:
        pipeline = RAGPipeline()
        response = pipeline.query("What does niacinamide do?")

        # Streaming (cho Streamlit)
        for token in pipeline.query_stream("Is fragrance safe?"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        weaviate_url: str = WEAVIATE_URL,
        ollama_url:   str = OLLAMA_BASE_URL,
        model:        str = OLLAMA_MODEL,
    ):
        # Connect Weaviate
        self._weaviate_client = get_client(url=weaviate_url)
        self._embedder        = LocalEmbedder()
        self._retriever       = IngredientRetriever(
            client   = self._weaviate_client,
            embedder = self._embedder,
        )

        # Load ingredient names for classifier
        ingredient_names = self._load_ingredient_names()
        self._classifier  = QueryClassifier(ingredient_names)
        self._prompt_builder = PromptBuilder()
        self._llm = OllamaLLM(base_url=ollama_url, model=model)

        print(f"[RAGPipeline] Ready")
        print(f"  Weaviate : {weaviate_url}")
        print(f"  LLM      : {model} @ {ollama_url}")
        print(f"  KB names : {len(ingredient_names)} ingredients indexed")

    def _load_ingredient_names(self) -> list[str]:
        """Fetch tất cả ingredient names từ Weaviate để build classifier index."""
        collection = self._weaviate_client.collections.get("Ingredient")
        response   = collection.query.fetch_objects(
            filters = Filter.by_property("chunk_type").equal("summary"),
            limit   = 3000,
        )
        return [obj.properties["ingredient_name"] for obj in response.objects]

    def _retrieve(self, classification: dict) -> list[dict]:
        """Retrieve chunks dựa trên query classification."""
        qtype   = classification["type"]
        filters = classification.get("filters", {})

        if qtype == "ingredient_specific":
            name = classification["ingredient_name"]
            # Fetch cả summary + detail chunks của ingredient này
            chunks = self._retriever.hybrid_search(
                query      = name,
                top_k      = TOP_K_SPECIFIC,
                alpha      = 0.7,           # lean toward vector for specific lookup
                chunk_type = None,          # both summary + detail
                dedup      = False,         # keep all chunks for this ingredient
            )
            # Filter chỉ lấy chunks của đúng ingredient đó
            chunks = [
                c for c in chunks
                if c.get("ingredient_name", "").lower() == name.lower()
            ]
            # Nếu không có → fallback to hybrid search rộng hơn
            if not chunks:
                chunks = self._retriever.hybrid_search(
                    query = name,
                    top_k = TOP_K_SPECIFIC,
                    alpha = 0.7,
                    dedup = True,
                )
        else:
            # Open-ended
            search_kwargs = {
                "top_k": TOP_K_OPEN,
                "alpha": ALPHA,
                "dedup": True,
            }
            if "concern_group" in filters:
                search_kwargs["concern_group"] = filters["concern_group"]
            if "categories" in filters:
                search_kwargs["categories"] = filters["categories"]

            chunks = self._retriever.hybrid_search(
                query=classification.get("raw_query", ""),
                **search_kwargs,
            )

        return chunks

    def query(self, user_query: str) -> RAGResponse:
        """
        Non-streaming query — trả về RAGResponse đầy đủ.
        """
        t0 = time.time()

        # Classify
        classification = self._classifier.classify(user_query)
        classification["raw_query"] = user_query

        # Retrieve
        chunks = self._retrieve(classification)

        if not chunks:
            return RAGResponse(
                query      = user_query,
                query_type = classification["type"],
                answer     = "I couldn't find relevant information for your query in the knowledge base.",
                sources    = [],
                chunks_used= 0,
                latency_ms = int((time.time() - t0) * 1000),
            )

        # Build prompt
        system_prompt, user_prompt = self._prompt_builder.build(
            query           = user_query,
            query_type      = classification["type"],
            chunks          = chunks,
            ingredient_name = classification.get("ingredient_name"),
        )

        # Generate
        answer = self._llm.generate(system_prompt, user_prompt)

        # Sources
        sources = list(dict.fromkeys(
            c.get("ingredient_name", "") for c in chunks
        ))

        return RAGResponse(
            query      = user_query,
            query_type = classification["type"],
            answer     = answer,
            sources    = sources,
            chunks_used= len(chunks),
            latency_ms = int((time.time() - t0) * 1000),
        )

    def query_stream(self, user_query: str):
        """
        Streaming query — yields (token | None, metadata).
        Dùng cho Streamlit st.write_stream().

        Yields:
            str tokens từ LLM
        After exhausting: call .last_response for metadata
        """
        classification = self._classifier.classify(user_query)
        classification["raw_query"] = user_query

        chunks = self._retrieve(classification)

        if not chunks:
            yield "I couldn't find relevant information for your query."
            return

        system_prompt, user_prompt = self._prompt_builder.build(
            query           = user_query,
            query_type      = classification["type"],
            chunks          = chunks,
            ingredient_name = classification.get("ingredient_name"),
        )

        self._last_chunks = chunks
        self._last_classification = classification

        yield from self._llm.generate_stream(system_prompt, user_prompt)

    @property
    def last_sources(self) -> list[str]:
        """Sources từ query_stream call cuối — dùng sau khi stream xong."""
        chunks = getattr(self, "_last_chunks", [])
        return list(dict.fromkeys(c.get("ingredient_name", "") for c in chunks))

    @property
    def last_query_type(self) -> str:
        clf = getattr(self, "_last_classification", {})
        return clf.get("type", "")

    def check_health(self) -> dict:
        """Check health của tất cả dependencies."""
        weaviate_ok = False
        ollama_ok   = self._llm.is_available()

        try:
            stats = self._retriever.collection_stats()
            weaviate_ok = stats["total"] > 0
        except Exception:
            pass

        return {
            "weaviate": weaviate_ok,
            "ollama":   ollama_ok,
            "model":    OLLAMA_MODEL,
        }

    def close(self):
        self._weaviate_client.close()


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pipeline = RAGPipeline()

    # Health check
    health = pipeline.check_health()
    print(f"\n[Health]")
    print(f"  Weaviate : {'✓' if health['weaviate'] else '✗'}")
    print(f"  Ollama   : {'✓' if health['ollama'] else '✗'} ({health['model']})")

    if not health["ollama"]:
        print("\n⚠ Ollama not available. Run: ollama serve")
        sys.exit(1)

    # Test queries
    test_queries = [
        "What does niacinamide do for skin?",
        "Is phenoxyethanol safe to use?",
        "What humectants are safe for sensitive skin?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        response = pipeline.query(query)
        print(f"Type    : {response.query_type}")
        print(f"Sources : {response.sources}")
        print(f"Latency : {response.latency_ms}ms")
        print(f"\nAnswer:\n{response.answer}")

    pipeline.close()