# Chunking Strategy

## Overview

Knowledge base gồm 2509 ingredient records từ Paula's Choice Ingredient Dictionary.  
Mỗi record được chia thành **2 loại chunk** trước khi đưa vào Weaviate.

---

## Tại sao cần chunking?

| Vấn đề | Chi tiết |
|--------|----------|
| Description quá dài | Max ~3979 chars, avg ~668 chars — vượt ngưỡng embedding tối ưu |
| Mất entity context | Chunk tách ra không biết đang nói về ingredient nào |
| Retrieval không đồng nhất | Summary query và detail query cần chunk khác nhau |

---

## Chunk Types

### Chunk Type 1 — Summary Chunk

**Mục đích:** Quick lookup, cross-ingredient query, filter-based search

**1 record = 1 summary chunk** (luôn có)

**Content:**
```
Ingredient: {name} | Categories: {categories} | Key Points: {key_points} |
Benefits: {benefits} | Safety: {safety_label} | Concern Group: {concern_group}
```

**Ví dụ:**
```
Ingredient: Niacinamide | Categories: Antioxidant, Humectant |
Key Points: Improves enlarged pores. Brightens dull tone. Boosts barrier strength |
Benefits: Anti-Aging, Pore Minimizer, Soothing | Safety: Very Safe |
Concern Group: No Concerns
```

**Size:** ~150–300 chars

---

### Chunk Type 2 — Detail Chunk

**Mục đích:** Q&A chi tiết ("Niacinamide hoạt động như thế nào?")

**Chỉ tạo khi description > 500 chars**  
**1 record có thể có 0–n detail chunks**

**Splitting rules:**
- Target chunk size: 300–500 chars
- Overlap: 50 chars (tránh mất context tại boundary)
- Split tại sentence boundary (`.`) gần nhất với target size

**Content:**
```
Ingredient: {name} [detail {i}/{total}] | {description_chunk}
```

**Ví dụ:**
```
Ingredient: Niacinamide [detail 1/2] | Niacinamide (also known as vitamin B3
or nicotinamide) is a unique skin-restoring ingredient that offers a multitude
of benefits for skin. It is best known for its ability to help visibly reduce
enlarged pores and improve uneven skin tone...

Ingredient: Niacinamide [detail 2/2] | ...Unlike many superstar anti-aging
ingredients, niacinamide is stable in the presence of heat and light.
Niacinamide offers benefits starting in concentrations as low as 0.2%...
```

**Size:** 300–500 chars per chunk, overlap 50 chars

---

## Ingredient Name Prefix — Lý do thiết kế

Mỗi chunk đều bắt đầu bằng `Ingredient: {name}` — áp dụng cho cả summary và detail chunk.

**Lý do kỹ thuật:**

```
Không có prefix:
  "...helps improve skin barrier and reduce redness..."
  → Vector nằm ở vùng generic "skin benefit"
  → Không gắn với entity cụ thể

Có prefix:
  "Ingredient: Niacinamide | ...helps improve skin barrier..."
  → Vector mang cả 2 signal: entity identity + semantic content
  → Retrieval chính xác hơn
```

**Lợi ích:**
- Embedding biết context là về ingredient cụ thể nào
- BM25 matching tốt hơn khi user nhắc tên ingredient trong query
- Chunk không mất entity context sau khi tách khỏi record gốc
- `[detail i/total]` tag cho LLM biết đây là partial description

**Potential downside đã xem xét:**  
Nếu prefix chiếm > 30% độ dài chunk, vector bị kéo về phía tên ingredient.  
→ Giải quyết bằng cách đảm bảo detail chunk đủ dài (≥ 300 chars).

---

## Weaviate Schema

```python
# Collection: Ingredient
{
    # ── Embedded field ───────────────────────────────
    "chunk_text":      str,   # text đưa vào embedding model

    # ── Metadata fields (filterable) ─────────────────
    "ingredient_name": str,   # filter / group by
    "chunk_type":      str,   # "summary" | "detail"
    "chunk_index":     int,   # 0 = summary, 1,2,3... = detail parts
    "chunk_total":     int,   # tổng số chunks của ingredient này

    "concern_group":   str,   # "no_concern" | "worth_knowing" | "potential_concern"
    "safety_label":    str,   # "Very Safe" | "Safe" | "Use with Awareness" | ...
    "safety_score":    int,   # 1–5 (internal, không expose ra UI)
    "categories":    list[str],  # ["Humectant", "Antioxidant", ...]
    "benefits":      list[str],  # ["Hydration", "Anti-Aging", ...]
    "warnings":      list[str],  # warning text nếu có

    "source_url":      str,   # Paula's Choice URL
    "date_modified":   str,
}
```

---

## Query Flow — Hybrid Search

Weaviate hybrid search kết hợp **BM25 + vector similarity**:

```python
# User query: "humectant nào an toàn cho da nhạy cảm?"

collection.query.hybrid(
    query   = "humectant safe sensitive skin",
    filters = Filter.by_property("concern_group").equal("no_concern")
            & Filter.by_property("categories").contains_any(["Humectant"]),
    limit   = 5,
    alpha   = 0.5    # 0.0 = BM25 only | 1.0 = vector only | 0.5 = balanced
)
```

**alpha tuning:**

| alpha | Khi nào dùng |
|-------|-------------|
| 0.3   | Query ngắn, keyword rõ ràng ("phenoxyethanol safe?") |
| 0.5   | Balanced — default cho MVP |
| 0.7   | Query dài, semantic ("ingredient nào phù hợp da khô nhạy cảm?") |

---

## Chunk Count Estimate

| Loại | Ước tính |
|------|---------|
| Summary chunks | 2509 (1 per ingredient) |
| Detail chunks | ~800–1000 (chỉ ingredients có description > 500 chars) |
| **Total** | **~3300–3500 chunks** |

---

## Retrieval — Post-processing

Sau khi Weaviate trả về top-k chunks, cần dedup trước khi đưa vào LLM:

```python
# Vấn đề: query có thể trả về cả summary + detail của cùng 1 ingredient
# → LLM nhận context trùng lặp

# Giải quyết: nếu đã có detail chunk → bỏ summary chunk của cùng ingredient
def dedup_chunks(chunks: list) -> list:
    seen = {}
    for chunk in chunks:
        name = chunk["ingredient_name"]
        if name not in seen or chunk["chunk_type"] == "detail":
            seen[name] = chunk
    return list(seen.values())
```

---

## File liên quan

| File | Vai trò |
|------|---------|
| `paula_choice_cleaning.ipynb` | Build + export cleaned KB |
| `chunker.py` | Tạo chunks từ cleaned KB |
| `weaviate_ingest.py` | Load chunks vào Weaviate |
| `concern_classifier.py` | Runtime lookup (Luồng 1, không qua RAG) |
