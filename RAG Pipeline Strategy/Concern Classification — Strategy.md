# Ingredient Concern Classification — Strategy

## Overview

Thay vì tính overall score (dễ gây bias, không có phương pháp khoa học đủ mạnh), hệ thống phân loại mỗi ingredient vào một trong 3 concern groups dựa trên 2 tiêu chí độc lập: **Safety Rating** và **Category Override**.

---

## SAFETY_MAP

Mapping từ Paula's Choice rating sang numeric value — dùng nội bộ để xác định group, **không expose ra UI dưới dạng điểm số**.

| Paula's Choice Rating | Numeric | Safety Label        |
|-----------------------|---------|---------------------|
| Best                  | 5       | Very Safe           |
| Good                  | 4       | Safe                |
| Average               | 3       | Use with Awareness  |
| Bad                   | 2       | Caution Advised     |
| Worst                 | 1       | Known Concern       |

> **Nguồn rating:** Paula's Choice Ingredient Dictionary — không phải hệ thống tự tính.

---

## Concern Groups

| Group             | Icon | Threshold                                                              |
|-------------------|------|------------------------------------------------------------------------|
| No Concerns       | ✓    | score 4–5 **AND** category không phải Irritant / Fragrance             |
| Worth Knowing     | ⚠    | score 3 **OR** (score 4–5 AND category là Preservative / Fragrance)   |
| Potential Concerns| ✕    | score 1–2 **OR** category là Irritant                                  |

---

## Classification Logic (Pseudocode)

```python
OVERRIDE_TO_WORTH_KNOWING = {"Preservative", "Fragrance: Synthetic and Natural"}
OVERRIDE_TO_CONCERN       = {"Irritant"}

def classify(ingredient):
    score      = SAFETY_MAP[ingredient.rating]
    categories = set(ingredient.categories)

    # Category override — takes priority over score
    if categories & OVERRIDE_TO_CONCERN:
        return "potential_concern"

    if categories & OVERRIDE_TO_WORTH_KNOWING:
        return "worth_knowing"

    # Score-based fallback
    if score >= 4:
        return "no_concern"
    elif score == 3:
        return "worth_knowing"
    else:  # score <= 2
        return "potential_concern"
```

---

## Tại sao không dùng Overall Score

| Vấn đề                    | Giải thích                                                                                 |
|---------------------------|--------------------------------------------------------------------------------------------|
| **Không có cơ sở khoa học** | Không có phương pháp peer-reviewed để aggregate safety score của nhiều ingredients lại     |
| **Position bias**         | INCI listing chỉ đảm bảo thứ tự giảm dần trên 1% — dưới 1% có thể liệt kê tự do          |
| **Mean pha loãng**        | 9 ingredients tốt + 1 Worst → mean cao, nhưng vẫn có known concern                        |
| **Precedent tiêu cực**    | Yuka, EWG, CosDNA đang bị chỉ trích vì scoring tự đặt ra không nhất quán                  |

**Thay thế:** Concern-based grouping + transparent sourcing — hiển thị đúng những gì data nói, không diễn giải thêm.

---

## Warning Generation

Warning được generate từ `categories` field trong knowledge base — **không** từ score.

```python
WARNING_TRIGGERS = {
    "Irritant":                        "May cause irritation — patch test recommended.",
    "Fragrance: Synthetic and Natural": "Common allergen — avoid if sensitive skin.",
    "Preservative":                    "Preservative — safe within regulated limits.",
}

def generate_warnings(ingredient):
    for category in ingredient.categories:
        if category in WARNING_TRIGGERS:
            yield f"{ingredient.name}: {WARNING_TRIGGERS[category]}"
```

---

## Data Flow (Runtime)

```
OCR Output
["Water", "Glycerin", "Niacinamide", "Phenoxyethanol", "Fragrance"]
          ↓
  Lookup trong Knowledge Base (paula_choice_cleaned.json)
          ↓
  classify(ingredient)      → concern group
  generate_warnings(ingredient) → warning text
          ↓
  Structured output cho UI
  {
    no_concern:       [Water, Glycerin, Niacinamide, Butylene Glycol]
    worth_knowing:    [Phenoxyethanol]
    potential_concern:[Fragrance]
    warnings:         ["Fragrance: Common allergen..."]
  }
```

---

## Disclaimer

Kết quả phân tích mang tính thông tin — không thay thế tư vấn của bác sĩ da liễu.
Rating nguồn từ Paula's Choice Ingredient Dictionary, cập nhật định kỳ.
