"""
concern_classifier.py
─────────────────────────────────────────────────────────────────────────────
Runtime module — Luồng 1 của app (không cần RAG / LLM)

Nhận ingredient list từ OCR pipeline, lookup trong Knowledge Base,
trả về grouped result + warnings cho UI.

Flow:
    OCR output
        │
        ▼
    normalize_name()          ← chuẩn hóa tên trước khi lookup
        │
        ▼
    lookup_ingredient()       ← exact match + fuzzy fallback
        │
        ▼
    classify_product()        ← group + aggregate warnings
        │
        ▼
    ProductAnalysisResult     ← structured output cho UI

Usage:
    from concern_classifier import ConcernClassifier

    clf = ConcernClassifier("paula_choice_cleaned.json")
    result = clf.classify(["Water", "Glycerin", "Niacinamide", "Fragrance"])
    print(result.to_dict())
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data classes (output schema) ─────────────────────────────────────────────

@dataclass
class IngredientResult:
    """Per-ingredient analysis result."""
    name: str                           # tên sau khi normalize
    matched_name: str                   # tên trong KB (có thể khác nếu fuzzy match)
    concern_group: str                  # no_concern | worth_knowing | potential_concern
    safety_label: str                   # Very Safe | Safe | Use with Awareness | ...
    safety_score: int                   # 1–5 (internal, không expose dưới dạng score UI)
    categories: list[str]
    warnings: list[str]
    key_points: list[str]
    description: str
    match_type: str                     # exact | fuzzy | not_found
    confidence: float                   # 1.0 = exact, 0.0–1.0 = fuzzy, 0.0 = not found


@dataclass
class ProductAnalysisResult:
    """Aggregated result cho toàn bộ sản phẩm."""
    total_detected: int
    total_matched: int
    total_not_found: int

    no_concern: list[IngredientResult]          = field(default_factory=list)
    worth_knowing: list[IngredientResult]       = field(default_factory=list)
    potential_concern: list[IngredientResult]   = field(default_factory=list)
    not_found: list[str]                        = field(default_factory=list)

    all_warnings: list[str]                     = field(default_factory=list)

    def to_dict(self) -> dict:
        def _fmt(items: list[IngredientResult]) -> list[dict]:
            return [
                {
                    "name":          i.matched_name,
                    "concern_group": i.concern_group,
                    "safety_label":  i.safety_label,
                    "categories":    i.categories,
                    "warnings":      i.warnings,
                    "key_points":    i.key_points,
                    "description":   i.description,
                    "match_type":    i.match_type,
                }
                for i in items
            ]


        return {
            "summary": {
                "total_detected":  self.total_detected,
                "total_matched":   self.total_matched,
                "total_not_found": self.total_not_found,
            },
            "groups": {
                "no_concern":        _fmt(self.no_concern),
                "worth_knowing":     _fmt(self.worth_knowing),
                "potential_concern": _fmt(self.potential_concern),
            },
            "not_found":   self.not_found,
            "all_warnings": self.all_warnings,
        }


# ── INCI Synonym map ──────────────────────────────────────────────────────────
# Cover các aliases phổ biến nhất trên nhãn sản phẩm
# Áp dụng SAU khi strip parenthetical và normalize
# Key: normalized alias → Value: normalized KB name

INCI_SYNONYMS: dict[str, str] = {
    "aqua":              "water",
    "parfum":            "parfum/fragrance",
    "fragrance":         "parfum/fragrance",
    "aroma":             "parfum/fragrance",
    "alcohol denat":     "denatured alcohol",
    "tocopherol":        "vitamin e",
    "ascorbic acid":     "vitamin c",
    "l-ascorbic acid":   "vitamin c",
    "ha":                "hyaluronic acid",
}


# ── Classifier ────────────────────────────────────────────────────────────────

class ConcernClassifier:
    """
    Load knowledge base một lần, reuse cho nhiều request.

    KB expected schema (từ paula_choice_cleaning.ipynb output):
    {
        "name":          str,
        "rating":        str,
        "safety_score":  int,
        "safety_label":  str,
        "concern_group": str,
        "description":   str,
        "key_points":    list[str],
        "benefits":      list[str],
        "categories":    list[str],
        "warnings":      list[str],
        ...
    }
    """

    def __init__(self, kb_path: str | Path):
        self.kb_path = Path(kb_path)
        self._kb: list[dict] = []
        self._index: dict[str, dict] = {}      # normalized_name → record
        self._load_kb()

    # ── KB loading ────────────────────────────────────────────────────────────

    def _load_kb(self) -> None:
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.kb_path}")

        with open(self.kb_path, encoding="utf-8") as f:
            self._kb = json.load(f)

        # Build lookup index: normalized name → record
        for record in self._kb:
            key = self._normalize(record["name"])
            self._index[key] = record

        print(f"[ConcernClassifier] Loaded {len(self._kb)} ingredients from KB")

    # ── Name normalization ────────────────────────────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        """
        Chuẩn hóa tên ingredient để so sánh.

        Xử lý các biến thể thường gặp từ OCR và nhãn sản phẩm:
            "NIACINAMIDE"              → "niacinamide"
            "Aqua (Water)"             → "water"        (strip paren + synonym)
            "Parfum (Fragrance)"       → "fragrance"    (strip paren + synonym)
            "Panthenol (Vitamin B5)"   → "panthenol"    (strip paren)
            "Hyaluronic Acid (XS)"     → "hyaluronic acid" (strip paren)
            "Phenoxyéthanol"           → "phenoxyethanol"  (unicode)
        """
        if not name:
            return ""

        # Unicode normalize (é → e)
        name = unicodedata.normalize("NFKD", name)
        name = name.encode("ascii", "ignore").decode("ascii")

        # Lowercase
        name = name.lower()

        # Strip parenthetical annotations: "Aqua (Water)" → "Aqua"
        # Covers: (Water), (Vitamin B5), (XS), (Fragrance), etc.
        name = re.sub(r"\s*\(.*?\)", "", name)

        # Collapse whitespace + strip
        name = re.sub(r"\s+", " ", name).strip()

        # Remove trailing punctuation noise từ OCR
        name = re.sub(r"[,;.\-]+$", "", name).strip()

        # INCI synonym map — apply sau khi strip paren
        name = INCI_SYNONYMS.get(name, name)

        return name

    # ── Lookup strategies ─────────────────────────────────────────────────────

    def _exact_lookup(self, name: str) -> Optional[dict]:
        """Exact match sau normalize."""
        return self._index.get(self._normalize(name))

    def _fuzzy_lookup(self, name: str, threshold: float = 0.80) -> Optional[tuple[dict, float]]:
        """
        Simple token-based fuzzy match — không dùng thư viện ngoài.

        Ưu tiên:
        1. Normalized name là substring của KB name (hoặc ngược lại)
        2. Token overlap ratio

        Return: (record, confidence) hoặc None
        """
        query_norm = self._normalize(name)
        query_tokens = set(query_norm.split())

        best_record = None
        best_score = 0.0

        for kb_norm, record in self._index.items():
            kb_tokens = set(kb_norm.split())

            # Token-level substring check — e.g. "aqua" token in "water (aqua)" tokens
            # Phải match ở token level, không phải character level
            # Tránh false positive: "mica" trong "unknownchemical"
            query_token_subset = query_tokens.issubset(kb_tokens)
            kb_token_subset    = kb_tokens.issubset(query_tokens)

            if query_token_subset or kb_token_subset:
                shorter = min(len(query_tokens), len(kb_tokens))
                longer  = max(len(query_tokens), len(kb_tokens))
                score = shorter / longer
                score = max(score, 0.85)
            else:
                # Token overlap (Jaccard)
                intersection = query_tokens & kb_tokens
                union = query_tokens | kb_tokens
                score = len(intersection) / len(union) if union else 0.0

            if score > best_score:
                best_score = score
                best_record = record

        if best_score >= threshold:
            return best_record, best_score
        return None

    def _lookup(self, name: str) -> tuple[Optional[dict], str, float]:
        """
        Lookup với fallback strategy.
        Return: (record, match_type, confidence)
        """
        # 1. Exact match
        record = self._exact_lookup(name)
        if record:
            return record, "exact", 1.0

        # 2. Fuzzy match
        result = self._fuzzy_lookup(name)
        if result:
            record, confidence = result
            return record, "fuzzy", confidence

        # 3. Not found
        return None, "not_found", 0.0

    # ── Main classify method ──────────────────────────────────────────────────

    def classify(self, ingredient_list: list[str]) -> ProductAnalysisResult:
        """
        Classify toàn bộ ingredient list từ OCR.

        Args:
            ingredient_list: list tên ingredients, e.g. từ OCR output
                             ["Water", "Glycerin", "Niacinamide", "Fragrance"]

        Returns:
            ProductAnalysisResult với đầy đủ grouped results + warnings
        """
        results: list[IngredientResult] = []
        not_found: list[str] = []

        for raw_name in ingredient_list:
            if not raw_name or not raw_name.strip():
                continue

            record, match_type, confidence = self._lookup(raw_name.strip())

            if match_type == "not_found":
                not_found.append(raw_name.strip())
                continue

            ing_result = IngredientResult(
                name=raw_name.strip(),
                matched_name=record["name"],
                concern_group=record.get("concern_group", "worth_knowing"),
                safety_label=record.get("safety_label", "Unknown"),
                safety_score=record.get("safety_score", 0),
                categories=record.get("categories", []),
                warnings=record.get("warnings", []),
                key_points=record.get("key_points", []),
                description=record.get("description", ""),
                match_type=match_type,
                confidence=confidence,
            )
            results.append(ing_result)

        # Aggregate all warnings (deduplicated)
        all_warnings = []
        seen_warnings = set()
        for r in results:
            for w in r.warnings:
                if w not in seen_warnings:
                    seen_warnings.add(w)
                    all_warnings.append(w)

        return ProductAnalysisResult(
            total_detected=len(ingredient_list),
            total_matched=len(results),
            total_not_found=len(not_found),
            no_concern       =[r for r in results if r.concern_group == "no_concern"],
            worth_knowing    =[r for r in results if r.concern_group == "worth_knowing"],
            potential_concern=[r for r in results if r.concern_group == "potential_concern"],
            not_found=not_found,
            all_warnings=all_warnings,
        )

    def get_ingredient(self, name: str) -> Optional[IngredientResult]:
        """
        Lookup single ingredient — dùng cho RAG pipeline khi cần
        fetch KB record sau retrieval.
        """
        record, match_type, confidence = self._lookup(name)
        if not record:
            return None

        return IngredientResult(
            name=name,
            matched_name=record["name"],
            concern_group=record.get("concern_group", "worth_knowing"),
            safety_label=record.get("safety_label", "Unknown"),
            safety_score=record.get("safety_score", 0),
            categories=record.get("categories", []),
            warnings=record.get("warnings", []),
            key_points=record.get("key_points", []),
            description=record.get("description", ""),
            match_type=match_type,
            confidence=confidence,
        )

    @property
    def kb_size(self) -> int:
        return len(self._kb)


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    kb_path = sys.argv[1] if len(sys.argv) > 1 else "paula_choice_cleaned.json"

    clf = ConcernClassifier(kb_path)

    # Mock OCR output
    test_ingredients = [
        "Water",
        "Glycerin",
        "Niacinamide",
        "Butylene Glycol",
        "Phenoxyethanol",
        "Parfum/Fragrance",
        "UnknownChemical XR-99",     # not_found case
        "NIACINAMIDE",               # case variation
        "glycerin",                  # lowercase
    ]

    print("\n" + "=" * 60)
    print("CONCERN CLASSIFIER — TEST RUN")
    print("=" * 60)
    print(f"Input ({len(test_ingredients)} ingredients): {test_ingredients}\n")

    result = clf.classify(test_ingredients)
    output = result.to_dict()

    # Summary
    s = output["summary"]
    print(f"Summary:")
    print(f"  Detected  : {s['total_detected']}")
    print(f"  Matched   : {s['total_matched']}")
    print(f"  Not found : {s['total_not_found']}")

    # Groups
    for group_key, label, icon in [
        ("no_concern",        "No Concerns",        "✓"),
        ("worth_knowing",     "Worth Knowing",       "⚠"),
        ("potential_concern", "Potential Concerns",  "✕"),
    ]:
        items = output["groups"][group_key]
        if not items:
            continue
        print(f"\n{icon} {label} ({len(items)})")
        for item in items:
            match_info = f"[{item['match_type']}]" if item['match_type'] != 'exact' else ""
            print(f"  {item['name']:40s} {item['safety_label']:20s} {match_info}")
            if item['warnings']:
                for w in item['warnings']:
                    print(f"    ↳ {w}")

    # Not found
    if output["not_found"]:
        print(f"\n? Not in KB ({len(output['not_found'])})")
        for name in output["not_found"]:
            print(f"  {name}")

    # All warnings
    if output["all_warnings"]:
        print(f"\n⚑ All Warnings:")
        for w in output["all_warnings"]:
            print(f"  • {w}")