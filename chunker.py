"""
chunker.py
─────────────────────────────────────────────────────────────────────────────
Tạo chunks từ paula_choice_cleaned.json để ingest vào Weaviate.

Chunk Types:
  - summary : 1 per ingredient, luôn có
              content = name + categories + key_points + benefits + safety
  - detail  : chỉ khi description > 500 chars
              content = name prefix + description (split tại sentence boundary)
              overlap = 50 chars

Usage:
    from chunker import Chunker

    chunker = Chunker("paula_choice_cleaned.json")
    chunks  = chunker.build_all()

    # Hoặc chạy trực tiếp để preview + export
    python chunker.py paula_choice_cleaned.json --output chunks.json
"""

from __future__ import annotations

import json
import re
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

DETAIL_THRESHOLD = 500      # description ngắn hơn → chỉ có summary chunk
CHUNK_TARGET     = 400      # target chars per detail chunk
CHUNK_OVERLAP    = 50       # overlap chars giữa các detail chunks


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    Một chunk sẵn sàng để embed + ingest vào Weaviate.

    chunk_text  → field được đưa vào embedding model
    Tất cả còn lại → metadata fields (filterable, không embed)
    """
    # ── Embedded ──────────────────────────────────────────
    chunk_text:      str

    # ── Identity ──────────────────────────────────────────
    ingredient_name: str
    chunk_type:      str        # "summary" | "detail"
    chunk_index:     int        # 0 = summary, 1,2,3... = detail parts
    chunk_total:     int        # tổng số chunks của ingredient này

    # ── Filterable metadata ───────────────────────────────
    concern_group:   str
    safety_label:    str
    safety_score:    int
    categories:      list[str]
    benefits:        list[str]
    warnings:        list[str]

    # ── Source ────────────────────────────────────────────
    source_url:      str
    date_modified:   str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Chunker ───────────────────────────────────────────────────────────────────

class Chunker:

    def __init__(
        self,
        kb_path: str | Path,
        detail_threshold: int = DETAIL_THRESHOLD,
        chunk_target:     int = CHUNK_TARGET,
        chunk_overlap:    int = CHUNK_OVERLAP,
    ):
        self.kb_path          = Path(kb_path)
        self.detail_threshold = detail_threshold
        self.chunk_target     = chunk_target
        self.chunk_overlap    = chunk_overlap
        self._kb: list[dict]  = []
        self._load()

    def _load(self) -> None:
        if not self.kb_path.exists():
            raise FileNotFoundError(f"KB not found: {self.kb_path}")
        with open(self.kb_path, encoding="utf-8") as f:
            self._kb = json.load(f)
        print(f"[Chunker] Loaded {len(self._kb)} records from {self.kb_path.name}")

    # ── Summary chunk ─────────────────────────────────────────────────────────

    def _build_summary_chunk(self, record: dict, chunk_total: int) -> Chunk:
        """
        1 per ingredient — structured info, không có description dài.

        Format:
            Ingredient: {name} | Categories: {cats} | Key Points: {kps} |
            Benefits: {benefits} | Safety: {label} | Concern Group: {group}
        """
        parts = [f"Ingredient: {record['name']}"]

        if record.get("categories"):
            parts.append("Categories: " + ", ".join(record["categories"]))

        if record.get("key_points"):
            parts.append("Key Points: " + ". ".join(record["key_points"]))

        if record.get("benefits"):
            parts.append("Benefits: " + ", ".join(record["benefits"]))

        if record.get("safety_label"):
            parts.append(f"Safety: {record['safety_label']}")

        if record.get("concern_group"):
            group_label = {
                "no_concern":        "No Concerns",
                "worth_knowing":     "Worth Knowing",
                "potential_concern": "Potential Concern",
            }.get(record["concern_group"], "")
            parts.append(f"Concern Group: {group_label}")

        return Chunk(
            chunk_text      = " | ".join(parts),
            ingredient_name = record["name"],
            chunk_type      = "summary",
            chunk_index     = 0,
            chunk_total     = chunk_total,
            concern_group   = record.get("concern_group", ""),
            safety_label    = record.get("safety_label", ""),
            safety_score    = record.get("safety_score", 0),
            categories      = record.get("categories", []),
            benefits        = record.get("benefits", []),
            warnings        = record.get("warnings", []),
            source_url      = record.get("source_url", ""),
            date_modified   = record.get("date_modified", ""),
        )

    # ── Detail chunks ─────────────────────────────────────────────────────────

    def _split_description(self, description: str) -> list[str]:
        """
        Split description tại sentence boundary gần nhất với chunk_target.
        Thêm overlap từ chunk trước vào đầu chunk hiếu tiếp theo.

        Rules:
        - Tách tại ". " (end of sentence)
        - Nếu không tìm được sentence boundary → tách tại space
        - Mỗi chunk >= 100 chars (tránh chunk quá nhỏ)
        """
        if len(description) <= self.chunk_target:
            return [description]

        chunks = []
        start  = 0
        text   = description

        while start < len(text):
            end = start + self.chunk_target

            if end >= len(text):
                # Last piece
                piece = text[start:].strip()
                if piece:
                    chunks.append(piece)
                break

            # Tìm sentence boundary gần end nhất (tìm ngược)
            boundary = text.rfind(". ", start, end)

            if boundary != -1 and boundary > start + 100:
                # Split sau dấu chấm (include the period)
                split_at = boundary + 1
            else:
                # Fallback: tách tại space
                boundary = text.rfind(" ", start, end)
                split_at = boundary if boundary > start + 100 else end

            piece = text[start:split_at].strip()
            if piece:
                chunks.append(piece)

            # Next start = split_at - overlap (giữ context)
            start = max(split_at - self.chunk_overlap, split_at)

        return chunks

    def _build_detail_chunks(
        self,
        record: dict,
        summary_chunk_total: int,
    ) -> list[Chunk]:
        """
        Split description thành 1+ detail chunks.
        Mỗi chunk có prefix: "Ingredient: {name} [detail {i}/{total}]"
        """
        description = record.get("description", "")
        if not description or len(description) <= self.detail_threshold:
            return []

        pieces = self._split_description(description)
        total  = len(pieces)

        detail_chunks = []
        for i, piece in enumerate(pieces, start=1):
            prefix    = f"Ingredient: {record['name']} [detail {i}/{total}]"
            chunk_text = f"{prefix} | {piece}"

            detail_chunks.append(Chunk(
                chunk_text      = chunk_text,
                ingredient_name = record["name"],
                chunk_type      = "detail",
                chunk_index     = i,                # 1-indexed
                chunk_total     = summary_chunk_total,
                concern_group   = record.get("concern_group", ""),
                safety_label    = record.get("safety_label", ""),
                safety_score    = record.get("safety_score", 0),
                categories      = record.get("categories", []),
                benefits        = record.get("benefits", []),
                warnings        = record.get("warnings", []),
                source_url      = record.get("source_url", ""),
                date_modified   = record.get("date_modified", ""),
            ))

        return detail_chunks

    # ── Main build ────────────────────────────────────────────────────────────

    def build(self, record: dict) -> list[Chunk]:
        """Build tất cả chunks cho 1 ingredient record."""
        detail_pieces = self._split_description(record.get("description", "")) \
            if len(record.get("description", "")) > self.detail_threshold else []

        # chunk_total = 1 (summary) + n (detail)
        chunk_total = 1 + len(detail_pieces)

        chunks = []
        chunks.append(self._build_summary_chunk(record, chunk_total))
        chunks.extend(self._build_detail_chunks(record, chunk_total))

        return chunks

    def build_all(self) -> list[Chunk]:
        """Build chunks cho toàn bộ KB."""
        all_chunks = []
        for record in self._kb:
            all_chunks.extend(self.build(record))

        # Stats
        summary_count = sum(1 for c in all_chunks if c.chunk_type == "summary")
        detail_count  = sum(1 for c in all_chunks if c.chunk_type == "detail")
        print(f"[Chunker] Built {len(all_chunks)} chunks total")
        print(f"          summary : {summary_count}")
        print(f"          detail  : {detail_count}")
        print(f"          avg chunk_text length: {sum(len(c.chunk_text) for c in all_chunks) // len(all_chunks)} chars")

        return all_chunks


# ── Post-retrieval dedup (dùng trong RAG pipeline) ────────────────────────────

def dedup_retrieved_chunks(chunks: list[dict]) -> list[dict]:
    """
    Sau khi Weaviate trả về top-k chunks, dedup theo ingredient_name.
    Nếu cùng 1 ingredient có cả summary + detail → ưu tiên giữ detail.

    Args:
        chunks: list of chunk dicts từ Weaviate response

    Returns:
        Deduplicated list
    """
    seen: dict[str, dict] = {}

    for chunk in chunks:
        name = chunk.get("ingredient_name", "")
        if name not in seen:
            seen[name] = chunk
        elif chunk.get("chunk_type") == "detail":
            # Detail chunk có description đầy đủ hơn summary
            seen[name] = chunk

    return list(seen.values())


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build chunks from cleaned KB")
    parser.add_argument("kb_path",           help="Path to paula_choice_cleaned.json")
    parser.add_argument("--output", "-o",    help="Output path for chunks JSON", default="chunks.json")
    parser.add_argument("--preview", "-p",   help="Ingredient name to preview chunks for")
    parser.add_argument("--threshold", "-t", type=int, default=DETAIL_THRESHOLD,
                        help=f"Detail chunk threshold (default: {DETAIL_THRESHOLD})")
    args = parser.parse_args()

    chunker = Chunker(
        kb_path          = args.kb_path,
        detail_threshold = args.threshold,
    )

    # Preview mode
    if args.preview:
        # Find record
        record = next(
            (r for r in chunker._kb if r["name"].lower() == args.preview.lower()),
            None
        )
        if not record:
            print(f"Ingredient '{args.preview}' not found in KB")
        else:
            chunks = chunker.build(record)
            print(f"\n{'='*60}")
            print(f"Preview: {record['name']} → {len(chunks)} chunks")
            print(f"{'='*60}")
            for chunk in chunks:
                print(f"\n[{chunk.chunk_type.upper()} | index={chunk.chunk_index} | total={chunk.chunk_total}]")
                print(f"Length: {len(chunk.chunk_text)} chars")
                print(f"Text:\n{chunk.chunk_text}")
        import sys; sys.exit(0)

    # Build all + export
    all_chunks = chunker.build_all()

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in all_chunks], f, ensure_ascii=False, indent=2)

    print(f"\n✅ Exported {len(all_chunks)} chunks → {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
