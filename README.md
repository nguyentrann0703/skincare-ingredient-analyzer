# 🧴 Cosmetic Ingredient Analyzer

An AI-powered pipeline that analyzes cosmetic product ingredients — classifying safety concerns and answering ingredient questions via RAG (Retrieval-Augmented Generation).

---

## Overview

This project has two core runtime pipelines:

| Pipeline | Input | Output |
|---|---|---|
| **Luồng 1 — Concern Classifier** | Ingredient list (from OCR or manual input) | Grouped safety analysis (no concern / worth knowing / potential concern) |
| **Luồng 2 — RAG Q&A** | Natural language question | LLM-generated answer grounded in ingredient KB |

Knowledge base: **Paula's Choice Ingredient Dictionary** (2,509 ingredients)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                │
│              Tab 1: Product Scan │ Tab 2: Ask AI          │
└──────────────────┬──────────────────────┬────────────────┘
                   │                      │
                   ▼                      ▼
      ┌────────────────────┐   ┌────────────────────────┐
      │  concern_classifier│   │     rag_pipeline.py     │
      │       .py          │   │                         │
      │                    │   │  QueryClassifier        │
      │  Exact match       │   │  → ingredient_specific  │
      │  Fuzzy match       │   │  → open_ended           │
      │  Token Jaccard     │   │                         │
      └────────┬───────────┘   │  IngredientRetriever    │
               │               │  → Weaviate hybrid      │
               │               │    search (BM25+vector) │
               │               │                         │
               │               │  OllamaLLM              │
               │               │  → Qwen2.5:7b           │
               │               └──────────┬──────────────┘
               │                          │
               └──────────┬───────────────┘
                          ▼
              ┌───────────────────────┐
              │  paula_choice_        │
              │  cleaned.json         │
              │  (Knowledge Base)     │
              └───────────────────────┘
```

---

## Project Structure

```
paula-scraper/
│
├── 📊 Data
│   ├── paula_choice_full_details.json   # Raw scraped data (2,533 records)
│   └── paula_choice_cleaned.json        # Cleaned KB (2,509 records) ← main input
│
├── 🧹 Data Pipeline
│   └── paula_choice_cleaning.ipynb      # Jupyter notebook: clean → classify → export
│
├── 🔍 Concern Classifier (Luồng 1)
│   └── concern_classifier.py            # Exact/fuzzy ingredient lookup, concern grouping
│
├── 🗄️ Vector DB Pipeline
│   ├── chunker.py                       # Split KB records → summary + detail chunks
│   ├── weaviate_ingest.py               # Embed (BAAI/bge-base-en-v1.5) + ingest to Weaviate
│   └── chunks.json                      # Pre-built chunks (generated, not committed)
│
├── 🤖 RAG Pipeline (Luồng 2)
│   └── rag_pipeline.py                  # QueryClassifier + Retriever + OllamaLLM
│
├── 🖥️ App
│   └── app.py                           # Streamlit UI (2 tabs)
│
├── 🐳 Infrastructure
│   └── docker-compose.yml               # Weaviate local instance
│
└── 📦 Dependencies
    └── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Qwen2.5:7b via Ollama (local) |
| Embedding | BAAI/bge-base-en-v1.5 via sentence-transformers |
| Vector DB | Weaviate 1.27 (Docker, local) |
| Search | Hybrid: BM25 + cosine vector (HNSW) |
| Language | Python 3.12 |

---

## Setup

### Prerequisites

- Mac (Apple Silicon recommended) / Linux
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/download)
- Conda

### 1. Create environment

```bash
conda create -n paula python=3.12 -y
conda activate paula
python -m pip install -r requirements.txt
```

### 2. Pull LLM model

```bash
ollama pull qwen2.5:7b
```

### 3. Start Weaviate

```bash
docker compose up -d
```

### 4. Ingest knowledge base

```bash
python weaviate_ingest.py \
    --kb    paula_choice_cleaned.json \
    --reset \
    --stats \
    --save-chunks chunks.json
```

First run will download BAAI/bge-base-en-v1.5 (~440MB). Embedding 7,459 chunks takes ~30–60s on Apple Silicon MPS.

### 5. Start Ollama server

```bash
# Run in a separate terminal
ollama serve
```

### 6. Run app

```bash
conda activate paula
python -m streamlit run app.py
```

Open http://localhost:8501

---

## Key Design Decisions

### Chunking Strategy

Each ingredient produces 2 chunk types:

| Type | Count | Content | Purpose |
|---|---|---|---|
| `summary` | 1 per ingredient | name + categories + key_points + safety | Quick lookup, filtering |
| `detail` | 0–n per ingredient | split description (400 char target, 50 overlap) | Deep Q&A |

Total: **7,459 chunks** from 2,509 ingredients.

### Hybrid Search

```
alpha = 0.5  →  BM25 (keyword) + cosine vector (semantic) balanced
```

Metadata filters available: `concern_group`, `categories`, `safety_score`, `chunk_type`

### Query Classification

Rule-based classifier (no LLM overhead):
- **ingredient_specific**: query contains a known ingredient name → direct lookup
- **open_ended**: everything else → hybrid search with extracted filters

---

## Concern Groups

| Group | Criteria | Examples |
|---|---|---|
| ✓ No Concerns | rating Best/Good, no override categories | Niacinamide, Glycerin |
| ⚠ Worth Knowing | rating Average, or Preservative/Fragrance category | Phenoxyethanol, Parfum |
| ✕ Potential Concern | rating Bad/Worst, or Irritant category | Formaldehyde, Alcohol Denat |

---

## Usage

### Tab 1 — Product Scan

Paste an ingredient list (newline or comma-separated):

```
Water, Glycerin, Niacinamide, Phenoxyethanol, Fragrance
```

Returns grouped concern analysis with safety details per ingredient.

### Tab 2 — Ask AI

Example queries:
- `What does niacinamide do for skin?`
- `Is phenoxyethanol safe to use?`
- `What humectants are safe for sensitive skin?`
- `Why is fragrance concerning in skincare?`

---

## Data Source

**Paula's Choice Ingredient Dictionary** — scraped from [paulaschoice.com](https://www.paulaschoice.com/ingredient-dictionary)

2,509 ingredients with: description, rating (Best/Good/Average/Bad/Worst), categories, key points, benefits, warnings, source URL.

---

## Roadmap

- [ ] OCR pipeline (YOLO + Tesseract) to scan physical product labels
- [ ] Multilingual support (Vietnamese queries)
- [ ] LLM parameter controls in UI (temperature, top-k)
- [ ] Product history / scan log
- [ ] Ingredient comparison mode
