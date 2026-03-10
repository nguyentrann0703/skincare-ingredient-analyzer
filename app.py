"""
app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit UI với 2 tabs:
  Tab 1 — Product Scan  : paste ingredient list → concern_classifier
  Tab 2 — Ask AI        : chat Q&A → RAG pipeline

Run:
    streamlit run app.py
"""

import streamlit as st
import time
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Ingredient Analyzer",
    page_icon  = "🧴",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main font */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Concern group badges */
    .badge-safe {
        background: #f0fdf4; color: #16a34a;
        border: 1px solid #bbf7d0;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 600;
    }
    .badge-warn {
        background: #fffbeb; color: #b45309;
        border: 1px solid #fde68a;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 600;
    }
    .badge-concern {
        background: #fef2f2; color: #dc2626;
        border: 1px solid #fecaca;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 600;
    }

    /* Source tag */
    .source-tag {
        font-family: 'DM Mono', monospace;
        font-size: 11px; color: #94a3b8;
    }

    /* Warning box */
    .warning-box {
        background: #fef2f2;
        border-left: 3px solid #ef4444;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 13px;
        color: #7f1d1d;
    }

    /* Chat message */
    .chat-meta {
        font-size: 11px; color: #94a3b8;
        font-family: 'DM Mono', monospace;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "classifier" not in st.session_state:
        st.session_state.classifier = None
    if "health" not in st.session_state:
        st.session_state.health = None

init_state()


# ── Load pipeline (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI pipeline...")
def load_pipeline():
    from rag_pipeline import RAGPipeline
    return RAGPipeline()

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_classifier():
    from concern_classifier import ConcernClassifier
    kb_path = Path("paula_choice_cleaned.json")
    if not kb_path.exists():
        return None
    return ConcernClassifier(str(kb_path))


# ── Helper: render ingredient card ───────────────────────────────────────────

def render_ingredient_card(ing: dict):
    """Render 1 ingredient row với expander."""
    group = ing.get("concern_group", "no_concern")

    icon_map  = {"no_concern": "✓", "worth_knowing": "⚠", "potential_concern": "✕"}
    badge_map = {"no_concern": "badge-safe", "worth_knowing": "badge-warn", "potential_concern": "badge-concern"}
    label_map = {"no_concern": "No Concerns", "worth_knowing": "Worth Knowing", "potential_concern": "Potential Concern"}

    icon        = icon_map.get(group, "?")
    badge_class = badge_map.get(group, "badge-safe")
    label       = label_map.get(group, group)
    match_tag   = f" `[{ing.get('match_type','')}]`" if ing.get("match_type") != "exact" else ""

    with st.expander(
        f"{icon}  **{ing.get('matched_name', ing.get('name', ''))}**{match_tag}"
    ):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                f'<span class="{badge_class}">{label}</span>&nbsp;&nbsp;'
                f'<span style="font-size:12px;color:#64748b">{ing.get("safety_label","")}</span>',
                unsafe_allow_html=True,
            )

            if ing.get("description"):
                st.markdown(f'<p style="font-size:13px;color:#334155;margin-top:8px">{ing["description"][:400]}{"..." if len(ing.get("description","")) > 400 else ""}</p>', unsafe_allow_html=True)

            if ing.get("key_points"):
                st.markdown("**Key Points:**")
                for kp in ing["key_points"][:3]:
                    st.markdown(f"- {kp}")

        with col2:
            if ing.get("categories"):
                st.markdown("**Categories**")
                for cat in ing["categories"][:4]:
                    st.markdown(f'`{cat}`')

            if ing.get("warnings"):
                for w in ing["warnings"]:
                    st.markdown(
                        f'<div class="warning-box">⚑ {w}</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown(
            '<p class="source-tag">Source: Paula\'s Choice Ingredient Dictionary</p>',
            unsafe_allow_html=True,
        )


# ── Helper: render concern groups ─────────────────────────────────────────────

def render_groups(result_dict: dict):
    """Render 3 concern groups từ ProductAnalysisResult.to_dict()."""
    groups = result_dict.get("groups", {})
    summary = result_dict.get("summary", {})

    # Summary bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Detected",  summary.get("total_detected", 0))
    col2.metric("✓ No Concerns",   len(groups.get("no_concern", [])),        delta=None)
    col3.metric("⚠ Worth Knowing", len(groups.get("worth_knowing", [])),     delta=None)
    col4.metric("✕ Concerns",      len(groups.get("potential_concern", [])), delta=None)

    st.divider()

    # Warnings box
    warnings = result_dict.get("all_warnings", [])
    if warnings:
        st.markdown("#### ⚑ Warnings")
        for w in warnings:
            st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)
        st.divider()

    # Ingredient groups
    st.markdown("#### Detected Ingredients")

    for group_key, group_label in [
        ("no_concern",        "✓ No Concerns"),
        ("worth_knowing",     "⚠ Worth Knowing"),
        ("potential_concern", "✕ Potential Concerns"),
    ]:
        items = groups.get(group_key, [])
        if not items:
            continue
        st.markdown(f"**{group_label}** ({len(items)})")
        for ing in items:
            render_ingredient_card(ing)

    # Not found
    not_found = result_dict.get("not_found", [])
    if not_found:
        st.markdown(f"**? Not in Knowledge Base** ({len(not_found)})")
        st.caption(", ".join(not_found))


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# 🧴 Ingredient Analyzer")
st.caption("Powered by RAG · Knowledge base: Paula's Choice Ingredient Dictionary")
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📋 Product Scan", "💬 Ask AI"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Product Scan
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Scan Ingredients")
    st.caption("Paste the ingredient list from a product label. One ingredient per line, or comma-separated.")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        ingredient_input = st.text_area(
            "Ingredient List",
            placeholder="Water\nGlycerin\nNiacinamide\nPhenoxyethanol\nFragrance",
            height=200,
        )

        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    with col_result:
        if analyze_btn and ingredient_input.strip():
            # Parse input — support cả newline và comma
            raw = ingredient_input.replace(",", "\n")
            ingredient_list = [
                line.strip() for line in raw.splitlines()
                if line.strip()
            ]

            if not ingredient_list:
                st.warning("No ingredients detected in input.")
            else:
                with st.spinner(f"Analyzing {len(ingredient_list)} ingredients..."):
                    classifier = load_classifier()

                    if classifier is None:
                        st.error("`paula_choice_cleaned.json` not found. Place it in the same folder as app.py.")
                    else:
                        result = classifier.classify(ingredient_list)
                        result_dict = result.to_dict()
                        render_groups(result_dict)

        elif analyze_btn:
            st.info("Please enter some ingredients first.")

        else:
            st.markdown("""
            **How to use:**
            1. Copy the ingredient list from the product label
            2. Paste it in the text area
            3. Click **Analyze**

            **What you'll get:**
            - Ingredients grouped by concern level
            - Safety information per ingredient
            - Warnings for known irritants and allergens

            *Data source: Paula's Choice Ingredient Dictionary*
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Ask AI (RAG)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Ask About Ingredients")
    st.caption("Ask anything about cosmetic ingredients — safety, benefits, comparisons.")

    # Health check sidebar info
    with st.sidebar:
        st.markdown("### System Status")
        if st.button("Check Status", use_container_width=True):
            with st.spinner("Checking..."):
                try:
                    pipeline = load_pipeline()
                    health   = pipeline.check_health()
                    st.session_state.health = health
                except Exception as e:
                    st.session_state.health = {"error": str(e)}

        health = st.session_state.health
        if health:
            if "error" in health:
                st.error(f"Pipeline error: {health['error']}")
            else:
                st.markdown(f"**Weaviate:** {'🟢' if health['weaviate'] else '🔴'}")
                st.markdown(f"**Ollama:** {'🟢' if health['ollama'] else '🔴'}")
                st.markdown(f"**Model:** `{health.get('model','')}`")

        st.divider()
        st.markdown("**Example queries:**")
        examples = [
            "What does niacinamide do for skin?",
            "Is phenoxyethanol safe?",
            "What humectants are safe for sensitive skin?",
            "Why is fragrance concerning in skincare?",
            "What are the best antioxidants for anti-aging?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
                st.session_state.pending_query = ex

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Handle example button click — inject vào input trước khi render
    pending = st.session_state.pop("pending_query", None)

    # Chat input — luôn ở bottom
    user_input = st.chat_input("Ask about ingredients...") or pending

    # Nếu có input mới → save vào session state + rerun
    # Sau rerun, history đã có user message → generate response
    if user_input:
        st.session_state.chat_history.append({
            "role":    "user",
            "content": user_input,
        })
        st.session_state.pending_response = user_input
        st.rerun()

    # Render toàn bộ chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.markdown(
                    f'<p class="chat-meta">{msg["meta"]}</p>',
                    unsafe_allow_html=True,
                )

    # Generate response nếu có pending
    if st.session_state.get("pending_response"):
        query = st.session_state.pop("pending_response")

        with st.chat_message("assistant"):
            try:
                pipeline = load_pipeline()

                if not pipeline.check_health()["ollama"]:
                    answer = "⚠️ Ollama is not running. Please start it with `ollama serve` in your terminal."
                    st.markdown(answer)
                    sources = []
                    qtype   = ""
                    latency = 0
                else:
                    t0           = time.time()
                    response_box = st.empty()
                    full_answer  = ""

                    for token in pipeline.query_stream(query):
                        full_answer += token
                        response_box.markdown(full_answer + "▌")

                    response_box.markdown(full_answer)

                    sources = pipeline.last_sources
                    qtype   = pipeline.last_query_type
                    latency = int((time.time() - t0) * 1000)
                    answer  = full_answer

                # Meta info
                meta_parts = []
                if sources:
                    meta_parts.append(f"Sources: {', '.join(sources[:5])}")
                if qtype:
                    meta_parts.append(f"Query type: {qtype.replace('_', ' ')}")
                if latency:
                    meta_parts.append(f"{latency}ms")

                meta = "  ·  ".join(meta_parts)
                if meta:
                    st.markdown(
                        f'<p class="chat-meta">{meta}</p>',
                        unsafe_allow_html=True,
                    )

                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": answer,
                    "meta":    meta,
                })

            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": error_msg,
                })