"""
app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit UI với 2 tabs:
  Tab 1 — Product Scan  : paste ingredient list → concern_classifier
  Tab 2 — Ask AI        : chat Q&A → RAG pipeline

Run:
    python -m streamlit run app.py
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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Group header pills ── */
    .group-header-safe {
        display: flex; align-items: center; gap: 10px;
        background: #f0fdf4; border: 1px solid #bbf7d0;
        border-radius: 12px; padding: 10px 16px; margin: 16px 0 6px 0;
    }
    .group-header-warn {
        display: flex; align-items: center; gap: 10px;
        background: #fffbeb; border: 1px solid #fde68a;
        border-radius: 12px; padding: 10px 16px; margin: 16px 0 6px 0;
    }
    .group-header-alert {
        display: flex; align-items: center; gap: 10px;
        background: #fef2f2; border: 1px solid #fecaca;
        border-radius: 12px; padding: 10px 16px; margin: 16px 0 6px 0;
    }
    .group-title-safe  { font-weight: 700; font-size: 16px; color: #15803d; letter-spacing: 0.05em; }
    .group-title-warn  { font-weight: 700; font-size: 16px; color: #b45309; letter-spacing: 0.05em; }
    .group-title-alert { font-weight: 700; font-size: 16px; color: #dc2626; letter-spacing: 0.05em; }
    .group-count {
        margin-left: auto;
        font-size: 12px; font-weight: 600;
        background: white; border-radius: 20px;
        padding: 2px 10px;
    }
    .count-safe  { color: #16a34a; border: 1px solid #bbf7d0; }
    .count-warn  { color: #b45309; border: 1px solid #fde68a; }
    .count-alert { color: #dc2626; border: 1px solid #fecaca; }

    /* ── Ingredient row ── */
    .ing-safety {
        font-size: 11px; font-weight: 600;
        padding: 2px 8px; border-radius: 20px; margin-left: 8px;
    }
    .safety-safe  { background: #f0fdf4; color: #16a34a; }
    .safety-warn  { background: #fffbeb; color: #b45309; }
    .safety-alert { background: #fef2f2; color: #dc2626; }

    /* ── Warning banner ── */
    .warning-banner {
        background: #fef2f2; border-left: 4px solid #ef4444;
        border-radius: 8px; padding: 12px 16px; margin: 10px 0;
        font-size: 13px; color: #7f1d1d;
    }

    /* ── Summary cards ── */
    .summary-row {
        display: flex; gap: 10px; margin-bottom: 16px;
    }
    .summary-card {
        flex: 1; text-align: center; border-radius: 12px;
        padding: 12px 8px; border: 1px solid;
    }
    .summary-card .num { font-size: 24px; font-weight: 700; }
    .summary-card .lbl { font-size: 11px; font-weight: 500; margin-top: 2px; }
    .sc-safe  { background: #f0fdf4; border-color: #bbf7d0; color: #16a34a; }
    .sc-warn  { background: #fffbeb; border-color: #fde68a; color: #b45309; }
    .sc-alert { background: #fef2f2; border-color: #fecaca; color: #dc2626; }
    .sc-total { background: #f8fafc; border-color: #e2e8f0; color: #475569; }

    /* ── Not found ── */
    .not-found-box {
        background: #f8fafc; border: 1px dashed #cbd5e1;
        border-radius: 10px; padding: 10px 14px; margin-top: 8px;
        font-size: 12px; color: #94a3b8;
    }

    /* ── Source tag ── */
    .source-tag {
        font-family: 'DM Mono', monospace;
        font-size: 11px; color: #94a3b8; margin-top: 6px;
    }

    /* Chat */
    .chat-meta {
        font-size: 11px; color: #94a3b8;
        font-family: 'DM Mono', monospace; margin-top: 4px;
    }

    /* ── Tab labels ── */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] > div > p {
        font-size: 18px !important;
        font-weight: 700 !important;
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
    if "scan_context" not in st.session_state:
        st.session_state.scan_context = None

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

@st.cache_resource(show_spinner="Loading OCR pipeline...")
def load_ocr_pipeline():
    from ocr_pipeline import OCRPipeline
    yolo_path = Path("models/best.pt")
    return OCRPipeline(yolo_path if yolo_path.exists() else None)


# ── Scan context helpers ───────────────────────────────────────────────────────

def save_scan_context(result_dict: dict):
    groups = result_dict.get("groups", {})
    st.session_state.scan_context = {
        "safe":      [i["name"] for i in groups.get("no_concern", [])],
        "concerns":  [i["name"] for i in groups.get("worth_knowing", [])],
        "alerts":    [i["name"] for i in groups.get("potential_concern", [])],
        "not_found": result_dict.get("not_found", []),
    }

def build_scan_context_prompt() -> str:
    ctx = st.session_state.get("scan_context")
    if not ctx:
        return ""
    parts = ["[Scanned Product Context]"]
    if ctx["safe"]:
        parts.append(f"Safe ingredients: {', '.join(ctx['safe'])}")
    if ctx["concerns"]:
        parts.append(f"Worth knowing: {', '.join(ctx['concerns'])}")
    if ctx["alerts"]:
        parts.append(f"Alert ingredients: {', '.join(ctx['alerts'])}")
    if ctx["not_found"]:
        parts.append(f"Not in KB: {', '.join(ctx['not_found'])}")
    return "\n".join(parts)


# ── Helper: render ingredient card ───────────────────────────────────────────

def render_ingredient_card(ing: dict):
    group = ing.get("concern_group", "no_concern")
    safety_cls   = {"no_concern": "safety-safe", "worth_knowing": "safety-warn", "potential_concern": "safety-alert"}
    safety_label = ing.get("safety_label", "")
    name         = ing.get("matched_name", ing.get("name", ""))
    scls         = safety_cls.get(group, "safety-safe")
    match_tag    = f" `[{ing.get('match_type','')}]`" if ing.get("match_type") != "exact" else ""

    with st.expander(f"**{name}**{match_tag}"):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                f'<span class="ing-safety {scls}">{safety_label}</span>',
                unsafe_allow_html=True,
            )
            if ing.get("description"):
                desc = ing["description"][:400] + ("…" if len(ing.get("description","")) > 400 else "")
                st.markdown(f'<div style="font-size:13px;color:#475569;margin-top:6px;line-height:1.5">{desc}</div>', unsafe_allow_html=True)
            if ing.get("key_points"):
                st.markdown("**Key Points:**")
                for kp in ing["key_points"][:3]:
                    st.markdown(f"- {kp}")
            if ing.get("warnings"):
                for w in ing["warnings"]:
                    st.markdown(f'<div class="warning-banner">⚑ {w}</div>', unsafe_allow_html=True)

        with col2:
            if ing.get("categories"):
                st.markdown("**Categories**")
                for cat in ing["categories"][:4]:
                    st.markdown(f'`{cat}`')

        st.markdown('<p class="source-tag">Source: Paula\'s Choice Ingredient Dictionary</p>', unsafe_allow_html=True)


# ── Helper: render concern groups ─────────────────────────────────────────────

def render_groups(result_dict: dict):
    groups  = result_dict.get("groups", {})
    summary = result_dict.get("summary", {})

    n_safe  = len(groups.get("no_concern", []))
    n_warn  = len(groups.get("worth_knowing", []))
    n_alert = len(groups.get("potential_concern", []))
    n_total = summary.get("total_detected", n_safe + n_warn + n_alert)

    st.markdown(f"""
    <div class="summary-row">
        <div class="summary-card sc-total"><div class="num">{n_total}</div><div class="lbl">Detected</div></div>
        <div class="summary-card sc-safe"><div class="num">{n_safe}</div><div class="lbl">Safe</div></div>
        <div class="summary-card sc-warn"><div class="num">{n_warn}</div><div class="lbl">Worth Knowing</div></div>
        <div class="summary-card sc-alert"><div class="num">{n_alert}</div><div class="lbl">Alert</div></div>
    </div>
    """, unsafe_allow_html=True)

    warnings = result_dict.get("all_warnings", [])
    if warnings:
        for w in warnings:
            st.markdown(f'<div class="warning-banner">⚑ {w}</div>', unsafe_allow_html=True)

    group_config = [
        ("no_concern",        "SAFE INGREDIENTS",  "safe",  "✓"),
        ("worth_knowing",     "POTENTIAL CONCERNS", "warn",  "⚠"),
        ("potential_concern", "ALERT",              "alert", "✕"),
    ]

    for group_key, group_label, style, icon in group_config:
        items = groups.get(group_key, [])
        if not items:
            continue
        st.markdown(f"""
        <div class="group-header-{style}">
            <span class="group-title-{style}">{icon} {group_label}</span>
            <span class="group-count count-{style}">{len(items)}</span>
        </div>
        """, unsafe_allow_html=True)
        for ing in items:
            render_ingredient_card(ing)

    not_found = result_dict.get("not_found", [])
    if not_found:
        st.markdown(f"""
        <div class="not-found-box">
            <strong>❓ Not in Knowledge Base ({len(not_found)})</strong><br>
            {", ".join(not_found)}
        </div>
        """, unsafe_allow_html=True)


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
    st.markdown("### Let's Scan Ingredients!")

    scan_camera, scan_manual = st.tabs(["📷 Camera Scan", "📝 Manual Input"])

    with scan_camera:
        

        ocr = load_ocr_pipeline()
        if ocr.mode == "mock":
            st.warning("⚠️ Running in **mock mode** — Place `models/best.pt` to enable YOLO detection.", icon=None)
        else:
            st.success("✅ YOLO model loaded — ingredient label detection active.")

        uploaded_image = st.file_uploader(
            "Upload product label photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Take a photo with your phone and upload it here",
        )

        use_camera = st.toggle("Use webcam instead", value=False)
        if use_camera:
            uploaded_image = st.camera_input("Take a photo of the ingredient label")

        if uploaded_image:
            from PIL import Image as PILImage, ImageOps
            import io
            image = ImageOps.exif_transpose(PILImage.open(io.BytesIO(uploaded_image.getvalue())))

            col_preview, col_result = st.columns([1, 1], gap="large")

            with col_preview:
                with st.spinner("Detecting ingredient label..."):
                    classifier = load_classifier()
                    kb_names = [r["name"] for r in classifier._kb] if classifier else None
                    ocr_result = ocr.run(image, kb_names=kb_names)

                st.image(
                    ocr_result.preview_image,
                    caption=f"{'✅ Label detected' if ocr_result.bbox_detected else '⬜ Full image (no detection)'} — {ocr_result.latency_ms}ms",
                    use_container_width=True,
                )

                if ocr_result.raw_text:
                    with st.expander("Raw OCR text"):
                        st.code(ocr_result.raw_text, language=None)

                if ocr_result.error:
                    st.error(f"OCR error: {ocr_result.error}")

            with col_result:
                if ocr_result.ingredient_list:
                    st.markdown(f"**{len(ocr_result.ingredient_list)} ingredients detected**")
                    with st.spinner("Analyzing ingredients..."):
                        classifier = load_classifier()
                        if classifier is None:
                            st.error("`paula_choice_cleaned.json` not found.")
                        else:
                            result = classifier.classify(ocr_result.ingredient_list)
                            result_dict = result.to_dict()
                            save_scan_context(result_dict)
                            render_groups(result_dict)
                else:
                    st.warning("No ingredients detected. Try better lighting or use Manual Input.")

    with scan_manual:
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
                raw = ingredient_input.replace(",", "\n")
                ingredient_list = [line.strip() for line in raw.splitlines() if line.strip()]
                if not ingredient_list:
                    st.warning("No ingredients detected in input.")
                else:
                    with st.spinner(f"Analyzing {len(ingredient_list)} ingredients..."):
                        classifier = load_classifier()
                        if classifier is None:
                            st.error("`paula_choice_cleaned.json` not found.")
                        else:
                            result = classifier.classify(ingredient_list)
                            result_dict = result.to_dict()
                            save_scan_context(result_dict)
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
    st.markdown("### Ask About Ingredients!")
    st.caption("Ask anything about cosmetic ingredients — safety, benefits, comparisons.")

    with st.sidebar:
        # ── System status ─────────────────────────────────────
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

        with st.expander("📋 Product Scan"):
            st.markdown("""
1. **Camera Scan** — upload a photo of the ingredient label, or use your webcam
2. **Manual Input** — paste or type the ingredient list directly

Results are grouped into **Safe**, **Potential Concerns**, and **Alert** — tap any ingredient to learn more.
            """)

        with st.expander("💬 Ask AI"):
            st.markdown("""
Ask anything about ingredients:
- *"What does glycerin do for skin?"*
- *"Is phenoxyethanol safe for sensitive skin?"*
- *"What's the difference between AHA and BHA?"*

💡 **Tip:** Scan a product first — the AI will automatically use that product's ingredient list as context for your questions.
            """)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.markdown(f'<p class="chat-meta">{msg["meta"]}</p>', unsafe_allow_html=True)

    # Generate pending response
    if st.session_state.get("pending_response"):
        query = st.session_state.pop("pending_response")

        with st.chat_message("assistant"):
            try:
                pipeline = load_pipeline()

                if not pipeline.check_health()["ollama"]:
                    answer = "⚠️ Ollama is not running. Please start it with `ollama serve` in your terminal."
                    st.markdown(answer)
                    sources, qtype, latency = [], "", 0
                else:
                    t0           = time.time()
                    response_box = st.empty()
                    full_answer  = ""

                    # Inject scan context nếu có
                    scan_ctx = build_scan_context_prompt()
                    augmented = f"{scan_ctx}\n\nQuestion: {query}" if scan_ctx else query

                    for token in pipeline.query_stream(augmented):
                        full_answer += token
                        response_box.markdown(full_answer + "▌")
                    response_box.markdown(full_answer)

                    sources = pipeline.last_sources
                    qtype   = pipeline.last_query_type
                    latency = int((time.time() - t0) * 1000)
                    answer  = full_answer

                meta_parts = []
                if sources:
                    meta_parts.append(f"Sources: {', '.join(sources[:5])}")
                if qtype:
                    meta_parts.append(f"Query type: {qtype.replace('_', ' ')}")
                if latency:
                    meta_parts.append(f"{latency}ms")

                meta = "  ·  ".join(meta_parts)
                if meta:
                    st.markdown(f'<p class="chat-meta">{meta}</p>', unsafe_allow_html=True)

                st.session_state.chat_history.append({"role": "assistant", "content": answer, "meta": meta})

            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    # Chat input — inside tab2
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask about ingredients...") or pending
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.pending_response = user_input
        st.rerun()