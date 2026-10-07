import streamlit as st
import re
import sys

from rag_pipeline import (
    TOP_K,
    build_evidence,
    generate_with_ollama,
    load_retrieval_resources,
    retrieve,
    retrieval_confidence,
    source_citation,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AUTOSAR HLD Insight",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# LOAD MODELS / DATABASE
# ============================================================

@st.cache_resource
def load_resources():
    return load_retrieval_resources()


embedding_model, collection = load_resources()


def select_example_question(question):
    st.session_state["question_input"] = question
    st.session_state["selected_example"] = question


def evidence_excerpt(document, query, max_chars=420):
    text = " ".join(document.split())

    if len(text) <= max_chars:
        return text

    stop_words = {
        "a", "an", "and", "are", "for", "from", "how", "is", "of",
        "the", "to", "what", "which", "who", "with",
    }
    query_terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", query)
        if len(term) > 2 and term.lower() not in stop_words
    ]

    term_variants = {
        "provides": ("provide", "provides", "provider"),
        "provider": ("provide", "provides", "provider"),
        "consumes": ("consume", "consumes", "consumer"),
        "consumer": ("consume", "consumes", "consumer"),
    }

    candidate_starts = {0}
    for term in query_terms:
        variants = term_variants.get(term, (term,))
        for variant in variants:
            for match in re.finditer(re.escape(variant), text, re.IGNORECASE):
                candidate_starts.add(
                    max(
                        0,
                        min(match.start() - max_chars // 3, len(text) - max_chars)
                    )
                )

    best_start = 0
    best_score = -1

    for start in candidate_starts:
        excerpt = text[start:start + max_chars].lower()
        score = sum(
            max(excerpt.count(variant) for variant in term_variants.get(term, (term,)))
            for term in query_terms
        )
        if score > best_score:
            best_start = start
            best_score = score

    excerpt = text[best_start:best_start + max_chars].strip()

    if best_start > 0:
        excerpt = "..." + excerpt
    if best_start + max_chars < len(text):
        excerpt += "..."

    return excerpt


# ============================================================
# HEADER
# ============================================================

st.title("🚗 AUTOSAR HLD Insight")

st.subheader(
    "AI-Assisted Architecture Analysis and Knowledge Retrieval System"
)

st.markdown(
    """
Ask questions about the **Vehicle Gateway Control System AUTOSAR
High-Level Design** and receive grounded answers based only on the
retrieved HLD evidence.
"""
)

st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown("### 🔎 Ask the HLD")

query = st.text_input(
    "Enter your question:",
    placeholder="e.g. Which component provides VehicleSpeed?",
    key="question_input"
)

st.markdown("#### Try asking:")

example_questions = [
    "Which component provides VehicleSpeed?",
    "Which component consumes BrakeStatus?",
    "What interface connects SensorInterface and VehicleControlComponent?",
    "Which communication protocol is primary for the explicit HLD signals?",
    "Explain the primary control data flow.",
]

example_columns = st.columns(len(example_questions))
for index, (column, example_question) in enumerate(
    zip(example_columns, example_questions)
):
    column.button(
        example_question,
        key=f"example_{index}",
        on_click=select_example_question,
        args=(example_question,),
        use_container_width=True
    )

analyze = st.button(
    "Analyze",
    type="primary",
    use_container_width=True
)
selected_example = st.session_state.pop("selected_example", None)


# ============================================================
# ANALYSIS
# ============================================================

if analyze or selected_example:

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Analyzing the AUTOSAR HLD..."):

            retrieved = retrieve(
                query,
                embedding_model,
                collection,
                n_results=TOP_K,
            )
            documents = retrieved["documents"]
            metadatas = retrieved["metadatas"]
            best_similarity = retrieved["best_similarity"]
            confidence = retrieval_confidence(best_similarity)
            evidence = build_evidence(documents, metadatas)

            # ------------------------------------------------
            # 7. Try Ollama
            # ------------------------------------------------

            try:

                answer = generate_with_ollama(query, evidence)
                response_mode = "LLM-generated"

            except Exception as error:
                print(f"Ollama generation failed: {error}", file=sys.stderr)

                answer = (
                    "LLM unavailable — showing retrieved HLD evidence."
                )

                response_mode = "Evidence fallback"


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.divider()

        st.markdown("### Answer")

        with st.container(border=True):
            st.markdown(answer)


        # ====================================================
        # METRICS
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Retrieval Confidence")
            st.markdown(f"**Retrieval Confidence: {confidence}**")
            st.caption(f"Similarity score: {best_similarity:.2f}")


        with col2:

            st.markdown("### ⚙️ Response Mode")

            if response_mode == "LLM-generated":

                st.success(
                    "🤖 LLM-generated"
                )

            else:

                st.warning(
                    "📄 Evidence fallback"
                )


        # ====================================================
        # SOURCES
        # ====================================================

        st.divider()

        st.markdown("### Sources & Evidence")

        for i in range(len(metadatas)):

            with st.expander(
                f"Source {i + 1}",
                expanded=response_mode == "Evidence fallback"
            ):
                st.markdown(f"**Page:** {metadatas[i]['page']}")
                st.markdown(
                    f"**Section:** {metadatas[i].get('section', 'N/A')}"
                )
                st.markdown("**Evidence:**")
                if response_mode == "Evidence fallback":
                    st.write(documents[i])
                else:
                    st.write(evidence_excerpt(documents[i], query))


        # ====================================================
        # TECHNICAL NOTE
        # ====================================================

        st.divider()

        st.caption(
            "This is a retrieval similarity indicator, not a probability "
            "of answer correctness."
        )