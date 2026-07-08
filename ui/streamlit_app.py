"""
Streamlit UI for the RAG app - lets you test /ask and /contradict without
Postman/curl.

Run: streamlit run ui/streamlit_app.py
(Requires the FastAPI backend to be running separately, default
http://127.0.0.1:8000.)
"""
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("RAG Document Q&A")

tab_ask, tab_contradict = st.tabs(["Ask a Question", "Compare Documents (Contradiction Check)"])


def _get_documents():
    try:
        resp = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()["documents"]
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        return []


with tab_ask:
    st.subheader("Ask a question about the ingested documents")

    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_area("Question", placeholder="e.g. How many remote work days are employees allowed?", height=80)
    with col2:
        top_k = st.slider("Top K", min_value=1, max_value=15, value=5)

    if st.button("Ask", type="primary", key="ask_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={"question": question, "top_k": top_k},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
                    data = None

            if data:
                st.markdown("### Answer")
                st.write(data["answer"])

                conf = data["confidence"]
                if data["low_confidence_warning"]:
                    st.warning(
                        f"Low confidence ({conf:.2f}). The retrieved context may not fully support this "
                        "answer - please verify against the source documents."
                    )
                else:
                    st.success(f"Confidence: {conf:.2f}")

                meta_col1, meta_col2 = st.columns(2)
                meta_col1.metric("Detected language", data["detected_language"])
                meta_col2.metric("Response time (s)", data["response_time_seconds"])

                st.markdown("### Citations")
                if data["citations"]:
                    for i, c in enumerate(data["citations"], start=1):
                        with st.expander(f"[{i}] {c['source']} - page {c['page']} (score: {c['relevance_score']:.2f})"):
                            st.caption(f"chunk_id: {c['chunk_id']}")
                            st.write(c["snippet"])
                else:
                    st.info("No citations - the documents did not contain relevant information.")

                with st.expander("Full retrieved context"):
                    for i, ctx in enumerate(data["retrieved_context"], start=1):
                        st.markdown(f"**[{i}]**")
                        st.write(ctx)

with tab_contradict:
    st.subheader("Compare two documents for contradictions")

    documents = _get_documents()
    if not documents:
        st.info("No documents ingested yet, or the API is unreachable. Run scripts/ingest.py and start the API.")
    else:
        doc_labels = {f"{d['doc_id']}  ({d['source']})": d["doc_id"] for d in documents}
        col1, col2 = st.columns(2)
        with col1:
            label_1 = st.selectbox("Document A", list(doc_labels.keys()), key="doc1")
        with col2:
            options_2 = [l for l in doc_labels.keys() if l != label_1] or list(doc_labels.keys())
            label_2 = st.selectbox("Document B", options_2, key="doc2")

        topic = st.text_input("Topic (optional)", placeholder="e.g. remote work days per week")
        top_k_c = st.slider("Top K per document", min_value=1, max_value=15, value=5, key="contradict_topk")

        if st.button("Compare", type="primary", key="contradict_btn"):
            with st.spinner("Retrieving relevant excerpts and analyzing..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/contradict",
                        json={
                            "doc_id_1": doc_labels[label_1],
                            "doc_id_2": doc_labels[label_2],
                            "topic": topic or None,
                            "top_k": top_k_c,
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
                    data = None

            if data:
                verdict = data["verdict"]
                verdict_display = {
                    "contradiction": ("Contradiction found", "error"),
                    "no_contradiction": ("No contradiction", "success"),
                    "insufficient_evidence": ("Insufficient evidence", "warning"),
                }
                text, kind = verdict_display.get(verdict, (verdict, "info"))
                getattr(st, kind)(text)

                st.markdown("### Reasoning")
                st.write(data["reasoning"])
                st.caption(f"Response time: {data['response_time_seconds']}s")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"#### Evidence from {doc_labels[label_1]}")
                    for c in data["evidence_doc_1"]:
                        with st.expander(f"{c['source']} - page {c['page']}"):
                            st.write(c["snippet"])
                with col_b:
                    st.markdown(f"#### Evidence from {doc_labels[label_2]}")
                    for c in data["evidence_doc_2"]:
                        with st.expander(f"{c['source']} - page {c['page']}"):
                            st.write(c["snippet"])
