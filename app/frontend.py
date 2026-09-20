"""
app/frontend.py

Matches "Streamlit Frontend Pod / app/frontend.py / Deployment: rag-frontend"
in the diagram, reached through the K8s Service rag-frontend-service
(NodePort 30085) shown as http://localhost:30085 for the User/Browser.
"""
import logging

import requests
import streamlit as st

from app.config import settings

logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Demo", page_icon="🔎", layout="wide")
st.title("🔎 LangGraph RAG Agent — Demo")

tab_chat, tab_ingest = st.tabs(["Ask a question", "Ingest a document"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Ask something about your ingested documents:")
    if st.button("Ask", type="primary") and question:
        with st.spinner("Running the LangGraph agent..."):
            try:
                logger.info("Query submitted question=%r", question)
                resp = requests.post(
                    f"{settings.backend_url}/api/v1/advanced/query",
                    json={"question": question},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info("Query succeeded strategy=%s", data.get("strategy"))
                st.session_state.history.append((question, data))
            except Exception as e:  # noqa: BLE001
                logger.exception("Query request failed")
                st.error(f"Request failed: {e}")

    for q, data in reversed(st.session_state.history):
        with st.container(border=True):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {data['answer']}")
            st.caption(f"Strategy: `{data['strategy']}`")
            with st.expander("Contexts used"):
                for c in data.get("contexts_used", []):
                    st.write(c)

with tab_ingest:
    text = st.text_area("Paste text to ingest into pgvector:", height=200)
    source = st.text_input("Source label", value="manual-upload")
    if st.button("Ingest") and text:
        try:
            logger.info("Ingest submitted source=%r len=%d", source, len(text))
            resp = requests.post(
                f"{settings.backend_url}/api/v1/advanced/ingest-async",
                json={"text": text, "source": source},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info("Ingest queued task_id=%s", data.get("task_id"))
            st.success(f"Queued as Celery task {data['task_id']}")
        except Exception as e:  # noqa: BLE001
            logger.exception("Ingest request failed")
            st.error(f"Request failed: {e}")
