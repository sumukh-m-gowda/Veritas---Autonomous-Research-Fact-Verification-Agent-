from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import embeddings

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def chunk_and_embed(raw_text: str, source: str, url: str, query: str, k: int = 3) -> list[dict]:
    """Chunk raw_text, embed into an in-memory FAISS index, return top-k chunks relevant to query."""
    docs = splitter.create_documents([raw_text], metadatas=[{"source": source, "url": url}])
    if not docs:
        return []

    vector_store = FAISS.from_documents(docs, embeddings)
    top_docs = vector_store.similarity_search(query, k=k)

    return [
        {"source": source, "url": url, "content": d.page_content, "relevance_score": None}
        for d in top_docs
    ]