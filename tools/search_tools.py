from __future__ import annotations

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

_ddg = DuckDuckGoSearchRun()
_wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))


@tool
def web_search(query: str) -> str:
    """Search the live web for a query. Use for current events, statistics, or anything time-sensitive."""
    try:
        return _ddg.invoke(query)
    except Exception as e:
        return f"web_search failed ({type(e).__name__}): {e}. Try a different tool or a rephrased query."


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a query. Use for stable, encyclopedic facts like biographies, historical events, and definitions."""
    try:
        return _wiki.invoke(query)
    except Exception as e:
        return f"wikipedia_search failed ({type(e).__name__}): {e}. Try web_search instead."


@tool
def fetch_url(url: str) -> str:
    """Fetch and return the text content of a specific web page URL, e.g. one found via search."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs)[:5000]
    except Exception as e:
        return f"fetch_url failed ({type(e).__name__}): {e}. Try a different source."