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
    return _ddg.invoke(query)


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a query. Use for stable, encyclopedic facts like biographies, historical events, and definitions."""
    return _wiki.invoke(query)


@tool
def fetch_url(url: str) -> str:
    """Fetch and return the text content of a specific web page URL, e.g. one found via search."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs)[:5000]