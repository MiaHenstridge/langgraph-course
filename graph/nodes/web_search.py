from typing import Any, Dict

from langchain_core.documents import Document 
from langchain_tavily import TavilySearch

from graph.state import GraphState


# initialize web search tool
web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_result = "\n".join(
        [res["content"] for res in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)   # create a document from the search result
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}