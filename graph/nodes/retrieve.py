from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve node which takes a GraphState 
    and outputs retrieved documents for the question"""
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
    }

