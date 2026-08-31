from typing import Literal
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2


# -------- Nodes
def draft_node(state: MessagesState):
    """Draft the initial response"""
    # returns AnswerQuestion
    response = first_responder.invoke({"messages": state["messages"]})  # first response with human message as input
    return {"messages": [response]}     # append response to the graph state


def revise_node(state: MessagesState):
    """Revise the answer based on tool results"""
    # invoke revisor chain and invoke with all messages so far
    response = revisor.invoke({"messages": state["messages"]})
    return {"message": [response]}      # append response to the graph state


# --------- Event loop for tool execution
# either execute tool or end
def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count"""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


# -------- Graph
builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("revise", revise_node)
builder.add_node("execute_tools", execute_tools)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="reflexion_agent.png")


if __name__ == "__main__":
    print("Hello Reflexion Agent")