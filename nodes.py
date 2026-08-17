from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools    # from react.py

load_dotenv()


# reasoning node
SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node
    """
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_MESSAGE}, 
        *state["messages"]
    ])
    return {"messages": [response]}


# tool node
tool_node = ToolNode(tools)
