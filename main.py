from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain


# define state schema
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# define node names and functions/logics for reflection and generation
REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    """Run the generation chain and invoke it with all messages so far"""
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}    # treat the reflection output as human message

# build graph
builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)   # first node to be called


# helper function to define conditional edges
def should_continue(state: MessageGraph):
    """Either continue reflection or end"""
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT})    # conditional edge from generate to either end or reflection
builder.add_edge(REFLECT, GENERATE) # deterministic edge from reflect to generate

# compile and visualize graph
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflection_agent.png")


if __name__ == "__main__":
    print("Hello Reflection Agent")
    inputs = {
        "messages": [
            HumanMessage(
                content="""
        Make this LinkedIn post better:

        Fraud doesn't always look different — sometimes it hides in plain sight.

        For my graduation thesis, I built THeGCN, a Temporal Heterophily-aware Graph Convolutional Network for detecting fraudulent borrowers in financial networks.

        Two ideas drove the design:

        🕒 Temporal context — fraud rings evolve over time, so I used temporal edge sampling (via TGL) instead of treating the graph as static.

        🎭 Heterophily-awareness — fraudsters often "mask" themselves by connecting to legitimate users. A dual-filter aggregator (adaptive low-pass + high-pass) helps the model see through that camouflage.

        Benchmarked against ML baselines (Naïve Bayes, Linear SVM, XGBoost) and GNNs (GraphSAGE, GAT, GATv2, FAGCN, TGAT) on DGraphFin — a real-world graph with 3.7M nodes and a 1.27% fraud rate — THeGCN came out on top on AP, Recall@k, and Precision@k.

        Key takeaway: temporal signals mattered more than spatial structure alone, and heterophily-aware aggregation really shines once you add engineered interaction features.

        Code's on GitHub if you want to dig into the details 👇

        #GraphNeuralNetworks #FraudDetection #MachineLearning #GNN #DataScience

                """
            ),
        ],
    }

    responses = graph.invoke(inputs)
    print(responses)
