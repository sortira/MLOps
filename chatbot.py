from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model

# --- Define State ---
class State(TypedDict):
    # messages is a list, updated by add_messages
    messages: Annotated[list, add_messages]

# Create graph builder
graph_builder = StateGraph(State)

# Load API key
with open("secret") as f:
    os.environ['GOOGLE_API_KEY'] = f.read().strip()

# Initialize model
llm = init_chat_model("google_genai:gemini-2.0-flash")

# Node function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile into runnable graph
graph = graph_builder.compile()

# Function to stream updates
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# CLI loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except EOFError:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
