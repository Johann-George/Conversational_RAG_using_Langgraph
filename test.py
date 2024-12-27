from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
import streamlit as st

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

state = MessagesState()

st.title("MBCET Chatbot")
user_input = st.text_input("Ask me anything about the college:")

if st.button("Submit"):
    for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config
    ):
        st.write(step["AI_messages"][-1].pretty_print())
