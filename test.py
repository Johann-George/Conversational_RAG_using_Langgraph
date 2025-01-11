import streamlit as st
from langchain_core.messages import HumanMessage
from streamlit_chat import message
from main import graph

# Specify an ID for the thread
config = {"configurable": {"thread_id": "5"}}

st.title("MBCET Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the college"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": f"{prompt}"})

    with st.chat_message("assistant"):
        messages = [HumanMessage(content=f"{prompt}")]
        response = graph.invoke(
            {"messages": messages},
            stream_mode="values",
            config=config
        )
        st.markdown(response['messages'][-1].content)

    st.session_state.messages.append({"role": "assistant", "content": f"{response['messages'][-1].content}"})
