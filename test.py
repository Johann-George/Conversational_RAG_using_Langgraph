from main import get_graph

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}
graph = get_graph()

while True:
    print("Enter the prompt:")
    input_message = str(input())
    for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config
    ):
        step["messages"][-1].pretty_print()
