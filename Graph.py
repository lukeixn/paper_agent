# class Graph:
#     def __init__(self):
#         self.nodes = {}
#         self.edges = {}
#     def add_node(self, name:str, obj):
#         self.nodes[name] = obj
#     def add_edge(self, src, dst):
#         self.edges[src] = dst
#     def run(self,state,start):
#         current = start

#         while current:

#             print(f"\n执行节点: {current}")

#             node_func = self.nodes[current]

#             state = node_func(state)

#             current = self.edges.get(current)

#         return state
# def node_a(state):
#     state["count"] += 1
#     print("A执行")
#     return state
# def node_b(state):
#     state["count"] += 1
#     print("B执行")
#     return state
# graph = Graph()

# graph.add_node("A", node_a)
# graph.add_node("B", node_b)

# graph.add_edge("A", "B")
# state = {
#     "count": 0
# }

# result = graph.run(
#     state,
#     start="A"
# )

# print(result)

from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph import START
from langgraph.graph import END
from  langgraph.checkpoint.memory import MemorySaver
mem=MemorySaver()
class State(TypedDict):
    count: int


def node_a(state):

    print("A")

    state["count"] += 1

    return state


def node_b(state):

    print("B")

    state["count"] += 1

    return state


graph = StateGraph(State)

graph.add_node("A", node_a)
graph.add_node("B", node_b)

graph.add_edge(START, "A")
graph.add_edge("A", "B")
graph.add_edge("B", END)
# graph.add_conditional_edges()
app = graph.compile(checkpointer=mem)
config = {
    "configurable": {
        "thread_id": "user_001"
    }
}
result = app.invoke(
    {
        "count": 10
    },
    config=config
)
snapshot = app.get_state(config)

print(snapshot.values)
print(result)