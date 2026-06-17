from typing import TypedDict
from typing_extensions import Annotated

import operator

from langgraph.graph import (
    StateGraph,
    START,
    END
)


class State(TypedDict):

    count: Annotated[
        int,
        operator.add
    ]


def node_a(state):

    print("A:", state)

    return {}


def node_b(state):

    print("B:", state)

    return {
        "count": 1
    }


def node_c(state):

    print("C:", state)

    return {
        "count": 1
    }


def node_d(state):

    print("D:", state)

    return {
        "count": 1
    }


def node_e(state):

    print("E:", state)

    return state


builder = StateGraph(State)

builder.add_node("A", node_a)
builder.add_node("B", node_b)
builder.add_node("C", node_c)
builder.add_node("D", node_d)
builder.add_node("E", node_e)

builder.add_edge(START, "A")

builder.add_edge("A", "B")
builder.add_edge("A", "C")
builder.add_edge("A", "D")

builder.add_edge(
    ["B", "C", "D"],
    "E"
)

builder.add_edge("E", END)

app = builder.compile()

result = app.invoke(
    {
        "count": 0
    }
)

print(result)