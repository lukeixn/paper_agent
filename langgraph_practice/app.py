from __future__ import annotations

import time
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

try:
    from .demo_core import AGENTS, create_initial_state, render_svg, stream_demo
except ImportError:  # Streamlit runs this file as a script.
    from demo_core import AGENTS, create_initial_state, render_svg, stream_demo


# This app is intentionally independent from the real Paper Agent UI.
# It exists only for practicing LangGraph concepts with visible feedback.


st.set_page_config(
    page_title="LangGraph Practice",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_progress(query: str) -> dict[str, Any]:
    return {
        "state": create_initial_state(query),
        "completed_nodes": set(),
        "completed_agents": set(),
        "active_node": None,
        "events": [],
    }


def draw(progress: dict[str, Any]) -> None:
    svg = render_svg(
        progress["state"],
        active_node=progress["active_node"],
        completed_nodes=progress["completed_nodes"],
        completed_agents=progress["completed_agents"],
    )
    components.html(svg, height=620, scrolling=False)


st.markdown(
    """
<style>
  .block-container { padding-top: 2rem; }
  [data-testid="stMetricValue"] { font-size: 1.7rem; }
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Practice Controls")
    query = st.text_area(
        "Question",
        value="Compare methods and limitations for KV cache compression.",
        height=100,
    )
    speed = st.slider(
        "Replay delay",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )
    st.caption("Try keywords: method, limitation, innovation, survey, publish.")
    run = st.button("Run LangGraph Demo", type="primary", use_container_width=True)


st.title("LangGraph Practice Lab")
st.write(
    "A minimal demo that mirrors this project: normalize question, retrieve papers, "
    "route to selected agents, run parallel branches, then reduce into a report."
)

left, right = st.columns([0.95, 1.25], gap="large")

with left:
    st.subheader("Knowledge Map")
    st.markdown(
        """
1. **State**: shared graph data contract.
2. **Node**: reads state, returns partial updates.
3. **Route**: decides which agents are needed.
4. **Send**: creates dynamic parallel branches.
5. **Reducer**: merges `agent_outputs` with `operator.add`.
6. **Stream**: drives realtime UI updates.
"""
    )

    st.subheader("Available Agents")
    for name, spec in AGENTS.items():
        st.markdown(f"- `{name}`: **{spec.title}** - {spec.focus}")

    metrics = st.empty()
    timeline = st.empty()
    answer_box = st.empty()

with right:
    graph_box = st.empty()


progress = init_progress(query)
with right:
    with graph_box.container():
        draw(progress)

if run:
    progress = init_progress(query)
    for event in stream_demo(query):
        node = event["node"]
        progress["state"] = event["state"]
        progress["active_node"] = node
        if node == "run_agent" and event.get("agent_name"):
            progress["completed_agents"].add(event["agent_name"])
            event_label = f"run_agent -> {event['agent_name']}"
        else:
            progress["completed_nodes"].add(node)
            event_label = node
        progress["events"].append(event_label)

        with right:
            with graph_box.container():
                draw(progress)

        state = progress["state"]
        with left:
            with metrics.container():
                cols = st.columns(3)
                cols[0].metric("Papers", len(state.get("papers", [])))
                cols[1].metric("Selected Agents", len(state.get("selected_agents", [])))
                cols[2].metric("Completed Agents", len(progress["completed_agents"]))
            timeline.markdown(
                "### Event Stream\n"
                + "\n".join(f"- `{item}`" for item in progress["events"])
            )
            if state.get("final_answer"):
                answer_box.markdown(state["final_answer"])

        time.sleep(speed)
