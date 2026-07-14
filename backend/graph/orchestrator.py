"""Build + compile the orchestration StateGraph with a durable checkpointer.

Flow::

    START -> router -> task_agent -> gatekeeper --(infeasible)--> END
                                          |
                                      (feasible)
                                          v
                                  rqgm_evaluator -> hitl (interrupt) -> END

The HITL interrupt is made durable by a SqliteSaver checkpointer; the API drives
it with one ``thread_id`` per session and resumes via ``Command(resume=...)``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from backend.graph.nodes import (
    gatekeeper_node,
    hitl_node,
    router_node,
    rqgm_evaluator_node,
    task_agent_node,
)
from backend.graph.state import GraphState

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _REPO_ROOT / "data" / "checkpoints"
_CHECKPOINT_DB = _CHECKPOINT_DIR / "graph.db"


def _route_after_gatekeeper(state: GraphState) -> str:
    """Physics gate: only feasible proposals reach the fuzzy evaluator."""
    return "evaluate" if state.get("feasible", True) else "reject"


def build_graph(checkpointer: Any | None = None):
    """Compile the graph. Pass a checkpointer (e.g. MemorySaver) for tests."""
    builder = StateGraph(GraphState)
    builder.add_node("router", router_node)
    builder.add_node("task_agent", task_agent_node)
    builder.add_node("gatekeeper", gatekeeper_node)
    builder.add_node("rqgm_evaluator", rqgm_evaluator_node)
    builder.add_node("hitl", hitl_node)

    builder.add_edge(START, "router")
    builder.add_edge("router", "task_agent")
    builder.add_edge("task_agent", "gatekeeper")
    builder.add_conditional_edges(
        "gatekeeper",
        _route_after_gatekeeper,
        {"evaluate": "rqgm_evaluator", "reject": END},
    )
    builder.add_edge("rqgm_evaluator", "hitl")
    builder.add_edge("hitl", END)

    if checkpointer is None:
        checkpointer = _default_checkpointer()
    return builder.compile(checkpointer=checkpointer)


def _default_checkpointer() -> SqliteSaver:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CHECKPOINT_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    return saver


# ---------------------------------------------------------------------------
# Compiled-graph singleton + run helpers
# ---------------------------------------------------------------------------
_graph = None


def get_compiled_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _extract_interrupt(result: Any, snapshot: Any) -> dict[str, Any] | None:
    # Modern LangGraph surfaces interrupts under "__interrupt__" in the result.
    interrupts = None
    if isinstance(result, dict):
        interrupts = result.get("__interrupt__")
    if interrupts:
        first = interrupts[0]
        return getattr(first, "value", first)
    # Fallback: inspect pending tasks on the snapshot.
    for task in getattr(snapshot, "tasks", []) or []:
        for itr in getattr(task, "interrupts", []) or []:
            return getattr(itr, "value", itr)
    return None


def _summarize(graph, result: Any, config: dict) -> dict[str, Any]:
    snapshot = graph.get_state(config)
    awaiting = bool(snapshot.next)
    return {
        "awaiting_hitl": awaiting,
        "hitl_request": _extract_interrupt(result, snapshot) if awaiting else None,
        "state": dict(snapshot.values),
        "next": list(snapshot.next),
    }


def start_run(inputs: dict[str, Any], thread_id: str, graph=None) -> dict[str, Any]:
    """Run the graph until completion or the HITL interrupt."""
    graph = graph or get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(inputs, config)
    return _summarize(graph, result, config)


def resume_run(decision: Any, thread_id: str, graph=None) -> dict[str, Any]:
    """Resume a paused run with the human decision."""
    graph = graph or get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=decision), config)
    return _summarize(graph, result, config)
