"""LangGraph orchestration (Hot/Cold path).

A StateGraph wires the deterministic and fuzzy halves together:

    router -> task_agent -> gatekeeper_node --(infeasible)--> END (hard reject)
                                 |
                             (feasible)
                                 v
                         rqgm_evaluator_node -> hitl_node (interrupt) -> END

Compiled with a SqliteSaver checkpointer so the HITL interrupt is durable and
resumable per ``thread_id`` (one per session).
"""

from backend.graph.state import GraphState  # noqa: F401

__all__ = ["GraphState"]
