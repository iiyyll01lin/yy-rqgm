"""The AGENT half of RQGM co-evolution (the task-agent that also evolves).

RQGM has two halves that co-evolve under an anti-reward-hacking asymmetry:

* the **evaluator** half (:mod:`backend.evaluator`) is gated by held-out ground
  truth (the anchor set) — it is the fuzzy judge that evolves *what "good" means*;
* the **agent** half (this package) is the task-agent that proposes concrete
  architectures. It evolves too, but it is scored/gated by the **epoch-FROZEN
  champion evaluator** — never by the held-out anchors, and it can neither read
  nor modify the evaluator's rubric or gate.

Within an evaluator epoch the champion rubric is frozen, so the agent cannot
observe or game a moving judge; it can only win by genuinely covering the failure
modes the current champion actually scores. At an evaluator epoch boundary the
agent archive's utilities are selectively erased (re-scored under the new
champion), mirroring the evaluator's own memory selective erasure. See
``docs/03-evaluator.md`` and ``docs/blueprint.md`` (Iron Law II).
"""
