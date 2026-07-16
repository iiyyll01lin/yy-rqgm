"""RQGM AGENT-half co-evolution (offline, deterministic).

Covers the four required properties:

* agent evolution IMPROVES utility (under the frozen champion evaluator);
* the agent GATE blocks a worse / no-op challenger;
* an EVALUATOR promotion triggers agent-utility selective erasure (coupling);
* the epoch-internal ANTI-HACK asymmetry — the agent cannot game the evaluator
  within an epoch (frozen rubric + zero gain from covering a champion blind spot),
  and agent evolution never reads/mutates the evaluator's rubric or gate.

All deterministic via the inference mock (LEMONADE_FORCE_MOCK=1 in conftest).
"""

from fastapi.testclient import TestClient

from backend.agent import agent_evolve, agent_versioning, coevolve, concrete_archive
from backend.agent import needs as needs_ds
from backend.agent.agent_program import AgentProgram, load_seed_program
from backend.agent.agent_score import score_program
from backend.evaluator import evolve, versioning
from backend.evaluator.mutation import mutate_rubric_text
from backend.memory.qdrant_store import EvolutionaryMemory


# ---------------------------------------------------------------------------
# Representation + hot path
# ---------------------------------------------------------------------------
def test_seed_program_and_versioning_defaults():
    seed = load_seed_program()
    assert seed.skills  # non-empty baseline
    assert agent_versioning.get_agent_epoch() == 0
    assert agent_versioning.get_champion_version() == "agent-champion-0"
    # champion program == seed at epoch 0
    assert agent_versioning.get_champion_program().genome_text() == seed.genome_text()


def test_genome_text_is_order_invariant():
    """A no-op reordering must NOT change the program identity (honesty control)."""
    a = AgentProgram("p", ["x", "y", "z"])
    b = AgentProgram("p", ["z", "y", "x"])
    assert a.genome_text() == b.genome_text()
    assert a.program_id() == b.program_id()


def test_hot_path_uses_champion_program():
    from backend.graph.nodes.task_agent import task_agent_node

    out = task_agent_node({"need": "predictive maintenance", "domain": "smart_manufacturing"})
    assert out["architecture"]
    assert out["agent_program_version"] == "agent-champion-0"


# ---------------------------------------------------------------------------
# Agent evolution improves utility (scored by the FROZEN champion evaluator)
# ---------------------------------------------------------------------------
def test_agent_evolution_improves_utility():
    proposal, frontier = agent_evolve.propose_agent_via_frontier(budget=8, seed=7)
    assert proposal  # frontier beat the incumbent
    champion = agent_versioning.get_champion_program()
    challenger = agent_versioning.get_challenger_program(proposal["challenger_id"])
    rubric = versioning.get_champion_rubric_text()
    val = needs_ds.load_needs(needs_ds.VAL)
    champ_u = score_program(champion, val, rubric_text=rubric, evaluator_epoch=0).utility
    chal_u = score_program(challenger, val, rubric_text=rubric, evaluator_epoch=0).utility
    assert chal_u > champ_u
    # the challenger covers strictly more skills than the seed champion
    assert set(challenger.skills) > set(champion.skills)


def test_agent_gate_promotes_better_and_advances_epoch():
    proposal, _ = agent_evolve.propose_agent_via_frontier(budget=8, seed=7)
    result = agent_evolve.approve_agent_challenger(proposal["challenger_id"])
    assert result["applied"] is True
    assert result["gate"]["passed"] is True
    assert result["agent_epoch_id"] == 1
    assert agent_versioning.get_agent_epoch() == 1
    # the hot path now adopts the promoted program
    assert agent_versioning.get_champion_version() == proposal["challenger_id"]


# ---------------------------------------------------------------------------
# Gate blocks worse / no-op challengers
# ---------------------------------------------------------------------------
def test_agent_gate_blocks_worse_challenger():
    """A challenger that covers FEWER skills scores lower and is rejected (tie/loss
    favours the incumbent), exactly like the evaluator code gate."""
    seed = load_seed_program()
    worse = AgentProgram(seed.system_prompt, ["typed_state_schema"])  # dropped 2 skills
    agent_versioning.register_challenger(
        version="agent-worse", program=worse, parent_version="agent-champion-0", evaluator_epoch=0
    )
    gate = agent_evolve.evaluate_agent_promotion("agent-worse")
    assert gate.passed is False
    assert gate.p1_non_inferior is False
    assert gate.challenger_utility <= gate.champion_utility


def test_no_op_program_change_does_not_win_gate():
    """Adding an IRRELEVANT (no-op) skill must not win the gate — the offline
    honesty control against reward-hacking the agent evolution itself."""
    seed = load_seed_program()
    # 'surrogate_validation' covers no need flaw here -> identical architectures.
    noop = seed.with_skill("surrogate_validation")
    agent_versioning.register_challenger(
        version="agent-noop", program=noop, parent_version="agent-champion-0", evaluator_epoch=0
    )
    gate = agent_evolve.evaluate_agent_promotion("agent-noop")
    assert gate.passed is False
    assert gate.utility_delta == 0.0          # a true no-op: zero utility change
    assert gate.p1_non_inferior is False       # tie favours the incumbent


# ---------------------------------------------------------------------------
# THE anti-hack asymmetry (agent cannot game the evaluator within an epoch)
# ---------------------------------------------------------------------------
def test_epoch_internal_anti_hack_asymmetry():
    """Within an evaluator epoch the agent CANNOT game the (frozen) judge:

    (a) the champion rubric text is byte-identical before/after agent evolution;
    (b) agent evolution never advances/mutates the evaluator epoch or champion;
    (c) covering a flaw the FROZEN champion does NOT catch (kpi_sensor_gaming, a
        champion-0 blind spot) yields ZERO utility gain — so the only way for the
        agent to gain is to genuinely cover a failure mode the evaluator scores.
    """
    rubric_before = versioning.get_champion_rubric_text()
    epoch_before = versioning.get_epoch()
    champ_before = versioning.get_champion_version()

    # A full agent evolution + promotion round.
    proposal, _ = agent_evolve.propose_agent_via_frontier(budget=8, seed=7)
    agent_evolve.approve_agent_challenger(proposal["challenger_id"])

    # (a) + (b): the evaluator half is untouched by agent evolution.
    assert versioning.get_champion_rubric_text() == rubric_before
    assert versioning.get_epoch() == epoch_before == 0
    assert versioning.get_champion_version() == champ_before == "champion-0"

    # (c): covering the champion-0 blind spot (kpi) is worthless within the epoch.
    rubric = versioning.get_champion_rubric_text()
    kpi_needs = [n for n in needs_ds.load_needs(needs_ds.VAL) if "kpi_sensor_gaming" in n.get("latent_flaws", [])]
    assert kpi_needs  # the dataset carries kpi needs
    base = load_seed_program()
    gamer = base.with_skill("reward_hacking_guard")  # "guards" a flaw the frozen judge can't see
    u_base = score_program(base, kpi_needs, rubric_text=rubric, evaluator_epoch=0).utility
    u_gamer = score_program(gamer, kpi_needs, rubric_text=rubric, evaluator_epoch=0).utility
    assert u_gamer == u_base  # zero gain: the frozen champion does not score kpi gaming


def test_agent_scored_by_frozen_champion_not_needs_ground_truth():
    """The agent's jury is the evaluator champion, tracked by evaluator versioning —
    changing the evaluator champion changes the score; the needs' labels never do."""
    seed = load_seed_program()
    val = needs_ds.load_needs(needs_ds.VAL)
    champ0 = versioning.get_champion_rubric_text()
    u0 = score_program(seed, val, rubric_text=champ0, evaluator_epoch=0).utility

    # Promote the evaluator (adds reward_hacking_resistance) -> the SAME program is
    # now scored by a DIFFERENT (stricter) champion, so its utility changes.
    better = mutate_rubric_text(
        champ0, [{"id": "reward_hacking_resistance", "text": "Penalize KPI gaming via sensor disabling."}],
        "challenger-rhr", 1,
    )
    versioning.register_challenger(version="challenger-rhr", rubric_text=better, parent_version="champion-0")
    evolve.approve_challenger("challenger-rhr", approve=True, memory=EvolutionaryMemory(collection="t_frozen"))
    champ1 = versioning.get_champion_rubric_text()
    u1 = score_program(seed, val, rubric_text=champ1, evaluator_epoch=versioning.get_epoch()).utility
    assert u1 < u0  # the stricter champion penalises the seed's uncovered kpi gaming


# ---------------------------------------------------------------------------
# Coupling: an EVALUATOR promotion erases stale agent utility
# ---------------------------------------------------------------------------
def test_evaluator_promotion_triggers_agent_utility_erasure():
    # 1. Evolve + promote the agent under the FROZEN champion-0 evaluator.
    proposal, _ = agent_evolve.propose_agent_via_frontier(budget=8, seed=7)
    agent_evolve.approve_agent_challenger(proposal["challenger_id"])
    assert agent_versioning.get_agent_epoch() == 1
    # 2. Baseline the champion agent's val utility under evaluator epoch 0.
    u_before = agent_evolve.measure_champion_utility()

    # 3. Promote the EVALUATOR (close the kpi blind spot).
    champ0 = versioning.get_champion_rubric_text()
    better = mutate_rubric_text(
        champ0, [{"id": "reward_hacking_resistance", "text": "Penalize KPI gaming via sensor disabling."}],
        "challenger-rhr", 1,
    )
    versioning.register_challenger(version="challenger-rhr", rubric_text=better, parent_version="champion-0")
    result = evolve.approve_challenger("challenger-rhr", approve=True, memory=EvolutionaryMemory(collection="t_erase"))
    assert result["applied"] is True
    assert versioning.get_epoch() == 1

    # 4. The coupling hook re-scored the agent archive under the new champion.
    erasure = result["agent_utility_erasure"]
    assert erasure["rescored"] >= 1
    assert erasure["utilities_dropped"] >= 1        # the gaming agent lost utility
    assert erasure["new_evaluator_epoch"] == 1
    u_after, ev_epoch = agent_versioning.get_champion_utility()
    assert ev_epoch == 1                            # re-stamped to the new evaluator epoch
    assert u_after < u_before                       # gaming a closed blind spot no longer pays


# ---------------------------------------------------------------------------
# Concrete-archive (DGM) linking: strong -> anchor, gamed -> adversarial
# ---------------------------------------------------------------------------
def test_concrete_archive_links_strong_and_gamed():
    seed = load_seed_program()
    dev = needs_ds.load_needs(needs_ds.DEV)
    archives = concrete_archive.build_archive(dev, seed)
    strong = concrete_archive.anchor_candidates(archives)
    gamed = concrete_archive.adversarial_samples(archives)
    assert strong  # some genuinely-strong architectures become anchor candidates
    assert gamed   # some gamed architectures become adversarial samples
    # every gamed sample targets a genuine champion-0 blind spot
    targets = {g["targets"] for g in gamed}
    assert targets <= {"kpi_sensor_gaming", "concept_drift_blind"}
    for g in gamed:
        assert g["split"] == "adversarial"
        assert g["provenance"] == "agent_concrete_archive"


# ---------------------------------------------------------------------------
# One full alternating co-evolution round
# ---------------------------------------------------------------------------
def test_coevolution_round_runs_and_promotes_both_halves():
    summary = coevolve.run_coevolution_round(
        agent_budget=6, evaluator_budget=6, seed=7,
        promote_agent=True, approve_evaluator=True,
    )
    # agent half advanced under the frozen evaluator
    assert summary["agent"]["promotion"]["applied"] is True
    assert summary["agent_epoch"] == 1
    # self-play mined the champion blind spots as adversarial samples
    assert set(summary["self_play"]["targets"]) <= {"kpi_sensor_gaming", "concept_drift_blind"}
    # evaluator promoted -> its epoch advanced -> triggered agent-utility erasure
    ev_promo = summary["evaluator"]["promotion"]
    assert ev_promo["applied"] is True
    assert versioning.get_epoch() == 1
    assert ev_promo["agent_utility_erasure"]["rescored"] >= 1


# ---------------------------------------------------------------------------
# API endpoints + report blocks
# ---------------------------------------------------------------------------
def test_agent_api_propose_promote_and_report():
    with TestClient(app_module().app) as c:
        prop = c.post("/api/admin/agent/propose").json()
        assert prop["challenger_id"].startswith("agent-challenger-")
        assert prop["frontier"]["size"] >= 1
        promo = c.post("/api/admin/agent/promote").json()
        assert promo["applied"] is True
        assert promo["agent_epoch_id"] == 1
        assert promo["gate"]["passed"] is True

        rep = c.get("/api/admin/report").json()
        assert rep["agent"]["agent_epoch"] == 1
        assert rep["agent"]["scored_by"] == "frozen_champion_evaluator"
        assert 0.0 <= rep["agent"]["val_utility"] <= 1.0
        assert set(rep["co_evolution"]) >= {"agent_epoch", "evaluator_epoch", "agent_mined_adversarial"}


def app_module():
    import backend.app.main as m

    return m
