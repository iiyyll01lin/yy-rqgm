"""p1-ledger: cross-epoch metrics time-series + promotion regression guard.

The ledger persists the champion's held-out quality metrics (val/test separation,
judge accuracy/κ, hack-ratio, over-acceptance) once per applied promotion. The
regression guard asserts a promotion NEVER reduces the untouched gold ``test``
split's separation or Cohen's κ.
"""

from backend.evaluator import evolve, report, versioning
from backend.memory.qdrant_store import EvolutionaryMemory


def test_metrics_snapshot_records_the_full_series_row():
    report.reset_metrics_ledger()
    row = report.record_metrics_snapshot()
    for key in report._LEDGER_KEYS:
        assert key in row
    assert row["epoch_id"] == versioning.get_epoch()
    # The core time-series metrics the plan calls for are present + typed.
    for k in ("val_separation", "test_separation", "hack_ratio", "over_acceptance_rate", "test_kappa"):
        assert isinstance(row[k], (int, float))
    ledger = report.load_metrics_ledger()
    assert ledger and ledger[-1]["champion_version"] == row["champion_version"]


def test_regression_violations_detects_test_drops():
    prev = {"test_separation": 0.50, "test_kappa": 0.80}
    assert report.regression_violations(prev, {"test_separation": 0.40, "test_kappa": 0.80})
    assert report.regression_violations(prev, {"test_separation": 0.50, "test_kappa": 0.70})
    # An improvement (or a tie) is not a violation.
    assert report.regression_violations(prev, {"test_separation": 0.60, "test_kappa": 0.90}) == []
    assert report.regression_violations(prev, {"test_separation": 0.50, "test_kappa": 0.80}) == []


def test_promotion_does_not_reduce_test_separation_or_kappa():
    report.reset_metrics_ledger()
    # Baseline snapshot of the epoch-0 champion (untouched gold test split).
    baseline = report.record_metrics_snapshot()
    assert baseline["epoch_id"] == 0

    prop = evolve.propose_challenger()  # adds reward_hacking_resistance
    result = evolve.approve_challenger(
        prop.version, approve=True, memory=EvolutionaryMemory(collection="t_ledger_guard")
    )
    assert result["applied"] is True

    ledger = report.load_metrics_ledger()
    assert len(ledger) >= 2  # baseline (epoch 0) + auto-recorded promotion (epoch 1)
    prev, curr = ledger[-2], ledger[-1]
    assert curr["epoch_id"] == prev["epoch_id"] + 1

    # THE GUARD: the promotion must not regress the gold test split.
    assert report.regression_violations(prev, curr) == []
    assert curr["test_separation"] >= prev["test_separation"]
    assert curr["test_kappa"] >= prev["test_kappa"]


def test_ledger_grows_one_row_per_applied_promotion():
    report.reset_metrics_ledger()
    assert report.load_metrics_ledger() == []
    prop = evolve.propose_challenger()
    # A vetoed (rejected) promotion must NOT append a row.
    evolve.approve_challenger(prop.version, approve=False)
    assert report.load_metrics_ledger() == []
    # An applied promotion appends exactly one row.
    prop2 = evolve.propose_challenger()
    evolve.approve_challenger(
        prop2.version, approve=True, memory=EvolutionaryMemory(collection="t_ledger_grow")
    )
    assert len(report.load_metrics_ledger()) == 1
