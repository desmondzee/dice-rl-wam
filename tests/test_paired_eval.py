import csv
import json
import math
from pathlib import Path

import pytest

from analysis.paired_eval import arm_of, bootstrap_ci, mcnemar_exact, paired, summarize, wilson, write_curve


def _write_arm(root, successes, settings=None):
    root.mkdir(parents=True, exist_ok=True)
    for (task, state, seed_index), success in successes.items():
        index = seed_index * 2 + state
        path = root / "episodes" / f"task_{task:02d}" / f"episode_{index:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"task_id": task, "init_state_id": state, "seed_index": seed_index, "success": success}))
    if settings is not None:
        (root / "settings.json").write_text(json.dumps(settings))
    return root


def test_wilson_bounds_match_known_values():
    p, low, high = wilson(50, 100)
    assert p == 0.5 and math.isclose(low, 0.40383, abs_tol=1e-4) and math.isclose(high, 0.59617, abs_tol=1e-4)
    _, low, high = wilson(0, 10)
    assert low == 0.0 and math.isclose(high, 0.27753, abs_tol=1e-4)
    assert all(math.isnan(value) for value in wilson(0, 0))


def test_mcnemar_exact_is_binomial_on_discordant_pairs():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert math.isclose(mcnemar_exact(5, 0), 2 / 32)
    assert math.isclose(mcnemar_exact(1, 6), 2 * (1 + 7) / 128)
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def test_paired_counts_only_shared_keys_and_bootstraps_the_difference():
    base = {(0, s, 0): s < 6 for s in range(10)}
    other = {(0, s, 0): s < 8 for s in range(10)}
    other[(0, 99, 0)] = True
    result = paired(base, other)
    assert (result["n"], result["wins"], result["losses"]) == (10, 2, 0)
    assert math.isclose(result["diff"], 0.2) and math.isclose(result["p"], 0.5)
    assert result["ci_low"] >= 0.0 and result["ci_high"] <= 0.5 and result["ci_low"] <= 0.2 <= result["ci_high"]
    low, high = bootstrap_ci([1.0] * 5)
    assert low == high == 1.0
    with pytest.raises(ValueError, match="No shared"):
        paired(base, {(1, 0, 0): True})


def test_arm_reads_settings_or_falls_back_to_directory_name(tmp_path):
    named = tmp_path / "heldout-k1-step320000"
    named.mkdir()
    assert arm_of(named) == (1, 320000)
    assert arm_of(_write_arm(tmp_path / "x", {}, settings={"k": 4, "step": 80000})) == (4, 80000)
    assert arm_of(tmp_path / "heldout-k4") == (4, 0)


def test_summarize_pairs_arms_per_task_and_writes_curve(tmp_path):
    keys = [(0, s, i) for s in range(2) for i in range(2)]
    sft = _write_arm(tmp_path / "sft", {k: k[1] == 0 for k in keys})
    k4 = _write_arm(tmp_path / "heldout-k4-step080000", {k: True for k in keys}, {"k": 4, "step": 80000})
    k1 = _write_arm(tmp_path / "heldout-k1-step080000", {k: k[2] == 0 for k in keys}, {"k": 1, "step": 80000})
    curve, lines = summarize(sft, [k4, k1])
    assert [(row["arm"], row["step"], row["success"]) for row in curve] == [("sft-k1", 0, 0.5), ("rl-k4", 80000, 1.0), ("rl-k1", 80000, 0.5)]
    text = "\n".join(lines)
    assert "rl-k4@80000 vs sft-k1: diff +0.500 (n=4, wins 2, losses 0), McNemar p=0.500" in text
    assert "rl-k4 vs rl-k1 @80000: diff +0.500 (n=4, wins 2, losses 0)" in text
    write_curve(tmp_path / "curve.csv", curve)
    with (tmp_path / "curve.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["task"] == "0" and rows[1]["arm"] == "rl-k4" and rows[1]["ci_low"] != ""
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path / "missing", [])
