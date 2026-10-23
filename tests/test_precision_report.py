"""Precision-report plumbing with storage sentinels, not model-effect simulations."""
import json

import numpy as np
import pytest

from kaggle.run_effect_cache import compare_precision
from models.checkpoints import save_cache


@pytest.mark.parametrize("case", ["matching", "mismatch", "correspondence"])
def test_precision_report_gates(tmp_path, case):
    metadata = {"record_ids": ["unit-test-storage-sentinel"], "heads": [[0, 0]],
                "config_hash": "unit", "population_hash": "unit", "code_hash": "unit"}
    half, full = tmp_path / "half.npz", tmp_path / "full.npz"
    save_cache(half, np.zeros((1, 1), dtype=np.float32), metadata)
    other = {**metadata, "heads": [[1, 0]]} if case == "correspondence" else metadata
    save_cache(full, np.full((1, 1), 1 if case == "mismatch" else 0, dtype=np.float32), other)
    report = tmp_path / "agreement.json"
    if case == "matching":
        compare_precision(half, full, 0.02, report)
        assert json.loads(report.read_text())["passed"]
    else:
        with pytest.raises(ValueError):
            compare_precision(half, full, 0.02, report)
        if case == "mismatch":
            assert not json.loads(report.read_text())["passed"]
