"""Storage/scheduling sentinel tests; these files are not research effect data."""
import numpy as np
import pytest

from models.checkpoints import digest, load_cache, merge_shards, save_cache
from models.effect_cache import row_shard, should_stop


def test_digest_key_order():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})


def test_atomic_roundtrip_and_resume_sentinels(tmp_path):
    path = tmp_path / "checkpoint.npz"
    values = np.full((2, 2), np.nan, dtype=np.float32)
    values[0] = 0  # Storage sentinel, not an observed effect.
    save_cache(path, values, {"identity": "a"})
    loaded, meta = load_cache(path, "a", (2, 2))
    np.testing.assert_array_equal(values, loaded)
    assert meta == {"identity": "a"}
    assert not path.with_suffix(".npz.tmp").exists()
    with pytest.raises(ValueError, match="mismatch"):
        load_cache(path, "different", (2, 2))
    with pytest.raises(ValueError, match="mismatch"):
        load_cache(path, "a", (1, 2))


def test_merge_requires_every_entry_and_disjoint_rows(tmp_path):
    paths = [tmp_path / f"worker-{rank}.npz" for rank in range(2)]
    shape = (3, 2)
    metadata = {"identity": "id"}
    output = tmp_path / "final.npz"
    arrays = [np.full(shape, np.nan, dtype=np.float32) for _ in paths]
    arrays[0][row_shard(3, 0, 2)] = 0
    for path, array in zip(paths, arrays):
        save_cache(path, array, metadata)
    assert not merge_shards(paths, "id", shape, metadata, output)
    assert not output.exists()
    arrays[1][row_shard(3, 1, 2)] = 0
    save_cache(paths[1], arrays[1], metadata)
    assert merge_shards(paths, "id", shape, metadata, output)
    values, meta = load_cache(output, "id", shape)
    assert np.isfinite(values).all() and meta["complete"]
    with np.load(output, allow_pickle=False) as archive:
        rows = archive["observations"]
        assert len(rows) == 6
        np.testing.assert_array_equal(rows["record_index"], [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(rows["head_index"], [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(rows["effect"], values.ravel())
    arrays[1][0] = 0
    save_cache(paths[1], arrays[1], metadata)
    with pytest.raises(ValueError, match="Overlapping"):
        merge_shards(paths, "id", shape, metadata, output)


@pytest.mark.parametrize("count", [8, 9, 1024])
def test_shards_are_disjoint_and_exhaustive(count):
    left, right = set(row_shard(count, 0, 2)), set(row_shard(count, 1, 2))
    assert not left & right
    assert left | right == set(range(count))


def test_deadline_margin():
    assert not should_stop(1000, 300, now=699)
    assert should_stop(1000, 300, now=700)
    assert should_stop(1000, 300, now=1001)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, 2.01, -2.01])
def test_corrupt_effect_values_rejected(tmp_path, bad):
    path = tmp_path / "bad.npz"
    save_cache(path, np.array([[bad]]), {"identity": "id"})
    with pytest.raises(ValueError, match="Invalid"):
        load_cache(path, "id", (1, 1))


def test_interrupted_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    values = np.full((1, 1), np.nan, dtype=np.float32)
    save_cache(path, values, {"identity": "id"})

    def fail(*args, **kwargs):
        raise OSError("unit-test interrupted write")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="interrupted write"):
        save_cache(path, values, {"identity": "different"})
    loaded, meta = load_cache(path, "id", (1, 1))
    np.testing.assert_array_equal(loaded, values)
