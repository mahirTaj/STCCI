"""Atomic, non-pickle cache files. Never mix runs with different identities."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_cache(path, effects, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        arrays = {"effects": effects, "metadata": json.dumps(metadata, sort_keys=True)}
        if metadata.get("complete"):
            if not np.isfinite(effects).all():
                raise ValueError("Cannot mark an incomplete cache complete")
            # One row per observed intervention, linked to ordered IDs/heads in metadata.
            observations = np.empty(effects.size, dtype=[("record_index", "i4"), ("head_index", "i4"),
                                                        ("effect", "f4")])
            record_indices, head_indices = np.indices(effects.shape)
            observations["record_index"] = record_indices.ravel()
            observations["head_index"] = head_indices.ravel()
            observations["effect"] = effects.ravel()
            arrays["observations"] = observations
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_cache(path, identity, shape):
    with np.load(path, allow_pickle=False) as archive:
        effects = archive["effects"].copy()
        metadata = json.loads(str(archive["metadata"]))
    if metadata["identity"] != identity or effects.shape != shape:
        raise ValueError("Checkpoint identity/shape mismatch; use a separate output directory")
    if np.isinf(effects).any() or (np.abs(effects[np.isfinite(effects)]) > 2).any():
        raise ValueError("Invalid cached effects")
    return effects, metadata


def merge_shards(paths, identity, shape, metadata, output):
    combined = np.full(shape, np.nan, dtype=np.float32)
    for path in paths:
        values, _ = load_cache(path, identity, shape)
        present = np.isfinite(values)
        if np.any(present & np.isfinite(combined)):
            raise ValueError("Overlapping shards")
        combined[present] = values[present]
    if not np.isfinite(combined).all():
        return False
    save_cache(output, combined, {**metadata, "complete": True})
    return True
