"""One CUDA device per independent worker; disjoint record rows, no DDP."""
import json
import logging
import signal
import time
from pathlib import Path

import numpy as np

from models.checkpoints import load_cache, save_cache


def row_shard(count, rank, world):
    return list(range(rank, count, world))


def should_stop(deadline, margin, now=None):
    return (time.time() if now is None else now) >= deadline - margin


def worker(rank, device_index, world, records, heads, config, metadata, directory,
           precision, deadline):
    import torch
    from models.intervention_hooks import effect_batch
    from models.load_gpt2 import load_bridge, validate_bridge

    directory = Path(directory)
    logger = logging.getLogger(f"worker-{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / f"worker-{rank}.log")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    shape = (len(records), len(heads))
    checkpoint = directory / f"worker-{rank}.npz"
    values = np.full(shape, np.nan, dtype=np.float32)
    worker_metadata = dict(metadata)
    if checkpoint.exists():
        values, worker_metadata = load_cache(checkpoint, metadata["identity"], shape)
        logger.info("resumed checkpoint=%s", checkpoint)
    owned = row_shard(len(records), rank, world)
    other = sorted(set(range(len(records))) - set(owned))
    if np.isfinite(values[other]).any():
        raise ValueError("Checkpoint contains another worker's rows")
    pending = [i for i in owned if not np.isfinite(values[i]).all()]
    if not pending:
        logger.info("shard already complete")
        return
    if should_stop(deadline, config["stop_margin_seconds"]) or stop_requested:
        save_cache(checkpoint, values, worker_metadata)
        return
    started = time.monotonic()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = load_bridge(config, device, dtype)
    diagnostics = validate_bridge(model, [records[i] for i in owned[:2]], config, heads, device, dtype)
    worker_metadata["validation"] = diagnostics
    worker_metadata["device"] = torch.cuda.get_device_name(device_index)
    (directory / f"validation-{rank}.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    logger.info("validated device=%s precision=%s", device, precision)
    completed_batches = 0
    compute_started = time.monotonic()
    try:
        for offset in range(0, len(pending), config["batch_size"]):
            if stop_requested or should_stop(deadline, config["stop_margin_seconds"]):
                logger.info("stopping at safe batch boundary")
                break
            indices = pending[offset:offset + config["batch_size"]]
            values[indices] = effect_batch(model, [records[i] for i in indices], heads, device)
            completed_batches += 1
            done = min(offset + len(indices), len(pending))
            elapsed = time.monotonic() - compute_started
            logger.info(json.dumps({"worker": rank, "completed": done, "pending_initial": len(pending),
                "eta_seconds": elapsed / done * (len(pending) - done),
                "elapsed_seconds": time.monotonic() - started,
                "allocated_mb": torch.cuda.memory_allocated(device_index) / 2**20,
                "reserved_mb": torch.cuda.memory_reserved(device_index) / 2**20,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device_index) / 2**20}))
            if completed_batches % config["checkpoint_batches"] == 0:
                save_cache(checkpoint, values, worker_metadata)
    finally:
        # Includes Python failures and cooperative stop, not SIGKILL/power loss.
        save_cache(checkpoint, values, worker_metadata)
