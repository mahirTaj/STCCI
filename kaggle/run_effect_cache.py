"""Thin CLI entry point. GPU execution is only performed when invoked by the user."""
import argparse
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from models.checkpoints import digest, load_cache, merge_shards
from models.effect_cache import should_stop, worker


def code_hash():
    paths = [ROOT / "requirements.txt", ROOT / "reports/population_manifest.json"]
    for folder in ("data", "models", "kaggle"):
        paths.extend(sorted((ROOT / folder).glob("*.py")))
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kill_test.json")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/stcci"))
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"),
                        help="Default root for a relative --dataset-path; read-only attached data")
    parser.add_argument("--dataset-path", type=Path, help="Exact official train.json, relative to input-dir or absolute")
    parser.add_argument("--hf-cache", type=Path, help="Default: output-dir/hf-cache")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--fp32-check", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="CPU-only official data validation; no model weights")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-batches", type=int)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args(argv)
    if len(set(args.devices)) != len(args.devices) or any(i < 0 for i in args.devices):
        parser.error("devices must be distinct nonnegative indices")
    for name in ("batch_size", "checkpoint_batches", "max_hours"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name} must be positive and finite")
    if args.max_hours is not None and args.max_hours > 7.5:
        parser.error("max_hours may only lower the conservative 7.5-hour script budget")
    return args


def run_precision(records, heads, config, base_metadata, directory, precision, devices, deadline):
    directory.mkdir(parents=True, exist_ok=True)
    shape = (len(records), len(heads))
    metadata = {**base_metadata, "precision": precision, "heads": heads,
                "record_ids": [r["record_id"] for r in records], "worker_devices": devices}
    metadata["identity"] = digest(metadata)
    metadata["complete"] = False
    final = directory / "effects.npz"
    if final.exists():
        values, saved = load_cache(final, metadata["identity"], shape)
        if not saved.get("complete") or not np.isfinite(values).all():
            raise ValueError("Invalid final cache")
        print(f"COMPLETE (already cached) {final}", flush=True)
        return final
    if should_stop(deadline, config["stop_margin_seconds"]):
        print(f"PAUSED before worker startup: {directory}", flush=True)
        return None
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, device, len(devices), records,
        heads, config, metadata, str(directory), precision, deadline)) for rank, device in enumerate(devices)]
    def request_stop(signum, frame):
        for process in processes:
            if process.pid and process.is_alive():
                process.terminate()  # POSIX SIGTERM is handled at safe worker boundaries.

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            for process in processes:
                process.join(timeout=0.2)
            if any(p.exitcode is not None and p.exitcode != 0 for p in processes):
                request_stop(None, None)
        failures = [p.exitcode for p in processes if p.exitcode != 0]
        if failures:
            raise RuntimeError(f"Worker failed (exit codes {failures}); inspect logs and traceback")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    paths = [directory / f"worker-{rank}.npz" for rank in range(len(devices))]
    if not all(path.exists() for path in paths):
        print(f"PAUSED {directory}; rerun the same command", flush=True)
        return None
    metadata["worker_diagnostics"] = [load_cache(p, metadata["identity"], shape)[1] for p in paths]
    complete = merge_shards(paths, metadata["identity"], shape, metadata, final)
    print(f"{'COMPLETE' if complete else 'PAUSED'} {final if complete else directory}", flush=True)
    return final if complete else None


def compare_precision(fp16_path, fp32_path, tolerance, report_path):
    with np.load(fp16_path, allow_pickle=False) as f:
        half = f["effects"]
        half_meta = json.loads(str(f["metadata"]))
    with np.load(fp32_path, allow_pickle=False) as f:
        full = f["effects"]
        full_meta = json.loads(str(f["metadata"]))
    for key in ("record_ids", "heads", "config_hash", "population_hash", "code_hash"):
        if half_meta[key] != full_meta[key]:
            raise ValueError(f"Precision cache {key} differs")
    errors = np.abs(half - full)
    passed = bool(np.isfinite(errors).all() and np.all(errors <= tolerance))
    report = {"passed": passed, "absolute_tolerance": tolerance,
              "max_abs_error": float(errors.max()), "mean_abs_error": float(errors.mean()),
              "per_head_max_abs_error": errors.max(axis=0).tolist(),
              "sign_disagreements": int(np.count_nonzero(np.sign(half) != np.sign(full))),
              "record_ids": half_meta["record_ids"], "heads": half_meta["heads"]}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'PRECISION_PASS' if passed else 'PRECISION_FAIL'} {report_path}", flush=True)
    if not passed:
        raise ValueError("fp16/fp32 bounded effects exceed the configured engineering tolerance")


def main(argv=None):
    started = time.time()
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("batch_size", "checkpoint_batches", "max_hours"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if config["population_rule"] != "sha256-within-template-round-robin-v1":
        raise ValueError("Unimplemented population rule")
    if config["model_id"] != "openai-community/gpt2":
        raise ValueError("This implementation is validated only for the specified GPT-2 model")
    if not 0 < config["max_hours"] <= 7.5 or config["checkpoint_batches"] < 1 or config["batch_size"] < 1:
        raise ValueError("Invalid operational config")
    deadline = started + config["max_hours"] * 3600
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = (args.hf_cache or args.output_dir / "hf-cache").resolve()
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Import after cache environment variables are set; children inherit them.
    from data.load_mib import prepare
    import torch
    packages = {name: version(name) for name in
                ("torch", "transformer-lens", "transformers", "datasets", "numpy", "huggingface-hub")}
    if not args.prepare_only:
        if not torch.cuda.is_available() or max(args.devices) >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA devices are unavailable; GPU cache must be run on Kaggle")
    dataset_path = args.dataset_path
    if dataset_path and not dataset_path.is_absolute():
        dataset_path = args.input_dir / dataset_path
    records, manifest, audit = prepare(config, args.output_dir / "preparation", hf_cache, dataset_path)
    if args.prepare_only:
        print("PREPARED (CPU only; no transformer forwards)", flush=True)
        return 0
    metadata = {"config": config, "config_hash": digest(config), "population_hash": manifest["population_hash"],
        "model_revision": config["model_revision"], "dataset_revision": config["dataset_revision"],
        "counterfactual": config["counterfactual"], "code_hash": code_hash(), "packages": packages,
        "preprocessing": audit, "cuda_runtime": torch.version.cuda,
        "matrix_layout": "effects[record_index, head_index]; one value per observed intervention",
        "hypothesis_source": "SPEC.md", "smoke": args.smoke, "fp32_check": args.fp32_check}
    if args.smoke:
        records, heads = records[:config["smoke_n"]], config["smoke_heads"]
        run_precision(records, heads, config, metadata, args.output_dir / "smoke", "fp16", args.devices, deadline)
    elif args.fp32_check:
        records, heads = records[:config["fp32_check_n"]], config["heads"]
        directory = args.output_dir / "fp32-check"
        half = run_precision(records, heads, config, metadata, directory / "fp16", "fp16", args.devices, deadline)
        if half is None:
            return 0
        full = run_precision(records, heads, config, metadata, directory / "fp32", "fp32", args.devices, deadline)
        if full:
            compare_precision(half, full, config["effect_agreement_atol"], directory / "agreement.json")
    else:
        run_precision(records, config["heads"], config, metadata, args.output_dir / "full", "fp16", args.devices, deadline)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
