"""Pinned official source acquisition and preprocessing audit."""
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from data.fixed_population import choose_population
from data.validate_counterfactuals import validate_record
from models.checkpoints import digest


def prepare(config, directory, hf_cache, dataset_path=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    official = Path(hf_hub_download(config["dataset_id"], "train.json", repo_type="dataset",
                                  revision=config["dataset_revision"], cache_dir=str(hf_cache)))
    source_sha = hashlib.sha256(official.read_bytes()).hexdigest()
    source = Path(dataset_path) if dataset_path else official
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise ValueError("Attached train.json differs from the pinned official source")
    dataset = load_dataset("json", data_files={"train": str(source)}, split="train",
                           cache_dir=str(hf_cache / "datasets"))
    if len(dataset) != 10000:
        raise ValueError("Official train record count differs from SPEC.md")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"],
                                             cache_dir=str(hf_cache))
    eligible, excluded = [], []
    for index, row in enumerate(dataset):
        try:
            eligible.append(validate_record(row, index, tokenizer, config))
        except ValueError as error:
            excluded.append({"source_index": index, "reason": str(error)})
    audit = {"source_records": len(dataset), "eligible": len(eligible), "excluded": len(excluded),
             "excluded_fraction": len(excluded) / len(dataset), "source_sha256": source_sha,
             "dataset_revision": config["dataset_revision"]}
    (directory / "preprocessing.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (directory / "exclusions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in excluded), encoding="utf-8")
    print(json.dumps(audit), flush=True)
    population = choose_population(eligible, config["population_n"])
    manifest = {"records": population, "population_hash": digest(population),
                "rule": config["population_rule"], "source_sha256": source_sha}
    frozen_path = Path(__file__).resolve().parents[1] / "reports/population_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (config["dataset_revision"] != frozen["dataset_revision"] or
        manifest["population_hash"] != frozen["population_hash"] or
        source_sha != frozen["source_sha256"] or
        [r["source_index"] for r in population] != frozen["source_indices"]):
        raise ValueError("Population differs from the frozen CPU-validated manifest; investigate before proceeding")
    path = directory / "population.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing population differs; use a new output directory")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return population, manifest, audit
