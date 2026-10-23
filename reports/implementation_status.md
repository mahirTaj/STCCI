# Implementation status — 2026-10-23

Requested scope: first implementation steps 1–7 in SPEC.md plus the Kaggle
cache infrastructure, smoke/precision modes, notebook and beginner guide.
Steps 5–7 have implementation and runtime gates; their model-forward validation
remains pending on Kaggle. Do not interpret source inspection or tensor unit
tests as successful GPT-2 experiments.

## Tested — CPU only

Environment: Windows, Python 3.12.3, torch 2.14.1+cpu, TransformerLens 4.2.0,
transformers 5.20.2, datasets 5.1.0. Full resolved versions: requirements.txt.

* Installed API signatures inspected before writing integrations: datasets
  `load_dataset`; HF `hf_hub_download` / `AutoTokenizer.from_pretrained`;
  TransformerBridge `boot_transformers`, `forward`, `run_with_hooks`,
  `run_with_cache`. Source inspection covered the HF boot loader and hook_z
  reshaping. No incompatible API requiring a SPEC workaround was encountered.
* `python -m pytest -q`: **34 passed**, no skipped tests, on the latest run.
  Tests cover real official-record tokenization and validation, deterministic
  template selection, masked padding, exact tensor coordinate replacement,
  no-op tensor identity, fp32 score arithmetic, checkpoint round trips,
  interrupted writes, identity and range rejection, disjoint shard assignment,
  complete-only merge and observation layout, deadline boundaries, precision
  report gates, invalid CLI inputs, CPU refusal, thin notebook structure and
  installed API contracts. Storage/index sentinels are unit-test inputs, not
  fabricated research observations or simulated statistical experiments.
* `python -m compileall -q data models kaggle tests`: passed.
* `python -m pip check`: `No broken requirements found.`
* `python kaggle/run_effect_cache.py --help`: passed.
* Actual CPU preparation:
  `python kaggle/run_effect_cache.py --prepare-only --output-dir outputs/cpu-preparation`.
  Read official pinned train split and actual pinned GPT-2 tokenizer; no model
  weights or model forwards. 10,000 source records, 10,000 eligible, zero
  exclusions (0%). Deterministically selected 1,024 records from 10 templates.
  Repeated successfully offline and checked the tracked frozen manifest.
* Population SHA-256:
  `dbf5ef395ca86ea09f8782fd5d1697ca4a5252307ac4f8be60d2359dbbc39c93`.
  Source JSON SHA-256:
  `2ee41861d725eee2b71da3846872d5e5ea29d9d8c9aa3a8954f8430584b2f243`.
  IDs and rule: reports/population_manifest.json. Full local tokens/manifest and
  preprocessing/exclusion files: outputs/cpu-preparation/preparation/ (ignored
  by Git, regenerated from the official revision on Kaggle).

Initial test run: six atomic-rename tests failed because Windows sandbox access
to its system temporary directory denied `os.replace`. Moving pytest temporary
files into `outputs/pytest` resolved those failures; atomic writes were retained.
Source downloads/install required network access; offline preparation replay
used the existing official source/tokenizer cache.

## Untested — user must run on Kaggle

* Linux dependency installation and CUDA compatibility of the pinned environment.
* Pretrained GPT-2 loading, bridge/reference logits, actual hook firing/shapes,
  no-op model output agreement and causal head-output replacement.
* Smoke-mode effects and completion, fp16/fp32 model-effect agreement, full cache.
* Two-device process execution, actual GPU checkpoint/resume, cooperative SIGTERM,
  deadline behavior around in-flight CUDA calls, measured VRAM/runtime and ETA.
* Kaggle account accelerator availability, quota, deadline and persistence rules;
  inspect your current account/UI. The script's 7.5-hour cap is its own budget,
  not a claim about Kaggle limits. Download/kernel stalls can exceed a soft stop.

No GPU experiment, final effect matrix, finite-population tau, statistical
coverage, power, TOST, superiority, novelty or GO/KILL result is reported.
The runtime validation thresholds are initial engineering gates, not measured
agreement or changed scientific MESOIs. Hypothesis and estimand remain as in
SPEC.md. Statistical inference and novelty work remain outside this step.
