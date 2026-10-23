# Changelog

## 2026-10-23 — notebook dependency-conflict fix

* Diagnosed supplied user logs: torchvision required torch 2.11.0 while the
  shared notebook environment had installed torch 2.14.1; model imports failed
  on torchvision::nms before GPT-2 loaded. Pip also reported unrelated shared
  package conflicts. The earlier global install instructions were inadequate.
* Added a standard-library setup helper that installs unchanged requirement
  versions in an isolated venv, selecting the official torch 2.14.1+cu126 build.
  Verified Python 3.13 Linux wheel listing on the official PyTorch index.
* Added import/environment checks, optional two-device CUDA tensor probes,
  and parent-process import validation before downloading data or spawning workers.
  Updated notebook and guide to use the isolated interpreter with checked exits.
* No hypothesis, estimand, MESOI, frozen scientific config or requirement
  version changed. Dataset/model revisions and the population manifest remain fixed.
* Tested/untested details for this fix: reports/environment_fix.md. GPU probes,
  isolated Linux installation and all model forwards remain untested locally.

## 2026-10-23

* Prepared the initially empty repo for the requested first implementation stage.
* Installed and inspected pinned TransformerLens 4.2.0 / datasets 5.1.0 APIs;
  froze the resolved CPU environment and official model/dataset revisions.
* Validated all 10,000 official training records on CPU with the real tokenizer.
  All passed implemented eligibility checks; selected 1,024 records from 10
  templates with the user-delegated deterministic rule. Saved a frozen manifest.
* Implemented last-token head patching, runtime HF/logit/hook/no-op gates,
  independent GPU row shards, atomic checkpoints, automatic resume, deadline
  handling, progress/memory logs and a single merged matrix/metadata/result table.
* Added separate smoke and fp32-check modes, a thin notebook and Kaggle guide.
* Tested: 34 CPU unit tests, offline repeated population preparation, compilation,
  CLI help and pip dependency consistency. Initial system-temp rename failures
  were resolved by using repo-local pytest temporary files.
* Untested: Linux/Kaggle/CUDA installation, GPT-2 forwards, actual hook/no-op
  numerical behavior, GPU multiprocessing/resume/termination, fp16/fp32 agreement,
  runtime/VRAM/ETA and final research cache. No GPU experiments were executed.
* Stopped before statistical inference, tau computation, novelty conclusions,
  calibration, confirmatory experiments or research decisions.
