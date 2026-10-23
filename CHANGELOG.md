# Changelog

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
