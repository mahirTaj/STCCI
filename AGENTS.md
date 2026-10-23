# AGENTS.md

## Project
Implement the research project described in SPEC.md (STCCI: selective inference after adaptive circuit discovery). SPEC.md is the source of truth.

## Rules
- Never report an experiment as run unless it was actually run. Separate "tested" from "untested" in every report.
- Never change the hypothesis, estimand, MESOI, or frozen configs without asking me first.
- Do not create, annotate, or synthesize data. Use only the existing datasets named in SPEC.md.
- Verify library APIs (TransformerLens, Hugging Face datasets) against the installed version before writing code that depends on them. If something in SPEC.md doesn't match reality, tell me instead of working around it silently.
- Do not claim facts about Kaggle limits or library behavior unless verified. Tell me what I should check myself.
- Work in small steps. Finish the step I asked for, report, and stop.

## Execution environment
- You have no GPU. Write code, run CPU unit tests, and prepare the repo.
- All GPU work runs on Kaggle (2x T4, 16 GB each, not pooled, fp16 only, no bf16) in a notebook that I run myself.
- Keep GPU code resumable (checkpoint and auto-resume), configurable by path, and runnable as a CLI script. Keep notebooks thin; put logic in the repo.
- Statistics run on CPU from the cached effect matrix.

## Style
- Python, PyTorch, pinned dependencies in requirements.txt.
- Add or update unit tests for anything you change.
- Keep a short CHANGELOG of what was done in each session.