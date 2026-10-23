# STCCI preparation and effect caching

Research protocol: [SPEC.md](SPEC.md). Start with [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md).
This stage implements data validation, frozen GPT-2 intervention plumbing and
resumable effect caching. Statistical methods, calibration, MESOI freezing,
the novelty gate and GO/KILL decisions are not implemented in this stage.

No GPU experiment has been run by Codex. Runtime validation must pass on your
Kaggle session before interpreting effects. CPU tests use unchanged official
records and mechanical tensor/index checks, never synthetic research data.

For offline tokenizer tests, first run
`python kaggle/run_effect_cache.py --prepare-only --output-dir outputs/cpu-preparation`
(CPU only, requires Internet to download the official data and tokenizer).
Run CPU tests with `python -m pytest -q`; real tokenizer tests explicitly skip
if that tokenizer cache is absent. Run the CLI with
`python kaggle/run_effect_cache.py --help`. All runtime paths are configurable.
Do not commit downloaded model weights, checkpoints, or output caches.
