# Notebook environment fix — 2026-10-23

## Evidence from user-supplied logs

The run validated 10,000 source records with zero exclusions, then both workers
failed importing TransformerLens. The nested exception was
`RuntimeError: operator torchvision::nms does not exist`; the outer exception
reported failure to import transformers.PreTrainedModel. The install log states
that torchvision 0.26.0+cu128 requires torch==2.11.0, while torch 2.14.1 was
installed. It also lists multiple conflicts with unrelated packages in the
shared notebook environment, and installed CUDA 13 packages.

The pandas/bottleneck warning and unauthenticated HF warning are separate from
the terminating torchvision traceback. These logs do not show successful model
loading, hook validation or effects. They show Python 3.13 paths and packages
named google-colab; they do not by themselves establish the assigned GPUs.

## Changes

* Keep requirements.txt and all scientific configuration/revisions unchanged.
* Create /kaggle/working/stcci-venv without system site packages. Never install
  research requirements into the notebook kernel's shared environment.
* Install torch==2.14.1+cu126 from the official CUDA 12.6 index before installing
  the existing requirements. The installed packaging library confirms this
  local build satisfies the unchanged torch==2.14.1 version specifier.
* Official listing checked on 2026-10-23:
  https://download.pytorch.org/whl/cu126/torch/
  includes torch-2.14.1+cu126-cp313-cp313-manylinux_2_28_x86_64.whl.
  Availability is not a claim of compatibility with the user's assigned driver.
* Run pip check and real model-library imports in that interpreter. Optionally
  require the expected CUDA build, two devices and a tiny fp16 tensor operation
  on both devices. These probes do not load models or create effect observations.
* Fail on model import errors in the parent before preprocessing/workers.
* Use the isolated interpreter throughout the thin notebook; checked subprocess
  calls surface command failures to the notebook.

## Tested — CPU only

`python -m pytest -q`: **41 passed**, no skips. Includes seven new environment
tests. `python kaggle/check_environment.py --require-isolated` passed against
the existing Windows Python 3.12.3 / torch 2.14.1+cpu environment, importing the
actual TransformerBridge and HF model class without loading weights.
`python -m compileall -q data models kaggle tests`, `python -m pip check` and
`git diff --check` passed. Git diff confirmed requirements.txt, scientific
config and frozen population manifest unchanged.

The setup command tests mock process launches: they check isolation, exact commands,
unchanged pins, failure propagation, original exception preservation and
preprocessing refusal on model import failure; they do not install a Linux
environment or simulate a GPU experiment.

## Untested

The isolated Python 3.13 Linux install, CUDA 12.6/driver compatibility, two-GPU
tensor probes, GPT-2 forwards, model hook/no-op checks, fp16/fp32 agreement,
effect caching, runtime and GPU resume remain pending user execution. No GPU
experiment was run by Codex. See KAGGLE_GUIDE.md for the exact recovery cells.

Existing official-data downloads can be retained. Code fingerprints change
with this fix, so prior successful effect checkpoints, if any, must remain
associated with their original code; never edit their metadata to reuse them
under a changed fingerprint. The supplied traceback failed before model loading.
