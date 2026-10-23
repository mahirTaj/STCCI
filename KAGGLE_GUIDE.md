# Kaggle effect-cache guide

This is a preparation workflow, not a report of a successful GPU run. Check your
account's current accelerator availability, quota, session deadline, persistence
rules and Internet access in the Kaggle UI. No quota or platform time allowance
is assumed here. The script's own default time budget is 7.5 hours including
startup; it reserves another 5 minutes before starting further work. Download
time or an in-flight CUDA call cannot be guaranteed to finish by a deadline.

## Recovery from the reported torchvision import failure

The failed run reached preprocessing but stopped before loading GPT-2. The
updated setup below isolates dependencies. First publish the local fix from
PowerShell in your local repo:

```powershell
git add CHANGELOG.md KAGGLE_GUIDE.md kaggle models/environment.py reports/environment_fix.md tests/test_environment_setup.py
git commit -m "Isolate Kaggle dependencies and check model imports early"
git push
```

Then in the existing notebook, replace the old install/run cells with:

```python
%cd /kaggle/working/STCCI
!git pull --ff-only
```

Confirm the pull succeeds, then run:

```python
import subprocess
import sys
subprocess.run([sys.executable, 'kaggle/setup_environment.py'], check=True)
STCCI_PYTHON = '/kaggle/working/stcci-venv/bin/python'
subprocess.run([STCCI_PYTHON, 'kaggle/check_environment.py',
                '--require-isolated', '--require-cuda'], check=True)
subprocess.run([STCCI_PYTHON, 'kaggle/run_effect_cache.py', '--smoke'], check=True)
```

Expected gates: ENVIRONMENT_READY, ENVIRONMENT_CHECK_PASS, then smoke COMPLETE.
Do not run the full cache until these pass. Keep the existing stcci data/download
directory; do not remove it to repair imports. If you have successful effect
checkpoints from an earlier code version, retain that code for resumption or
use a separate output directory for this fix; code identity checks intentionally
reject mixing them. The supplied failing run did not reach model loading.

## 1. Push this folder to GitHub

Create an empty GitHub repository in your browser. From PowerShell in this folder:

```powershell
git add .
git commit -m "Prepare STCCI Kaggle effect cache"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/STCCI.git
git push -u origin main
```

Replace YOUR_NAME. If `origin` already exists, inspect `git remote -v` and use
`git remote set-url origin ...` only if it is the repository you intend to use.
Expected: Git prints an uploaded branch. Check GitHub contains SPEC.md,
requirements.txt, models/, data/, kaggle/, and configs/. Generated files and
the local .venv are ignored. Authenticate through Git/GitHub's normal credential
flow; never paste a personal access token into a public notebook.

## 2. Create a notebook

In Kaggle create a Python notebook, or import `kaggle/kaggle_runner.ipynb`.
In its settings choose the available **GPU T4 x2** accelerator and enable Internet.
Verify the displayed accelerator and your remaining allowance yourself. If x2
is unavailable, stop and resolve the account/settings issue; the default runner
requires two independent CUDA devices.

## 3. Clone and install (paste as separate cells)

```python
!nvidia-smi
```

Expected: two devices in the report. This command reports the actual session;
the guide makes no guarantee about the hardware assigned by Kaggle.

```python
REPO_URL = "https://github.com/YOUR_NAME/STCCI.git"
!git clone {REPO_URL} /kaggle/working/STCCI
%cd /kaggle/working/STCCI
!git rev-parse HEAD
```

Expected: clone completes and prints a commit hash. Record this hash. If the
folder exists after reconnecting, use `%cd /kaggle/working/STCCI` and inspect
`!git status` and `!git rev-parse HEAD`; do not delete the folder or blindly pull
new code into a partially completed cache run. For exact replay, check out the
original commit in a fresh clone.

```python
import subprocess
import sys
subprocess.run([sys.executable, 'kaggle/setup_environment.py'], check=True)
STCCI_PYTHON = '/kaggle/working/stcci-venv/bin/python'
subprocess.run([STCCI_PYTHON, 'kaggle/check_environment.py',
                '--require-isolated', '--require-cuda'], check=True)
```

Expected: `No broken requirements found.`, `ENVIRONMENT_READY`, then
`ENVIRONMENT_CHECK_PASS (no model forwards)`. The final check prints the exact
interpreter, package versions, CUDA runtime, and both assigned device names. It
runs a tiny fp16 tensor operation on each GPU, without loading GPT-2 or any data.
Stop on any error before running smoke.

The setup helper creates a virtual environment **without system site packages**
at /kaggle/working/stcci-venv, with `with_pip=False` to bypass the reported
ensurepip bootstrap failure. It uses the existing interpreter's pip with its
documented `--python` option to install into that isolated environment; it does
not install into the existing interpreter. This also works for a venv with no
pip of its own. Documentation: https://pip.pypa.io/en/stable/topics/python-option/
It installs the unchanged torch 2.14.1 version
using the explicit `+cu126` build from PyTorch's official CUDA 12.6 index, then
the unchanged requirements.txt. It never installs into the notebook kernel.
Official Python 3.13 Linux wheel availability was verified on 2026-10-23 at
https://download.pytorch.org/whl/cu126/torch/; execution on your GPUs remains
unverified until this check passes. The earlier generic install pulled CUDA 13
packages in the supplied user log; this setup explicitly selects CUDA 12.6.

Always use STCCI_PYTHON below. Using bare `python` again selects the shared
environment and can reproduce the original conflict. No kernel restart is
needed for the external interpreter to use its newly installed packages. The
notebook's own shared environment is not repaired by this helper.

## 4. Run smoke first

```python
subprocess.run([STCCI_PYTHON, 'kaggle/run_effect_cache.py', '--smoke'], check=True)
```

Expected (illustrative messages, not observed GPU output): preprocessing counts
and exclusions, then two worker logs containing `validated`, `completed`,
`eta_seconds`, `allocated_mb`, `reserved_mb`, and `elapsed_seconds`. Success
prints `COMPLETE` and `/kaggle/working/stcci/smoke/effects.npz` (8 records,
2 heads). Minutes is a smoke design target, not measured runtime. The CLI
checks HF/bridge logits, hook dimensions, no-op replacement, and finite bounded
effects before writing results. Read any traceback; do not proceed on failure.

`check=True` raises if the CLI exits with an error. Also look for COMPLETE and
the final file; a clean time-budget pause returns successfully with PAUSED.

## 5. Check fp16/fp32 agreement

```python
subprocess.run([STCCI_PYTHON, 'kaggle/run_effect_cache.py', '--fp32-check'], check=True)
```

Expected: separate resumable fp16/fp32 caches on the fixed first 8 population
records across all 24 heads, then `PRECISION_PASS` or a nonzero failure and an
agreement report. This compares bounded effects as float32 with the absolute
tolerance recorded in config. It also reports sign disagreements and per-head
errors; inspect these before any later analysis. This is an engineering gate,
not evidence of statistical coverage or general fp16 accuracy.

## 6. Run the full cache

```python
subprocess.run([STCCI_PYTHON, 'kaggle/run_effect_cache.py'], check=True)
```

Expected: same checks, 1,024 records and 24 heads; two independent processes each
own disjoint record rows. Checkpoints default to every 5 completed batches.
`COMPLETE` means all entries are finite and the merged `effects.npz` exists.
`PAUSED` means the script stopped for its time budget or a termination request;
run the same command again with the same config and output directory.

Paths and operational settings:

```python
subprocess.run([STCCI_PYTHON, 'kaggle/run_effect_cache.py',
    '--output-dir', '/kaggle/working/stcci', '--input-dir', '/kaggle/input',
    '--checkpoint-batches', '1', '--max-hours', '7'], check=True)
```

Outputs default to /kaggle/working/stcci; attached resume data defaults to
/kaggle/input. Hugging Face downloads use an output-local HF cache. Optional
`--dataset-path /kaggle/input/YOUR_DATA/train.json` accepts only the exact pinned
official train JSON; its SHA-256 must match the downloaded revision. The script
does not silently trust a relabeled or modified attached dataset.

## 7. Watch progress

Worker messages print as batches finish. To inspect saved logs in a separate
cell when the session permits concurrent cells, or after the run:

```python
from pathlib import Path
for p in Path('/kaggle/working/stcci').rglob('worker-*.log'):
    print(p, '\n', '\n'.join(p.read_text().splitlines()[-3:]))
!nvidia-smi
```

ETA is based on completed batches, excluding uncertain future overhead; it is
not a platform guarantee. Long first downloads are not a hung effect batch.

## 8. If the session dies or pauses

If the files still exist, run the exact same command. It automatically reloads
the matching worker checkpoints and skips finished batches. Up to the unsaved
checkpoint interval can be recomputed after an abrupt kill. A cooperative stop
saves at the next safe batch boundary. A GPU call already running cannot be
interrupted safely by the Python deadline logic.

If the environment is being discarded, save/download the **entire**
`/kaggle/working/stcci` directory first, including population, preparation audit,
worker checkpoints and logs. Verify Kaggle's current save/version and output
retention behavior yourself. A GitHub clone alone does not restore checkpoints.
Create a Kaggle dataset from that directory if desired, then attach it to the
next notebook. Restore by copying it into the writable output path:

```python
import shutil
shutil.copytree('/kaggle/input/YOUR_RESUME_DATASET/stcci',
                '/kaggle/working/stcci', dirs_exist_ok=True)
```

Adjust the source to the actual attached directory layout. Copy only into a
fresh output directory; do not combine checkpoints from unrelated runs. Then
clone the same code commit, install the same requirements, and rerun. A config,
population or code identity mismatch is an error; do not edit checkpoint metadata
to bypass it. Use a separate output directory for a new run.

## 9. Download and bring the cache back

After COMPLETE, the final file is:

`/kaggle/working/stcci/full/effects.npz`

In Kaggle's output/file panel download this file (save a notebook version with
outputs if needed, verifying the current UI). Preserve the full output directory
as an audit/resume artifact. Alternatively create a Kaggle dataset from these
outputs and attach it to a later CPU statistics notebook; no GPU is needed to
read the matrix.

Locally create `outputs/full/` under this repo and copy the downloaded file there.
Read it without pickle:

```python
import json
import numpy as np
with np.load('outputs/full/effects.npz', allow_pickle=False) as f:
    Y = f['effects']
    metadata = json.loads(str(f['metadata']))
assert Y.shape == (1024, 24) and np.isfinite(Y).all()
print(Y.shape, metadata['model_revision'], metadata['config_hash'])
```

Expected: `(1024, 24)` and the pinned revision/config hashes. The metadata
contains ordered record IDs, heads, full processing config, population hash,
validation diagnostics, code hash and package versions. Matrix position `[i,j]`
is one observed intervention result for record ID i and head j. No tau or
statistics are computed in this stage. The same file also contains
`observations`, a numeric table with one row per observed intervention
(`record_index`, `head_index`, `effect`), linked to the metadata IDs and heads.

## Common errors

* **No CUDA/two devices:** check accelerator settings and actual device count.
  CPU execution of the cache is intentionally refused. `--devices 0` is available
  for a deliberate single-device diagnostic in a separate output directory.
* **Internet/download/authentication:** check Internet settings and public URLs
  in reports/sources.md. Do not add credentials to code. Retry the same revision.
* **CUDA out of memory:** stop and inspect other allocations. A smaller
  `--batch-size` is supported, but changes the run identity; use a fresh output
  directory and keep the validation report. Memory figures are measured logs.
* **API/hook/logit/no-op mismatch:** stop; keep traceback, package versions and
  validation output. Do not switch hook names or libraries to get past the gate.
* **Too few eligible records:** review preprocessing.json and exclusions.jsonl.
  Do not lower N or generate replacement records for the full study.
* **Checkpoint mismatch/corruption:** restore a consistent backup or start in a
  fresh directory; never merge different configs or population manifests.
* **PAUSED:** preserve files and rerun the same command. **Worker failed:** inspect
  logs/tracebacks and rerun only after diagnosing; no complete matrix is emitted.
* **`torchvision::nms does not exist` / `Could not import PreTrainedModel`:**
  the supplied failure log shows torchvision 0.26.0+cu128 requiring torch 2.11.0
  alongside installed torch 2.14.1. Transformers' import reached torchvision
  and failed before GPT-2 loaded. Run the isolated setup above and use its
  interpreter for the check and smoke. Do not uninstall random notebook packages
  or change the research config to bypass it.
* **Pandas/bottleneck warning or pip conflicts with Colab/RAPIDS/gradio/etc.:**
  these appeared in the supplied shared-environment log. The isolated environment
  excludes those unrelated preinstalled packages. To check it, run
  `subprocess.run([sys.executable, '-m', 'pip', '--python', STCCI_PYTHON, 'check'], check=True)`
  in a Kaggle Code cell after the setup cell. A shared-kernel pip check without
  `--python` does not check the research venv.
* **`ensurepip` exits with status 1:** update the repo and rerun the same setup
  cell. The revised helper creates the venv without ensurepip and routes existing
  pip into it using `--python`. It can reuse the partially created isolated venv;
  do not delete outputs or change requirements. The supplied traceback omitted
  ensurepip's underlying output, so its exact cause is undetermined.
* **Existing pip missing or `--python` unsupported / venv creation fails:** keep
  the full printed error and report it. The helper stops before installing
  research packages; it does not fall back to the shared environment.
* **CUDA check fails after imports succeed:** keep the printed build versions,
  traceback and `nvidia-smi` output. Wheel availability is not proof of driver
  or GPU compatibility; do not proceed to full caching.
