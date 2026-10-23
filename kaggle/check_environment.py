"""Environment/import checks; optional CUDA tensor check, no models or datasets."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(require_isolated=False, require_cuda=False):
    if require_isolated:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("Use /kaggle/working/stcci-venv/bin/python, not the notebook kernel's python")
        from kaggle.setup_environment import check_venv_config
        check_venv_config(sys.prefix)
    from models.environment import verify_model_imports
    verify_model_imports()
    import torch

    info = {"python": sys.executable, "python_version": sys.version,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "packages": {name: version(name) for name in
                         ("transformer-lens", "transformers", "datasets", "numpy", "pandas")}}
    if require_cuda:
        if torch.version.cuda != "12.6" or torch.__version__.split("+")[0] != "2.14.1":
            raise RuntimeError("Expected the pinned torch 2.14.1 CUDA 12.6 build; rerun setup_environment.py")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("Two CUDA devices are required; check accelerator settings and nvidia-smi")
        info["devices"] = []
        for index in (0, 1):
            # An actual small fp16 kernel checks this installed build against the assigned GPU/driver.
            # This is an environment probe, never a transformer experiment or effect observation.
            with torch.cuda.device(index):
                tensor = torch.ones((2, 2), dtype=torch.float16, device=f"cuda:{index}")
                result = (tensor @ tensor).float().cpu()
                if not bool((result == 2).all()):
                    raise RuntimeError(f"CUDA tensor check failed on device {index}")
                info["devices"].append({"index": index, "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index))})
    print(json.dumps(info, indent=2), flush=True)
    print("ENVIRONMENT_CHECK_PASS (no model forwards)", flush=True)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-isolated", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    check(args.require_isolated, args.require_cuda)


if __name__ == "__main__":
    main()
