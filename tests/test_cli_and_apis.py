import inspect
import json
from pathlib import Path

import pytest

from kaggle.run_effect_cache import main, parse_args


def test_installed_api_contracts():
    from transformer_lens.model_bridge import TransformerBridge
    from datasets import load_dataset
    for parameter in ("revision", "dtype", "device"):
        assert parameter in inspect.signature(TransformerBridge.boot_transformers).parameters
    assert "fwd_hooks" in inspect.signature(TransformerBridge.run_with_hooks).parameters
    for parameter in ("split", "cache_dir", "data_files"):
        assert parameter in inspect.signature(load_dataset).parameters


@pytest.mark.parametrize("args", [["--devices", "0", "0"], ["--max-hours", "nan"],
    ["--checkpoint-batches", "0"], ["--batch-size", "-1"], ["--smoke", "--fp32-check"],
    ["--max-hours", "8"]])
def test_invalid_cli_args(args):
    with pytest.raises(SystemExit):
        parse_args(args)


def test_defaults_and_notebook_is_thin():
    args = parse_args([])
    assert str(args.output_dir).replace("\\", "/") == "/kaggle/working/stcci"
    assert args.devices == [0, 1]
    nb = json.loads(Path("kaggle/kaggle_runner.ipynb").read_text())
    assert all(not cell.get("outputs") for cell in nb["cells"])
    source = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert "def " not in source and "effect_batch" not in source


def test_cache_refuses_cpu_before_data_loading(tmp_path, monkeypatch):
    import torch
    import data.load_mib
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("GPU cache must fail before downloading data or loading models")

    monkeypatch.setattr(data.load_mib, "prepare", forbidden)
    with pytest.raises(RuntimeError, match="CUDA devices are unavailable"):
        main(["--smoke", "--output-dir", str(tmp_path)])
