"""CPU-only tests of environment isolation, fail-fast checks and setup commands."""
from pathlib import Path
import subprocess

from packaging.specifiers import SpecifierSet
import pytest

from kaggle import setup_environment


def test_setup_installs_into_venv_and_stops_on_errors(tmp_path, monkeypatch):
    target = tmp_path / "research-venv"
    builds, commands = [], []

    class Builder:
        def __init__(self, **kwargs):
            builds.append(kwargs)

        def create(self, directory):
            assert directory == target

    def run(command, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(setup_environment.venv, "EnvBuilder", Builder)
    monkeypatch.setattr(setup_environment.subprocess, "run", run)
    setup_environment.setup(target)
    assert builds == [{"with_pip": True, "system_site_packages": False}]
    assert len(commands) == 4
    python = str(setup_environment.isolated_python(target))
    assert all(command[0] == python for command in commands)
    assert commands[0][4:] == ["torch==2.14.1+cu126", "--index-url", "https://download.pytorch.org/whl/cu126"]
    assert commands[1][-1] == str(setup_environment.ROOT / "requirements.txt")
    assert commands[2][1:] == ["-m", "pip", "check"]
    assert commands[3][-1] == "--require-isolated"

    commands.clear()

    def fail(command, check):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup_environment.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        setup_environment.setup(target)
    assert len(commands) == 1


def test_setup_refuses_system_site_packages(tmp_path):
    (tmp_path / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(ValueError, match="system site packages"):
        setup_environment.check_venv_config(tmp_path)
    (tmp_path / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    setup_environment.check_venv_config(tmp_path)


def test_setup_does_not_overwrite_unrelated_directory(tmp_path):
    with pytest.raises(ValueError, match="not a virtual environment"):
        setup_environment.setup(tmp_path)


def test_cuda_build_satisfies_unchanged_torch_pin():
    pins = (setup_environment.ROOT / "requirements.txt").read_text().splitlines()
    assert "torch==2.14.1" in pins
    assert SpecifierSet("==2.14.1").contains("2.14.1+cu126")


def test_model_import_error_preserves_original_cause(monkeypatch):
    import builtins
    from models.environment import verify_model_imports
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "transformer_lens.model_bridge":
            raise RuntimeError("unit-test missing torchvision::nms")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(RuntimeError, match="setup_environment.py") as caught:
        verify_model_imports()
    assert "torchvision::nms" in str(caught.value.__cause__)


def test_runner_checks_model_imports_before_data_loading(tmp_path, monkeypatch):
    import torch
    import data.load_mib
    import models.environment
    from kaggle.run_effect_cache import main

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def broken():
        raise RuntimeError("unit-test import failure")

    def forbidden(*args, **kwargs):
        pytest.fail("Data loading must not start when the model stack cannot import")

    monkeypatch.setattr(models.environment, "verify_model_imports", broken)
    monkeypatch.setattr(data.load_mib, "prepare", forbidden)
    with pytest.raises(RuntimeError, match="unit-test import failure"):
        main(["--smoke", "--output-dir", str(tmp_path)])


def test_notebook_uses_isolated_interpreter():
    import json
    nb = json.loads(Path("kaggle/kaggle_runner.ipynb").read_text())
    by_id = {cell["id"]: "".join(cell["source"]) for cell in nb["cells"]}
    assert "setup_environment.py" in by_id["cell-3"]
    assert "--require-cuda" in by_id["cell-3"]
    for cell_id in ("cell-4", "cell-5", "cell-6"):
        assert "subprocess.run([STCCI_PYTHON" in by_id[cell_id]
        assert "check=True" in by_id[cell_id]
