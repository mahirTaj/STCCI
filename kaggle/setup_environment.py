"""Install pinned research dependencies in an isolated environment, never the notebook kernel."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
TORCH_WHEEL = "torch==2.14.1+cu126"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"


def isolated_python(directory):
    directory = Path(directory)
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def check_venv_config(directory):
    config = Path(directory) / "pyvenv.cfg"
    if config.exists():
        settings = dict(line.split("=", 1) for line in config.read_text().splitlines() if "=" in line)
        normalized = {key.strip(): value.strip().lower() for key, value in settings.items()}
        if normalized.get("include-system-site-packages") != "false":
            raise ValueError("Refusing to reuse an environment that exposes system site packages; choose a new --venv-dir")


def setup(directory):
    directory = Path(directory).resolve()
    if directory.exists() and not (directory / "pyvenv.cfg").exists():
        raise ValueError("Existing target is not a virtual environment; choose a new --venv-dir")
    check_venv_config(directory)
    # Avoid ensurepip: the supplied notebook traceback shows that bootstrap fails.
    # pip --python explicitly supports managing a venv with no pip installed.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(directory)
    python = str(isolated_python(directory))
    pip = [sys.executable, "-m", "pip", "--python", python]
    commands = [
        [*pip, "--version"],
        [*pip, "install", TORCH_WHEEL, "--index-url", TORCH_INDEX],
        [*pip, "install", "-r", str(ROOT / "requirements.txt")],
        [*pip, "check"],
        [python, str(ROOT / "kaggle/check_environment.py"), "--require-isolated"],
    ]
    for command in commands:
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    print(f"ENVIRONMENT_READY {python}", flush=True)
    print("Next: run this interpreter with kaggle/check_environment.py --require-isolated --require-cuda", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-dir", type=Path, default=Path("/kaggle/working/stcci-venv"))
    args = parser.parse_args(argv)
    setup(args.venv_dir)


if __name__ == "__main__":
    main()
