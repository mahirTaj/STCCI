import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/kill_test.json").read_text())


@pytest.fixture
def official_rows():
    return json.loads((ROOT / "tests/fixtures/official_records.json").read_text())


@pytest.fixture
def tokenizer(config):
    from transformers import AutoTokenizer
    # Offline test: use the real pinned tokenizer from the documented CPU preparation.
    import os
    cache = Path(os.environ.get("STCCI_TEST_HF_CACHE", ROOT / "outputs/cpu-preparation/hf-cache"))
    try:
        return AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"],
                                            cache_dir=str(cache), local_files_only=True)
    except OSError:
        pytest.skip("Real tokenizer not cached: run --prepare-only first; no mock tokenizer is substituted")
