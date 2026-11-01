"""config.py's promise, enforced: the knobs live in one place.

`config.py` has named this file in its docstring from the start. A hardcoded
tag would not fail loudly -- it would silently diverge from the run every
other module describes, and the numbers would stop being comparable.
"""

import re
from pathlib import Path

from memproj import config

PKG = Path(config.__file__).resolve().parent


def _modules():
    for path in PKG.rglob("*.py"):
        if path.name == "config.py":
            continue
        yield path, path.read_text(encoding="utf-8")


def test_no_module_but_config_names_a_model():
    tags = {config.MODEL, config.EMBED_MODEL}
    for path, text in _modules():
        found = {tag for tag in tags if tag in text}
        assert not found, f"{path.name} hardcodes {sorted(found)}"


def test_no_module_but_config_pins_a_context_length():
    literal_ctx = re.compile(r'"num_ctx"\s*:\s*\d+')
    for path, text in _modules():
        assert not literal_ctx.search(text), \
            f"{path.name} hardcodes num_ctx; import config.ollama_options()"


def test_ollama_options_carries_the_three_pinned_settings():
    assert config.ollama_options() == {
        "num_ctx": config.NUM_CTX,
        "temperature": config.TEMPERATURE,
        "seed": config.SEED,
    }


def test_the_four_experiment_knobs_exist():
    for name in ("MODEL", "NUM_CTX", "TEMPERATURE", "SEED"):
        assert hasattr(config, name), f"config.{name} is missing"
    assert config.TEMPERATURE == 0.0
