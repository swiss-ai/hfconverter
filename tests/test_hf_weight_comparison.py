"""Pure-Python tests for comparison metadata/index helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("compare_hf_weights.py")
_SPEC = importlib.util.spec_from_file_location("compare_hf_weights", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_comparator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_comparator)


def test_validate_index_accepts_exact_header_keymap() -> None:
    available = {
        "layer.0.weight": "model-00001-of-00002.safetensors",
        "norm.weight": "model-00002-of-00002.safetensors",
    }
    index = {"metadata": {"total_size": 128}, "weight_map": available}

    assert _comparator.validate_index(index, available) == available


@pytest.mark.parametrize(
    ("weight_map", "available", "message"),
    [
        (
            {"indexed.only": "model-00001-of-00001.safetensors"},
            {},
            "absent from shard headers",
        ),
        ({}, {"header.only": "model.safetensors"}, "absent from index"),
        (
            {"weight": "wrong.safetensors"},
            {"weight": "model.safetensors"},
            "wrong shard",
        ),
    ],
)
def test_validate_index_rejects_inconsistent_mapping(
    weight_map: dict[str, str], available: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _comparator.validate_index({"weight_map": weight_map}, available)


def test_validate_index_requires_string_mapping() -> None:
    with pytest.raises(ValueError, match="string-to-string"):
        _comparator.validate_index({"weight_map": {"weight": 12}}, {})


def test_metadata_serializes_shape_and_dtype() -> None:
    info = (Path("model.safetensors"), (2, 3), "F32")

    assert _comparator._metadata(info) == {"shape": [2, 3], "dtype": "F32"}
