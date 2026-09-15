"""Regression coverage for fork tokenizer metadata in tensor-loading passes."""

import pickle
import sys
from argparse import Namespace
from dataclasses import make_dataclass
from types import ModuleType

import pytest
import torch

pytest.importorskip("megatron.core")

import megatron_mock
from exporter import reader


class _UnexpectedMetadata:
    pass


@pytest.mark.parametrize("with_tokenizer_metadata", [False, True])
def test_load_selected_tensors_with_fork_metadata(
    tmp_path, dist_env, monkeypatch, with_tokenizer_metadata
):
    checkpoint = tmp_path / "iter_0000100"
    expected = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    args = Namespace(hidden_size=4)
    module_name = "megatron.core.tokenizers.utils.tokenizer_extra_metadata"
    with monkeypatch.context() as patch:
        if with_tokenizer_metadata:
            for end in range(3, len(module_name.split(".")) + 1):
                name = ".".join(module_name.split(".")[:end])
                if name not in sys.modules:
                    patch.setitem(sys.modules, name, ModuleType(name))
            module = sys.modules[module_name]
            special = make_dataclass(
                "ModelSpecialTokens", ["full_ids"], frozen=True,
            )
            extra = make_dataclass(
                "TokenizerExtraMetadata", ["special_tokens"], frozen=True,
            )
            special.__module__ = extra.__module__ = module_name
            patch.setattr(module, "ModelSpecialTokens", special, raising=False)
            patch.setattr(module, "TokenizerExtraMetadata", extra, raising=False)
            args.tokenizer_extra_metadata = extra(special([1, 2]))
        megatron_mock.save_synthetic_checkpoint(
            {"model.weight": expected}, args, checkpoint,
            extra_tensors={"optimizer.state": torch.ones(64)},
        )

    # The fork-only classes are unavailable when the stock container loads the file.
    allowed_before = set(torch.serialization.get_safe_globals())
    loaded_args, _ = reader.load_args(checkpoint)
    assert loaded_args.hidden_size == 4
    for _ in range(2):  # Streaming revisits common.pt for each batch of tensors.
        tensors = reader.load_tensors(checkpoint, keys=["model.weight"])
        assert set(tensors) == {"model.weight"}
        assert tensors["model.weight"].dtype == expected.dtype
        assert torch.equal(tensors["model.weight"], expected)
    assert set(torch.serialization.get_safe_globals()) == allowed_before


def test_tensor_loader_still_rejects_unrelated_globals(tmp_path, dist_env):
    checkpoint = tmp_path / "iter_0000100"
    megatron_mock.save_synthetic_checkpoint(
        {"model.weight": torch.ones(4)},
        Namespace(extra=_UnexpectedMetadata()), checkpoint,
    )
    allowed_before = set(torch.serialization.get_safe_globals())
    with pytest.raises(pickle.UnpicklingError, match="_UnexpectedMetadata"):
        reader.load_tensors(checkpoint, keys=["model.weight"])
    assert set(torch.serialization.get_safe_globals()) == allowed_before
