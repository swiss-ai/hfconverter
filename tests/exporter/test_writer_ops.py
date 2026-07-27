"""Operational properties of the export: dtypes, sharding, tokenizer, atomicity, memory bound.

These close gaps an adversarial review found in the original suite -- each is a property a REAL
conversion depends on but the tiny all-fp32 single-shard round trips never touched.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import megatron_mock  # noqa: E402
from exporter import mapping, output_claim, reader, writer  # noqa: E402
from exporter.export import export_checkpoint  # noqa: E402

FP32_BUFFER_SUFFIXES = ("e_score_correction_bias", "qb_beta")


def _save_ckpt(tmp_path, *, dtype=torch.float32, expert_bias=True, extra_tensors=None):
    model = megatron_mock.build_tiny_model(
        sandwich_norm=True, moe_latent_size=None, use_quantile_balancing=True, seed=3
    )
    if dtype != torch.float32:
        # parameters only: the fp32 router buffers must survive the cast, exactly as in a real
        # bf16 run (checkpoints carry bf16 params alongside fp32 router buffers)
        for module in model.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                setattr(module, name, torch.nn.Parameter(param.data.to(dtype)))
    tensors = megatron_mock.to_megatron_tensors(
        model, model.config, expert_bias_present=expert_bias
    )
    args = megatron_mock.make_args_namespace(
        model.config, expert_bias_present=expert_bias, bf16=(dtype == torch.bfloat16)
    )
    ckpt = tmp_path / "iter_0000100"
    megatron_mock.save_synthetic_checkpoint(
        tensors, args, ckpt, extra_tensors=extra_tensors, iteration=100
    )
    return model, ckpt


def _all_written(out_dir):
    written = {}
    for name in os.listdir(str(out_dir)):
        if name.endswith(".safetensors"):
            written.update(load_file(os.path.join(str(out_dir), name)))
    return written


class TestBf16Export:
    """Every real fork checkpoint is bf16 params + fp32 router buffers; nothing tested that."""

    def test_bf16_params_and_fp32_router_buffers_survive_the_export(
        self, dist_env, export_api, tmp_path
    ):
        model, ckpt = _save_ckpt(tmp_path, dtype=torch.bfloat16)
        out_dir = tmp_path / "hf"
        export_api(ckpt, out_dir)

        written = _all_written(out_dir)
        assert written["model.embed_tokens.weight"].dtype == torch.bfloat16
        for key, tensor in written.items():
            expected = torch.float32 if key.endswith(FP32_BUFFER_SUFFIXES) else torch.bfloat16
            assert tensor.dtype == expected, f"{key}: {tensor.dtype}"

        config = json.loads((out_dir / "config.json").read_text())
        assert config["dtype"] == "bfloat16"

        # D2: bit-identical, not merely close
        source = model.state_dict()
        assert torch.equal(
            written["model.layers.0.self_attn.q_proj.weight"],
            source["model.layers.0.self_attn.q_proj.weight"],
        )

    def test_mixed_parameter_dtypes_fail_before_shard_planning(self):
        config = megatron_mock.tiny_export_config(False, None, False)
        plan = mapping.build_plan(config.to_dict(), expert_bias_present=True)
        metadata = {
            row.megatron_key: SimpleNamespace(
                global_shape=row.shape,
                dtype=row.dtype or torch.bfloat16,
            )
            for row in plan.rows
        }
        metadata["output_layer.weight"].dtype = torch.float32

        with pytest.raises(ValueError, match="mixed parameter dtypes") as error:
            mapping.validate_metadata(plan, metadata)
        assert "output_layer.weight" in str(error.value)


class TestGroupLimitedExport:
    def test_group_geometry_reaches_saved_hf_config(self, dist_env, export_api, tmp_path):
        model = megatron_mock.build_tiny_model(
            sandwich_norm=False,
            moe_latent_size=None,
            use_quantile_balancing=False,
            seed=3,
            n_group=3,
            topk_group=1,
        )
        tensors = megatron_mock.to_megatron_tensors(
            model, model.config, expert_bias_present=True
        )
        args = megatron_mock.make_args_namespace(
            model.config, expert_bias_present=True
        )
        ckpt = tmp_path / "iter_0000100"
        megatron_mock.save_synthetic_checkpoint(tensors, args, ckpt, iteration=100)

        out_dir = tmp_path / "hf"
        export_api(ckpt, out_dir)

        saved_config = json.loads((out_dir / "config.json").read_text())
        assert saved_config["n_group"] == 3
        assert saved_config["topk_group"] == 1


class TestSharding:
    def test_multi_shard_writes_an_index_covering_every_tensor(
        self, dist_env, export_api, tmp_path
    ):
        _, ckpt = _save_ckpt(tmp_path)
        out_dir = tmp_path / "hf"
        export_api(ckpt, out_dir, "--max-shard-size", "20KB")

        names = sorted(os.listdir(str(out_dir)))
        shards = [n for n in names if n.startswith("model-") and n.endswith(".safetensors")]
        assert len(shards) > 1, f"expected a sharded set, got {names}"
        assert "model.safetensors" not in names

        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        written = _all_written(out_dir)
        assert set(index["weight_map"]) == set(written)
        assert index["metadata"]["total_size"] == sum(
            t.numel() * t.element_size() for t in written.values()
        )
        for key, filename in index["weight_map"].items():
            assert filename in shards, (key, filename)


class TestTokenizerAndConfigIds:
    def test_token_ids_land_in_config_json_not_only_generation_config(
        self, dist_env, export_api, tmp_path
    ):
        _, ckpt = _save_ckpt(tmp_path)
        tok_dir = tmp_path / "tok"
        tok_dir.mkdir()
        (tok_dir / "tokenizer_config.json").write_text(
            json.dumps({"bos_token_id": 11, "eos_token_id": 12, "pad_token_id": 13})
        )
        (tok_dir / "tokenizer.json").write_text(json.dumps({"added_tokens": []}))

        out_dir = tmp_path / "hf"
        export_api(ckpt, out_dir, "--tokenizer-dir", str(tok_dir))

        config = json.loads((out_dir / "config.json").read_text())
        generation = json.loads((out_dir / "generation_config.json").read_text())
        # the two files in one directory must not disagree
        for field, value in (("bos_token_id", 11), ("eos_token_id", 12), ("pad_token_id", 13)):
            assert config[field] == value, f"config.json {field}"
            assert generation[field] == value, f"generation_config.json {field}"
        assert (out_dir / "tokenizer_config.json").is_file()

    def test_bad_tokenizer_dir_fails_before_anything_is_written(self, dist_env, tmp_path):
        _, ckpt = _save_ckpt(tmp_path)
        out_dir = tmp_path / "hf"
        with pytest.raises(ValueError, match="tokenizer-dir"):
            export_checkpoint(ckpt, out_dir, tokenizer_dir=tmp_path / "nope")
        assert not out_dir.exists() or not any(out_dir.iterdir())

    def test_tokenizer_larger_than_the_vocab_is_refused(self, dist_env, tmp_path):
        # Both cluster entry points default TOKENIZER_DIR to the 1.5b-era 200064-entry
        # tokenizer. Pointing it at a smaller-vocab checkpoint used to exit 0 and only fail
        # later, at the embedding lookup, on ordinary text.
        _, ckpt = _save_ckpt(tmp_path)
        tok_dir = tmp_path / "tok-too-big"
        tok_dir.mkdir()
        (tok_dir / "tokenizer_config.json").write_text(json.dumps({"bos_token_id": 1}))
        (tok_dir / "tokenizer.json").write_text(
            json.dumps({"added_tokens": [], "model": {"vocab": {f"t{i}": i for i in range(999)}}})
        )
        out_dir = tmp_path / "hf"
        with pytest.raises(ValueError, match="vocab_size"):
            export_checkpoint(ckpt, out_dir, tokenizer_dir=tok_dir)

    def test_tokenizer_within_the_vocab_is_accepted(self, dist_env, export_api, tmp_path):
        _, ckpt = _save_ckpt(tmp_path)
        tok_dir = tmp_path / "tok-fits"
        tok_dir.mkdir()
        (tok_dir / "tokenizer_config.json").write_text(json.dumps({"bos_token_id": 1}))
        (tok_dir / "tokenizer.json").write_text(
            json.dumps({"added_tokens": [], "model": {"vocab": {f"t{i}": i for i in range(64)}}})
        )
        out_dir = tmp_path / "hf"
        export_api(ckpt, out_dir, "--tokenizer-dir", str(tok_dir))
        assert (out_dir / "tokenizer.json").is_file()


class TestAtomicity:
    def test_only_one_concurrent_export_can_claim_an_empty_output(self, tmp_path):
        out_dir = tmp_path / "hf"
        out_dir.mkdir()
        start = threading.Barrier(2)

        def compete():
            start.wait()
            try:
                return output_claim.claim_output_dir(out_dir)
            except ValueError as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [future.result() for future in (pool.submit(compete), pool.submit(compete))]

        claims = [result for result in results if isinstance(result, output_claim.OutputClaim)]
        failures = [result for result in results if isinstance(result, ValueError)]
        assert len(claims) == 1
        assert len(failures) == 1
        assert "already claimed" in str(failures[0])
        assert claims[0].marker.read_text() == claims[0].contents

        output_claim.release_output_claim(claims[0])
        assert not (out_dir / output_claim.INCOMPLETE_MARKER).exists()

    def test_completion_refuses_to_remove_a_marker_owned_by_someone_else(self, tmp_path):
        claim = output_claim.claim_output_dir(tmp_path / "hf")
        claim.marker.write_text("claim_token=someone-else\n")

        with pytest.raises(RuntimeError, match="ownership marker changed"):
            output_claim.release_output_claim(claim)
        assert claim.marker.is_file()

    def test_conversion_info_is_written_last_and_certifies_the_export(
        self, dist_env, export_api, tmp_path
    ):
        _, ckpt = _save_ckpt(tmp_path)
        out_dir = tmp_path / "hf"
        export_api(ckpt, out_dir)
        assert (out_dir / "conversion_info.json").is_file()
        assert not (out_dir / ".export_incomplete").exists()

        info = json.loads((out_dir / "conversion_info.json").read_text())
        assert info["settings"]["verify"] is True
        assert info["exporter_git_hash"]  # sha, sha-dirty, or "unknown" -- never empty

    def test_a_failure_mid_write_leaves_a_detectable_incomplete_marker(
        self, dist_env, tmp_path, monkeypatch
    ):
        _, ckpt = _save_ckpt(tmp_path)
        out_dir = tmp_path / "hf"

        def boom(*_args, **_kwargs):
            raise RuntimeError("disk exploded")

        monkeypatch.setattr(writer, "copy_module_files", boom)
        with pytest.raises(RuntimeError, match="disk exploded"):
            export_checkpoint(ckpt, out_dir)

        # The directory holds weights but is NOT certified: no conversion_info.json, and the marker
        # says so -- a half-written export cannot be mistaken for a good one.
        assert (out_dir / ".export_incomplete").is_file()
        assert "claim_token=" in (out_dir / ".export_incomplete").read_text()
        assert not (out_dir / "conversion_info.json").exists()


class TestFilteredLoad:
    def test_optimizer_tensors_are_never_materialized(self, dist_env, tmp_path):
        # The memory bound: mcore's load_plain_tensors reads EVERY tensor, optimizer state included
        # (fp32, several times the bf16 weights). The exporter must only ask for what it maps.
        extra = {
            "optimizer.state.exp_avg.decoder.layers.0.mlp.linear_fc1.weight": torch.zeros(
                128, 32, dtype=torch.float32
            ),
            "optimizer.state.exp_avg_sq.decoder.layers.0.mlp.linear_fc1.weight": torch.zeros(
                128, 32, dtype=torch.float32
            ),
        }
        _, ckpt = _save_ckpt(tmp_path, extra_tensors=extra)

        with reader.single_rank_pg():
            everything = reader.load_tensors(ckpt)
            model_only = reader.load_tensors(
                ckpt, keys=[k for k in everything if not k.startswith("optimizer.")]
            )
        assert any(k.startswith("optimizer.") for k in everything)
        assert not any(k.startswith("optimizer.") for k in model_only)
        assert len(model_only) == len(everything) - len(extra)

    def test_a_missing_expected_tensor_is_named(self, dist_env, tmp_path):
        _, ckpt = _save_ckpt(tmp_path)
        with reader.single_rank_pg():
            with pytest.raises(ValueError, match="does.not.exist"):
                reader.load_tensors(ckpt, keys=["decoder.layers.9.does.not.exist"])
