"""The streaming exporter plans the shard layout from
torch_dist METADATA and writes one shard at a time, loading each Megatron source only while some
shard still needs it — instead of materializing the whole model. Two properties matter and neither
is touched by the single-shard round trips elsewhere:

  1. plan_shards (metadata-only, via the hub factory + size/id stand-ins) makes byte-for-byte the
     SAME decision the real torch packer makes on materialized tensors. If it drifts, the streamed
     files stop matching the shipped exports and the cheap sha256 regression would false-alarm.
  2. A multi-shard export whose SOURCE tensors straddle shard boundaries reloads bitwise-identically
     to the source model — the residency bookkeeping (a source stays resident from its first shard
     to its last) must not drop or misorder anything.
"""

import json
import os

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402
from huggingface_hub import split_torch_state_dict_into_shards  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import megatron_mock  # noqa: E402
from exporter import config_from_args, mapping, reader, writer  # noqa: E402
from exporter.export import export_checkpoint  # noqa: E402
from modeling_apertus2 import Apertus2ForCausalLM  # noqa: E402

# small enough to force many shards on the tiny model: the expert fc1 stack (one source) fans out
# into 2*E per-expert tensors that land across several shards (straddling), and the 16 KB embedding
# exceeds it -> exercises the hub packer's "single tensor over the cap gets its own shard" branch.
TINY_SHARD = "5KB"


def _build_and_save(tmp_path, sandwich, latent, qb, expert_bias, seed=7):
    model = megatron_mock.build_tiny_model(
        sandwich, latent, qb, seed=seed, zero_expert_bias=not expert_bias
    )
    tensors = megatron_mock.to_megatron_tensors(
        model, model.config, expert_bias_present=expert_bias
    )
    args = megatron_mock.make_args_namespace(model.config, expert_bias_present=expert_bias)
    ckpt = tmp_path / "iter_0000100"
    megatron_mock.save_synthetic_checkpoint(tensors, args, ckpt, iteration=100)
    return model, ckpt


def _all_written(out_dir):
    written = {}
    for name in os.listdir(str(out_dir)):
        if name.endswith(".safetensors"):
            written.update(load_file(os.path.join(str(out_dir), name)))
    return written


class TestPlanShardsMatchesHub:
    """plan_shards() from metadata == split_torch_state_dict_into_shards() from real tensors."""

    def test_layout_is_byte_for_byte_the_hub_decision(self, dist_env, tmp_path):
        _, ckpt = _build_and_save(tmp_path, True, megatron_mock.TINY_LATENT, True, True)

        args, _ = reader.load_args(ckpt)
        derived = config_from_args.derive_config(args)
        metadata = reader.load_metadata(ckpt)
        model_keys, _ = mapping.partition_universe(set(metadata), False)
        plan = mapping.build_plan(derived.kwargs, derived.expert_bias_present)
        params_dtype = mapping.params_dtype(plan, lambda k: metadata[k].dtype)
        specs = mapping.plan_hf_tensors(plan, params_dtype)

        # the real owned tensors the OLD path would have packed (same emit order as `specs`)
        with reader.single_rank_pg():
            tensors = reader.load_tensors(ckpt, keys=model_keys)
        owned = {k: writer._owned_contiguous(v) for k, v in mapping.convert(plan, tensors).items()}
        assert list(owned) == [s.hf_key for s in specs], "emit order drifted from plan_hf_tensors"

        for size in (TINY_SHARD, "1KB", "5GB"):  # multi-shard, heavily-sharded, single-shard
            hub = split_torch_state_dict_into_shards(owned, max_shard_size=size)
            mine = writer.plan_shards(specs, size)
            assert mine.is_sharded == hub.is_sharded, size
            assert mine.filename_to_tensors == hub.filename_to_tensors, size
            assert mine.tensor_to_filename == hub.tensor_to_filename, size
            assert dict(mine.metadata) == dict(hub.metadata), size  # incl. total_size


class TestStreamingMultiShardBitwise:
    @pytest.mark.parametrize(
        "sandwich,latent,qb,expert_bias",
        [
            pytest.param(True, megatron_mock.TINY_LATENT, True, True, id="sandwich-latent-qb-bias"),
            pytest.param(False, None, False, False, id="no-bias-synth-zeros"),
        ],
    )
    def test_straddling_sources_reload_bitwise(
        self, dist_env, tmp_path, sandwich, latent, qb, expert_bias
    ):
        model, ckpt = _build_and_save(tmp_path, sandwich, latent, qb, expert_bias)
        out_dir = tmp_path / "hf"

        # verify_load exercises the disk-backed _DiskShardTensors reference reader across shards
        summary = export_checkpoint(ckpt, out_dir, max_shard_size=TINY_SHARD, verify_load=True)

        names = sorted(os.listdir(str(out_dir)))
        shards = [n for n in names if n.startswith("model-") and n.endswith(".safetensors")]
        assert len(shards) > 1, f"expected a sharded set to force straddling, got {names}"
        assert "model.safetensors" not in names
        assert (out_dir / "model.safetensors.index.json").is_file()

        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        written = _all_written(out_dir)
        assert set(index["weight_map"]) == set(written)
        assert summary.num_tensors == len(written)

        # the whole point: a straddled export loads back bitwise-identical to the source model
        reloaded, info = Apertus2ForCausalLM.from_pretrained(
            str(out_dir), dtype=torch.float32, output_loading_info=True
        )
        assert not info["missing_keys"], info["missing_keys"]
        assert not info["unexpected_keys"], info["unexpected_keys"]
        source_state = model.state_dict()
        restored_state = reloaded.state_dict()
        assert set(source_state) == set(restored_state)
        for key, source in source_state.items():
            restored = restored_state[key]
            assert restored.dtype == source.dtype, f"{key}: {restored.dtype} != {source.dtype}"
            assert torch.equal(restored, source), f"{key}: values differ across the streamed roundtrip"

    def test_shard_size_does_not_change_tensor_bytes(self, dist_env, tmp_path):
        """The same checkpoint exported single-shard vs heavily-sharded is bitwise-equal per key —
        proving the streaming split is a pure output-layout change, not a data change."""
        _, ckpt = _build_and_save(tmp_path, True, megatron_mock.TINY_LATENT, True, True)
        one = tmp_path / "one"
        many = tmp_path / "many"
        export_checkpoint(ckpt, one, max_shard_size="5GB")
        export_checkpoint(ckpt, many, max_shard_size=TINY_SHARD)

        a = _all_written(one)
        b = _all_written(many)
        assert a.keys() == b.keys()
        for key in a:
            assert a[key].dtype == b[key].dtype, key
            assert torch.equal(a[key], b[key]), f"{key}: bytes changed with shard size"
