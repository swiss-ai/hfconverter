"""The exporter must account for every checkpoint tensor exactly once.

Via megatron_mock injections on an otherwise-good tiny checkpoint:
  (a) an extra unknown tensor key   -> SystemExit/ValueError naming it verbatim + the
                                       "new fork feature?" hint
  (b) a deleted expected key        -> error naming it verbatim
  (c) optimizer.* dummy tensors     -> dropped by default (export succeeds; count recorded in
                                       conversion_info.json) but fatal under --strict-optimizer
  (d) mlp.experts.weight1 present   -> error pointing at OffloadingExpertsMLP

Error text is searched across the exception chain, stderr, stdout, and logging (the
export_expect_failure fixture) so the assertion holds wherever the exporter surfaces it.
"""

import json
import os

import pytest

pytest.importorskip("megatron.core")

import torch  # noqa: E402

import exporter.export  # noqa: E402,F401  (module-level on purpose: collection surfaces the dependency)
import megatron_mock  # noqa: E402


def _make_checkpoint(tmp_path, extra_tensors=None, drop_key=None, name="ckpt"):
    """Good (F,F, no QB, expert_bias on) tiny checkpoint with optional mutations."""
    model = megatron_mock.build_tiny_model(False, None, False, seed=0)
    tensors = megatron_mock.to_megatron_tensors(model, model.config, expert_bias_present=True)
    if drop_key is not None:
        del tensors[drop_key]
    args = megatron_mock.make_args_namespace(model.config, expert_bias_present=True)
    checkpoint_dir = tmp_path / name
    megatron_mock.save_synthetic_checkpoint(
        tensors, args, checkpoint_dir, extra_tensors=extra_tensors
    )
    return checkpoint_dir


UNKNOWN_KEY = "decoder.layers.1.mlp.mystery_gadget.weight"
MISSING_KEY = "decoder.layers.1.pre_mlp_layernorm.weight"
OFFLOADING_KEY = "decoder.layers.1.mlp.experts.weight1"
OPTIMIZER_KEYS = {
    "optimizer.state.exp_avg.decoder.layers.1.mlp.router.weight": torch.randn(8, 32),
    "optimizer.state.exp_avg_sq.decoder.layers.1.mlp.router.weight": torch.randn(8, 32),
}


class TestBijection:
    def test_unknown_key_fails_naming_it_verbatim(
        self, dist_env, export_expect_failure, tmp_path
    ):
        checkpoint_dir = _make_checkpoint(
            tmp_path, extra_tensors={UNKNOWN_KEY: torch.randn(4, 4)}
        )
        blob = export_expect_failure(checkpoint_dir, tmp_path / "out")
        assert UNKNOWN_KEY in blob, f"error must name the unknown key verbatim; got:\n{blob}"
        assert "new fork feature" in blob, (
            f"error must carry the 'new fork feature?' hint; got:\n{blob}"
        )

    def test_missing_expected_key_fails_naming_it(
        self, dist_env, export_expect_failure, tmp_path
    ):
        checkpoint_dir = _make_checkpoint(tmp_path, drop_key=MISSING_KEY)
        blob = export_expect_failure(checkpoint_dir, tmp_path / "out")
        assert MISSING_KEY in blob, f"error must name the missing key verbatim; got:\n{blob}"

    def test_optimizer_keys_dropped_by_default_and_recorded(
        self, dist_env, export_api, tmp_path
    ):
        checkpoint_dir = _make_checkpoint(tmp_path, extra_tensors=dict(OPTIMIZER_KEYS))
        output_dir = tmp_path / "out"
        export_api(checkpoint_dir, output_dir)  # must succeed

        assert os.path.isfile(output_dir / "config.json")
        info_path = output_dir / "conversion_info.json"
        assert os.path.isfile(info_path)
        with open(info_path) as f:
            info = json.load(f)

        def _records_drop_count(node):
            """Accept a count == 2 or a 2-element list under any key mentioning the drop."""
            if isinstance(node, dict):
                for key, value in node.items():
                    key_l = str(key).lower()
                    if "drop" in key_l or "optimizer" in key_l:
                        if value == 2:
                            return True
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            return True
                    if _records_drop_count(value):
                        return True
            elif isinstance(node, list):
                return any(_records_drop_count(item) for item in node)
            return False

        assert _records_drop_count(info), (
            "conversion_info.json must record the dropped optimizer-key count (2); got: "
            f"{json.dumps(info)[:2000]}"
        )

    def test_optimizer_keys_fatal_with_strict_flag(
        self, dist_env, export_expect_failure, tmp_path
    ):
        checkpoint_dir = _make_checkpoint(tmp_path, extra_tensors=dict(OPTIMIZER_KEYS))
        blob = export_expect_failure(
            checkpoint_dir, tmp_path / "out_strict", "--strict-optimizer"
        )
        assert "optimizer" in blob.lower(), (
            f"--strict-optimizer failure should mention the optimizer keys; got:\n{blob}"
        )

    def test_offloading_experts_key_fails_pointing_at_the_feature(
        self, dist_env, export_expect_failure, tmp_path
    ):
        checkpoint_dir = _make_checkpoint(
            tmp_path, extra_tensors={OFFLOADING_KEY: torch.randn(8, 16, 32)}
        )
        blob = export_expect_failure(checkpoint_dir, tmp_path / "out")
        assert "ffloading" in blob, (  # matches Offloading/offloading
            f"error must point at OffloadingExpertsMLP; got:\n{blob}"
        )
