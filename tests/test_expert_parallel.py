"""End-to-end regression coverage for native Transformers expert parallelism.

EP is the only supported multi-rank mode (TP is intentionally rejected): ``ep_router``
remaps the gate's ``(logits, weights, indices)`` triple to rank-local expert ids with a
sentinel, ``grouped_gemm`` shards both stacked expert banks on the expert axis, and
``moe_tp_experts`` all-reduces the routed output.  The single-process tests below pin the
gate/experts interface that machinery depends on; the two-rank test runs the real
``from_pretrained(distributed_config=...)`` path on CPU/Gloo.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers.utils import is_torch_greater_or_equal

from configuration_apertus2 import Apertus2Config
from modeling_apertus2 import Apertus2ForCausalLM
from conftest import TINY_HIDDEN, TINY_N_EXPERTS, TINY_TOPK


WORKER = Path(__file__).with_name("_expert_parallel_worker.py")
TORCHRUN_AVAILABLE = (
    torch.distributed.is_available()
    and torch.distributed.is_gloo_available()
    and is_torch_greater_or_equal("2.5")
    and importlib.util.find_spec("torch.distributed.run") is not None
    and WORKER.is_file()
)


def test_ep_plan_is_published_and_tp_plan_is_not(make_model):
    assert Apertus2Config.base_model_tp_plan is None
    assert Apertus2Config.base_model_ep_plan == {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    model = make_model()
    assert model.model._ep_plan == Apertus2Config.base_model_ep_plan


def test_gate_returns_expert_parallel_triple(make_model):
    """RouterParallel unpacks exactly (logits, weights, indices) and reads num_experts."""
    model = make_model(use_quantile_balancing=False)
    gate = model.model.layers[1].mlp.gate
    assert gate.num_experts == TINY_N_EXPERTS

    x = torch.randn(6, TINY_HIDDEN, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        logits, weights, indices = gate(x)
        replay_indices, replay_weights = gate.route_tokens_to_experts(logits)

    assert logits.shape == (6, TINY_N_EXPERTS)
    assert logits.dtype == torch.float32
    assert weights.shape == indices.shape == (6, TINY_TOPK)
    # Forward must dispatch exactly what route_tokens_to_experts decides.
    assert torch.equal(indices, replay_indices)
    assert torch.equal(weights, replay_weights)


def test_naive_experts_skip_ep_sentinel(make_model):
    """Index ``num_experts`` marks a non-local expert and must contribute nothing."""
    model = make_model()
    experts = model.model.layers[1].mlp.experts
    tokens = torch.randn(5, TINY_HIDDEN, generator=torch.Generator().manual_seed(5))

    all_sentinel = torch.full((5, TINY_TOPK), experts.num_experts, dtype=torch.long)
    with torch.no_grad():
        out = experts(tokens, all_sentinel, torch.zeros(5, TINY_TOPK))
    assert torch.equal(out, torch.zeros_like(tokens))

    # A sentinel slot with zero weight must equal a duplicated real slot with zero weight.
    mixed = torch.tensor([[3, experts.num_experts]] * 5, dtype=torch.long)
    duplicated = torch.tensor([[3, 3]] * 5, dtype=torch.long)
    slot_weights = torch.tensor([[0.7, 0.0]] * 5)
    with torch.no_grad():
        out_mixed = experts(tokens, mixed, slot_weights)
        out_duplicated = experts(tokens, duplicated, slot_weights)
    assert torch.equal(out_mixed, out_duplicated)


@pytest.mark.skipif(not TORCHRUN_AVAILABLE, reason="torch.distributed.run unavailable")
def test_two_rank_ep_matches_unsharded_forward_and_generation(make_model, tmp_path):
    model_dir = tmp_path / "model"
    model = make_model(
        sandwich_norm=True,
        moe_latent_size=24,
        use_quantile_balancing=True,
        hidden_size=48,
        intermediate_size=96,
    )
    # Keep routing invariant to tiny distributed roundoff while still exercising nonzero
    # expert weights.  QB selects the LARGEST -qb_beta, so beta offsets pin the selection
    # to expert 0 (rank 0's shard) and expert E/2 (rank 1's shard): both EP ranks compute
    # a nonzero partial output and the all-reduce is exercised for real.
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            if not model.config.is_moe_layer(layer_idx):
                continue
            layer.mlp.gate.weight.zero_()
            beta = torch.zeros(model.config.n_routed_experts, dtype=torch.float32)
            beta[0] = -2.0
            beta[model.config.n_routed_experts // 2] = -1.0
            layer.mlp.gate.qb_beta.copy_(beta)
    model.save_pretrained(model_dir)

    reference = Apertus2ForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
    ).eval()
    input_ids = torch.tensor([[5, 7, 11, 13]], dtype=torch.long)
    with torch.inference_mode():
        logits = reference(input_ids=input_ids, use_cache=False).logits
        cached = reference(input_ids=input_ids, use_cache=True)
        decode_input_ids = cached.logits[:, -1:].argmax(dim=-1)
        decode_logits = reference(
            input_ids=decode_input_ids,
            past_key_values=cached.past_key_values,
            use_cache=True,
        ).logits
        generated = reference.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )

    payload_path = tmp_path / "reference.pt"
    torch.save(
        {
            "input_ids": input_ids,
            "logits": logits,
            "decode_input_ids": decode_input_ids,
            "decode_logits": decode_logits,
            "generated": generated,
        },
        payload_path,
    )

    env = os.environ.copy()
    for stale in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "PYTHONPATH"):
        env.pop(stale, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(tmp_path / "hf-home"),
            "HF_HUB_OFFLINE": "1",
            "HF_MODULES_CACHE": str(tmp_path / "hf-modules"),
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(WORKER),
            str(model_dir),
            str(payload_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
    )
    assert process.returncode == 0, (
        f"expert-parallel worker failed rc={process.returncode}\n"
        f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    )
    assert process.stdout.count("EP_OK rank=") == 2, process.stdout
