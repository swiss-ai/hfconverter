#!/usr/bin/env python
"""Two-rank CPU worker used by ``test_expert_parallel.py``.

The parent test saves a self-describing Apertus2 checkpoint and reference outputs.
Every rank loads that checkpoint through ``AutoModelForCausalLM`` with expert
parallelism enabled, then checks both sharding and numerical equivalence: the stacked
expert banks are split on the expert axis while attention, the router, the shared
expert, and the latent projections stay replicated.  ``CUDA_VISIBLE_DEVICES`` is empty
in the parent so this remains a portable Gloo test and does not consume CI GPUs.
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig


def main() -> None:
    model_dir, payload_path = sys.argv[1:]
    torch.set_num_threads(1)

    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            dtype=torch.float32,
            distributed_config=DistributedConfig(enable_expert_parallel=True),
        ).eval()

        assert dist.is_initialized()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 2
        assert model.tp_size == world_size
        assert model.config.distributed_config.enable_expert_parallel

        config = model.config
        local_experts = config.n_routed_experts // world_size

        # Attention is untouched by EP: projections keep their full shapes.
        attention = model.model.layers[0].self_attn
        assert attention.q_proj.weight.shape == (
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        )
        assert attention.k_proj.weight.shape == (
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
        )

        moe = model.model.layers[config.first_k_dense_replace].mlp
        # The router is replicated and keeps GLOBAL expert bookkeeping; RouterParallel
        # derives the local count from num_experts / ep_size at forward time.
        assert moe.gate.weight.shape == (config.n_routed_experts, config.hidden_size)
        assert moe.gate.num_experts == config.n_routed_experts
        # The expert banks are sharded on the expert axis, and the plan rewrites the
        # experts module's num_experts to the LOCAL count (the sentinel value).
        assert moe.experts.gate_up_proj.shape == (
            local_experts,
            2 * config.moe_intermediate_size,
            config.moe_latent_size,
        )
        assert moe.experts.down_proj.shape == (
            local_experts,
            config.moe_latent_size,
            config.moe_intermediate_size,
        )
        assert moe.experts.num_experts == local_experts
        # Shared expert and latent projections are replicated.
        assert moe.shared_experts.gate_proj.weight.shape[1] == config.hidden_size
        assert moe.latent_down_proj.weight.shape == (config.moe_latent_size, config.hidden_size)
        assert model.lm_head.weight.shape == (config.vocab_size, config.hidden_size)

        device = next(model.parameters()).device
        input_ids = payload["input_ids"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits.cpu()
            cached = model(input_ids=input_ids, use_cache=True)
            assert cached.past_key_values.layers[0].keys.shape == (
                input_ids.shape[0],
                config.num_key_value_heads,
                input_ids.shape[1],
                config.head_dim,
            )
            cached_logits = cached.logits.cpu()
            decoded = model(
                input_ids=payload["decode_input_ids"].to(device),
                past_key_values=cached.past_key_values,
                use_cache=True,
            )
            assert decoded.past_key_values.layers[0].keys.shape[-2] == input_ids.shape[1] + 1
            decode_logits = decoded.logits.cpu()
            generated = model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
                use_cache=True,
            ).cpu()

        torch.testing.assert_close(logits, payload["logits"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cached_logits, payload["logits"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(decode_logits, payload["decode_logits"], rtol=1e-5, atol=1e-6)
        assert torch.equal(generated, payload["generated"])

        max_difference = (logits - payload["logits"]).abs().max().item()
        print(f"EP_OK rank={rank} max_abs_difference={max_difference:.9g}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
