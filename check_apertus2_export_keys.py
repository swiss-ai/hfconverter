"""Run the generic converter with Apertus2-specific HF export-key validation.

This wrapper only supplies the expected key set. Conversion is implemented in
convert_checkpoint.py; no weights are loaded or transformed by this module.
"""

from __future__ import annotations

from typing import Any

from convert_checkpoint import main as _main


def _expected_hf_keys(bridge: Any) -> set[str]:
    """Resolve registry HF names, expanding Apertus2's routed-expert wildcard."""
    config = bridge.hf_pretrained
    experts = getattr(config, "n_routed_experts", None)
    if not isinstance(experts, int) or experts < 1:
        raise ValueError(
            f"Expected positive Apertus2 n_routed_experts, got {experts!r}"
        )
    keys: set[str] = set()
    for mapping in bridge._model_bridge.mapping_registry().mappings:
        value = mapping.hf_param
        names = value.values() if isinstance(value, dict) else (value,)
        for name in names:
            if "*" not in name:
                keys.add(name)
            elif name.count("*") == 1 and ".experts.*." in name:
                keys.update(
                    name.replace(".experts.*.", f".experts.{index}.", 1)
                    for index in range(experts)
                )
            else:
                raise ValueError(f"Unexpected Apertus2 HF mapping wildcard: {name}")
    if not keys:
        raise RuntimeError("Apertus2 registry returned no expected HF keys")
    return keys


def main(argv: list[str] | None = None) -> None:
    _main(argv, expected_keys_fn=_expected_hf_keys)


if __name__ == "__main__":
    main()
