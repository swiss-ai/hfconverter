"""Fixtures for the exporter test suite.

Tiny geometry and the mock live in ``megatron_mock.py`` so plain modules can share them
without pytest.

This conftest must import cleanly on the old py3.11 venv that has NO megatron-core: nothing
here imports megatron at module level. Every test module in this directory starts with
`pytest.importorskip("megatron.core")` instead.

PARENT-CONFTEST RE-EXPORT SHIELD (load-bearing, do not remove): pytest imports every
conftest.py under the bare module name 'conftest', and the last one imported owns
sys.modules['conftest']. The existing tests/test_modeling_apertus2.py does
`from conftest import EMBEDDING_MULTIPLIER, ...`; once THIS file exists it can be the module
that import resolves to (verified on pytest 9.1.1). We therefore load the parent
tests/conftest.py under an explicit private name and re-export all of its public names, so
the class tests keep working no matter which conftest owns the bare name. For the same
reason, test modules in THIS directory never `import conftest` — they import megatron_mock
(unique basename) and use fixtures.
"""

import importlib
import importlib.util
import inspect
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_HERE)
REPO_ROOT = os.path.dirname(_TESTS_DIR)
for _path in (REPO_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- parent conftest re-export shield (see module docstring) ---
_parent_spec = importlib.util.spec_from_file_location(
    "_apertus_parent_tests_conftest", os.path.join(_TESTS_DIR, "conftest.py")
)
_parent_conftest = importlib.util.module_from_spec(_parent_spec)
sys.modules["_apertus_parent_tests_conftest"] = _parent_conftest
_parent_spec.loader.exec_module(_parent_conftest)
globals().update(
    {
        _name: getattr(_parent_conftest, _name)
        for _name in dir(_parent_conftest)
        if not _name.startswith("_")
    }
)

import megatron_mock  # noqa: E402  (light import: no megatron at module level)


# ---------------------------------------------------------------------------
# CPU shim + world_size=1 gloo default process group (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dist_env():
    """Install the CPU shim and guarantee a ws=1 gloo default pg (tcp init, free port).

    Yields the idempotent `ensure_default_pg` so tests/mock code can re-establish the group
    if the exporter created-and-destroyed its own in between. Torn down at session end.
    """
    megatron_mock.install_cpu_shim()
    megatron_mock.ensure_default_pg()
    yield megatron_mock.ensure_default_pg
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# tiny-geometry factories (exporter-consistent multipliers; see megatron_mock docstring)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_export_config():
    """Factory for tiny Apertus2Config over the {sandwich} x {latent} x {QB} matrix."""

    def _make(
        sandwich_norm=False, moe_latent_size=None, use_quantile_balancing=False, **overrides
    ):
        return megatron_mock.tiny_export_config(
            sandwich_norm, moe_latent_size, use_quantile_balancing, **overrides
        )

    return _make


@pytest.fixture
def make_export_model():
    """Factory for a tiny, seeded, eval-mode Apertus2ForCausalLM with non-zero fp32
    router buffers (so copy-vs-synthesize bugs are observable)."""

    def _make(
        sandwich_norm=False,
        moe_latent_size=None,
        use_quantile_balancing=False,
        seed=0,
        **kwargs,
    ):
        return megatron_mock.build_tiny_model(
            sandwich_norm, moe_latent_size, use_quantile_balancing, seed=seed, **kwargs
        )

    return _make


# ---------------------------------------------------------------------------
# Exporter invocation helpers.
# ---------------------------------------------------------------------------


def _invoke_exporter_main(argv):
    """Call exporter.export's main() in-process, tolerating main(argv) or main()+sys.argv."""
    export_module = importlib.import_module("exporter.export")
    main = getattr(export_module, "main", None)
    assert callable(main), (
        "exporter/export.py must expose a main() entry point "
        "(python -m exporter.export)"
    )
    parameters = inspect.signature(main).parameters.values()
    accepts_argv = any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        for p in parameters
    )
    if accepts_argv:
        return main(list(argv))
    original_argv = sys.argv
    sys.argv = ["exporter.export", *argv]
    try:
        return main()
    finally:
        sys.argv = original_argv


def run_export(checkpoint_dir, output_dir, *extra_argv):
    """Run the exporter; returns normally only on success (raises SystemExit on rc != 0)."""
    argv = [
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-dir",
        str(output_dir),
        *extra_argv,
    ]
    rc = _invoke_exporter_main(argv)
    if rc not in (None, 0):
        raise SystemExit(rc)
    return rc


def exception_chain_text(exc):
    """Flatten an exception and its __cause__/__context__ chain into one searchable blob."""
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(f"{type(exc).__name__}: {exc}")
        if isinstance(exc, SystemExit):
            parts.append(f"exit_code={exc.code!r}")
        exc = exc.__cause__ or exc.__context__
    return "\n".join(parts)


@pytest.fixture
def export_api(dist_env):
    """Successful-export runner (in-process python API)."""
    return run_export


@pytest.fixture
def export_expect_failure(dist_env, capsys, caplog):
    """Failing-export runner: asserts SystemExit/ValueError and returns every place the
    error message could live (exception chain + stderr + stdout + logging) as one blob."""

    def _run(checkpoint_dir, output_dir, *extra_argv):
        with pytest.raises((SystemExit, ValueError)) as excinfo:
            run_export(checkpoint_dir, output_dir, *extra_argv)
        captured = capsys.readouterr()
        return "\n".join(
            [exception_chain_text(excinfo.value), captured.err, captured.out, caplog.text]
        )

    return _run
