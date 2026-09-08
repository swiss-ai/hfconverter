"""Reject an image-installed Megatron before reading or converting a checkpoint."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--expected-commit')
    args = parser.parse_args()
    checkout = args.checkout.resolve(strict=True)
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if args.expected_commit and commit != args.expected_commit:
        raise RuntimeError(f'Megatron commit mismatch: expected {args.expected_commit}, got {commit}')
    origins = {}
    # Resolve packages without importing Torch/TE or modifying sys.path. This
    # checks the environment that the following python/torchrun process inherits.
    for name in ['megatron.core', 'megatron.training', 'model_provider', 'gpt_builders']:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        if spec is None or spec.origin is None:
            raise RuntimeError(f'{name} cannot be resolved; export PYTHONPATH={checkout}')
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(checkout):
            raise RuntimeError(f'{name} resolves to {origin}; expected code under {checkout}. '
                               f'Export PYTHONPATH={checkout} before conversion.')
        origins[name] = str(origin)
    selected_version = runpy.run_path(str(checkout/'megatron/core/package_info.py'))['__version__']
    try:
        installed_version = importlib.metadata.version('megatron-core')
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    print('MEGATRON_SOURCE_VERIFIED ' + json.dumps({
        'python_executable': sys.executable, 'commit': commit, 'selected_mcore_version': selected_version,
        'installed_distribution_version': installed_version, 'module_origins': origins,
    }), flush=True)


if __name__ == '__main__':
    main()
