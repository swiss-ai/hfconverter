"""The converter must reject fallback to a different installed Megatron."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECK = Path(__file__).resolve().parents[1] / 'check_megatron_source.py'


class SourceGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ['selected', 'image']:
            repo = self.root / name
            for package in ['megatron', 'megatron/core', 'megatron/training']:
                directory = repo / package
                directory.mkdir(parents=True, exist_ok=True)
                (directory / '__init__.py').write_text('')
            for module in ['model_provider.py', 'gpt_builders.py']:
                (repo / module).write_text('')
            (repo / 'megatron/core/package_info.py').write_text('__version__ = "0.16.0rc0"\n')
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c',
                            'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)

    def run_guard(self, resolved, *args):
        env = {**os.environ, 'PYTHONPATH': str(self.root / resolved)}
        return subprocess.run([sys.executable, "-S", str(CHECK), str(self.root / 'selected'), *args],
                              cwd=self.root, env=env, text=True, capture_output=True)

    def test_accepts_selected_checkout(self):
        result = self.run_guard('selected')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_image_megatron_when_pythonpath_is_wrong(self):
        result = self.run_guard('image')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(self.root / 'image/megatron/core'), result.stderr)

    def test_rejects_wrong_checkpoint_source_commit(self):
        result = self.run_guard('selected', '--expected-commit', '0' * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('commit mismatch', result.stderr)

    def test_missing_megatron_explains_required_pythonpath(self):
        result = self.run_guard('absent')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('export PYTHONPATH=', result.stderr)


if __name__ == '__main__':
    unittest.main()
