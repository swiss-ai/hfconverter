"""Make the standalone conversion utilities importable by their focused tests."""

import os
import sys

_TESTS_CLUSTER = os.path.dirname(os.path.abspath(__file__))
CLUSTER_DIR = os.path.join(os.path.dirname(os.path.dirname(_TESTS_CLUSTER)), "cluster")

for _p in (CLUSTER_DIR, _TESTS_CLUSTER):
    if _p not in sys.path:
        sys.path.insert(0, _p)
