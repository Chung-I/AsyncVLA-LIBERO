"""Repo-wide pytest conftest.

`vla-scripts/` (hyphenated, matching the existing `vla-scripts/train_asyncvla.py`
convention) is not a valid Python package name, so it isn't picked up by the editable
install's package discovery. Tests that need `import vla_scripts.<module>` (e.g.
`tests/test_train_overfit.py` importing `train_asyncvla_libero`) register it here as a
namespace package pointing at the `vla-scripts/` directory, so no source file needs to
move or be duplicated.
"""

import sys
import types
from pathlib import Path

_VLA_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "vla-scripts"

if "vla_scripts" not in sys.modules:
    _pkg = types.ModuleType("vla_scripts")
    _pkg.__path__ = [str(_VLA_SCRIPTS_DIR)]
    sys.modules["vla_scripts"] = _pkg
