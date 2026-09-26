"""
Vendored core of the official Dataset Inference repository
(https://github.com/cleverhans-lab/dataset-inference, commit 24baf46).
See SOURCE.md for provenance, hashes and the exact list of modifications.

`attacks.py` is an upstream copy (2-line torch-2.x patch, see SOURCE.md) that does `import ipdb` at module top.
ipdb is a debugger that the vendored functions never call at runtime (the only
references are commented-out `ipdb.set_trace()` lines) and it is not installed
in the evaluation environment, so a placeholder module is registered here
BEFORE the submodules are imported, so no import line has to be touched.
"""
import sys
import types

if "ipdb" not in sys.modules:
    try:
        import ipdb  # noqa: F401  (use the real package when it is installed)
    except ImportError:
        _stub = types.ModuleType("ipdb")
        _stub.set_trace = lambda *a, **k: None
        sys.modules["ipdb"] = _stub
