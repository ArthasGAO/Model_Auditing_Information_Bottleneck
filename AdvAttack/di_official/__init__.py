"""Dataset Inference baseline evaluated through the official code path.

Layout:
    di_vendored/            official code (attacks.py upstream copy + 2-line torch-2.x patch; see SOURCE.md)
    di_official_adapter.py  bridges framework models / datasets to that code
Entry point: main_DI_official_eval.py (repo root).
"""
