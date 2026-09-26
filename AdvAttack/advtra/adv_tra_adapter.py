# -*- coding: utf-8 -*-
"""
ADV-TRA Adapter.

Bridges your framework (NormalizedModel + raw_train_set + group indices) to
the authors' released ADV-TRA code (expects an argparse namespace, reads data
and models from disk via fixed paths, returns results via print statements).

High-level contract:

  - You pass in a `wrapped_source_model` -- this should be a model that takes
    raw [0, 1] images and returns logits. A NormalizedModel-wrapped classifier
    is the intended use case; a bare [0,1]-trained classifier would also work.

  - You pass in a raw_dataset (e.g., dataset_obj.raw_train_set) plus a list
    of base_indices. The adapter loads those samples, moves them to the
    device, and writes a data_log.pth file at the path the authors' code
    expects.

  - For extraction, the adapter monkey-patches the vendored module's
    `build_model` symbol so it returns your wrapped model (already loaded
    with the correct weights, no state_dict load needed). We also save a
    dummy state_dict at the path the authors' code will try to load, because
    their code unconditionally calls `torch.load(...)`.

  - For verification, the same trick is applied for the suspect model.

  - After extraction, trajectories are saved to disk at
    `{fingerprint_path}/{dataset}/trajectory_{length}/{1..N}/`.

  - Verification returns (detection_rate, mutation_rates) as a tuple.

Usage (see usage_template.py for a complete example):

    args = build_args(
        dataset_name="cifar10",
        num_classes=10,
        data_path="./results/advtra",
        model_path="./results/advtra/models",
        fingerprint_path="./results/advtra/fingerprints",
        num_trajectories=100,
        length=4,
        tra_classes=10,
        max_iteration=300,
        initial_stepsize=0.05,
        tra_lr=0.05,
        factor_lc=0.9,
        factor_re=0.95,
        threshold=0.5,
        device="cuda:0",
    )

    run_extraction(args, wrapped_source, raw_train_set, base_indices=group_A[:200])
    det_rate, mut_rates = run_verification(args, wrapped_suspect)
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Import the vendored module directly so we can rebind its symbols.
# Support both package-style (from .advtra_vendored import adv_gen) and
# flat-import (import adv_gen after adding the vendored dir to sys.path).
try:
    from .advtra_vendored import adv_gen as _adv_gen
except ImportError:
    try:
        from advtra_vendored import adv_gen as _adv_gen
    except ImportError:
        import os as _os
        import sys as _sys
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "advtra_vendored"))
        #import adv_gen as _adv_gen


# =====================================================================
# 1. Build the args namespace the authors' code expects
# =====================================================================

def build_args(
    *,
    dataset_name: str,
    num_classes: int,
    data_path: str,
    model_path: str,
    fingerprint_path: str,
    num_trajectories: int = 100,
    length: int = 4,
    tra_classes: int = 10,
    max_iteration: int = 300,
    initial_stepsize: float = 0.05,
    tra_lr: float = 0.05,
    factor_lc: float = 0.9,
    factor_re: float = 0.95,
    threshold: float = 0.5,
    device: str = "cuda:0",
    suspect_path: str = None,
    extraction_log_path: str = None,
) -> SimpleNamespace:
    """Construct the argparse-style namespace the vendored code reads from.

    All trajectory-related defaults match the paper (for CIFAR) / the
    reference code (where the two disagree, we follow the code):
      - length=4 (2l=4, so l=2)
      - tra_classes=10 (m=9 chained bilateral trajectories)
      - factor_lc=0.9 (length-control factor, alpha_lc in the paper)
      - factor_re=0.95 (brake factor; note the paper says 0.9 but the
        released code uses 0.95)
      - threshold=0.5 (Thr_mut in the paper)

    Paths are used as follows by the vendored code:
      - data_path/{dataset}/allocated_data/data_log.pth
          -- the X_train / y_train tensor dict (we write this in write_data_log)
      - model_path/{dataset}/source_model.pth
          -- where the code expects the source checkpoint (we write a dummy
             one; the monkey-patched build_model supplies the real model)
      - fingerprint_path/{dataset}/trajectory_{length}/{1..N}/tra_log.pth
          -- where extracted trajectories are saved
      - suspect_path
          -- where the code expects the suspect checkpoint (we write a dummy
             one; the monkey-patched build_model supplies the real suspect)
      - extraction_log_path (optional)
          -- if set, each base-sample attempt will be logged as one line
             showing the class traversal path and where (if anywhere) it
             failed. Useful for debugging low success rates.
    """
    return SimpleNamespace(
        dataset=dataset_name,
        num_classes=num_classes,
        data_path=data_path,
        model_path=model_path,
        fingerprint_path=fingerprint_path,
        num_trajectories=num_trajectories,
        length=length,
        tra_classes=tra_classes,
        max_iteration=max_iteration,
        initial_stepsize=initial_stepsize,
        tra_lr=tra_lr,
        factor_lc=factor_lc,
        factor_re=factor_re,
        threshold=threshold,
        device=device,
        suspect_path=suspect_path,
        extraction_log_path=extraction_log_path,
    )


# =====================================================================
# 2. Data plumbing: write a data_log.pth from your raw dataset
# =====================================================================

def write_data_log(
    args: SimpleNamespace,
    raw_dataset: Dataset,
    indices: Sequence[int],
) -> str:
    """Build the data_log.pth the authors' generate_trajectory expects.

    Only X_train and y_train are read by their code (on the CIFAR path).
    We still include empty placeholders for the other four keys so the
    file format matches what their allocate_data would have produced.

    Args:
        args: args namespace built by build_args (for args.data_path / dataset).
        raw_dataset: a dataset returning (x in [0,1], y) tuples. Typically
            your CIFAR10Dataset.raw_train_set.
        indices: sample indices to include. The vendored code uses the FIRST
            2 * num_trajectories of these, so pass at least that many. We
            recommend 2 * num_trajectories to leave headroom for samples
            that fail Eq. 3 during boundary probing.

    Returns:
        The path where data_log.pth was written.
    """
    if len(indices) < 2 * args.num_trajectories:
        print(
            f"[warn] only {len(indices)} base indices provided but the "
            f"vendored code will read 2 * num_trajectories = "
            f"{2 * args.num_trajectories} of them. Extraction may stop "
            f"early if too many samples fail boundary probing."
        )

    xs, ys = [], []
    for i in indices:
        x, y = raw_dataset[int(i)]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        xs.append(x.unsqueeze(0))
        ys.append(int(y))
    X_train = torch.cat(xs, dim=0)
    y_train = torch.tensor(ys, dtype=torch.long)

    empty_x = torch.empty(0, *X_train.shape[1:])
    empty_y = torch.empty(0, dtype=torch.long)

    data_log = {
        "X_train": X_train,
        "y_train": y_train,
        "X_attack": empty_x,
        "y_attack": empty_y,
        "X_remain": empty_x,
        "y_remain": empty_y,
    }

    save_dir = os.path.join(args.data_path, args.dataset, "allocated_data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "data_log.pth")
    torch.save(data_log, save_path)
    return save_path


# =====================================================================
# 3. The model shim: makes a pre-loaded model compatible with the
#    vendored code's build_model + load_state_dict + .to(device) calls
# =====================================================================

class _ModelShim(nn.Module):
    """Forwards to a pre-loaded model while absorbing the calls that the
    vendored code makes during its own setup path.

    The vendored code does, roughly:
        m = build_model(args)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        ... m(x), m.zero_grad(), etc.

    We already have the real (wrapped) model loaded and on the right device,
    so load_state_dict is a no-op and .to(device) / .eval() just delegate.
    """

    def __init__(self, actual_model: nn.Module):
        super().__init__()
        # Stored as a direct attribute (not a submodule) so that
        # state_dict()/load_state_dict() don't try to manage its parameters.
        # Using object.__setattr__ bypasses nn.Module's submodule registration.
        object.__setattr__(self, "_actual_model", actual_model)

    def load_state_dict(self, *args, **kwargs):
        # No-op: the real weights were loaded by the caller before wrapping.
        return

    def state_dict(self, *args, **kwargs):
        # Return the real model's state dict if anyone asks.
        return self._actual_model.state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self._actual_model.to(*args, **kwargs)
        return self

    def eval(self):
        self._actual_model.eval()
        return self

    def train(self, mode: bool = True):
        self._actual_model.train(mode)
        return self

    def zero_grad(self, *args, **kwargs):
        self._actual_model.zero_grad(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._actual_model.parameters(*args, **kwargs)

    def modules(self):
        return self._actual_model.modules()

    def forward(self, x):
        return self._actual_model(x)

    def __call__(self, x):
        return self._actual_model(x)


@contextlib.contextmanager
def _inject_model(wrapped_model: nn.Module):
    """Context manager: temporarily replace the vendored code's `build_model`
    symbol with a lambda returning the given wrapped model (in a shim).

    This is what bridges your pre-loaded NormalizedModel to the authors'
    code, which otherwise would try to construct a model from scratch and
    load a checkpoint from disk.
    """
    shim = _ModelShim(wrapped_model)
    original = _adv_gen.build_model
    _adv_gen.build_model = lambda a: shim
    try:
        yield shim
    finally:
        _adv_gen.build_model = original


def _ensure_dummy_checkpoint(path: str) -> None:
    """Write an empty state_dict at `path` so the vendored code's
    torch.load(...) doesn't crash. The shim ignores what gets loaded."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({}, path)


def _count_saved_trajectories(args: SimpleNamespace) -> int:
    """Count the number of surface trajectories actually saved on disk.

    The vendored generate_trajectory writes trajectories to:
        {fingerprint_path}/{dataset}/trajectory_{length}/{1..N}/

    where N can be less than args.num_trajectories if some base samples
    failed boundary probing (the outer loop silently skips those with
    a "This basic sample cannot generate a stable adversarial trajectory"
    message).

    The saved indices are dense starting from 1, so we count the max
    consecutive index with a tra_log.pth file.
    """
    fp_dir = os.path.join(
        args.fingerprint_path, args.dataset, f"trajectory_{args.length}"
    )
    if not os.path.isdir(fp_dir):
        return 0

    count = 0
    for idx in range(1, args.num_trajectories + 1):
        traj_file = os.path.join(fp_dir, str(idx), "tra_log.pth")
        if os.path.isfile(traj_file):
            count += 1
        else:
            # Indices are dense starting from 1 -- stop at first gap
            break
    return count


# =====================================================================
# 4. Top-level API: run_extraction and run_verification
# =====================================================================

def run_extraction(
    args: SimpleNamespace,
    wrapped_source_model: nn.Module,
    raw_dataset: Dataset,
    base_indices: Sequence[int],
) -> str:
    """Run the authors' generate_trajectory end-to-end using your model.

    Steps performed:
      1. Write data_log.pth from (raw_dataset, base_indices).
      2. Write a dummy source_model.pth so the vendored code's torch.load works.
      3. Monkey-patch build_model so it returns wrapped_source_model.
      4. Call generate_trajectory(args) -- this is the authors' original code.
      5. Restore the original build_model symbol.

    Returns:
        The path to the fingerprint directory. Each trajectory is saved as
        {fingerprint_path}/{dataset}/trajectory_{length}/{1..N}/tra_log.pth
        and pred_log.pth -- the exact layout expected by run_verification.
    """
    wrapped_source_model.eval()

    write_data_log(args, raw_dataset, base_indices)

    dummy_source = os.path.join(args.model_path, args.dataset, "source_model.pth")
    _ensure_dummy_checkpoint(dummy_source)

    requested = args.num_trajectories
    with _inject_model(wrapped_source_model):
        _adv_gen.generate_trajectory(args)

    # Sync args.num_trajectories to how many were actually saved on disk.
    # The vendored code can save fewer than requested if base samples failed
    # boundary probing (each failure is reported with
    # "This basic sample cannot generate a stable adversarial trajectory").
    # Without this sync, later verification would crash trying to read
    # trajectory files that don't exist.
    actual = _count_saved_trajectories(args)
    if actual < requested:
        print(
            f"[advtra] WARNING: requested {requested} trajectories but only "
            f"{actual} were saved. Base-sample budget was insufficient to "
            f"achieve the target count. Updating args.num_trajectories = {actual}."
        )
        print(
            f"[advtra] To get more trajectories, increase base_indices (pass "
            f"more base samples) or relax step-size hyperparameters."
        )
    args.num_trajectories = actual

    return os.path.join(
        args.fingerprint_path, args.dataset, f"trajectory_{args.length}"
    )


def run_verification(
    args: SimpleNamespace,
    wrapped_suspect_model: nn.Module,
    suspect_path: str = None,
) -> Tuple[float, List[float]]:
    """Run the authors' verify_trajectory against the saved fingerprint.

    The vendored code reads the fingerprint from:
        {fingerprint_path}/{dataset}/trajectory_{length}/{1..num_trajectories}

    So you must have already called run_extraction with the same args.

    Args:
        args: the args namespace used for extraction (so path + length match).
        wrapped_suspect_model: your suspect model, wrapped with NormalizedModel
            if your framework normalizes inside the model.
        suspect_path: optional override for args.suspect_path. If provided,
            args.suspect_path is updated in-place. A dummy state_dict is
            written here to satisfy the vendored code's torch.load call.

    Returns:
        (detection_rate, mutation_rates) where:
          - detection_rate: fraction of trajectories with r_mut < threshold
          - mutation_rates: list of per-trajectory r_mut values (length
            num_trajectories)
    """
    wrapped_suspect_model.eval()

    if suspect_path is not None:
        args.suspect_path = suspect_path
    if args.suspect_path is None:
        # Default location if none was specified at build_args time
        args.suspect_path = os.path.join(
            args.model_path, args.dataset, "_suspect_dummy.pth"
        )

    _ensure_dummy_checkpoint(args.suspect_path)

    # Sync args.num_trajectories to disk reality in case fewer were saved
    # than originally requested (e.g. when loading a pre-existing fingerprint).
    # The vendored verify_trajectory iterates range(1, num_trajectories+1) and
    # crashes if any file is missing.
    actual = _count_saved_trajectories(args)
    if actual == 0:
        raise RuntimeError(
            f"No trajectories found at {args.fingerprint_path}/{args.dataset}"
            f"/trajectory_{args.length}/. Run run_extraction first."
        )
    if actual != args.num_trajectories:
        args.num_trajectories = actual

    with _inject_model(wrapped_suspect_model):
        detection_rate, mutation_rates = _adv_gen.verify_trajectory(args)

    return detection_rate, mutation_rates


# =====================================================================
# 5. Convenience: a pretty result object (optional)
# =====================================================================

class AdvTraResult:
    """Thin value object wrapping the tuple returned by run_verification."""

    def __init__(self, detection_rate: float, mutation_rates: List[float], threshold: float):
        self.detection_rate = detection_rate
        self.mutation_rates = mutation_rates
        self.threshold = threshold
        self.mean_mutation_rate = float(np.mean(mutation_rates)) if mutation_rates else 0.0
        self.num_trajectories = len(mutation_rates)

    def __str__(self) -> str:
        return (
            f"AdvTraResult(\n"
            f"  detection_rate      = {self.detection_rate:.4f}\n"
            f"  mean_mutation_rate  = {self.mean_mutation_rate:.4f}\n"
            f"  num_trajectories    = {self.num_trajectories}\n"
            f"  threshold           = {self.threshold}\n"
            f")"
        )


def run_verification_pretty(
    args: SimpleNamespace,
    wrapped_suspect_model: nn.Module,
    suspect_path: str = None,
) -> AdvTraResult:
    """Like run_verification but returns an AdvTraResult object."""
    det_rate, mut_rates = run_verification(args, wrapped_suspect_model, suspect_path)
    return AdvTraResult(det_rate, mut_rates, args.threshold)