# -*- coding: utf-8 -*-
"""
Dataset Inference (official code) adapter.

Bridges this framework (NormalizedModel-wrapped checkpoints, CIFAR dataset
objects with raw [0,1] splits, group_A index files) to the released Dataset
Inference code in `di_vendored/`:

  * feature extraction  = official `get_random_label_only` -> `rand_steps`
                          (Blind Walk, `--feature_type rand`, black-box)
  * ownership decision  = official notebook protocol (CIFAR10_rand.ipynb +
                          notebooks/utils.generate_table)

Contract with the official code:
  - `model` takes raw [0,1] images and returns logits (NormalizedModel puts the
    normalization inside the model, which is upstream's `--normalize 1`).
  - `loader` yields (X, y) batches in [0,1] with NO augmentation and NO
    shuffling (upstream: `train_shuffle=False` -> transform_train = transform_test).
  - `args` is an argparse-like namespace; the `rand` path reads
    `batch_size`, `regressor_embed`, `dataset` and writes `distance`.
  - `num_images / batch_size` must be an integer: upstream stops the loader
    loop with `if i+1 >= num_images/batch_size: break`, so a batch size that
    does not divide num_images yields MORE than num_images rows and the
    notebook's `reshape(num_images, 30)` fails.

Feature files are written in the upstream directory layout
    {root}/{DATASET}/model_{name}_normalized/{train,test}_rand_vulnerability.pt
(train = private, test = public) so the official notebook can re-read them
after editing only its `names` list. The victim is always named "teacher".
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .di_vendored import attacks as _attacks            # noqa: F401  (upstream file + 2-line torch-2.x patch, see SOURCE.md)
from .di_vendored import generate_features as _gf
from .di_vendored import notebook_protocol as _nb

_OFFICIAL_DATASET_NAMES = {
    "CIFAR-10": "CIFAR10", "CIFAR10": "CIFAR10",
    "CIFAR-100": "CIFAR100", "CIFAR100": "CIFAR100",
    "SVHN": "SVHN",
}


def official_dataset_name(framework_name: str) -> str:
    """Map this framework's dataset name to upstream's `args.dataset` value.

    In the `rand` path `args.dataset` only matters for the SVHN branch of
    `rand_steps` (doubled noise, 100 steps); CIFAR-10 and CIFAR-100 behave
    identically.
    """
    try:
        return _OFFICIAL_DATASET_NAMES[framework_name]
    except KeyError:
        raise ValueError(f"No upstream dataset name for {framework_name!r}")


# =====================================================================
# 1. args namespace + loaders
# =====================================================================
def build_args(dataset_name: str, batch_size: int = 500, regressor_embed: int = 0) -> SimpleNamespace:
    """The subset of upstream `params.parse_args()` that the rand path reads.

    Upstream README: `python generate_features.py --batch_size 500 ...
    --feature_type rand`; `regressor_embed` defaults to 0.
    """
    return SimpleNamespace(
        dataset=dataset_name,
        batch_size=int(batch_size),
        regressor_embed=int(regressor_embed),
        feature_type="rand",
        distance=None,          # set per noise family inside get_random_label_only
    )


def make_loader(dataset, indices, batch_size: int) -> DataLoader:
    """Unshuffled, single-process loader over `dataset[indices]` (raw [0,1])."""
    subset = Subset(dataset, [int(i) for i in indices])
    return DataLoader(subset, batch_size=int(batch_size), shuffle=False, num_workers=0)


# =====================================================================
# 2. Feature extraction (official rand_steps)
# =====================================================================
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _Tee(io.TextIOBase):
    """Pass stdout through unchanged while keeping a copy for parsing."""

    def __init__(self, stream):
        self.stream = stream
        self.parts = []

    def write(self, s):
        self.stream.write(s)
        self.parts.append(s)
        return len(s)

    def flush(self):
        self.stream.flush()


_FAILED_RE = re.compile(r"Number of steps = (\d+) \| Failed to convert = (\d+)")
_FAMILIES = ("linf", "l2", "l1")   # upstream loop order inside get_random_label_only


def parse_walk_log(text: str, num_images: int, batch_size: int) -> dict:
    """Turn upstream's per-walk prints into exact unflipped counts.

    `rand_steps` prints one line per call; `get_random_label_only` calls it
    batch by batch, and inside each batch 10 times for linf, then 10 for l2,
    then 10 for l1. So the k-th line of a batch belongs to family k // 10.
    """
    lines = _FAILED_RE.findall(text)
    n_batches = num_images // batch_size
    expected = n_batches * 30
    out = {"walks_per_family": 10 * num_images, "lines_parsed": len(lines), "lines_expected": expected}
    if len(lines) != expected:
        out["warning"] = "unexpected number of rand_steps lines; counts not attributed"
        return out
    unflipped = {f: 0 for f in _FAMILIES}
    steps_seen = set()
    for i, (steps, failed) in enumerate(lines):
        unflipped[_FAMILIES[(i % 30) // 10]] += int(failed)
        steps_seen.add(int(steps))
    out["unflipped"] = unflipped
    out["unflipped_frac"] = {f: unflipped[f] / (10 * num_images) for f in _FAMILIES}
    out["steps_printed"] = sorted(steps_seen)
    return out


def extract_rand_features(model, loader, args, device, num_images: int = 1000):
    """Run upstream `get_random_label_only` on `model`.

    Returns (features, walk_stats):
      features   `[num_images, 10, 3]` CPU tensor (10 random draws x [linf, l2, l1])
      walk_stats exact unflipped-walk counts per family, parsed from the lines
                 upstream prints (the prints still reach the console).

    The call is wrapped in `torch.no_grad()`: upstream evaluates `model(X)` and
    `model(X+delta)` outside `rand_steps` with autograd enabled but never uses
    the graph; the walk itself is already under no_grad upstream. This only
    saves memory, it does not change any number.
    """
    if num_images % args.batch_size != 0:
        raise ValueError(
            f"batch_size={args.batch_size} must divide num_images={num_images} "
            "(upstream loop exit condition; see module docstring)."
        )
    _gf.device = torch.device(device)
    model.eval()
    tee = _Tee(sys.stdout)
    with torch.no_grad(), contextlib.redirect_stdout(tee):
        full_d = _gf.get_random_label_only(args, loader, model, num_images=num_images)
    walk_stats = parse_walk_log("".join(tee.parts), num_images, args.batch_size)
    return full_d.detach().cpu(), walk_stats


# =====================================================================
# 3. Upstream file layout
# =====================================================================
def feature_dir(root, dataset_name: str, name: str) -> Path:
    return Path(root) / dataset_name / f"model_{name}_normalized"


def features_exist(root, dataset_name: str, name: str) -> bool:
    d = feature_dir(root, dataset_name, name)
    return (d / "train_rand_vulnerability.pt").is_file() and (d / "test_rand_vulnerability.pt").is_file()


def save_features(root, dataset_name: str, name: str, train_d: torch.Tensor, test_d: torch.Tensor, meta: dict) -> Path:
    d = feature_dir(root, dataset_name, name)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(train_d, d / "train_rand_vulnerability.pt")   # private (upstream: train split)
    torch.save(test_d, d / "test_rand_vulnerability.pt")     # public  (upstream: test split)
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return d


def load_meta(root, dataset_name: str, name: str) -> dict:
    p = feature_dir(root, dataset_name, name) / "meta.json"
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_cached_names(root, dataset_name: str) -> list[str]:
    base = Path(root) / dataset_name
    if not base.is_dir():
        return []
    names = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("model_") and d.name.endswith("_normalized"):
            name = d.name[len("model_"):-len("_normalized")]
            if features_exist(root, dataset_name, name):
                names.append(name)
    return names


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def extract_and_save(
    *, root, dataset_name: str, name: str, model, private_loader, public_loader,
    args, device, num_images: int, seed: int, meta_extra: dict | None = None, force: bool = False,
) -> Path:
    """Extract (public then private, the upstream `feature_extractor` order,
    with ONE seed reset before the pair as upstream does) and write the files.
    Skips work when both files exist and `force` is False."""
    if features_exist(root, dataset_name, name) and not force:
        # The cache is only valid for the same extraction settings; refuse to
        # relabel seed-0 features as another seed (or another image count).
        cached = load_meta(root, dataset_name, name)
        for key, want in (("feature_seed", int(seed)), ("num_images", int(num_images)),
                          ("batch_size", int(args.batch_size))):
            if key in cached and cached[key] != want:
                raise ValueError(
                    f"cached features for {name!r} were extracted with {key}={cached[key]}, "
                    f"requested {want}. Use --force-features or a different --features-root."
                )
        # The walk budget / noise scales live inside the vendored rand_steps, so
        # an edit there (e.g. steps = 200) is only visible through the file hash.
        cur_sha = sha256_file(Path(__file__).parent / "di_vendored" / "attacks.py")
        if cached.get("vendored_attacks_sha256") and cached["vendored_attacks_sha256"] != cur_sha:
            raise ValueError(
                f"cached features for {name!r} were extracted with a different di_vendored/attacks.py "
                f"(sha256 {cached['vendored_attacks_sha256'][:12]}... vs current {cur_sha[:12]}...). "
                "Mixing walk settings across models is invalid: use a separate --features-root "
                "(teacher included) or --force-features."
            )
        print(f"[di_official] cached features for {name!r}, skipping extraction")
        return feature_dir(root, dataset_name, name)

    _seed_everything(seed)
    t0 = time.time()
    test_d, walks_public = extract_rand_features(model, public_loader, args, device, num_images=num_images)
    t1 = time.time()
    train_d, walks_private = extract_rand_features(model, private_loader, args, device, num_images=num_images)
    t2 = time.time()

    meta = {
        # exact "Failed to convert" totals per noise family (see parse_walk_log)
        "walks_private": walks_private,
        "walks_public": walks_public,
        "name": name,
        "dataset": dataset_name,
        "num_images": int(num_images),
        "batch_size": int(args.batch_size),
        "feature_seed": int(seed),
        "order": "public(test) then private(train), single seed reset before the pair",
        "seconds_public": round(t1 - t0, 2),
        "seconds_private": round(t2 - t1, 2),
        "train_shape": list(train_d.shape),
        "test_shape": list(test_d.shape),
        "torch": torch.__version__,
        "vendored_attacks_sha256": sha256_file(Path(__file__).parent / "di_vendored" / "attacks.py"),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if meta_extra:
        meta.update(meta_extra)
    d = save_features(root, dataset_name, name, train_d, test_d, meta)
    print(f"[di_official] features for {name!r} -> {d}  "
          f"(public {t1 - t0:.1f}s, private {t2 - t1:.1f}s)")
    return d


# =====================================================================
# 4. Notebook protocol
# =====================================================================
def run_notebook_protocol(
    *, root, dataset_name: str, names: list[str], split_index: int = 500, seed: int = 0,
    selected_m: int = 10, total_inner_rep: int = 100, num_images: int = 1000,
    regressor_epochs: int = 1000, per_name_reseed: bool = True,
) -> dict:
    """Run CIFAR10_rand.ipynb cells 3, 8-18 and 29 over `names`.

    Returns {name: {"p_value_welch", "mean_diff_welch", "p_value_m", "mean_diff_m"}}
      * *_welch : cell 16/18, ONE Welch t-test on rows [split_index:]
                  (public > private, `alternative="greater"`, equal_var=False)
      * *_m     : cell 29 / generate_table(selected_m), the paper's table
                  protocol: `total_inner_rep` random draws of `selected_m`
                  rows per side, harmonic mean of the p-values.

    `names` must contain "teacher" (the victim). Only the victim's features
    are used for standardization and for training the regressor, so the
    regressor is identical for every run over the same victim feature files
    (seeded, CPU, full-batch SGD).

    per_name_reseed: the notebook draws generate_table's random subsets from
    the global torch RNG in `names` order, so a model's m-protocol p-value
    would depend on which other models are in the run. With True (default),
    the RNG is reset to `seed` before each model's draws, making every row
    independent of the rest of the list. Set False to reproduce the notebook's
    sequential stream exactly.
    """
    if "teacher" not in names:
        raise ValueError('names must include "teacher" (the victim).')

    _nb.set_notebook_seed(seed)                                                          # cell 3
    trains, tests, mean_cifar, std_cifar = _nb.load_features(                            # cell 8
        str(Path(root) / dataset_name), names)
    trains_n, tests_n, a_num = _nb.normalize_and_flatten(                                # cell 9
        trains, tests, names, mean_cifar, std_cifar, v_type="rand", num_images=num_images)
    train, y = _nb.build_regressor_training_set(trains_n, tests_n, split_index)         # cell 10
    model, optimizer = _nb.build_regressor(a_num)                                        # cell 11
    _nb.train_regressor(model, optimizer, train, y, epochs=regressor_epochs)             # cell 12
    outputs_tr, outputs_te = _nb.score_features(model, trains_n, tests_n, names)        # cell 15
    outputs_tr, outputs_te = _nb.hold_out(outputs_tr, outputs_te, names, split_index)   # cell 17

    results = {}
    for name in names:                                                                   # cell 18
        print(f"{name}")
        _nb.print_inference(outputs_tr[name], outputs_te[name])
        pval, mean_diff = _nb.inference_stats(outputs_tr[name], outputs_te[name])
        if per_name_reseed:
            torch.manual_seed(seed)
        tab = _nb.generate_table(outputs_tr, outputs_te, [name],                         # cell 29
                                 selected_m=selected_m, total_inner_rep=total_inner_rep, order=[name])
        results[name] = {
            "p_value_welch": pval,
            "mean_diff_welch": mean_diff,
            "p_value_m": float(tab.loc[name, "p_value"]),
            "mean_diff_m": float(tab.loc[name, "mean_diff"]),
        }
    return results
