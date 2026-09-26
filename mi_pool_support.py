"""Lightweight, testable subset-cache and MI resume helpers (no Torch imports)."""
import csv
import math
import os
from decimal import Decimal
from pathlib import Path
import tempfile

import numpy as np


def sizes_from_rates(training_size, rates):
    if type(training_size) is not int or training_size <= 0:
        raise ValueError("training_size must be a positive integer")
    if not rates or len(set(rates)) != len(rates):
        raise ValueError("Rates must be nonempty and unique")
    sizes = []
    for rate in rates:
        if type(rate) not in (int, float) or not 0 < rate <= 1:
            raise ValueError("Use rates in (0, 1], e.g. 0.05")
        count = Decimal(str(rate)) * training_size
        if count != count.to_integral_value() or count < 1:
            raise ValueError(f"Nonintegral sample count: {training_size} * {rate}")
        sizes.append(int(count))
    return sizes


def positive_ints(values, name):
    values = list(values)
    if not values or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


def create_nested_balanced_subsets(dataset, group_A, save_dir, subset_sizes,
                                   num_classes=10, seed=42, force_rebuild=False):
    """Exact sizes, fixed remainder allocation, legacy-compatible nested sets.

    The old per-class shuffle is preserved. One independent seeded class order
    allocates remainders at every size, making each set a prefix of a balanced
    round-robin sequence. Existing arrays (including their order) are retained.
    An incompatible cache raises instead of invalidating already computed MI.
    force_rebuild=True explicitly rebuilds; do not use it with published MI.
    """
    requested = positive_ints(subset_sizes, "subset_sizes")
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be positive")
    group = np.asarray(group_A)
    if group.ndim != 1 or not np.issubdtype(group.dtype, np.integer) or len(set(group.tolist())) != len(group):
        raise ValueError("group_A must contain unique integer indices")
    if (group < 0).any() or (group >= len(dataset)).any():
        raise ValueError("group_A indices out of dataset bounds")
    target = Path(save_dir) / f"nested_subsets_seed{seed}.npz"
    cached = {}
    if target.exists() and not force_rebuild:
        with np.load(target, allow_pickle=False) as data:
            for key in data.files:
                if not key.startswith("size_") or not key[5:].isdigit():
                    raise ValueError(f"Unexpected cache key {key}: {target}")
                cached[int(key[5:])] = data[key].copy()
    # Prefer labels directly, avoiding image decode/augmentation during validation.
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "labels", None)
    classes = [[] for _ in range(num_classes)]
    for idx in group:
        label = int(labels[int(idx)] if labels is not None else dataset[int(idx)][1])
        if not 0 <= label < num_classes:
            raise ValueError(f"Invalid class label {label}")
        classes[label].append(int(idx))
    rng = np.random.default_rng(seed)
    for c in range(num_classes):
        classes[c] = np.asarray(classes[c], dtype=np.int64)
        rng.shuffle(classes[c])  # Exactly the legacy RNG call order.
    class_order = np.random.default_rng(seed).permutation(num_classes)
    rank = np.argsort(class_order)
    result = dict(cached)
    for size in sorted(set(requested) | set(cached)):
        q, remainder = divmod(size, num_classes)
        counts = q + (rank < remainder).astype(int)
        if size <= 0 or any(n > len(classes[c]) for c, n in enumerate(counts)):
            raise ValueError(f"Cannot create balanced subset size={size} from group_A")
        indices = np.concatenate([classes[c][:n] for c, n in enumerate(counts)])
        if size in cached:
            old = cached[size]
            if (old.ndim != 1 or not np.issubdtype(old.dtype, np.integer)
                    or len(old) != size or not np.array_equal(np.sort(old), np.sort(indices))):
                raise ValueError(f"Incompatible cached subset size={size}: {target}; "
                                 "cache preserved, do not rebuild while reusing old MI")
        else:
            np.random.default_rng(seed + size).shuffle(indices)
            result[size] = indices
    if force_rebuild or set(result) != set(cached):
        target.parent.mkdir(parents=True, exist_ok=True)
        # Same-directory temporary file + atomic replace prevents a partial NPZ.
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            np.savez(temporary, **{f"size_{s}": result[s] for s in sorted(result)})
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return result


def missing_mi_grid(csv_path, model_name, scenario, seed, rate, in_sizes, bins):
    """Validate model rows, return missing pairs and reusable out-sample MI.

    Invalid/duplicate rows are errors, never silently treated as completed.
    Existing rows at other sizes are kept and can supply out-sample MI by bin.
    This is a single-writer resume protocol, not a concurrent CSV job queue.
    """
    sizes = positive_ints(in_sizes, "in_sizes")
    bins = positive_ints(bins, "bins")
    present, out = set(), {}
    path = Path(csv_path)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["model_name"] != model_name:
                    continue
                size_f, bin_f = float(row["in_size"]), float(row["bins"])
                if not size_f.is_integer() or not bin_f.is_integer() or min(size_f, bin_f) <= 0:
                    raise ValueError(f"Invalid MI grid key: {model_name}")
                key = (int(size_f), int(bin_f))
                if key in present:
                    raise ValueError(f"Duplicate MI row: {model_name}, {key}")
                if row["Scenario"] != scenario or float(row["seed"]) != seed or float(row["rate"]) != rate:
                    raise ValueError(f"MI identity mismatch: {model_name}, {key}")
                vals = [float(row[c]) for c in ["I(X;T)-In", "I(T;Y)-In", "I(X;T)-Out", "I(T;Y)-Out"]]
                if not all(math.isfinite(v) for v in vals):
                    raise ValueError(f"Nonfinite MI: {model_name}, {key}")
                out_size = float(row["out_size"])
                if not out_size.is_integer() or out_size <= 0:
                    raise ValueError(f"Invalid out_size: {model_name}")
                value = (*vals[2:], int(out_size))
                if key[1] in out and out[key[1]] != value:
                    raise ValueError(f"Inconsistent out-sample MI: {model_name}, bins={key[1]}")
                present.add(key)
                out[key[1]] = value
    return [(s, b) for s in sizes for b in bins if (s, b) not in present], out
