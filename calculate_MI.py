import argparse
import csv
import math
import os
import tempfile
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # MI runs in deterministic mode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import util

device = util.device

# The (in_size, bins) grid every table uses. in_size = rate x training group size,
# i.e. 250 ... 25000 for the 25000-image scenarios. 70 cells per model.
IN_SIZE_RATES = [0.01, 0.05, 0.10, 0.20, 0.50, 0.75, 1.00]
BINS = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
SUBSET_SEED = 42          # seed of the nested probe subsets, shared by every table
BATCH_SIZE = 128


# ===========================================================================
# 1. Estimator: softmax -> B equal-width bins -> plug-in I(X;T) = H(T) and I(T;Y)
# ===========================================================================
def _infer_num_classes(net):
    if hasattr(net, "fc"):
        return net.fc.out_features
    elif hasattr(net, "classifier"):
        last_linear = [m for m in net.classifier.modules()
                       if isinstance(m, nn.Linear)][-1]
        return last_linear.out_features
    elif hasattr(net, "num_classes"):
        return net.num_classes
    raise ValueError("Cannot automatically infer num_classes from model.")


def collect_logits(net, data_loader, device):
    """Logits (N, K) and one-hot labels (N, K_y) of the model over the loader; the bins
    sweep reuses them."""
    start_time = time.time()
    num_classes = _infer_num_classes(net)
    layer_T_list, label_list = [], []

    net.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            layer_T_list.append(net(inputs).detach())
            label_list.append(F.one_hot(targets.detach(), num_classes=num_classes).float())

    layer_T = torch.cat(layer_T_list, dim=0).to(dtype=torch.float32)
    label_matrix = torch.cat(label_list, dim=0).to(dtype=torch.float32)
    print(f"logits inference costs: {time.time() - start_time:.1f}s")
    return layer_T, label_matrix


def MI_formula_cal(matrix, p1, p2):
    mask = matrix > 0
    denom = p1[:, None] * p2[None, :]
    ratio = matrix / denom
    log_ratio = torch.log2(ratio)
    return (matrix * log_ratio)[mask].sum()


def mi_from_logits(layer_T, label_matrix, num_intervals=50):
    """(I(X;T), I(T;Y)) in bits after bucketising the softmax into num_intervals bins."""
    device = layer_T.device
    N = layer_T.shape[0]

    T_soft = torch.softmax(layer_T, dim=1)
    bins = torch.linspace(0, 1, num_intervals + 1, device=device, dtype=torch.float32)
    T_discrete = torch.bucketize(T_soft, bins, right=True) - 1
    T_discrete = T_discrete.clamp(0, num_intervals - 1).contiguous()

    unique_T, inverse_idx = torch.unique(T_discrete, dim=0, return_inverse=True)
    K_unique = unique_T.shape[0]

    T_counts = torch.zeros(K_unique, device=device, dtype=torch.float32)
    T_counts.index_add_(0, inverse_idx,
                        torch.ones(N, device=device, dtype=torch.float32))
    p_T = T_counts / N
    mask_T = p_T > 0
    I_X_T = -(p_T[mask_T] * torch.log2(p_T[mask_T])).sum()

    K_y = label_matrix.shape[1]
    TY_counts = torch.zeros((K_unique, K_y), device=device, dtype=torch.float32)
    TY_counts.index_add_(0, inverse_idx, label_matrix)
    TY_matrix = TY_counts / N
    P_T_marg = TY_matrix.sum(dim=1)
    P_Y_marg = TY_matrix.sum(dim=0)
    I_T_Y = MI_formula_cal(TY_matrix, P_T_marg, P_Y_marg)
    return I_X_T.item(), I_T_Y.item()


# ===========================================================================
# 2. Nested class-balanced probe subsets of group_A (cached under Indices/<dataset>/)
# ===========================================================================
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
                                   num_classes=10, seed=42, force_rebuild=False, filename=None):
    requested = positive_ints(subset_sizes, "subset_sizes")
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be positive")
    group = np.asarray(group_A)
    if group.ndim != 1 or not np.issubdtype(group.dtype, np.integer) or len(set(group.tolist())) != len(group):
        raise ValueError("group_A must contain unique integer indices")
    if (group < 0).any() or (group >= len(dataset)).any():
        raise ValueError("group_A indices out of dataset bounds")
    target = Path(save_dir) / (filename or f"nested_subsets_seed{seed}.npz")
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


def nested_subsets_filename(group_size, subset_seed=SUBSET_SEED):
    """Cache file of the nested probe subsets under Indices/<dataset>/."""
    return f"nested_subsets_{int(group_size)}_seed{subset_seed}.npz"


# ===========================================================================
# 3. Master table: one row per (model, in_size, bins), single-writer resume protocol
# ===========================================================================
MI_COLUMNS = [
    "Scenario", "seed", "rate", "model_name", "bins", "in_size", "I(X;T)", "I(T;Y)", "timestamp",
    "stage", "victim_scenario", "suspect_arch", "training_size", "in_size_rate",
    "strategy", "sparsity", "achieved_sparsity", "kd_method", "aux_dataset", "checkpoint",
]


def ensure_table(csv_path):
    """Create the table with MI_COLUMNS, or refuse a file whose header differs."""
    csv_path = Path(csv_path)
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream), None)
        if header != MI_COLUMNS:
            raise ValueError(f"{csv_path} has a different header; not appending to it")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=MI_COLUMNS).writeheader()


def append_row(csv_path, row):
    with Path(csv_path).open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=MI_COLUMNS).writerow({c: row.get(c, "") for c in MI_COLUMNS})


def missing_mi_grid(csv_path, model_name, scenario, seed, rate, in_sizes, bins):
    """Validate the model's rows and return the (in_size, bins) cells still missing.

    Invalid/duplicate rows are errors, never silently treated as completed.
    This is a single-writer resume protocol, not a concurrent CSV job queue.
    """
    sizes = positive_ints(in_sizes, "in_sizes")
    bins = positive_ints(bins, "bins")
    present = set()
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
                vals = [float(row[c]) for c in ["I(X;T)", "I(T;Y)"]]
                if not all(math.isfinite(v) for v in vals):
                    raise ValueError(f"Nonfinite MI: {model_name}, {key}")
                present.add(key)
    return [(s, b) for s in sizes for b in bins if (s, b) not in present]


# ===========================================================================
# 4. Measuring one model
# ===========================================================================
def mi_grid(net, dataset_obj, ds_cfg, group_size, num_classes, in_sizes, bins, *,
            needed_pairs=None, subset_seed=SUBSET_SEED, batch_size=BATCH_SIZE, model_name=""):
    """(I(X;T), I(T;Y)) on the nested group_A subsets for every (in_size, bins).

    Inference runs once per needed in_size; the bins sweep reuses the cached logits.
    `needed_pairs` restricts the work to the cells still missing from a table.
    Returns {(in_size, bins): (ixt, ity)}.
    """
    in_sizes = positive_ints(in_sizes, "in_sizes")
    bins = positive_ints(bins, "bins")
    needed = set(needed_pairs) if needed_pairs is not None else {(s, b) for s in in_sizes for b in bins}
    if not needed:
        return {}
    idx_dir = util.indices_dir(ds_cfg)
    group_A = util.load_group_A(dataset_obj, ds_cfg, group_size, num_classes)
    in_sample_set = dataset_obj.in_sample_set
    # The probe subsets are a function of (group_A, subset_seed); group_A of (dataset,
    # group_size, GROUP_SEED). The cache name carries the group size so two scenarios
    # of one dataset with different training sizes never collide.
    nested = create_nested_balanced_subsets(dataset=in_sample_set, group_A=group_A, save_dir=str(idx_dir),
                                            subset_sizes=in_sizes, num_classes=num_classes,
                                            seed=subset_seed, force_rebuild=False,
                                            filename=nested_subsets_filename(group_size, subset_seed))
    needed_sizes = sorted({s for s, _ in needed})
    needed_bins = sorted({b for _, b in needed})

    net.eval()
    # ---- phase 1: inference, once per in_size ----
    in_cache = {}
    for in_size in needed_sizes:
        subset = dataset_obj.subset("train", nested[in_size].tolist(), clean=True)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=True)
        print(f"  [Phase 1] inference on the group_A subset (size={in_size})")
        in_cache[in_size] = collect_logits(net, loader, device)

    # ---- phase 2: MI on cached logits ----
    results = {}
    for in_size in needed_sizes:
        logits, labels = in_cache[in_size]
        for nb in needed_bins:
            if (in_size, nb) not in needed:
                continue
            ixt, ity = mi_from_logits(logits, labels, num_intervals=nb)
            if not np.isfinite([ixt, ity]).all():
                raise ValueError(f"Nonfinite MI: {model_name}, in_size={in_size}, bins={nb}")
            results[(in_size, nb)] = (ixt, ity)
    del in_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _identity(stage, plan, ids):
    """The identity columns of one model, from the ids the trainer used to name it."""
    scenario = plan["Scenario_Name"]
    row = {"Scenario": scenario, "model_name": util.model_name(stage, plan, **ids), "stage": stage,
           "victim_scenario": scenario if stage in ("victim", "negative") else util.victim_scenario(plan),
           "suspect_arch": util.suspect_arch(stage, plan), "rate": round(float(ids["rate"]), 2)}
    if stage in ("victim", "negative", "distillation", "extraction"):
        row["seed"] = ids["seed"]
    else:                                                   # fine_tune / prune: the recovery seed
        row["seed"], row["strategy"] = ids["ft_seed"], ids["strategy"]
    if stage == "prune":
        row["sparsity"] = round(float(ids["sparsity"]), 6)
    if stage == "distillation":
        row["kd_method"] = ids["method"]
    if stage == "extraction":
        row["aux_dataset"] = plan.get("Auxiliary_Dataset", util.victim_dataset_cfg(plan)).get("name", "")
    return row


def measure_model(net, plan, stage, ids, *, checkpoint=None, csv_path=None, in_size_rates=None,
                  bins=None, extra=None, batch_size=BATCH_SIZE, dataset=None):
    """Measure an in-memory model on the grid and append the missing cells to the stage table.
    Returns the number of rows written."""
    ds_cfg = util.victim_dataset_cfg(plan)
    dataset_obj, num_classes, group_size = dataset or util.build_dataset_from_yaml(ds_cfg)
    rates = list(IN_SIZE_RATES if in_size_rates is None else in_size_rates)
    in_sizes = sizes_from_rates(group_size, rates)
    bins = positive_ints(list(BINS if bins is None else bins), "bins")
    csv_path = Path(csv_path) if csv_path is not None else util.mi_table_path(stage)

    row = _identity(stage, plan, ids)
    row["training_size"] = group_size
    if checkpoint is not None:
        row["checkpoint"] = str(Path(checkpoint).resolve())
    if extra:
        row.update(extra)

    ensure_table(csv_path)
    missing = missing_mi_grid(csv_path, row["model_name"], row["Scenario"], row["seed"], row["rate"],
                              in_sizes, bins)
    if not missing:
        print(f"[SKIP] {row['model_name']}: all {len(in_sizes) * len(bins)} MI cells already in {csv_path}")
        return 0
    print(f"==> [{row['model_name']}] {len(missing)} cell(s) to compute -> {csv_path}")
    net = net.to(device)
    results = mi_grid(net, dataset_obj, ds_cfg, group_size, num_classes, in_sizes, bins,
                      needed_pairs=missing, batch_size=batch_size, model_name=row["model_name"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    written = 0
    for (in_size, nb) in sorted(results):
        ixt, ity = results[(in_size, nb)]
        print(f"  in_size={in_size:>5} bins={nb:>3}: I(X;T)={ixt:.4f} I(T;Y)={ity:.4f}")
        append_row(csv_path, {**row, "bins": nb, "in_size": in_size,
                              "in_size_rate": f"{in_size / group_size:g}",
                              "I(X;T)": f"{ixt:.6f}", "I(T;Y)": f"{ity:.6f}",
                              "timestamp": timestamp})
        written += 1
    return written


def measure_checkpoint(stage, plan, ckpt_path, ids, **kwargs):
    """Build the architecture the stage produces, load `ckpt_path`, and measure it."""
    ds_cfg = util.victim_dataset_cfg(plan)
    dataset = util.build_dataset_from_yaml(ds_cfg)
    num_classes = dataset[1]
    net = util.build_model(util.suspect_arch(stage, plan), num_classes)
    net.load_state_dict(util.load_state(ckpt_path, map_location=device))
    net.to(device).eval()
    extra = dict(kwargs.pop("extra", None) or {})
    if stage == "prune" and "achieved_sparsity" not in extra:
        extra["achieved_sparsity"] = f"{util.check_pruned_weights(net):.6f}"
    return measure_model(net, plan, stage, ids, checkpoint=ckpt_path, extra=extra, dataset=dataset, **kwargs)


# ===========================================================================
# 5. CLI: measure every trained model of a stage that is not yet in its table
# ===========================================================================
def measure_stage(stage, plan_filters=(), seeds=None, in_size_rates=None, bins=None,
                  plan_dir=None, data_root=None, deterministic=True):
    plans = util.plan_files(stage, plan_filters, plan_dir)
    if not plans:
        raise SystemExit(f"No plans in {plan_dir or util.stage_plan_dir(stage)} matching {list(plan_filters)}")
    written = absent = 0
    for plan_path in plans:
        plan = util.load_plan(plan_path, data_root)
        for ids in util.model_grid(stage, plan, seeds=seeds):
            ckpt = util.model_dir(stage, plan, **ids) / util.BEST_CHECKPOINT
            if not util.checkpoint_exists(ckpt):
                print(f"[ABSENT] {ckpt}")
                absent += 1
                continue
            util.set_seed(SUBSET_SEED, deterministic=deterministic)
            written += measure_checkpoint(stage, plan, ckpt, ids, in_size_rates=in_size_rates, bins=bins)
    print(f"\n==> {stage}: {written} new row(s) in {util.mi_table_path(stage)}; {absent} checkpoint(s) absent")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the MI grid of every trained model of a stage")
    parser.add_argument("--stage", required=True, choices=sorted(util.STAGES))
    parser.add_argument("--plans", nargs="*", default=(), help="plan file stems or `_` tokens, e.g. CIFAR10_ResNet18 or VGG16")
    parser.add_argument("--plan-dir", default=None, help="plan folder (default: the stage's own)")
    parser.add_argument("--seeds", default=None, help="e.g. 42:122 or 0,1,2 (default: the stage's own)")
    parser.add_argument("--rates", default=None, help="in_size rates, e.g. 0.05,1.0 (default: full grid)")
    parser.add_argument("--bins", default=None, help="e.g. 50 or 5,10,50 (default: full grid)")
    parser.add_argument("--data-root", default=None, help="override Dataset.root_dir of every plan")
    parser.add_argument("--output-root", default=None, help="where saved_models/, saved_logs/, Indices/ live")
    args = parser.parse_args(argv)
    if args.output_root:
        util.set_output_root(args.output_root)
    seeds = util.parse_seeds(args.seeds) if args.seeds else None
    rates = [float(x) for x in args.rates.split(",")] if args.rates else None
    bins = [int(x) for x in args.bins.split(",")] if args.bins else None
    measure_stage(args.stage, args.plans, seeds, rates, bins, args.plan_dir, args.data_root)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
