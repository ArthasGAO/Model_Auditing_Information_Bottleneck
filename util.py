import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import yaml

from Dataset.CIFAR_10 import CIFAR10Dataset
from Dataset.CIFAR_100 import CIFAR100Dataset
from KnowledgeDistillation.DKD import DKD
from KnowledgeDistillation.KD import KD
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_worker(worker_id):
    """DataLoader worker seed derived deterministically from the main seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)   # undo a previous deterministic MI pass


STAGES = {
    "victim":       {"plan_dir": "saved_exp_plan/victim"},
    "negative":     {"plan_dir": "saved_exp_plan/victim"},   # same recipes, disjoint split
    "fine_tune":    {"plan_dir": "saved_exp_plan/fine_tune"},
    "prune":        {"plan_dir": "saved_exp_plan/prune"},
    "distillation": {"plan_dir": "saved_exp_plan/distillation"},
    "extraction":   {"plan_dir": "saved_exp_plan/extraction"},
}
MODEL_ROOT = Path("saved_models")
LOG_ROOT = Path("saved_logs")
INDICES_ROOT = Path("Indices")

VICTIM_SEED = 42        # the one victim per scenario
VICTIM_RATE = 1.0       # victim trains on group_A
NEGATIVE_RATE = 0.0     # negatives and every positive train on the disjoint group_B
GROUP_SEED = 42         # seed of the frozen group_A / group_B draw
BEST_CHECKPOINT = "best_epoch.pth"


def stage_plan_dir(stage) -> Path:
    return Path(STAGES[stage]["plan_dir"])


def stage_model_root(stage) -> Path:
    return MODEL_ROOT / stage


def stage_log_root(stage) -> Path:
    return LOG_ROOT / stage


def model_name(stage, plan, **ids):
    scen = plan["Scenario_Name"]
    if stage in ("victim", "negative"):
        return f"{scen}_{ids['seed']}_{round(float(ids['rate']), 2)}"
    if stage == "fine_tune":
        return (f"{scen}_{ids['model_seed']}_{round(float(ids['rate']), 2)}_{ids['strategy']}"
                f"_ftsize={ids['ft_size']}_ftseed={ids['ft_seed']}")
    if stage == "prune":
        return (f"{scen}_{ids['model_seed']}_{round(float(ids['rate']), 2)}"
                f"_sparsity={round(float(ids['sparsity']), 6)}_{ids['strategy']}"
                f"_ftsize={ids['ft_size']}_ftseed={ids['ft_seed']}")
    if stage == "distillation":
        return f"{scen}_{ids['method']}_{ids['seed']}_{round(float(ids['rate']), 2)}"
    if stage == "extraction":
        return f"{scen}_{ids['seed']}_{round(float(ids['rate']), 2)}"
    raise ValueError(f"Unknown stage {stage!r}; known: {sorted(STAGES)}")


def model_dir(stage, plan, **ids) -> Path:
    return stage_model_root(stage) / model_name(stage, plan, **ids)


def training_log_path(stage, name) -> Path:
    return stage_log_root(stage) / "Performance" / f"training_log_{name}.csv"


def mi_table_path(stage) -> Path:
    """The single-writer MI master table of one stage."""
    return stage_log_root(stage) / f"MI_{stage}.csv"


def indices_dir(dataset_cfg) -> Path:
    """Where the frozen index arrays of one dataset live (group_A, group_B, nested MI subsets)."""
    name = dataset_cfg["name"] if isinstance(dataset_cfg, dict) else str(dataset_cfg)
    return INDICES_ROOT / name


def victim_scenario(plan):
    if "Victim_Scenario" in plan:
        return plan["Victim_Scenario"]
    raise KeyError(
        f"Plan {plan.get('Scenario_Name')!r} has no `Victim_Scenario` key. Every positive "
        "plan must name its victim with `Victim_Scenario: <victim Scenario_Name>`."
    )


def victim_dir(plan, seed=VICTIM_SEED) -> Path:
    return stage_model_root("victim") / f"{victim_scenario(plan)}_{seed}_{round(VICTIM_RATE, 2)}"


def victim_checkpoint(plan, seed=VICTIM_SEED) -> Path:
    path = victim_dir(plan, seed) / BEST_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(
            f"Victim checkpoint not found: {path}. Train it first with main_victim.py "
            f"(plan whose Scenario_Name is {victim_scenario(plan)!r})."
        )
    return path


def victim_dataset_cfg(plan):
    if "Victim" in plan and isinstance(plan["Victim"], dict) and "Dataset" in plan["Victim"]:
        return plan["Victim"]["Dataset"]
    return plan["Dataset"]


def victim_arch(plan):
    if "Victim" in plan and isinstance(plan["Victim"], dict) and "Model" in plan["Victim"]:
        return plan["Victim"]["Model"]
    if "Teacher_Model" in plan:
        return plan["Teacher_Model"]["teacher_name"]
    return plan["Model"]


def suspect_arch(stage, plan):
    if stage == "distillation":
        return plan["Student_Model"]["student_name"]
    if stage == "extraction":
        return plan["Substitute"].get("Model", victim_arch(plan))
    return plan["Model"]


def plan_files(stage, filters=(), plan_dir=None):
    files = sorted(Path(plan_dir if plan_dir else stage_plan_dir(stage)).glob("*.yaml"))
    if not filters:
        return files
    chosen = []
    for f in (f.lower() for f in filters):
        hits = [p for p in files if f == p.stem.lower() or f in p.stem.lower().split("_")]
        if not hits:
            hits = [p for p in files if f in p.name.lower()]
        chosen.extend(p for p in hits if p not in chosen)
    return sorted(chosen)


def parse_seeds(spec):
    """'42' | '0,1,2' | '42:122' (half-open) -> list of ints."""
    spec = str(spec).strip()
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in spec.split(",") if s.strip()]


def set_output_root(root):
    global MODEL_ROOT, LOG_ROOT, INDICES_ROOT
    root = Path(root)
    MODEL_ROOT, LOG_ROOT, INDICES_ROOT = root / "saved_models", root / "saved_logs", root / "Indices"


def build_argparser(stage, description):
    """The command line shared by the six training entry points."""
    import argparse
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plans", nargs="*", default=(), metavar="NAME",
                   help="only these plans: a file stem (CIFAR10_ResNet18) or one of its `_` tokens (VGG16)")
    p.add_argument("--plan-dir", default=None, help=f"plan folder (default {STAGES[stage]['plan_dir']})")
    p.add_argument("--seeds", default=None,
                   help=f"'42' | '0,1,2' | '42:122' half-open (default {_seed_summary(stage)})")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers (default 4; 0 on small machines)")
    p.add_argument("--data-root", default=None, help="override Dataset.root_dir of every plan (CIFAR download dir)")
    p.add_argument("--output-root", default=None, help="rebase saved_models/, saved_logs/, Indices/ under this folder")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the plan's epoch count; for smoke tests only, the recipe is the plan's")
    p.add_argument("--no-skip", action="store_true", help="retrain cells whose best_epoch.pth already exists")
    p.add_argument("--no-mi", action="store_true", help="do not measure MI after training (calculate_MI.py can do it later)")
    return p


def _seed_summary(stage):
    seeds = DEFAULT_SEEDS[stage]
    return f"{seeds[0]}:{seeds[-1] + 1}" if len(seeds) > 1 else str(seeds[0])


def apply_common_args(args):
    if args.output_root:
        set_output_root(args.output_root)
    return args


def load_plan(plan_path, data_root=None):
    plan = process_yaml_file(plan_path)
    if plan is None:
        raise FileNotFoundError(plan_path)
    if data_root:
        for key in ("Dataset", "FT_Dataset", "Auxiliary_Dataset"):
            if isinstance(plan.get(key), dict):
                plan[key]["root_dir"] = data_root
        if isinstance(plan.get("Victim"), dict) and isinstance(plan["Victim"].get("Dataset"), dict):
            plan["Victim"]["Dataset"]["root_dir"] = data_root
    return plan


def checkpoint_exists(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def _to_tuple_if_list(x):
    return tuple(x) if isinstance(x, list) else x

def normalize_transform_specs(specs):
    if not specs:
        return specs

    out = []
    for spec in specs:
        if spec is None:
            continue
        spec = dict(spec)  # shallow copy
        params = dict(spec.get("params", {}) or {})

        # Common list->tuple conversions
        for k in ["size", "scale", "ratio", "mean", "std"]:
            if k in params:
                params[k] = _to_tuple_if_list(params[k])

        spec["params"] = params
        out.append(spec)
    return out

def process_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def build_dataset_from_yaml(ds_cfg):
    if isinstance(ds_cfg, str):
        ds_name, ds_params = ds_cfg.strip(), {}
    elif isinstance(ds_cfg, dict):
        ds_name = str(ds_cfg.get("name", ds_cfg.get("dataset", ""))).strip()
        ds_params = ds_cfg
    else:
        raise ValueError(f"Invalid Dataset config: {ds_cfg} (type={type(ds_cfg)})")

    normalization = ds_params.get("normalization", "cifar10" if ds_name == "CIFAR-10" else "cifar100")
    root_dir = ds_params.get("root_dir", ds_params.get("root", "./data"))
    loading = ds_params.get("loading", "torchvision")
    img_size = int(ds_params.get("img_size", 32))
    download = bool(ds_params.get("download", True))
    group_size = int(ds_params.get("group_size", 50000))
    train_tf_specs = normalize_transform_specs(ds_params.get("train_transforms", None))
    test_tf_specs = normalize_transform_specs(ds_params.get("test_transforms", None))

    if ds_name == "CIFAR-10":
        ds_obj = CIFAR10Dataset(normalization=normalization, loading=loading, root_dir=root_dir,
                                img_size=img_size, train_transforms=train_tf_specs,
                                test_transforms=test_tf_specs, download=download)
        num_classes = 10
    elif ds_name in ("CIFAR-100", "CIFAR-100_COPY"):
        ds_obj = CIFAR100Dataset(normalization=normalization, loading=loading, root_dir=root_dir,
                                 img_size=img_size, train_transforms=train_tf_specs,
                                 test_transforms=test_tf_specs, download=download)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset {ds_name!r}; this repository covers CIFAR-10 and CIFAR-100")
    return ds_obj, num_classes, group_size


ARCHITECTURES = ("ResNet-18", "VGG16")


def build_model(model_name, num_classes):
    """Classifier returning logits; the same class serves every stage, including the
    teacher and student of a distillation run."""
    if model_name == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    if model_name == "VGG16":
        return ModifiedVGG16(num_classes=num_classes)
    raise ValueError(f"Unsupported model {model_name!r}; this repository covers {ARCHITECTURES}")


def build_optimizer_scheduler(params, optimizer_name, optimizer_params, scheduler_name,
                              scheduler_params, epochs):
    """torch optimizer + scheduler from plan fields. `T_max: auto` -> epochs."""
    optimizer = getattr(optim, optimizer_name)(params, **optimizer_params)
    scheduler = None
    if scheduler_name:
        s_params = dict(scheduler_params or {})
        if s_params.get("T_max") == "auto":
            s_params["T_max"] = epochs
        scheduler = getattr(lr_sched, scheduler_name)(optimizer, **s_params)
    return optimizer, scheduler


def build_base_criterion(aug_cfg: dict, use_mixup: bool = False):
    """Ground-truth CE term. Label smoothing uses torch's built-in, which computes the
    same (1-eps)*NLL + eps*mean(-logprob) as timm's LabelSmoothingCrossEntropy."""
    if use_mixup:
        raise ValueError("mixup is not supported in this repository")
    ls = float(aug_cfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()


def process_experiment_setup(data, epochs=None):
    """Victim / negative plan -> dataset, fresh model, optimizer, scheduler, epochs.
    `epochs` overrides the plan's count and sizes the scheduler accordingly."""
    result = {}
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(data.get("Dataset"))
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size

    model = build_model(data.get("Model", ""), num_classes)
    result["Model"] = model

    optimizer_cfg = data.get("Optimizer", {})
    scheduler_cfg = data.get("Scheduler", {}) or {}
    result["Epochs"] = int(data.get("Epochs", 100) if epochs is None else epochs)
    optimizer, scheduler = build_optimizer_scheduler(
        model.parameters(), optimizer_cfg.get("name", "Adam"), optimizer_cfg.get("params", {"lr": 1e-3}),
        scheduler_cfg.get("name"), scheduler_cfg.get("params", {}), result["Epochs"])
    result["Optimizer"] = optimizer
    result["Scheduler"] = scheduler
    return result


def process_experiment_ft_setup(data):
    """Fine-tuning plan -> victim dataset, FT dataset, model factory, per-strategy configs."""
    result = {}
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(data.get("Dataset"))
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size

    # FT_Dataset only sizes the fine-tuning split, group_B of the same dataset (see determine_ft_dataset)
    result["FT_GroupSize"] = int((data.get("FT_Dataset") or {}).get("group_size", group_size))

    model_name_ = data.get("Model", "")
    result["Model_Factory"] = lambda: build_model(model_name_, num_classes)

    result["FT_Setups"] = {}
    scheduler_cfg = data.get("Scheduler", None)
    for opt_cfg in data.get("Optimizers", []):
        strategy = opt_cfg["strategy"]
        if strategy in result["FT_Setups"]:
            raise ValueError(f"Duplicate fine-tune strategy found in YAML: {strategy}")
        result["FT_Setups"][strategy] = {
            "Optimizer_Name": opt_cfg.get("name", "Adam"),
            "Optimizer_Params": opt_cfg.get("params", {}).copy(),
            "Epochs": opt_cfg.get("Epochs", 100),
            "Scheduler_Name": scheduler_cfg.get("name") if scheduler_cfg else None,
            "Scheduler_Params": scheduler_cfg.get("params", {}).copy() if scheduler_cfg else None,
        }
    return result


def process_experiment_prune_setup(data):
    """Pruning plan -> victim dataset, FT dataset, model factory, per-sparsity configs."""
    result = {}
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(data.get("Dataset"))
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size

    model_name_ = data.get("Model", "")
    result["Model_Factory"] = lambda: build_model(model_name_, num_classes)

    # FT_Dataset only sizes the fine-tuning split, group_B of the same dataset (see determine_ft_dataset)
    result["FT_GroupSize"] = int((data.get("FT_Dataset") or {}).get("group_size", group_size))

    result["Prune_Setups"] = {}
    scheduler_cfg = data.get("Scheduler", None)
    for opt_cfg in data.get("Optimizers", []):
        s_key = round(float(opt_cfg["sparsity"]), 6)
        if s_key in result["Prune_Setups"]:
            raise ValueError(f"Duplicate sparsity setting found in YAML: {s_key}")
        result["Prune_Setups"][s_key] = {
            "Optimizer_Name": opt_cfg.get("name", "Adam"),
            "Optimizer_Params": opt_cfg.get("params", {}).copy(),
            "Epochs": opt_cfg.get("Epochs", 100),
            "Scheduler_Name": scheduler_cfg.get("name") if scheduler_cfg else None,
            "Scheduler_Params": scheduler_cfg.get("params", {}).copy() if scheduler_cfg else None,
        }
    return result


def sparsity_levels_from_setup(exp_setup, fallback=None):
    levels = sorted(exp_setup.get("Prune_Setups", {}))
    return levels if levels else list(fallback or [])


def process_experiment_kd_setup(data, teacher_ckpt=None):
    """Distillation plan -> dataset, a (student, teacher) factory and per-method configs.

    The teacher is the victim named by `Victim_Scenario`; its checkpoint is resolved
    through victim_checkpoint() unless `teacher_ckpt` is given explicitly.
    """
    result = {}
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["Epochs"] = data.get("Epochs", 200)
    result["GroupSize"] = group_size

    teacher_cfg = data.get("Teacher_Model", {})
    student_cfg = data.get("Student_Model", {})
    teacher_name = teacher_cfg.get("teacher_name", "ResNet-18")
    student_name = student_cfg.get("student_name", "ResNet-18")
    ckpt = Path(teacher_ckpt) if teacher_ckpt is not None else victim_checkpoint(data)
    result["Teacher_Checkpoint"] = ckpt

    def create_st_te_model():
        teacher_model = build_model(teacher_name, num_classes)
        teacher_model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        student_model = build_model(student_name, num_classes)
        return student_model, teacher_model


    aug_cfg = data.get("Augmentation", {}) or {}
    if bool(aug_cfg.get("use_mixup", False)):
        raise ValueError("Augmentation.use_mixup is not supported for distillation")
    ce_criterion = build_base_criterion(aug_cfg, use_mixup=False)

    result["KD_Setups"] = {}
    opt_template = data.get("Optimizer", {})
    sched_template = data.get("Scheduler", {})
    for d_cfg in data.get("Distillation", []):
        method_name_ = d_cfg.get("name")
        params = d_cfg.get("params", {}).copy()
        if method_name_ in result["KD_Setups"]:
            raise ValueError(f"Duplicate distillation method found in YAML: {method_name_}")

        def make_bundle(method_name_=method_name_, params=params):
            student, teacher = create_st_te_model()
            if method_name_ == "KD":
                return KD(student, teacher,
                          temperature=params.get("TEMPERATURE", 4),
                          ce_weight=params.get("CE_WEIGHT", 0.1),
                          kd_weight=params.get("KD_WEIGHT", 0.9),
                          ce_criterion=ce_criterion)
            if method_name_ == "DKD":
                return DKD(student, teacher,
                           ce_weight=params.get("CE_WEIGHT", 1.0),
                           alpha=params.get("ALPHA", 1.0),
                           beta=params.get("BETA", 8.0),
                           temperature=params.get("T", 4.0),
                           warmup=params.get("WARMUP", 20),
                           ce_criterion=ce_criterion)
            raise ValueError(f"Unsupported distillation method: {method_name_} (KD, DKD)")

        result["KD_Setups"][method_name_] = {
            "Builder": make_bundle,
            "Optimizer_Name": opt_template.get("name", "SGD"),
            "Optimizer_Params": opt_template.get("params", {}).copy(),
            "Scheduler_Name": sched_template.get("name") if sched_template else None,
            "Scheduler_Params": sched_template.get("params", {}).copy() if sched_template else None,
        }
    return result



def save_indices(indices, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(indices, dtype=np.int64))

def load_indices(path):
    return np.load(path).tolist()


def create_or_load_group_A(
    dataset,
    save_dir,
    group_size=25000,
    num_classes=10,
    seed=42,
    force_rebuild=False
):
    save_path = Path(save_dir) / f"group_A_{group_size}_seed{seed}.npy"

    if save_path.exists() and not force_rebuild:
        print(f"[INFO] Loading Group A from {save_path}")
        return load_indices(save_path)

    print("[INFO] Creating new Group A")

    rng = np.random.default_rng(seed)
    n_per_class = group_size // num_classes

    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
    else:
        labels = np.asarray([dataset[i][1] for i in range(len(dataset))])

    # Group indices by class in one vectorized pass
    class_to_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

    group_A = []
    for c in range(num_classes):
        indices = class_to_indices[c].copy()
        rng.shuffle(indices)
        group_A.extend(indices[:n_per_class].tolist())

    rng.shuffle(group_A)
    save_indices(group_A, save_path)
    return group_A


def create_or_load_group_B(
    save_dir,
    overlap_rate,
    group_A_indices,
    dataset,
    group_size=10000,
    num_classes=10,
    seed=42,
    force_rebuild=False
):
    overlap_rate = round(overlap_rate, 2)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    A_size = len(group_A_indices)
    save_path = Path(save_dir) / f"group_B_{A_size}_{overlap_rate}_{group_size}_seed{seed}.npy"

    if save_path.exists() and not force_rebuild:
        print(f"[INFO] Loading Group B from {save_path}")
        return np.load(save_path).tolist()  # return list, consistent with group_A

    print(f"[INFO] Creating Group B (overlap={overlap_rate})")

    rng = np.random.default_rng(seed)
    n_per_class = group_size // num_classes
    K_c = int(n_per_class * overlap_rate)  # K samples drawn from A (overlap)
    U_c = n_per_class - K_c                # U samples drawn from non-A (unique)

    if hasattr(dataset, "targets"):
        all_labels = np.asarray(dataset.targets)
    elif hasattr(dataset, "Y"):
        all_labels = np.asarray(dataset.Y)
    else:
        all_labels = np.asarray([dataset[i][1] for i in range(len(dataset))])

    group_A_arr = np.asarray(group_A_indices)
    in_A_mask = np.zeros(len(all_labels), dtype=bool)
    in_A_mask[group_A_arr] = True

    group_B = []
    for c in range(num_classes):
        class_mask = (all_labels == c)
        # Candidates IN A and of class c (for the K_c overlap samples)
        A_candidates = np.where(class_mask & in_A_mask)[0]
        # Candidates NOT IN A and of class c (for the U_c unique samples)
        nonA_candidates = np.where(class_mask & ~in_A_mask)[0]

        if len(A_candidates) < K_c:
            raise ValueError(
                f"Not enough overlap samples in class {c}. "
                f"Have {len(A_candidates)}, need {K_c}"
            )
        if len(nonA_candidates) < U_c:
            raise ValueError(
                f"Not enough unique samples in class {c}. "
                f"Have {len(nonA_candidates)}, need {U_c}"
            )

        rng.shuffle(A_candidates)
        rng.shuffle(nonA_candidates)
        group_B.extend(A_candidates[:K_c].tolist())
        group_B.extend(nonA_candidates[:U_c].tolist())

    rng.shuffle(group_B)
    np.save(save_path, np.asarray(group_B))
    return group_B



def load_group_A(dataset_obj, ds_cfg, group_size, num_classes):
    """The victim's frozen training indices for the dataset a plan describes."""
    return create_or_load_group_A(dataset=dataset_obj.train_set, save_dir=str(indices_dir(ds_cfg)),
                                  group_size=group_size, num_classes=num_classes,
                                  seed=GROUP_SEED, force_rebuild=False)


def load_group_B(dataset_obj, ds_cfg, group_A, group_size, num_classes, overlap_rate=NEGATIVE_RATE):
    """The attacker's frozen split: `overlap_rate` of it drawn from group_A, the rest disjoint."""
    return create_or_load_group_B(dataset=dataset_obj.train_set, save_dir=str(indices_dir(ds_cfg)),
                                  group_A_indices=group_A, group_size=group_size,
                                  num_classes=num_classes, overlap_rate=overlap_rate,
                                  seed=GROUP_SEED, force_rebuild=False)


def determine_ft_dataset(exp_yaml, exp_setup, group_A):
    """Fine-tuning data: group_B of the victim's dataset (|group_B| = FT_GroupSize) and its test set."""
    ft_name = (exp_yaml.get("FT_Dataset") or {}).get("name", exp_yaml["Dataset"]["name"])
    if ft_name != exp_yaml["Dataset"]["name"]:
        raise ValueError(
            f"FT_Dataset.name={ft_name!r} differs from Dataset.name={exp_yaml['Dataset']['name']!r}; "
            "only same-distribution fine-tuning (group_B of the victim's dataset) is supported here.")
    group_B = load_group_B(exp_setup["Dataset"], exp_yaml["Dataset"], group_A,
                           exp_setup["FT_GroupSize"], exp_setup["NumClasses"])
    ft_data_train = exp_setup["Dataset"].subset("train", group_B, clean=False)
    ft_data_val = exp_setup["Dataset"].test_set
    return ft_data_train, ft_data_val


def train_one_epoch(net, trainloader, optimizer, criterion, epoch, device):
    print(f'\nEpoch: {epoch}')
    net.train()
    running_loss = correct = total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)   # to avoid gradient spikes
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
    return {"train_loss": running_loss / total, "train_acc": 100. * correct / total}


def train_one_epoch_kd(distiller, trainloader, optimizer, epoch, device):
    print(f'\nEpoch: {epoch}')
    distiller.train()                        # student in train mode, teacher kept in eval
    running_loss = correct = total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, losses_dict = distiller(image=inputs, target=targets, epoch=epoch)   # epoch drives the DKD warmup
        loss = sum(losses_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(distiller.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        correct += logits.argmax(1).eq(targets).sum().item()
    return {"train_loss": running_loss / total, "train_acc": 100. * correct / total}


def evaluate1(net, test_loader, criterion, device):
    net.eval()
    test_loss = correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            test_loss += criterion(outputs, targets).item() * targets.size(0)
            total += targets.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
    return {"test_loss": test_loss / total, "test_acc": 100. * correct / total}


def query_victim(victim_net, dataloader, device, temperature=1.0):
    """Query the victim model (blackbox) on the auxiliary dataset.

    Returns:
        all_inputs: tensor of all input samples [N, C, H, W]
        all_outputs: tensor of soft-label predictions [N, num_classes]
    """
    victim_net.eval()
    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = victim_net(inputs)
            # Convert logits to soft probabilities
            soft_labels = torch.softmax(outputs / temperature, dim=1)

            all_inputs.append(inputs.cpu())
            all_outputs.append(soft_labels.cpu())

    all_inputs = torch.cat(all_inputs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    return all_inputs, all_outputs


def train_one_epoch_knockoff(net, trainloader, optimizer, criterion, epoch, device):
    """One epoch on the victim's soft labels; accuracy is measured against their argmax."""
    print(f'\nEpoch: {epoch}')
    net.train()
    running_loss = correct = total = 0
    for inputs, soft_targets in trainloader:
        inputs, soft_targets = inputs.to(device), soft_targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, soft_targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        correct += outputs.argmax(1).eq(soft_targets.argmax(1)).sum().item()
    return {"train_loss": running_loss / total, "train_acc": 100. * correct / total}


def evaluate_fidelity(victim_net, substitute_net, dataloader, device):
    victim_net.eval()
    substitute_net.eval()

    agree = 0
    total = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            v_pred = victim_net(inputs).argmax(dim=1)
            s_pred = substitute_net(inputs).argmax(dim=1)

            agree += (v_pred == s_pred).sum().item()
            total += inputs.size(0)

    return 100.0 * agree / total


def setup_finetune(model, strategy, num_classes=None, device='cuda',
                   deit_finetune_norm_in_ll=False):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        head_attr = "fc"
        head_modules = [model.fc]
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        head_attr = "classifier"
        head_modules = [model.classifier[-1]]
        if not isinstance(head_modules[0], nn.Linear):
            raise ValueError("Last element in model.classifier is not a Linear layer.")
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        head_attr = "head"
        head_modules = [model.head]
        if hasattr(model, "head_dist") and isinstance(model.head_dist, nn.Linear):
            head_modules.append(model.head_dist)
    else:
        raise ValueError(
            "Model does not have a supported head (.fc / .classifier[-1] / .head)."
        )

    in_features = head_modules[0].in_features
    out_features = head_modules[0].out_features

    if num_classes is None:
        num_classes = out_features

    def replace_heads():
        new_modules = []
        for _ in head_modules:
            new_head = nn.Linear(in_features, num_classes).to(device)
            nn.init.xavier_uniform_(new_head.weight)
            nn.init.zeros_(new_head.bias)
            new_modules.append(new_head)

        if head_attr == "fc":
            model.fc = new_modules[0]
        elif head_attr == "classifier":
            model.classifier[-1] = new_modules[0]
        else:  # "head"
            model.head = new_modules[0]
            if len(new_modules) > 1:
                model.head_dist = new_modules[1]
        return new_modules

    # --- Strategy A: Fine-tune Last Layer ---
    if strategy == 'FT-LL':
        for p in model.parameters():
            p.requires_grad = False

        if num_classes != out_features:
            head_modules = replace_heads()

        for m in head_modules:
            for p in m.parameters():
                p.requires_grad = True

        if head_attr == "head" and deit_finetune_norm_in_ll:
            if hasattr(model, "norm") and model.norm is not None:
                for p in model.norm.parameters():
                    p.requires_grad = True

    # --- Strategy B: Fine-tune All Layers ---
    elif strategy == 'FT-AL':
        for p in model.parameters():
            p.requires_grad = True

    # --- Strategy C: Re-train All Layers（reinit head）---
    elif strategy == 'RT-AL':
        replace_heads()
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown fine-tuning strategy: {strategy}")

    return model


def set_backbone_eval_norm_dropout(model):
    """FT-LL: keep BatchNorm statistics and Dropout frozen while the classifier trains."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
            m.eval()


def ft_one_epoch(net, trainloader, optimizer, criterion, epoch, device, strategy,
                 freeze_backbone_norm=True):
    print(f'\nEpoch: {epoch}')
    net.train()
    if strategy == "FT-LL" and freeze_backbone_norm:
        set_backbone_eval_norm_dropout(net)
    running_loss = correct = total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
    return {"train_loss": running_loss / total, "train_acc": 100. * correct / total}


def prune_model_global(model, amount, exclude_patterns=None):
    if exclude_patterns is None:
        exclude_patterns = []

    # Gather all parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if any(pat in name for pat in exclude_patterns):
            continue  # caller asked for this layer to stay dense
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # Apply global magnitude pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    return model


def check_pruned_weights(net):
    zeros = total = 0
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach()
            zeros += int((w == 0).sum())
            total += w.numel()
    return zeros / total if total else 0.0


def load_state(path, map_location="cpu"):
    """A state_dict from a .pth that may wrap it under `model` / `state_dict`."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


LOG_COLUMNS = ["Epoch", "Train_Loss", "Train_Acc", "Test_Loss", "Test_Acc"]


def _save_atomic(obj, path):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_training(*, scenario_name, epochs, step_fn, eval_net, testloader, criterion, optimizer,
                 scheduler, save_dir, log_file, best_ckpt_start_frac, device=device, export_state=None):
    """Epoch loop shared by every trainer: step_fn(epoch) trains, evaluate1 scores the test set,
    one CSV row per epoch, and the best test-accuracy state from epoch int(epochs * frac) on is
    written to save_dir/best_epoch.pth after the last epoch (so its presence means completion)."""
    save_dir, log_file = Path(save_dir), Path(log_file)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", newline="") as f:          # one log per model; a rerun starts it afresh
        csv.writer(f).writerow(LOG_COLUMNS)
    export = export_state or (lambda net: net.state_dict())

    best_test_acc, best_state = -1.0, None
    best_ckpt_from = int(epochs * best_ckpt_start_frac)
    for epoch in range(epochs):
        train = step_fn(epoch)
        if scheduler is not None:
            scheduler.step()
        test = evaluate1(eval_net, testloader, criterion, device)
        print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']} | train {train['train_acc']:.2f}% "
              f"| test {test['test_acc']:.2f}%")
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train["train_loss"], train["train_acc"], test["test_loss"], test["test_acc"]])
        if epoch >= best_ckpt_from and test["test_acc"] > best_test_acc:
            best_test_acc = test["test_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in export(eval_net).items()}
    if best_state is None:
        raise RuntimeError(f"{scenario_name}: no epoch fell in the best-checkpoint window (epochs={epochs})")
    best_path = save_dir / BEST_CHECKPOINT
    _save_atomic(best_state, best_path)
    return {"best_test_acc": best_test_acc, "best_checkpoint": best_path}
DEFAULT_SEEDS = {
    "victim": [VICTIM_SEED],
    "negative": list(range(42, 122)),      
    "fine_tune": [0, 1, 2],
    "prune": [0, 1, 2],
    "distillation": [0, 1, 2],
    "extraction": [0, 1, 2],
}
PRUNE_STRATEGY = "FT-AL"                   # pruning recovers with FT-AL only


def model_grid(stage, plan, seeds=None, rate=None):
    """Yield the ids of every model cell a plan defines for a stage, in training order."""
    seeds = list(DEFAULT_SEEDS[stage] if seeds is None else seeds)
    if stage in ("victim", "negative"):
        r = (VICTIM_RATE if stage == "victim" else NEGATIVE_RATE) if rate is None else rate
        for seed in seeds:
            yield {"seed": seed, "rate": r}
    elif stage == "fine_tune":
        ft_size = plan["FT_Dataset"]["group_size"]
        for ft_seed in seeds:
            for opt in plan.get("Optimizers", []):
                yield {"model_seed": VICTIM_SEED, "rate": VICTIM_RATE, "strategy": opt["strategy"],
                       "ft_size": ft_size, "ft_seed": ft_seed}
    elif stage == "prune":
        ft_size = plan["FT_Dataset"]["group_size"]
        for ft_seed in seeds:
            for opt in plan.get("Optimizers", []):
                yield {"model_seed": VICTIM_SEED, "rate": VICTIM_RATE,
                       "sparsity": round(float(opt["sparsity"]), 6), "strategy": PRUNE_STRATEGY,
                       "ft_size": ft_size, "ft_seed": ft_seed}
    elif stage == "distillation":
        r = NEGATIVE_RATE if rate is None else rate
        for seed in seeds:
            for method in plan.get("Distillation", []):
                yield {"method": method["name"], "seed": seed, "rate": r}
    elif stage == "extraction":
        for seed in seeds:
            yield {"seed": seed, "rate": VICTIM_RATE}
    else:
        raise ValueError(f"Unknown stage {stage!r}; known: {sorted(STAGES)}")
