import numpy as np
import pandas as pd 
import yaml
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score
from Dataset.CIFAR_10 import CIFAR10Dataset, CIFAR10PseudoLabelDataset, CIFARNetDataset
from Dataset.CIFAR_100 import CIFAR100Dataset
from Dataset.MNIST import MNISTDataset
import torch.nn.functional as F
from Dataset.Tiny_ImageNet import TinyImageNetDataset
from KnowledgeDistillation.KD import KD
from KnowledgeDistillation.DKD import DKD
from KnowledgeDistillation.FitNet import FitNet
from Model.ResNet_18 import ResNet18, ResNet34, ResNet50
from Model.ResNet_18_dist import ResNet18_dist, ResNet10_dist, ResNet34_dist
from Model.VGG16 import ModifiedVGG16, ModifiedVGG19
from Model.VGG16_dist import VGG16_Teacher, VGG8_Student, VGG19_Teacher
from Model.MLP import MNIST_MLP
from Model.DeiT import DeiTKDStudent
from torch.utils.data import Subset, ConcatDataset, Dataset
import torch.nn.utils.prune as prune
import torch.optim.lr_scheduler as lr_sched
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from collections import defaultdict
from pathlib import Path
import pickle
from typing import List, Optional, Dict, Union
from Model.DeiT import DistillationLoss, FusionCELoss
import timm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------
# Helpers for YAML transform specs
# -----------------------------
def _to_tuple_if_list(x):
    return tuple(x) if isinstance(x, list) else x

def normalize_transform_specs(specs):
    """
    Convert YAML-friendly list params -> tuples for torchvision transforms.
    Safe to call on None/empty.
    """
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


# -----------------------------
# Dataset factory (supports old + new YAML)
# -----------------------------
def build_dataset_from_yaml(ds_cfg):
    """
    Supports:
      Old: Dataset: "CIFAR-100"
      New: Dataset:
             name: CIFAR-100
             img_size: 32
             normalization: cifar100
             train_transforms: [...]
             test_transforms: [...]
    """

    # Backward compatible parsing
    if isinstance(ds_cfg, str):
        ds_name = ds_cfg.strip()
        ds_params = {}
    elif isinstance(ds_cfg, dict):
        ds_name = str(ds_cfg.get("name", ds_cfg.get("dataset", ""))).strip()
        ds_params = ds_cfg
    else:
        raise ValueError(f"Invalid Dataset config: {ds_cfg} (type={type(ds_cfg)})")

    # Pull new optional dataset params (safe defaults)
    normalization = ds_params.get("normalization", "cifar100")
    root_dir = ds_params.get("root_dir", ds_params.get("root", "./data"))
    loading = ds_params.get("loading", "torchvision")
    img_size = int(ds_params.get("img_size", 32))
    download = bool(ds_params.get("download", True))
    group_size = int(ds_params.get("group_size", 50000))

    train_tf_specs = normalize_transform_specs(ds_params.get("train_transforms", None))
    test_tf_specs  = normalize_transform_specs(ds_params.get("test_transforms", None))

    # Instantiate dataset wrappers
    if ds_name == "MNIST":
        # If you later refactor MNISTDataset similarly, pass transforms into it too.
        # For now, keep old behavior.
        ds_obj = MNISTDataset()
        num_classes = 10

    elif ds_name == "CIFAR-10":
        # If you refactor CIFAR10Dataset to accept transform specs, wire it here similarly.
        ds_obj = CIFAR10Dataset(
            normalization=normalization,
            loading=loading,
            root_dir=root_dir,
            img_size=img_size,
            train_transforms=train_tf_specs,
            test_transforms=test_tf_specs,
            download=download,
        )
        num_classes = 10

    elif ds_name in ["CIFAR-100", "CIFAR-100_COPY"]:
        # IMPORTANT: This assumes you are using the refactored CIFAR100Dataset
        # that accepts train_transforms and test_transforms.
        ds_obj = CIFAR100Dataset(
            normalization=normalization,
            loading=loading,
            root_dir=root_dir,
            img_size=img_size,
            train_transforms=train_tf_specs,
            test_transforms=test_tf_specs,
            download=download,
        )
        num_classes = 100
    
    elif ds_name == "PseudoLabelCIFAR-10":
        ds_obj = CIFAR10PseudoLabelDataset(
            train_transforms=train_tf_specs,
            test_transforms=test_tf_specs,
        )
        num_classes = 10
        
    elif ds_name == "CIFARNet":
        ds_obj = CIFARNetDataset(
            normalization=normalization,       # Keeps victim CIFAR-10 stats
            img_size=img_size,                 # Triggers the 64x64 -> 32x32 downsample
            train_transforms=train_tf_specs,
            test_transforms=test_tf_specs,
        )
        num_classes = 10

    elif ds_name == "TinyImageNet":      # <-- 加这一支
        ds_obj = TinyImageNetDataset(
            normalization=ds_cfg.get("normalization", "tiny_imagenet"),
            root_dir=ds_cfg.get("root_dir", "./data/tiny-imagenet-200"),
            img_size=ds_cfg.get("img_size", 64),
            train_transforms=ds_cfg.get("train_transforms"),
            test_transforms=ds_cfg.get("test_transforms"),
            download=ds_cfg.get("download", True),
        )

        num_classes = 200
    else:
        return None, None, group_size

    return ds_obj, num_classes, group_size


def process_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        print(f"this time experiment blueprint:{data}")
        return data

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def process_experiment_setup(data):
    result = {}

    # -----------------------------
    # Dataset setup (now supports YAML-driven transforms)
    # -----------------------------
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size

    # -----------------------------
    # Model setup
    # -----------------------------
    model_name = data.get("Model", "")
    if model_name == "MLP":
        model = MNIST_MLP()
    elif model_name == "ResNet-18":
        model = ResNet18(num_classes=num_classes)
    elif model_name == "ResNet-34":
        model = ResNet34(num_classes=num_classes)
    elif model_name == "ResNet-50":
        model = ResNet50(num_classes=num_classes)
    elif model_name == "ResNet-10":
        model = ResNet10_dist(num_classes=num_classes)
    elif model_name == "VGG16":
        model = ModifiedVGG16(num_classes=num_classes)
    elif model_name == "VGG19":
        # Bigger same-family victim for the VGG16 KD row
        # (saved_exp_plan/train_plan/teacher_plan/*VGG19*).
        model = ModifiedVGG19(num_classes=num_classes)
    elif model_name == "VGG8":
        model = VGG8_Student(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    result["Model"] = model

    # -----------------------------
    # Optimizer setup
    # -----------------------------
    optimizer_cfg = data.get("Optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "Adam")
    optimizer_params = optimizer_cfg.get("params", {"lr": 1e-3})

    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    result["Optimizer"] = optimizer

    # -----------------------------
    # Scheduler setup (new)
    # -----------------------------
    scheduler_cfg = data.get("Scheduler", {})

    scheduler = None
    if scheduler_cfg:
        scheduler_name = scheduler_cfg.get("name", None)
        scheduler_params = scheduler_cfg.get("params", {})

        if scheduler_name is not None:
            scheduler_class = getattr(lr_sched, scheduler_name)
            scheduler = scheduler_class(optimizer, **scheduler_params)

    result["Scheduler"] = scheduler

    # -----------------------------
    # Epoch setup
    # -----------------------------
    result["Epochs"] = data.get("Epochs", 100)

    return result


def process_experiment_ft_setup(data):
    result = {}

    # -----------------------------
    # Dataset setup
    # -----------------------------
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size

    # -----------------------------
    # FT Dataset setup
    # -----------------------------
    result["FT_Dataset"] = None
    ft_data_cfg = data.get("FT_Dataset")
    if ft_data_cfg:
        ft_dataset_obj, ft_num_classes, ft_group_size = build_dataset_from_yaml(ft_data_cfg)
        result["FT_Dataset"] = ft_dataset_obj
        result["FT_NumClasses"] = ft_num_classes
        result["FT_GroupSize"] = ft_group_size

    # -----------------------------
    # Model factory setup
    # -----------------------------
    model_name = data.get("Model", "")

    def build_model():
        if model_name == "MLP":
            return MNIST_MLP()
        elif model_name == "ResNet-18":
            return ResNet18(num_classes=num_classes)
        elif model_name == "VGG16":
            return ModifiedVGG16(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    result["Model_Name"] = model_name
    result["Model_Factory"] = build_model

    # -----------------------------
    # Per-strategy FT configs
    # -----------------------------
    result["FT_Setups"] = {}

    optimizer_cfg_list = data.get("Optimizers", [])
    scheduler_cfg = data.get("Scheduler", None)

    for opt_cfg in optimizer_cfg_list:
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


def process_experiment_ft_setup_deit(data):
    """
    DeiT 版本的 fine-tune setup。
    与 CNN 版 process_experiment_ft_setup 的唯一区别:
      - Model_Factory 用 timm.create_model 构建 DeiT(解析 dict 形式的 Model 字段)
    其余(Dataset / FT_Dataset / FT_Setups)逻辑完全一致。
    """
    result = {}
 
    # ---- Dataset ----
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size
 
    # ---- FT Dataset ----
    result["FT_Dataset"] = None
    ft_data_cfg = data.get("FT_Dataset")
    if ft_data_cfg:
        ft_dataset_obj, ft_num_classes, ft_group_size = build_dataset_from_yaml(ft_data_cfg)
        result["FT_Dataset"] = ft_dataset_obj
        result["FT_NumClasses"] = ft_num_classes
        result["FT_GroupSize"] = ft_group_size
 
    # ---- Model factory(关键差异:构建 DeiT)----
    model_cfg = data.get("Model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "DeiT ft setup 期望 Model 是一个 dict(含 model_name/img_size/patch_size 等),"
            f"但收到: {type(model_cfg)}。请用结构化的 Model 配置。"
        )
 
    deit_model_name = model_cfg.get("model_name", "deit_tiny_patch16_224")
    img_size = int(model_cfg.get("img_size", 32))
    patch_size = int(model_cfg.get("patch_size", 4))
    pretrained = bool(model_cfg.get("pretrained", False))
    drop_path_rate = float(model_cfg.get("drop_path_rate", 0.1))
 
    def build_model():
        # 与训练时 build_deit_student 完全一致的构建,保证结构匹配 checkpoint
        kwargs = dict(
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            patch_size=patch_size,
        )
        try:
            model = timm.create_model(deit_model_name, **kwargs)
        except TypeError:
            kwargs.pop("img_size", None)
            model = timm.create_model(deit_model_name, **kwargs)
        return model
 
    result["Model_Name"] = deit_model_name
    result["Model_Factory"] = build_model
 
    # ---- Per-strategy FT configs(与 CNN 版一致)----
    result["FT_Setups"] = {}
    optimizer_cfg_list = data.get("Optimizers", [])
    scheduler_cfg = data.get("Scheduler", None)
 
    for opt_cfg in optimizer_cfg_list:
        strategy = opt_cfg["strategy"]
        if strategy in result["FT_Setups"]:
            raise ValueError(f"Duplicate fine-tune strategy in YAML: {strategy}")
        result["FT_Setups"][strategy] = {
            "Optimizer_Name": opt_cfg.get("name", "AdamW"),
            "Optimizer_Params": opt_cfg.get("params", {}).copy(),
            "Epochs": opt_cfg.get("Epochs", 100),
            "Scheduler_Name": scheduler_cfg.get("name") if scheduler_cfg else None,
            "Scheduler_Params": scheduler_cfg.get("params", {}).copy() if scheduler_cfg else None,
        }
 
    return result


def process_experiment_prune_setup(data):
    result = {}

    # -----------------------------
    # Dataset setup (now supports YAML-driven transforms)
    # -----------------------------
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size

    # -----------------------------
    # Model factory setup
    # -----------------------------
    model_name = data.get("Model", "")

    def build_model():
        if model_name == "MLP":
            return MNIST_MLP()
        elif model_name == "ResNet-18":
            return ResNet18(num_classes=num_classes)
        elif model_name == "VGG16":
            return ModifiedVGG16(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    result["Model_Name"] = model_name
    result["Model_Factory"] = build_model

    # -----------------------------
    # FT dataset setup
    # -----------------------------
    result["FT_Dataset"] = None
    ft_data_cfg = data.get("FT_Dataset")
    if ft_data_cfg:
        ft_dataset_obj, ft_num_classes, ft_group_size = build_dataset_from_yaml(ft_data_cfg)
        result["FT_Dataset"] = ft_dataset_obj
        result["FT_NumClasses"] = ft_num_classes
        result["FT_GroupSize"] = ft_group_size

    # -----------------------------
    # Optimizer setup
    # -----------------------------
     # -----------------------------
    # Sparsity-specific configs
    # -----------------------------
    result["Prune_Setups"] = {}

    optimizer_cfg_list = data.get("Optimizers", [])
    scheduler_cfg = data.get("Scheduler", None)

    for opt_cfg in optimizer_cfg_list:
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


def process_experiment_prune_setup_deit(data):
    """
    DeiT 版本的 pruning setup。

    与 CNN 版 process_experiment_prune_setup 的唯一区别:
      - Model_Factory 用 timm.create_model 构建 DeiT(解析 dict 形式的 Model 字段),
        构建参数与 process_experiment_ft_setup_deit / build_deit_student 完全一致,
        否则 state_dict 结构对不上 saved_models/vanilla/Transformer_Models 下的 checkpoint。
    其余(Dataset / FT_Dataset / 按 sparsity 索引的 Prune_Setups)与 CNN 版逐字一致。

    旧入口 process_experiment_prune_setup 保留不动,供 CNN(ResNet/VGG)剪枝路径使用。
    """
    result = {}

    # ---- Dataset ----
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size

    # ---- FT Dataset ----
    result["FT_Dataset"] = None
    ft_data_cfg = data.get("FT_Dataset")
    if ft_data_cfg:
        ft_dataset_obj, ft_num_classes, ft_group_size = build_dataset_from_yaml(ft_data_cfg)
        result["FT_Dataset"] = ft_dataset_obj
        result["FT_NumClasses"] = ft_num_classes
        result["FT_GroupSize"] = ft_group_size

    # ---- Model factory(关键差异:构建 DeiT)----
    model_cfg = data.get("Model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "DeiT prune setup 期望 Model 是一个 dict(含 model_name/img_size/patch_size 等),"
            f"但收到: {type(model_cfg)}。请用结构化的 Model 配置。"
        )

    deit_model_name = model_cfg.get("model_name", "deit_tiny_patch16_224")
    img_size = int(model_cfg.get("img_size", 32))
    patch_size = int(model_cfg.get("patch_size", 4))
    pretrained = bool(model_cfg.get("pretrained", False))
    drop_path_rate = float(model_cfg.get("drop_path_rate", 0.1))

    def build_model():
        # 与训练时 build_deit_student 完全一致的构建,保证结构匹配 checkpoint
        kwargs = dict(
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            patch_size=patch_size,
        )
        try:
            model = timm.create_model(deit_model_name, **kwargs)
        except TypeError:
            kwargs.pop("img_size", None)
            model = timm.create_model(deit_model_name, **kwargs)
        return model

    result["Model_Name"] = deit_model_name
    result["Model_Factory"] = build_model

    # ---- Per-sparsity prune configs(与 CNN 版一致)----
    result["Prune_Setups"] = {}
    optimizer_cfg_list = data.get("Optimizers", [])
    scheduler_cfg = data.get("Scheduler", None)

    for opt_cfg in optimizer_cfg_list:
        s_key = round(float(opt_cfg["sparsity"]), 6)

        if s_key in result["Prune_Setups"]:
            raise ValueError(f"Duplicate sparsity setting found in YAML: {s_key}")

        result["Prune_Setups"][s_key] = {
            "Optimizer_Name": opt_cfg.get("name", "AdamW"),
            "Optimizer_Params": opt_cfg.get("params", {}).copy(),
            "Epochs": opt_cfg.get("Epochs", 100),
            "Scheduler_Name": scheduler_cfg.get("name") if scheduler_cfg else None,
            "Scheduler_Params": scheduler_cfg.get("params", {}).copy() if scheduler_cfg else None,
        }

    return result


def sparsity_levels_from_setup(exp_setup, fallback=None):
    """
    剪枝档位以 YAML plan 为唯一事实来源,返回升序的 sparsity 列表。

    为什么需要:main_prune.py / calculate_MI_prune.py 里的模块级 SPARSITY 常量,和 plan
    里 Optimizers 各项的 sparsity 字段,是两处独立维护的。一旦 plan 只写了 {0.2, 0.8}
    而常量还停在 [0.2, 0.4, 0.6, 0.8]:
      - main_prune 跑完 0.2 就在第二档抛
            KeyError: No pruning config found for sparsity=0.4
        后面的档位永远跑不到;
      - calculate_MI_prune 会去找根本不存在的 checkpoint 目录。
    反过来 plan 里加了一档而常量忘了同步,那一档就被静默跳过 —— 更难发现。

    plan 里没有任何 sparsity 配置时退回 fallback(即旧的模块级常量),保持老行为。
    """
    levels = sorted(exp_setup.get("Prune_Setups", {}))
    return levels if levels else list(fallback or [])


def _load_cnn_teacher(teacher_model, state, teacher_name):
    """Load a CNN teacher checkpoint (original create_st_te_model behaviour).

    Lifted out of create_st_te_model unchanged when a DeiT teacher branch was
    added; a ViT must load strictly into its backbone instead, never through
    the order-mapping fallback below.
    """
    try:
        # This works natively for ResNet18
        teacher_model.load_state_dict(state, strict=True)
    except RuntimeError:
        print(f"Strict loading failed for {teacher_name}. Mapping weights by order (VGG compatibility)...")
        # This catches the VGG key mismatch and maps by order
        new_state_dict = teacher_model.state_dict()

        for (old_key, old_val), (new_key, new_val) in zip(state.items(), new_state_dict.items()):
            if old_val.shape == new_val.shape:
                new_state_dict[new_key] = old_val
            else:
                raise ValueError(f"Shape mismatch! Old {old_key} ({old_val.shape}) vs New {new_key} ({new_val.shape})")

        teacher_model.load_state_dict(new_state_dict, strict=True)


def process_experiment_kd_setup(data):
    result = {}
    # -----------------------------
    # 1. Dataset and Basic Config
    # -----------------------------
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["Epochs"] = data.get("Epochs", 200)
    result["GroupSize"] = group_size
    img_size = ds_cfg.get("img_size", 32)

    # -----------------------------
    # 2. Model Templates (Teacher & Student)
    # -----------------------------
    teacher_cfg = data.get("Teacher_Model", {})
    student_cfg = data.get("Student_Model", {})

    def create_st_te_model():
        teacher_name = teacher_cfg.get("teacher_name", "resnet18")
        teacher_ckpt = teacher_cfg.get("teacher_ckpt", "").strip()
        teacher_is_deit = False

        if teacher_name == "ResNet-18":
            teacher_model = ResNet18_dist(num_classes=num_classes)
        elif teacher_name == "ResNet-34":
            # Bigger same-family victim (saved_exp_plan/kd_plan_rn34). Same
            # module naming as the vanilla ResNet34, so strict load works.
            teacher_model = ResNet34_dist(num_classes=num_classes)
        elif teacher_name == "VGG16":
            teacher_model = VGG16_Teacher(num_classes=num_classes)
        elif teacher_name == "VGG19":
            # Bigger same-family victim for a VGG16 student. Its checkpoint is
            # written by main_train_teacher.py through ModifiedVGG19, whose key
            # NAMES differ (features.* vs stage1..5.*) but whose shapes match in
            # order, so the VGG fallback below maps it across.
            teacher_model = VGG19_Teacher(num_classes=num_classes)
        elif teacher_name == "DeiT":
            # Self-distillation teacher: the victim IS the student's own
            # architecture. Wrapped in DeiTKDStudent so the distiller sees the
            # same (logits, feats) contract as for a DeiT student -- the class
            # name says "Student" but it is just the timm adapter, and for a
            # distilled backbone split_deit_outputs returns the averaged
            # (head + head_dist)/2 logits in eval mode, which is the victim's
            # deployed output. Settings come from the Teacher_Model block the
            # same way the student reads Student_Model.
            teacher_model = DeiTKDStudent(
                build_deit_student({"Model": teacher_cfg}, num_classes)
            )
            teacher_is_deit = True
        else:
            raise ValueError(f"Teacher Model architecture not supported: {teacher_name}")

        if teacher_ckpt:
            state = torch.load(teacher_ckpt, map_location="cpu")
            if teacher_is_deit:
                # The victim checkpoint is a bare timm state_dict while the
                # wrapper prefixes every key with "backbone.", so load one
                # level in. Deliberately NO order-mapping fallback: on a ViT
                # that would silently pair unrelated tensors of equal shape.
                teacher_model.backbone.load_state_dict(state, strict=True)
            else:
                _load_cnn_teacher(teacher_model, state, teacher_name)
        else:
            raise ValueError("No teacher model path to load!")

        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        student_name = student_cfg.get("student_name", "resnet10")
        if student_name == "ResNet-10":
            student_model = ResNet10_dist(num_classes=num_classes)
        elif student_name == "VGG8":
            student_model = VGG8_Student(num_classes=num_classes)
        elif student_name == "ResNet-18":
            # Same-architecture student (Born-Again style); matches the
            # ResNet-18 negative pool in saved_models/vanilla/CNN_Models.
            student_model = ResNet18_dist(num_classes=num_classes)
        elif student_name == "VGG16":
            # Cross-family student; VGG16_Teacher is just the deep VGG config,
            # the name refers to depth, not to the teacher role.
            student_model = VGG16_Teacher(num_classes=num_classes)
        elif student_name == "DeiT":
            # Transformer student. build_deit_student reads its settings from a
            # "Model" block, so the Student_Model block is passed through under
            # that key (model_name / img_size / patch_size / drop_path_rate).
            student_model = DeiTKDStudent(
                build_deit_student({"Model": student_cfg}, num_classes)
            )
        else:
            raise ValueError(f"Student Model architecture not supported: {student_name}")
        
        return student_model, teacher_model
    
    result["Create_ST_TE_Model"] = create_st_te_model

    # -----------------------------
    # 3. Loop through ALL Distillation methods
    # -----------------------------
    result["KD_Setups"] = {}

    distill_list = data.get("Distillation", [])
    opt_template = data.get("Optimizer", {})
    sched_template = data.get("Scheduler", {})

    # ---- Ground-truth CE term -------------------------------------------
    # An optional "Augmentation" block lets a KD plan reuse the exact criterion
    # its negative-model counterpart trained with. It is passed through the same
    # build_base_criterion() that process_experiment_setup_deit() uses, so the
    # student and its baseline are bit-identical on the CE side and differ only
    # by the teacher term. Absent the block this is nn.CrossEntropyLoss(), i.e.
    # the previous plain-CE behaviour.
    aug_cfg = data.get("Augmentation", {}) or {}
    if bool(aug_cfg.get("use_mixup", False)):
        raise ValueError(
            "Augmentation.use_mixup is not supported for distillation: "
            "train_one_epoch_kd has no mixup path, and SoftTargetCrossEntropy "
            "would receive hard labels. Set use_mixup: False."
        )
    ce_criterion = build_base_criterion(aug_cfg, use_mixup=False)
    result["CE_Criterion"] = ce_criterion
    print(f"[KD] ground-truth CE term: {ce_criterion.__class__.__name__}"
          f"(label_smoothing={float(aug_cfg.get('label_smoothing', 0.0))})")

    for d_cfg in distill_list:
        method_name = d_cfg.get("name")
        params = d_cfg.get("params", {}).copy()

        if method_name in result["KD_Setups"]:
            raise ValueError(f"Duplicate distillation method found in YAML: {method_name}")
        
        def make_bundle(method_name=method_name, params=params):
            student, teacher = create_st_te_model()

            if method_name == "KD":
                distiller = KD(
                    student,
                    teacher,
                    temperature=params.get("TEMPERATURE", 4),
                    ce_weight=params.get("CE_WEIGHT", 0.1),
                    kd_weight=params.get("KD_WEIGHT", 0.9),
                    ce_criterion=ce_criterion,
                )
            elif method_name == "DKD":
                distiller = DKD(
                    student,
                    teacher,
                    ce_weight=params.get("CE_WEIGHT", 1.0),
                    alpha=params.get("ALPHA", 1.0),
                    beta=params.get("BETA", 8.0),
                    temperature=params.get("T", 4.0),
                    warmup=params.get("WARMUP", 20),
                    ce_criterion=ce_criterion,
                )
            elif method_name == "FitNet":
                distiller = FitNet(
                    student,
                    teacher,
                    ce_weight=params.get("CE_WEIGHT", 1.0),
                    feat_weight=params.get("FEAT_WEIGHT", 100.0),
                    hint_layer=params.get("HINT_LAYER", 2),
                    input_size=(img_size, img_size),
                    ce_criterion=ce_criterion,
                )
            else:
                raise ValueError(f"Unsupported distillation method: {method_name}")

            return distiller
        
        result["KD_Setups"][method_name] = {
            "Builder": make_bundle,
            "Optimizer_Name": opt_template.get("name", "SGD"),
            "Optimizer_Params": opt_template.get("params", {}).copy(),
            "Scheduler_Name": sched_template.get("name") if sched_template else None,
            "Scheduler_Params": sched_template.get("params", {}).copy() if sched_template else None,
        }

    return result


def create_train_subset(train_set, length, seed):
    n = len(train_set)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    idx_a, idx_b = perm[:length], perm[length:]

    ds_a = Subset(train_set, idx_a)
    ds_b = Subset(train_set, idx_b)

    # show sizes
    print(f"Created subsets: ds_a={len(ds_a)}, ds_b={len(ds_b)} (total={n})")

    return ds_a, ds_b


def train_one_epoch(net, trainloader, optimizer, criterion, epoch, device):
    print(f'\nEpoch: {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) # to avoid gradient spikes
        optimizer.step()

        running_loss += loss.item() * len(targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # store for sklearn metrics
        all_preds.append(predicted.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    # concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # accuracy
    avg_loss = running_loss / total
    avg_acc = 100. * correct / total

    # --------------------------
    # sklearn metrics (weighted)
    # --------------------------
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    print(f'\nEpoch: {epoch} ends')
    return {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1
            }


def train_one_epoch_kd(distiller, trainloader, optimizer, epoch, device):
    print(f'\nEpoch: {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    # distiller.train() automatically sets student to train() and teacher to eval()
    distiller.train() 
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # ---------------------------------------------------
        # KD Forward Pass
        # We pass the epoch parameter specifically for DKD warmup
        # ---------------------------------------------------
        logits, losses_dict = distiller(image=inputs, target=targets, epoch=epoch)
        
        # The total loss is the sum of CE loss and Distillation/Feature loss
        loss = sum(losses_dict.values())
        
        loss.backward()
        
        # Gradient clipping: applied to distiller (frozen teacher gradients are None, so it's safe)
        torch.nn.utils.clip_grad_norm_(distiller.parameters(), max_norm=1.0) 
        
        optimizer.step()

        # ---------------------------------------------------
        # Tracking & Metrics
        # ---------------------------------------------------
        # loss.item() is the batch mean, so multiply by batch size for total running loss
        running_loss += loss.item() * targets.size(0) 
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # store for sklearn metrics
        all_preds.append(predicted.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    # concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # average metrics over the epoch
    avg_loss = running_loss / total
    avg_acc = 100. * correct / total

    # --------------------------
    # sklearn metrics (weighted)
    # --------------------------
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    print(f'\nEpoch: {epoch} ends')
    return {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1
            }


def train_one_epoch_mix(net, trainloader, optimizer, criterion, epoch, device):
    print(f'\nEpoch: {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0

    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)   # NOTE: soft labels [B, C] when mixup/cutmix enabled

        optimizer.zero_grad(set_to_none=True)

        outputs = net(inputs)
        loss = criterion(outputs, targets)   # soft_target_cross_entropy
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        running_loss += loss.item() * bs

        # logging accuracy: convert soft target to hard label
        hard_targets = targets.argmax(dim=1)
        _, predicted = outputs.max(1)
        total += bs
        correct += predicted.eq(hard_targets).sum().item()

    avg_loss = running_loss / total
    avg_acc = 100. * correct / total

    print(f'\nEpoch: {epoch} ends')
    return {
        "train_loss": avg_loss,
        "train_acc": avg_acc,
        # train precision/recall/f1 not meaningful with mixup/cutmix:
        "train_precision": float("nan"),
        "train_recall": float("nan"),
        "train_f1": float("nan"),
    }


def evaluate1(net, test_loader, criterion, device): # Used to calculate the test performance
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            test_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

            # store for sklearn metrics
            all_preds.append(predicted.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    # concat all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = test_loss / total
    avg_acc = 100. * correct / total

    # weighted-averaged metrics over the whole test set
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return {
            "test_loss": avg_loss,
            "test_acc": avg_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
            }


def evaluate2(net, data_loader, device, num_intervals, verbose): # Used tp calculate the MI metric
    
    if hasattr(net, "fc"):
        num_classes = net.fc.out_features
    elif hasattr(net, "classifier"):
        # assume last Linear layer in classifier
        last_linear = [m for m in net.classifier.modules() if isinstance(m, nn.Linear)][-1]
        num_classes = last_linear.out_features
    elif hasattr(net, "num_classes"): # this is made for the timm ViT model class
        num_classes = net.num_classes
    else:
        raise ValueError("Cannot automatically infer num_classes from model.")

    #print(f"Out features are {num_classes}.")

    layer_T_array, label_matrix = None, None

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0] # this is in case of the knowledge distillation case !
 
            layer_T_array, label_matrix = calculate_MI_input_gpu(layer_T_array, label_matrix
                                                            ,outputs, targets, num_classes)

    #effective_mask = min(mask, layer_T_array.shape[0]) # Up to now, we try to keep in and out samples size the same.
    #print(effective_mask)
    # ===== MI calculation =====

    #print("MI calculation begins:")
    '''value_xt, value_ty = MI_cal_gpu_v1(layer_T_array.to(device=device, dtype=torch.float32), 
                                       label_matrix.to(device=device, dtype=torch.float32),
                                       num_intervals=50)'''
    
    value_xt, value_ty = MI_cal_gpu_v2(layer_T_array.to(device=device, dtype=torch.float32), 
                                       label_matrix.to(device=device, dtype=torch.float32),
                                       num_intervals=num_intervals, verbose=verbose)
    #value_xt, value_ty = MI_cal_v2(label_matrix.numpy(), layer_T_array.numpy(), effective_mask)
    #print("MI calculation ends.")

    return value_xt, value_ty
 

# collect layer_T_array for MI analysis
def calculate_MI_input(layer_T_array, label_matrix, outputs, targets, num_classes):
    outputs_cpu = outputs.detach().cpu()
    targets_cpu = targets.detach().cpu()


    if layer_T_array is None:
        layer_T_array = outputs.detach().cpu()
    else:
        layer_T_array = torch.cat((layer_T_array, outputs_cpu), dim=0)

    # accumulate one-hot labels
    batch_onehot = F.one_hot(targets_cpu, num_classes=num_classes).float().cpu()

    if label_matrix is None:
        label_matrix = batch_onehot
    else:
        label_matrix = torch.cat((label_matrix, batch_onehot), dim=0)
    
    return layer_T_array, label_matrix


def calculate_MI_input_gpu(layer_T_array, label_matrix, outputs, targets, num_classes):
    """Accumulate one batch of model outputs and one-hot labels."""
    
    outputs_detached = outputs.detach()
    targets_detached = targets.detach()
 
    if layer_T_array is None:
        layer_T_array = outputs_detached
    else:
        layer_T_array = torch.cat((layer_T_array, outputs_detached), dim=0)
 
    batch_onehot = F.one_hot(targets_detached, num_classes=num_classes).float()
    if label_matrix is None:
        label_matrix = batch_onehot
    else:
        label_matrix = torch.cat((label_matrix, batch_onehot), dim=0)
 
    return layer_T_array, label_matrix


def evaluate(scenario_name, net, in_sample_loader, out_sample_loader, criterion, epoch, 
             best_acc, device): # MI_flag means if needed to calculate the MI
    
    if hasattr(net, "fc"):
        num_classes = net.fc.out_features
    elif hasattr(net, "classifier"):
        # assume last Linear layer in classifier
        last_linear = [m for m in net.classifier.modules() if isinstance(m, nn.Linear)][-1]
        num_classes = last_linear.out_features
    else:
        num_classes = 10
        #raise ValueError("Cannot automatically infer num_classes from model.")

    print(f"Out features are {num_classes}.")

    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    layer_T_array_in, layer_T_array_out = None, None
    label_matrix_in, label_matrix_out = None, None

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(out_sample_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            test_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

            if (num_classes) == 100:
                if (epoch % 5 == 0) or (epoch == 199):
                    layer_T_array_out, label_matrix_out = calculate_MI_input(layer_T_array_out, label_matrix_out
                                                                     ,outputs, targets, num_classes)
            else: 
                layer_T_array_out, label_matrix_out = calculate_MI_input(layer_T_array_out, label_matrix_out
                                                                     ,outputs, targets, num_classes)

        for batch_idx, (inputs, targets) in enumerate(in_sample_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            if (num_classes) == 100:
                if (epoch % 5 == 0) or (epoch == 199):
                    layer_T_array_in, label_matrix_in = calculate_MI_input(layer_T_array_in, label_matrix_in
                                                                     ,outputs, targets, num_classes)
                    effective_mask = min(mask, layer_T_array_in.shape[0]) # Up to now, we try to keep in and out samples size the same.
            else: 
                layer_T_array_in, label_matrix_in = calculate_MI_input(layer_T_array_in, label_matrix_in
                                                                     ,outputs, targets, num_classes)
                effective_mask = min(mask, layer_T_array_in.shape[0]) # Up to now, we try to keep in and out samples size the same.

    avg_loss = test_loss / total
    avg_acc = 100. * correct / total

    # ===== MI calculation =====

    if (num_classes) == 100:
        if (epoch % 5 == 0) or (epoch == 199):
            print("MI calculation begins:")
            value_xt_out, value_ty_out = MI_cal_v2(label_matrix_out.numpy(), layer_T_array_out.numpy(), effective_mask)
            value_xt_in, value_ty_in = MI_cal_v2(label_matrix_in.numpy(), layer_T_array_in.numpy(), effective_mask)
            print("MI calculation ends.")
        else:
            value_xt_out, value_ty_out = -1, -1
            value_xt_in, value_ty_in = -1, -1
    else: 
        print("MI calculation begins:")
        value_xt_out, value_ty_out = MI_cal_v2(label_matrix_out.numpy(), layer_T_array_out.numpy(), effective_mask)
        value_xt_in, value_ty_in = MI_cal_v2(label_matrix_in.numpy(), layer_T_array_in.numpy(), effective_mask)
        print("MI calculation ends.")

    # Save best model
    if avg_acc > best_acc:
        print(f"Saving model (acc improved from {best_acc:.2f}% → {avg_acc:.2f}%)")
        
        os.makedirs(f'./saved_models/{scenario_name}', exist_ok=True)
        torch.save(net.state_dict(), f'./saved_models/{scenario_name}/best_{epoch}.pth')
        best_acc = avg_acc

    return {
            "test_loss": avg_loss,
            "test_acc": avg_acc,
            "best_acc": best_acc,
            "value_xt_in": value_xt_in,
            "value_ty_in": value_ty_in,
            "value_xt_out": value_xt_out,
            "value_ty_out": value_ty_out
            }


def setup_finetune(model, strategy, num_classes=None, device='cuda',
                   deit_finetune_norm_in_ll=False):
    """
    为三种 fine-tune 策略配置模型（参数的 requires_grad）:
      - FT-LL: 只 fine-tune 最后一层（分类头），其余全冻结
      - FT-AL: fine-tune 所有层
      - RT-AL: 重置分类头，然后 fine-tune 所有层

    自动识别 ResNet（.fc）、VGG（.classifier）、DeiT（.head [+ .head_dist]）。

    DeiT distilled（deit_*_distilled_*）有两个分类头:head 学 ground truth，
    head_dist 学 teacher。eval 时 timm 恒定返回 (head + head_dist)/2，两者共同
    构成分类器，因此三种策略都把它们当作一个整体:要解冻一起解冻，要重置一起重置。
    只动其中一个等于只处理了半个分类器 —— FT-LL 会因另一半被钉死而受限（head_dist
    的输出退化为逐样本常量偏置，线性的 head 无法逐样本抵消它），RT-AL 会让
    head_dist 保留原模型知识并继续占推理 logits 的 50%。

    deit_finetune_norm_in_ll: 仅对 DeiT 的 FT-LL 生效。
        False -> FT-LL 只解冻分类头参数（严格对齐 CNN，推荐）
        True  -> FT-LL 额外解冻 final LayerNorm（model.norm）的参数
    """
    # --- 识别分类头位置 ---
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        # ResNet 风格
        head_attr = "fc"
        head_modules = [model.fc]
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # VGG 风格:分类头是 classifier 的最后一个元素
        head_attr = "classifier"
        head_modules = [model.classifier[-1]]
        if not isinstance(head_modules[0], nn.Linear):
            raise ValueError("Last element in model.classifier is not a Linear layer.")
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        # DeiT / ViT:plain 只有 head；distilled 还有 head_dist，两者同属“最后一层”
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
        """安全替换分类头；DeiT distilled 时两个头一并重置（各自独立初始化）。"""
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
        # 先冻结全部
        for p in model.parameters():
            p.requires_grad = False

        # 类别数变了才替换头
        if num_classes != out_features:
            head_modules = replace_heads()

        # 只解冻分类头（distilled 时包含 head_dist）
        for m in head_modules:
            for p in m.parameters():
                p.requires_grad = True

        # DeiT 特有（可选）:额外解冻 final LayerNorm 的参数
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
    """
    把 backbone 里所有"会因 train 模式改变行为"的子模块设为 eval。
 
    为什么需要:net.train() 会把整个模型(含被冻结的 backbone)切到 train 模式。
    对被冻结的 backbone,我们要确保它前向行为完全确定、不漂移:
      - CNN  : BatchNorm 在 train 模式会更新 running_mean/var(即使 weight 冻结),
               必须设 eval 防止统计漂移;Dropout 也设 eval。
      - DeiT : LayerNorm 无 running 统计(train/eval 行为相同),设 eval 更明确;
               DropPath(stochastic depth)在 train 模式会随机跳层,污染冻结
               backbone 的前向,必须设 eval 关掉。
 
    用于 FT-LL。FT-AL / RT-AL 不调用此函数(它们要训练整个网络)。
    """
    for m in model.modules():
        # CNN 的 BatchNorm 和通用 Dropout
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
            m.eval()
        # transformer 的 LayerNorm
        elif isinstance(m, nn.LayerNorm):
            m.eval()
        # timm 的 DropPath / stochastic depth(用类名判断,跨 timm 版本最稳)
        elif m.__class__.__name__ in ("DropPath", "StochasticDepth"):
            m.eval()


def ft_one_epoch(net, trainloader, optimizer, criterion, epoch, device, strategy,
                 freeze_backbone_norm=True):
    """
    单 epoch fine-tune 训练循环。

    freeze_backbone_norm: 仅对 FT-LL 生效。
        True  -> 调用 set_backbone_eval_norm_dropout，冻结 backbone 运行行为（严格 FT-LL）
        False -> 不冻结（允许 backbone 的 norm/dropout/droppath 在 train 模式活跃）
    """
    print(f'\nEpoch: {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    net.train()

    # FT-LL:把被冻结的 backbone 的 norm/dropout/droppath 设回 eval
    # 否则 net.train() 会覆盖它们，导致冻结的 backbone 行为漂移/带随机性
    if strategy == "FT-LL" and freeze_backbone_norm:
        set_backbone_eval_norm_dropout(net)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        # DeiT distilled 在 distilled_training=True 的 train 模式下返回 (cls, dist)。
        # 按推理口径融合两个头，而不是丢弃 dist:eval 时 timm 恒定输出 (cls+dist)/2，
        # 只取 cls 会让训练去优化一个推理时并不使用的函数。
        # 不能复用 logits_for_inference —— 它带 @torch.no_grad()，会切断梯度。
        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            outputs = (outputs[0] + outputs[1]) / 2.0

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(targets)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.append(predicted.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = running_loss / total
    avg_acc = 100. * correct / total

    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    print(f'\nEpoch: {epoch} ends')
    return {
        "train_loss": avg_loss,
        "train_acc": avg_acc,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": f1,
    }


def sanity_check_finetune(model, ckpt_path, strategy, device='cuda',
                          deit_finetune_norm_in_ll=False):
    """
    对比 fine-tune 后的模型与原始 checkpoint,验证各策略行为是否符合预期。
    支持 ResNet(.fc)、VGG(.classifier)、DeiT(.head / .head_dist)。
 
    与 setup_finetune 对齐的开关:
      deit_finetune_norm_in_ll: 仅对 DeiT 的 FT-LL 生效。
        False -> FT-LL 时 final LayerNorm(model.norm)应保持冻结(算 backbone,不应变)
        True  -> FT-LL 时 final LayerNorm 被解冻训练(应变,不计入 backbone 违规)
      必须与调用 setup_finetune 时传的同名参数一致,否则会误判。
 
      （DeiT distilled 的 head_dist 恒被归为分类头，与 setup_finetune 的处理保持一致，不再可配。）
    """
    # ---- checkpoint 加载(加固:weights_only + 解包)----
    original_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(original_state, dict) and "model" in original_state:
        original_state = original_state["model"]
    elif isinstance(original_state, dict) and "state_dict" in original_state:
        original_state = original_state["state_dict"]
 
    current_state = model.state_dict()
 
    # ---- 逐参数比较(用 get 防 KeyError)----
    changed = []
    unchanged = []
    missing = []   # 当前模型有、但 checkpoint 没有的 key(如 RT-AL 重置后的新 head)
    for name, param in current_state.items():
        if name not in original_state:
            missing.append(name)
            changed.append(name)   # checkpoint 里没有 -> 视为变化
            continue
        original_param = original_state[name]
        if param.shape != original_param.shape:
            changed.append(name)   # 形状不同(如 head 类别数变了)-> 视为变化
        elif torch.equal(param.to(device), original_param.to(device)):
            unchanged.append(name)
        else:
            changed.append(name)
 
    # ---- 识别分类头参数名 ----
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        head_names = {n for n in current_state if n.startswith("fc.")}
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last_idx = len(model.classifier) - 1
        head_names = {n for n in current_state if n.startswith(f"classifier.{last_idx}.")}
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        head_names = {n for n in current_state if n.startswith("head.")}
        if hasattr(model, "head_dist") and isinstance(model.head_dist, nn.Linear):
            head_names |= {n for n in current_state if n.startswith("head_dist.")}
    else:
        raise ValueError("Model does not have a supported head (.fc / .classifier[-1] / .head).")
 
    # ---- DeiT FT-LL:决定 final norm 算不算"允许变化" ----
    # 当 deit_finetune_norm_in_ll=True 且策略是 FT-LL 时,model.norm 被有意解冻,
    # 它的变化是预期的,不应算作 backbone 违规。把它从 backbone 检查中豁免。
    is_deit = hasattr(model, "head") and isinstance(model.head, nn.Linear)
    norm_names = {n for n in current_state if n.startswith("norm.")} if is_deit else set()
    ll_norm_exempt = (strategy == "FT-LL" and deit_finetune_norm_in_ll and is_deit)
 
    # backbone = 全部 - head - (FT-LL 豁免的 norm)
    head_or_exempt = set(head_names)
    if ll_norm_exempt:
        head_or_exempt |= norm_names
    backbone_names = set(current_state.keys()) - head_or_exempt
 
    # ---- 打印概况 ----
    print(f"\n{'='*50}")
    print(f"  Sanity Check: {strategy}"
          + (f"  (DeiT, norm_in_ll={deit_finetune_norm_in_ll})" if is_deit else ""))
    print(f"{'='*50}")
    print(f"  Total parameters:   {len(current_state)}")
    print(f"  Changed:            {len(changed)}")
    print(f"  Unchanged:          {len(unchanged)}")
    if missing:
        print(f"  (Not in checkpoint: {len(missing)}, e.g. reinitialized head)")
 
    passed = True
 
    if strategy == "FT-LL":
        backbone_changed = [n for n in changed if n in backbone_names]
        head_unchanged = [n for n in unchanged if n in head_names]
 
        if backbone_changed:
            print(f"\n  [FAIL] Backbone layers changed (should be frozen):")
            for n in backbone_changed:
                print(f"         - {n}")
            passed = False
 
        if head_unchanged:
            print(f"\n  [FAIL] Head layers unchanged (should be updated):")
            for n in head_unchanged:
                print(f"         - {n}")
            passed = False
 
        # DeiT 额外提示:确认 final norm 行为符合开关
        if is_deit and ll_norm_exempt:
            norm_changed = [n for n in changed if n in norm_names]
            if norm_changed:
                print(f"\n  [INFO] final norm changed as expected "
                      f"(deit_finetune_norm_in_ll=True): {norm_changed}")
            else:
                print(f"\n  [WARN] deit_finetune_norm_in_ll=True 但 final norm 未变化,"
                      f"检查是否真的解冻了 norm。")
 
    elif strategy == "FT-AL":
        if not any(n in backbone_names for n in changed):
            print(f"\n  [FAIL] No backbone layers changed (all should be trainable)")
            passed = False
        if not any(n in head_names for n in changed):
            print(f"\n  [FAIL] Head layers unchanged (should be updated)")
            passed = False
 
    elif strategy == "RT-AL":
        if not any(n in head_names for n in changed):
            print(f"\n  [FAIL] Head layers unchanged (should be reinitialized and updated)")
            passed = False
        if not any(n in backbone_names for n in changed):
            print(f"\n  [FAIL] No backbone layers changed (all should be trainable)")
            passed = False
 
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
 
    if passed:
        print(f"\n  [PASS] All layers behaved as expected for {strategy}")
 
    print(f"\n  Changed layers:")
    for n in changed:
        if n in head_names:
            tag = "(head)"
        elif ll_norm_exempt and n in norm_names:
            tag = "(final-norm, exempt)"
        else:
            tag = "(backbone)"
        print(f"    - {n} {tag}")
 
    print(f"{'='*50}\n")
    return passed



def prune_model_global(model, amount, exclude_patterns=None):
    """
    Global unstructured pruning (magnitude-based).
    Removes the `amount` fraction of the smallest-magnitude weights globally.

    exclude_patterns: optional list of name substrings kept OUT of the pruned
        set entirely, e.g. ["features.14", "features.15", "features.16"] to
        spare VGG16's last conv stage. `amount` is then a fraction of what is
        left, so the whole-model sparsity ends up lower than `amount`; measure
        it with check_pruned_weights(model, exclude_patterns=<the same list>)
        to get a number comparable to `amount`. Omitting it (the default)
        reproduces the original behaviour exactly: every Conv2d/Linear except
        a DeiT patch_embed Conv2d.
    """
    if exclude_patterns is None:
        exclude_patterns = []

    # Gather all parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and "patch_embed" in name:
            continue  # skip patch embedding
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

    # Optionally remove the pruning reparameterization (make zeros permanent)
    '''for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')'''

    return model


def check_pruned_weights(model, exclude_patterns=None):
    """
    Inspect the sparsity (percentage of zero weights) in each pruned layer.
    Also prints global sparsity across the entire model.
    
    exclude_patterns: list of name substrings to skip (e.g., ["patch_embed"])
    """
    if exclude_patterns is None:
        exclude_patterns = []

    total_weights = 0
    total_zero = 0

    print("\n=== PRUNING CHECKER ===")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if any(pat in name for pat in exclude_patterns):
                continue
            if hasattr(module, "weight"):
                w = module.weight.data.cpu().numpy()
                num_total = w.size
                num_zero = (w == 0).sum()
                layer_sparsity = num_zero / num_total * 100

                total_weights += num_total
                total_zero += num_zero

    global_sparsity = total_zero / total_weights * 100
    print(f"\nGLOBAL SPARSITY: {global_sparsity:.2f}%\n")
    return global_sparsity


def setup_ft_optimizer(model_name, ):
    pass

###############################################################################
# MI Calculation
##############################################################################

NUM_INTERVALS=50
#NUM_LABEL=10
mask=40000

def MI_cal_v2(label_matrix, layer_T, NUM_TEST_MASK):
    start_time = time.time()
    NUM_LABEL = label_matrix.shape[1]

    MI_XT=0
    MI_TY=0
    # This part is to transfrom logits to probabilities through softmax.
    layer_T = np.exp(layer_T - np.max(layer_T,axis=1,keepdims=True)) #numerical stabilization: prevent potential overflow issues for exponentiation.
    layer_T /= np.sum( layer_T,axis=1,keepdims=True)
    
    layer_T = Discretize_v2(layer_T)
    XT_matrix = np.zeros((NUM_TEST_MASK,NUM_TEST_MASK)) # 1000*1000
    Non_repeat=[]
    mark_list=[]
    for i in range(NUM_TEST_MASK):
        pre_mark_size = len(mark_list)
        if i==0:
            Non_repeat.append(i)
            mark_list.append(i)
            XT_matrix[i,i]=1
        else:
            for j in range(len(Non_repeat)):
                if (layer_T[i] ==layer_T[ Non_repeat[j] ]).all():
                    mark_list.append(Non_repeat[j])
                    XT_matrix[i,Non_repeat[j]]=1
                    break
        if pre_mark_size == len(mark_list):
            Non_repeat.append(Non_repeat[-1]+1)
            mark_list.append(Non_repeat[-1])
            XT_matrix[i,Non_repeat[-1]]=1
    
    XT_matrix = np.delete(XT_matrix,range(len(Non_repeat),NUM_TEST_MASK),axis=1)				
    XT_matrix = XT_matrix/NUM_TEST_MASK
    P_X_for_layer_T = np.sum(XT_matrix,axis=1)
    P_layer_T_for_X= np.sum(XT_matrix,axis=0)
    for i in range(XT_matrix.shape[0]):
        for j in range(XT_matrix.shape[1]):
            if XT_matrix[i,j]==0:
                pass
            else:
                MI_XT+=XT_matrix[i,j]*np.log2(XT_matrix[i,j]/(P_X_for_layer_T[i]*P_layer_T_for_X[j]))

    TY_matrix = np.zeros((len(Non_repeat),NUM_LABEL))
    mark_list = np.array(mark_list)
    for i in range(len(Non_repeat)):
        TY_matrix[i,:] = np.sum(label_matrix[  np.where(mark_list==i)[0]  , : ] ,axis=0 )
    TY_matrix = TY_matrix/NUM_TEST_MASK
    P_layer_T_for_Y = np.sum(TY_matrix,axis=1)
    P_Y_for_layer_T = np.sum(TY_matrix,axis=0)
    for i in range(TY_matrix.shape[0]):
        for j in range(TY_matrix.shape[1]):
            if TY_matrix[i,j]==0:
                pass
            else:
                MI_TY+=TY_matrix[i,j]*np.log2(TY_matrix[i,j]/(P_layer_T_for_Y[i]*P_Y_for_layer_T[j]))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(elapsed_time)
    return MI_XT,MI_TY



#将连续数据转换为离散数据
def Discretize_v2(layer_T):	
    labels = np.arange(NUM_INTERVALS) # [0, 1, 2, 3 ..., 49]
    bins = np.arange(NUM_INTERVALS+1) # [0, 1, 2, 3 ..., 49, 50]
    bins = bins/float(NUM_INTERVALS)  # [0, 0.02, 0.04, ..., 1.0]
    
    for i in range(layer_T.shape[1]):
        temp = pd.cut(layer_T[:,i],bins,labels=labels)
        layer_T[:,i] = np.array(temp)
    return layer_T

#---------------------GPU calculation version---------------------------------
def MI_formula_cal(matrix, p1, p2): #p1, p2 represents marginals
    mask = matrix > 0
    denom = p1[:, None] * p2[None, :]
    ratio = matrix / denom
    log_ratio = torch.log2(ratio)
    MI = (matrix * log_ratio)[mask].sum()
    return MI

def find_match_indices(layer_T, non_repeat, i):
    # layer_T: [N, C]
    nr_idx = torch.tensor(non_repeat, device=layer_T.device, dtype=torch.long)
    # Compare layer_T[i] to all non_repeat rows at once
    eq = (layer_T[i].unsqueeze(0) == layer_T[nr_idx])   # [len(nonrepeat), C]
    return eq.all(dim=1)  # boolean vector length = len(nonrepeat)

    
def MI_cal_gpu_v1(layer_T, label_matrix, num_intervals=50):
    start_time = time.time()
    device = layer_T.device
    layer_T = torch.softmax(layer_T, dim=1)

    bins = torch.linspace(0, 1, num_intervals + 1, device=device, dtype=torch.float32)
    layer_T_discrete = torch.bucketize(layer_T, bins, right=True) - 1
    layer_T_discrete = layer_T_discrete.clamp(0, num_intervals - 1)

    layer_T = layer_T_discrete.contiguous()
    N, C = layer_T.shape

    XT_matrix = torch.zeros((N, N), device=device, dtype=torch.float32)
    non_repeat, mark_list = [], []

    for i in range(N):
        if i == 0:
            # First row always unique
            non_repeat.append(i)
            mark_list.append(i)
            XT_matrix[i, i] = 1
            continue
    
        matches = find_match_indices(layer_T, non_repeat, i)
    
        if matches.any():
            j = matches.nonzero()[0].item()  # First matched index in non_repeat
            match_idx = non_repeat[j]
            mark_list.append(match_idx)
            XT_matrix[i, match_idx] = 1
        else:
            # No match found → new unique pattern
            new_idx = non_repeat[-1] + 1
            non_repeat.append(new_idx)
            mark_list.append(new_idx)
            XT_matrix[i, new_idx] = 1

    N_unique = len(non_repeat)
    XT_matrix = XT_matrix[:, :N_unique]
    XT_matrix = XT_matrix / N

    P_X_for_layer_T = XT_matrix.sum(dim=1)
    P_layer_T_for_X = XT_matrix.sum(dim=0)

    I_X_T = MI_formula_cal(XT_matrix, P_X_for_layer_T, P_layer_T_for_X)

    mark_t = torch.tensor(mark_list, device=device, dtype=torch.long)
    K = len(non_repeat)
    
    TY_matrix = torch.zeros((K, label_matrix.shape[1]), 
                            device=device, dtype=torch.float32)
    
    TY_matrix.index_add_(0, mark_t, label_matrix)
    
    TY_matrix = TY_matrix / N

    P_layer_T_for_Y = TY_matrix.sum(dim=1)
    P_Y_for_layer_T = TY_matrix.sum(dim=0)

    I_T_Y = MI_formula_cal(TY_matrix, P_layer_T_for_Y, P_Y_for_layer_T)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

    return I_X_T.item(), I_T_Y.item()


def MI_cal_gpu_v2(layer_T, label_matrix, num_intervals=50, verbose=False):
    """Compute I(X; T_tilde) and I(T_tilde; Y) via binning.
 
    Args:
        layer_T:      (N, C) logits tensor on GPU. Will be softmaxed internally.
        label_matrix: (N, K_y) one-hot label tensor on GPU.
        num_intervals: number of equal-width bins per output dimension (default 50).
 
    Returns:
        (I_X_T, I_T_Y): two Python floats, in bits.
 
    Notes:
        - Under the empirical input distribution (each input is unique with
          probability 1/N) and a deterministic mapping X -> T, we have
          H(T_tilde | X) = 0, so I(X; T_tilde) = H(T_tilde). This function
          uses that identity directly to avoid building an N x N matrix.
        - Two samples are in the same equivalence class iff their bin-pattern
          vectors are componentwise equal. Grouping is done via torch.unique
          along dim=0, which guarantees this partition exactly.
    """
    start_time = time.time()
    device = layer_T.device
    N = layer_T.shape[0]
 
    # Step 1: softmax + per-coordinate equal-width binning on [0, 1].
    # Identical to the original implementation.
    T_soft = torch.softmax(layer_T, dim=1)
    bins = torch.linspace(0, 1, num_intervals + 1, device=device, dtype=torch.float32)
    T_discrete = torch.bucketize(T_soft, bins, right=True) - 1
    T_discrete = T_discrete.clamp(0, num_intervals - 1).contiguous()
 
    # Step 2: group samples by their bin-pattern vector.
    # `inverse_idx[i]` is the unique-pattern ID of sample i.
    # This single call replaces the entire O(N^2) Python loop and fixes the
    # off-by-one partition bug in the original implementation.
    unique_T, inverse_idx = torch.unique(T_discrete, dim=0, return_inverse=True)
    K_unique = unique_T.shape[0]
 
    # Step 3: I(X; T_tilde) = H(T_tilde) under the empirical input distribution.
    # Compute the bin-pattern marginal by counting samples per group, then take
    # the entropy in bits. This is mathematically equivalent to running
    # MI_formula_cal on the (N x U) XT_matrix the original built, but avoids
    # the O(N^2) allocation entirely.
    T_counts = torch.zeros(K_unique, device=device, dtype=torch.float32)
    T_counts.index_add_(
        0, inverse_idx, torch.ones(N, device=device, dtype=torch.float32)
    )
    p_T = T_counts / N
    mask_T = p_T > 0
    I_X_T = -(p_T[mask_T] * torch.log2(p_T[mask_T])).sum()
 
    # Step 4: build the (T_tilde, Y) joint-count matrix.
    # For each sample i, accumulate its one-hot label into row inverse_idx[i].
    # Same construction as the original (identical to its TY_matrix), just
    # built from a correct partition.
    K_y = label_matrix.shape[1]
    TY_matrix_counts = torch.zeros((K_unique, K_y), device=device, dtype=torch.float32)
    TY_matrix_counts.index_add_(0, inverse_idx, label_matrix)
    TY_matrix = TY_matrix_counts / N
 
    # Step 5: I(T_tilde; Y) via the original MI formula.
    # We use the SAME MI_formula_cal as the original here so the only point of
    # divergence between this and the original is the partition (which we've
    # corrected). All MI-formula-level rounding behavior is preserved.
    P_T_marg = TY_matrix.sum(dim=1)
    P_Y_marg = TY_matrix.sum(dim=0)
    I_T_Y = MI_formula_cal(TY_matrix, P_T_marg, P_Y_marg)

    if not verbose:
        return I_X_T.item(), I_T_Y.item()
    
    debug_data = {
        # Continuous prediction vectors, shape (N, K). Row i is sample i.
        "T_soft": T_soft.detach().cpu().numpy(),

        # Discretized prediction vectors, shape (N, K). Row i is sample i.
        # T_discrete[i] is the K-tuple of bin indices for sample i. Two
        # samples have T_discrete[i] == T_discrete[j] iff they fall into
        # the same equivalence class.
        "T_discrete": T_discrete.detach().cpu().numpy(),

        # The unique bin patterns observed across all N samples, shape (U, K).
        # This is the "menu" of pattern types.
        "unique_T": unique_T.detach().cpu().numpy(),

        # Mapping sample index -> unique-pattern row index, shape (N,).
        # inverse_idx[i] = k means "sample i corresponds to unique_T[k]".
        # Equivalently: T_discrete[i] == unique_T[inverse_idx[i]].
        "inverse_idx": inverse_idx.detach().cpu().numpy(),

        # Sample count per unique pattern, shape (U,).
        # pattern_counts[k] = how many samples fall in pattern k.
        # This is the (un-normalized) empirical distribution of T_tilde.
        "pattern_counts": T_counts.detach().cpu().numpy().astype("int64"),

        # Joint count matrix, shape (U, K_y).
        # label_counts[k, y] = how many samples are in pattern k AND have label y.
        # This is the (un-normalized) empirical joint distribution of (T_tilde, Y).
        "label_counts": TY_matrix_counts.detach().cpu().numpy().astype("int64"),

        # Metadata so downstream code can reconstruct context without
        # needing to know how the data was produced.
        "N": N,
        "K": layer_T.shape[1],
        "K_y": K_y,
        "U": K_unique,
        "num_intervals": num_intervals,
        "I_X_T": I_X_T.item(),
        "I_T_Y": I_T_Y.item(),
    }

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"MI calculation time is: {elapsed_time} !")
 
    return I_X_T.item(), I_T_Y.item()


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

    # Fast path: use dataset.targets if the dataset exposes it (CIFAR10, CIFAR100,
    # TinyImageNet, and most torchvision datasets do). This avoids triggering
    # __getitem__ for each sample, which would load the image and run the full
    # transform pipeline -- wasteful when we only need the labels.
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


def create_or_load_subset_from_group(
    dataset,
    group_A,
    save_dir,
    subset_size=10000,
    num_classes=10,
    seed=42,
    force_rebuild=False
):
    save_path = Path(save_dir + f"/group_A_subset_{subset_size}_from_{len(group_A)}_seed{seed}.npy")

    if save_path.exists() and not force_rebuild:
        print(f"[INFO] Loading subset from {save_path}")
        return load_indices(save_path)

    print("[INFO] Creating new balanced subset from Group A")

    rng = np.random.default_rng(seed)
    n_per_class = subset_size // num_classes

    # Build class-to-indices mapping using only group_A indices
    class_to_indices = {}
    for idx in group_A:
        _, y = dataset[idx]
        class_to_indices.setdefault(y, []).append(idx)

    subset = []
    for c in range(num_classes):
        indices = np.array(class_to_indices[c])
        rng.shuffle(indices)
        subset.extend(indices[:n_per_class])

    rng.shuffle(subset)
    save_indices(subset, save_path)
    return subset


def create_nested_balanced_subsets(
    dataset, group_A, save_dir, subset_sizes,
    num_classes=10, seed=42, force_rebuild=False,
):
    """Create nested class-balanced subsets of group_A at the requested sizes.
 
    Properties guaranteed:
      - All subsets are class-balanced (each class contributes size/num_classes).
      - subset[s1] is a subset of subset[s2] whenever s1 < s2 (nesting).
      - Reproducible: same (group_A, seed, subset_sizes) -> same subsets.
 
    Returns:
        dict mapping subset_size -> np.ndarray of indices.
    """
    save_path = Path(save_dir) / f"nested_subsets_seed{seed}.npz"
    if save_path.exists() and not force_rebuild:
        print(f"[INFO] Loading nested subsets from {save_path}")
        data = np.load(save_path)
        return {int(k.split("_")[1]): data[k] for k in data.files}
 
    print(f"[INFO] Building nested subsets for sizes {subset_sizes}")
    rng = np.random.default_rng(seed)
 
    # Group group_A indices by class label, then shuffle each class ONCE.
    # This single shuffle determines all subset memberships across sizes.
    class_indices = {c: [] for c in range(num_classes)}
    for idx in group_A:
        _, y = dataset[idx]
        class_indices[int(y)].append(idx)
    for c in range(num_classes):
        arr = np.array(class_indices[c])
        rng.shuffle(arr)
        class_indices[c] = arr
 
    # Validate that requested sizes are achievable
    max_per_class = min(len(v) for v in class_indices.values())
    for s in subset_sizes:
        if s % num_classes != 0:
            raise ValueError(
                f"subset_size {s} not divisible by num_classes {num_classes}"
            )
        n_per_class = s // num_classes
        if n_per_class > max_per_class:
            raise ValueError(
                f"subset_size {s} requires {n_per_class} samples per class, "
                f"but smallest class has only {max_per_class} in group_A"
            )
 
    # Build each subset: take the first n_per_class from each class.
    # Nesting follows automatically because the per-class arrays are fixed.
    subsets = {}
    for s in subset_sizes:
        n_per_class = s // num_classes
        idx = np.concatenate([
            class_indices[c][:n_per_class] for c in range(num_classes)
        ])
        # Optional final shuffle for loader iteration order; doesn't change set
        order_rng = np.random.default_rng(seed + s)
        order_rng.shuffle(idx)
        subsets[s] = idx
 
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **{f"size_{s}": subsets[s] for s in subset_sizes})
    print(f"[INFO] Saved nested subsets to {save_path}")
    return subsets


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

    # Fast path: read labels without triggering image IO + transforms.
    # Same convention as create_or_load_group_A.
    if hasattr(dataset, "targets"):
        all_labels = np.asarray(dataset.targets)
    elif hasattr(dataset, "Y"):
        all_labels = np.asarray(dataset.Y)
    else:
        all_labels = np.asarray([dataset[i][1] for i in range(len(dataset))])

    # Build a boolean mask: True at positions belonging to group_A.
    # This replaces the dict-of-lists + set() approach with O(N) numpy ops.
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


def create_or_load_group_B_superset(
    dataset,
    group_A,
    save_path,
    groupB_size=30000,
    num_classes=10,
    seed=42,
    force_rebuild=False,
):
    """
    Group B is a superset of Group A with a larger, class-balanced size.

    Example:
      |A|=25000 (2500/class), groupB_size=30000 (3000/class)
      => B = A + 500/class additional samples, drawn from dataset \\ A

    Returns:
      group_B: list[int]
    """
    save_path = Path(save_path)

    if save_path.exists() and not force_rebuild:
        print(f"[INFO] Loading Group B from {save_path}")
        return load_indices(save_path)

    group_A = list(group_A)
    size_A = len(group_A)

    if groupB_size < size_A:
        raise ValueError(f"groupB_size={groupB_size} must be >= len(group_A)={size_A}")
    if groupB_size % num_classes != 0:
        raise ValueError(f"groupB_size={groupB_size} must be divisible by num_classes={num_classes}")
    if size_A % num_classes != 0:
        raise ValueError(f"len(group_A)={size_A} must be divisible by num_classes={num_classes}")

    rng = np.random.default_rng(seed)
    per_class_B = groupB_size // num_classes

    # One pass: build class lists + idx->class mapping
    class_to_indices = {c: [] for c in range(num_classes)}
    idx_to_class = {}

    for idx, (_, y) in enumerate(dataset):
        y = int(y)
        class_to_indices[y].append(idx)
        idx_to_class[idx] = y

    # Count A per class
    A_count = {c: 0 for c in range(num_classes)}
    for idx in group_A:
        c = idx_to_class[idx]
        A_count[c] += 1

    # Validate A doesn't already exceed target per class
    for c in range(num_classes):
        if A_count[c] > per_class_B:
            raise ValueError(
                f"Group A already has {A_count[c]} samples in class {c}, "
                f"but Group B target per class is {per_class_B}. "
                f"Choose a larger groupB_size."
            )

    # For each class, sample additional indices from pool excluding A
    A_set = set(group_A)
    extra_indices = []

    for c in range(num_classes):
        need = per_class_B - A_count[c]
        if need == 0:
            continue

        pool = np.array([i for i in class_to_indices[c] if i not in A_set])
        if len(pool) < need:
            raise ValueError(
                f"Not enough remaining samples for class {c}: "
                f"need {need}, but only {len(pool)} available outside Group A."
            )

        rng.shuffle(pool)
        extra_indices.extend(pool[:need].tolist())

    group_B = group_A + extra_indices
    rng.shuffle(group_B)

    if len(group_B) != groupB_size:
        raise RuntimeError(f"Internal error: expected |B|={groupB_size}, got {len(group_B)}")

    save_indices(group_B, save_path)
    print(f"[INFO] Created Group B superset: |A|={size_A}, |B|={len(group_B)} saved to {save_path}")
    return group_B


def get_targets(ds):
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets)
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels)

    if isinstance(ds, Subset):
        base_y = get_targets(ds.dataset)
        return base_y[np.asarray(ds.indices)]

    if isinstance(ds, ConcatDataset):
        return np.concatenate([get_targets(d) for d in ds.datasets])

    # slow fallback
    return np.asarray([ds[i][1] for i in range(len(ds))])


def group_check(train_set, group_A):
    import numpy as np
    from collections import Counter

    ys = get_targets(train_set)[np.asarray(group_A)]
    print("Subset size:", len(group_A))
    print("Unique classes in subset:", len(np.unique(ys)))
    print("Min label:", ys.min(), "Max label:", ys.max())

    cnt = Counter(ys.tolist())
    print("Smallest class counts:", sorted(cnt.items(), key=lambda x: x[1])[:5])
    print("Largest class counts:", sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:5])


def prepare_group_subset(train_set, group_A, rate, save_dir):
    group_check(train_set, group_A)
    
    train_subset = None

    if rate == 1:
        train_subset = Subset(train_set, group_A) # this is the same portion, no need to do further division since the training seed
                                              # keeps the same here.
    elif rate < 1:
        group_B = create_or_load_group_B(train_set, group_A, overlap_rate=rate,
                                    save_path=save_dir + f"/group_B_overlap_{rate}.npy")

        group_check(train_set, group_B)
        train_subset = Subset(train_set, group_B)
    else:# supersut containing the groupA with rate additional training samples
        base_length = len(group_A)
        group_B = create_or_load_group_B_superset(train_set, group_A, groupB_size=base_length+rate,
                                                  save_path=save_dir + f"/group_B_super_{rate}.npy")
        print(f"group_B length:{len(group_B)}")
        train_subset = Subset(train_set, group_B)
    
    return train_subset


def load_pickel_dataset(pkl_path):

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X = data["data"]
    Y = data["extrapolated_targets"]

    return X, Y


def extract_ft_balanced_train_val_indices(
    Y,
    split_size, # Renamed from total_size for clarity
    seed=42
):
    """
    Generate disjoint class-balanced subset indices for train and val.
    Returns:
        train_idx (np.ndarray), val_idx (np.ndarray)
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y)
    classes = np.unique(Y)
    num_classes = len(classes)

    if split_size % num_classes != 0:
        raise ValueError(f"split_size={split_size} not divisible by num_classes={num_classes}")

    per_class = split_size // num_classes
    train_idx = []
    val_idx = []

    for c in classes:
        idx_c = np.where(Y == c)[0]
        rng.shuffle(idx_c)

        needed = 2 * per_class
        if len(idx_c) < needed:
            raise ValueError(f"Class {c} has only {len(idx_c)} samples; need {needed}.")
            
        # We only care about the indices!
        train_idx.append(idx_c[:per_class])
        val_idx.append(idx_c[per_class:needed])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    # Return ONLY the indices
    return train_idx, val_idx


import os
import re
from pathlib import Path
from typing import Optional, Union


def save_checkpoint(path, net, optimizer, scheduler, epoch, best_acc, extra=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint_from_epoch(
    model_dir: Path,
    epoch: Optional[Union[int, str]] = None,
    pattern: str = r"(?:epoch|ep)[-_]?(-?\d+)",  # matches "epoch_12", "ep-12", etc.
) -> Optional[Path]:
    """
    Select a .pth checkpoint from a directory.

    Args:
        model_dir: directory containing .pth checkpoints
        epoch:
            - None or "latest": pick newest by mtime (your current behavior)
            - int: pick checkpoint whose filename contains that epoch number (best-effort)
        pattern: regex used to extract epoch from filename

    Returns:
        Path to selected checkpoint, or None if not found / no match.
    """
    ckpt_files = list(model_dir.glob("*.pth"))
    if not ckpt_files:
        return None

    # Default: latest by modification time
    if epoch is None or (isinstance(epoch, str) and epoch.lower() in {"latest", "newest", "last"}):
        return max(ckpt_files, key=lambda p: p.stat().st_mtime)

    # Epoch-based selection
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"epoch must be a non-negative int or None/'latest', got: {epoch}")

    # Build epoch -> candidates mapping
    epoch_candidates = []
    for p in ckpt_files:
        m = re.search(pattern, p.stem)
        if m:
            e = int(m.group(1))
            if e == epoch:
                epoch_candidates.append(p)

    if not epoch_candidates:
        return None

    # If multiple matches, pick the newest among them
    return max(epoch_candidates, key=lambda p: p.stat().st_mtime)


def load_last_checkpoint(model_dir: Path):
    """
    Load the most recent .pth checkpoint in a model directory.
    Returns the checkpoint path or None if not found.
    """
    ckpt_files = sorted(model_dir.glob("*.pth"), key=os.path.getmtime)
    if not ckpt_files:
        return None
    last_ckpt = ckpt_files[-1]  # the newest one
    print(f"[📦] Loading last checkpoint: {last_ckpt.name}")
    return last_ckpt

def load_best_checkpoint(model_dir: Path, filename: str = "best_epoch.pth") -> Path | None:
    """
    Return the path to the best checkpoint file (default: best_epoch.pth) if it exists.
    """
    model_dir = Path(model_dir)
    best_ckpt = model_dir / filename

    if best_ckpt.exists() and best_ckpt.is_file():
        print(f"[📦] Loading best checkpoint: {best_ckpt.name}")
        return best_ckpt, filename

    return None, None


def build_epoch_to_ckpt_map(
    model_dir: Path,
    gap: int = 1,
    pattern: str = r"(?:epoch|ep)[-_]?(-?\d+)",
) -> Dict[int, Path]:
    """
    Build a mapping epoch -> checkpoint path with:
      1) All negative epochs always included (if exist)
      2) Non-negative epochs subsampled by gap
      3) Final (largest) epoch always included

    If multiple checkpoints match the same epoch,
    the newest by modification time is kept.

    Args:
        model_dir: directory containing .pth checkpoints
        gap: minimum gap between selected non-negative epochs
        pattern: regex to extract epoch number

    Returns:
        Dict[int, Path] mapping selected epoch -> checkpoint path
    """
    ckpt_files = list(Path(model_dir).glob("*.pth"))

    # -----------------------------
    # Step 1: build epoch -> newest ckpt map
    # -----------------------------
    epoch_map: Dict[int, Path] = {}
    for p in ckpt_files:
        m = re.search(pattern, p.stem)
        if not m:
            continue
        e = int(m.group(1))
        if e not in epoch_map or p.stat().st_mtime > epoch_map[e].stat().st_mtime:
            epoch_map[e] = p

    if not epoch_map:
        return {}

    # -----------------------------
    # Step 2: separate epochs
    # -----------------------------
    negative_epochs = sorted(e for e in epoch_map if e < 0)
    nonneg_epochs = sorted(e for e in epoch_map if e >= 0)

    selected_epochs: List[int] = []

    # -----------------------------
    # Step 3: always include negative epochs
    # -----------------------------
    selected_epochs.extend(negative_epochs)

    # -----------------------------
    # Step 4: gap-based selection on non-negative epochs
    # -----------------------------
    if nonneg_epochs:
        current = nonneg_epochs[0]
        selected_epochs.append(current)

        for e in nonneg_epochs[1:]:
            if e >= current + gap:
                selected_epochs.append(e)
                current = e

        # -----------------------------
        # Step 5: always include final epoch
        # -----------------------------
        last_epoch = nonneg_epochs[-1]
        if last_epoch not in selected_epochs:
            selected_epochs.append(last_epoch)

    # -----------------------------
    # Step 6: return epoch -> path map
    # -----------------------------
    return {e: epoch_map[e] for e in sorted(selected_epochs)}


def remove_prune_mask(net):
    parameters_to_prune = []
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')


def remove_prune_mask_safe(net):
    """
    remove_prune_mask 的安全版本:只对真正挂了 weight 剪枝 mask 的模块调 prune.remove。

    为什么需要:prune_model_global 对 DeiT 会显式跳过 patch_embed 的 Conv2d,该模块
    没有 weight_orig/weight_mask。旧版 remove_prune_mask 无条件对所有 Conv2d/Linear
    调 prune.remove,碰到它就抛
        ValueError: Parameter 'weight' of module ... has to be pruned before
                    pruning can be removed
    在 CNN(ResNet/VGG,所有 Conv2d/Linear 都被剪)上两者行为完全等价。

    旧入口 remove_prune_mask 保留不动,供既有 CNN 剪枝脚本继续使用。

    返回实际固化了多少个模块,便于调用方做 sanity check。
    """
    removed = 0
    for _, module in net.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        # weight_orig 只有在 weight 被剪过之后才存在,这是最精确的判据
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
            removed += 1
    return removed


def sample_subset(dataset, sample_size, seed):
    """
    Randomly sample `sample_size` items from a given dataset.
    The selection is reproducible given the same seed.
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=sample_size, replace=False)
    return Subset(dataset, indices)


# =====================================================
# 2. Utilities for DeiT output handling + metrics
# =====================================================
def split_deit_outputs(outputs):
    """
    timm DeiT distilled models may return:
      - tuple/list: (cls_logits, dist_logits)
      - tensor: logits (already merged or non-distilled model)
    """
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return outputs[0], outputs[1]
    return outputs, None


def process_experiment_setup_deit(data):
    result = {}

    # -----------------------------
    # Dataset setup (now supports YAML-driven transforms)
    # -----------------------------
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    print(num_classes)
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size
    print(result["GroupSize"])

    # -----------------------------
    # Model setup
    # -----------------------------
    student_model = build_deit_student(data, num_classes)
    result["Student Model"] = student_model

    teacher_model = build_teacher_if_any(data, num_classes)
    result["Teacher Model"] = teacher_model

    # -----------------------------
    # Epoch setup
    # -----------------------------
    result["Epochs"] = data.get("Epochs", 100)

    # -----------------------------
    # Data Augmentation setup (Optional)-Mixup/Cutmix
    # -----------------------------
    aug_cfg = data.get("Augmentation", {})
    mixup_fn = build_timm_mixup(num_classes=num_classes, aug_cfg=aug_cfg)
    result["MixupFn"] = mixup_fn

    # ---- Base criterion consistent with mixup ----
    base_criterion = build_base_criterion(aug_cfg, use_mixup=(mixup_fn is not None))

    # -----------------------------
    # Optimizer setup
    # -----------------------------
    optimizer_cfg = data.get("Optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "Adam")
    optimizer_params = optimizer_cfg.get("params", {"lr": 1e-3})

    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(student_model.parameters(), **optimizer_params)
    result["Optimizer"] = optimizer

    # -----------------------------
    # Scheduler setup (new)
    # -----------------------------
    scheduler_cfg = data.get("Scheduler", {})

    scheduler = None
    if scheduler_cfg:
        scheduler_name = scheduler_cfg.get("name", None)
        scheduler_params = scheduler_cfg.get("params", {})

        if scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            result["Optimizer"],
                            T_max=int(scheduler_params.get("T_max", result["Epochs"])),
                            eta_min=float(scheduler_params.get("eta_min", 1e-6)),
                        )
        elif scheduler_name == "WarmupCosineAnnealingLR":
            scheduler =  build_warmup_cosine_scheduler(
                            result["Optimizer"],
                            total_epochs=int(scheduler_params.get("T_max", result["Epochs"])),
                            warmup_epochs=int(scheduler_params.get("warmup_epochs", 10)),
                            warmup_start_factor=float(scheduler_params.get("warmup_start_factor", 0.1)),
                            eta_min=float(scheduler_params.get("eta_min", 1e-6)),
                        )
            
    result["Scheduler"] = scheduler

    # -----------------------------
    # Distillation setup
    # -----------------------------
    dist_cfg = data.get("Distillation", {})

    enabled = bool(dist_cfg.get("enabled", True))  # default True if Distillation block exists
    print(f"the DeiT used the distilled {enabled} or not !")
    distill_type = str(dist_cfg.get("type", "hard")).lower()

    if enabled and distill_type != "none":
        # This flag is critical for timm 1.0.24 VisionTransformerDistilled to return tuple in train mode.
        if hasattr(result["Student Model"], "distilled_training"):
            result["Student Model"].distilled_training = True

    alpha = float(dist_cfg.get("alpha", 0.5))
    tau = float(dist_cfg.get("tau", 2.0))

    criterion = DistillationLoss(
        teacher=teacher_model,
        base_criterion=base_criterion,
        distill_type=distill_type,
        alpha=alpha,
        tau=tau,
    )
    
    result["Distillation"] = criterion

    return result


def build_deit_student(exp_yaml, num_classes):
    deit_cfg = exp_yaml.get("Model", {})
    model_name = deit_cfg.get("model_name", "deit_tiny_patch16_224")
    pretrained = bool(deit_cfg.get("pretrained", False))
    img_size = int(deit_cfg.get("img_size", 224))
    patch_size = int(deit_cfg.get("patch_size", 16))
    drop_path_rate = float(deit_cfg.get("drop_path_rate", 0.0))

    # Create model robustly: some timm models accept img_size directly, others need dynamic_img_size
    kwargs = dict(
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        img_size=img_size,
        patch_size=patch_size
    )

    # Try passing img_size; fall back if unsupported
    try:
        # kwargs["img_size"] = img_size
        model = timm.create_model(model_name, **kwargs)
    except TypeError:
        # fallback: remove img_size override
        kwargs.pop("img_size", None)
        model = timm.create_model(model_name, **kwargs)

    return model


def build_teacher_if_any(exp_yaml, num_classes):
    dist_cfg = exp_yaml.get("Distillation", {})
    enabled = bool(dist_cfg.get("enabled", True))

    if not enabled:
        return None
    
    teacher_name = dist_cfg.get("teacher_name", "resnet18")
    teacher_ckpt = dist_cfg.get("teacher_ckpt", "").strip()

    if teacher_name == "ResNet-18":
        teacher_model = ResNet18(num_classes=num_classes)
    elif teacher_name == "VGG16":
        teacher_model = ModifiedVGG16(num_classes=num_classes)
    else:
        raise ValueError("teacher model class is not supported now!")

    if teacher_ckpt:
        state = torch.load(teacher_ckpt, map_location="cpu")
        teacher_model.load_state_dict(state, strict=True)
    else:
        raise ValueError("Distillation.enabled=true but teacher_ckpt is empty. Provide a CIFAR-trained teacher checkpoint.")

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.to(device)
    return teacher_model


@torch.no_grad()
def logits_for_inference(cls_logits, dist_logits):
    if dist_logits is None:
        return cls_logits
    return (cls_logits + dist_logits) / 2.0


def hard_targets_from_maybe_soft(targets):
    # Mixup/CutMix yields soft targets shape [B, C]
    if targets.ndim == 2:
        return targets.argmax(dim=1)
    return targets


def compute_cls_metrics(y_true_np, y_pred_np):
    # match your macro metrics style
    precision = precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    return precision, recall, f1


def train_one_epoch_deit(net, loader, optimizer, criterion, epoch, device, mixup_fn=None):
    print(f'\nEpoch: {epoch}')
    net.train()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = criterion(inputs, outputs, targets) if isinstance(criterion, (DistillationLoss, FusionCELoss)) else criterion(
            split_deit_outputs(outputs)[0], targets
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) # to avoid gradient spikes
        optimizer.step()

        total_loss += loss.item()

        # metrics: use inference logits (avg heads if available)
        cls_logits, dist_logits = split_deit_outputs(outputs)
        logits = logits_for_inference(cls_logits, dist_logits)
        preds = logits.argmax(dim=1)

        hard_t = hard_targets_from_maybe_soft(targets)

        all_preds.append(preds.detach().cpu())
        all_targets.append(hard_t.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    train_acc = float((all_preds == all_targets).mean())
    precision, recall, f1 = compute_cls_metrics(all_targets, all_preds)

    print(f'\nEpoch: {epoch} ends.')
    return {
        "train_loss": total_loss / max(1, len(loader)),
        "train_acc": train_acc,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": f1,
    }


@torch.no_grad()
def evaluate_deit(net, loader, criterion_ce, device):
    net.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = net(inputs)
        cls_logits, dist_logits = split_deit_outputs(outputs)
        logits = logits_for_inference(cls_logits, dist_logits)

        loss = criterion_ce(logits, targets)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    test_acc = float((all_preds == all_targets).mean())
    precision, recall, f1 = compute_cls_metrics(all_targets, all_preds)

    return {
        "test_loss": total_loss / max(1, len(loader)),
        "test_acc": test_acc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    }


@torch.no_grad()
def evaluate_deit_ft(net, loader, criterion_ce, device):
    """
    DeiT 评估，但指标口径对齐 ft_one_epoch / evaluate1，而非 train_one_epoch_deit。

    为什么需要单独一个函数：evaluate_deit 与 train_one_epoch_deit 是一套自洽口径
    （0-1 / macro / per-batch），用于 DeiT 预训练；ft_one_epoch 与 evaluate1 是另一套
    （0-100 / weighted / per-sample），用于 fine-tune。fine-tune 路径复用 ft_one_epoch，
    若评估仍用 evaluate_deit，同一行 CSV 里 Train_* 与 Test_* 将不可比。
    而直接改 evaluate_deit 会破坏预训练路径的自洽性与 best_epoch 的筛选口径。

    DeiT 特有的多头处理（split + average）与 evaluate_deit 保持一致。
    """
    net.eval()
    test_loss = 0.0
    all_preds, all_targets = [], []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = net(inputs)
        cls_logits, dist_logits = split_deit_outputs(outputs)
        logits = logits_for_inference(cls_logits, dist_logits)

        loss = criterion_ce(logits, targets)
        test_loss += loss.item() * targets.size(0)

        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    total = len(all_targets)

    return {
        "test_loss": test_loss / total,
        "test_acc": 100. * float((all_preds == all_targets).mean()),
        "test_precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "test_recall": recall_score(all_targets, all_preds, average="weighted", zero_division=0),
        "test_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
    }


def build_warmup_cosine_scheduler(
    optimizer,
    total_epochs: int,
    warmup_epochs: int = 10,
    warmup_start_factor: float = 0.1,
    eta_min: float = 1e-6,
):
    """
    Epoch-based warmup + cosine scheduler.

    - Warmup: linearly increase LR from (base_lr * warmup_start_factor) to base_lr over warmup_epochs.
    - Cosine: cosine annealing from base_lr to eta_min over the remaining epochs.
    """
    warmup_epochs = int(warmup_epochs)
    total_epochs = int(total_epochs)

    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError(f"warmup_epochs must be in [0, total_epochs-1], got {warmup_epochs} with total_epochs={total_epochs}")

    # No warmup -> plain cosine
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)

    # Phase 1: warmup
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Phase 2: cosine (remaining epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min,
    )

    # Switch to cosine after warmup_epochs
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    return scheduler


def build_timm_mixup(num_classes: int, aug_cfg: dict):
    """
    Returns timm.data.Mixup object or None.
    """
    if not bool(aug_cfg.get("use_mixup", False)):
        return None

    mixup_alpha = float(aug_cfg.get("mixup_alpha", 0.8))
    cutmix_alpha = float(aug_cfg.get("cutmix_alpha", 1.0))
    prob = float(aug_cfg.get("prob", 1.0))
    switch_prob = float(aug_cfg.get("switch_prob", 0.5))
    label_smoothing = float(aug_cfg.get("label_smoothing", 0.0))

    # timm Mixup: when both mixup and cutmix are set, it will sample between them
    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_minmax=None,          # or a tuple like (0.2, 1.0) if you want constrained cutmix area
        prob=prob,
        switch_prob=switch_prob,
        mode="batch",                # 'batch' is typical
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )
    return mixup_fn


def build_base_criterion(aug_cfg: dict, use_mixup: bool):
    if use_mixup:
        return SoftTargetCrossEntropy()
    else:
        ls = float(aug_cfg.get("label_smoothing", 0.0))
        if ls > 0:
            return LabelSmoothingCrossEntropy(smoothing=ls)
        return nn.CrossEntropyLoss()
    

def _reinit_linear(layer: nn.Linear):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def setup_finetune_deit(
    model,
    strategy: str,
    num_classes: int,
    device="cuda",
    train_distill_head: bool = False,  # True if you want to update head_dist too
    reinit_head: bool = False,         # if True, reinit new heads (RT-AL-like)
):
    """
    Fine-tuning setup for timm DeiT/ViT-style models.

    strategy:
      - 'FT-LL': freeze backbone, train head(s) only
      - 'FT-AL': train all params

    train_distill_head:
      - False: only train model.head
      - True: train model.head and model.head_dist (if exists)

    Notes:
      - timm DeiT models typically expose .head and optionally .head_dist.
      - This function replaces heads to match num_classes.
    """

    # ---- identify heads ----
    if not hasattr(model, "head") or not isinstance(model.head, nn.Linear):
        raise ValueError("Expected a timm DeiT/ViT model with a Linear `model.head`.")

    has_dist = hasattr(model, "head_dist") and isinstance(getattr(model, "head_dist"), nn.Linear)

    embed_dim = model.head.in_features

    out_features = model.head.out_features

    # ---- apply strategy ----
    if strategy == "FT-LL":
        # freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # unfreeze class head
        for p in model.head.parameters():
            p.requires_grad = True

        # optionally unfreeze distill head too
        if train_distill_head and has_dist:
            for p in model.head_dist.parameters():
                p.requires_grad = True

    elif strategy == "FT-AL":
        for p in model.parameters():
            p.requires_grad = True

        # optional: if you explicitly DON'T want to train distill head even in FT-AL
        # you can freeze it here, but usually for FT-AL you keep all trainable.

    else:
        raise ValueError(f"Unknown fine-tuning strategy: {strategy}")

    return model


def process_experiment_setup_deit_ft(data):
    result = {}

    # -----------------------------
    # Dataset setup (now supports YAML-driven transforms)
    # -----------------------------
    ds_cfg = data.get("Dataset")
    dataset_obj, num_classes, group_size = build_dataset_from_yaml(ds_cfg)
    print(num_classes)
    result["Dataset"] = dataset_obj
    result["NumClasses"] = num_classes
    result["GroupSize"] = group_size
    print(result["GroupSize"])

    # -----------------------------
    # Model setup
    # -----------------------------
    student_model = build_deit_student(data, num_classes)
    result["Student Model"] = student_model

    teacher_model = build_teacher_if_any(data, num_classes)
    result["Teacher Model"] = teacher_model

    # -----------------------------
    # Epoch setup
    # -----------------------------
    result["Epochs"] = data.get("Epochs", 100)

    # -----------------------------
    # Data Augmentation setup (Optional)-Mixup/Cutmix
    # -----------------------------
    aug_cfg = data.get("Augmentation", {})
    mixup_fn = build_timm_mixup(num_classes=num_classes, aug_cfg=aug_cfg)
    result["MixupFn"] = mixup_fn

    # ---- Base criterion consistent with mixup ----
    base_criterion = build_base_criterion(aug_cfg, use_mixup=(mixup_fn is not None))

    # -----------------------------
    # Optimizer setup
    # -----------------------------
    optimizer_cfg = data.get("Optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "Adam")
    optimizer_params = optimizer_cfg.get("params", {"lr": 1e-3})

    optimizer_class = getattr(optim, optimizer_name)
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = optimizer_class(trainable_params, **optimizer_params)

    result["Optimizer"] = optimizer

    # -----------------------------
    # Scheduler setup (new)
    # -----------------------------
    scheduler_cfg = data.get("Scheduler", {})

    scheduler = None
    if scheduler_cfg:
        scheduler_params = scheduler_cfg.get("params", {})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        result["Optimizer"],
                        T_max=int(scheduler_params.get("T_max", result["Epochs"])),
                        eta_min=float(scheduler_params.get("eta_min", 1e-6)),
                    ) # In fine-tuning, we do not expect to use warmup scheduler since the overall epochs is small
            
    result["Scheduler"] = scheduler

    # -----------------------------
    # Distillation setup
    # -----------------------------
    dist_cfg = data.get("Distillation", {})

    enabled = bool(dist_cfg.get("enabled", True))  # default True if Distillation block exists
    distill_type = str(dist_cfg.get("type", "hard")).lower()

    if enabled and distill_type != "none":
        # This flag is critical for timm 1.0.24 VisionTransformerDistilled to return tuple in train mode.
        if hasattr(result["Student Model"], "distilled_training"):
            result["Student Model"].distilled_training = True

    alpha = float(dist_cfg.get("alpha", 0.5))

    criterion = FusionCELoss(
        base_criterion=base_criterion,
        alpha=alpha,
    )
    
    result["Distillation"] = criterion

    return result




import numpy as np
from typing import Sequence, Optional, Tuple, Union, List, Dict, Literal

Split = Literal["train", "test"]
TaggedIndex = Tuple[Split, int]

def create_class_balanced_mix_train_test(
    train_dataset,
    test_dataset,
    train_in_indices: Sequence[int],            # e.g., group_A indices into train_dataset
    num_classes: int = 10,
    total_size: Optional[int] = None,           # total mixed size (divisible by num_classes)
    per_class: Optional[int] = None,            # samples per class in final mixed set
    frac_in_from_train: Optional[float] = None, # fraction per class coming from train_in_indices
    per_class_from_train: Optional[int] = None, # exact count per class from train
    seed: int = 42,
    replace: bool = False,
    shuffle: bool = True,
    return_parts: bool = False,
    test_out_indices: Optional[Sequence[int]] = None,  # optionally restrict test pool
) -> Union[
    List[TaggedIndex],
    Tuple[List[TaggedIndex], List[TaggedIndex], List[TaggedIndex]]
]:
    """
    Build a class-balanced mixed set where:
      - "in" samples come from train_dataset using train_in_indices (e.g., group_A)
      - "out" samples come from test_dataset (default: all test indices, or test_out_indices if provided)

    Returns:
      mixed: List[(split, idx)] where split in {"train","test"} and idx is index into that dataset.
      Optionally also returns (picked_train, picked_test).
    """
    rng = np.random.default_rng(seed)

    train_in = np.asarray(list(train_in_indices), dtype=int)
    test_pool = np.arange(len(test_dataset), dtype=int) if test_out_indices is None \
                else np.asarray(list(test_out_indices), dtype=int)

    # ---- determine per_class ----
    if per_class is None:
        if total_size is None:
            raise ValueError("Provide either total_size or per_class.")
        if total_size % num_classes != 0:
            raise ValueError(f"total_size={total_size} must be divisible by num_classes={num_classes}.")
        per_class = total_size // num_classes
    else:
        per_class = int(per_class)

    # ---- determine per_class_from_train ----
    if per_class_from_train is None:
        if frac_in_from_train is None:
            raise ValueError("Provide either frac_in_from_train or per_class_from_train.")
        if not (0.0 <= frac_in_from_train <= 1.0):
            raise ValueError("frac_in_from_train must be in [0, 1].")
        per_class_from_train = int(round(per_class * frac_in_from_train))
    else:
        per_class_from_train = int(per_class_from_train)

    per_class_from_test = per_class - per_class_from_train
    if per_class_from_test < 0:
        raise ValueError("per_class_from_train cannot exceed per_class.")

    # ---- build pools by class from TRAIN and TEST ----
    train_by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx in train_in:
        _, y = train_dataset[int(idx)]
        train_by_class[int(y)].append(int(idx))

    test_by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx in test_pool:
        _, y = test_dataset[int(idx)]
        test_by_class[int(y)].append(int(idx))

    picked_train: List[TaggedIndex] = []
    picked_test: List[TaggedIndex] = []

    # ---- sample per class ----
    for c in range(num_classes):
        Tr_c = np.asarray(train_by_class[c], dtype=int)
        Te_c = np.asarray(test_by_class[c], dtype=int)

        if not replace:
            if per_class_from_train > len(Tr_c):
                raise ValueError(f"Class {c}: need {per_class_from_train} from TRAIN, only {len(Tr_c)} available.")
            if per_class_from_test > len(Te_c):
                raise ValueError(f"Class {c}: need {per_class_from_test} from TEST, only {len(Te_c)} available.")

        if per_class_from_train > 0:
            chosen = rng.choice(Tr_c, size=per_class_from_train, replace=replace)
            picked_train.extend([("train", int(i)) for i in chosen])

        if per_class_from_test > 0:
            chosen = rng.choice(Te_c, size=per_class_from_test, replace=replace)
            picked_test.extend([("test", int(i)) for i in chosen])

    mixed = picked_train + picked_test
    if shuffle:
        rng.shuffle(mixed)

    if return_parts:
        return mixed, picked_train, picked_test
    return mixed


class MixedSplitDataset(Dataset):
    def __init__(
        self,
        train_dataset,
        test_dataset,
        tagged_indices: List[TaggedIndex],
        return_split: bool = False,   # if True, returns (x,y,split_id)
        split_ids: Tuple[int, int] = (0, 1),  # (train_id, test_id)
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tagged_indices = tagged_indices
        self.return_split = return_split
        self.train_id, self.test_id = split_ids

    def __len__(self):
        return len(self.tagged_indices)

    def __getitem__(self, i: int):
        split, idx = self.tagged_indices[i]
        if split == "train":
            x, y = self.train_dataset[idx]
            if self.return_split:
                return x, y, self.train_id
            return x, y
        elif split == "test":
            x, y = self.test_dataset[idx]
            if self.return_split:
                return x, y, self.test_id
            return x, y
        else:
            raise ValueError(f"Unknown split tag: {split}")


def verify_optimizer_param_binding(optimizer, model, label=""):
    """Check if optimizer's parameters are the same objects as model's parameters."""
    model_params = set(id(p) for p in model.parameters())
    optim_params = set(id(p) for p in optimizer.param_groups[0]['params'])
    
    shared = model_params & optim_params
    only_model = model_params - optim_params
    only_optim = optim_params - model_params
    
    print(f"\n[🔍 Param Binding Check: {label}]")
    print(f"  Model params:     {len(model_params)}")
    print(f"  Optimizer params:  {len(optim_params)}")
    print(f"  Shared (correct):  {len(shared)}")
    print(f"  Only in model:     {len(only_model)}")
    print(f"  Only in optimizer: {len(only_optim)}")
    
    if only_model or only_optim:
        print("  [❌] MISMATCH — optimizer is NOT bound to this model!")
    else:
        print("  [✅] All parameters match.")


def snapshot_params(model):
    """Take a frozen copy of all parameter values."""
    return {name: p.clone().detach() for name, p in model.named_parameters()}

def verify_params_changed(before, after, label=""):
    """Compare two snapshots and report which params moved."""
    changed = 0
    unchanged = 0
    for name in before:
        if torch.equal(before[name], after[name]):
            unchanged += 1
        else:
            changed += 1
    
    print(f"\n[🔍 Weight Update Check: {label}]")
    print(f"  Changed:   {changed}")
    print(f"  Unchanged: {unchanged}")
    if changed == 0:
        print("  [❌] NO parameters were updated — optimizer is likely stale!")
    else:
        print(f"  [✅] {changed} parameter groups updated.")


def determine_ft_dataset(exp_yaml, exp_setup, group_A):
    if exp_yaml["FT_Dataset"]["name"] == "PseudoLabelCIFAR-10":
        pkl_path = "E:/Experiment/data/ti_500K_pseudo_labeled.pickle"
        X, Y = load_pickel_dataset(pkl_path)
        idx_tr, idx_va = extract_ft_balanced_train_val_indices(
            Y, split_size=exp_setup["FT_GroupSize"], seed=42
        )

        exp_setup["FT_Dataset"].set_data(X, Y)
        ft_data_train = exp_setup["FT_Dataset"].subset("train", idx_tr, clean=False)
        ft_data_val = exp_setup["FT_Dataset"].subset("train", idx_va, clean=True)

    elif exp_yaml["FT_Dataset"]["name"] == "CIFARNet":
        hf_train_labels = exp_setup["FT_Dataset"].train_set.hf_dataset["label"]
        idx_tr, idx_va = extract_ft_balanced_train_val_indices(
            Y=hf_train_labels,
            split_size=exp_setup["FT_GroupSize"],
            seed=42
        )
        ft_data_train = exp_setup["FT_Dataset"].subset("train", idx_tr, clean=False)
        ft_data_val = exp_setup["FT_Dataset"].subset("train", idx_va, clean=True)

    else:
        group_B = create_or_load_group_B(
            save_dir=f'./Indices/{exp_yaml["Dataset"]["name"]}/',
            overlap_rate=0.0,
            group_A_indices=group_A,
            dataset=exp_setup["Dataset"].train_set,
            group_size=exp_setup["FT_GroupSize"],
            num_classes=exp_setup["NumClasses"],
            seed=42,
            force_rebuild=False
        )
        ft_data_train = exp_setup["Dataset"].subset("train", group_B, clean=False)
        ft_data_val = exp_setup["Dataset"].test_set

    return ft_data_train, ft_data_val


def dbg(tag, msg):
    print(f"[DEBUG][{tag}] {msg}")


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

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Convert logits to soft probabilities
            soft_labels = torch.softmax(outputs / temperature, dim=1)

            all_inputs.append(inputs.cpu())
            all_outputs.append(soft_labels.cpu())

    all_inputs = torch.cat(all_inputs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    return all_inputs, all_outputs


def train_one_epoch_knockoff(net, trainloader, optimizer, criterion, epoch, device):
    """Training loop for knockoff attack with soft labels."""
    print(f'\nEpoch: {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    net.train()
    for batch_idx, (inputs, soft_targets) in enumerate(trainloader):
        inputs, soft_targets = inputs.to(device), soft_targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, soft_targets)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        _, hard_targets = soft_targets.max(1)  # convert soft labels to hard for accuracy
        total += inputs.size(0)
        correct += predicted.eq(hard_targets).sum().item()

        all_preds.append(predicted.detach().cpu().numpy())
        all_targets.append(hard_targets.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = running_loss / total
    avg_acc = 100. * correct / total

    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    print(f'\nEpoch: {epoch} ends')
    return {
        "train_loss": avg_loss,
        "train_acc": avg_acc,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": f1
    }

def evaluate_fidelity(victim_net, substitute_net, dataloader, device):
    victim_net.eval()
    substitute_net.eval()

    agree = 0
    total = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            v_out = victim_net(inputs)
            if isinstance(v_out, tuple):
                v_out = v_out[0]
            s_out = substitute_net(inputs)
            if isinstance(s_out, tuple):
                s_out = s_out[0]

            v_pred = v_out.argmax(dim=1)
            s_pred = s_out.argmax(dim=1)

            agree += (v_pred == s_pred).sum().item()
            total += inputs.size(0)

    return 100.0 * agree / total


def partition_indices_class_balanced(dataset, indices, num_partitions=10, seed=42):
    """
    Split `indices` into `num_partitions` disjoint, class-balanced partitions.

    Parameters
    ----------
    dataset : torch Dataset
        The dataset (e.g. CIFAR-10 train_set) to look up labels from.
    indices : array-like
        The pool of indices to partition (e.g. group_B).
    num_partitions : int
        Number of partitions to create.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    partitions : list of np.ndarray
        Each element is an array of indices for that partition.
    """
    rng = np.random.default_rng(seed)

    # ── Group indices by class ─────────────────────────────────────
    class_to_indices = defaultdict(list)
    for idx in indices:
        label = dataset.targets[idx] if hasattr(dataset, 'targets') else dataset[idx][1]
        class_to_indices[label].append(idx)

    # ── Shuffle within each class, then deal into partitions ───────
    partitions = [[] for _ in range(num_partitions)]

    for cls in sorted(class_to_indices.keys()):
        cls_indices = np.array(class_to_indices[cls])
        rng.shuffle(cls_indices)

        per_partition = len(cls_indices) // num_partitions
        if per_partition * num_partitions != len(cls_indices):
            print(f"Warning: class {cls} has {len(cls_indices)} samples, "
                  f"not evenly divisible by {num_partitions}. "
                  f"Using first {per_partition * num_partitions}.")

        for i in range(num_partitions):
            partitions[i].extend(cls_indices[i * per_partition : (i + 1) * per_partition])

    # Shuffle within each partition so classes aren't in order
    partitions = [np.array(p) for p in partitions]
    for p in partitions:
        rng.shuffle(p)

    return partitions



import subprocess
import time

def get_gpu_temp():
    """Query GPU temperature via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        temps = [int(t.strip()) for t in result.stdout.strip().split("\n")]
        return max(temps)  # return hottest GPU if multi-GPU
    except Exception as e:
        print(f"[WARNING] Could not read GPU temp: {e}")
        return None


def wait_for_cool_gpu(threshold=82, check_interval=30):
    """Block execution until GPU temp drops below threshold."""
    temp = get_gpu_temp()
    if temp is None:
        return

    if temp >= threshold:
        print(f"[THERMAL] GPU at {temp}°C (threshold: {threshold}°C). Pausing...")

    while temp is not None and temp >= threshold:
        time.sleep(check_interval)
        temp = get_gpu_temp()
        print(f"[THERMAL] GPU at {temp}°C, waiting...")

    if temp is not None:
        print(f"[THERMAL] GPU at {temp}°C. Resuming.")



def collect_logits(net, data_loader, device):
    """Run the model over the loader once and return cached logits + one-hot labels.
 
    This is the expensive step. The returned tensors can be reused for any
    number of MI computations at different bin counts.
 
    Returns:
        layer_T:      (N, K) float32 logits tensor on `device`.
        label_matrix: (N, K_y) float32 one-hot label tensor on `device`.
    """
    if hasattr(net, "fc"):
        num_classes = net.fc.out_features
    elif hasattr(net, "classifier"):
        # assume last Linear layer in classifier
        last_linear = [m for m in net.classifier.modules() if isinstance(m, nn.Linear)][-1]
        num_classes = last_linear.out_features
    elif hasattr(net, "num_classes"): # this is made for the timm ViT model class
        num_classes = net.num_classes
    else:
        raise ValueError("Cannot automatically infer num_classes from model.")
    
    layer_T_list, label_list = [], []
 
    net.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            layer_T_list.append(outputs.detach())
            label_list.append(F.one_hot(targets.detach(),
                                        num_classes=num_classes).float())
 
    layer_T = torch.cat(layer_T_list, dim=0).to(dtype=torch.float32)
    label_matrix = torch.cat(label_list, dim=0).to(dtype=torch.float32)
    return layer_T, label_matrix


def mi_from_logits(layer_T, label_matrix, num_intervals=50, verbose=False):
    """Compute (I(X;T), I(T;Y)) from cached logits at a given bin count.
 
    This is the function to call repeatedly for bin sweeps. It does NOT run
    the model; it only does softmax + binning + counting + entropy.
 
    Returns (I_X_T, I_T_Y) when verbose=False, or
            (I_X_T, I_T_Y, debug_data) when verbose=True.
    """
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
 
    if not verbose:
        return I_X_T.item(), I_T_Y.item()
 
    debug_data = {
        "T_soft":         T_soft.detach().cpu().numpy().astype(np.float32),
        "T_discrete":     T_discrete.detach().cpu().numpy().astype(np.int16),
        "unique_T":       unique_T.detach().cpu().numpy().astype(np.int16),
        "inverse_idx":    inverse_idx.detach().cpu().numpy().astype(np.int32),
        "pattern_counts": T_counts.detach().cpu().numpy().astype(np.int64),
        "label_counts":   TY_counts.detach().cpu().numpy().astype(np.int64),
        "N":              np.int64(N),
        "K":              np.int64(layer_T.shape[1]),
        "K_y":            np.int64(K_y),
        "U":              np.int64(K_unique),
        "num_intervals":  np.int64(num_intervals),
        "I_X_T":          np.float64(I_X_T.item()),
        "I_T_Y":          np.float64(I_T_Y.item()),
    }
    return I_X_T.item(), I_T_Y.item(), debug_data
