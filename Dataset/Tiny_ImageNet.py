import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


# =============================================================
# Core file-based Tiny-ImageNet dataset (analogous to torchvision.CIFAR10)
# =============================================================
class TinyImageNet(Dataset):
    """
    File-based Tiny-ImageNet loader for the Stanford layout:

        tiny-imagenet-200/
        ├── wnids.txt
        ├── words.txt
        ├── train/<wnid>/images/*.JPEG
        └── val/
            ├── val_annotations.txt
            └── images/*.JPEG   (flat)

    The 'test' split in the original release is unlabeled, so here 'train'
    corresponds to the 100k labeled training images and 'val' corresponds
    to the 10k labeled validation images (used as the held-out test split).
    """
    SPLITS = ("train", "val")

    def __init__(self, root: str, split: str = "train", transform=None, download: bool = False):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.transform = transform

        if download:
            self._maybe_download()

        if not self.root.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet root not found: {self.root}. "
                f"Pass download=True or extract tiny-imagenet-200.zip manually."
            )

        # Canonical class order from wnids.txt so class indices are reproducible.
        with open(self.root / "wnids.txt") as f:
            self.wnids = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.classes = self.wnids  # torchvision-compatible attribute

        # Human-readable names (optional, for diagnostics).
        self.idx_to_name = {}
        words_path = self.root / "words.txt"
        if words_path.exists():
            with open(words_path) as f:
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2 and parts[0] in self.class_to_idx:
                        self.idx_to_name[self.class_to_idx[parts[0]]] = parts[1]

        self.samples: List = []
        if split == "train":
            for wnid in self.wnids:
                img_dir = self.root / "train" / wnid / "images"
                if not img_dir.exists():
                    continue
                for name in sorted(os.listdir(img_dir)):
                    if name.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((img_dir / name, self.class_to_idx[wnid]))
        else:  # val
            ann = self.root / "val" / "val_annotations.txt"
            img_root = self.root / "val" / "images"
            with open(ann) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    fname, wnid = parts[0], parts[1]
                    if wnid in self.class_to_idx:
                        self.samples.append((img_root / fname, self.class_to_idx[wnid]))

        # Expose targets for downstream tooling (label-stratified splits, etc.)
        self.targets = [t for _, t in self.samples]

    def _maybe_download(self):
        """Download + extract the Stanford Tiny-ImageNet zip if not already present."""
        if self.root.exists() and (self.root / "wnids.txt").exists():
            return
        import urllib.request
        import zipfile

        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        parent = self.root.parent
        parent.mkdir(parents=True, exist_ok=True)
        zip_path = parent / "tiny-imagenet-200.zip"

        if not zip_path.exists():
            print(f"[TinyImageNet] Downloading from {url} ...")
            urllib.request.urlretrieve(url, zip_path)

        print(f"[TinyImageNet] Extracting to {parent} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(parent)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # ~1% of Tiny-ImageNet images are single-channel grayscale; force RGB.
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# =============================================================
# TinyImageNetDataset — structured wrapper matching CIFAR10Dataset
# =============================================================
class TinyImageNetDataset(Dataset):
    """
    Tiny-ImageNet wrapper with YAML-driven transform pipelines.

    Mirrors the CIFAR10Dataset interface:
      - train_set: uses train_transforms (aug)
      - test_set:  uses test_transforms (clean)   <-- built from the 'val' split
      - in_sample_set: train data with test_transforms (clean probing)
      - raw_test_set: val split, ToTensor only (for adversarial eval)
      - raw_train_set: train split, spatial aug only, no normalization
      - raw_train_clean_set: train split, ToTensor only

    Transform specs (list of dicts):
      [{"name": "RandomCrop", "params": {"size": 64, "padding": 8}, "enabled": True}, ...]

    Notes
    -----
    * Tiny-ImageNet has no labeled 'test' split in the public release.
      We treat the 'val' split as the test split (standard practice).
    * Default img_size is 64. If you set img_size != 64, Resize is added to
      the default test pipeline so shapes stay consistent.
    """

    def __init__(
        self,
        normalization: str = "tiny_imagenet",
        loading: str = "file",
        root_dir: str = "./data/tiny-imagenet-200",
        img_size: int = 64,
        train_transforms: Optional[List[Dict[str, Any]]] = None,
        test_transforms: Optional[List[Dict[str, Any]]] = None,
        build_dataset: bool = True,
        download: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.loading = loading
        self.normalization = normalization
        self.img_size = int(img_size)
        self.download = download

        stats = self.set_normalization(normalization)
        self.mean = stats["mean"]
        self.std = stats["std"]

        # Build transform pipelines (YAML-driven or defaults)
        self.train_transform = self._build_transform_pipeline(train_transforms, is_train=True)
        self.test_transform = self._build_transform_pipeline(test_transforms, is_train=False)

        # Raw pipelines for adversarial workflows (no normalization).
        # If img_size != 64, resize first so PGD still operates in pixel space
        # at the intended resolution.
        raw_steps = []
        if self.img_size != 64:
            raw_steps.append(transforms.Resize((self.img_size, self.img_size)))
        raw_steps.append(transforms.ToTensor())
        self.raw_transform = transforms.Compose(raw_steps)

        raw_train_steps = []
        if self.img_size != 64:
            raw_train_steps.append(transforms.Resize((self.img_size, self.img_size)))
        raw_train_steps.extend([
            transforms.RandomCrop(self.img_size, padding=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        self.raw_train_transform = transforms.Compose(raw_train_steps)

        self.train_set = None
        self.test_set = None
        self.in_sample_set = None
        self.raw_test_set = None
        self.raw_train_set = None
        self.raw_train_clean_set = None

        if build_dataset:
            self._build_datasets()

    # ---------------------------
    # Build datasets
    # ---------------------------
    def _build_datasets(self):
        # 'train' split of Tiny-ImageNet with training aug
        self.train_set = self.get_dataset(train=True, transform=self.train_transform, download=self.download)
        # 'val' split used as test (labeled held-out)
        self.test_set = self.get_dataset(train=False, transform=self.test_transform, download=self.download)
        # Train data, test-time transform (clean probing for fingerprinting, RobD, etc.)
        self.in_sample_set = self.get_dataset(train=True, transform=self.test_transform, download=self.download)

        # Raw variants for adversarial evaluation (PGD operates in [0,1] pixel space)
        self.raw_test_set = self.get_dataset(train=False, transform=self.raw_transform, download=self.download)
        self.raw_train_set = self.get_dataset(train=True, transform=self.raw_train_transform, download=self.download)
        self.raw_train_clean_set = self.get_dataset(train=True, transform=self.raw_transform, download=self.download)

    # ---------------------------
    # Normalization
    # ---------------------------
    def set_normalization(self, normalization: str):
        if normalization == "tiny_imagenet":
            # Computed over Tiny-ImageNet training set
            mean = (0.4802, 0.4481, 0.3975)
            std = (0.2770, 0.2691, 0.2821)
        elif normalization == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif normalization == "cifar10":
            # Occasionally useful for cross-dataset transfer experiments
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif normalization == "cifar100":
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        else:
            raise NotImplementedError(f"Unknown normalization: {normalization}")
        return {"mean": mean, "std": std}

    # ---------------------------
    # Transform factory
    # ---------------------------
    def _build_transform_pipeline(self, specs: Optional[List[Dict[str, Any]]], is_train: bool):
        """If specs is None or empty -> fallback to defaults."""
        if not specs:
            return self._default_transform(is_train=is_train)

        steps = []
        for spec in specs:
            if spec is None:
                continue
            if not spec.get("enabled", True):
                continue
            name = spec.get("name")
            if not name:
                raise ValueError(f"Transform spec missing 'name': {spec}")
            params = spec.get("params", {}) or {}
            steps.append(self._make_transform(name, params))
        return transforms.Compose(steps)

    def _default_transform(self, is_train: bool):
        if is_train:
            steps = []
            # If caller upscaled, Resize first so RandomCrop targets img_size
            if self.img_size != 64:
                steps.append(transforms.Resize((self.img_size, self.img_size)))
            steps.extend([
                transforms.RandomCrop(self.img_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            return transforms.Compose(steps)
        else:
            steps = []
            if self.img_size != 64:
                steps.append(transforms.Resize((self.img_size, self.img_size)))
            steps.extend([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            return transforms.Compose(steps)

    def _make_transform(self, name: str, params: Dict[str, Any]):
        name = name.strip()

        # ---- geometric ----
        if name == "Resize":
            size = params.get("size", self.img_size)
            if isinstance(size, int):
                size = (size, size)
            return transforms.Resize(size=size)

        if name == "RandomCrop":
            size = params.get("size", self.img_size)
            return transforms.RandomCrop(
                size=size,
                padding=params.get("padding", 0),
                pad_if_needed=params.get("pad_if_needed", False),
                padding_mode=params.get("padding_mode", "constant"),
            )

        if name == "RandomResizedCrop":
            return transforms.RandomResizedCrop(
                size=params.get("size", self.img_size),
                scale=params.get("scale", (0.8, 1.0)),
                ratio=params.get("ratio", (3 / 4, 4 / 3)),
                interpolation=params.get("interpolation", transforms.InterpolationMode.BILINEAR),
            )

        if name == "RandomHorizontalFlip":
            return transforms.RandomHorizontalFlip(p=params.get("p", 0.5))

        if name == "ColorJitter":
            return transforms.ColorJitter(
                brightness=params.get("brightness", 0.0),
                contrast=params.get("contrast", 0.0),
                saturation=params.get("saturation", 0.0),
                hue=params.get("hue", 0.0),
            )

        # ---- policy-based ----
        if name == "RandAugment":
            return transforms.RandAugment(
                num_ops=params.get("num_ops", 2),
                magnitude=params.get("magnitude", 9),
            )

        if name == "AutoAugment":
            # Tiny-ImageNet has no dedicated policy; ImageNet is the closest match.
            policy = params.get("policy", "IMAGENET")
            policy_enum = getattr(transforms.AutoAugmentPolicy, policy)
            return transforms.AutoAugment(policy_enum)

        # ---- tensor + normalize ----
        if name == "ToTensor":
            return transforms.ToTensor()

        if name == "Normalize":
            return transforms.Normalize(
                mean=params.get("mean", self.mean),
                std=params.get("std", self.std),
            )

        # ---- tensor-level ----
        if name == "RandomErasing":
            return transforms.RandomErasing(
                p=params.get("p", 0.25),
                scale=params.get("scale", (0.02, 0.33)),
                ratio=params.get("ratio", (0.3, 3.3)),
                value=params.get("value", 0),
                inplace=params.get("inplace", False),
            )

        raise NotImplementedError(f"Unknown transform name: '{name}'. Add it in _make_transform().")

    # ---------------------------
    # Dataset retrieval
    # ---------------------------
    def get_dataset(self, train: bool, transform, download: bool = True):
        split = "train" if train else "val"
        if self.loading == "file":
            return TinyImageNet(
                root=self.root_dir,
                split=split,
                transform=transform,
                download=download,
            )
        elif self.loading == "huggingface":
            # Alternate path: HF mirror (zh-plus/tiny-imagenet has train+valid splits).
            from datasets import load_dataset
            hf = load_dataset("zh-plus/tiny-imagenet")
            hf_split = hf["train"] if train else hf["valid"]
            return _HFTinyImageNetWrapper(hf_split, transform=transform)
        elif self.loading == "custom":
            raise NotImplementedError("Custom Tiny-ImageNet loader not implemented.")
        else:
            raise NotImplementedError(f"Unknown loading mode: {self.loading}")

    def subset(self, split: str, indices, clean: bool = False):
        """
        Create a torch.utils.data.Subset from one of the internal datasets.

        split:
          - "train"      -> train_set (or in_sample_set if clean=True)
          - "test"       -> test_set
          - "raw_test"   -> raw_test_set
          - "raw_train"  -> raw_train_set
        """
        split = split.lower().strip()
        if split == "train":
            base = self.in_sample_set if clean else self.train_set
        elif split == "test":
            base = self.test_set
        elif split == "raw_test":
            base = self.raw_test_set
        elif split == "raw_train":
            base = self.raw_train_set
        else:
            raise ValueError(f"Unknown split: {split}")

        if base is None:
            raise ValueError(f"Base dataset is None for split={split}, clean={clean}")

        return Subset(base, indices)


# =============================================================
# Optional HF wrapper (kept here so the file is self-contained)
# =============================================================
class _HFTinyImageNetWrapper(Dataset):
    """Thin wrapper around a Hugging Face Tiny-ImageNet split."""
    def __init__(self, hf_split, transform=None):
        self.hf = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        item = self.hf[idx]
        img_key = "image" if "image" in item else "img"
        img = item[img_key].convert("RGB")
        label = int(item["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label