from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


# =============================================================
# 1. Base class: torchvision CIFAR-10
# =============================================================
class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 wrapper with YAML-driven transform pipelines.

    Standard sets:
      - train_set:           train data, train_transforms (aug + normalize)
      - test_set:            test  data, test_transforms  (clean + normalize)
      - in_sample_set:       train data, test_transforms  (clean probing)

    Raw sets (for adversarial training and other [0,1]-pixel work):
      - raw_train_set:       train data, RandomCrop+HFlip+ToTensor (no normalize)
      - raw_train_clean_set: train data, ToTensor only (no aug, no normalize)
      - raw_test_set:        test  data, ToTensor only (no aug, no normalize)

    Why raw sets exist:
      Adversarial attacks like PGD must operate on raw [0,1] pixels because
      they clip against the input domain. If normalization is baked into the
      pipeline, PGD's clipping breaks. Raw sets feed unnormalized images;
      the model itself should wrap normalization (NormalizedModel pattern).
    """

    def __init__(
        self,
        normalization: str = "cifar10",
        loading: str = "torchvision",
        root_dir: str = "./data",
        img_size: int = 32,
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

        # Build transform pipelines.
        self.train_transform = self._build_transform_pipeline(train_transforms, is_train=True)
        self.test_transform = self._build_transform_pipeline(test_transforms, is_train=False)

        # Raw transforms for AT and other unnormalized workflows.
        self.raw_transform = transforms.Compose([ # this is the transform only without normalization. it applies for the clean data in [0, 1] space.
            transforms.ToTensor(),
        ])
        self.raw_train_transform = transforms.Compose([
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        self.train_set = None
        self.test_set = None
        # New: test images but using train augmentation.
        self.test_aug_set = None

        self.in_sample_set = None

        self.raw_train_set = None
        self.raw_train_clean_set = None
        self.raw_test_set = None

        if build_dataset:
            self._build_datasets()

    # ---------------------------
    # Build datasets
    # ---------------------------
    def _build_datasets(self):
        self.train_set = self.get_dataset(
            train=True, transform=self.train_transform, download=self.download
        )
        self.test_set = self.get_dataset(
            train=False, transform=self.test_transform, download=self.download
        )
        self.test_aug_set = self.get_dataset( # test data with train style data augmentation
                                train=False,
                                transform=self.train_transform,
                                download=self.download
                            )
        self.in_sample_set = self.get_dataset(# train data without data augmentation (clean)
            train=True, transform=self.test_transform, download=self.download
        )

        self.raw_train_set = self.get_dataset(# train data without normalization (still with augmentation);suitable for [0,1]
            train=True, transform=self.raw_train_transform, download=self.download
        )
        self.raw_train_clean_set = self.get_dataset(#train data only with to.Tensor; suitable for [0,1]
            train=True, transform=self.raw_transform, download=self.download
        )
        self.raw_test_set = self.get_dataset(#test data only with to.Tensor; suitable for [0,1]
            train=False, transform=self.raw_transform, download=self.download
        )

    # ---------------------------
    # Normalization
    # ---------------------------
    def set_normalization(self, normalization: str):
        if normalization == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif normalization == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError(f"Unknown normalization: {normalization}")
        return {"mean": mean, "std": std}

    # ---------------------------
    # Transform factory
    # ---------------------------
    def _build_transform_pipeline(self, specs: Optional[List[Dict[str, Any]]], is_train: bool):
        if not specs:
            return self._default_transform(is_train=is_train)

        steps = []
        for spec in specs:
            if spec is None:
                continue
            enabled = spec.get("enabled", True)
            if not enabled:
                continue

            name = spec.get("name", None)
            if not name:
                raise ValueError(f"Transform spec missing 'name': {spec}")

            params = spec.get("params", {}) or {}
            steps.append(self._make_transform(name, params))

        return transforms.Compose(steps)

    def _default_transform(self, is_train: bool):
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(self.img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            steps = []
            if self.img_size != 32:
                steps.append(transforms.Resize((self.img_size, self.img_size)))
            steps.extend([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            return transforms.Compose(steps)

    def _make_transform(self, name: str, params: Dict[str, Any]):
        name = name.strip()

        if name == "Resize":
            size = params.get("size", self.img_size)
            if isinstance(size, int):
                size = (size, size)
            return transforms.Resize(size=size)

        if name == "RandomCrop":
            return transforms.RandomCrop(
                size=params.get("size", self.img_size),
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

        if name == "RandAugment":
            return transforms.RandAugment(
                num_ops=params.get("num_ops", 2),
                magnitude=params.get("magnitude", 9),
            )

        if name == "AutoAugment":
            policy = params.get("policy", "CIFAR10")
            policy_enum = getattr(transforms.AutoAugmentPolicy, policy)
            return transforms.AutoAugment(policy_enum)

        if name == "ToTensor":
            return transforms.ToTensor()

        if name == "Normalize":
            return transforms.Normalize(
                mean=params.get("mean", self.mean),
                std=params.get("std", self.std),
            )

        if name == "RandomErasing":
            return transforms.RandomErasing(
                p=params.get("p", 0.25),
                scale=params.get("scale", (0.02, 0.33)),
                ratio=params.get("ratio", (0.3, 3.3)),
                value=params.get("value", 0),
                inplace=params.get("inplace", False),
            )

        raise NotImplementedError(
            f"Unknown transform name: '{name}'. Add it in _make_transform()."
        )

    # ---------------------------
    # Dataset retrieval
    # ---------------------------
    def get_dataset(self, train: bool, transform, download: bool = True):
        if self.loading == "torchvision":
            return CIFAR10(
                root=self.root_dir,
                train=train,
                transform=transform,
                download=download,
            )
        elif self.loading == "custom":
            raise NotImplementedError("Custom CIFAR-10 loader not implemented.")
        else:
            raise NotImplementedError(f"Unknown loading mode: {self.loading}")

    # ---------------------------
    # Subset selector
    # ---------------------------
    def subset(self, split: str, indices, clean: bool = False):
        """
        Create a torch.utils.data.Subset from one of the internal datasets.

        split:
          - "train":            train_set (aug+normalize) / in_sample_set if clean=True
          - "test":             test_set
          - "raw_train":        raw_train_set (aug, no normalize)
          - "raw_train_clean":  raw_train_clean_set (no aug, no normalize)
          - "raw_test":         raw_test_set
        """

        split = split.lower().strip()

        if split == "train":
            base = self.in_sample_set if clean else self.train_set
        elif split == "test":
            base = self.test_set if clean else self.test_aug_set
        elif split == "raw_train":
            base = self.raw_train_set
        elif split == "raw_train_clean":
            base = self.raw_train_clean_set
        elif split == "raw_test":
            base = self.raw_test_set
        else:
            raise ValueError(f"Unknown split: {split}")

        if base is None:
            raise ValueError(
                f"Base dataset is None for split={split}, clean={clean}. "
                f"This dataset variant may not provide that split."
            )

        return Subset(base, indices)


# =============================================================
# 2. Numpy-backed dataset (used by pseudo-labels)
# =============================================================
class NumpyImageDataset(Dataset):
    """
    CIFAR-compatible dataset backed by numpy arrays.
    """

    def __init__(self, X, Y, transform=None):
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert X.shape[0] == Y.shape[0]

        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]                # (H, W, C), uint8
        label = int(self.Y[idx])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# =============================================================
# 3. Pseudo-label CIFAR-10 (500K from 80M Tiny Images)
# =============================================================
class CIFAR10PseudoLabelDataset(CIFAR10Dataset):
    """
    Pseudo-labeled CIFAR-10 wrapper for numpy-array data.

    No test split is provided; the upstream pickle is purely train-only
    pseudo-labeled data.

    Sets that ARE provided:
      - train_set            (aug + normalize)
      - in_sample_set        (clean + normalize, for probing)
      - raw_train_set        (aug, no normalize, for AT)
      - raw_train_clean_set  (no aug, no normalize)

    Sets that are NOT provided:
      - test_set            -> None (no clean test labels exist)
      - raw_test_set        -> None
    """

    def __init__(
        self,
        normalization: str = "cifar10",
        img_size: int = 32,
        train_transforms: Optional[List[Dict[str, Any]]] = None,
        test_transforms: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            normalization=normalization,
            loading="custom",
            root_dir="./data",
            img_size=img_size,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            build_dataset=False,
            download=False,
        )

        self.X = None
        self.Y = None
        self.train_set = None
        self.test_set = None
        self.in_sample_set = None
        self.raw_train_set = None
        self.raw_train_clean_set = None
        self.raw_test_set = None

    def set_data(self, X: np.ndarray, Y: np.ndarray):
        """
        Attach numpy arrays and build the internal datasets.

        X: (N, H, W, 3) uint8
        Y: (N,)
        """

        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be numpy arrays.")
        if X.ndim != 4 or X.shape[-1] != 3:
            raise ValueError(f"X must have shape (N, H, W, 3). Got {X.shape}.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have same length. Got {X.shape[0]} vs {Y.shape[0]}."
            )
        if X.dtype != np.uint8:
            raise ValueError(f"X must be uint8 (0-255) for PIL. Got {X.dtype}.")

        self.X = X
        self.Y = Y

        # Normalized variants
        self.train_set = NumpyImageDataset(
            X=self.X, Y=self.Y, transform=self.train_transform
        )
        self.in_sample_set = NumpyImageDataset(
            X=self.X, Y=self.Y, transform=self.test_transform
        )

        # Raw variants for AT (and other [0,1]-pixel workflows).
        self.raw_train_set = NumpyImageDataset(
            X=self.X, Y=self.Y, transform=self.raw_train_transform
        )
        self.raw_train_clean_set = NumpyImageDataset(
            X=self.X, Y=self.Y, transform=self.raw_transform
        )

        # Pseudo-labeled data has no clean test split.
        self.test_set = None
        self.raw_test_set = None

        return self


# =============================================================
# 4. CIFARNet (HF, 190K from ImageNet)
# =============================================================
try:
    from datasets import load_dataset
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


class HFImageDataset(Dataset):
    """
    PyTorch Dataset wrapper for a Hugging Face image dataset.
    Forces a resize to match CIFAR-10 size and applies the parent transform.
    """

    def __init__(self, hf_dataset, transform=None, force_size=32):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.force_size = force_size

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        img_key = "img" if "img" in item else "image"
        img = item[img_key].convert("RGB")
        label = int(item["label"])

        if self.force_size and img.size != (self.force_size, self.force_size):
            img = img.resize((self.force_size, self.force_size), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class CIFARNetDataset(CIFAR10Dataset):
    """
    HF CIFARNet (ImageNet-derived) wrapper.

    Train: 190,000 images.
    Test:  10,000 images.

    All raw_* sets are populated:
      - raw_train_set        (aug, no normalize)
      - raw_train_clean_set  (no aug, no normalize)
      - raw_test_set         (no aug, no normalize)
    """

    def __init__(
        self,
        normalization: str = "cifar10",   # keep CIFAR-10 stats for the victim model
        img_size: int = 32,
        train_transforms: Optional[List[Dict[str, Any]]] = None,
        test_transforms: Optional[List[Dict[str, Any]]] = None,
    ):
        if not _HAS_HF:
            raise ImportError(
                "CIFARNetDataset requires `datasets`. "
                "Install via: pip install datasets"
            )

        super().__init__(
            normalization=normalization,
            loading="custom",
            root_dir="./data",
            img_size=img_size,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            build_dataset=False,
            download=False,
        )

        print("[INFO] Loading CIFARNet from Hugging Face...")
        hf_data = load_dataset("EleutherAI/cifarnet")

        # Normalized variants
        self.train_set = HFImageDataset(
            hf_data["train"],
            transform=self.train_transform,
            force_size=self.img_size,
        )
        self.test_set = HFImageDataset(
            hf_data["test"],
            transform=self.test_transform,
            force_size=self.img_size,
        )
        self.in_sample_set = HFImageDataset(
            hf_data["train"],
            transform=self.test_transform,
            force_size=self.img_size,
        )

        # Raw variants for AT and other [0,1]-pixel workflows.
        self.raw_train_set = HFImageDataset(
            hf_data["train"],
            transform=self.raw_train_transform,
            force_size=self.img_size,
        )
        self.raw_train_clean_set = HFImageDataset(
            hf_data["train"],
            transform=self.raw_transform,
            force_size=self.img_size,
        )
        self.raw_test_set = HFImageDataset(
            hf_data["test"],
            transform=self.raw_transform,
            force_size=self.img_size,
        )