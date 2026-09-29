from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset


class CIFAR100Dataset(Dataset):
    def __init__(
        self,
        normalization: str = "cifar100",
        loading: str = "torchvision",
        root_dir: str = "./data",
        img_size: int = 32,
        train_transforms: Optional[List[Dict[str, Any]]] = None,
        test_transforms: Optional[List[Dict[str, Any]]] = None,
        download: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.loading = loading
        self.normalization = normalization
        self.img_size = int(img_size)
        self.download = download

        # Normalization stats
        self.mean, self.std = self._set_normalization(normalization)

        # YAML-driven transforms
        self.train_transform = self._build_transform_pipeline(train_transforms, is_train=True)
        self.test_transform = self._build_transform_pipeline(test_transforms, is_train=False)


        # Normalized variants
        self.train_set = self._get_dataset(train=True, transform=self.train_transform)
        self.test_set = self._get_dataset(train=False, transform=self.test_transform)
        self.in_sample_set = self._get_dataset(train=True, transform=self.test_transform)


    # ---------------------------
    # Normalization setup
    # ---------------------------
    def _set_normalization(self, normalization: str):
        if normalization == "cifar100":
            mean = (0.5071, 0.4865, 0.4409)
            std = (0.2673, 0.2564, 0.2762)
        elif normalization == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError(f"Unknown normalization: {normalization}")
        return mean, std

    # ---------------------------
    # Transform factory
    # ---------------------------
    def _build_transform_pipeline(self, specs: Optional[List[Dict[str, Any]]], is_train: bool):
        """
        Build transforms.Compose from a list of transform specs.

        If specs is None or empty:
          - Train defaults to standard CIFAR crop/flip + normalize
          - Test defaults to ToTensor + normalize (plus Resize if img_size != 32)
        """
        if not specs:
            return self._default_transform(is_train=is_train)

        steps = []
        for spec in specs:
            if not spec:
                continue
            enabled = spec.get("enabled", True)
            if not enabled:
                continue

            name = spec.get("name")
            if not name:
                raise ValueError(f"Transform spec missing 'name': {spec}")

            params = spec.get("params", {}) or {}
            t = self._make_transform(name, params)
            steps.append(t)

        return transforms.Compose(steps)

    def _default_transform(self, is_train: bool):
        steps = []
        if not is_train:
            if self.img_size != 32:
                steps.append(transforms.Resize((self.img_size, self.img_size)))
            steps.extend([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            return transforms.Compose(steps)

        # train default
        steps.extend([
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        return transforms.Compose(steps)

    def _make_transform(self, name: str, params: Dict[str, Any]):
        name = name.strip()

        # --- Geometric / basic ---
        if name == "Resize":
            size = params.get("size", self.img_size)
            if isinstance(size, int):
                size = (size, size)
            return transforms.Resize(size=size)

        if name == "RandomCrop":
            size = params.get("size", self.img_size)
            padding = params.get("padding", 0)
            pad_if_needed = params.get("pad_if_needed", False)
            padding_mode = params.get("padding_mode", "constant")
            return transforms.RandomCrop(
                size=size,
                padding=padding,
                pad_if_needed=pad_if_needed,
                padding_mode=padding_mode,
            )

        if name == "RandomResizedCrop":
            size = params.get("size", self.img_size)
            scale = params.get("scale", (0.8, 1.0))
            ratio = params.get("ratio", (3 / 4, 4 / 3))
            interpolation = params.get("interpolation", transforms.InterpolationMode.BILINEAR)
            return transforms.RandomResizedCrop(
                size=size,
                scale=scale,
                ratio=ratio,
                interpolation=interpolation,
            )

        if name == "RandomHorizontalFlip":
            p = params.get("p", 0.5)
            return transforms.RandomHorizontalFlip(p=p)

        if name == "ColorJitter":
            return transforms.ColorJitter(
                brightness=params.get("brightness", 0.0),
                contrast=params.get("contrast", 0.0),
                saturation=params.get("saturation", 0.0),
                hue=params.get("hue", 0.0),
            )

        # --- Policy-based ---
        if name == "RandAugment":
            num_ops = params.get("num_ops", 2)
            magnitude = params.get("magnitude", 9)
            return transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)

        if name == "AutoAugment":
            policy = params.get("policy", "CIFAR10")
            policy_enum = getattr(transforms.AutoAugmentPolicy, policy)
            return transforms.AutoAugment(policy_enum)

        # --- Tensor & normalization ---
        if name == "ToTensor":
            return transforms.ToTensor()

        if name == "Normalize":
            mean = params.get("mean", self.mean)
            std = params.get("std", self.std)
            return transforms.Normalize(mean=mean, std=std)

        # --- Tensor-level regularization ---
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
    def _get_dataset(self, train: bool, transform, download: Optional[bool] = None):
        if download is None:
            download = self.download

        if self.loading == "torchvision":
            return CIFAR100(
                root=self.root_dir,
                train=train,
                transform=transform,
                download=download,
            )
        raise NotImplementedError(f"Unknown loading mode: {self.loading}")

    # ---------------------------
    # Subset selector
    # ---------------------------
    def subset(self, split: str, indices, clean: bool = False):
        split = split.lower().strip()

        if split == "train":
            base = self.in_sample_set if clean else self.train_set
        elif split == "test":
            base = self.test_set
        else:
            raise ValueError(f"Unknown split: {split}")

        if base is None:
            raise ValueError(
                f"Base dataset is None for split={split}, clean={clean}."
            )

        return Subset(base, indices)