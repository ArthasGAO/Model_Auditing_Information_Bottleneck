from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


# =============================================================
# 1. Base class: torchvision CIFAR-10
# =============================================================
class CIFAR10Dataset(Dataset):
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

        self.train_set = None
        self.test_set = None
        self.in_sample_set = None


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
        self.in_sample_set = self.get_dataset(# train data without data augmentation (clean)
            train=True, transform=self.test_transform, download=self.download
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
        """

        split = split.lower().strip()

        if split == "train":
            base = self.in_sample_set if clean else self.train_set
        elif split == "test":
            base = self.test_set
        else:
            raise ValueError(f"Unknown split: {split}")

        if base is None:
            raise ValueError(
                f"Base dataset is None for split={split}, clean={clean}. "
                f"This dataset variant may not provide that split."
            )

        return Subset(base, indices)
