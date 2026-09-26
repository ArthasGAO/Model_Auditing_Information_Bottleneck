from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import SVHN


class _SVHNLabelRemapWrapper(Dataset):
    def __init__(self, base_dataset, remap_zero=True):
        self.base = base_dataset
        self.remap_zero = remap_zero

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = int(y)
        if self.remap_zero and y == 10:
            y = 0
        return x, y


class SVHNDataset(Dataset):
    def __init__(
        self,
        normalization="svhn",          # "svhn" | "imagenet"
        loading="torchvision",
        root_dir="./data",
        build_dataset=True,
        use_extra=False,
        remap_zero=True,

        # augmentation toggles
        random_crop=True,
        crop_padding=4,
        horizontal_flip=False,

        # NEW: color jitter
        use_color_jitter=False,
        cj_brightness=0.2,
        cj_contrast=0.2,
        cj_saturation=0.2,
        cj_hue=0.05,
    ):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.in_sample_set = None

        self.root_dir = root_dir
        self.loading = loading
        self.normalization = normalization
        self.use_extra = bool(use_extra)
        self.remap_zero = bool(remap_zero)

        stats = self.set_normalization(normalization)
        self.mean = stats["mean"]
        self.std = stats["std"]

        # ===== Transforms (mimic CIFAR style) =====
        train_tf = []

        if random_crop:
            train_tf.append(transforms.RandomCrop(32, padding=crop_padding))

        # NEW: ColorJitter (apply on PIL image, so keep it before ToTensor)
        if use_color_jitter:
            train_tf.append(
                transforms.ColorJitter(
                    brightness=cj_brightness,
                    contrast=cj_contrast,
                    saturation=cj_saturation,
                    hue=cj_hue,
                )
            )

        if horizontal_flip:
            train_tf.append(transforms.RandomHorizontalFlip())

        train_tf += [
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]
        self.train_transfroms = transforms.Compose(train_tf)

        self.test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        if build_dataset:
            self._build_datasets()

    def _build_datasets(self):
        train_base = self.get_dataset(split="train", transform=self.train_transfroms)
        train_wrapped = _SVHNLabelRemapWrapper(train_base, remap_zero=self.remap_zero)

        if self.use_extra:
            extra_base = self.get_dataset(split="extra", transform=self.train_transfroms)
            extra_wrapped = _SVHNLabelRemapWrapper(extra_base, remap_zero=self.remap_zero)
            from torch.utils.data import ConcatDataset
            self.train_set = ConcatDataset([train_wrapped, extra_wrapped])
        else:
            self.train_set = train_wrapped

        test_base = self.get_dataset(split="test", transform=self.test_transforms)
        self.test_set = _SVHNLabelRemapWrapper(test_base, remap_zero=self.remap_zero)

        in_base = self.get_dataset(split="train", transform=self.test_transforms)
        in_wrapped = _SVHNLabelRemapWrapper(in_base, remap_zero=self.remap_zero)

        if self.use_extra:
            extra_in_base = self.get_dataset(split="extra", transform=self.test_transforms)
            extra_in_wrapped = _SVHNLabelRemapWrapper(extra_in_base, remap_zero=self.remap_zero)
            from torch.utils.data import ConcatDataset
            self.in_sample_set = ConcatDataset([in_wrapped, extra_in_wrapped])
        else:
            self.in_sample_set = in_wrapped

    def set_normalization(self, normalization):
        if normalization == "svhn":
            mean = (0.4377, 0.4438, 0.4728)
            std = (0.1980, 0.2010, 0.1970)
        elif normalization == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError(f"Unknown normalization: {normalization}")
        return {"mean": mean, "std": std}

    def get_dataset(self, split, transform, download=True):
        if self.loading == "torchvision":
            dataset = SVHN(
                root=self.root_dir,
                split=split,
                transform=transform,
                download=download,
            )
        elif self.loading == "custom":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return dataset
