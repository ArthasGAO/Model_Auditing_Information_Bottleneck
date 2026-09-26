"""DFMS-HL: Data-Free Model Stealing in a Hard-Label setting.

Sanyal, Addepalli, Babu. "Towards Data-Free Model Stealing in a Hard Label
Setting", CVPR 2022. Reference code: github.com/val-iisc/Hard-Label-Model-Stealing.

This module holds the attack's building blocks; main_dfms_extraction.py wires
them into the framework (victim zoo, group_B proxy pools, logging). Every
stage below corresponds to one script in the reference repo:

    train_dcgan            code/train_generator/dcgan.py                 (S1)
    make_val_set           code/train_student/generate_val_data.py       (S2, S6)
    train_clone_offline    code/train_student/train_student.py           (S3, S5)
    train_divgan           code/train_generator/train_gen.py             (S4)
    alternate_train        code/train_generator/train_generator_clone.py (S7)

Tensor spaces. The attack juggles three:
    pixel  [0, 1]      canonical in-memory form of every image here
    gan    [-1, 1]     what G emits (tanh) and what D consumes
    victim (x-m)/s     the victim's own normalisation; the clone lives here
                       too so its checkpoint drops into calculate_MI_* and
                       build_mea_matrices unchanged
The reference code uses a single (0.5, 0.5) normalisation for everything, so
its GAN space *is* its classifier space. Our victims were trained under the
dataset's mean/std, hence the explicit ImageSpace bridge.

Deliberate deviations from the reference code, all documented in
main_dfms_extraction.py: no EMA weight-averaging bookkeeping, warmup without
the reference's off-by-one, augmentation done on GPU tensors instead of PIL,
proxy labels queried once and cached, per-image AutoAugment kept.
"""
import math
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision.transforms import v2


# =============================================================================
# Proxy class lists (paper Sec. 4.1; reference dcgan.py hard-codes the same)
# =============================================================================
# CIFAR-100 superclasses with no CIFAR-10 counterpart: every animal superclass,
# both vehicle superclasses and "people" are left out.
CIFAR100_40_UNRELATED = [
    "orchid", "poppy", "rose", "sunflower", "tulip",                      # flowers
    "bottle", "bowl", "can", "cup", "plate",                              # food containers
    "apple", "mushroom", "orange", "pear", "sweet_pepper",                # fruit and vegetables
    "clock", "keyboard", "lamp", "telephone", "television",               # household electrical devices
    "bed", "chair", "couch", "table", "wardrobe",                         # household furniture
    "maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree",    # trees
    "bridge", "castle", "house", "road", "skyscraper",                    # large man-made outdoor things
    "cloud", "forest", "mountain", "plain", "sea",                        # large natural outdoor scenes
]
# The paper's "10 random classes", drawn from the 40 above.
CIFAR100_10_RANDOM = ["plate", "rose", "castle", "keyboard", "house",
                      "forest", "road", "television", "bottle", "wardrobe"]

CLASS_FILTERS = {
    "cifar100_40_unrelated": CIFAR100_40_UNRELATED,
    "cifar100_10_random": CIFAR100_10_RANDOM,
}


def resolve_class_filter(spec, class_to_idx):
    """Return a sorted list of class indices, or None for "all"."""
    if spec is None or (isinstance(spec, str) and spec.lower() == "all"):
        return None
    names = CLASS_FILTERS[spec] if isinstance(spec, str) else list(spec)
    missing = [n for n in names if n not in class_to_idx]
    if missing:
        raise KeyError(f"class filter names not in dataset: {missing}")
    return sorted(class_to_idx[n] for n in names)


# =============================================================================
# Tensor-space bridge and the black-box victim
# =============================================================================
class ImageSpace:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    def to_victim(self, x01):
        return (x01 - self.mean) / self.std

    @staticmethod
    def to_gan(x01):
        return x01 * 2.0 - 1.0

    @staticmethod
    def from_gan(g):
        return (g + 1.0) * 0.5


def logits_of(out):
    """timm DeiT models return a tuple in some modes; CNNs return a tensor."""
    return out[0] if isinstance(out, (tuple, list)) else out


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


class VictimOracle:
    """Hard-label black-box access. Every call through `hard_labels` is an
    attacker query and is counted; evaluation code must not go through it."""

    def __init__(self, net, device, batch_size=1000):
        self.net = net.eval()
        self.device = device
        self.batch_size = batch_size
        self.queries = 0

    @torch.no_grad()
    def hard_labels(self, x_victim):
        preds = []
        for chunk in x_victim.split(self.batch_size):
            out = logits_of(self.net(chunk.to(self.device, non_blocking=True)))
            preds.append(out.argmax(1))
        self.queries += int(x_victim.shape[0])
        return torch.cat(preds)


# =============================================================================
# Data: proxy bank, GPU augmentation, synthetic shapes
# =============================================================================
def pad_crop_flip(x, pad=4):
    """Per-sample RandomCrop(H, padding=pad) + RandomHorizontalFlip on a GPU
    batch in pixel space. Zero padding, matching torchvision's default used
    by every other training script in this repo."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    di = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    dj = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = di[:, None] + torch.arange(H, device=x.device)[None, :]
    cols = dj[:, None] + torch.arange(W, device=x.device)[None, :]
    b = torch.arange(B, device=x.device)[:, None, None, None]
    c = torch.arange(C, device=x.device)[None, :, None, None]
    out = xp[b, c, rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def to_pixel(x):
    return x.float().div(255.0) if x.dtype == torch.uint8 else x


class ProxyBank:
    """Unlabelled proxy images as uint8 on the GPU. Serves shuffled,
    augmented pixel-space batches without DataLoader workers (the reference
    code's RandomCrop+Flip on PIL is reproduced exactly by pad_crop_flip)."""

    def __init__(self, x_uint8, device):
        assert x_uint8.dtype == torch.uint8 and x_uint8.dim() == 4
        self.x = x_uint8.to(device)
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def num_batches(self, batch_size):
        return math.ceil(len(self) / batch_size)

    def batches(self, batch_size, augment=True):
        perm = torch.randperm(len(self), device=self.device)
        for s in range(0, len(self), batch_size):
            x = to_pixel(self.x[perm[s:s + batch_size]])
            yield pad_crop_flip(x) if augment else x


class BatchAutoAugment:
    """torchvision's CIFAR-10 AutoAugment policy, the same 25 sub-policies the
    reference auto_augment.py implements, applied per image (one sub-policy
    draw per image) on uint8 GPU tensors."""

    def __init__(self, policy="cifar10"):
        self.aa = v2.AutoAugment(getattr(v2.AutoAugmentPolicy, policy.upper()))

    def __call__(self, x01):
        x8 = (x01.clamp(0, 1) * 255).round().to(torch.uint8)
        out = torch.stack([self.aa(img) for img in x8])
        return out.float().div_(255.0)


@torch.no_grad()
def natural_proxy_uint8(dataset_obj, indices, batch_size=1000):
    """Clean (ToTensor-only) images of `indices` from the dataset wrapper's
    raw_train_clean_set, packed as uint8 N x 3 x H x W."""
    from torch.utils.data import DataLoader
    sub = dataset_obj.subset("raw_train_clean", list(indices))
    xs = []
    for x, _ in DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0):
        xs.append((x * 255).round().to(torch.uint8))
    return torch.cat(xs)


def generate_synthetic_shapes(n, num_shapes=50, min_size=5, max_size=10,
                              canvas=100, grey=True, seed=0, out_size=32):
    """The paper's synthetic proxy (Appendix), following generate_synthetic_data.py:
    `num_shapes` triangles / rectangles / circles / ellipses of random colour
    at random positions on a white canvas, background recoloured to one random
    RGB, 4x4 box blur, nearest-neighbour resize to 32x32, then grey-scale by
    copying the G channel. skimage.draw.random_shapes is replaced by PIL so
    the repo does not gain a dependency; shape statistics are the same."""
    rng = np.random.RandomState(seed)
    imgs = np.empty((n, out_size, out_size, 3), dtype=np.uint8)
    kinds = ["circle", "ellipse", "rectangle", "triangle"]
    for k in range(n):
        img = Image.new("RGB", (canvas, canvas), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for _ in range(num_shapes):
            kind = kinds[rng.randint(len(kinds))]
            colour = tuple(int(c) for c in rng.randint(0, 255, 3))   # 0..254, 255 = background
            w = int(rng.randint(min_size, max_size + 1))
            h = w if kind == "circle" else int(rng.randint(min_size, max_size + 1))
            x0 = int(rng.randint(0, canvas - w))
            y0 = int(rng.randint(0, canvas - h))
            if kind in ("circle", "ellipse"):
                draw.ellipse([x0, y0, x0 + w, y0 + h], fill=colour)
            elif kind == "rectangle":
                draw.rectangle([x0, y0, x0 + w, y0 + h], fill=colour)
            else:
                pts = [(x0 + int(rng.randint(0, w + 1)), y0 + int(rng.randint(0, h + 1))) for _ in range(3)]
                draw.polygon(pts, fill=colour)
        arr = np.array(img)
        bg = rng.randint(0, 255, 3)
        for c in range(3):                      # per-channel test, as in the reference
            ch = arr[..., c]
            ch[ch == 255] = bg[c]
        arr = cv2.blur(arr, (4, 4))
        arr = cv2.resize(arr, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        if grey:
            arr[..., 0] = arr[..., 1]
            arr[..., 2] = arr[..., 1]
        imgs[k] = arr
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()


# =============================================================================
# Metrics and losses
# =============================================================================
def balanced_agreement(pred, target, num_classes):
    """Mean per-class agreement (recall of the victim's label), classes with
    no samples ignored. This is the reference code's "val acc"."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    cm.index_put_((target.cpu().long(), pred.cpu().long()), torch.ones_like(target.cpu().long()),
                  accumulate=True)
    row = cm.sum(1)
    valid = row > 0
    per_class = cm.diag()[valid].float() / row[valid].float()
    return 100.0 * per_class.mean().item()


def label_entropy(counts):
    """Entropy (nats) of a class-count vector; ln(K) means perfectly uniform."""
    p = counts.float() / counts.sum().clamp(min=1)
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def weighted_prf(targets, preds):
    return {
        "train_precision": precision_score(targets, preds, average="weighted", zero_division=0),
        "train_recall": recall_score(targets, preds, average="weighted", zero_division=0),
        "train_f1": f1_score(targets, preds, average="weighted", zero_division=0),
    }


def diversity_loss(logits, temp=1.0, eps=1e-5):
    """L_class_div = sum_j a_j log a_j, a = batch-mean softmax (paper Eq. 3).
    Minimising it pushes the *batch* label distribution towards uniform."""
    p = F.softmax(logits / temp, dim=1)
    a = p.mean(0)
    return (a * torch.log(a + eps)).sum()


def sample_entropy_loss(logits, eps=1e-5):
    """Per-sample prediction entropy; the reference's `classification_loss`,
    weighted by c_l which every run script sets to 0."""
    p = F.softmax(logits, dim=1)
    return (-(p * torch.log(p + eps)).sum(1)).mean()


@torch.no_grad()
def predict(clone, x01, space, batch_size=1000):
    clone.eval()
    preds = []
    for chunk in x01.split(batch_size):
        chunk = to_pixel(chunk).to(space.mean.device, non_blocking=True).float()
        preds.append(logits_of(clone(space.to_victim(chunk))).argmax(1))
    return torch.cat(preds)


@torch.no_grad()
def sample_generator(G, n, nz, device, batch_size=500):
    """n pixel-space images from G in eval mode (BN running stats)."""
    G.eval()
    out = []
    for s in range(0, n, batch_size):
        b = min(batch_size, n - s)
        out.append(ImageSpace.from_gan(G(torch.randn(b, nz, 1, 1, device=device))))
    return torch.cat(out)


# =============================================================================
# S1  DCGAN pre-training on the proxy (dcgan.py, --train_with_teacher off)
# =============================================================================
def _gan_optims(G, D, lr, beta1):
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    return optG, optD


def train_dcgan(G, D, bank, space, epochs, batch_size, nz, lr, beta1, device,
                on_epoch, max_iters=None):
    bce = nn.BCELoss()
    optG, optD = _gan_optims(G, D, lr, beta1)
    for epoch in range(epochs):
        t0 = time.time()
        G.train(); D.train()
        agg = defaultdict(float); n = 0
        for it, real01 in enumerate(bank.batches(batch_size)):
            if max_iters is not None and it >= max_iters:
                break
            real = space.to_gan(real01)
            b = real.size(0)
            ones = torch.ones(b, device=device)
            zeros = torch.zeros(b, device=device)

            # (1) D: maximise log D(x) + log(1 - D(G(z)))
            D.zero_grad(set_to_none=True)
            out_r = D(real)
            errD_real = bce(out_r, ones)
            errD_real.backward()
            fake = G(torch.randn(b, nz, 1, 1, device=device))
            out_f = D(fake.detach())
            errD_fake = bce(out_f, zeros)
            errD_fake.backward()
            optD.step()

            # (2) G: maximise log D(G(z))
            G.zero_grad(set_to_none=True)
            out_g = D(fake)
            errG = bce(out_g, ones)
            errG.backward()
            optG.step()

            agg["loss_d_real"] += errD_real.item() * b
            agg["loss_d_fake"] += errD_fake.item() * b
            agg["loss_g_adv"] += errG.item() * b
            agg["D_x"] += out_r.mean().item() * b
            agg["D_G_z"] += out_g.mean().item() * b
            agg["acc_d"] += ((out_r > 0.5).sum() + (out_f <= 0.5).sum()).item() / 2.0
            n += b
        stats = {k: v / max(n, 1) for k, v in agg.items()}
        stats["epoch_time"] = time.time() - t0
        on_epoch(epoch, stats)


# =============================================================================
# S2 / S6  Victim-labelled generator samples used as the attacker's "val set"
# =============================================================================
def make_val_set(G, oracle, space, n, nz, device):
    x01 = sample_generator(G, n, nz, device)
    y = oracle.hard_labels(space.to_victim(x01))
    return x01.half().cpu(), y.cpu()


# =============================================================================
# S3 / S5  Offline clone training on victim-labelled proxy + generator samples
# =============================================================================
def train_clone_offline(clone, parts, batch_size, epochs, optimizer, scheduler,
                        space, device, on_epoch, grad_clip=None, max_iters=None):
    """`parts` is a list of {"x": images (uint8 or pixel float, on device),
    "y": victim labels, "ratio": r}. Each step concatenates int(batch_size*r)
    samples from every part; an epoch is min over parts of ceil(N/bs) steps,
    exactly the reference's lock-step iteration over separate loaders."""
    ce = nn.CrossEntropyLoss()
    sizes = [max(1, int(batch_size * p["ratio"])) for p in parts]
    steps = min(math.ceil(len(p["x"]) / bs) for p, bs in zip(parts, sizes))
    if max_iters is not None:
        steps = min(steps, max_iters)
    for epoch in range(epochs):
        t0 = time.time()
        clone.train()
        perms = [torch.randperm(len(p["x"]), device=device) for p in parts]
        loss_sum, correct, total = 0.0, 0, 0
        preds_all, targs_all = [], []
        for step in range(steps):
            xs, ys = [], []
            for p, bs, perm in zip(parts, sizes, perms):
                idx = perm[step * bs:(step + 1) * bs]
                xs.append(pad_crop_flip(to_pixel(p["x"][idx])))
                ys.append(p["y"][idx])
            x = torch.cat(xs)
            y = torch.cat(ys)
            optimizer.zero_grad(set_to_none=True)
            out = logits_of(clone(space.to_victim(x)))
            loss = ce(out, y)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(clone.parameters(), max_norm=grad_clip)
            optimizer.step()
            pred = out.argmax(1)
            loss_sum += loss.item() * y.size(0)
            correct += (pred == y).sum().item()
            total += y.size(0)
            preds_all.append(pred.cpu()); targs_all.append(y.cpu())
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        preds_np = torch.cat(preds_all).numpy(); targs_np = torch.cat(targs_all).numpy()
        stats = {"train_loss": loss_sum / max(total, 1), "train_acc": 100.0 * correct / max(total, 1),
                 "lr": lr, "epoch_time": time.time() - t0, "steps": steps}
        stats.update(weighted_prf(targs_np, preds_np))
        on_epoch(epoch, stats)


# =============================================================================
# S4  DivGAN: keep training G against the (frozen) clone's class-diversity
#     loss + the adversarial loss on proxy data (train_gen.py, c_l=0, d_l=10)
# =============================================================================
def train_divgan(G, D, clone, bank, space, epochs, batch_size, nz, lr, beta1,
                 lambda_div, lambda_cls, temp, num_classes, device, on_epoch, max_iters=None):
    bce = nn.BCELoss()
    optG, optD = _gan_optims(G, D, lr, beta1)
    clone.eval()
    set_requires_grad(clone, False)
    try:
        for epoch in range(epochs):
            t0 = time.time()
            G.train(); D.train()
            agg = defaultdict(float); n = 0
            counts = torch.zeros(num_classes, dtype=torch.long, device=device)
            for it, real01 in enumerate(bank.batches(batch_size)):
                if max_iters is not None and it >= max_iters:
                    break
                real = space.to_gan(real01)
                b = real.size(0)
                ones = torch.ones(b, device=device)
                zeros = torch.zeros(b, device=device)

                # (1) D step
                D.zero_grad(set_to_none=True)
                out_r = D(real); errD_real = bce(out_r, ones); errD_real.backward()
                fake = G(torch.randn(b, nz, 1, 1, device=device))
                out_f = D(fake.detach()); errD_fake = bce(out_f, zeros); errD_fake.backward()
                optD.step()

                # (2) G step: adversarial + lambda_div * class-diversity through the clone
                G.zero_grad(set_to_none=True)
                logits = logits_of(clone(space.to_victim(space.from_gan(fake))))
                div = diversity_loss(logits, temp)
                ent = sample_entropy_loss(logits)
                out_g = D(fake)
                adv = bce(out_g, ones)
                errG = adv + lambda_cls * ent + lambda_div * div
                errG.backward()
                optG.step()

                counts += torch.bincount(logits.argmax(1), minlength=num_classes)
                agg["loss_d_real"] += errD_real.item() * b
                agg["loss_d_fake"] += errD_fake.item() * b
                agg["loss_g_adv"] += adv.item() * b
                agg["loss_div"] += div.item() * b
                agg["loss_ent"] += ent.item() * b
                agg["D_x"] += out_r.mean().item() * b
                agg["D_G_z"] += out_g.mean().item() * b
                agg["acc_d"] += ((out_r > 0.5).sum() + (out_f <= 0.5).sum()).item() / 2.0
                n += b
            stats = {k: v / max(n, 1) for k, v in agg.items()}
            stats["label_entropy"] = label_entropy(counts)     # of the clone's argmax on fakes
            stats["epoch_time"] = time.time() - t0
            on_epoch(epoch, stats)
    finally:
        set_requires_grad(clone, True)


# =============================================================================
# S7  Alternating generator / clone training (train_generator_clone.py)
# =============================================================================
def alternate_train(G, D, clone, oracle, bank, space, epochs, batch_size, nz,
                    lr_gan, beta1, lambda_div, lambda_cls, temp, optC, schedC,
                    autoaug, num_classes, device, on_epoch, grad_clip=None, max_iters=None):
    """Per iteration, in the reference order:
        1. z -> G(z); AutoAugment; victim hard labels (the query); one CE step
           on the clone.
        2. G step on the *un-augmented* G(z): BCE(D(G(z)), 1)
           + lambda_div * class-diversity(clone) [+ lambda_cls * entropy].
           The clone is in eval mode and only provides gradients.
        3. D step: proxy batch as real, G(z).detach() as fake.
    D is freshly initialised by the caller, as in the reference (--netD is
    never passed); the clone starts from the S5 checkpoint."""
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    optG, optD = _gan_optims(G, D, lr_gan, beta1)
    for epoch in range(epochs):
        t0 = time.time()
        agg = defaultdict(float); n = 0
        correct = 0
        counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        preds_all, targs_all = [], []
        for it, real01 in enumerate(bank.batches(batch_size)):
            if max_iters is not None and it >= max_iters:
                break
            b = real01.size(0)
            ones = torch.ones(b, device=device)
            zeros = torch.zeros(b, device=device)

            # ---- 1. generate, augment, query, clone step
            G.train()
            G.zero_grad(set_to_none=True)
            fake = G(torch.randn(b, nz, 1, 1, device=device))
            x01 = space.from_gan(fake).detach()
            x_in = autoaug(x01) if autoaug is not None else x01
            x_v = space.to_victim(x_in)
            y = oracle.hard_labels(x_v)

            clone.train()
            set_requires_grad(clone, True)
            optC.zero_grad(set_to_none=True)
            out_c = logits_of(clone(x_v))
            loss_c = ce(out_c, y)
            loss_c.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(clone.parameters(), max_norm=grad_clip)
            optC.step()
            pred_c = out_c.argmax(1)
            correct += (pred_c == y).sum().item()
            counts += torch.bincount(y, minlength=num_classes)
            preds_all.append(pred_c.cpu()); targs_all.append(y.cpu())

            # ---- 2. G step through the frozen clone
            clone.eval()
            set_requires_grad(clone, False)
            D.train()
            logits = logits_of(clone(space.to_victim(space.from_gan(fake))))
            div = diversity_loss(logits, temp)
            ent = sample_entropy_loss(logits)
            out_g = D(fake)
            adv = bce(out_g, ones)
            errG = adv + lambda_cls * ent + lambda_div * div
            errG.backward()
            optG.step()

            # ---- 3. D step
            D.zero_grad(set_to_none=True)
            out_r = D(space.to_gan(real01)); errD_real = bce(out_r, ones); errD_real.backward()
            out_f = D(fake.detach()); errD_fake = bce(out_f, zeros); errD_fake.backward()
            optD.step()

            agg["train_loss"] += loss_c.item() * b
            agg["loss_d_real"] += errD_real.item() * b
            agg["loss_d_fake"] += errD_fake.item() * b
            agg["loss_g_adv"] += adv.item() * b
            agg["loss_div"] += div.item() * b
            agg["loss_ent"] += ent.item() * b
            agg["D_x"] += out_r.mean().item() * b
            agg["D_G_z"] += out_g.mean().item() * b
            agg["acc_d"] += ((out_r > 0.5).sum() + (out_f <= 0.5).sum()).item() / 2.0
            n += b
        set_requires_grad(clone, True)
        lr = optC.param_groups[0]["lr"]
        schedC.step()
        stats = {k: v / max(n, 1) for k, v in agg.items()}
        stats["train_acc"] = 100.0 * correct / max(n, 1)          # agreement with the victim on G(z)
        stats["label_entropy"] = label_entropy(counts)              # of the victim's labels on G(z)
        stats["lr"] = lr
        stats["epoch_time"] = time.time() - t0
        stats["steps"] = n // max(batch_size, 1) if n else 0
        if preds_all:
            stats.update(weighted_prf(torch.cat(targs_all).numpy(), torch.cat(preds_all).numpy()))
        on_epoch(epoch, stats)
