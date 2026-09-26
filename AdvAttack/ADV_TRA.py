# -*- coding: utf-8 -*-
"""
ADV-TRA: Adversarial-Trajectory Fingerprinting.

Implements the method from Xu et al., "United We Stand, Divided We Fall:
Fingerprinting Deep Neural Networks via Adversarial Trajectories" (NeurIPS 2024).

The four trajectory-generation stages (Section 4.1):
  1. Trajectory Initialization  (Eq. 2)
  2. Boundary Probing           (Eqs. 3-6)
  3. Trajectory Bilateralization (Eq. 7)
  4. Trajectory Connection -> Surface Trajectory (Eq. 8)

And the verification phase (Section 4.2):
  - r_mut: mutation rate (Eq. 9), a Hamming-style disagreement between suspect
    and source predictions along a trajectory.
  - Two-stage decision: per-trajectory threshold on r_mut, then detection-rate
    across all trajectories.

All trajectory operations run in raw [0, 1] pixel space. The source/suspect
models are expected to be wrapped with NormalizedModel so that normalization
happens inside the forward pass (same pattern as our RobD and IPGuard code).

Typical usage:

    from data.cifar10_dataset import CIFAR10Dataset
    from utils.normalized_model import NormalizedModel
    from models.resnet import ResNet18
    from Baseline.adv_tra import AdvTraPipeline

    dataset_obj = CIFAR10Dataset(config)

    victim_raw = ResNet18(num_classes=10)
    victim_raw.load_state_dict(torch.load(victim_ckpt))
    victim = NormalizedModel(victim_raw, dataset_obj.mean, dataset_obj.std).to(device)

    pipeline = AdvTraPipeline(source_model=victim, num_classes=10, device=device)

    # Extract fingerprint from the source model's raw training data
    fingerprint = pipeline.extract_fingerprint(
        base_dataset=dataset_obj.raw_train_set,
        base_indices=group_A[:200],           # 2x budget for skipped samples
    )
    pipeline.save_fingerprint('./results/advtra/victim_fp.pt')

    # Verify a suspect
    suspect_raw = ResNet18(num_classes=10)
    suspect_raw.load_state_dict(torch.load(suspect_ckpt))
    suspect = NormalizedModel(suspect_raw, dataset_obj.mean, dataset_obj.std).to(device)

    result = pipeline.verify_suspect(suspect)
    print(result)
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


# =====================================================================
# 1. Result container for verify_suspect()
# =====================================================================

@dataclass
class VerificationResult:
    """Returned by AdvTraPipeline.verify_suspect()."""
    detection_rate: float                  # fraction of trajectories with r_mut < threshold
    mean_mutation_rate: float              # average r_mut across trajectories
    mutation_rates: List[float] = field(default_factory=list)
    num_trajectories: int = 0
    threshold_mut: float = 0.5

    def __str__(self) -> str:
        return (
            f"VerificationResult(\n"
            f"  detection_rate      = {self.detection_rate:.4f}\n"
            f"  mean_mutation_rate  = {self.mean_mutation_rate:.4f}\n"
            f"  num_trajectories    = {self.num_trajectories}\n"
            f"  threshold_mut       = {self.threshold_mut}\n"
            f")"
        )


# =====================================================================
# 2. Low-level trajectory update (Eq. 2 in paper)
# =====================================================================

def _targeted_step(
    x: torch.Tensor,
    step_size: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    """One targeted FGSM-style update:  x_{i+1} = x_i - s_i * sign(grad).

    Clipping to [0, 1] is done by masking out gradient components that would
    push pixels out of range -- matches the reference implementation's
    `next_adv_sample` and keeps x valid without a separate clamp step.
    """
    sign_grad = grad.sign()
    proposed = x - step_size * sign_grad
    oob = ((proposed > 1.0) | (proposed < 0.0)).float()
    sign_grad = sign_grad * (1.0 - oob)
    return x - step_size * sign_grad


# =====================================================================
# 3. AdvTraPipeline: the main entry point
# =====================================================================

class AdvTraPipeline:
    """End-to-end ADV-TRA: fingerprint extraction + suspect verification.

    All hyperparameters use paper notation in the docs but are exposed here
    with plain names. The two length-related parameters are the ones most
    likely to surprise:
      - `bilateral_length` is 2l (paper), the full length of one T_bi.
      - `num_classes_traversed` is the number of classes the surface
        trajectory visits including the base class, so it builds
        (num_classes_traversed - 1) bilateral trajectories chained together.

    Paper defaults for CIFAR: bilateral_length=4 (l=2), num_classes_traversed=10.
    """

    # ----- construction -------------------------------------------------

    def __init__(
        self,
        source_model: nn.Module,
        num_classes: int,
        # Trajectory shape
        bilateral_length: int = 4,
        num_classes_traversed: int = 10,
        # Step-size optimization
        max_iterations: int = 300,
        initial_stepsize: float = 0.05,
        step_lr: float = 0.05,
        alpha_brk: float = 0.9,
        alpha_lc: float = 0.9,
        coef_brk: float = 1.0,
        coef_fwd: float = 1.0,
        warmup_fraction: float = 0.25,
        # Verification
        threshold_mut: float = 0.5,
        num_trajectories: int = 100,
        # Logistics
        device: str = "cuda",
        verbose: bool = True,
        seed: int = 0,
    ):
        assert bilateral_length % 2 == 0, "bilateral_length (2l) must be even"
        assert 0.0 < alpha_brk < 1.0, "alpha_brk must be in (0, 1)"
        assert 0.0 < alpha_lc < 1.0, "alpha_lc must be in (0, 1)"
        assert num_classes_traversed <= num_classes, (
            "num_classes_traversed cannot exceed num_classes"
        )

        self.source_model = source_model.eval()
        self.num_classes = num_classes

        self.bilateral_length = bilateral_length
        self.half_length = bilateral_length // 2        # == l in paper
        self.num_classes_traversed = num_classes_traversed

        self.max_iterations = max_iterations
        self.initial_stepsize = initial_stepsize
        self.step_lr = step_lr
        self.alpha_brk = alpha_brk
        self.alpha_lc = alpha_lc
        self.coef_brk = coef_brk
        self.coef_fwd = coef_fwd
        self.warmup_fraction = warmup_fraction

        self.threshold_mut = threshold_mut
        self.num_trajectories = num_trajectories

        self.device = device
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        # Populated by extract_fingerprint()
        self.fingerprint: Optional[List[dict]] = None

    # ----- public API ---------------------------------------------------

    def extract_fingerprint(
        self,
        base_dataset: Dataset,
        base_indices: Optional[Sequence[int]] = None,
    ) -> List[dict]:
        """Generate `num_trajectories` surface trajectories from the source model.

        Args:
            base_dataset: a torchvision-style dataset returning (image, label)
                pairs, where image is a raw [0, 1] tensor. Typically
                `dataset_obj.raw_train_set`.
            base_indices: optional list of indices into base_dataset. If None,
                uses the first `2 * num_trajectories` samples. We recommend
                supplying ~2x the target count to accommodate base samples
                the source misclassifies or for which boundary probing fails.

        Returns:
            A list of fingerprint records, one per successful trajectory:
                {
                    "trajectories": List[Tensor],  # bilateral trajectories
                    "predictions":  List[Tensor],  # source labels along each
                    "base_label":   int,           # ground-truth class of x_0
                }
            The list is also stored on the pipeline as `self.fingerprint`.
        """
        if base_indices is None:
            base_indices = list(range(min(2 * self.num_trajectories, len(base_dataset))))

        self.source_model.eval()
        fingerprint: List[dict] = []
        attempted = 0

        for idx in base_indices:
            if len(fingerprint) >= self.num_trajectories:
                break
            attempted += 1

            x0, y0 = self._load_sample(base_dataset, idx)

            # Skip if the source misclassifies the base sample: class c_0 is
            # then ambiguous and trajectory semantics break down.
            with torch.no_grad():
                if int(self.source_model(x0).argmax(dim=1).item()) != int(y0.item()):
                    if self.verbose:
                        print(f"  [skip idx={idx}] source misclassifies base sample")
                    continue

            record = self._build_surface_trajectory(x0, y0)
            if record is None:
                if self.verbose:
                    print(f"  [fail idx={idx}] trajectory generation did not converge")
                continue

            fingerprint.append(record)
            if self.verbose:
                print(
                    f"  [ok   idx={idx}] trajectory "
                    f"{len(fingerprint)}/{self.num_trajectories} built"
                )

        if self.verbose:
            rate = len(fingerprint) / max(attempted, 1)
            print(
                f"\nExtracted {len(fingerprint)} trajectories from "
                f"{attempted} base samples (success rate {rate:.2%})"
            )

        self.fingerprint = fingerprint
        return fingerprint

    def verify_suspect(
        self,
        suspect_model: nn.Module,
        batch_size: int = 256,
    ) -> VerificationResult:
        """Verify a suspect model against the extracted fingerprint.

        Two-stage decision (Section 4.2):
          1. Per-trajectory: r_mut < threshold_mut -> "matched"
          2. Final:          detection_rate = matched / total trajectories
        """
        if self.fingerprint is None:
            raise RuntimeError(
                "No fingerprint available. Call extract_fingerprint() first, "
                "or load_fingerprint()."
            )

        suspect_model.eval()
        mutation_rates: List[float] = []
        num_matched = 0

        for i, record in enumerate(self.fingerprint):
            r_mut = self._compute_mutation_rate(
                suspect_model,
                record["trajectories"],
                record["predictions"],
                batch_size=batch_size,
            )
            mutation_rates.append(r_mut)
            if r_mut < self.threshold_mut:
                num_matched += 1

            if self.verbose and (i + 1) % 10 == 0:
                print(
                    f"  verified {i + 1}/{len(self.fingerprint)} "
                    f"(running det. rate: {num_matched / (i + 1):.3f})"
                )

        result = VerificationResult(
            detection_rate=num_matched / len(self.fingerprint),
            mean_mutation_rate=float(np.mean(mutation_rates)),
            mutation_rates=mutation_rates,
            num_trajectories=len(self.fingerprint),
            threshold_mut=self.threshold_mut,
        )
        if self.verbose:
            print(f"\n{result}")
        return result

    def save_fingerprint(self, path: str) -> None:
        """Persist the fingerprint to disk (CPU tensors)."""
        if self.fingerprint is None:
            raise RuntimeError("Nothing to save; call extract_fingerprint() first.")
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        # Trajectories are already on CPU; ensure predictions are too.
        to_save = []
        for r in self.fingerprint:
            to_save.append({
                "trajectories": [t.cpu() for t in r["trajectories"]],
                "predictions": [p.cpu() for p in r["predictions"]],
                "base_label": r["base_label"],
            })
        torch.save(to_save, path)
        if self.verbose:
            print(f"Saved fingerprint ({len(to_save)} trajectories) to {path}")

    def load_fingerprint(self, path: str) -> List[dict]:
        """Load a previously-saved fingerprint."""
        self.fingerprint = torch.load(path, map_location="cpu", weights_only=False)
        if self.verbose:
            print(f"Loaded fingerprint ({len(self.fingerprint)} trajectories) from {path}")
        return self.fingerprint

    # ----- internal: trajectory construction ----------------------------

    def _load_sample(
        self,
        dataset: Dataset,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch (x, y) from dataset and move to device with correct shapes."""
        x, y = dataset[idx]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.unsqueeze(0).to(self.device)              # [1, C, H, W]
        y = torch.tensor([int(y)], device=self.device, dtype=torch.long)
        return x, y

    def _build_surface_trajectory(
        self,
        x0: torch.Tensor,
        y0: torch.Tensor,
    ) -> Optional[dict]:
        """Stage 4: chain (num_classes_traversed - 1) bilateral trajectories."""
        # Pick target classes, excluding the base class y0
        all_classes = np.arange(self.num_classes)
        candidates = np.delete(all_classes, int(y0.item()))
        chosen = self.rng.choice(
            candidates,
            size=self.num_classes_traversed - 1,
            replace=False,
        )

        traj_list: List[torch.Tensor] = []
        pred_list: List[torch.Tensor] = []
        current_x = x0.clone()

        for c in chosen:
            target = torch.tensor([int(c)], device=self.device, dtype=torch.long)

            # Stages 1 + 2: optimize step sizes so Eq. 3 holds
            step_sizes = self._probe_boundary(current_x, target)
            if step_sizes is None:
                return None

            # Stage 3: bilateralize
            bi_traj = self._bilateralize(current_x, target, step_sizes)

            # Record source-model predictions along the bilateral trajectory
            # (these become the reference labels used in verification)
            with torch.no_grad():
                bi_preds = self.source_model(bi_traj).argmax(dim=1)

            traj_list.append(bi_traj.detach().cpu())
            pred_list.append(bi_preds.detach().cpu())

            # Chain: next base sample = last point of this bilateral
            current_x = bi_traj[-1:].detach().clone()

        return {
            "trajectories": traj_list,
            "predictions": pred_list,
            "base_label": int(y0.item()),
        }

    def _forward_unilateral(
        self,
        x0: torch.Tensor,
        target: torch.Tensor,
        step_sizes: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the trajectory update (Eq. 2) for `len(step_sizes)` steps.

        Returns the full trajectory [x_0, ..., x_l] and the per-step gradients.
        """
        traj: List[torch.Tensor] = [x0.detach().clone()]
        grads: List[torch.Tensor] = []

        x = traj[0].clone()
        for i, s_i in enumerate(step_sizes):
            self.source_model.zero_grad(set_to_none=True)
            x.requires_grad_(True)

            logits = self.source_model(x)
            # Targeted cross-entropy: minimizing this drives prediction toward
            # `target`, matching Eq. 2's `-s_i * sign(grad L)`.
            loss = F.cross_entropy(logits, target)
            loss.backward()

            g = x.grad.detach().clone()
            grads.append(g)

            with torch.no_grad():
                x_next = _targeted_step(x.detach(), s_i.detach(), g)
            traj.append(x_next.clone())
            x = x_next

        return traj, grads

    def _check_boundary(
        self,
        traj: List[torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[int, int, List[int]]:
        """Check Eq. 3 on a unilateral trajectory.

        Returns (undershoot, overshoot, preds):
          - overshoot=1 if some earlier x_i (i<l) already hits target
          - undershoot=1 if the last x_l does NOT hit target (and no overshoot)
          - both zero ==> Eq. 3 satisfied
        """
        tgt = int(target.item())
        preds: List[int] = []
        with torch.no_grad():
            for x in traj[1:]:                          # exclude x_0
                preds.append(int(self.source_model(x).argmax(dim=1).item()))

        l = len(traj) - 1
        overshoot = 0
        undershoot = 0
        for i, p in enumerate(preds):
            if i < l - 1 and p == tgt:
                overshoot = 1
            if i == l - 1 and p != tgt and overshoot == 0:
                undershoot = 1
        return undershoot, overshoot, preds

    def _probe_boundary(
        self,
        x0: torch.Tensor,
        target: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        """Stages 1 + 2: alternate (i) gradient descent on step sizes via
        L_brk + L_fwd, and (ii) global rescaling by alpha_lc, until Eq. 3
        holds and step sizes are monotonically decreasing.

        Returns the optimized step sizes, or None if no valid config is found.
        """
        l = self.half_length
        step_sizes = [
            torch.tensor(self.initial_stepsize, device=self.device, dtype=torch.float32)
            for _ in range(l)
        ]

        best: Optional[List[torch.Tensor]] = None
        warmup_iters = int(self.warmup_fraction * self.max_iterations)

        for epoch in range(self.max_iterations):
            # (i) Build trajectory and check boundary condition ------------
            traj, _ = self._forward_unilateral(x0, target, step_sizes)
            undershoot, overshoot, _ = self._check_boundary(traj, target)
            eq3_ok = (undershoot == 0 and overshoot == 0)

            if eq3_ok:
                best = [s.detach().clone() for s in step_sizes]
                if epoch >= warmup_iters:
                    return best

            # (ii) Gradient descent on L_brk + L_fwd -----------------------
            s_grad = [s.detach().clone().requires_grad_(True) for s in step_sizes]
            loss = torch.tensor(0.0, device=self.device)

            # L_brk (Eq. 4): force s_{i+1} approx alpha_brk * s_i
            for i in range(l - 1):
                diff = self.alpha_brk * s_grad[i] - s_grad[i + 1]
                loss = loss + self.coef_brk * diff.pow(2)

            # L_fwd (Eq. 5): penalize any s_i < 0
            for i in range(l):
                if s_grad[i].item() < 0:
                    loss = loss - self.coef_fwd * s_grad[i]

            if loss.requires_grad and loss.item() != 0.0:
                loss.backward()
                with torch.no_grad():
                    for i in range(l):
                        g = s_grad[i].grad
                        if g is not None:
                            step_sizes[i] = (s_grad[i].detach() - self.step_lr * g).clone()
                        else:
                            step_sizes[i] = s_grad[i].detach().clone()
            else:
                for i in range(l):
                    step_sizes[i] = s_grad[i].detach().clone()

            # (iii) Length control: rescale all step sizes globally --------
            if overshoot == 1:
                scale = self.alpha_lc
            elif undershoot == 1:
                scale = 1.0 / self.alpha_lc
            else:
                scale = 1.0

            if scale != 1.0:
                with torch.no_grad():
                    for i in range(l):
                        step_sizes[i] = step_sizes[i] * scale

            # Safety check: diverged step sizes -> give up on this base sample
            if any(s.item() > 1.0 for s in step_sizes):
                return best

        return best

    def _bilateralize(
        self,
        x0: torch.Tensor,
        target: torch.Tensor,
        step_sizes: List[torch.Tensor],
    ) -> torch.Tensor:
        """Stage 3: run 2l steps using the concatenated sequence s + reverse(s),
        starting from x_0. Returns a tensor of shape [2l+1, C, H, W].
        """
        bilateral_steps = list(step_sizes) + list(reversed(step_sizes))
        traj, _ = self._forward_unilateral(x0, target, bilateral_steps)
        return torch.cat(traj, dim=0)

    # ----- internal: verification --------------------------------------

    def _compute_mutation_rate(
        self,
        suspect_model: nn.Module,
        traj_list: List[torch.Tensor],
        source_pred_list: List[torch.Tensor],
        batch_size: int = 256,
    ) -> float:
        """Eq. 9: fraction of trajectory points where suspect disagrees with source."""
        all_traj = torch.cat(traj_list, dim=0).to(self.device)
        all_src = torch.cat(source_pred_list, dim=0).to(self.device)

        preds = []
        with torch.no_grad():
            for start in range(0, all_traj.shape[0], batch_size):
                chunk = all_traj[start : start + batch_size]
                preds.append(suspect_model(chunk).argmax(dim=1))
        preds = torch.cat(preds, dim=0)

        return (preds != all_src).float().mean().item()


# =====================================================================
# 4. Example usage (commented out; drop into your own driver script)
# =====================================================================

if __name__ == "__main__":
    # Example wiring that mirrors the RobD / DatasetInference usage pattern.
    # Uncomment and adapt to your paths.
    #
    # import torch
    # from data.cifar10_dataset import CIFAR10Dataset
    # from utils.normalized_model import NormalizedModel
    # from models.resnet import ResNet18
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # # Build the dataset object (same one you use for RobD / IPGuard)
    # dataset_obj = CIFAR10Dataset(config)   # your existing config dict
    #
    # # Load the source (victim) model and wrap with NormalizedModel
    # victim_raw = ResNet18(num_classes=10)
    # victim_raw.load_state_dict(torch.load("./saved_models/.../victim.pth"))
    # victim = NormalizedModel(victim_raw, dataset_obj.mean, dataset_obj.std).to(device)
    #
    # # --- Extraction ---------------------------------------------------
    # pipeline = AdvTraPipeline(
    #     source_model=victim,
    #     num_classes=10,
    #     bilateral_length=4,          # 2l=4, l=2
    #     num_classes_traversed=10,    # -> 9 chained bilateral trajectories
    #     num_trajectories=100,
    #     device=device,
    #     seed=0,
    # )
    #
    # # Use the raw training set (ToTensor only, no augmentation) and the
    # # same group_A indices used for victim training.
    # import numpy as np
    # group_A = np.load("./saved_Indices/group_A_25000_seed42.npy")
    #
    # pipeline.extract_fingerprint(
    #     base_dataset=dataset_obj.raw_train_set,
    #     base_indices=group_A[:200],          # ~2x the trajectory budget
    # )
    # pipeline.save_fingerprint("./results/advtra/victim_fp.pt")
    #
    # # --- Verification -------------------------------------------------
    # suspect_raw = ResNet18(num_classes=10)
    # suspect_raw.load_state_dict(torch.load("./saved_models/.../suspect.pth"))
    # suspect = NormalizedModel(suspect_raw, dataset_obj.mean, dataset_obj.std).to(device)
    #
    # result = pipeline.verify_suspect(suspect)
    # print(result)
    print("AdvTraPipeline ready.")
    print("Components:")
    print("  - extract_fingerprint(base_dataset, base_indices): build N surface trajectories")
    print("  - verify_suspect(suspect_model): return VerificationResult")
    print("  - save_fingerprint(path) / load_fingerprint(path)")