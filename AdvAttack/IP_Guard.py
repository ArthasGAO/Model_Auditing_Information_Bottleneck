"""
Fingerprint Generation via Classification Boundary Points

Implements the optimization from Equation (5) of the paper:
    min_x ReLU(Z_i(x) - Z_j(x) + k) + ReLU(max_{t!=i,j} Z_t(x) - Z_i(x))

where:
    - i is the source label (initial predicted class of x before optimization)
    - j is the target label (the class x is pushed toward)
    - k controls the robustness-uniqueness tradeoff
    - Z_c(x) is the logit for class c

At convergence (objective = 0):
    - Term 1 satisfied: Z_j(x) >= Z_i(x) + k, so j has the highest logit
      among {i, j}. The model predicts class j (or is right on the boundary
      when k=0).
    - Term 2 satisfied: Z_i(x) >= max_{t!=i,j} Z_t(x), so no third class
      overtakes i. The competition is strictly between i and j.

    Therefore the optimized fingerprint point is predicted as class j by
    the target model, not class i. The point has crossed the boundary.

Ownership verification:
    Query both the victim and suspect models on fingerprint points and
    compute the matching rate (fraction of agreeing predictions). A stolen
    model shares the victim's boundary geometry and should agree on these
    boundary-sensitive points. An independent model has a different boundary
    and will disagree.

Initialization strategies:
    - T (Training): initialize x from a real training example predicted as class i
    - R (Random):   initialize x ~ Uniform[0, 1]^d

Target label selection:
    - R (Random):      randomly select j != i
    - L (Least-likely): j = argmin_{c != i} Z_c(x_init)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class IPGuardGenerator:
    """
    Generates fingerprint data points near the classification boundary
    of a target model by solving the optimization in Eq. (5).
    """

    def __init__(
        self,
        model,
        num_classes,
        k=0.0,
        max_iters=1000,
        lr=0.01,
        init_strategy="T",      # "T" = Training example, "R" = Random
        target_strategy="R",    # "R" = Random, "L" = Least-likely
        clip_min=0.0,
        clip_max=1.0,
        device="cuda",
        verbose=True,
    ):
        """
        Args:
            model:            the target model (should be NormalizedModel wrapping
                              a raw model, so it accepts [0,1] inputs)
            num_classes:      number of output classes
            k:                margin parameter balancing robustness vs uniqueness
            max_iters:        maximum optimization iterations per fingerprint point
            lr:               Adam learning rate
            init_strategy:    "T" (training example) or "R" (random)
            target_strategy:  "R" (random) or "L" (least-likely)
            clip_min:         minimum pixel value (0.0 for raw images)
            clip_max:         maximum pixel value (1.0 for raw images)
            device:           torch device
            verbose:          print progress info
        """
        self.model = model
        self.num_classes = num_classes
        self.k = k
        self.max_iters = max_iters
        self.lr = lr
        self.init_strategy = init_strategy.upper()
        self.target_strategy = target_strategy.upper()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device
        self.verbose = verbose

        assert self.init_strategy in ("T", "R"), \
            f"init_strategy must be 'T' or 'R', got '{self.init_strategy}'"
        assert self.target_strategy in ("R", "L"), \
            f"target_strategy must be 'R' or 'L', got '{self.target_strategy}'"

    def _compute_objective(self, logits, i, j):
        """
        Compute Eq. (5):
            ReLU(Z_i(x) - Z_j(x) + k) + ReLU(max_{t!=i,j} Z_t(x) - Z_i(x))

        Args:
            logits: (1, num_classes) or (B, num_classes) tensor
            i:      source label index (int or tensor)
            j:      target label index (int or tensor)

        Returns:
            scalar loss value
        """
        z_i = logits[:, i]  # logit of source class
        z_j = logits[:, j]  # logit of target class

        # Term 1: ReLU(Z_i - Z_j + k)
        # Zero when Z_j >= Z_i + k, i.e., target class j's logit exceeds
        # source class i's logit by at least k. At convergence the model
        # predicts class j (the point has crossed the boundary from i to j).
        term1 = F.relu(z_i - z_j + self.k)

        # Term 2: ReLU(max_{t != i,j} Z_t - Z_i)
        # Ensures class i remains the second-highest logit (after j),
        # so no third class interferes. At convergence the logit ranking
        # is: Z_j >= Z_i >= all others.
        mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
        mask[i] = False
        mask[j] = False
        if mask.any():
            z_others = logits[:, mask]
            max_other = z_others.max(dim=1).values
            term2 = F.relu(max_other - z_i)
        else:
            # Only 2 classes, no third class to worry about
            term2 = torch.zeros_like(term1)

        loss = (term1 + term2).mean()
        return loss

    def _select_target_label(self, logits, i):
        """
        Select target label j based on the chosen strategy.

        Args:
            logits: (1, num_classes) tensor from the model
            i:      source label index

        Returns:
            j: target label index (int)
        """
        if self.target_strategy == "R":
            # Random: pick any class != i
            candidates = [c for c in range(self.num_classes) if c != i]
            j = np.random.choice(candidates)
        elif self.target_strategy == "L":
            # Least-likely: class with smallest logit (excluding i)
            logits_np = logits.detach().cpu().numpy().flatten()
            logits_np[i] = float('inf')  # exclude source class
            j = int(np.argmin(logits_np))
        return j

    def _initialize_point(self, train_dataset, input_shape):
        """
        Initialize x and determine source label i.

        Args:
            train_dataset:  dataset of (image, label) pairs (raw [0,1] images)
                            Required if init_strategy == "T", can be None for "R"
            input_shape:    (C, H, W) shape for random initialization

        Returns:
            x_init: (1, C, H, W) tensor on device
            i:      source label (predicted class of x_init)
        """
        self.model.eval()

        if self.init_strategy == "T":
            # Training example: randomly pick a sample
            assert train_dataset is not None, \
                "Training dataset required for init_strategy='T'"
            idx = np.random.randint(len(train_dataset))
            x_init, _ = train_dataset[idx]
            if not isinstance(x_init, torch.Tensor):
                x_init = torch.tensor(x_init, dtype=torch.float32)
            x_init = x_init.unsqueeze(0).to(self.device)
        elif self.init_strategy == "R":
            # Random: uniform sample from [0, 1]^d
            x_init = torch.rand(1, *input_shape, device=self.device)

        # Determine source label i = predicted class
        with torch.no_grad():
            logits = self.model(x_init)
            i = logits.argmax(dim=1).item()

        return x_init, i, logits

    def generate_single(self, train_dataset=None, input_shape=(3, 32, 32)):
        """
        Generate a single fingerprint point near the classification boundary.

        The optimization starts from x predicted as class i, and pushes x
        across the boundary so that the model predicts class j at convergence.
        The resulting point sits in a boundary region specific to this model.

        Args:
            train_dataset:  dataset for initialization (needed if init_strategy="T")
            input_shape:    (C, H, W) for random init

        Returns:
            dict with keys:
                'point':        (C, H, W) tensor, the fingerprint point
                'source_label': int, class i (initial predicted class before opt)
                'target_label': int, class j (predicted class after opt, if converged)
                'final_loss':   float, final objective value
                'converged':    bool, whether objective reached 0
                'iters':        int, number of iterations used
        """
        self.model.eval()

        # Step 1: Initialize x and get source label i
        x_init, i, init_logits = self._initialize_point(train_dataset, input_shape)

        # Step 2: Select target label j
        j = self._select_target_label(init_logits, i)

        # Step 3: Optimize x using Adam
        # x is the optimization variable, so we need it to require grad
        x = x_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=self.lr)

        final_loss = float('inf')
        converged = False
        iters_used = 0

        for iteration in range(self.max_iters):
            optimizer.zero_grad()
            logits = self.model(x)
            loss = self._compute_objective(logits, i, j)

            current_loss = loss.item()
            iters_used = iteration + 1

            # Check convergence: objective == 0
            tol = 1e-6
            if current_loss <= tol:
                converged = True
                final_loss = 0.0
                break

            loss.backward()
            optimizer.step()

            # Project x back to valid range [clip_min, clip_max]
            with torch.no_grad():
                x.clamp_(self.clip_min, self.clip_max)

            final_loss = current_loss

        # Record the victim model's final prediction on the converged point
        with torch.no_grad():
            final_logits = self.model(x)
            final_pred = final_logits.argmax(dim=1).item()

        result = {
            'point': x.detach().squeeze(0).cpu(),
            'source_label': i,
            'target_label': j,
            'victim_pred': final_pred,     # victim's prediction on the optimized point
            'final_loss': final_loss,
            'converged': converged,
            'iters': iters_used,
        }

        return result

    def generate(self, n_points, train_dataset=None, input_shape=(3, 32, 32)):
        """
        Generate n fingerprint points with diverse (i, j) pairs.

        Args:
            n_points:       number of fingerprint points to generate
            train_dataset:  dataset for initialization (needed if init_strategy="T")
            input_shape:    (C, H, W)

        Returns:
            dict with keys:
                'points':        (n_points, C, H, W) tensor
                'source_labels': list of ints (class i, before optimization)
                'target_labels': list of ints (class j, optimization target)
                'victim_preds':  list of ints (victim's prediction on optimized point)
                'final_losses':  list of floats
                'converged':     list of bools
                'iters':         list of ints
        """
        all_points = []
        source_labels = []
        target_labels = []
        victim_preds = []
        final_losses = []
        converged_flags = []
        iters_list = []

        n_converged = 0

        for idx in range(n_points):
            result = self.generate_single(
                train_dataset=train_dataset,
                input_shape=input_shape,
            )

            all_points.append(result['point'])
            source_labels.append(result['source_label'])
            target_labels.append(result['target_label'])
            victim_preds.append(result['victim_pred'])
            final_losses.append(result['final_loss'])
            converged_flags.append(result['converged'])
            iters_list.append(result['iters'])

            if result['converged']:
                n_converged += 1

            if self.verbose and (idx + 1) % 10 == 0:
                print(
                    f"  [{idx+1}/{n_points}] "
                    f"converged={n_converged}/{idx+1} "
                    f"(i={result['source_label']}, j={result['target_label']}) "
                    f"loss={result['final_loss']:.6f} "
                    f"iters={result['iters']}"
                )

        points_tensor = torch.stack(all_points, dim=0)

        if self.verbose:
            print(f"\nFingerprint generation complete:")
            print(f"  Total: {n_points}, Converged: {n_converged} "
                  f"({100*n_converged/n_points:.1f}%)")
            avg_loss = np.mean(final_losses)
            print(f"  Avg final loss: {avg_loss:.6f}")

        return {
            'points': points_tensor,
            'source_labels': source_labels,
            'target_labels': target_labels,
            'victim_preds': victim_preds,
            'final_losses': final_losses,
            'converged': converged_flags,
            'iters': iters_list,
        }

    def save_fingerprints(self, fingerprint_data, save_path):
        """Save generated fingerprints to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fingerprint_data, save_path)
        if self.verbose:
            print(f"Fingerprints saved to {save_path}")

    @staticmethod
    def load_fingerprints(load_path):
        """Load previously generated fingerprints."""
        return torch.load(load_path, map_location='cpu', weights_only=False)


def verify_fingerprint(suspect_model, fingerprint_data, device='cuda', batch_size=256):
    """
    Verify model ownership by computing the matching rate on fingerprint points.

    Matching rate = fraction of fingerprint points where the suspect model's
    predicted label matches the victim (target) model's predicted label.

    These fingerprint points were optimized to sit on the victim's decision
    boundary, where predictions are most sensitive to boundary geometry.
    A stolen model inherited a similar boundary from the victim, so it should
    agree on most of these points (high matching rate). An independent model
    has its own boundary in different locations and will disagree on many
    points (low matching rate).

    This works in a black-box setting: only predicted labels are needed
    from the suspect model, no access to weights or gradients.

    Args:
        victim_model:     the owner's (target) model (NormalizedModel)
        suspect_model:    the suspect model to verify (NormalizedModel)
        fingerprint_data: dict from FingerprintGenerator.generate()
        device:           torch device

    Returns:
        dict with:
            'matching_rate':    fraction of points where both models agree
            'victim_preds':     victim's predicted labels on fingerprint points
            'suspect_preds':    suspect's predicted labels on fingerprint points
    """
    suspect_model.eval()

    points = fingerprint_data['points']
    target_labels = torch.as_tensor(fingerprint_data['victim_preds']).long()

    suspect_preds_all = []

    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size].to(device)
            logits = suspect_model(batch)
            suspect_preds_all.append(logits.argmax(dim=1).cpu())

    suspect_preds = torch.cat(suspect_preds_all, dim=0)
    agreement = (target_labels == suspect_preds).float().mean().item()

    return {
        "matching_rate": agreement,
        "target_labels": target_labels.numpy(),
        "suspect_preds": suspect_preds.numpy(),
    }


# =====================================================================
# Demo / Usage Example
# =====================================================================
if __name__ == "__main__":
    """
    Example usage with your existing pipeline components.
    Adjust paths and imports to match your project structure.
    """

    # ---- Pseudo-code showing integration with your pipeline ----
    #
    # from models.resnet import ResNet18
    # from utils.normalized_model import NormalizedModel
    # from data.cifar10_dataset import CIFAR10Dataset
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # # Load victim model
    # dataset_obj = CIFAR10Dataset(config)
    # victim_raw = ResNet18(num_classes=10)
    # victim_raw.load_state_dict(torch.load('path/to/victim.pth'))
    # victim_model = NormalizedModel(victim_raw, dataset_obj.mean, dataset_obj.std).to(device)
    #
    # # Get raw [0,1] training data for initialization
    # # (ToTensor only, no normalization or augmentation)
    # raw_train_set = dataset_obj.raw_train_set
    #
    # # Generate fingerprints with Training init + Random target (T, R)
    # gen_TR = FingerprintGenerator(
    #     model=victim_model,
    #     num_classes=10,
    #     k=0.0,           # on the boundary
    #     max_iters=1000,
    #     lr=0.01,
    #     init_strategy="T",
    #     target_strategy="R",
    #     device=device,
    # )
    # fp_TR = gen_TR.generate(n_points=100, train_dataset=raw_train_set)
    # gen_TR.save_fingerprints(fp_TR, './fingerprints/victim_TR_k0.pt')
    #
    # # Generate fingerprints with Training init + Least-likely target (T, L)
    # gen_TL = FingerprintGenerator(
    #     model=victim_model,
    #     num_classes=10,
    #     k=0.0,
    #     max_iters=1000,
    #     lr=0.01,
    #     init_strategy="T",
    #     target_strategy="L",
    #     device=device,
    # )
    # fp_TL = gen_TL.generate(n_points=100, train_dataset=raw_train_set)
    #
    # # Verify against a suspect model
    # suspect_raw = ResNet18(num_classes=10)
    # suspect_raw.load_state_dict(torch.load('path/to/suspect.pth'))
    # suspect_model = NormalizedModel(suspect_raw, dataset_obj.mean, dataset_obj.std).to(device)
    #
    # result = verify_fingerprint(victim_model, suspect_model, fp_TR, device=device)
    # print(f"Matching rate: {result['matching_rate']:.4f}")
    # # High matching rate → suspect likely derived from victim (stolen)
    # # Low matching rate  → suspect likely independent

    print("FingerprintGenerator ready for integration.")
    print("Supported configurations:")
    print("  Init:   T (Training example), R (Random)")
    print("  Target: R (Random label),     L (Least-likely label)")
    print("  Combinations: (T,R), (T,L), (R,R), (R,L)")