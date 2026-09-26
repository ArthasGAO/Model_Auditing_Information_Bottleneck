"""
Dataset Inference via Blind Walk
================================

Implements the black-box dataset inference pipeline from the paper:
    "Dataset Inference: Ownership Resolution in Machine Learning"

This version is aligned with the reference `rand_steps` implementation:

    * Directions are the RAW sampled noise (NOT sign()'d, NOT unit-normalized),
      so the three noise families stay genuinely distinct.
    * The walk advances in integer multiples of the base noise (delta_base * mag),
      mirroring "take k steps in the same direction".
    * The distance feature is the MATCHED L_p norm of the perturbation at the flip:
          uniform noise  -> L_inf
          gaussian noise -> L_2
          laplace noise  -> L_1
    * Default budget: 50 steps (100 for SVHN), reference noise scales
      uni=std=0.005, laplace-scale=0.01 (doubled for SVHN).

Pipeline:
    1. Blind Walk: black-box embedding generation (30-dim distance vector)
    2. Regressor g_V: 2-layer tanh network trained with margin-style loss.
       Inputs are standardized per-dimension (the three L_p blocks now live on
       very different scales, so standardization keeps any one block from
       dominating the tanh).
    3. Hypothesis testing: one-sided t-test to decide ownership.

Integration notes:
    - Expects a NormalizedModel wrapper (operates on raw [0,1] inputs)
    - Uses the same CIFAR10Dataset raw_train_set / raw_test_set pattern
"""

import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy import stats


# Maps each noise family to the L_p norm used to measure its distance.
# (uniform -> L_inf, gaussian -> L_2, laplace -> L_1), matching the paper.
_DIST_P = {"uniform": float("inf"), "gaussian": 2.0, "laplace": 1.0}
_DIST_ORDER = ["uniform", "gaussian", "laplace"]


# =====================================================================
# 1. Blind Walk — black-box embedding generation
# =====================================================================
class BlindWalk:
    """
    Generates distance-to-boundary embeddings by walking in random directions
    from an input until the model's prediction flips.

    For each of three noise families (uniform, gaussian, laplace), `n_samples`
    raw directions are drawn. A walk advances as `delta_base * mag` with
    mag = 1, 2, 3, ... and stops when the prediction flips (or at max_steps).
    The recorded feature is the MATCHED L_p norm of the final perturbation:
    L_inf for uniform, L_2 for gaussian, L_1 for laplace.

    Final embedding dimension: 3 * n_samples (default 30), laid out as
    [uniform block | gaussian block | laplace block].
    """

    def __init__(
        self,
        model,
        n_samples=10,
        noise_uniform=0.005,    # uniform noise scale  -> L_inf walks
        noise_gaussian=0.005,   # gaussian std          -> L_2 walks
        noise_laplace=0.01,     # laplace scale         -> L_1 walks
        max_steps=50,
        clip_min=0.0,
        clip_max=1.0,
        device="cuda",
        verbose=False,
    ):
        """
        Args:
            model:          NormalizedModel wrapping the classifier, accepts [0,1] inputs
            n_samples:      directions per noise family (default 10 -> 30-dim embedding)
            noise_uniform:  scale of U(-s, s) noise (paired with L_inf distance)
            noise_gaussian: std of N(0, s) noise     (paired with L_2 distance)
            noise_laplace:  scale of Laplace(0, s)    (paired with L_1 distance)
            max_steps:      max integer multiple of the base noise (50 / 100 for SVHN)
            clip_min/max:   pixel bounds (0 / 1 for raw images)
            device:         torch device
            verbose:        print progress
        """
        self.model = model
        self.n_samples = n_samples
        self.noise_uniform = noise_uniform
        self.noise_gaussian = noise_gaussian
        self.noise_laplace = noise_laplace
        self.max_steps = max_steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device
        self.verbose = verbose
        self.embedding_dim = 3 * n_samples  # uniform + gaussian + laplace

    # -----------------------------------------------------------------
    # Direction sampling — RAW noise, no sign, no normalization
    # -----------------------------------------------------------------
    def _sample_direction(self, shape, distribution):
        """
        Sample one raw noise direction. Magnitude and shape are preserved
        (this is what keeps the three families distinct). The matched L_p
        norm used later is determined by `distribution`, not here.
        """
        if distribution == "uniform":       # -> L_inf distance
            return torch.empty(shape, device=self.device).uniform_(
                -self.noise_uniform, self.noise_uniform
            )
        elif distribution == "gaussian":    # -> L_2 distance
            return torch.normal(
                0.0, self.noise_gaussian, size=tuple(shape), device=self.device
            )
        elif distribution == "laplace":     # -> L_1 distance
            # Laplace(0, b) via inverse CDF of U(-0.5, 0.5); matches np.random.laplace.
            u = torch.empty(shape, device=self.device).uniform_(-0.5 + 1e-6, 0.5 - 1e-6)
            return -self.noise_laplace * torch.sign(u) * torch.log1p(-2.0 * torch.abs(u))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    @staticmethod
    def _logits(out):
        """Return classification logits as a single tensor.

        DeiT/timm distilled models can return a (cls_logits, dist_logits) tuple
        (e.g. in distilled-training mode); CNNs return a plain tensor. In eval
        mode timm usually merges the two heads, but we unwrap defensively so the
        walk is architecture-agnostic (ResNet / VGG / DeiT).
        """
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    @staticmethod
    def _lp_norm(delta, p_per_walk):
        """
        Per-walk L_p norm of the perturbation, where p differs by walk.

        Args:
            delta:       (N, C, H, W) perturbations
            p_per_walk:  (N,) float tensor with values in {1.0, 2.0, inf}
        Returns:
            (N,) float tensor of matched-norm distances
        """
        flat = delta.flatten(start_dim=1)  # (N, D)
        out = torch.empty(flat.shape[0], device=flat.device, dtype=torch.float32)

        is_inf = torch.isinf(p_per_walk)
        if is_inf.any():
            out[is_inf] = flat[is_inf].abs().amax(dim=1)
        for p in (1.0, 2.0):
            m = (p_per_walk == p)
            if m.any():
                out[m] = flat[m].norm(p=p, dim=1)
        return out

    @torch.no_grad()
    def _batched_walk(self, x_starts, y_starts, deltas_base, p_per_walk):
        """
        Run many walks in parallel. Each walk has its own start point, label,
        raw direction, and matched norm. A walk advances as delta_base * mag
        (mag = 1..max_steps) and is frozen at the first mag where it flips.

        Args:
            x_starts:    (N, C, H, W) starting points (raw [0,1])
            y_starts:    (N,) integer ground-truth labels
            deltas_base: (N, C, H, W) RAW noise directions
            p_per_walk:  (N,) matched L_p order per walk (1.0 / 2.0 / inf)

        Returns:
            distances:  (N,) matched-L_p distance at flip (or at max_steps if no flip)
            flipped:    (N,) bool — True if the walk flipped before the cap
        """
        self.model.eval()
        N = x_starts.shape[0]

        # flip_mag[i] = first mag at which walk i flipped; 0 means "not yet flipped"
        flip_mag = torch.zeros(N, dtype=torch.int64, device=self.device)
        active = torch.ones(N, dtype=torch.bool, device=self.device)

        for mag in range(1, self.max_steps + 1):
            idx = active.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break

            # perturbation = delta_base * mag, clipped so x+delta stays in [clip_min, clip_max]
            delta = deltas_base[idx] * mag
            delta = torch.min(
                torch.max(delta, self.clip_min - x_starts[idx]),
                self.clip_max - x_starts[idx],
            )
            preds = self._logits(self.model(x_starts[idx] + delta)).argmax(dim=1)
            flipped_now = preds != y_starts[idx]

            newly = idx[flipped_now]
            flip_mag[newly] = mag
            active[newly] = False

        flipped = flip_mag > 0
        # walks that never flipped use the cap magnitude (max_steps)
        final_mag = torch.where(
            flipped, flip_mag, torch.full_like(flip_mag, self.max_steps)
        )

        # rebuild the final perturbation and measure its matched L_p norm
        delta_final = deltas_base * final_mag.view(-1, 1, 1, 1).float()
        delta_final = torch.min(
            torch.max(delta_final, self.clip_min - x_starts),
            self.clip_max - x_starts,
        )
        distances = self._lp_norm(delta_final, p_per_walk)
        return distances, flipped

    # -----------------------------------------------------------------
    # Helpers to build the (deltas, p_per_walk) for one point's 3 blocks
    # -----------------------------------------------------------------
    def _build_walks_for_point(self, shape_one):
        """Return (deltas, p_list) for one point: [uniform | gaussian | laplace]."""
        deltas_list, p_list = [], []
        for distribution in _DIST_ORDER:
            p = _DIST_P[distribution]
            for _ in range(self.n_samples):
                deltas_list.append(self._sample_direction(shape_one, distribution))
                p_list.append(p)
        return deltas_list, p_list

    def compute_embedding(self, x, y, return_per_dist=False):
        """
        Compute the 3*n_samples embedding for one point.

        Returns:
            embedding: (3 * n_samples,) numpy array of matched-L_p distances
            (optional) per_dist_stats: dict if return_per_dist=True
        """
        self.model.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)  # (1, C, H, W)

        total = 3 * self.n_samples
        x_rep = x.expand(total, -1, -1, -1).contiguous()
        y_rep = torch.full((total,), int(y), dtype=torch.long, device=self.device)

        deltas_list, p_list = self._build_walks_for_point(x.shape)  # x.shape = (1,C,H,W)
        deltas = torch.cat(deltas_list, dim=0)                       # (total, C, H, W)
        p_per_walk = torch.tensor(p_list, device=self.device)

        distances, _ = self._batched_walk(x_rep, y_rep, deltas, p_per_walk)
        distances_np = distances.cpu().numpy().astype(np.float32)

        if return_per_dist:
            per_dist_stats = {}
            n = self.n_samples
            for k, distribution in enumerate(_DIST_ORDER):
                slc = distances_np[k * n:(k + 1) * n]
                per_dist_stats[distribution] = {
                    "mean": float(slc.mean()),
                    "min": float(slc.min()),
                    "max": float(slc.max()),
                }
            return distances_np, per_dist_stats
        return distances_np

    def compute_embeddings(self, dataset, indices=None, max_points=None,
                           point_batch_size=100, print_every=50, tag=""):
        """
        Compute embeddings for a set of points using full batching.

        Returns:
            embeddings: (N, 3 * n_samples) numpy array, layout [U-block | G-block | L-block]
            labels:     (N,) numpy array of ground-truth labels
        """
        if indices is None:
            indices = list(range(len(dataset)))
        if max_points is not None:
            indices = indices[:max_points]

        n_total = len(indices)
        prefix = f"[{tag}] " if tag else ""
        walks_per_point = 3 * self.n_samples

        # per-point matched-norm pattern (same for every point), repeated per batch
        p_one = ([float("inf")] * self.n_samples
                 + [2.0] * self.n_samples
                 + [1.0] * self.n_samples)

        print(f"{prefix}Computing {n_total} embeddings "
              f"(n_samples={self.n_samples}, max_steps={self.max_steps}, "
              f"scales U/G/L={self.noise_uniform}/{self.noise_gaussian}/{self.noise_laplace}, "
              f"point_batch={point_batch_size})")
        print(f"{prefix}  -> {point_batch_size * walks_per_point} walks per forward pass")

        all_embeddings = np.zeros((n_total, walks_per_point), dtype=np.float32)
        all_labels     = np.zeros(n_total, dtype=np.int64)
        all_flipped    = np.zeros((n_total, walks_per_point), dtype=bool)

        t_start = time.time()
        t_last  = t_start
        processed = 0

        for batch_start in range(0, n_total, point_batch_size):
            batch_idx = indices[batch_start: batch_start + point_batch_size]
            B = len(batch_idx)

            xs, ys = [], []
            for idx in batch_idx:
                x, y = dataset[idx]
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                if isinstance(y, torch.Tensor):
                    y = y.item()
                xs.append(x)
                ys.append(int(y))

            x_batch = torch.stack(xs, dim=0).to(self.device)               # (B, C, H, W)
            y_batch = torch.tensor(ys, dtype=torch.long, device=self.device)
            shape_one = (1,) + tuple(x_batch.shape[1:])                     # (1, C, H, W)

            # Build B * walks_per_point raw directions + matched-norm orders.
            deltas_list, p_list = [], []
            for _ in range(B):
                d, p = self._build_walks_for_point(shape_one)
                deltas_list.extend(d)
                p_list.extend(p)
            deltas = torch.cat(deltas_list, dim=0)                          # (B*wpp, C, H, W)
            p_per_walk = torch.tensor(p_list, device=self.device)

            # replicate each point walks_per_point times so indices align
            x_rep = x_batch.repeat_interleave(walks_per_point, dim=0)
            y_rep = y_batch.repeat_interleave(walks_per_point, dim=0)

            distances, flipped = self._batched_walk(x_rep, y_rep, deltas, p_per_walk)

            emb_batch     = distances.view(B, walks_per_point).cpu().numpy()
            flipped_batch = flipped.view(B, walks_per_point).cpu().numpy()

            all_embeddings[batch_start: batch_start + B] = emb_batch
            all_labels    [batch_start: batch_start + B] = np.array(ys)
            all_flipped   [batch_start: batch_start + B] = flipped_batch
            processed += B

            if print_every > 0 and processed // print_every > (processed - B) // print_every:
                t_now   = time.time()
                rate    = B / (t_now - t_last) if (t_now - t_last) > 0 else 0.0
                eta     = (n_total - processed) / rate if rate > 0 else 0.0
                seen    = all_embeddings[:processed]
                n = self.n_samples
                mean_u = seen[:, 0:n].mean()
                mean_g = seen[:, n:2 * n].mean()
                mean_l = seen[:, 2 * n:3 * n].mean()
                print(f"{prefix}  [{processed}/{n_total}] "
                      f"elapsed={t_now - t_start:.1f}s  rate={rate:.1f} pts/s  ETA={eta:.1f}s  |  "
                      f"mean dist (Linf={mean_u:.4f}, L2={mean_g:.4f}, L1={mean_l:.4f})")
                t_last = t_now

        # ---------- Summary (per-block, since the three norms live on different scales) ----------
        t_total = time.time() - t_start
        n = self.n_samples
        flip_u = all_flipped[:, 0:n].mean()
        flip_g = all_flipped[:, n:2 * n].mean()
        flip_l = all_flipped[:, 2 * n:3 * n].mean()
        capped_frac = 1.0 - all_flipped.mean()

        print(f"\n{prefix}===== Summary =====")
        print(f"{prefix}Total wall time: {t_total:.1f}s "
              f"({n_total / max(t_total, 1e-9):.1f} points/s average)")
        print(f"{prefix}Per-block mean distance: "
              f"Linf={all_embeddings[:, 0:n].mean():.4f}  "
              f"L2={all_embeddings[:, n:2*n].mean():.4f}  "
              f"L1={all_embeddings[:, 2*n:3*n].mean():.4f}")
        print(f"{prefix}Per-block flip rate: "
              f"Linf={flip_u*100:.2f}%  L2={flip_g*100:.2f}%  L1={flip_l*100:.2f}%")
        print(f"{prefix}Overall did-not-flip fraction: {capped_frac*100:.2f}%")
        if capped_frac > 0.2:
            print(f"{prefix}  [WARNING] >20% of walks never flipped — consider raising the "
                  f"noise scales or max_steps so the signal isn't dominated by the cap.")

        return all_embeddings, all_labels


# =====================================================================
# 2. Confidence Regressor g_V
# =====================================================================
class ConfidenceRegressor(nn.Module):
    """
    Two-layer tanh network. Inputs are standardized per-dimension using buffers
    fixed at training time (the three L_p blocks span very different scales, so
    without this the L1 block would dominate and saturate the tanh). The buffers
    travel with state_dict, so train-time and verify-time normalization match.

    Training loss (per-sample): L(x, b) = -b * g_V(x)
        b = +1 for public points  -> push g_V large
        b = -1 for private points -> push g_V small
    At test time, smaller scores indicate "looks like private data."
    """

    def __init__(self, input_dim=30, hidden_dim=30):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def set_normalization(self, mean, std):
        with torch.no_grad():
            self.feat_mean.copy_(mean)
            self.feat_std.copy_(std.clamp_min(1e-8))

    def forward(self, x):
        x = (x - self.feat_mean) / self.feat_std
        h = torch.tanh(self.fc1(x))
        return self.fc2(h).squeeze(-1)  # (B,) scalar per sample


def margin_loss(scores, b):
    """Paper's loss: L(x, b) = -b * g_V(x), b in {-1 (private), +1 (public)}."""
    return (-b * scores).mean()


def train_regressor(
    embeddings_private,
    embeddings_public,
    input_dim=30,
    hidden_dim=30,
    epochs=100,
    lr=1e-3,
    batch_size=64,
    device="cuda",
    verbose=True,
):
    """Train g_V on victim-model embeddings from private and public sets."""
    X_priv = torch.tensor(embeddings_private, dtype=torch.float32)
    X_pub  = torch.tensor(embeddings_public,  dtype=torch.float32)
    b_priv = -torch.ones(len(X_priv))
    b_pub  = +torch.ones(len(X_pub))

    X = torch.cat([X_priv, X_pub], dim=0).to(device)
    b = torch.cat([b_priv, b_pub], dim=0).to(device)

    regressor = ConfidenceRegressor(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    # Fix standardization from the training embeddings (saved with the model).
    regressor.set_normalization(X.mean(dim=0), X.std(dim=0))

    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)

    n = len(X)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = margin_loss(regressor(X[idx]), b[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 10 == 0:
            regressor.eval()
            with torch.no_grad():
                s_priv = regressor(X_priv.to(device)).mean().item()
                s_pub  = regressor(X_pub.to(device)).mean().item()
            regressor.train()
            print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss / n_batches:.4f}  "
                  f"mean(g_V | private)={s_priv:.4f}  mean(g_V | public)={s_pub:.4f}")

    regressor.eval()
    return regressor


# =====================================================================
# 3. Hypothesis testing (dataset inference decision)
# =====================================================================
def hypothesis_test(scores_private, scores_public, alpha=0.05):
    """
    One-sided Welch t-test. Alternative: mean(public) > mean(private)
    (private gets smaller g_V scores -> consistent with theft).
    """
    mean_priv = float(np.mean(scores_private))
    mean_pub  = float(np.mean(scores_public))

    t_stat, p_two_sided = stats.ttest_ind(scores_public, scores_private, equal_var=False)
    if t_stat > 0:
        p_one_sided = p_two_sided / 2.0
    else:
        p_one_sided = 1.0 - (p_two_sided / 2.0)

    stolen = (p_one_sided < alpha) and (mean_pub > mean_priv)
    return {
        "stolen": bool(stolen),
        "p_value": float(p_one_sided),
        "mean_private": mean_priv,
        "mean_public": mean_pub,
        "delta": mean_pub - mean_priv,
        "t_stat": float(t_stat),
    }


# =====================================================================
# 4. End-to-end pipeline wrapper
# =====================================================================
class DatasetInferencePipeline:
    """
    Wraps Blind Walk embedding generation + g_V training + verification.
    Constructor stays callable as DatasetInferencePipeline(victim_model, device=...).
    """

    def __init__(
        self,
        victim_model,
        n_samples=10,
        noise_uniform=0.005,
        noise_gaussian=0.005,
        noise_laplace=0.01,
        max_steps=50,
        point_batch_size=100,
        device="cuda",
    ):
        self.victim_model = victim_model
        self.n_samples = n_samples
        self.noise_uniform = noise_uniform
        self.noise_gaussian = noise_gaussian
        self.noise_laplace = noise_laplace
        self.max_steps = max_steps
        self.point_batch_size = point_batch_size
        self.device = device
        self.embedding_dim = 3 * n_samples
        self.regressor = None

    def _make_blindwalk(self, model):
        return BlindWalk(
            model=model,
            n_samples=self.n_samples,
            noise_uniform=self.noise_uniform,
            noise_gaussian=self.noise_gaussian,
            noise_laplace=self.noise_laplace,
            max_steps=self.max_steps,
            device=self.device,
            verbose=True,
        )

    @staticmethod
    def _take(indices, dataset, n):
        if indices is not None:
            return list(indices)[:n]
        return list(range(min(n, len(dataset))))

    def train_regressor(
        self,
        private_dataset,
        public_dataset,
        private_indices=None,
        public_indices=None,
        n_train_samples=500,
        regressor_epochs=100,
        regressor_lr=1e-3,
    ):
        """Generate Blind Walk embeddings on the VICTIM model and train g_V (balanced)."""
        bw = self._make_blindwalk(self.victim_model)
        priv_idx = self._take(private_indices, private_dataset, n_train_samples)
        pub_idx  = self._take(public_indices,  public_dataset,  n_train_samples)

        print("==> Generating victim-model Blind Walk embeddings (private)...")
        emb_priv, _ = bw.compute_embeddings(
            private_dataset, indices=priv_idx, tag="victim/private",
            point_batch_size=self.point_batch_size,
        )
        print("==> Generating victim-model Blind Walk embeddings (public)...")
        emb_pub, _ = bw.compute_embeddings(
            public_dataset, indices=pub_idx, tag="victim/public",
            point_batch_size=self.point_batch_size,
        )

        print(f"  Private embeddings shape: {emb_priv.shape}")
        print(f"  Public  embeddings shape: {emb_pub.shape}")

        print("==> Training confidence regressor g_V...")
        self.regressor = train_regressor(
            embeddings_private=emb_priv,
            embeddings_public=emb_pub,
            input_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            epochs=regressor_epochs,
            lr=regressor_lr,
            device=self.device,
        )
        return self.regressor

    def verify_suspect(
        self,
        suspect_model,
        private_dataset,
        public_dataset,
        private_indices=None,
        public_indices=None,
        n_test_samples=200,
        alpha=0.05,
    ):
        """Run dataset inference on a suspect model (balanced private/public)."""
        assert self.regressor is not None, \
            "Call train_regressor() (or load_regressor()) before verify_suspect()."

        bw = self._make_blindwalk(suspect_model)
        priv_idx = self._take(private_indices, private_dataset, n_test_samples)
        pub_idx  = self._take(public_indices,  public_dataset,  n_test_samples)

        print("==> Generating suspect-model Blind Walk embeddings (private)...")
        emb_priv_sus, _ = bw.compute_embeddings(
            private_dataset, indices=priv_idx, tag="suspect/private",
            point_batch_size=self.point_batch_size,
        )
        print("==> Generating suspect-model Blind Walk embeddings (public)...")
        emb_pub_sus, _ = bw.compute_embeddings(
            public_dataset, indices=pub_idx, tag="suspect/public",
            point_batch_size=self.point_batch_size,
        )

        print(f"  Mean embedding (suspect/private): {emb_priv_sus.mean():.4f}")
        print(f"  Mean embedding (suspect/public):  {emb_pub_sus.mean():.4f}")
        print("  If STOLEN: private distances > public (private further from boundary)")

        self.regressor.eval()
        with torch.no_grad():
            s_priv = self.regressor(
                torch.tensor(emb_priv_sus, dtype=torch.float32, device=self.device)
            ).cpu().numpy()
            s_pub = self.regressor(
                torch.tensor(emb_pub_sus, dtype=torch.float32, device=self.device)
            ).cpu().numpy()

        result = hypothesis_test(s_priv, s_pub, alpha=alpha)
        result["scores_private"] = s_priv
        result["scores_public"]  = s_pub

        print(f"\n==> Verification result:")
        print(f"  mean(g_V | private) = {result['mean_private']:.4f}")
        print(f"  mean(g_V | public)  = {result['mean_public']:.4f}")
        print(f"  delta (pub - priv)  = {result['delta']:.4f}")
        print(f"  t-statistic         = {result['t_stat']:.4f}")
        print(f"  p-value (one-sided) = {result['p_value']:.6e}")
        print(f"  Decision @ alpha={alpha}: {'STOLEN' if result['stolen'] else 'inconclusive'}")
        return result

    def save_regressor(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.regressor.state_dict(),   # includes feat_mean / feat_std buffers
            "embedding_dim": self.embedding_dim,
            "n_samples": self.n_samples,
            "noise_uniform": self.noise_uniform,
            "noise_gaussian": self.noise_gaussian,
            "noise_laplace": self.noise_laplace,
            "max_steps": self.max_steps,
        }, save_path)
        print(f"Regressor saved to {save_path}")

    def load_regressor(self, load_path):
        ckpt = torch.load(load_path, map_location=self.device, weights_only=False)
        self.regressor = ConfidenceRegressor(
            input_dim=ckpt["embedding_dim"],
            hidden_dim=ckpt["embedding_dim"],
        ).to(self.device)
        self.regressor.load_state_dict(ckpt["state_dict"])
        self.regressor.eval()
        print(f"Regressor loaded from {load_path}")


if __name__ == "__main__":
    print("DatasetInferencePipeline ready (rand_steps-aligned).")
    print("  - BlindWalk: raw matched noise, integer-multiple walk, matched L_p distance")
    print("  - ConfidenceRegressor: 2-layer tanh with per-dim input standardization")
    print("  - hypothesis_test: one-sided Welch t-test for ownership")