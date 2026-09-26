"""
Paper-aligned variant of the self-implemented Dataset Inference (AdvAttack/DI.py).

Maini, Yaghini, Papernot, "Dataset Inference: Ownership Resolution in Machine
Learning", ICLR 2021. Exactly three changes relative to AdvAttack/DI.py, which
this module imports from and never modifies:

  1. Bounded confidence scores. g_V's output goes through tanh, so the loss
     L(x, b) = -b * g_V(x) (paper Sec. 6.2) can no longer be lowered just by
     scaling the weights. Same as the official notebook's final nn.Tanh().
     Everything else in g_V is unchanged: per-dimension input standardization,
     Linear(30, 30) - tanh - Linear(30, 1), Adam 1e-3, batch 64, same epochs.
  2. Disjoint images. g_V is trained on the victim's embeddings of one set of
     private / public images and every suspect is tested on a different,
     non-overlapping set (the caller passes both index lists; overlap raises).
  3. m-sample protocol (paper Sec. 6.2 / Table 1, official
     notebooks/utils.generate_table): reveal m samples per side, one-sided Welch
     t-test (public > private), repeat `reps` times, aggregate the p-values by
     their harmonic mean. The full-sample Welch test on all held-out images is
     still reported alongside, via the unchanged AdvAttack.DI.hypothesis_test.

The Blind Walk (BlindWalk: raw noise directions, integer-multiple walk, matched
L_p distance, 50-step cap) is the unchanged class from AdvAttack/DI.py.
"""
import numpy as np
import torch
from pathlib import Path
from scipy.stats import hmean, ttest_ind

from AdvAttack.DI import BlindWalk, ConfidenceRegressor, margin_loss, hypothesis_test

PROTOCOL_TAG = "paper_v1"   # stored in every saved regressor; loads are checked against it


# =====================================================================
# 1. g_V with a bounded (tanh) output
# =====================================================================
class ConfidenceRegressorTanh(ConfidenceRegressor):
    """ConfidenceRegressor (same layers, same parameter names) + tanh on the output."""

    def forward(self, x):
        return torch.tanh(super().forward(x))


def train_regressor_tanh(
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
    """Body identical to AdvAttack.DI.train_regressor except the regressor class."""
    X_priv = torch.tensor(embeddings_private, dtype=torch.float32)
    X_pub  = torch.tensor(embeddings_public,  dtype=torch.float32)
    b_priv = -torch.ones(len(X_priv))
    b_pub  = +torch.ones(len(X_pub))

    X = torch.cat([X_priv, X_pub], dim=0).to(device)
    b = torch.cat([b_priv, b_pub], dim=0).to(device)

    regressor = ConfidenceRegressorTanh(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
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
# 2. m-sample protocol (official generate_table semantics)
# =====================================================================
def m_sample_test(scores_private, scores_public, m=10, reps=100, generator=None):
    """Harmonic mean of `reps` one-sided Welch p-values on m revealed samples per side.

    Mirrors notebooks/utils.generate_table: each repetition draws
    positions = randperm(total)[:m] and tests public[positions] > private[positions]
    (the same positions index both arrays, as upstream does); the returned
    mean_diff is the average of (mean public - mean private) over repetitions.
    With a torch.Generator seeded s this equals generate_table after
    torch.manual_seed(s). scipy >= 1.9 returns 0 / nan from hmean instead of
    raising, so upstream's `except: hm = 1.0` is reproduced but rarely fires;
    `n_degenerate` counts draws whose p-value is 0 or nan (zero-variance
    samples, e.g. saturated tanh scores) so such cases are visible.
    """
    # Keep the input dtype (float32 scores): upstream runs ttest_ind on float32 arrays too.
    priv = np.asarray(scores_private)
    pub = np.asarray(scores_public)
    if priv.shape != pub.shape or priv.ndim != 1:
        raise ValueError("m_sample_test expects two 1-D score arrays of equal length.")
    total = priv.shape[0]
    if not 2 <= m <= total:
        raise ValueError(f"m={m} must lie in [2, {total}].")
    p_values, diffs = [], []
    for _ in range(reps):
        positions = torch.randperm(total, generator=generator)[:m].numpy()
        _, p = ttest_ind(pub[positions], priv[positions], alternative="greater", equal_var=False)
        p_values.append(float(p))
        diffs.append(pub[positions].mean() - priv[positions].mean())   # averaged as upstream does
    try:
        p_hm = float(hmean(p_values))
    except Exception:
        p_hm = 1.0
    n_degenerate = int(sum(1 for p in p_values if not np.isfinite(p) or p == 0.0))
    return {"p_value_m": p_hm, "mean_diff_m": float(np.mean(diffs)),
            "n_degenerate": n_degenerate, "p_values": p_values}


# =====================================================================
# 3. Pipeline with explicit, disjoint train / test index sets
# =====================================================================
def check_disjoint(train_idx, test_idx, what):
    overlap = set(map(int, train_idx)) & set(map(int, test_idx))
    if overlap:
        raise ValueError(f"{what}: {len(overlap)} image(s) used both to train g_V and to test "
                         f"(e.g. {sorted(overlap)[:5]}); the paper-aligned protocol forbids this.")


class PaperDIPipeline:
    """Blind Walk + tanh g_V + disjoint held-out test + m-sample protocol.

    Walk defaults are those of AdvAttack.DI.DatasetInferencePipeline.
    """

    def __init__(self, victim_model, n_samples=10, noise_uniform=0.005, noise_gaussian=0.005,
                 noise_laplace=0.01, max_steps=50, point_batch_size=100, device="cuda"):
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
        self.train_indices = None   # (private, public) the current g_V was trained on

    def _make_blindwalk(self, model):
        return BlindWalk(model=model, n_samples=self.n_samples,
                         noise_uniform=self.noise_uniform, noise_gaussian=self.noise_gaussian,
                         noise_laplace=self.noise_laplace, max_steps=self.max_steps,
                         device=self.device, verbose=True)

    def _embed(self, model, private_dataset, public_dataset, private_idx, public_idx, tag):
        bw = self._make_blindwalk(model)
        print(f"==> Blind Walk embeddings ({tag}, private)...")
        emb_priv, _ = bw.compute_embeddings(private_dataset, indices=list(private_idx),
                                            tag=f"{tag}/private", point_batch_size=self.point_batch_size)
        print(f"==> Blind Walk embeddings ({tag}, public)...")
        emb_pub, _ = bw.compute_embeddings(public_dataset, indices=list(public_idx),
                                           tag=f"{tag}/public", point_batch_size=self.point_batch_size)
        return emb_priv, emb_pub

    def train_regressor(self, private_dataset, public_dataset, private_idx, public_idx,
                        regressor_epochs=30, regressor_lr=1e-3):
        """Victim embeddings on the TRAIN index sets -> tanh g_V."""
        emb_priv, emb_pub = self._embed(self.victim_model, private_dataset, public_dataset,
                                        private_idx, public_idx, "victim")
        print(f"  Private embeddings shape: {emb_priv.shape}\n  Public  embeddings shape: {emb_pub.shape}")
        print("==> Training confidence regressor g_V (tanh output)...")
        self.regressor = train_regressor_tanh(
            embeddings_private=emb_priv, embeddings_public=emb_pub,
            input_dim=self.embedding_dim, hidden_dim=self.embedding_dim,
            epochs=regressor_epochs, lr=regressor_lr, device=self.device,
        )
        self.train_indices = (np.asarray(private_idx), np.asarray(public_idx))
        return self.regressor

    def verify_suspect(self, suspect_model, private_dataset, public_dataset, private_idx, public_idx,
                       alpha=0.05, m=10, reps=100, m_generator=None):
        """Suspect embeddings on the held-out TEST index sets -> full Welch test + m-sample test."""
        assert self.regressor is not None, "train_regressor() or load_regressor() first."
        if self.train_indices is None:
            raise ValueError("The g_V training index sets are unknown; cannot check disjointness.")
        check_disjoint(self.train_indices[0], private_idx, "private")
        check_disjoint(self.train_indices[1], public_idx, "public")

        emb_priv, emb_pub = self._embed(suspect_model, private_dataset, public_dataset,
                                        private_idx, public_idx, "suspect")
        self.regressor.eval()
        with torch.no_grad():
            s_priv = self.regressor(torch.tensor(emb_priv, dtype=torch.float32, device=self.device)).cpu().numpy()
            s_pub = self.regressor(torch.tensor(emb_pub, dtype=torch.float32, device=self.device)).cpu().numpy()

        result = hypothesis_test(s_priv, s_pub, alpha=alpha)          # full held-out sample
        mres = m_sample_test(s_priv, s_pub, m=m, reps=reps, generator=m_generator)
        result.update({
            "p_value_m": mres["p_value_m"], "mean_diff_m": mres["mean_diff_m"],
            "n_degenerate_m": mres["n_degenerate"],
            "stolen_m": bool(mres["p_value_m"] < alpha),
            "saturated_frac": float(np.mean(np.abs(np.concatenate([s_priv, s_pub])) > 0.999)),
            "scores_private": s_priv, "scores_public": s_pub,
        })
        print(f"  full Welch: delta={result['delta']:.4f} p={result['p_value']:.3e} | "
              f"m={m} x {reps}: mean_diff={mres['mean_diff_m']:.4f} p_hm={mres['p_value_m']:.3e}")
        return result

    def save_regressor(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.regressor.state_dict(),
            "head": "tanh", "protocol": PROTOCOL_TAG,
            "train_private_indices": self.train_indices[0].tolist(),
            "train_public_indices": self.train_indices[1].tolist(),
            "embedding_dim": self.embedding_dim, "n_samples": self.n_samples,
            "noise_uniform": self.noise_uniform, "noise_gaussian": self.noise_gaussian,
            "noise_laplace": self.noise_laplace, "max_steps": self.max_steps,
        }, save_path)

    def load_regressor(self, load_path):
        ckpt = torch.load(load_path, map_location=self.device, weights_only=False)
        if ckpt.get("head") != "tanh" or ckpt.get("protocol") != PROTOCOL_TAG:
            raise ValueError(f"{load_path} is not a {PROTOCOL_TAG} tanh regressor "
                             "(a DI.py regressor has the same keys but no output tanh).")
        self.regressor = ConfidenceRegressorTanh(ckpt["embedding_dim"], ckpt["embedding_dim"]).to(self.device)
        self.regressor.load_state_dict(ckpt["state_dict"])
        self.regressor.eval()
        self.train_indices = (np.asarray(ckpt["train_private_indices"]),
                              np.asarray(ckpt["train_public_indices"]))
