"""
AdvAttack/boundary_band.py

Phase-1 decision-boundary distance gate ("band membership").

Given a served model and a query x0, decide whether x0 lies within distance d of
the nearest decision boundary (the "band") and flag it as a potential fingerprint
probe. This is a READ-ONLY scalar test: x0 is never perturbed, projected, or
clamped. We estimate the input-space distance to the boundary with a closed-form
first-order (DeepFool-style) estimate, optionally curvature-corrected.

COORDINATE SYSTEM (Option A — fixed for all call sites)
--------------------------------------------------------
Differentiate through NormalizedModel with raw [0,1] inputs. Normalization is the
model's first internal layer (NOT a data-pipeline transform), so autograd folds the
1/std factor into grad_x g by the chain rule and the distance -- hence the threshold
d -- is in RAW-PIXEL L2 units. This matches the space PGD / DeepJudge / IPGuard use.

The estimator itself is coordinate-agnostic: it differentiates w.r.t. whatever
(x, model) it is handed. Callers (see main_boundary_band_eval.py) are responsible
for passing a NormalizedModel and raw [0,1] tensors. Do NOT feed pre-normalized
tensors and do NOT measure d in normalized units.

NO CLAMPING / NO BOX CONSTRAINT anywhere in this file: delta is evaluated AT x0, the
point is never moved, so there is no perturbed image to project onto [0,1]. We
estimate a geometric distance to the boundary surface (defined over all of R^D), not
a realizable adversarial image. (The one exception is the internal finite-difference
probe used for curvature, which is likewise left unclamped -- see second_order_query.)

Two-phase structure (mirrors the MI pipeline's collect / summarize split):
  * collect_boundary_stats(...)  -- ONE expensive pass; caches per-query
                                    (margin, ||grad g||, j*, [kappa]) as tensors.
  * collect_boundary_stats_batched(...) -- same pass, GPU-batched over queries:
                                    1 forward + (1+K) backwards per BATCH instead
                                    of per query. Same schema / same numbers up to
                                    kernel-order float noise.
  * posthoc_verify_batch(...)    -- Eq. (13) check of the constraints dropped in
                                    the first-order relaxation, evaluated at the
                                    candidate point x_hat = x0 + r^(1).
  * band_from_stats(stats, d)    -- CHEAP sweep over d on cached stats, so
                                    re-tuning d or comparing first vs second order
                                    never re-runs gradients.
  * calibrate_d(stats, pct)      -- pick d from a low percentile of the benign Delta.
"""

import math
import torch
import numpy as np


# Small floor on the gradient norm to guard the ||grad g|| -> 0 ill-conditioned case.
EPS_GRAD = 1e-12


# =====================================================================
# Core autograd primitive: DeepFool-style per-class input gradients.
# This is the reused pattern from attack/deepfool.py::_construct_jacobian;
# _input_grads is the subset variant (only backprop the classes we need,
# which is what makes top-K cheap on CIFAR-100 / TinyImageNet).
# =====================================================================
def _as_logits(out):
    """Collapse a (logits, features) / (cls, dist) tuple to the logits tensor."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _input_grads(logits, x, class_ids):
    """Input-space gradients d(logit_c)/dx for each c in `class_ids`.

    `logits` is the length-C logit vector for a single query (still attached to the
    graph whose leaf is `x`). Returns a tensor of shape (len(class_ids), *x.shape).

    This is inherently a per-class backward loop -- it is NOT batched away, exactly
    as in the DeepFool reference implementation.
    """
    grads = []
    n = len(class_ids)
    for k, c in enumerate(class_ids):
        grad = torch.autograd.grad(
            logits[c],
            x,
            retain_graph=(k + 1 < n),
            create_graph=False,
            only_inputs=True,
        )[0]
        grads.append(grad.detach())
    return torch.stack(grads)


def _construct_jacobian(logits, x):
    """Faithful equivalent of attack/deepfool.py::_construct_jacobian: the full
    all-class input Jacobian. Kept for reuse / parity; the estimator uses the
    subset primitive `_input_grads` so top-K only pays for the classes it needs."""
    return _input_grads(logits, x, list(range(int(logits.shape[0]))))


def _margin_value_and_grad(model, x_point, y, j):
    """Return (g, grad_x g) for the pairwise margin g = z_y - z_j at `x_point`.

    Fresh forward+backward at the given point (used by the curvature probe). No
    argmax is recomputed: y and j are held fixed to the reference query's choice.
    """
    x = x_point.clone().detach().requires_grad_(True)
    logits = _as_logits(model(x))[0]          # (C,)  -- batch dim is 1
    g = logits[y] - logits[j]
    grad = torch.autograd.grad(g, x)[0]
    return g.detach(), grad.detach()


# =====================================================================
# Per-query first-order estimate.
# =====================================================================
def first_order_query(model, x0, topk=None, eps_grad=EPS_GRAD):
    """One DeepFool-style first-order band estimate for a single query x0.

    Args:
        model:    NormalizedModel (accepts raw [0,1]); logits, not softmax.
        x0:       (1, C, H, W) or (C, H, W) raw [0,1] tensor on the model's device.
        topk:     number of competing classes to score (nearest competitors by
                  logit). None or >= C-1 => evaluate all competitors (exact for
                  CIFAR-10). Use a small K for large-class models.
        eps_grad: floor on ||grad g|| to guard the ill-conditioned case.

    Returns a dict:
        y            predicted class y0 = argmax_c f(x0)_c              (int)
        j_star       nearest competitor (argmin_j delta1_j)            (int)
        delta1       Delta^(1) = min_j delta1_j  (raw-pixel L2)        (float)
        margin_star  g_{y,j*}(x0) >= 0                                 (float)
        gradnorm_star ||grad g_{y,j*}(x0)||_2                          (float)
        grad_star    grad_x g_{y,j*}(x0)  (detached tensor, x0 shape)  -- for curvature
        comp_ids     competitor class ids scored                       (list[int])
        comp_delta   delta1_j for each scored competitor               (list[float])
        ill          True if the winning competitor hit the grad-norm floor
    """
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)

    x = x0.clone().detach().requires_grad_(True)
    logits = _as_logits(model(x))[0]              # (C,)
    C = int(logits.shape[0])
    y = int(logits.argmax().item())

    # Competitor set: nearest classes by logit (largest logits, excluding y).
    comp_all = [c for c in range(C) if c != y]
    if topk is not None and topk < len(comp_all):
        order = sorted(comp_all, key=lambda c: float(logits[c].item()), reverse=True)
        comp_ids = order[:topk]
    else:
        comp_ids = comp_all

    # One backward per needed class: {y} U competitors. Reuses the DeepFool pattern.
    grads = _input_grads(logits, x, [y] + comp_ids)     # (1+K, 1, C, H, W)
    grad_y = grads[0]
    logits = logits.detach()

    best = None  # (delta, j, margin, gradnorm, grad_g, ill)
    comp_delta = []
    for idx, j in enumerate(comp_ids):
        margin = float(logits[y] - logits[j])           # g_{y,j} >= 0
        grad_g = grad_y - grads[idx + 1]                # grad_x g_{y,j}
        gnorm = float(torch.norm(grad_g.flatten(), p=2))
        ill = gnorm < eps_grad
        delta_j = margin / max(gnorm, eps_grad)
        comp_delta.append(delta_j)
        if best is None or delta_j < best[0]:
            best = (delta_j, j, margin, gnorm, grad_g, ill)

    delta1, j_star, margin_star, gradnorm_star, grad_star, ill = best
    return {
        "y": y,
        "j_star": int(j_star),
        "delta1": float(delta1),
        "margin_star": float(margin_star),
        "gradnorm_star": float(gradnorm_star),
        "grad_star": grad_star.detach(),
        "comp_ids": comp_ids,
        "comp_delta": comp_delta,
        "ill": bool(ill),
    }


# =====================================================================
# Optional per-query second-order (curvature) refinement.
# =====================================================================
def second_order_query(model, x0, y, j_star, margin, grad0,
                       fd_r=1e-2, eps_grad=EPS_GRAD, corr_tol=0.0):
    """Curvature-corrected band distance along the first-order boundary direction.

    Uses ONE Hessian-vector product via finite difference (no clamping of the
    probe point):
        H z ~= (grad g(x0 + h z) - grad g(x0)) / h ,   z = grad0
    with h chosen so the probe step has L2 norm ~ fd_r (stabilizes the difference
    without changing the estimator). Then
        kappa = (grad0^T H grad0) / ||grad0||^2
        delta2 = (||grad0|| - sqrt(||grad0||^2 - 2 kappa g)) / kappa
    Radicand < 0 (no crossing along v) => delta2 = +inf. |kappa| ~ 0 => delta2 falls
    back to the first-order delta (its analytic limit).

    Returns dict: delta2, kappa, correction, radicand_neg, skipped.
      correction = kappa g^2 / (2 ||grad0||^3); if |correction| <= corr_tol the
      first-order estimate already suffices and delta2 is set to delta1 (skipped=True).
    """
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)

    gnorm = float(torch.norm(grad0.flatten(), p=2))
    delta1 = margin / max(gnorm, eps_grad)

    if gnorm < eps_grad:
        return {"delta2": delta1, "kappa": 0.0, "correction": 0.0,
                "radicand_neg": False, "skipped": True}

    # Probe step of L2 norm ~ fd_r along z = grad0 (h = fd_r / ||grad0||).
    h = fd_r / gnorm
    x1 = x0 + h * grad0                              # deliberately unclamped
    _, grad1 = _margin_value_and_grad(model, x1, y, j_star)
    Hz = (grad1 - grad0) / h                         # ~ H grad0
    kappa = float(torch.dot(grad0.flatten(), Hz.flatten()) / (gnorm ** 2))

    correction = kappa * (margin ** 2) / (2.0 * gnorm ** 3 + eps_grad)
    if abs(correction) <= corr_tol:
        return {"delta2": delta1, "kappa": kappa, "correction": float(correction),
                "radicand_neg": False, "skipped": True}

    if abs(kappa) < eps_grad:
        return {"delta2": delta1, "kappa": kappa, "correction": float(correction),
                "radicand_neg": False, "skipped": True}

    radicand = gnorm ** 2 - 2.0 * kappa * margin
    if radicand < 0.0:
        return {"delta2": float("inf"), "kappa": kappa, "correction": float(correction),
                "radicand_neg": True, "skipped": False}

    delta2 = (gnorm - math.sqrt(radicand)) / kappa
    # Guard the wrong root (negative distance for kappa<0): fall back to first order.
    if delta2 < 0.0:
        delta2 = delta1
    return {"delta2": float(delta2), "kappa": kappa, "correction": float(correction),
            "radicand_neg": False, "skipped": False}


# =====================================================================
# Phase 1a: collect (expensive, run once).
# =====================================================================
def _iter_queries(data, max_samples=None):
    """Yield (x_single_with_batch_dim, label_int_or_-1) from a DataLoader of
    (x, y) batches, a single (x, y) tuple, or a bare tensor of images."""
    seen = 0
    if isinstance(data, torch.Tensor):
        if data.dim() == 3:
            data = data.unsqueeze(0)
        batches = [(data, None)]

    elif isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], torch.Tensor):
        xb, yb = data
        if xb.dim() == 3:
            xb = xb.unsqueeze(0)
        batches = [(xb, yb)]

    else:
        batches = data

    for batch in batches:
        if isinstance(batch, (tuple, list)):
            xb, yb = batch[0], (batch[1] if len(batch) > 1 else None)
        else:
            xb, yb = batch, None

        if xb.dim() == 3:
            xb = xb.unsqueeze(0)

        for i in range(xb.shape[0]):
            label = int(yb[i].item()) if yb is not None else -1
            yield xb[i:i + 1], label

            seen += 1
            if max_samples is not None and seen >= max_samples:
                return


def collect_boundary_stats(model, data, device="cuda", topk=None,
                           compute_curvature=False, fd_r=1e-2, corr_tol=0.0,
                           eps_grad=EPS_GRAD, max_samples=None, verbose=True):
    """ONE expensive pass. Returns a dict of CPU tensors, one entry per query:

        delta1        (N,)  first-order nearest-boundary distance
        j_star        (N,)  nearest competitor id            (long)
        y_pred        (N,)  predicted class                  (long)
        label         (N,)  ground-truth label or -1         (long)
        margin_star   (N,)
        gradnorm_star (N,)
        ill           (N,)  bool, grad-norm floor hit
        delta2        (N,)  present iff compute_curvature (else == delta1)
        kappa         (N,)  present iff compute_curvature
        correction    (N,)  present iff compute_curvature

    Uses model.eval() but NOT torch.no_grad() (input gradients are required).
    """
    model.eval()
    delta1, jstar, ypred, labels = [], [], [], []
    margins, gnorms, ills = [], [], []
    delta2, kappas, corrs = [], [], []

    n = 0
    for x_single, label in _iter_queries(data, max_samples):
        x0 = x_single.to(device)
        fo = first_order_query(model, x0, topk=topk, eps_grad=eps_grad)

        delta1.append(fo["delta1"])
        jstar.append(fo["j_star"])
        ypred.append(fo["y"])
        labels.append(label)
        margins.append(fo["margin_star"])
        gnorms.append(fo["gradnorm_star"])
        ills.append(fo["ill"])

        if compute_curvature:
            so = second_order_query(
                model, x0, fo["y"], fo["j_star"], fo["margin_star"],
                fo["grad_star"], fd_r=fd_r, eps_grad=eps_grad, corr_tol=corr_tol,
            )
            delta2.append(so["delta2"])
            kappas.append(so["kappa"])
            corrs.append(so["correction"])

        n += 1
        if verbose and n % 200 == 0:
            print(f"  [boundary_band] processed {n} queries "
                  f"(mean delta1={np.mean(delta1):.4f})")

    stats = {
        "delta1": torch.tensor(delta1, dtype=torch.float32),
        "j_star": torch.tensor(jstar, dtype=torch.long),
        "y_pred": torch.tensor(ypred, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long),
        "margin_star": torch.tensor(margins, dtype=torch.float32),
        "gradnorm_star": torch.tensor(gnorms, dtype=torch.float32),
        "ill": torch.tensor(ills, dtype=torch.bool),
    }
    if compute_curvature:
        stats["delta2"] = torch.tensor(delta2, dtype=torch.float32)
        stats["kappa"] = torch.tensor(kappas, dtype=torch.float32)
        stats["correction"] = torch.tensor(corrs, dtype=torch.float32)
    else:
        stats["delta2"] = stats["delta1"].clone()  # keep the schema uniform
    return stats


# =====================================================================
# Batched variants: same estimator, GPU-parallel over the query dimension.
#
# Correctness precondition: model.eval() (no cross-sample coupling -- BN uses
# running stats, dropout off). Then for a batch x = (x_1..x_B) the gradient of a
# SUMMED selected logit,  s = sum_i z_{c_i}(x_i),  w.r.t. x recovers each sample's
# own gradient, because d z_c(x_i) / d x_{i'} = 0 for i != i'. This turns the
# per-query loop of collect_boundary_stats into 1 forward + (1+K) backwards per
# BATCH, with all margin/norm/argmin math vectorized.
# =====================================================================
def _competitor_ids_batch(logits, y, topk):
    """Per-sample competitor class ids, (B, K) long.

    Mirrors first_order_query's ordering so tie-breaking agrees:
      topk=None (or >= C-1)  -> all C-1 competitors in ascending class-id order;
      topk=K                 -> the K nearest competitors by logit, largest first.
    """
    B, C = logits.shape
    if topk is None or topk >= C - 1:
        ids = torch.arange(C, device=logits.device).expand(B, C)
        keep = ids != y.unsqueeze(1)
        return ids[keep].view(B, C - 1)
    masked = logits.detach().clone()
    masked.scatter_(1, y.unsqueeze(1), float("-inf"))
    return masked.topk(topk, dim=1).indices


def first_order_batch(model, xb, topk=None, eps_grad=EPS_GRAD):
    """Batched first_order_query: one forward + (1+K) backwards for the WHOLE batch.

    Args match first_order_query except xb is (B, C, H, W) (a single (C, H, W)
    query is auto-unsqueezed). Returns a dict of per-sample tensors, detached, on
    the input device:

        y             (B,)  long   predicted class
        j_star        (B,)  long   nearest competitor
        delta1        (B,)         Delta^(1) per query (raw-pixel L2)
        margin_star   (B,)
        gradnorm_star (B,)
        grad_star     (B, C, H, W) grad_x g_{y,j*} -- for curvature / post-hoc
        ill           (B,)  bool   grad-norm floor hit by the winner
        comp_ids      (B, K) long  competitor ids scored
        comp_delta    (B, K)       delta1_j per competitor
    """
    if xb.dim() == 3:
        xb = xb.unsqueeze(0)

    x = xb.clone().detach().requires_grad_(True)
    logits = _as_logits(model(x))                       # (B, C)
    B = x.shape[0]
    y = logits.detach().argmax(dim=1)                   # (B,)
    comp_ids = _competitor_ids_batch(logits, y, topk)   # (B, K)
    K = comp_ids.shape[1]

    # Summed-selected-logit trick (see section banner): per-sample grad_x z_y.
    grad_y = torch.autograd.grad(
        logits.gather(1, y.unsqueeze(1)).sum(), x, retain_graph=True)[0]

    logits_d = logits.detach()
    margins = logits_d.gather(1, y.unsqueeze(1)) - logits_d.gather(1, comp_ids)  # (B, K)

    best_delta = torch.full((B,), float("inf"), device=x.device)
    best_j = torch.zeros(B, dtype=torch.long, device=x.device)
    best_margin = torch.zeros_like(best_delta)
    best_gnorm = torch.zeros_like(best_delta)
    best_ill = torch.zeros(B, dtype=torch.bool, device=x.device)
    best_grad = torch.zeros_like(grad_y)
    comp_delta = torch.empty(B, K, device=x.device, dtype=margins.dtype)

    bview = (B,) + (1,) * (x.dim() - 1)
    for k in range(K):
        s = logits.gather(1, comp_ids[:, k:k + 1]).sum()
        grad_k = torch.autograd.grad(s, x, retain_graph=(k + 1 < K))[0]
        grad_g = grad_y - grad_k                        # grad_x g_{y,j} per sample
        gnorm = grad_g.flatten(1).norm(p=2, dim=1)      # (B,)
        delta_k = margins[:, k] / gnorm.clamp_min(eps_grad)
        comp_delta[:, k] = delta_k

        better = delta_k < best_delta                   # strict <, as in the loop version
        best_delta = torch.where(better, delta_k, best_delta)
        best_j = torch.where(better, comp_ids[:, k], best_j)
        best_margin = torch.where(better, margins[:, k], best_margin)
        best_gnorm = torch.where(better, gnorm, best_gnorm)
        best_ill = torch.where(better, gnorm < eps_grad, best_ill)
        best_grad = torch.where(better.view(bview), grad_g, best_grad)

    return {
        "y": y,
        "j_star": best_j,
        "delta1": best_delta,
        "margin_star": best_margin,
        "gradnorm_star": best_gnorm,
        "grad_star": best_grad.detach(),
        "ill": best_ill,
        "comp_ids": comp_ids,
        "comp_delta": comp_delta,
    }


def second_order_batch(model, x0, y, j_star, margin, grad0,
                       fd_r=1e-2, eps_grad=EPS_GRAD, corr_tol=0.0):
    """Batched second_order_query: ONE extra forward+backward for the whole batch.

    Args are the per-sample tensors from first_order_batch (y/j_star long (B,),
    margin (B,), grad0 = grad_star (B, C, H, W)). The per-sample fallback logic of
    the scalar version (ill-conditioned skip, |correction| <= corr_tol skip,
    |kappa| ~ 0 skip, radicand < 0 => +inf, wrong-root guard) is applied
    element-wise. Returns dict of (B,) tensors: delta2, kappa, correction,
    radicand_neg, skipped.
    """
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    B = x0.shape[0]
    bview = (B,) + (1,) * (x0.dim() - 1)

    gnorm = grad0.flatten(1).norm(p=2, dim=1)                     # (B,)
    gnorm_safe = gnorm.clamp_min(eps_grad)
    delta1 = margin / gnorm_safe

    # Probe step of L2 norm ~ fd_r along z = grad0 (h = fd_r / ||grad0||),
    # deliberately unclamped -- same policy as second_order_query.
    h = fd_r / gnorm_safe
    x1 = (x0 + h.view(bview) * grad0).clone().detach().requires_grad_(True)
    logits = _as_logits(model(x1))
    g = (logits.gather(1, y.unsqueeze(1)) - logits.gather(1, j_star.unsqueeze(1))).sum()
    grad1 = torch.autograd.grad(g, x1)[0]

    Hz = (grad1 - grad0) / h.view(bview)                          # ~ H grad0 per sample
    kappa = (grad0.flatten(1) * Hz.flatten(1)).sum(dim=1) / gnorm_safe.pow(2)
    correction = kappa * margin.pow(2) / (2.0 * gnorm.pow(3) + eps_grad)

    ill = gnorm < eps_grad
    skipped = ill | (correction.abs() <= corr_tol) | (kappa.abs() < eps_grad)
    radicand = gnorm.pow(2) - 2.0 * kappa * margin
    radicand_neg = (~skipped) & (radicand < 0.0)

    kappa_safe = torch.where(kappa.abs() < eps_grad, torch.ones_like(kappa), kappa)
    delta2 = (gnorm - radicand.clamp_min(0.0).sqrt()) / kappa_safe
    delta2 = torch.where(delta2 < 0.0, delta1, delta2)            # wrong-root guard
    delta2 = torch.where(radicand_neg, torch.full_like(delta1, float("inf")), delta2)
    delta2 = torch.where(skipped, delta1, delta2)

    # Match the scalar version's ill-conditioned early return (kappa = corr = 0).
    kappa = torch.where(ill, torch.zeros_like(kappa), kappa)
    correction = torch.where(ill, torch.zeros_like(correction), correction)

    return {"delta2": delta2, "kappa": kappa, "correction": correction,
            "radicand_neg": radicand_neg, "skipped": skipped}


# =====================================================================
# Post-hoc verification of the dropped constraints (guide Sec. 2.4, Eq. 13).
# =====================================================================
def posthoc_verify_batch(model, x0, y, j_star, margin, grad_star, eps_grad=EPS_GRAD):
    """Check the inequality constraints dropped in the first-order relaxation.

    Candidate point per query:  x_hat = x0 + r^(1),  with the first-order step
        r^(1) = -(g / ||grad g||^2) grad g          (g = g_{y,j*} at x0).
    Verification (Eq. 13):  g_{y,l}(x_hat) >= 0  for all l not in {y, j*} --
    at the candidate point no THIRD class overtakes y, so the y-vs-j* pairwise
    boundary is the operative one. ONE forward pass, no gradients.

    Two facts kept distinct (per the guide): x_hat lies on the LINEARIZED
    hyperplane, not on the true boundary, so g_{y,j*}(x_hat) != 0 in general; its
    value is returned as `lin_resid` (small iff the linearization is accurate).
    And the check only tests dominance of the remaining classes -- it cannot
    certify boundary membership. x_hat is deliberately NOT clamped to [0,1]
    (file-wide no-clamping policy: this is a geometric probe, not an image).

    Args are per-sample tensors from first_order_batch (a B=1 slice built from
    first_order_query's outputs works identically).

    Returns dict of per-sample tensors:
        ok         (B,) bool  all dropped constraints hold at x_hat
        n_viol     (B,) long  number of violated constraints
        min_other  (B,)       min_l g_{y,l}(x_hat) over l not in {y, j*}
        lin_resid  (B,)       g_{y,j*}(x_hat), the linearization residual
    """
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    B = x0.shape[0]
    bview = (B,) + (1,) * (x0.dim() - 1)

    gnorm2 = grad_star.flatten(1).pow(2).sum(dim=1)               # ||grad g||^2
    coef = margin / gnorm2.clamp_min(eps_grad ** 2)
    x_hat = x0 - coef.view(bview) * grad_star                     # x0 + r^(1)

    with torch.no_grad():
        logits = _as_logits(model(x_hat))                         # (B, C)

    g_all = logits.gather(1, y.unsqueeze(1)) - logits             # g_{y,l}; g_{y,y} = 0
    others = torch.ones_like(g_all, dtype=torch.bool)
    others.scatter_(1, y.unsqueeze(1), False)
    others.scatter_(1, j_star.unsqueeze(1), False)

    min_other = g_all.masked_fill(~others, float("inf")).min(dim=1).values
    n_viol = ((g_all < 0.0) & others).sum(dim=1)
    lin_resid = g_all.gather(1, j_star.unsqueeze(1)).squeeze(1)

    return {"ok": min_other >= 0.0, "n_viol": n_viol,
            "min_other": min_other, "lin_resid": lin_resid}


def _iter_query_batches(data, max_samples=None):
    """Batch-wise twin of _iter_queries: yield (x_batch, labels_or_None) from a
    DataLoader, a single (x, y) tuple, or a bare tensor, truncating the last batch
    so at most max_samples queries are yielded in total."""
    if isinstance(data, torch.Tensor):
        batches = [(data, None)]
    elif isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], torch.Tensor):
        batches = [data]
    else:
        batches = data

    seen = 0
    for batch in batches:
        if isinstance(batch, (tuple, list)):
            xb, yb = batch[0], (batch[1] if len(batch) > 1 else None)
        else:
            xb, yb = batch, None
        if xb.dim() == 3:
            xb = xb.unsqueeze(0)

        if max_samples is not None:
            room = max_samples - seen
            if room <= 0:
                return
            xb = xb[:room]
            yb = yb[:room] if yb is not None else None

        seen += xb.shape[0]
        yield xb, yb


def collect_boundary_stats_batched(model, data, device="cuda", topk=None,
                                   compute_curvature=False, fd_r=1e-2, corr_tol=0.0,
                                   eps_grad=EPS_GRAD, max_samples=None, verbose=True,
                                   posthoc=False):
    """Batched drop-in for collect_boundary_stats: same schema, same numbers up to
    kernel-order float noise, but each backward covers a whole batch. The batch
    size is whatever `data` yields (DataLoader batch size, or the full tensor for
    bare-tensor / (X, y) inputs) -- it is a speed/VRAM knob, not an accuracy knob.

    Extra keys when posthoc=True (Eq. 13 dropped-constraint check at x_hat):
        ph_ok        (N,) bool   all remaining classes dominated at x_hat
        ph_n_viol    (N,) long   violated-constraint count
        ph_min_other (N,)        min_l g_{y,l}(x_hat), l not in {y, j*}
        ph_lin_resid (N,)        g_{y,j*}(x_hat) linearization residual

    Uses model.eval() -- required for correctness here, not just convention (see
    the batched-variants section banner).
    """
    model.eval()
    out = {"delta1": [], "j_star": [], "y_pred": [], "label": [],
           "margin_star": [], "gradnorm_star": [], "ill": []}
    if compute_curvature:
        out.update({"delta2": [], "kappa": [], "correction": []})
    if posthoc:
        out.update({"ph_ok": [], "ph_n_viol": [], "ph_min_other": [], "ph_lin_resid": []})

    n, next_print = 0, 200
    for xb, yb in _iter_query_batches(data, max_samples):
        x0 = xb.to(device)
        B = x0.shape[0]
        fo = first_order_batch(model, x0, topk=topk, eps_grad=eps_grad)

        out["delta1"].append(fo["delta1"].cpu())
        out["j_star"].append(fo["j_star"].cpu())
        out["y_pred"].append(fo["y"].cpu())
        out["label"].append(yb.long().cpu() if yb is not None
                            else torch.full((B,), -1, dtype=torch.long))
        out["margin_star"].append(fo["margin_star"].cpu())
        out["gradnorm_star"].append(fo["gradnorm_star"].cpu())
        out["ill"].append(fo["ill"].cpu())

        if compute_curvature:
            so = second_order_batch(model, x0, fo["y"], fo["j_star"],
                                    fo["margin_star"], fo["grad_star"],
                                    fd_r=fd_r, eps_grad=eps_grad, corr_tol=corr_tol)
            out["delta2"].append(so["delta2"].cpu())
            out["kappa"].append(so["kappa"].cpu())
            out["correction"].append(so["correction"].cpu())

        if posthoc:
            ph = posthoc_verify_batch(model, x0, fo["y"], fo["j_star"],
                                      fo["margin_star"], fo["grad_star"],
                                      eps_grad=eps_grad)
            out["ph_ok"].append(ph["ok"].cpu())
            out["ph_n_viol"].append(ph["n_viol"].cpu())
            out["ph_min_other"].append(ph["min_other"].cpu())
            out["ph_lin_resid"].append(ph["lin_resid"].cpu())

        n += B
        if verbose and n >= next_print:
            mean_d = torch.cat(out["delta1"]).mean().item()
            print(f"  [boundary_band] processed {n} queries "
                  f"(mean delta1={mean_d:.4f})")
            next_print = n + 200

    if n == 0:
        raise ValueError("collect_boundary_stats_batched: no queries in `data`.")

    stats = {k: torch.cat(v) for k, v in out.items()}
    for k in ("delta1", "margin_star", "gradnorm_star",
              "delta2", "kappa", "correction", "ph_min_other", "ph_lin_resid"):
        if k in stats:
            stats[k] = stats[k].float()
    if not compute_curvature:
        stats["delta2"] = stats["delta1"].clone()  # keep the schema uniform
    return stats


# =====================================================================
# Phase 1b: band decision + calibration (cheap, sweeps over cached stats).
# =====================================================================
def _delta_of(stats, order):
    if order == "second":
        return stats["delta2"]
    if order == "first":
        return stats["delta1"]
    raise ValueError(f"order must be 'first' or 'second', got {order!r}")


def band_from_stats(stats, d, order="first"):
    """Cheap band decision on cached stats.

    Returns dict:
        flags      (N,) bool  == 1[Delta <= d]
        band_rate  float       fraction flagged
        delta      (N,)        the Delta series used (first or second order)
    """
    delta = _delta_of(stats, order)
    flags = delta <= d
    return {
        "flags": flags,
        "band_rate": float(flags.float().mean().item()) if len(flags) else 0.0,
        "delta": delta,
    }


def calibrate_d(stats, percentile=1.0, order="first"):
    """Pick d from a low percentile of the (benign) Delta distribution, in the same
    raw-pixel L2 units as the gradients. Ill-conditioned / non-finite entries are
    dropped before the percentile so a few exploded deltas don't skew calibration."""
    delta = _delta_of(stats, order).numpy()
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        raise ValueError("calibrate_d: no finite Delta values to calibrate on.")
    return float(np.percentile(finite, percentile))


def summarize(stats, order="first"):
    """Small distribution summary for logging/inspection."""
    delta = _delta_of(stats, order).numpy()
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p01": float("nan"), "p05": float("nan"), "n_ill": int(stats["ill"].sum())}
    return {
        "n": int(delta.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "n_ill": int(stats["ill"].sum().item()),
    }


# =====================================================================
# Debug checks from the implementation guide (opt-in).
# =====================================================================
def debug_checks(model, x0, device="cuda", topk=None, atol=1e-3):
    """Guide's recommended sanity checks for a single query. Returns a dict of
    residuals; raises AssertionError if a check fails beyond `atol` (relative).

      * vector-norm check:      ||r*|| ~= Delta^(1)
      * linearized-boundary:    g_{y,j*} + <grad g, r*> ~= 0
    where r* = -(g/||grad g||^2) grad g.
    """
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    x0 = x0.to(device)
    fo = first_order_query(model, x0, topk=topk)
    g = fo["margin_star"]
    grad = fo["grad_star"]
    gnorm2 = float(torch.dot(grad.flatten(), grad.flatten()))
    r_star = -(g / gnorm2) * grad

    norm_r = float(torch.norm(r_star.flatten(), p=2))
    vec_norm_resid = abs(norm_r - fo["delta1"])
    lin_resid = abs(g + float(torch.dot(grad.flatten(), r_star.flatten())))

    scale = max(abs(g), 1e-8)
    assert vec_norm_resid <= atol * max(fo["delta1"], 1e-8), \
        f"vector-norm check failed: ||r*||={norm_r} vs delta1={fo['delta1']}"
    assert lin_resid <= atol * scale, \
        f"linearized-boundary check failed: residual={lin_resid}"
    return {"vec_norm_resid": vec_norm_resid, "lin_resid": lin_resid,
            "delta1": fo["delta1"]}


# #####################################################################
# Lp generalization (ADDITIVE -- nothing above this banner is modified).
#
# Implements Lp_generalization_implementation_guide.md. Everything above is the
# L2 (self-dual) path; these functions generalize the FIRST-ORDER estimator to a
# user-facing distance metric p, with the dual exponent q = p/(p-1) acting on the
# input-side gradient norm.
#
#   * Numerator (margin g_{k,j}) is p-INVARIANT.               (guide S0)
#   * Denominator uses the dual q-norm ||grad g||_q.            (guide S1)
#   * Perturbation direction r uses the sign(a)|a|^(q-1) form.  (guide S2)
#   * Post-hoc dominance is kept p-INDEPENDENT (guide S8 default): the criterion
#     g_{k,l}(x0+r) >= 0 is a pure class-membership test; only r is p-correct.
#
# Caller ALWAYS passes p (never q); q is derived internally (guide S0 convention).
# Regression contract: p=2 reproduces the L2 functions element-wise.
# #####################################################################
def _dual_q(p):
    """Dual exponent q = p/(p-1) for the input-side norm (guide S0).

    Explicit endpoints:  p=2 -> q=2 (self-dual);  p=inf -> q=1;  p=1 -> q=inf.
    General p in (1, inf) via the formula. p < 1 is rejected (outside DeepFool's
    [1, inf) framework; p=1 is the experimental single-coordinate branch).
    """
    if p == float("inf"):
        return 1.0
    if p == 1:
        return float("inf")
    if p <= 1:
        raise ValueError(f"p must be >= 1 for the Lp band distance, got {p!r}.")
    return p / (p - 1.0)


def _dual_norm_rows(grad_flat, q):
    """||.||_q along dim=1 for a (B, D) tensor. Handles q in {1, 2, inf} and any
    general q >= 1 (torch.linalg.vector_norm covers ord=inf as max|.|)."""
    if q == float("inf"):
        return grad_flat.abs().amax(dim=1)
    return torch.linalg.vector_norm(grad_flat, ord=q, dim=1)


def _first_order_direction_lp(a, b, q, eps_grad=EPS_GRAD):
    """Single-shot Lp perturbation direction r (guide S2), batched over dim=0.

        r = -b * sign(a) |a|^(q-1) / ||a||_q^q ,     a = grad g_{k,j},  b = g_{k,j} >= 0.

    q=inf (p=1) is the singular limit: mass on argmax_i|a_i| only (guide S2 branch).
    For finite q the general expression is evaluated in a max-normalized form so
    |a|^(q-1) cannot over/underflow (guide S7): with m = max_i|a_i| and u = |a|/m,

        r = -(b/m) * sign(a) u^(q-1) / sum_i u_i^q .

    Sanity (verified in the Lp test):
      q=2  ->  -b a / ||a||_2^2         (== the L2 residual; regression)
      q=1  ->  -b sign(a) / ||a||_1     (FGSM direction, p=inf)
      q=inf->  single-coordinate branch (p=1).
    """
    a_flat = a.flatten(1)                                   # (B, D)
    r_flat = torch.zeros_like(a_flat)

    if q == float("inf"):
        idx = a_flat.abs().argmax(dim=1, keepdim=True)      # (B, 1) top-|a| coord
        ai = a_flat.gather(1, idx).squeeze(1)               # signed value there
        ai_safe = torch.where(ai.abs() < eps_grad,
                              torch.full_like(ai, eps_grad), ai)
        r_flat.scatter_(1, idx, (-b / ai_safe).unsqueeze(1))
        return r_flat.view_as(a)

    absa = a_flat.abs()
    m = absa.amax(dim=1, keepdim=True).clamp_min(eps_grad)  # (B, 1)
    u = absa / m                                            # in [0, 1]
    su = u.pow(q).sum(dim=1, keepdim=True).clamp_min(eps_grad)   # sum_i u_i^q
    r_flat = -(b.unsqueeze(1) / m) * a_flat.sign() * u.pow(q - 1.0) / su
    return r_flat.view_as(a)


def first_order_batch_lp(model, xb, p=2.0, topk=None, eps_grad=EPS_GRAD):
    """Batched first-order Lp band estimate (guide S1, S2, S4). Mirrors
    first_order_batch but the per-competitor distance uses the dual q-norm

        delta_{j,p} = g_{k,j} / ||grad g_{k,j}||_q ,    q = p/(p-1),

    and the winner's p-correct direction r_star is returned for the post-hoc
    check. Requires model.eval() (summed-selected-logit batching; same precondition
    as first_order_batch). Returns per-sample tensors on the input device:

        y, j_star, delta1, margin_star,
        gradnorm_star  = ||grad g||_q  (NOTE: q-norm, equals L2 only when p=2),
        grad_star      = grad_x g_{k,j*}   (for r / diagnostics),
        r_star         = -b sign(a)|a|^(q-1)/||a||_q^q  (guide S2, p-correct),
        ill            = winner hit the q-norm floor,
        comp_ids, comp_delta,
        p, q           = python floats (metric bookkeeping; guide S4 facet logging).
    """
    q = _dual_q(p)
    if xb.dim() == 3:
        xb = xb.unsqueeze(0)

    x = xb.clone().detach().requires_grad_(True)
    logits = _as_logits(model(x))                       # (B, C)
    B = x.shape[0]
    y = logits.detach().argmax(dim=1)
    comp_ids = _competitor_ids_batch(logits, y, topk)   # (B, K)
    K = comp_ids.shape[1]

    grad_y = torch.autograd.grad(
        logits.gather(1, y.unsqueeze(1)).sum(), x, retain_graph=True)[0]

    logits_d = logits.detach()
    margins = logits_d.gather(1, y.unsqueeze(1)) - logits_d.gather(1, comp_ids)  # (B, K)

    best_delta = torch.full((B,), float("inf"), device=x.device)
    best_j = torch.zeros(B, dtype=torch.long, device=x.device)
    best_margin = torch.zeros_like(best_delta)
    best_gnorm = torch.zeros_like(best_delta)            # q-norm of the winner
    best_ill = torch.zeros(B, dtype=torch.bool, device=x.device)
    best_grad = torch.zeros_like(grad_y)
    comp_delta = torch.empty(B, K, device=x.device, dtype=margins.dtype)

    bview = (B,) + (1,) * (x.dim() - 1)
    for k in range(K):
        s = logits.gather(1, comp_ids[:, k:k + 1]).sum()
        grad_k = torch.autograd.grad(s, x, retain_graph=(k + 1 < K))[0]
        grad_g = grad_y - grad_k                         # grad_x g_{y,j}
        gnorm_q = _dual_norm_rows(grad_g.flatten(1), q)  # ||grad g||_q  (guide S1)
        delta_k = margins[:, k] / gnorm_q.clamp_min(eps_grad)
        comp_delta[:, k] = delta_k

        better = delta_k < best_delta                    # strict <, as in L2 path
        best_delta = torch.where(better, delta_k, best_delta)
        best_j = torch.where(better, comp_ids[:, k], best_j)
        best_margin = torch.where(better, margins[:, k], best_margin)
        best_gnorm = torch.where(better, gnorm_q, best_gnorm)
        best_ill = torch.where(better, gnorm_q < eps_grad, best_ill)
        best_grad = torch.where(better.view(bview), grad_g, best_grad)

    r_star = _first_order_direction_lp(best_grad, best_margin, q, eps_grad)

    return {
        "y": y,
        "j_star": best_j,
        "delta1": best_delta,
        "margin_star": best_margin,
        "gradnorm_star": best_gnorm,
        "grad_star": best_grad.detach(),
        "r_star": r_star.detach(),
        "ill": best_ill,
        "comp_ids": comp_ids,
        "comp_delta": comp_delta,
        "p": float(p),
        "q": float(q),
    }


def posthoc_verify_batch_lp(model, x0, y, j_star, r_star):
    """Post-hoc dominance verification (guide S3), p-INDEPENDENT criterion.

    Identical membership test to posthoc_verify_batch -- g_{y,l}(x_hat) >= 0 for all
    l not in {y, j*} -- but the candidate point is built from the p-CORRECT direction
    of guide S2:  x_hat = x0 + r_star  (NOT the L2 residual). Passing the wrong r is
    exactly the "stale L2 residual" pitfall the guide warns against, so r_star must
    come from first_order_batch_lp with the same p.

    (guide S8 decision: kept p-independent. If you later need a p-COUPLED band
    criterion, this function is where that redesign lands.)

    One forward pass, no gradients. Returns per-sample tensors:
        ok, n_viol, min_other, lin_resid   (same schema as posthoc_verify_batch).
    """
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    x_hat = x0 + r_star                                  # p-correct candidate point

    with torch.no_grad():
        logits = _as_logits(model(x_hat))                # (B, C)

    g_all = logits.gather(1, y.unsqueeze(1)) - logits    # g_{y,l}; g_{y,y}=0
    others = torch.ones_like(g_all, dtype=torch.bool)
    others.scatter_(1, y.unsqueeze(1), False)
    others.scatter_(1, j_star.unsqueeze(1), False)

    min_other = g_all.masked_fill(~others, float("inf")).min(dim=1).values
    n_viol = ((g_all < 0.0) & others).sum(dim=1)
    lin_resid = g_all.gather(1, j_star.unsqueeze(1)).squeeze(1)

    return {"ok": min_other >= 0.0, "n_viol": n_viol,
            "min_other": min_other, "lin_resid": lin_resid}


def collect_boundary_stats_batched_lp(model, data, p=2.0, device="cuda", topk=None,
                                      eps_grad=EPS_GRAD, max_samples=None,
                                      verbose=True, posthoc=True):
    """Lp drop-in for collect_boundary_stats_batched (first order only).

    Same output schema as the L2 collector, so every downstream helper
    (band_from_stats, calibrate_d, summarize, stats_to_df) works unchanged -- but
    `delta1` now measures the Lp distance and `gradnorm_star` the dual q-norm.
    `delta2` is set to a copy of `delta1` (no Lp curvature term; keeps the schema
    uniform for band_from_stats/order="first"). Extra ph_* keys when posthoc=True.

    CROSS-p WARNING (guide S6): absolute delta values are NOT comparable across p
    (Linf is 1-2 orders smaller than L2 purely from ||a||_1 >= ||a||_2, a unit
    shift). Compare only dimensionless separability -- see lp_separability_metrics.
    """
    _dual_q(p)                                           # validate p early (raises if p < 1)
    model.eval()
    out = {"delta1": [], "j_star": [], "y_pred": [], "label": [],
           "margin_star": [], "gradnorm_star": [], "ill": []}
    if posthoc:
        out.update({"ph_ok": [], "ph_n_viol": [], "ph_min_other": [], "ph_lin_resid": []})

    n, next_print = 0, 200
    for xb, yb in _iter_query_batches(data, max_samples):
        x0 = xb.to(device)
        B = x0.shape[0]
        fo = first_order_batch_lp(model, x0, p=p, topk=topk, eps_grad=eps_grad)

        out["delta1"].append(fo["delta1"].cpu())
        out["j_star"].append(fo["j_star"].cpu())
        out["y_pred"].append(fo["y"].cpu())
        out["label"].append(yb.long().cpu() if yb is not None
                            else torch.full((B,), -1, dtype=torch.long))
        out["margin_star"].append(fo["margin_star"].cpu())
        out["gradnorm_star"].append(fo["gradnorm_star"].cpu())
        out["ill"].append(fo["ill"].cpu())

        if posthoc:
            ph = posthoc_verify_batch_lp(model, x0, fo["y"], fo["j_star"],
                                         fo["r_star"])
            out["ph_ok"].append(ph["ok"].cpu())
            out["ph_n_viol"].append(ph["n_viol"].cpu())
            out["ph_min_other"].append(ph["min_other"].cpu())
            out["ph_lin_resid"].append(ph["lin_resid"].cpu())

        n += B
        if verbose and n >= next_print:
            mean_d = torch.cat(out["delta1"]).mean().item()
            print(f"  [boundary_band Lp p={p}] processed {n} queries "
                  f"(mean delta1={mean_d:.6f})")
            next_print = n + 200

    if n == 0:
        raise ValueError("collect_boundary_stats_batched_lp: no queries in `data`.")

    stats = {k: torch.cat(v) for k, v in out.items()}
    for k in ("delta1", "margin_star", "gradnorm_star", "ph_min_other", "ph_lin_resid"):
        if k in stats:
            stats[k] = stats[k].float()
    stats["delta2"] = stats["delta1"].clone()            # schema parity (no Lp curvature)
    return stats


# ---------------------------------------------------------------------
# Dimensionless separability + diagnostics (guide S5, S6). NO cross-p
# absolute comparison -- only scale-free scores that survive the unit shift.
# ---------------------------------------------------------------------
def _finite(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def _auroc_small_delta(test_delta, fp_delta):
    """AUROC with fingerprints as positives and SMALL delta as the positive signal
    (score = -delta), tie-corrected (average ranks). 0.5 = no separation, 1.0 =
    every fp is nearer the boundary than every benign point."""
    t = _finite(test_delta)
    f = _finite(fp_delta)
    if t.size == 0 or f.size == 0:
        return float("nan")
    s = -np.concatenate([t, f])                          # higher score = more fp-like
    y = np.concatenate([np.zeros(t.size), np.ones(f.size)])
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:                                    # average ranks over ties
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos, n_neg = f.size, t.size
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def lp_separability_metrics(test_delta, fp_delta, fpr=0.01):
    """Scale-free separability of fingerprints vs benign at a fixed p (guide S6).

    These are the ONLY cross-p comparable quantities -- report them, never the raw
    delta box plots side by side across p.

        auroc          : rank separability (fp=pos, small delta=pos); 0.5 = none.
        median_ratio   : median(fp) / median(test); < 1 means fp sits nearer.
        recall_at_fpr  : fraction of fp flagged when d = test's (100*fpr)-th pct,
                         i.e. detection at a benign FPR of `fpr` (band-gate recall).
        d_at_fpr       : the threshold used (test percentile), for reference.
    """
    t = _finite(test_delta)
    f = _finite(fp_delta)
    if t.size == 0 or f.size == 0:
        raise ValueError("lp_separability_metrics: empty test or fp delta after filtering.")
    d_at_fpr = float(np.percentile(t, 100.0 * fpr))
    return {
        "auroc": _auroc_small_delta(t, f),
        "median_ratio": float(np.median(f) / np.median(t)),
        "recall_at_fpr": float(np.mean(f <= d_at_fpr)),
        "fpr": float(fpr),
        "d_at_fpr": d_at_fpr,
        "n_test": int(t.size),
        "n_fp": int(f.size),
    }


def topk_agreement_diagnostic(model, data, p, K, device="cuda",
                              max_samples=None, eps_grad=EPS_GRAD):
    """Full-C vs top-K nearest-competitor agreement at metric p (guide S5).

    Top-K ranks competitors by MARGIN only, ignoring the q-norm denominator; as
    q -> 1 (p -> inf) the denominator's weight grows and top-K is likelier to miss
    the true j*. Run this on a small subset per p: if jstar_agreement drops for
    large p, raise K or use full-C (topk=None) for that p.

    Returns: p, q, K, n, jstar_agreement (fraction of queries whose j* matches
    between full-C and top-K), and delta_rel_mad (median |dK-dFull|/dFull).
    """
    full = collect_boundary_stats_batched_lp(model, data, p=p, device=device,
                                             topk=None, max_samples=max_samples,
                                             eps_grad=eps_grad, verbose=False)
    tk = collect_boundary_stats_batched_lp(model, data, p=p, device=device,
                                           topk=K, max_samples=max_samples,
                                           eps_grad=eps_grad, verbose=False)
    agree = float((full["j_star"] == tk["j_star"]).float().mean().item())
    df = full["delta1"].numpy()
    dk = tk["delta1"].numpy()
    m = np.isfinite(df) & np.isfinite(dk) & (df > 0)
    rel = np.abs(dk[m] - df[m]) / df[m]
    return {
        "p": float(p), "q": float(_dual_q(p)), "K": int(K),
        "n": int(len(df)),
        "jstar_agreement": agree,
        "delta_rel_mad": float(np.median(rel)) if rel.size else float("nan"),
    }
