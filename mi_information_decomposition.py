"""Paired empirical information decomposition for equations (4) and (5).

V and S are full quantized softmax-vector states, not continuous logits.
All values are in bits. X denotes the input of a fixed deterministic model
and quantizer, so I(X;S|V)=H(S|V). This module does not select checkpoints,
change existing MI tables, or establish population-estimation accuracy.
"""

import numpy as np


def _discrete_vector(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must contain integer states")
    return value


def _check_pairing(victim_sample_ids, suspect_sample_ids, n):
    v_ids = np.asarray(victim_sample_ids)
    s_ids = np.asarray(suspect_sample_ids)
    if v_ids.ndim != 1 or s_ids.ndim != 1 or len(v_ids) != n or len(s_ids) != n:
        raise ValueError("Both sample-ID vectors must have exactly N entries")
    if not np.array_equal(v_ids, s_ids):
        raise ValueError("Victim and suspect sample IDs must match in order")
    if len(np.unique(v_ids)) != n:
        raise ValueError("This probe interface requires unique sample IDs")


def _entropy(*vectors):
    _, counts = np.unique(np.column_stack(vectors), axis=0, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def _occupancy(*vectors):
    _, counts = np.unique(np.column_stack(vectors), axis=0, return_counts=True)
    n = int(counts.sum())
    return {
        "unique_states": len(counts),
        "unique_fraction": len(counts) / n,
        "singleton_sample_fraction": int(np.sum(counts == 1)) / n,
        "max_state_count": int(counts.max()),
    }


def decomposition_from_states(victim_states, suspect_states, labels, *,
                              victim_sample_ids, suspect_sample_ids):
    """Compute four terms from aligned discrete state IDs and ground-truth Y.

    Use each model's own inverse_idx from MI_check.mi_from_logits(verbose=True).
    The two sets of state IDs need not have the same codebook. Sample IDs must
    identify the same underlying images and order; labels alone cannot prove
    this. The caller must also ensure fixed clean transforms, class semantics,
    model identity, and a matching quantization specification.

    G_X = I(X;S|V), L_X = I(X;V|S)
    G_Y = I(S;Y|V), L_Y = I(V;Y|S)

    Entropies use one joint empirical distribution without bias correction.
    Tiny negative values from floating-point subtraction are retained so
    numerical issues remain inspectable. This is a descriptive estimator.
    """
    v = _discrete_vector(victim_states, "victim_states")
    s = _discrete_vector(suspect_states, "suspect_states")
    y = _discrete_vector(labels, "labels")
    if len(v) != len(s) or len(v) != len(y):
        raise ValueError("Victim states, suspect states and labels must have equal length")
    _check_pairing(victim_sample_ids, suspect_sample_ids, len(y))

    hv, hs, hy = _entropy(v), _entropy(s), _entropy(y)
    hvs = _entropy(v, s)
    hvy, hsy, hvsy = _entropy(v, y), _entropy(s, y), _entropy(v, s, y)
    gy = hvy + hvs - hv - hvsy
    ly = hsy + hvs - hs - hvsy
    gx, lx = hvs - hv, hvs - hs
    ity_v, ity_s = hv + hy - hvy, hs + hy - hsy
    result = {
        "N": len(y),
        "G_X": gx, "L_X": lx, "G_Y": gy, "L_Y": ly,
        "I_X_V": hv, "I_X_S": hs,
        "I_V_Y": ity_v, "I_S_Y": ity_s,
        "I_VS_Y": hvs + hy - hvsy,
        "delta_X": hs - hv, "delta_Y": ity_s - ity_v,
        "H_Y": hy, "H_VS": hvs,
        "H_Y_given_V": hvy - hv, "H_Y_given_S": hsy - hs,
        "H_Y_given_VS": hvsy - hvs,
        "eq4_residual": (hs - hv) - (gx - lx),
        "eq5_residual": (ity_s - ity_v) - (gy - ly),
        "occupancy_victim": _occupancy(v),
        "occupancy_suspect": _occupancy(s),
        "occupancy_pair": _occupancy(v, s),
    }
    tol = 1e-10
    if min(gx, lx, gy, ly) < -tol or gy > gx + tol or ly > lx + tol:
        raise ArithmeticError("Empirical information bounds failed")
    return result


def decomposition_from_logits(victim_logits, suspect_logits,
                              victim_label_matrix, suspect_label_matrix, *,
                              victim_sample_ids, suspect_sample_ids,
                              num_intervals=50):
    """Reuse the authoritative MI_check quantizer; requires the project runtime.

    Logits and one-hot labels come from MI_check.collect_logits on the SAME
    ordered clean images. For wrapped AT models, the caller must apply each
    model's expected normalization exactly once. Returns original MI_check
    marginals as well as float64 count estimates and their residuals.
    """
    import torch
    from MI_check import mi_from_logits

    if isinstance(num_intervals, bool) or not isinstance(num_intervals, int) or num_intervals < 1:
        raise ValueError("num_intervals must be a positive integer")
    if victim_logits.ndim != 2 or victim_logits.shape != suspect_logits.shape:
        raise ValueError("Both logits tensors must have the same [N, C] shape")
    n, c = victim_logits.shape
    if n == 0 or c < 2:
        raise ValueError("Expected nonempty classification logits")
    _check_pairing(victim_sample_ids, suspect_sample_ids, n)
    for logits in (victim_logits, suspect_logits):
        if not torch.isfinite(logits).all():
            raise ValueError("Logits must be finite")
    for labels in (victim_label_matrix, suspect_label_matrix):
        if labels.shape != (n, c):
            raise ValueError("One-hot labels must have shape [N, C]")
        if not ((labels == 0) | (labels == 1)).all() or not (labels.sum(dim=1) == 1).all():
            raise ValueError("Labels must be one-hot ground-truth labels")
    if not torch.equal(victim_label_matrix.cpu(), suspect_label_matrix.cpu()):
        raise ValueError("Ground-truth labels must match for paired inputs")

    vx, vy, vd = mi_from_logits(
        victim_logits, victim_label_matrix.to(victim_logits.device),
        num_intervals=num_intervals, verbose=True,
    )
    sx, sy, sd = mi_from_logits(
        suspect_logits, suspect_label_matrix.to(suspect_logits.device),
        num_intervals=num_intervals, verbose=True,
    )
    result = decomposition_from_states(
        vd["inverse_idx"], sd["inverse_idx"],
        victim_label_matrix.argmax(dim=1).detach().cpu().numpy(),
        victim_sample_ids=victim_sample_ids, suspect_sample_ids=suspect_sample_ids,
    )
    result["num_intervals"] = num_intervals
    result["MI_check_marginals"] = {"I_X_V": vx, "I_V_Y": vy, "I_X_S": sx, "I_S_Y": sy}
    result["MI_check_marginal_residuals"] = {
        key: result[key] - value for key, value in result["MI_check_marginals"].items()
    }
    return result
