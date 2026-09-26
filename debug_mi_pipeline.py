"""
Focused diagnostic for the FT-LL constant I(T;Y) bug.

This script prints out everything needed to identify whether:
  (A) Labels match model argmax (pseudo-label leakage)
  (B) num_classes is mismatched (wrong fc layer size)
  (C) Pattern-label alignment is suspiciously perfect

Run this in place of one FT-LL evaluation and inspect the output.
"""

import numpy as np
import torch
import torch.nn.functional as F


def diagnose_mi_pipeline(net, data_loader, device, max_batches=None, num_classes_override=None):
    """Run full diagnostic on one (net, loader) combination."""
    print("=" * 70)
    print("DIAGNOSTIC: MI PIPELINE END-TO-END")
    print("=" * 70)

    # ---- Check model architecture ----
    print("\n[1] Model architecture inspection:")
    if hasattr(net, "fc"):
        print(f"    net.fc.out_features = {net.fc.out_features}")
        print(f"    net.fc.in_features  = {net.fc.in_features}")
    elif hasattr(net, "classifier"):
        from torch.nn import Linear
        linears = [m for m in net.classifier.modules() if isinstance(m, Linear)]
        for i, l in enumerate(linears):
            print(f"    classifier.linear[{i}]: in={l.in_features}, out={l.out_features}")
    if hasattr(net, "num_classes"):
        print(f"    net.num_classes = {net.num_classes}")

    inferred_K = _infer_num_classes(net)
    print(f"    INFERRED num_classes = {inferred_K}")
    if num_classes_override is not None:
        print(f"    OVERRIDE num_classes = {num_classes_override}")
        K_used = num_classes_override
    else:
        K_used = inferred_K

    # ---- Run inference batch by batch with full instrumentation ----
    print(f"\n[2] Running inference with num_classes={K_used}:")
    net.eval()
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for bi, (inputs, targets) in enumerate(data_loader):
            if max_batches is not None and bi >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if bi == 0:
                print(f"\n    --- First batch sanity ---")
                print(f"    inputs.shape:  {tuple(inputs.shape)}")
                print(f"    targets.shape: {tuple(targets.shape)}")
                print(f"    targets dtype: {targets.dtype}")
                print(f"    targets first 20: {targets[:20].cpu().tolist()}")
                print(f"    targets unique: {targets.unique().cpu().tolist()}")
                print(f"    targets.max()=  {targets.max().item()}, "
                      f"targets.min()={targets.min().item()}")
                print(f"    outputs.shape: {tuple(outputs.shape)}")
                print(f"    outputs[0, :10]: {outputs[0, :10].cpu().tolist()}")
                # Model argmax vs targets
                preds = outputs.argmax(dim=1)
                print(f"    preds first 20: {preds[:20].cpu().tolist()}")
                agreement = (preds == targets).float().mean().item()
                print(f"    AGREEMENT (preds == targets): {agreement:.4f}")
                if agreement > 0.99:
                    print(f"    [!!] HIGHLY SUSPICIOUS: model argmax nearly identical to targets")
                    print(f"    [!!] Either: (a) FT-LL was trained on this exact data,")
                    print(f"    [!!]         (b) targets are pseudo-labels = model predictions,")
                    print(f"    [!!]         (c) some other label-leakage")

            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())

    layer_T = torch.cat(all_outputs, dim=0).to(dtype=torch.float32)
    all_targets = torch.cat(all_targets, dim=0)
    N = layer_T.shape[0]

    # ---- Bulk statistics ----
    print(f"\n[3] Bulk statistics (N={N} samples):")

    # Targets distribution
    target_counts = torch.bincount(all_targets, minlength=K_used).cpu().numpy()
    print(f"    target distribution (per class):")
    for c, count in enumerate(target_counts):
        if count > 0 or c < 10:
            print(f"        class {c:>3}: {count} samples")
    nonzero_target_classes = (target_counts > 0).sum()
    print(f"    distinct target classes: {nonzero_target_classes}")

    # Overall preds vs targets
    overall_preds = layer_T.argmax(dim=1)
    overall_agreement = (overall_preds == all_targets).float().mean().item()
    print(f"\n    overall (preds == targets) accuracy: {overall_agreement:.4f}")

    if overall_agreement > 0.99:
        print(f"    [!!] Model has ~100% accuracy on this evaluation set.")
        print(f"    [!!] If this is supposed to be an OUT-OF-DISTRIBUTION dataset,")
        print(f"    [!!] something is wrong (data leakage / wrong dataloader).")

    # ---- Now do the MI pipeline with diagnostics ----
    print(f"\n[4] MI pipeline diagnostic at bins=50:")
    label_matrix = F.one_hot(all_targets, num_classes=K_used).float().to(device)
    print(f"    label_matrix.shape: {tuple(label_matrix.shape)}")
    print(f"    label_matrix sum: {label_matrix.sum().item()} (should be N={N})")
    print(f"    label_matrix.sum(dim=0) (per-class total): "
          f"{label_matrix.sum(dim=0).cpu().tolist()[:15]}")

    # Softmax + binning
    num_intervals = 50
    T_soft = torch.softmax(layer_T, dim=1)
    bins = torch.linspace(0, 1, num_intervals + 1, device=device, dtype=torch.float32)
    T_discrete = torch.bucketize(T_soft, bins, right=True) - 1
    T_discrete = T_discrete.clamp(0, num_intervals - 1).contiguous()

    unique_T, inverse_idx = torch.unique(T_discrete, dim=0, return_inverse=True)
    U = unique_T.shape[0]
    print(f"\n    bins={num_intervals}: U={U} unique patterns")

    # Compute counts
    T_counts = torch.zeros(U, device=device, dtype=torch.float32)
    T_counts.index_add_(0, inverse_idx, torch.ones(N, device=device))

    TY_counts = torch.zeros((U, K_used), device=device, dtype=torch.float32)
    TY_counts.index_add_(0, inverse_idx, label_matrix)

    # Check label purity per pattern
    print(f"\n[5] Per-pattern label purity (top 10 largest patterns):")
    pattern_counts_cpu = T_counts.cpu().numpy()
    label_counts_cpu = TY_counts.cpu().numpy()
    top_pattern_idx = np.argsort(pattern_counts_cpu)[::-1][:10]
    for rank, pi in enumerate(top_pattern_idx):
        size = int(pattern_counts_cpu[pi])
        lbl_dist = label_counts_cpu[pi]
        nonzero_labels = (lbl_dist > 0).sum()
        max_label_share = lbl_dist.max() / size if size > 0 else 0
        modal_class = lbl_dist.argmax()
        print(f"    pattern {pi:>4} (size={size:>5}): "
              f"{nonzero_labels} distinct labels, "
              f"modal class={modal_class} ({lbl_dist[modal_class]:.0f}, "
              f"{max_label_share*100:.1f}% pure)")

    # Compute MI
    p_T = T_counts / N
    mask_T = p_T > 0
    I_X_T = -(p_T[mask_T] * torch.log2(p_T[mask_T])).sum().item()

    TY_matrix = TY_counts / N
    P_T_marg = TY_matrix.sum(dim=1)
    P_Y_marg = TY_matrix.sum(dim=0)

    mi_mask = TY_matrix > 0
    denom = P_T_marg[:, None] * P_Y_marg[None, :]
    ratio = TY_matrix / denom
    log_ratio = torch.log2(ratio)
    I_T_Y = (TY_matrix * log_ratio)[mi_mask].sum().item()

    print(f"\n[6] Final MI:")
    print(f"    I(X;T) = {I_X_T:.6f}")
    print(f"    I(T;Y) = {I_T_Y:.6f}")
    print(f"    log2(num distinct target classes) = "
          f"{np.log2(nonzero_target_classes):.6f}")
    print(f"    log2(K_used) = {np.log2(K_used):.6f}")

    if abs(I_T_Y - np.log2(nonzero_target_classes)) < 1e-3:
        print(f"\n    [!!] I(T;Y) EQUALS log2(distinct_classes) EXACTLY.")
        print(f"    [!!] This means H(Y|T) = 0: every pattern has only one label.")
        print(f"    [!!] Possible causes:")
        print(f"    [!!]   - Model has 100% accuracy on this set (memorization/leakage)")
        print(f"    [!!]   - Labels are pseudo-labels = model predictions")


def _infer_num_classes(net):
    if hasattr(net, "fc"):
        return net.fc.out_features
    elif hasattr(net, "classifier"):
        import torch.nn as nn
        last_linear = [m for m in net.classifier.modules()
                       if isinstance(m, nn.Linear)][-1]
        return last_linear.out_features
    elif hasattr(net, "num_classes"):
        return net.num_classes
    raise ValueError("Cannot automatically infer num_classes.")


# ---- Usage example ----
if __name__ == "__main__":
    # Paste this into your main_epochs after loading the model:
    #
    #     from debug_mi_pipeline import diagnose_mi_pipeline
    #     diagnose_mi_pipeline(net, in_loaders[25000], device)
    #
    # It will print everything needed to identify the bug.
    pass