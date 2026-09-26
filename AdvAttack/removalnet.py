#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
RemovalNet fingerprint-removal attack, ported into the E:/Experiment framework.

Core algorithm (feature_maximize / logit_maximize / deepremoval_step) is a faithful
copy of the author's original at:
    RemovalNet/attack/RemovalNet/removalnet.py
    (Yao et al., "RemovalNet: DNN Fingerprint Removal Attacks", IEEE TDSC 2023)

Only the I/O seams are adapted to this framework:
  - intermediate-feature access is attached to the plain Model/ResNet_18.ResNet
    instance via `attach_resnet_feature_methods` (the author's own monkey-patch pattern),
    with the CIFAR stem (NO maxpool) instead of the ImageNet stem;
  - evaluation uses this framework's `evaluate1` (passed in), logging is CSV,
    checkpoints follow the saved_models/<task>/<scenario>/ convention;
  - the per-dataset auto-hyperparameter selection and the random LR jitter from the
    author's get_args() are NOT replicated; hyperparameters come from the YAML so runs
    stay reproducible under torch.use_deterministic_algorithms(True).

Author's ResNet layer-index scheme (kept faithfully):
    layer_index 1 -> stem (conv1+bn1+relu)        [NO maxpool here: CIFAR ResNet]
    layer_index 2 -> layer1
    layer_index 3 -> layer2
    layer_index 4 -> layer3
    layer_index 5 -> layer4
"""

import os
import csv
import math
import types
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim


# ----------------------------------------------------------------------------
# Feature-access methods attached to a plain ResNet (Model/ResNet_18.ResNet).
# This mirrors the author's resnet.py, but layerx1 omits the maxpool because the
# E:/Experiment ResNet is CIFAR-style (no maxpool in its forward path), so that
# fed_forward(.) + mid_forward(.) compose into exactly the model's normal forward().
# ----------------------------------------------------------------------------
def attach_resnet_feature_methods(model):
    def layerx1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x.contiguous()

    def layerx2(self, x):
        return self.layer1(x).contiguous()

    def layerx3(self, x):
        return self.layer2(x).contiguous()

    def layerx4(self, x):
        return self.layer3(x).contiguous()

    def layerx5(self, x):
        return self.layer4(x).contiguous()

    def fed_forward(self, x, layer_index):
        """Feed x from the head down to $layer_index, return that activation."""
        x = self.layerx1(x)
        if layer_index == 1:
            return x.contiguous()
        x = self.layerx2(x)
        if layer_index == 2:
            return x.contiguous()
        x = self.layerx3(x)
        if layer_index == 3:
            return x.contiguous()
        x = self.layerx4(x)
        if layer_index == 4:
            return x.contiguous()
        x = self.layerx5(x)
        if layer_index == 5:
            return x.contiguous()
        return x.contiguous()

    def mid_forward(self, x, layer_index):
        """Feed an activation captured at $layer_index onward to the logits."""
        if layer_index == 1:
            x = self.layerx2(x)
            x = self.layerx3(x)
            x = self.layerx4(x)
            x = self.layerx5(x)
        if layer_index == 2:
            x = self.layerx3(x)
            x = self.layerx4(x)
            x = self.layerx5(x)
        if layer_index == 3:
            x = self.layerx4(x)
            x = self.layerx5(x)
        if layer_index == 4:
            x = self.layerx5(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x.contiguous()

    model.layerx1 = types.MethodType(layerx1, model)
    model.layerx2 = types.MethodType(layerx2, model)
    model.layerx3 = types.MethodType(layerx3, model)
    model.layerx4 = types.MethodType(layerx4, model)
    model.layerx5 = types.MethodType(layerx5, model)
    model.fed_forward = types.MethodType(fed_forward, model)
    model.mid_forward = types.MethodType(mid_forward, model)
    return model


def batch_fed_forward(model, x, layer_index, batch_size=200):
    """Faithful copy of the author's attack/ops.batch_fed_forward (memory-safe)."""
    steps = math.ceil(len(x) / batch_size)
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for step in range(steps):
            off = step * batch_size
            batch_x = x[off: off + batch_size].clone().to(device)
            batch_out = model.fed_forward(batch_x, layer_index=layer_index).detach().cpu()
            outputs.append(batch_out)
        outputs = torch.cat(outputs).detach().cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def _as_logits(out):
    """Collapse (cls, dist) tuple/list model outputs (e.g. distilled DeiT) to a
    single logits tensor. No-op for plain tensor outputs (ResNet, plain DeiT)."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def view_learning_state(data, file_path, fontsize=30):
    """Faithful port of the author's utils/vis.view_learning_state.

    Saves one PDF per tracked key as `<file_path>_<KEY>.pdf` (e.g. LR_ACC.pdf,
    LR_LOSS_DIST.pdf, LR_LOSS_KL.pdf, LR_LOSS_CE.pdf). Only deviation from the
    author: text annotations use the actual recorded iteration (data["t"]) rather
    than the hard-coded index*20, so positions stay correct for any test_interval.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    for key in data["keys"]:
        if len(data.get(key, [])) == 0:
            continue
        plt.figure(figsize=(16, 12), dpi=100)
        plt.cla()
        plt.grid()
        x = np.array(data["t"], dtype=np.int32)
        y = np.array(data[key], dtype=np.float64)

        plt.plot(x, y, label="Surrogate Model", linewidth=5, marker="*", markersize=10, linestyle="solid")
        plt.plot(x, np.repeat(np.min(y), len(x)), color="black", linewidth=3, linestyle="dashdot")
        plt.text(float(x[int(np.argmin(y))]), float(np.min(y)) - 3, round(float(np.min(y)), 2), fontsize=fontsize - 5)
        plt.plot(x, np.repeat(np.max(y), len(x)), color="black", linewidth=3, linestyle="dashdot")
        plt.text(float(x[int(np.argmax(y))]), float(np.max(y)) + 3, round(float(np.max(y)), 2), fontsize=fontsize - 5)
        last_y = float(y[-1])
        plt.text(float(x[-1]), last_y + 3, round(last_y, 2), fontsize=fontsize - 5)

        plt.xlabel("Iteration", fontsize=fontsize)
        plt.ylabel(key.upper(), fontsize=fontsize)
        if key == "acc":
            plt.ylim(0, 100.0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="best", numpoints=1, fontsize=fontsize)
        fpath = file_path + f"_{key.upper()}.pdf"
        plt.savefig(fpath)
        plt.close()
        print(f"-> saving fig: {fpath}")


class RemovalNet:
    """Faithful port of the author's RemovalNet attack class.

    Args:
        model_T:   frozen victim (oracle that supplies pseudo-labels)
        model_t:   trainable copy (the surrogate being purified)
        train_loader: attacker's surrogate data (images; labels are ignored/overwritten)
        test_loader:  victim's clean test set (for accuracy logging)
        cfg:       namespace with layer/ydist/alpha/beta/gamma/T/poison_steps/
                   shuffle_ratio/lr/momentum/weight_decay/iterations/test_interval/
                   save_interval/batch_size
        device, criterion, evaluate_fn: framework-provided I/O
        log_file, save_dir, scenario_name: output bookkeeping
    """

    def __init__(self, model_T, model_t, train_loader, test_loader, cfg, device,
                 criterion, evaluate_fn, log_file, save_dir, scenario_name, plot_dir=None):
        self.cfg = cfg
        self.device = device
        self.batch_size = int(cfg.batch_size)
        self.model_T = model_T.to(device)
        self.model_t = model_t.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.evaluate_fn = evaluate_fn
        self.log_file = log_file
        self.save_dir = save_dir
        self.scenario_name = scenario_name
        self.plot_dir = plot_dir
        # author's per-iteration learning-curve tracker (acc + 3 component losses)
        self.learning_data = {
            "t": [], "acc": [], "loss_dist": [], "loss_kl": [], "loss_ce": [],
            "keys": ["acc", "loss_dist", "loss_kl", "loss_ce"],
        }
        os.makedirs(self.save_dir, exist_ok=True)
        if self.plot_dir is not None:
            os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(osp.dirname(self.log_file), exist_ok=True)
        if not osp.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["Scenario", "Step", "Train_Loss", "Loss_Dist", "Loss_KL", "Loss_CE",
                     "Test_Loss", "Test_Acc", "Test_Precision", "Test_Recall", "Test_F1"])

    # ------------------------------------------------------------------ #
    # The following four methods are copied verbatim from the author.
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize(vs):
        return [(v - torch.min(v)) / (torch.max(v) - torch.min(v) + 1e-6) for v in vs]

    @staticmethod
    def distance_cost(a, b, ydist):
        batch_size = len(a)
        if ydist == "l2":
            loss_dist = torch.norm(a - b, p=2)
        elif ydist == "cosine":
            loss_dist = F.cosine_similarity(a.view(batch_size, -1), b.view(batch_size, -1), dim=1).sum()
        elif ydist == "kl":
            loss_dist = F.kl_div(a, b, reduction="sum")
        elif ydist == "angle":
            loss_dist = (a / a.norm() - b / b.norm()).norm()
        else:
            raise NotImplementedError()
        return loss_dist

    def shuffle_features(self, outputs, ratio=1.0):
        """Shuffle a random subset of channels per sample (author's version)."""
        channels = outputs[0].shape[0]
        size = math.ceil(channels * ratio)
        for i, layer in enumerate(outputs):
            idx = np.arange(0, channels)
            np.random.shuffle(idx)
            idx_rnd = idx[:size]
            idx_ord = np.sort(idx_rnd)
            outputs[i][idx_ord] = layer[idx_rnd].clone()
        return outputs.detach()

    def feature_maximize(self, t, model, x, y, l, lr=0.01, shuffle_ratio=0.1,
                         poison_steps=20, ydist="l2"):
        # z = f(x)^{l}  (latent space); craft z_prime pushed away from z but still correct
        z = batch_fed_forward(model, x, layer_index=l, batch_size=self.batch_size).detach().contiguous().to(self.device)

        ''' Add a highly non-linear random noise matrix using the tangent of a uniform distribution
            in order to violently kicks the starting point out of any immdiate local minimum, giving the
            gradient descent a better chance of finding a truly distinct representation'''
        z_prime = z.clone() + torch.tan(torch.rand(z.shape, device=self.device))
        self.shuffle_features(z_prime, ratio=shuffle_ratio)
        for step in range(poison_steps):
            z_prime = self.shuffle_features(z_prime, ratio=shuffle_ratio).detach()
            z_prime.requires_grad = True
            logit = model.mid_forward(z_prime, layer_index=l)

            loss_dist = 0.1 * torch.log(self.distance_cost(z_prime, z, ydist))
            loss_ce = F.cross_entropy(logit, y)
            loss = loss_ce - loss_dist
            grad = torch.autograd.grad(loss, [z_prime], retain_graph=False, create_graph=False)[0]
            z_prime = z_prime - lr * grad.sign() # It ensures the embedded watermark is thoroughly scrambled across the entire latent space, 
                                                 # rather than just breaking a few isolated nodes that happen to have high gradient values.
            z_prime = z_prime.detach()
        return z_prime.clone().detach()

    def logit_maximize(self, t, model, x, y, ydist="l2"):
        with torch.no_grad():
            logits = F.softmax(_as_logits(model(x)), dim=1) # this is actually the prediction vector
            batch_size = len(logits)
            if ydist == "l2":
                dist = torch.cdist(logits, logits)
                idxs = torch.argmax(dist, dim=1).tolist()
            elif ydist == "cosine":
                from torchmetrics.functional import pairwise_cosine_similarity  # lazy: avoid hard dep
                dist = pairwise_cosine_similarity(logits, logits)
                idxs = torch.argmin(dist, dim=1).tolist()
            else:
                raise NotImplementedError()
            logits_prime = logits.clone()
            for i in range(batch_size):
                for beta in np.arange(0.3, 1, 0.1):
                    out = beta * logits[i] + (1 - beta) * logits[idxs[i]]
                    if out.argmax(dim=0) == y[i]:
                        logits_prime[i] = out
                        break
            return logits_prime.detach()

    def deepremoval_step(self, model_T, model_t, optimizer, x, y, l, t=0):
        cfg = self.cfg
        model_T.eval()
        model_t.eval()

        # remove feature-level fingerprints
        feats_prime = self.feature_maximize(
            t, model=model_t, x=x, y=y, l=l, ydist=cfg.ydist,
            shuffle_ratio=cfg.shuffle_ratio, poison_steps=cfg.poison_steps).detach()
        # remove logit-level fingerprints
        logit_prime = self.logit_maximize(t, model=model_t, x=x, y=y, ydist=cfg.ydist).detach()

        model_t.train()
        optimizer.zero_grad()
        logit = model_t(x)
        feats = model_t.fed_forward(x, layer_index=l)

        loss_dist = F.mse_loss(feats, feats_prime, reduction="mean")
        loss_kl = cfg.T * cfg.T * F.kl_div(
            F.log_softmax(logit / cfg.T, dim=1),
            F.softmax(logit_prime / cfg.T, dim=1), reduction='batchmean')
        loss_ce = F.cross_entropy(logit, y)

        loss = cfg.alpha * loss_kl + cfg.beta * loss_dist + cfg.gamma * loss_ce
        loss.backward()
        optimizer.step()
        return loss, loss_dist, loss_kl, loss_ce

    def deepremoval_step_logit_only(self, model_T, model_t, optimizer, x, y, l, t=0):
        """Ablation: LOGIT-LEVEL ONLY (decision-boundary) removal.

        Skips feature_maximize / fed_forward entirely (no white-box layer access),
        so the objective is  loss = alpha*loss_kl + gamma*loss_ce.  loss_dist is
        reported as 0 so the CSV/plot columns stay aligned with the full attack.
        """
        cfg = self.cfg
        model_T.eval()
        model_t.eval()

        # remove logit-level fingerprints only
        logit_prime = self.logit_maximize(t, model=model_t, x=x, y=y, ydist=cfg.ydist).detach()

        model_t.train()
        optimizer.zero_grad()
        logit = _as_logits(model_t(x))

        loss_dist = torch.zeros((), device=self.device)  # no latent term in this mode
        loss_kl = cfg.T * cfg.T * F.kl_div(
            F.log_softmax(logit / cfg.T, dim=1),
            F.softmax(logit_prime / cfg.T, dim=1), reduction='batchmean')
        loss_ce = F.cross_entropy(logit, y)

        loss = cfg.alpha * loss_kl + cfg.gamma * loss_ce
        loss.backward()
        optimizer.step()
        return loss, loss_dist, loss_kl, loss_ce

    def deepremoval_step_ce_only(self, model_T, model_t, optimizer, x, y, l, t=0):
        """Ablation: CE-ONLY (no removal terms at all).

        Equivalent to RemovalNet with alpha=beta=0 -> a plain fine-tune of the
        surrogate on the victim's hard labels (`y` is the victim's argmax, set in
        the loop), run through the same iteration-based pipeline. No
        feature_maximize / logit_maximize. loss = gamma * CE. loss_dist and
        loss_kl are reported as 0 so the CSV/plot columns stay aligned.
        """
        cfg = self.cfg
        model_t.train()
        optimizer.zero_grad()
        logit = _as_logits(model_t(x))

        loss_dist = torch.zeros((), device=self.device)  # no latent term
        loss_kl = torch.zeros((), device=self.device)    # no logit-removal term
        loss_ce = F.cross_entropy(logit, y)

        loss = cfg.gamma * loss_ce
        loss.backward()
        optimizer.step()
        return loss, loss_dist, loss_kl, loss_ce

    # ------------------------------------------------------------------ #
    # Main loop: iteration-based (faithful to the author's deepremoval()).
    # I/O (eval / CSV / checkpoint) adapted to this framework.
    # ------------------------------------------------------------------ #
    def deepremoval(self):
        cfg = self.cfg
        iterations = cfg.iterations + 1

        # optimizer is SGD by default (faithful to the author / ResNet path);
        # transformers (DeiT) typically need AdamW -> selectable via cfg.optimizer.
        opt_name = str(getattr(cfg, "optimizer", "sgd")).lower()
        if opt_name == "adamw":
            optimizer = optim.AdamW(self.model_t.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif opt_name == "adam":
            optimizer = optim.Adam(self.model_t.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            optimizer = optim.SGD(self.model_t.parameters(), lr=cfg.lr,
                                  momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cfg.iterations * 1.5))

        ce_only = bool(getattr(cfg, "ce_only", False))
        logit_only = bool(getattr(cfg, "logit_only", False))
        if ce_only:
            step_fn = self.deepremoval_step_ce_only
            mode_str = "CE-ONLY (fine-tune on victim labels, no removal terms)"
        elif logit_only:
            step_fn = self.deepremoval_step_logit_only
            mode_str = "LOGIT-ONLY (no layer term)"
        else:
            step_fn = self.deepremoval_step
            mode_str = "FULL (logit+layer)"
        print(f"[RemovalNet] mode={mode_str}")

        loader = iter(self.train_loader)
        for step in range(0, 1 + iterations):
            try:
                batch, label = next(loader)
            except StopIteration:
                loader = iter(self.train_loader)
                batch, label = next(loader)
            x = batch.to(self.device)
            # labels come from the VICTIM oracle, not ground truth (author's design)
            y = self.model_T(x).argmax(dim=1).long().detach()

            l = int(cfg.layer)
            loss, loss_dist, loss_kl, loss_ce = step_fn(
                self.model_T, self.model_t, optimizer, x=x, y=y, l=l, t=step)
            scheduler.step()

            if step == 0 or step % cfg.test_interval == 0 or step == iterations - 1:
                vloss, vdist, vkl, vce = (float(loss.detach()), float(loss_dist.detach()),
                                          float(loss_kl.detach()), float(loss_ce.detach()))
                test_result = self.evaluate_fn(self.model_t, self.test_loader, self.criterion, self.device)
                self._log_row(step, vloss, vdist, vkl, vce, test_result)
                print(f"[RemovalNet] step={step} lr={optimizer.param_groups[0]['lr']:.6f} "
                      f"loss={vloss:.4f} dist={vdist:.4f} kl={vkl:.4f} "
                      f"ce={vce:.4f} test_acc={test_result['test_acc']:.3f}")

                # author's learning-curve tracking + plots (saved into the saved_logs tree)
                if self.plot_dir is not None:
                    self.learning_data["t"].append(step)
                    self.learning_data["acc"].append(float(test_result["test_acc"]))
                    self.learning_data["loss_dist"].append(vdist)
                    self.learning_data["loss_kl"].append(vkl)
                    self.learning_data["loss_ce"].append(vce)
                    view_learning_state(self.learning_data, file_path=osp.join(self.plot_dir, "LR"))
                    torch.save(self.learning_data, osp.join(self.plot_dir, "learning_state.pt"))

            save_every = int(getattr(cfg, "save_interval", cfg.test_interval))
            if step % save_every == 0 or step == iterations - 1:
                self._save_ckpt(step)
        self._save_ckpt(step)
        print(f"[RemovalNet] done. final model saved under {self.save_dir}")

    def _log_row(self, step, loss, loss_dist, loss_kl, loss_ce, test_result):
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                self.scenario_name, step,
                float(loss), float(loss_dist), float(loss_kl), float(loss_ce),
                test_result["test_loss"], test_result["test_acc"],
                test_result["test_precision"], test_result["test_recall"], test_result["test_f1"],
            ])

    def _save_ckpt(self, step):
        path = osp.join(self.save_dir, f"step_{step}.pth")
        torch.save(self.model_t.state_dict(), path)
        # keep model_t trainable for the next step
        self.model_t.to(self.device).train()
