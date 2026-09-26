"""Shared pieces for "attack + added adversarial term" training (one-stage AT).

Every family script (main_ft_at.py, later main_prune_at.py / main_kd_at.py /
main_knockoff_extraction_at.py) trains on

    L = clean_weight * L_family(f(x), t) + adv_weight * L_family(f(x_adv), t)

with the weights used AS GIVEN (no normalisation). The advisor's "add one
term" form is clean_weight = 1, adv_weight = lambda. x_adv is generated
against the model being trained; t is the family's own target on the clean x.

BatchNorm policy (`bn_policy`; attack generation always runs in eval mode, as
every util_adv epoch function does). Default: "frozen".
  * "both"       both loss forwards run in train mode and BOTH update
                 running_mean / running_var. This is util_adv.at_one_epoch's
                 mixed mode and the standard Madry / TRADES / ARD practice
                 when training FROM SCRATCH; see the diagnosis below for why
                 it fails when fine-tuning a non-robust model.
  * "clean"      adversarial forward in train mode under freeze_bn_stats, so
                 only clean inputs update BN (util_adv.at_one_epoch_clean_bn).
  * "clean_eval" adversarial forward in eval mode
                 (util_adv.at_one_epoch_clean_bn_eval_aligned).
  * "frozen"     every BatchNorm layer stays in eval mode for BOTH forwards:
                 running statistics are never updated (they remain the
                 starting model's clean statistics), affine parameters train.
                 The trained, attacked and deployed functions are then one and
                 the same. Standard "freeze BN" fine-tuning practice.
  With no clean term (clean_weight == 0) the first three reduce to the pure
  mode (adversarial forward in train mode, updating BN, as main_at.py does);
  "frozen" stays frozen.

Diagnosis behind "frozen" (2026-09-22, FT-AL lambda=1 eps 8/255, 25k, lr 0.001):
  under "both" the SAME weights give 91% clean / 0.4% robust with clean BN
  statistics and 29% clean / 15% robust with adversarial statistics; the
  network learned a batch-statistics switch, not robust features (AdvProp's
  observation). Under "clean_eval" recalibrating BN on clean data restores the
  clean accuracy (16% -> 87% on a 30-epoch at_evasion checkpoint) but the
  robustness gained is small (7%), because the attack was generated against a
  model whose running statistics had drifted away from its weights.

Earlier evidence (saved_logs/at_evasion, DFMS base model, eps 8/255): mix=0.5/0.8
runs under bn=clean_eval collapse to ~10% clean accuracy from epoch 0 (final
16% / 7% robust); bn=clean keeps 89% clean but ends with 0.5% robust accuracy at
inference; the both-in-train-mode loop ends at 79% clean / 33% robust in the
log, but see the switch diagnosis above before trusting that number.

The caller owns optimizer.zero_grad() / loss.backward() / optimizer.step().
"""
import re

import torch
import torch.nn as nn

from util_adv import freeze_bn_stats, parse_attack_configs

AT_EXTRA_COLUMNS = ['Robust_Acc', 'Train_Clean_Acc', 'Train_Adv_Acc', 'LR', 'Epoch_Seconds',
                    'AT_Attack', 'AT_eps', 'AT_steps', 'AT_lambda', 'AT_bn', 'Robust_N']
BN_POLICIES = ("frozen", "both", "clean", "clean_eval")


def set_bn_eval(model):
    """Put every BatchNorm layer in eval mode (running stats frozen, affine still trains)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def logits_of(outputs):
    """Plain logits, or the DeiT-distilled train-mode pair averaged as util.ft_one_epoch does."""
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 2:
            return (outputs[0] + outputs[1]) / 2.0
        return outputs[0]
    return outputs


def make_adv_generator(attack_fn, attack_kwargs, target_fn=None):
    """gen(model, x, target) -> x_adv, with the family's target mapped by target_fn."""
    target_fn = target_fn or (lambda t: t)

    def gen(model, x, target):
        return attack_fn(model, x, target_fn(target), **attack_kwargs)
    return gen


def mixed_adversarial_step(model, images, target, loss_fn, adv_generator,
                           clean_weight, adv_weight, post_train_hook=None, logits_fn=logits_of,
                           bn_policy="frozen"):
    """Forward both terms for one batch and return the weighted loss (no backward).

    `model` is the NormalizedModel-wrapped network fed raw [0,1] images.
    `post_train_hook` runs right after model.train() (FT-LL uses it to pin the
    frozen backbone's BatchNorm/Dropout back to eval, as util.ft_one_epoch does).
    `bn_policy` is one of BN_POLICIES; see the module docstring.
    """
    if clean_weight < 0 or adv_weight < 0 or clean_weight + adv_weight <= 0:
        raise ValueError("clean_weight and adv_weight must be >= 0 with a positive sum.")
    if bn_policy not in BN_POLICIES:
        raise ValueError(f"bn_policy must be one of {BN_POLICIES}, got {bn_policy!r}")

    adv_images = adv_logits = adv_loss = clean_logits = clean_loss = None
    mixed = clean_weight > 0 and adv_weight > 0
    if adv_weight > 0:
        model.eval()                                          # generation: inference-time BN, no stat update
        adv_images = adv_generator(model, images, target).detach()
        if mixed and bn_policy == "clean_eval":
            adv_logits = logits_fn(model(adv_images))        # eval-mode forward, gradients kept
            adv_loss = loss_fn(adv_logits, target)

    model.train()
    if post_train_hook is not None:
        post_train_hook()
    if bn_policy == "frozen":
        set_bn_eval(model)                                   # both forwards use the fixed running stats
    if clean_weight > 0:
        clean_logits = logits_fn(model(images))              # train mode: updates BN (unless frozen)
        clean_loss = loss_fn(clean_logits, target)
    if adv_weight > 0 and adv_loss is None:
        if mixed and bn_policy == "clean":
            with freeze_bn_stats(model):                     # batch statistics, running stats untouched
                adv_logits = logits_fn(model(adv_images))
        else:                                                # "both" / "frozen", or pure adversarial mode
            adv_logits = logits_fn(model(adv_images))
        adv_loss = loss_fn(adv_logits, target)

    loss = None
    if clean_loss is not None:
        loss = clean_weight * clean_loss
    if adv_loss is not None:
        loss = adv_weight * adv_loss if loss is None else loss + adv_weight * adv_loss
    return {"loss": loss, "clean_logits": clean_logits, "adv_logits": adv_logits,
            "clean_loss": clean_loss, "adv_loss": adv_loss}


def parse_additive_at_config(exp_yaml):
    """Read the AdversarialTraining/Attack blocks of a one-stage plan.

    AdversarialTraining:
      form: additive      # L = L_family(x) + lambda * L_family(x_adv)
      lambda: 1.0
      bn_policy: frozen   # frozen | both | clean | clean_eval (see module docstring)
    Attack: {name: PGD, eps: ..., steps: ...}   # main_at.py grid semantics

    Returns (attack_configs, clean_weight=1.0, adv_weight=lambda, bn_policy).
    """
    at_cfg = exp_yaml.get("AdversarialTraining") or {}
    if not isinstance(at_cfg, dict):
        raise ValueError("AdversarialTraining must be a YAML mapping.")
    form = str(at_cfg.get("form", "additive")).lower()
    if form != "additive":
        raise ValueError(f"Unsupported AdversarialTraining.form: {form!r} (only 'additive').")
    try:
        lam = float(at_cfg.get("lambda", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("AdversarialTraining.lambda must be a non-negative number.") from exc
    if lam < 0:
        raise ValueError("AdversarialTraining.lambda must be >= 0.")
    bn_policy = str(at_cfg.get("bn_policy", "frozen")).lower()
    if bn_policy not in BN_POLICIES:
        raise ValueError(f"AdversarialTraining.bn_policy must be one of {BN_POLICIES}.")
    return parse_attack_configs(exp_yaml), 1.0, lam, bn_policy


def at_suffix(attack_name, attack_kwargs, lam, run_tag, bn_policy="frozen", extra=None):
    """Name suffix appended to the family's own scenario name.

    `extra` (ordered dict of short key -> value) adds family-specific fields
    between the bn policy and the run tag, e.g. {"inner": "hard"} for knockoff.
    """
    if not re.fullmatch(r"[A-Za-z0-9-]{1,32}", run_tag):
        raise ValueError("run_tag must contain 1-32 letters, digits or hyphens.")
    if bn_policy not in BN_POLICIES:
        raise ValueError(f"bn_policy must be one of {BN_POLICIES}, got {bn_policy!r}")
    param_str = "_".join(f"{k}={v}" for k, v in sorted(attack_kwargs.items()))
    extra_str = "".join(f"_{k}={v}" for k, v in (extra or {}).items())
    return f"_AT{attack_name}_{param_str}_lambda={lam:g}_bn={bn_policy}{extra_str}_run={run_tag}"


def filter_attacks(attack_configs, eps_values):
    """Keep only the attack configs whose eps is in eps_values (None = keep all)."""
    if eps_values is None:
        return list(attack_configs)
    wanted = [float(e) for e in eps_values]
    kept = [cfg for cfg in attack_configs
            if any(abs(float(cfg[2].get("eps", float("nan"))) - e) < 1e-9 for e in wanted)]
    if not kept:
        raise ValueError(f"No attack config in the plan matches eps {wanted}")
    return kept


@torch.no_grad()
def batch_accuracy(logits, target):
    return (logits.argmax(1) == target).sum().item()
