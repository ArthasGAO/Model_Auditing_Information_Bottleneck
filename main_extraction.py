import csv
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import calculate_MI
import util

STAGE = "extraction"
DETERMINISTIC = True           
BEST_CKPT_START_FRAC = 0.0  
BATCH_SIZE = 128
FIDELITY_COLUMNS = ["model_name", "victim_test_acc", "substitute_best_acc", "accuracy_recovery", "fidelity"]


class SoftLabeledSubset(Dataset):
    """Pairs augmented_dataset[i] with soft_labels[i]; both must follow the same index list."""

    def __init__(self, aug_dataset, soft_labels):
        assert len(aug_dataset) == len(soft_labels), f"length mismatch: {len(aug_dataset)} vs {len(soft_labels)}"
        self.aug_dataset = aug_dataset
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.aug_dataset)

    def __getitem__(self, idx):
        img, _ = self.aug_dataset[idx]           # discard the hard label
        return img, self.soft_labels[idx]


def soft_label_loss(logits, soft_targets):
    """KL(soft_targets || softmax(logits)), averaged over the batch."""
    return nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(logits, dim=1), soft_targets)


def load_victim(plan):
    victim_ds, num_classes, _ = util.build_dataset_from_yaml(util.victim_dataset_cfg(plan))
    victim = util.build_model(util.victim_arch(plan), num_classes).to(util.device)
    ckpt = util.victim_checkpoint(plan)
    victim.load_state_dict(util.load_state(ckpt, map_location=util.device))
    victim.eval()
    print(f"[victim] {util.victim_scenario(plan)} loaded from {ckpt}")
    return victim, victim_ds, num_classes


def query_set(plan):
    """(aux dataset wrapper, query indices): group_B of the auxiliary dataset."""
    aux_cfg = plan.get("Auxiliary_Dataset", util.victim_dataset_cfg(plan))
    aux_ds, aux_classes, aux_group_size = util.build_dataset_from_yaml(aux_cfg)
    group_A = util.load_group_A(aux_ds, aux_cfg, aux_group_size, aux_classes)
    group_B = util.load_group_B(aux_ds, aux_cfg, group_A, aux_group_size, aux_classes, overlap_rate=util.NEGATIVE_RATE)
    return aux_ds, group_B


def train_extraction(plan, plan_path, seed, *, workers=4, epochs=None, skip_existing=True, measure_mi=True):
    """Query the victim once, train one substitute, record fidelity, measure MI."""
    ids = {"seed": seed, "rate": util.VICTIM_RATE}
    name = util.model_name(STAGE, plan, **ids)
    save_dir = util.model_dir(STAGE, plan, **ids)
    best = save_dir / util.BEST_CHECKPOINT

    if skip_existing and util.checkpoint_exists(best):
        print(f"[SKIP] {name}: {best} exists")
    else:
        util.set_seed(seed, deterministic=DETERMINISTIC)
        generator = torch.Generator()
        generator.manual_seed(seed)
        victim, victim_ds, num_classes = load_victim(plan)
        aux_ds, query_idx = query_set(plan)

        # ---- query the victim on clean images ----
        clean_queries = aux_ds.subset("train", query_idx, clean=True)
        query_loader = DataLoader(clean_queries, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        stolen_inputs, soft_labels = util.query_victim(victim, query_loader, util.device)
        print(f"[query] {soft_labels.shape[0]} query-response pairs, {soft_labels.shape[1]} classes")
        for i in (0, len(query_idx) // 2, len(query_idx) - 1):      # clean and augmented views share indices
            img_clean, _ = clean_queries[i]
            assert torch.allclose(img_clean, stolen_inputs[i], atol=1e-6), f"index alignment broken at i={i}"
        del stolen_inputs

        # ---- substitute training data: augmented views of the same images, soft labels ----
        stolen = SoftLabeledSubset(aux_ds.subset("train", query_idx, clean=False), soft_labels)
        trainloader = DataLoader(stolen, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers,
                                 worker_init_fn=util.seed_worker, generator=generator,
                                 persistent_workers=workers > 0, pin_memory=True)
        testloader = DataLoader(victim_ds.test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        substitute = util.build_model(util.suspect_arch(STAGE, plan), num_classes).to(util.device)
        opt_cfg, sched_cfg = plan.get("Optimizer", {}), plan.get("Scheduler", {}) or {}
        n_epochs = plan.get("Epochs", 100) if epochs is None else epochs
        optimizer, scheduler = util.build_optimizer_scheduler(
            substitute.parameters(), opt_cfg.get("name", "Adam"), opt_cfg.get("params", {"lr": 1e-3}),
            sched_cfg.get("name"), sched_cfg.get("params", {}), n_epochs)
        eval_criterion = nn.CrossEntropyLoss()
        victim_acc = util.evaluate1(victim, testloader, eval_criterion, util.device)["test_acc"]
        print(f"[victim] test acc {victim_acc:.2f}%")

        def step(epoch):
            return util.train_one_epoch_knockoff(substitute, trainloader, optimizer, soft_label_loss, epoch, util.device)

        print(f"\n[{STAGE}] {name}: substitute {util.suspect_arch(STAGE, plan)}, {n_epochs} epoch(s)")
        result = util.run_training(
            scenario_name=name, epochs=n_epochs, step_fn=step, eval_net=substitute, testloader=testloader,
            criterion=eval_criterion, optimizer=optimizer, scheduler=scheduler, save_dir=save_dir,
            log_file=util.training_log_path(STAGE, name), best_ckpt_start_frac=BEST_CKPT_START_FRAC)

        # ---- fidelity of the best substitute to the victim ----
        substitute.load_state_dict(util.load_state(best, map_location=util.device))
        substitute.eval()
        fidelity = util.evaluate_fidelity(victim, substitute, testloader, util.device)
        recovery = 100.0 * result["best_test_acc"] / victim_acc
        print(f"==> {name}: victim {victim_acc:.2f}%  substitute best {result['best_test_acc']:.2f}%  "
              f"recovery {recovery:.1f}%  fidelity {fidelity:.1f}%")
        summary = util.stage_log_root(STAGE) / "fidelity.csv"
        new_file = not summary.exists()
        with summary.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIDELITY_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow({"model_name": name, "victim_test_acc": f"{victim_acc:.4f}",
                             "substitute_best_acc": f"{result['best_test_acc']:.4f}",
                             "accuracy_recovery": f"{recovery:.4f}", "fidelity": f"{fidelity:.4f}"})
        del victim, substitute, optimizer, scheduler, trainloader, testloader, stolen, soft_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if measure_mi:
        util.set_seed(calculate_MI.SUBSET_SEED, deterministic=True)
        calculate_MI.measure_checkpoint(STAGE, plan, best, ids)
    return save_dir


def run(argv=None):
    args = util.apply_common_args(util.build_argparser(STAGE, __doc__).parse_args(argv))
    plans = util.plan_files(STAGE, args.plans, args.plan_dir)
    if not plans:
        raise SystemExit(f"No plan files in {args.plan_dir or util.stage_plan_dir(STAGE)} matching {list(args.plans)}")
    seeds = util.parse_seeds(args.seeds) if args.seeds else util.DEFAULT_SEEDS[STAGE]

    loaded = [(p, util.load_plan(p, args.data_root)) for p in plans]
    cells = [(p, plan, ids) for p, plan in loaded for ids in util.model_grid(STAGE, plan, seeds=seeds)]
    done = sum(util.checkpoint_exists(util.model_dir(STAGE, plan, **ids) / util.BEST_CHECKPOINT) for _, plan, ids in cells)
    print(f"[{STAGE}] {len(plans)} plan(s): " + ", ".join(p.name for p in plans))
    print(f"[{STAGE}] seeds {seeds}, deterministic {DETERMINISTIC}, workers {args.workers}"
          + (f", epochs override {args.epochs}" if args.epochs else ""))
    print(f"[{STAGE}] TOTAL {len(cells)} cell(s): {done} already trained, "
          f"{len(cells) - done if not args.no_skip else len(cells)} to train; MI {'off' if args.no_mi else 'on'}")

    for plan_path, plan, ids in cells:
        print(f"\n========== {plan_path.name}  seed {ids['seed']} ==========")
        train_extraction(plan, plan_path, ids["seed"], workers=args.workers, epochs=args.epochs,
                         skip_existing=not args.no_skip, measure_mi=not args.no_mi)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    run()
