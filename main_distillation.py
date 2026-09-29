import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import calculate_MI
import util

STAGE = "distillation"
DETERMINISTIC = True         
BEST_CKPT_START_FRAC = 0.0 
BATCH_SIZE = 256


def train_distillation(plan, plan_path, seed, *, methods=None, rate=util.NEGATIVE_RATE, workers=4,
                       epochs=None, skip_existing=True, measure_mi=True):
    """Distil one victim into a student with every requested method for one seed."""
    setup = util.process_experiment_kd_setup(plan)          # resolves the teacher = victim checkpoint
    methods = list(setup["KD_Setups"]) if methods is None else list(methods)
    for method in methods:
        if method not in setup["KD_Setups"]:
            raise KeyError(f"{plan_path.name}: no Distillation entry named {method}")
    dataset = setup["Dataset"]
    criterion = nn.CrossEntropyLoss()                        # evaluation only; the distiller owns its loss
    transfer_idx = None

    for method in methods:
        ids = {"method": method, "seed": seed, "rate": rate}
        name = util.model_name(STAGE, plan, **ids)
        save_dir = util.model_dir(STAGE, plan, **ids)
        best = save_dir / util.BEST_CHECKPOINT
        if skip_existing and util.checkpoint_exists(best):
            print(f"[SKIP] {name}: {best} exists")
            continue

        util.set_seed(seed, deterministic=DETERMINISTIC)
        if transfer_idx is None:
            group_A = util.load_group_A(dataset, plan["Dataset"], setup["GroupSize"], setup["NumClasses"])
            transfer_idx = util.load_group_B(dataset, plan["Dataset"], group_A, setup["GroupSize"],
                                             setup["NumClasses"], overlap_rate=rate)
            print(f"[DATA] transfer set: {len(transfer_idx)} images, overlap with group_A "
                  f"{len(set(transfer_idx) & set(group_A))}")
        generator = torch.Generator()
        generator.manual_seed(seed)
        trainloader = DataLoader(dataset.subset("train", transfer_idx, clean=False), batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=workers, worker_init_fn=util.seed_worker,
                                 generator=generator, persistent_workers=workers > 0, pin_memory=True)
        testloader = DataLoader(dataset.test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                pin_memory=True)

        cfg = setup["KD_Setups"][method]
        n_epochs = setup["Epochs"] if epochs is None else epochs
        distiller = cfg["Builder"]().to(util.device)          # fresh student + frozen teacher
        params = distiller.get_learnable_parameters()
        if not params:
            raise ValueError(f"{name}: the distiller exposes no trainable parameters")
        optimizer, scheduler = util.build_optimizer_scheduler(
            params, cfg["Optimizer_Name"], cfg["Optimizer_Params"],
            cfg["Scheduler_Name"], cfg["Scheduler_Params"], n_epochs)

        def step(epoch, distiller=distiller, trainloader=trainloader, optimizer=optimizer):
            return util.train_one_epoch_kd(distiller, trainloader, optimizer, epoch, util.device)

        print(f"\n[{STAGE}] {name}: {method}, {n_epochs} epoch(s), teacher {setup['Teacher_Checkpoint']}")
        result = util.run_training(
            scenario_name=name, epochs=n_epochs, step_fn=step, eval_net=distiller.student,
            testloader=testloader, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            save_dir=save_dir, log_file=util.training_log_path(STAGE, name),
            best_ckpt_start_frac=BEST_CKPT_START_FRAC)
        print(f"==> {name}: best test acc {result['best_test_acc']:.2f}% -> {result['best_checkpoint']}")
        del distiller, optimizer, scheduler, trainloader, testloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if measure_mi:
        for method in methods:
            ids = {"method": method, "seed": seed, "rate": rate}
            best = util.model_dir(STAGE, plan, **ids) / util.BEST_CHECKPOINT
            util.set_seed(calculate_MI.SUBSET_SEED, deterministic=True)
            calculate_MI.measure_checkpoint(STAGE, plan, best, ids, plan_path=plan_path)


def run(argv=None):
    parser = util.build_argparser(STAGE, __doc__)
    parser.add_argument("--methods", nargs="*", default=None, choices=("KD", "DKD"),
                        help="subset of the plan's Distillation methods (default: all in the plan)")
    parser.add_argument("--rate", type=float, default=util.NEGATIVE_RATE,
                        help="overlap of the transfer set with the victim's group_A (default 0.0, disjoint)")
    args = util.apply_common_args(parser.parse_args(argv))
    plans = util.plan_files(STAGE, args.plans, args.plan_dir)
    if not plans:
        raise SystemExit(f"No plan files in {args.plan_dir or util.stage_plan_dir(STAGE)} matching {list(args.plans)}")
    seeds = util.parse_seeds(args.seeds) if args.seeds else util.DEFAULT_SEEDS[STAGE]

    loaded = [(p, util.load_plan(p, args.data_root)) for p in plans]
    cells = [(p, plan, ids) for p, plan in loaded for ids in util.model_grid(STAGE, plan, seeds=seeds, rate=args.rate)
             if args.methods is None or ids["method"] in args.methods]
    done = sum(util.checkpoint_exists(util.model_dir(STAGE, plan, **ids) / util.BEST_CHECKPOINT) for _, plan, ids in cells)
    print(f"[{STAGE}] {len(plans)} plan(s): " + ", ".join(p.name for p in plans))
    print(f"[{STAGE}] seeds {seeds}, methods {args.methods or 'all in plan'}, transfer overlap {args.rate}, "
          f"deterministic {DETERMINISTIC}, workers {args.workers}"
          + (f", epochs override {args.epochs}" if args.epochs else ""))
    print(f"[{STAGE}] TOTAL {len(cells)} cell(s): {done} already trained, "
          f"{len(cells) - done if not args.no_skip else len(cells)} to train; MI {'off' if args.no_mi else 'on'}")

    for plan_path, plan in loaded:
        for seed in seeds:
            print(f"\n========== {plan_path.name}  seed {seed} ==========")
            train_distillation(plan, plan_path, seed, methods=args.methods, rate=args.rate, workers=args.workers,
                               epochs=args.epochs, skip_existing=not args.no_skip, measure_mi=not args.no_mi)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    run()
