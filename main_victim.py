import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import calculate_MI
import util

STAGE = "victim"

DETERMINISTIC = False
BEST_CKPT_START_FRAC = 0.7
BATCH_SIZE = 128


def train_base_model(stage, plan, plan_path, seed, rate, *, workers=4, epochs=None,
                     skip_existing=True, measure_mi=True):
    """Train one victim (rate 1.0) or negative (rate 0.0) and measure its MI.

    rate is the overlap of the training split with group_A: 1.0 makes the split
    group_A itself (the victim), 0.0 the disjoint group_B (a negative).
    Returns the model folder.
    """
    ids = {"seed": seed, "rate": rate}
    name = util.model_name(stage, plan, **ids)
    save_dir = util.model_dir(stage, plan, **ids)
    best = save_dir / util.BEST_CHECKPOINT

    if skip_existing and util.checkpoint_exists(best):
        print(f"[SKIP] {name}: {best} exists")
    else:
        util.set_seed(seed, deterministic=DETERMINISTIC)
        generator = torch.Generator()
        generator.manual_seed(seed)

        setup = util.process_experiment_setup(plan, epochs=epochs)
        dataset = setup["Dataset"]
        group_A = util.load_group_A(dataset, plan["Dataset"], setup["GroupSize"], setup["NumClasses"])
        train_idx = util.load_group_B(dataset, plan["Dataset"], group_A, setup["GroupSize"],
                                      setup["NumClasses"], overlap_rate=rate)
        print(f"[DATA] {name}: {len(train_idx)} training images, "
              f"{len(set(train_idx) & set(group_A))} shared with group_A")

        trainloader = DataLoader(dataset.subset("train", train_idx, clean=False), batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=workers, worker_init_fn=util.seed_worker,
                                 generator=generator, persistent_workers=workers > 0, pin_memory=True)
        test_workers = min(2, workers)
        testloader = DataLoader(dataset.test_set, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=test_workers, persistent_workers=test_workers > 0, pin_memory=True)

        net = setup["Model"].to(util.device)
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = setup["Optimizer"], setup["Scheduler"]
        n_epochs = setup["Epochs"]

        def step(epoch):
            return util.train_one_epoch(net, trainloader, optimizer, criterion, epoch, util.device)

        result = util.run_training(
            scenario_name=name, epochs=n_epochs, step_fn=step, eval_net=net, testloader=testloader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, save_dir=save_dir,
            log_file=util.training_log_path(stage, name), best_ckpt_start_frac=BEST_CKPT_START_FRAC)
        print(f"==> {name}: best test acc {result['best_test_acc']:.2f}% -> {result['best_checkpoint']}")
        del net, optimizer, scheduler, trainloader, testloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if measure_mi:
        util.set_seed(calculate_MI.SUBSET_SEED, deterministic=True)
        calculate_MI.measure_checkpoint(stage, plan, best, ids)
    return save_dir


def run(stage, argv=None):
    """Command line driver shared by main_victim.py and main_negative.py."""
    args = util.apply_common_args(util.build_argparser(stage, __doc__).parse_args(argv))
    plans = util.plan_files(stage, args.plans, args.plan_dir)
    if not plans:
        raise SystemExit(f"No plan files in {args.plan_dir or util.stage_plan_dir(stage)} matching {list(args.plans)}")
    seeds = util.parse_seeds(args.seeds) if args.seeds else util.DEFAULT_SEEDS[stage]
    rate = util.VICTIM_RATE if stage == "victim" else util.NEGATIVE_RATE

    loaded = [(p, util.load_plan(p, args.data_root)) for p in plans]
    cells = [(p, plan, s) for p, plan in loaded for s in seeds]
    done = sum(util.checkpoint_exists(util.model_dir(stage, plan, seed=s, rate=rate) / util.BEST_CHECKPOINT)
               for _, plan, s in cells)
    print(f"[{stage}] {len(plans)} plan(s): " + ", ".join(p.name for p in plans))
    print(f"[{stage}] seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}), rate {rate}, deterministic {DETERMINISTIC}, "
          f"workers {args.workers}" + (f", epochs override {args.epochs}" if args.epochs else ""))
    print(f"[{stage}] TOTAL {len(cells)} cell(s): {done} already trained, "
          f"{len(cells) - done if not args.no_skip else len(cells)} to train; MI {'off' if args.no_mi else 'on'}")

    for plan_path, plan, seed in cells:
        print(f"\n========== {plan_path.name}  seed {seed} ==========")
        train_base_model(stage, plan, plan_path, seed, rate, workers=args.workers, epochs=args.epochs,
                         skip_existing=not args.no_skip, measure_mi=not args.no_mi)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    run(STAGE)
