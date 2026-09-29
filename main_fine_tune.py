import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import calculate_MI
import util

STAGE = "fine_tune"
DETERMINISTIC = False          
BEST_CKPT_START_FRAC = 0.3    
BATCH_SIZE = 128


def make_loaders(ft_train, ft_val, seed, workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    trainloader = DataLoader(ft_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers,
                             worker_init_fn=util.seed_worker, generator=generator,
                             persistent_workers=workers > 0, pin_memory=True)
    test_workers = min(2, workers)
    testloader = DataLoader(ft_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=test_workers,
                            persistent_workers=test_workers > 0, pin_memory=True)
    return trainloader, testloader


def train_fine_tune(plan, plan_path, ft_seed, *, strategies=None, workers=4, epochs=None,
                    skip_existing=True, measure_mi=True):
    """Fine-tune one victim with every requested strategy for one fine-tuning seed."""
    setup = util.process_experiment_ft_setup(plan)
    victim_ckpt = util.victim_checkpoint(plan)
    strategies = list(setup["FT_Setups"]) if strategies is None else list(strategies)
    for strategy in strategies:
        if strategy not in setup["FT_Setups"]:
            raise KeyError(f"{plan_path.name}: no Optimizers entry for strategy={strategy}")
    base_ids = {"model_seed": util.VICTIM_SEED, "rate": util.VICTIM_RATE,
                "ft_size": setup["FT_GroupSize"], "ft_seed": ft_seed}
    criterion = nn.CrossEntropyLoss()
    group_A = ft_data = None

    for strategy in strategies:
        ids = {**base_ids, "strategy": strategy}
        name = util.model_name(STAGE, plan, **ids)
        save_dir = util.model_dir(STAGE, plan, **ids)
        best = save_dir / util.BEST_CHECKPOINT
        if skip_existing and util.checkpoint_exists(best):
            print(f"[SKIP] {name}: {best} exists")
            continue

        util.set_seed(ft_seed, deterministic=DETERMINISTIC)
        if ft_data is None:
            group_A = util.load_group_A(setup["Dataset"], plan["Dataset"], setup["GroupSize"], setup["NumClasses"])
            ft_data = util.determine_ft_dataset(plan, setup, group_A)
        trainloader, testloader = make_loaders(*ft_data, ft_seed, workers)

        cfg = setup["FT_Setups"][strategy]
        n_epochs = cfg["Epochs"] if epochs is None else epochs
        net = setup["Model_Factory"]().to(util.device)
        net.load_state_dict(util.load_state(victim_ckpt, map_location=util.device))
        net = util.setup_finetune(model=net, strategy=strategy, device=util.device)
        optimizer, scheduler = util.build_optimizer_scheduler(
            filter(lambda p: p.requires_grad, net.parameters()), cfg["Optimizer_Name"], cfg["Optimizer_Params"],
            cfg["Scheduler_Name"], cfg["Scheduler_Params"], n_epochs)

        def step(epoch, net=net, trainloader=trainloader, optimizer=optimizer, strategy=strategy):
            return util.ft_one_epoch(net, trainloader, optimizer, criterion, epoch, util.device, strategy)

        print(f"\n[{STAGE}] {name}: {strategy}, {n_epochs} epoch(s), lr {cfg['Optimizer_Params'].get('lr')}")
        result = util.run_training(
            scenario_name=name, epochs=n_epochs, step_fn=step, eval_net=net, testloader=testloader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, save_dir=save_dir,
            log_file=util.training_log_path(STAGE, name), best_ckpt_start_frac=BEST_CKPT_START_FRAC)
        print(f"==> {name}: best test acc {result['best_test_acc']:.2f}% -> {result['best_checkpoint']}")
        del net, optimizer, scheduler, trainloader, testloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if measure_mi:
        for strategy in strategies:
            ids = {**base_ids, "strategy": strategy}
            best = util.model_dir(STAGE, plan, **ids) / util.BEST_CHECKPOINT
            util.set_seed(calculate_MI.SUBSET_SEED, deterministic=True)
            calculate_MI.measure_checkpoint(STAGE, plan, best, ids)


def run(argv=None):
    parser = util.build_argparser(STAGE, __doc__)
    parser.add_argument("--strategies", nargs="*", default=None, choices=("FT-LL", "FT-AL", "RT-AL"),
                        help="subset of the plan's strategies (default: all in the plan)")
    args = util.apply_common_args(parser.parse_args(argv))
    plans = util.plan_files(STAGE, args.plans, args.plan_dir)
    if not plans:
        raise SystemExit(f"No plan files in {args.plan_dir or util.stage_plan_dir(STAGE)} matching {list(args.plans)}")
    seeds = util.parse_seeds(args.seeds) if args.seeds else util.DEFAULT_SEEDS[STAGE]

    loaded = [(p, util.load_plan(p, args.data_root)) for p in plans]
    cells = [(p, plan, ids) for p, plan in loaded for ids in util.model_grid(STAGE, plan, seeds=seeds)
             if args.strategies is None or ids["strategy"] in args.strategies]
    done = sum(util.checkpoint_exists(util.model_dir(STAGE, plan, **ids) / util.BEST_CHECKPOINT) for _, plan, ids in cells)
    print(f"[{STAGE}] {len(plans)} plan(s): " + ", ".join(p.name for p in plans))
    print(f"[{STAGE}] seeds {seeds}, strategies {args.strategies or 'all in plan'}, deterministic {DETERMINISTIC}, "
          f"workers {args.workers}" + (f", epochs override {args.epochs}" if args.epochs else ""))
    print(f"[{STAGE}] TOTAL {len(cells)} cell(s): {done} already trained, "
          f"{len(cells) - done if not args.no_skip else len(cells)} to train; MI {'off' if args.no_mi else 'on'}")

    for plan_path, plan in loaded:
        for ft_seed in seeds:
            print(f"\n========== {plan_path.name}  ft_seed {ft_seed} ==========")
            train_fine_tune(plan, plan_path, ft_seed, strategies=args.strategies, workers=args.workers,
                            epochs=args.epochs, skip_existing=not args.no_skip, measure_mi=not args.no_mi)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    run()
