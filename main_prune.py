import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # see main_victim.py

import torch
import torch.nn as nn

import calculate_MI
import util
from main_fine_tune import make_loaders

STAGE = "prune"
DETERMINISTIC = False          
BEST_CKPT_START_FRAC = 0.2     
STRATEGY = util.PRUNE_STRATEGY  


def dense_state(net):
    """state_dict with the pruning masks folded into the weights (what gets saved): a pruned
    tensor is stored as `<name>_orig` and `<name>_mask`; unpruned ones keep their plain name."""
    state = net.state_dict()
    dense = {}
    for key, value in state.items():
        if key.endswith("_mask"):
            continue
        if key.endswith("_orig"):
            key, value = key[:-len("_orig")], value * state[key[:-len("_orig")] + "_mask"]
        dense[key] = value.detach().clone()
    return dense


def train_prune(plan, plan_path, ft_seed, *, sparsities=None, workers=4, epochs=None,
                skip_existing=True, measure_mi=True):
    """Prune + recover one victim at every requested sparsity for one recovery seed."""
    setup = util.process_experiment_prune_setup(plan)
    prune_exclude = plan.get("Prune_Exclude") or None      # substrings of module names kept dense
    if isinstance(prune_exclude, str):
        prune_exclude = [prune_exclude]
    victim_ckpt = util.victim_checkpoint(plan)
    victim_state = util.load_state(victim_ckpt, map_location=util.device)
    levels = util.sparsity_levels_from_setup(setup) if sparsities is None else [round(float(s), 6) for s in sparsities]
    for s in levels:
        if s not in setup["Prune_Setups"]:
            raise KeyError(f"{plan_path.name}: no Optimizers entry for sparsity={s}")
    base_ids = {"model_seed": util.VICTIM_SEED, "rate": util.VICTIM_RATE, "strategy": STRATEGY,
                "ft_size": setup["FT_GroupSize"], "ft_seed": ft_seed}
    criterion = nn.CrossEntropyLoss()
    group_A = ft_data = None

    for s in levels:
        ids = {**base_ids, "sparsity": s}
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

        cfg = setup["Prune_Setups"][s]
        n_epochs = cfg["Epochs"] if epochs is None else epochs
        net = setup["Model_Factory"]().to(util.device)
        net.load_state_dict(victim_state)
        net = util.prune_model_global(model=net, amount=s, exclude_patterns=prune_exclude)
        print(f"[{STAGE}] {name}: pruned to sparsity {s}" + (f" sparing {prune_exclude}" if prune_exclude else ""))
        net = util.setup_finetune(model=net, strategy=STRATEGY, device=util.device)
        optimizer, scheduler = util.build_optimizer_scheduler(
            filter(lambda p: p.requires_grad, net.parameters()), cfg["Optimizer_Name"], cfg["Optimizer_Params"],
            cfg["Scheduler_Name"], cfg["Scheduler_Params"], n_epochs)

        def step(epoch, net=net, trainloader=trainloader, optimizer=optimizer):
            return util.ft_one_epoch(net, trainloader, optimizer, criterion, epoch, util.device, STRATEGY)

        print(f"[{STAGE}] {name}: recovery {STRATEGY}, {n_epochs} epoch(s), lr {cfg['Optimizer_Params'].get('lr')}")
        result = util.run_training(
            scenario_name=name, epochs=n_epochs, step_fn=step, eval_net=net, testloader=testloader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, save_dir=save_dir,
            log_file=util.training_log_path(STAGE, name), best_ckpt_start_frac=BEST_CKPT_START_FRAC,
            export_state=dense_state)
        print(f"==> {name}: best test acc {result['best_test_acc']:.2f}% -> {result['best_checkpoint']}")
        print(f"[{STAGE}] {name}: sparsity after recovery {util.check_pruned_weights(net):.4f}")
        del net, optimizer, scheduler, trainloader, testloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if measure_mi:
        for s in levels:
            ids = {**base_ids, "sparsity": s}
            best = util.model_dir(STAGE, plan, **ids) / util.BEST_CHECKPOINT
            util.set_seed(calculate_MI.SUBSET_SEED, deterministic=True)
            calculate_MI.measure_checkpoint(STAGE, plan, best, ids)


def run(argv=None):
    parser = util.build_argparser(STAGE, __doc__)
    parser.add_argument("--sparsities", nargs="*", type=float, default=None,
                        help="subset of the plan's sparsity levels (default: all in the plan)")
    args = util.apply_common_args(parser.parse_args(argv))
    plans = util.plan_files(STAGE, args.plans, args.plan_dir)
    if not plans:
        raise SystemExit(f"No plan files in {args.plan_dir or util.stage_plan_dir(STAGE)} matching {list(args.plans)}")
    seeds = util.parse_seeds(args.seeds) if args.seeds else util.DEFAULT_SEEDS[STAGE]
    wanted = None if args.sparsities is None else {round(float(s), 6) for s in args.sparsities}

    loaded = [(p, util.load_plan(p, args.data_root)) for p in plans]
    cells = [(p, plan, ids) for p, plan in loaded for ids in util.model_grid(STAGE, plan, seeds=seeds)
             if wanted is None or ids["sparsity"] in wanted]
    done = sum(util.checkpoint_exists(util.model_dir(STAGE, plan, **ids) / util.BEST_CHECKPOINT) for _, plan, ids in cells)
    print(f"[{STAGE}] {len(plans)} plan(s): " + ", ".join(p.name for p in plans))
    print(f"[{STAGE}] seeds {seeds}, sparsities {sorted(wanted) if wanted else 'all in plan'}, "
          f"deterministic {DETERMINISTIC}, workers {args.workers}"
          + (f", epochs override {args.epochs}" if args.epochs else ""))
    print(f"[{STAGE}] TOTAL {len(cells)} cell(s): {done} already trained, "
          f"{len(cells) - done if not args.no_skip else len(cells)} to train; MI {'off' if args.no_mi else 'on'}")

    for plan_path, plan in loaded:
        for ft_seed in seeds:
            print(f"\n========== {plan_path.name}  ft_seed {ft_seed} ==========")
            train_prune(plan, plan_path, ft_seed, sparsities=args.sparsities, workers=args.workers,
                        epochs=args.epochs, skip_existing=not args.no_skip, measure_mi=not args.no_mi)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    run()
