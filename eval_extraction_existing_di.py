"""Evaluate extraction checkpoints using the existing CIFAR-10 ResNet-18 DI regressor.

This entry point never trains a regressor. Run --preflight before evaluation.
See docs/di_extraction.md for selection, protocol, and result interpretation.
"""
import argparse
import contextlib
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
VICTIM = 'CIFAR-10_ResNet-18_25000'
REGRESSOR = ROOT / 'saved_models/di_regressor/victim=ResNet-18_CIFAR-10_25000_seed=42_overlap=1.0_n=1000.pt.augmented_stale.bak'
METRICS = ['Mean_Private', 'Mean_Public', 'Delta', 'T_Stat', 'P_Value', 'Stolen']
COLUMNS = ['Case_ID', 'Method', 'Extraction_Seed', 'Victim_Arch', 'Suspect_Arch',
           'Checkpoint_Path', 'Checkpoint_SHA256', 'Config_ID', 'Regressor_Path',
           'Regressor_SHA256', 'N_Test_Per_Set', 'Alpha', 'Status', *METRICS,
           'Seconds', 'Updated_At', 'Error']


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def atomic_write(path, contents):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(contents, encoding='utf-8')
    for attempt in range(120):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 119:
                raise
            time.sleep(0.5)


def save_table(path, rows):
    import io
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buf.getvalue())


def discover_cases(patterns, checkpoint_mode):
    """Resolve architecture from YAML; never infer it from Same/Cross labels."""
    import yaml
    plans = {}
    for folder in ['extraction_plan', 'extraction_plan_crossarch', 'dfms_plan']:
        for path in sorted((ROOT / 'saved_exp_plan' / folder).rglob('*.yaml')):
            plan = yaml.safe_load(path.read_text(encoding='utf-8'))
            victim = plan.get('Victim', {})
            if victim.get('Model_Name') != VICTIM:
                continue
            if (victim.get('Model') != 'ResNet-18' or
                    victim.get('Dataset', {}).get('name') != 'CIFAR-10' or
                    victim.get('Seed', 42) != 42 or victim.get('Overlap', 1.0) != 1.0):
                raise ValueError(f'Incompatible victim declaration: {path}')
            name = plan['Scenario_Name']
            model = plan['Substitute']['Model']
            if name in plans and plans[name]['model'] != model:
                raise ValueError(f'Conflicting substitute architectures for {name}')
            plans.setdefault(name, dict(model=model, yaml=path.as_posix()))
    cases = []
    root = ROOT / 'saved_models/extraction_vanilla'
    for folder in sorted(root.rglob(f'{VICTIM}_*')):
        if not folder.is_dir() or not any(fnmatch.fnmatchcase(folder.name, p) for p in patterns):
            continue
        match = re.fullmatch(r'(.+)_(\d+)_1\.0', folder.name)
        if not match or match[1] not in plans:
            raise ValueError(f'No matching extraction YAML for {folder}')
        plan = plans[match[1]]
        method = next((m for m in ['Knockoff', 'JBA', 'DFMS'] if f'_{m}_' in match[1]), None)
        if method is None:
            raise ValueError(f'Unsupported extraction method: {folder}')
        if checkpoint_mode == 'best':
            checkpoint = folder / 'best_epoch.pth'
        elif method in ('JBA', 'DFMS'):
            checkpoint = folder / 'final_round.pth'
        else:
            numbered = [p for p in folder.glob('epoch_*.pth') if re.fullmatch(r'epoch_\d+', p.stem)]
            if not numbered:
                raise FileNotFoundError(f'No numbered final checkpoint in {folder}')
            checkpoint = max(numbered, key=lambda p: int(p.stem.split('_')[1]))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        cases.append(dict(case_id=f'{folder.name}/{checkpoint.name}', method=method,
                          extraction_seed=int(match[2]), model=plan['model'], yaml=plan['yaml'],
                          checkpoint=checkpoint.as_posix(), sha256=sha256(checkpoint)))
    if not cases:
        raise ValueError('No matching extraction checkpoints. Check --case glob.')
    return cases


def load_suspect(case, dataset, device):
    import torch
    from util import build_deit_student
    from util_adv import build_model, NormalizedModel
    cfg = case['model']
    if isinstance(cfg, dict):
        net = build_deit_student({'Model': {**cfg, 'pretrained': False}}, 10)
    else:
        net = build_model(cfg, 10)
    state = torch.load(case['checkpoint'], map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    elif 'state_dict' in state:
        state = state['state_dict']
    if any(k.startswith('base_model.') for k in state):
        state = {k.removeprefix('base_model.'): v for k, v in state.items() if k not in ('mean', 'std')}
    net.load_state_dict(state, strict=True)
    # Matches extraction scripts' evaluation on victim-normalized test images.
    return NormalizedModel(net, dataset.mean, dataset.std).to(device).eval()


@contextlib.contextmanager
def output_lock(output):
    # OS lock is released even after a crash; the lock file can remain on disk.
    with (output / 'run.lock').open('a+b') as file:
        file.write(b'0')
        file.flush()
        file.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='Strict-load and forward-check every selected model; no DI evaluation.')
    parser.add_argument('--case', action='append', help='Folder-name glob, repeatable; default all models from the supported victim.')
    parser.add_argument('--checkpoint', choices=['best', 'last'], default='best')
    parser.add_argument('--regressor', type=Path, default=REGRESSOR, help='Existing regressor for this exact victim; never auto-trained.')
    parser.add_argument('--n-test', type=int, default=1000, help='Samples in EACH private/public pool.')
    parser.add_argument('--point-batch-size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output', type=Path, default=ROOT / 'saved_logs/at_eval/DI/extraction_existing_regressor')
    args = parser.parse_args()
    os.chdir(ROOT)
    if not 2 <= args.n_test <= 10000 or args.point_batch_size < 1 or not 0 < args.alpha < 1:
        parser.error('Require 2 <= n-test <= 10000, positive point batch size, and 0 < alpha < 1.')
    import numpy as np
    import torch
    from AdvAttack.DI import DatasetInferencePipeline, ConfidenceRegressor
    from util import build_dataset_from_yaml
    from main_DI_eval import set_seed
    torch.set_num_threads(4)
    cases = discover_cases(args.case or ['*'], args.checkpoint)
    args.regressor = args.regressor.resolve(strict=True)
    checkpoint = torch.load(args.regressor, map_location='cpu', weights_only=True)
    walk = {k: checkpoint[k] for k in ['n_samples', 'noise_uniform', 'noise_gaussian', 'noise_laplace', 'max_steps']}
    if checkpoint['embedding_dim'] != 3 * walk['n_samples']:
        raise ValueError('Regressor dimensions do not match its saved walk settings.')
    regressor = ConfidenceRegressor(checkpoint['embedding_dim'], checkpoint['embedding_dim'])
    regressor.load_state_dict(checkpoint['state_dict'], strict=True)
    indices_path = ROOT / 'Indices/CIFAR-10/group_A_subset_10000_from_25000_seed42.npy'
    private_indices = np.load(indices_path)[:args.n_test]
    if len(private_indices) != args.n_test or len(set(private_indices.tolist())) != args.n_test:
        raise ValueError('Insufficient or duplicate private indices.')
    public_indices = list(range(args.n_test))
    config = dict(victim=VICTIM, victim_seed=42, victim_overlap=1.0,
                  regressor=args.regressor.as_posix(), regressor_sha256=sha256(args.regressor),
                  trained_in_this_run=False, walk=walk, n_test_per_set=args.n_test,
                  point_batch_size=args.point_batch_size, alpha=args.alpha, seed=args.seed,
                  device=args.device, checkpoint_mode=args.checkpoint,
                  input_protocol='existing_victim_regressor_legacy_input_v1',
                  private_transform='RandomCrop(32,padding=4),RandomHorizontalFlip(p=0.5),ToTensor',
                  public_transform='ToTensor', normalization='cifar10',
                  private_indices=private_indices.tolist(), public_indices=public_indices,
                  historical_training_indices='unknown; no disjointness claim',
                  cases=cases, implementation_sha256={p: sha256(ROOT / p) for p in
                  [Path(__file__).name, 'AdvAttack/DI.py', 'util.py', 'util_adv.py',
                   'main_DI_eval.py', 'Dataset/CIFAR_10.py', 'Model/ResNet_18.py', 'Model/VGG16.py']})
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    set_seed(args.seed)
    dataset, _, _ = build_dataset_from_yaml(dict(name='CIFAR-10', normalization='cifar10',
                                               img_size=32, group_size=25000, download=False))
    if not np.issubdtype(private_indices.dtype, np.integer) or min(private_indices) < 0 or max(private_indices) >= len(dataset.raw_train_set):
        raise ValueError('Private indices are invalid for the dataset.')
    if len(dataset.raw_test_set) < args.n_test:
        raise ValueError('Insufficient public images.')
    if args.preflight:
        for case in cases:
            net = load_suspect(case, dataset, 'cpu')
            with torch.no_grad():
                logits = net(dataset.raw_test_set[0][0].unsqueeze(0))
            if logits.shape != (1, 10) or not torch.isfinite(logits).all():
                raise ValueError(f'Invalid logits: {case["case_id"]}')
            print(f'PASS {case["case_id"]}', flush=True)
            del net
        print(f'Preflight passed: {len(cases)} checkpoints; regressor strict-loaded; NO training. Config={config_id}')
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the project PyTorch environment or --device cpu.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        manifest = output / 'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8'))['config_id'] != config_id:
                raise ValueError('Output contains a different configuration. Use a different --output directory.')
        else:
            if (output / 'master.csv').exists():
                raise ValueError('Existing master has no manifest; use a different output directory.')
            atomic_write(manifest, json.dumps(dict(config_id=config_id, config=config), indent=2))
        rows = []
        if (output / 'master.csv').exists():
            with (output / 'master.csv').open(encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            if (len(rows) != len(cases) or [r['Case_ID'] for r in rows] != [c['case_id'] for c in cases]
                    or any(r['Config_ID'] != config_id for r in rows)):
                raise ValueError('Master rows do not match the manifest.')
        else:
            for case in cases:
                row = dict.fromkeys(COLUMNS, '')
                row.update(Case_ID=case['case_id'], Method=case['method'], Extraction_Seed=case['extraction_seed'],
                           Victim_Arch='ResNet-18', Suspect_Arch=json.dumps(case['model'], sort_keys=True),
                           Checkpoint_Path=case['checkpoint'], Checkpoint_SHA256=case['sha256'],
                           Config_ID=config_id, Regressor_Path=config['regressor'],
                           Regressor_SHA256=config['regressor_sha256'], N_Test_Per_Set=args.n_test,
                           Alpha=args.alpha, Status='pending')
                rows.append(row)
        save_table(output / 'master.csv', rows)
        pipeline = DatasetInferencePipeline(None, **walk, point_batch_size=args.point_batch_size, device=args.device)
        pipeline.regressor = regressor.to(args.device).eval()

        def status(state, **extra):
            atomic_write(output / 'status.json', json.dumps(dict(state=state, pid=os.getpid(),
                         total=len(rows), completed=sum(r['Status'] == 'complete' for r in rows),
                         failed=sum(r['Status'] == 'failed' for r in rows), **extra), indent=2))

        print(f'Existing regressor only: {args.regressor}\nOutput: {output}\nCases: {len(cases)}; no training.', flush=True)
        try:
            for case, row in zip(cases, rows):
                if row['Status'] == 'complete':
                    continue
                start = time.monotonic()
                net = None
                row['Status'] = 'running'
                save_table(output / 'master.csv', rows)
                status('running', case_id=case['case_id'])
                try:
                    set_seed(args.seed)
                    net = load_suspect(case, dataset, args.device)
                    result = pipeline.verify_suspect(net, dataset.raw_train_set, dataset.raw_test_set,
                              private_indices=private_indices, public_indices=public_indices,
                              n_test_samples=args.n_test, alpha=args.alpha)
                    values = {name: int(result[name.lower()]) if name == 'Stolen' else float(result[name.lower()])
                              for name in METRICS}
                    if not all(np.isfinite(value) for value in values.values()):
                        raise ValueError(f'Non-finite DI statistics: {values}')
                    row.update(values, Status='complete', Error='')
                    print(f'RESULT {case["case_id"]}: {json.dumps(values)}', flush=True)
                except Exception:
                    row.update(Status='failed', Error=traceback.format_exc())
                    print(row['Error'], flush=True)
                finally:
                    row.update(Seconds=round(time.monotonic() - start, 3), Updated_At=time.strftime('%Y-%m-%dT%H:%M:%S'))
                    save_table(output / 'master.csv', rows)
                    del net
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
            failed = any(r['Status'] != 'complete' for r in rows)
            status('completed_with_errors' if failed else 'complete')
        except BaseException:
            status('interrupted_or_failed')
            raise
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
