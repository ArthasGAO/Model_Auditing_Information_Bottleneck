"""Coordinate the original IPGuard, DeepJudge and ADV_TRA YAML entry points."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    for attempt in range(120):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 119:
                raise
            time.sleep(0.5)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', type=Path, default=ROOT / 'saved_exp_plan/kd_eval_plan')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    import yaml
    import torch
    from util import build_dataset_from_yaml
    from util_adv import load_positive_suspect
    import main_IPGUARD_eval as ip
    import main_DEEPJUDGE_eval as dj
    import main_adv_tra as adv
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Use the CUDA PyTorch environment for this evaluation.')
    plans = [(p, yaml.safe_load(p.read_text(encoding='utf-8'))) for p in sorted(args.plan_dir.glob('*.yaml'))]
    if not plans:
        raise ValueError('No YAML evaluation plans found.')
    outputs = {p['Evaluation_Output'] for _, p in plans}
    if len(outputs) != 1:
        raise ValueError('All plans must share Evaluation_Output.')
    output = (ROOT / outputs.pop()).resolve()
    cases = []
    for path, plan in plans:
        cfg = plan['Positive']
        if cfg.get('Checkpoint_Format') != 'kd' or cfg['State'] != 'best':
            raise ValueError('This coordinator requires explicit KD best-checkpoint plans.')
        for directory in cfg['Model_Path']:
            checkpoint = ROOT / 'saved_models' / directory.lstrip('/') / 'best_epoch.pth'
            cases.append(dict(case_id=checkpoint.parent.name, path=checkpoint.as_posix(),
                              sha256=sha(checkpoint), architecture=cfg['Model'],
                              model_config=cfg.get('Model_Config'), plan=path.name))
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate cases across YAML plans.')
    # These assets are the same victim protocol used for the fine-tuning sweep.
    ip_dir = ROOT / 'Indices/CIFAR-10/IPGuard/victim=ResNet-18_CIFAR-10_25000_seed=42_overlap=1.0_k=5_size=100'
    tra_dir = ROOT / 'results/advtra1/CIFAR-10_ResNet-18_25000_42_1.0/fingerprints/cifar10/trajectory_8'
    caches = [ip_dir / f'{tag}.pt' for tag in ['TR', 'TL', 'RR', 'RL']]
    caches += [tra_dir / str(i) / name for i in range(1, 101) for name in ['tra_log.pth', 'pred_log.pth']]
    caches += [ROOT / 'Indices/adv_examples/CIFAR-10_ResNet-18_PGD_eps=0.03_step_size=0.003_steps=10.pt']
    threshold_root = ROOT / 'saved_logs/at_eval/DeepJudge/Threshold'
    caches += [threshold_root / 'RobD/victim=ResNet-18_CIFAR-10_25000_seed=42_overlap=1.0_RobD_PGD_eps=0.03_step_size=0.003_steps=10_threshold.json',
               threshold_root / 'JSD/victim=ResNet-18_CIFAR-10_25000_seed=42_overlap=1.0_JSD_clean_threshold.json',
               ROOT / 'saved_models/vanilla/CNN_Models/CIFAR-10_ResNet-18_25000_42_1.0/best_epoch.pth']
    config = dict(cases=cases, plans={p.name: {'sha256': sha(p), 'settings': plan} for p, plan in plans},
                  caches={p.relative_to(ROOT).as_posix(): sha(p) for p in caches},
                  code={name: sha(ROOT / name) for name in [Path(__file__).name, 'util_adv.py',
                        'Model/kd_eval.py', 'Model/ResNet_18_dist.py', 'Model/VGG16_dist.py', 'Model/DeiT.py',
                        'main_IPGUARD_eval.py', 'main_DEEPJUDGE_eval.py', 'main_adv_tra.py',
                        'AdvAttack/advtra/adv_tra_adapter.py']})
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    if args.preflight:
        dataset, classes, _ = build_dataset_from_yaml(plans[0][1]['Victim']['Dataset'])
        raw = torch.stack([dataset.raw_test_set[i][0] for i in range(2)]).cuda()
        for case in cases:
            net = load_positive_suspect(case['architecture'], classes, dataset, case['path'],
                                       checkpoint_format='kd', model_config=case['model_config'])
            with torch.no_grad():
                logits = net(raw)
                native, _ = net.base_model.student((raw - net.mean) / net.std)
            assert logits.shape == (2, classes) and torch.isfinite(logits).all()
            torch.testing.assert_close(logits, native, rtol=0, atol=0)
            state = torch.load(case['path'], map_location='cpu', weights_only=True)
            print(f'PASS {case["case_id"]}: {len(state)} tensors, first key={next(iter(state))}; native logits identical', flush=True)
            del net, state
            torch.cuda.empty_cache()
        print(f'Preflight passed: {len(cases)} KD checkpoints; {len(caches)} cached assets; config={config_id}', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    # OS lock releases after a crash without requiring stale-lock deletion.
    with (output / 'run.lock').open('a+b') as lock:
        import msvcrt
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        manifest = output / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['config_id'] != config_id:
            raise ValueError('Existing output has different inputs/code. Choose a new output in the YAMLs.')
        write(manifest, json.dumps(dict(config_id=config_id, config=config), indent=2))
        jobs_path = output / 'jobs.json'
        jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else {}

        def combine():
            ip_rows = read_csv(output / 'IPGuard/master.csv')
            dj_rows = read_csv(output / 'DeepJudge/master.csv')
            adv_rows = [r for p in (output / 'ADV_TRA').glob('*.csv') for r in read_csv(p)]
            rows = []
            for case in cases:
                name = case['case_id']
                row = dict(Case_ID=name, Suspect_Arch=case['architecture'], Positive=1,
                           Checkpoint_Path=case['path'], Checkpoint_SHA256=case['sha256'], Config_ID=config_id)
                ip_case = {r['FP_Config']: r for r in ip_rows if r['Scenario_Name'] == name}
                row['IPGuard_Status'] = 'complete' if set(ip_case) == {'TR', 'TL', 'RR', 'RL'} else 'pending'
                for tag in ['TR', 'TL', 'RR', 'RL']:
                    row[f'IPGuard_{tag}'] = ip_case.get(tag, {}).get('Matching_Rate', '')
                row['IPGuard_Mean'] = sum(float(row[f'IPGuard_{t}']) for t in ip_case) / 4 if len(ip_case) == 4 else ''
                for method, source, field, metrics in [
                    ('DeepJudge', dj_rows, 'Scenario_Name', ['RobD', 'JSD_Suspect', 'Stolen']),
                    ('ADV_TRA', adv_rows, 'Scenario', ['Detection_Rate', 'Mean_Mutation_Rate', 'Stolen'])]:
                    found = [r for r in source if r[field] == name]
                    row[f'{method}_Status'] = 'complete' if found else 'pending'
                    for metric in metrics:
                        row[f'{method}_{metric}'] = found[-1][metric] if found else ''
                rows.append(row)
            buf = io.StringIO(newline='')
            writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            write(output / 'master.csv', buf.getvalue())
            return rows

        with (output / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                for method, fn in [('IPGuard', ip.main_ipguard), ('ADV_TRA', adv.main_adv_tra_pos), ('DeepJudge', dj.main_robd_jsd)]:
                    for path, plan in plans:
                        job = f'{method}/{path.name}'
                        if jobs.get(job) == 'complete':
                            continue
                        combine()
                        write(output / 'status.json', json.dumps(dict(state='running', pid=os.getpid(), job=job,
                              completed_jobs=sum(v == 'complete' for v in jobs.values()), total_jobs=3 * len(plans)), indent=2))
                        print(f'\nSTART {job}', flush=True)
                        try:
                            ip.set_seed(42)
                            fn(str(path))
                            rows = combine()
                            expected = {c['case_id'] for c in cases if c['plan'] == path.name}
                            if any(r[f'{method}_Status'] != 'complete' for r in rows if r['Case_ID'] in expected):
                                raise RuntimeError(f'Missing results after {job}')
                            jobs[job] = 'complete'
                        except Exception:
                            jobs[job] = 'failed'
                            traceback.print_exc()
                        finally:
                            write(jobs_path, json.dumps(jobs, indent=2))
                            combine()
                            torch.cuda.empty_cache()
                failed = any(v != 'complete' for v in jobs.values())
                write(output / 'status.json', json.dumps(dict(state='completed_with_errors' if failed else 'complete',
                      pid=os.getpid(), completed_jobs=sum(v == 'complete' for v in jobs.values()), total_jobs=3 * len(plans)), indent=2))
                print(f'Finished. Results: {output / "master.csv"}', flush=True)
                if failed:
                    raise SystemExit(1)


if __name__ == '__main__':
    main()
